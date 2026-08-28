# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-dimensional reward for RPCO agentic RL.

Reward set (VisionCreator-R1, arXiv:2603.08812, §4.1/§4.3):

  R_reflect — continuous last-image quality from live ``judge_image`` C/A
      (mean of the **final** successful judge, not the first ``good_enough=YES``).
      Optionally mixed with a light lexical regularizer and a rewrite-delta
      term: ``0.70 * last_CA + 0.20 * coverage + 0.10 * max(0, last_CA - first_CA)``.
      ``good_enough`` is not used as a binary gate on this dim. Also emitted as
      ``reward_correctness`` / ``reward_aesthetics`` (last image) and
      ``first_correctness`` / ``first_aesthetics``.
  R_plan    — requirement coverage of the agent's plan lines against the
      reference subtasks (per-subtask best token overlap, then mean). Applied
      only on plan rows (``gt.reference_subtasks``). Zero when no plan text.
  R_format  — rule-based ratio over structural checks: well-formed tool calls,
      judge after the last generate, terminal policy-sampled Done, and the
      task-type tags (Plan + Reflection on plan rows; Reflection on reflect rows).
  R_tool_call — tool-call presence, the same ``f_tool_call`` as PR 1
      (``agentic_reward.py``): 1.0 iff any tool call was parsed, else 0.
      Emitted as ``reward_tool_call`` so PR 1 and RPCO share one WandB series.
  R_result  — output count/type match against ``expected_num_images``.
      Plan rows: exact count match. Reflect rows (lenient stop-validity):
      terminal Done + ≥1 image + (count ≤ expected OR last judge YES).
  R_done    — logged-only (not in W): PR 1's ``f_done`` closed-loop indicator
      (valid terminal context + policy-sampled ``Done.`` or forced stop cue),
      emitted as ``reward_done`` for the ``agentic_reward/done`` WandB series.
  R_terminal_no_penalty — an additive -1.0 scalar penalty when the final
      successful judge verdict is ``good_enough=NO``. Component rewards remain
      visible for diagnosis, but a failed terminal verdict cannot have positive
      total reward. The penalty is placed on the final response token by verl's
      reward manager, so it trains the sampled terminal decision.
  R_missing_tools_penalty — an additive -1.0 scalar penalty when the trajectory
      omits a ``generate_image`` call or a ``judge_image`` call. Without this,
      generate-only rollouts still earn ~0.3–0.4 from format/tool_call/result
      while judge+NO is floored at ≤0, so the policy learns to skip the judge.
      Same magnitude as ``R_terminal_no_penalty`` so skip-judge cannot beat
      an honest failed verdict.
  R_good_enough_floor_lift — raises a valid trajectory whose final successful
      sidecar verdict is ``good_enough=YES`` to at least 0.80. This protects the
      sparse success signal when the rollout reaches a good image near the
      generate/turn cap but lacks budget for a perfect Reflection/Done suffix.
      Cleanly closed YES trajectories retain their naturally higher score.

Base: ``base_score = (1/|W|) * sum(w_i * R_i)`` over the active set W (dims
with ``w_* > 0`` that apply to the row's task type). Final:
``score = base_score + R_terminal_no_penalty + R_missing_tools_penalty``,
clipped to [-1, 1]. Default weights are 1.0 except ``w_reflect=1.5`` so
last-image C/A outranks format/tool presence. ``RPCO_W_*`` env vars override
parquet-baked ``w_*`` without a rebuild.

Gating kept from PR 1: no ``generate_image`` / no successful PNG → score 0 and
``rollout_valid=0`` (rollout is discarded from the GRPO update). Env-injected
``Reflection`` (``agentic_forced_reflection=1``) never earns credit.
"""

from __future__ import annotations

import os
import re
from typing import Any

from verl_omni.utils.reward_score.agentic_reward import (
    _BLOCKED_GENERATE_RE,
    _PATH_RE,
    _TOOL_CALL_RE,
    _TOOL_OK,
    _as_dict,
    _assistant_prose,
    _extract_tool_calls,
    _gen_image_prompts,
    _has_agent_reflection_prose,
    _has_successful_generated_image,
    _iter_successful_judge_scores,
    _judge_parse_stats,
    _num_generate_after_first_yes,
    _ordered_tool_names,
    _policy_terminal_decision,
)
from verl_omni.utils.reward_score.agentic_reward import (
    _zero_result as _ca_zero_result,
)

DIMS = ("reflect", "plan", "format", "tool_call", "result")
GOOD_ENOUGH_SCORE_FLOOR = 0.80
# Slightly overweight last-image C/A so GRPO prefers better generate_image
# prompts / useful rewrites over protocol-only dims.
DEFAULT_WEIGHTS: dict[str, float] = {
    "reflect": 1.5,
    "plan": 1.0,
    "format": 1.0,
    "tool_call": 1.0,
    "result": 1.0,
}
# Always emit these so Ray `_postprocess` (keys taken from sample 0) cannot
# KeyError when one rollout is valid and another hits an early-zero path.
_SCHEMA_EXTRAS: dict[str, float | str | int | None] = {
    **{f"reward_{dim}": 0.0 for dim in DIMS},
    "reward_done": 0.0,
    "reward_terminal_no_penalty": 0.0,
    "reward_missing_tools_penalty": 0.0,
    "reward_good_enough_floor_lift": 0.0,
    "score_before_terminal_penalty": 0.0,
    "final_good_enough": -1,
    "reward_correctness": 0.0,
    "reward_aesthetics": 0.0,
    "first_correctness": 0.0,
    "first_aesthetics": 0.0,
    "reward_reflect_delta": 0.0,
    "rewrite_improve_frac": 0.0,
    "n_images_to_best": 0,
    "terminal_done": 0,
    "terminal_policy_reflection": 0,
    "forced_reflection_context": 0,
    "n_successful_generates": 0,
    "expected_num_images": 0,
    "task_type": "",
}


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    out = _ca_zero_result(method=method)
    out.update(_SCHEMA_EXTRAS)
    # Keep last-image C/A (and first-image / delta) so WandB can plot continuous
    # quality. Facet breakdowns from PR 1 remain unused here.
    for key in list(out):
        if key.startswith("reward_correctness_") or key.startswith("reward_aesthetics_"):
            out.pop(key, None)
    return out


_PLAN_HEADER_RE = re.compile(r"\bPlan\s*:", re.IGNORECASE)
_PLAN_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_FINDINGS_RE = re.compile(r"(?im)^\s*(?:findings|suggested_fixes)\s*:\s*(.*)$")
_TOKEN_RE = re.compile(r"[a-z0-9_']+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _coverage(candidate: str, reference: str) -> float:
    """Fraction of the reference tokens covered by the candidate."""
    ref_tokens = _tokens(reference)
    if not ref_tokens:
        return 0.0
    return len(_tokens(candidate) & ref_tokens) / len(ref_tokens)


def _best_coverage(candidates: list[str], reference: str) -> float:
    if not candidates:
        return 0.0
    return max(_coverage(candidate, reference) for candidate in candidates)


def _count_successful_generates(text: str) -> int:
    """Count successful live ``generate_image`` PNGs (``agentic_tool ok=1``)."""
    n = 0
    for line in (text or "").splitlines():
        if _TOOL_OK.search(line) and any(path.lower().endswith(".png") for path in _PATH_RE.findall(line)):
            n += 1
    return n


def _judge_feedback_text(text: str) -> str:
    """Concatenated findings/suggested_fixes from judge observations."""
    return " ".join(match.group(1) for match in _FINDINGS_RE.finditer(text or "")).strip()


def _extract_plan_lines(text: str) -> list[str]:
    """Numbered/bulleted plan items from the agent's prose (tool blocks stripped)."""
    prose = _assistant_prose(text)
    header = _PLAN_HEADER_RE.search(prose)
    body = prose[header.end() :] if header else prose
    lines = [match.group(1).strip() for match in _PLAN_ITEM_RE.finditer(body)]
    return [line for line in lines if len(_tokens(line)) >= 4]


def _reflection_text(blob: str) -> str:
    """Policy-sampled ``Reflection:`` prose (injected cues already stripped)."""
    prose = _assistant_prose(blob)
    match = re.search(r"\bReflection\s*:(.*?)(?:\bDone\.\s*$|$)", prose, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _reference_reflection_text(gt: dict[str, Any]) -> str:
    steps = gt.get("reference_steps") or []
    return " ".join(str(step.get("reflection") or "") for step in steps if isinstance(step, dict)).strip()


def _judge_ca_series(blob: str) -> list[tuple[float, float]]:
    """Successful ``judge_image`` C/A pairs in order. ``good_enough`` is ignored."""
    return [(float(c), float(a)) for c, a, _, _ in _iter_successful_judge_scores(blob)]


def _first_last_judge_ca(blob: str) -> tuple[float, float, float, float]:
    """Continuous first/last-image C/A. Zeros when no successful judge."""
    series = _judge_ca_series(blob)
    if not series:
        return 0.0, 0.0, 0.0, 0.0
    first_c, first_a = series[0]
    last_c, last_a = series[-1]
    return first_c, first_a, last_c, last_a


def _rewrite_improve_stats(blob: str) -> tuple[float, int]:
    """``(frac of judge-to-judge C/A lifts, 1-indexed judge of max C/A)``."""
    series = _judge_ca_series(blob)
    if not series:
        return 0.0, 0
    cas = [0.5 * (c + a) for c, a in series]
    if len(cas) >= 2:
        n_up = sum(1 for i in range(1, len(cas)) if cas[i] > cas[i - 1])
        frac = n_up / float(len(cas) - 1)
    else:
        frac = 0.0
    best_i = max(range(len(cas)), key=lambda i: (cas[i], -i))
    return float(frac), int(best_i + 1)


def _reflection_reward(blob: str, *, gt: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """``(R_reflect, last_c, last_a, first_c, first_a)``.

    Last-image quality is the mean of the **final** successful judge's C/A
    (continuous in [0, 1]). Lexical coverage of GT / judge feedback is a light
    regularizer, not a 50/50 mix. Rewrite delta ``max(0, last_CA - first_CA)``
    credits improvements without punishing a strong first image.
    """
    first_c, first_a, last_c, last_a = _first_last_judge_ca(blob)
    last_ca = 0.5 * (last_c + last_a)
    first_ca = 0.5 * (first_c + first_a)
    delta = max(0.0, last_ca - first_ca)

    coverage = 0.0
    reference = _reference_reflection_text(gt)
    if reference:
        coverage = _coverage(_reflection_text(blob), reference)
    else:
        feedback = _judge_feedback_text(blob)
        if feedback:
            coverage = _coverage(_reflection_text(blob), feedback)

    if last_ca <= 0.0 and coverage <= 0.0:
        return 0.0, last_c, last_a, first_c, first_a
    r_reflect = 0.70 * last_ca + 0.20 * coverage + 0.10 * delta
    return float(r_reflect), last_c, last_a, first_c, first_a


def _plan_reward(blob: str, *, gt: dict[str, Any]) -> float:
    """R_plan"""
    subtasks = [str(s).strip() for s in (gt.get("reference_subtasks") or []) if str(s).strip()]
    if not subtasks:
        return 0.0
    plan_lines = _extract_plan_lines(blob)
    if not plan_lines:
        return 0.0
    return sum(_best_coverage(plan_lines, subtask) for subtask in subtasks) / len(subtasks)


def _format_reward(blob: str, *, task_type: str, n_successful_gens: int) -> float:
    checks: list[bool] = []
    raw_blocks = len(_TOOL_CALL_RE.findall(blob))
    calls = _extract_tool_calls(blob)
    checks.append(raw_blocks > 0 and len(calls) == raw_blocks)  # well-formed tool calls
    names = _ordered_tool_names(calls)
    gen_idxs = [i for i, name in enumerate(names) if name == "generate_image"]
    judge_idxs = [i for i, name in enumerate(names) if name == "judge_image"]
    checks.append(n_successful_gens >= 1)
    checks.append(bool(judge_idxs) and (not gen_idxs or max(judge_idxs) > max(gen_idxs)))
    terminal_done, policy_reflection, _ = _policy_terminal_decision(blob)
    checks.append(terminal_done)
    if task_type == "plan":
        checks.append(bool(_extract_plan_lines(blob)))
        checks.append(policy_reflection)
    else:
        checks.append(_has_agent_reflection_prose(_assistant_prose(blob)))
    return sum(checks) / len(checks)


def _result_reward(
    blob: str,
    *,
    task_type: str,
    expected: int,
    n_successful_gens: int,
    terminal_done: bool,
    blocked: bool,
) -> float:
    if blocked or not terminal_done or n_successful_gens < 1:
        return 0.0
    if task_type == "plan":
        return 1.0 if n_successful_gens == expected else 0.0
    # Reflect rows (lenient stop-validity): stop is valid when the final judge
    # said YES (early stop) or the generated count stays within the reference.
    hits = _iter_successful_judge_scores(blob)
    last_yes = bool(hits) and hits[-1][2] is True
    return 1.0 if (n_successful_gens <= expected or last_yes) else 0.0


def _env_weight_override(dim: str) -> float | None:
    """``RPCO_W_REFLECT`` / ``RPCO_W_PLAN`` / … beat parquet-baked ``w_*``."""
    raw = os.environ.get(f"RPCO_W_{dim.upper()}")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _weight_raw(gt: dict[str, Any], extra_info: dict[str, Any], dim: str) -> Any:
    env_w = _env_weight_override(dim)
    if env_w is not None:
        return env_w
    raw = extra_info.get(f"w_{dim}")
    if raw is None:
        raw = gt.get(f"w_{dim}")
    # Existing UniCoT parquet baked ``w_tool`` before the dim was renamed.
    if dim == "tool_call" and raw is None:
        raw = extra_info.get("w_tool")
        if raw is None:
            raw = gt.get("w_tool")
    return raw


def _active_weights(gt: dict[str, Any], extra_info: dict[str, Any], *, task_type: str) -> dict[str, float]:
    weights = {}
    for dim in DIMS:
        if dim == "plan" and task_type != "plan":
            continue
        raw = _weight_raw(gt, extra_info, dim)
        default = DEFAULT_WEIGHTS.get(dim, 1.0)
        try:
            weight = float(raw if raw is not None else default)
        except (TypeError, ValueError):
            weight = default
        if weight > 0.0:
            weights[dim] = weight
    return weights


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str | int | None]:
    """Score an agentic trajectory with the RPCO multi-dimensional reward set."""
    del data_source, kwargs
    extra_info = dict(extra_info or {})
    gt = _as_dict(ground_truth)
    blob = solution_str or ""

    task_type = str(gt.get("task_type") or extra_info.get("task_type") or "reflect")
    if task_type not in {"reflect", "plan"}:
        task_type = "reflect"
    try:
        expected = int(gt.get("expected_num_images", extra_info.get("expected_num_images", 1)))
    except (TypeError, ValueError):
        expected = 1

    if not blob.strip():
        out = _zero_result(method="agentic_multidim_empty")
        out["task_type"] = task_type
        out["expected_num_images"] = int(expected)
        return out

    calls = _extract_tool_calls(blob)
    prompts = _gen_image_prompts(calls)
    names = _ordered_tool_names(calls)
    n_reflect = sum(1 for n in names if n == "judge_image")
    n_judge_ok, n_judge_fail, judge_parse_rate = _judge_parse_stats(blob)
    terminal_done, terminal_policy_reflection, forced_context = _policy_terminal_decision(blob)
    n_successful_gens = _count_successful_generates(blob)
    blocked = bool(_BLOCKED_GENERATE_RE.search(blob))

    out = _zero_result(method="agentic_multidim")
    out["num_hermes_tool_calls"] = int(len(calls))
    out["num_generate_image_prompts"] = int(len(prompts))
    out["num_judge_image_calls"] = int(n_reflect)
    out["judge_parse_ok"] = int(n_judge_ok)
    out["judge_parse_fail"] = int(n_judge_fail)
    out["judge_parse_ok_rate"] = float(judge_parse_rate)
    if not prompts:
        out["rollout_valid"] = 0
        out["score"] = 0.0
        out["task_type"] = task_type
        out["expected_num_images"] = int(expected)
        return out
    if not _has_successful_generated_image(blob):
        out["rollout_valid"] = 0
        out["score"] = 0.0
        out["task_type"] = task_type
        out["expected_num_images"] = int(expected)
        return out

    n_rewrite_after_yes = _num_generate_after_first_yes(blob, calls)
    # PR 1's closed-loop indicator (agentic_reward.py ``f_done``): logged to
    # WandB as ``reward_done``; the count-match R_result keeps its own logic.
    valid_terminal_context = bool(n_judge_ok > 0 and not blocked and n_rewrite_after_yes == 0)
    closed = bool(valid_terminal_context and terminal_done and (terminal_policy_reflection or forced_context))
    f_tool_call = 1.0 if calls else 0.0
    f_done = 1.0 if closed else 0.0

    r_reflect, last_c, last_a, first_c, first_a = _reflection_reward(blob, gt=gt)
    rewrite_improve_frac, n_images_to_best = _rewrite_improve_stats(blob)
    rewards = {
        "reflect": r_reflect,
        "plan": _plan_reward(blob, gt=gt),
        "format": _format_reward(blob, task_type=task_type, n_successful_gens=n_successful_gens),
        "tool_call": f_tool_call,
        "result": _result_reward(
            blob,
            task_type=task_type,
            expected=expected,
            n_successful_gens=n_successful_gens,
            terminal_done=terminal_done,
            blocked=blocked,
        ),
    }
    weights = _active_weights(gt, extra_info, task_type=task_type)
    w_sum = sum(weights.values())
    base_score = sum(weights[dim] * rewards[dim] for dim in weights) / w_sum if w_sum > 0 else 0.0
    judge_hits = _iter_successful_judge_scores(blob)
    final_good_enough = judge_hits[-1][2] if judge_hits else None
    # This is deliberately outside the weighted average. A terminal NO is task
    # failure regardless of high C/A or protocol-component rewards. Subtracting
    # one keeps continuous ordering among failed samples while ensuring every
    # failed sample scores <= 0 and every successful sample remains unchanged.
    terminal_no_penalty = -1.0 if final_good_enough is False else 0.0
    # Hard protocol gate: both tools must appear as parsed Hermes calls.
    # Format/result soft-miss alone left generate-only trajectories at ~+0.35,
    # which beat judge+NO (~-0.4) and collapsed the closed loop after force-first.
    called_generate = any(name == "generate_image" for name in names)
    called_judge = any(name == "judge_image" for name in names)
    missing_tools_penalty = 0.0 if (called_generate and called_judge) else -1.0
    penalized_score = base_score + terminal_no_penalty + missing_tools_penalty
    # A parsed live YES is the scarce task-success event. Preserve that signal
    # even when it arrives at the turn cap before a perfect terminal suffix.
    # Require both parsed calls and a successful image so prose cannot fake it.
    valid_good_enough = bool(
        final_good_enough is True
        and called_generate
        and called_judge
        and n_successful_gens >= 1
        and n_judge_ok > 0
        and not blocked
    )
    good_enough_floor_lift = max(0.0, GOOD_ENOUGH_SCORE_FLOOR - penalized_score) if valid_good_enough else 0.0
    score = max(-1.0, min(1.0, penalized_score + good_enough_floor_lift))

    out.update(
        {
            "score": float(score),
            "score_before_terminal_penalty": float(base_score),
            "reward_terminal_no_penalty": float(terminal_no_penalty),
            "reward_missing_tools_penalty": float(missing_tools_penalty),
            "reward_good_enough_floor_lift": float(good_enough_floor_lift),
            "final_good_enough": (
                1 if final_good_enough is True else 0 if final_good_enough is False else -1
            ),
            **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
            "reward_done": float(f_done),
            "reward_correctness": float(last_c),
            "reward_aesthetics": float(last_a),
            "first_correctness": float(first_c),
            "first_aesthetics": float(first_a),
            "reward_reflect_delta": float(max(0.0, 0.5 * (last_c + last_a) - 0.5 * (first_c + first_a))),
            "rewrite_improve_frac": float(rewrite_improve_frac),
            "n_images_to_best": int(n_images_to_best),
            "protocol_ok": int(rewards["format"] == 1.0),
            "rewrite_after_yes": int(n_rewrite_after_yes),
            "rollout_valid": 1,
            "terminal_done": int(terminal_done),
            "terminal_policy_reflection": int(terminal_policy_reflection),
            "forced_reflection_context": int(forced_context),
            "n_successful_generates": int(n_successful_gens),
            "expected_num_images": int(expected),
            "task_type": task_type,
            "method": "agentic_multidim",
        }
    )
    # Full schema merge so Ray reward workers never KeyError on missing keys.
    schema = _zero_result(method=str(out.get("method") or "agentic_multidim"))
    schema.update(out)
    return schema
