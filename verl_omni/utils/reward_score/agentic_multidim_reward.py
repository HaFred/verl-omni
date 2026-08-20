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
"""Multi-dimensional reward for RPCO stage-3 multi-task RL (PR 2 of RFC #302).

Reward set (VisionCreator-R1, arXiv:2603.08812, §4.1/§4.3):

  R_reflect — satisfied-checkpoint ratio on the final accepted image. The live
      ``judge_image`` facets are the checkpoints; quality is the mean of the
      last successful judge's correctness/aesthetics. When a UniCoT reference
      summary exists (``gt.reference_steps``), lexical coverage of the agent's
      ``Reflection:`` against that reference is blended in (0.5 quality +
      0.5 coverage), grounding the signal in the dataset's eval summaries.
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

Total: ``score = (1/|W|) * sum(w_i * R_i)`` over the active set W (dims with
``w_* > 0`` that apply to the row's task type). Default weights are 1.0 (paper
default); the UniCoT builder bakes per-row weights into ``ground_truth``.
``w_* > 0`` that apply to the row's task type). Default weights are 1.0 (paper
default); the UniCoT builder bakes per-row weights into ``ground_truth``.

Gating kept from PR 1: no ``generate_image`` / no successful PNG → score 0 and
``rollout_valid=0`` (rollout is discarded from the GRPO update). Env-injected
``Reflection`` (``agentic_forced_reflection=1``) never earns credit.
"""

from __future__ import annotations

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
    _zero_result as _pr1_zero_result,
)

DIMS = ("reflect", "plan", "format", "tool_call", "result")
# Always emit these so Ray `_postprocess` (keys taken from sample 0) cannot
# KeyError when one rollout is valid and another hits an early-zero path.
_SCHEMA_EXTRAS: dict[str, float | str | int | None] = {
    **{f"reward_{dim}": 0.0 for dim in DIMS},
    "reward_done": 0.0,
    "terminal_done": 0,
    "terminal_policy_reflection": 0,
    "forced_reflection_context": 0,
    "n_successful_generates": 0,
    "expected_num_images": 0,
    "task_type": "",
}


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    out = _pr1_zero_result(method=method)
    out.update(_SCHEMA_EXTRAS)
    # PR 1 C/A mix terms are not part of the RPCO score; drop the inherited
    # stub zeros so WandB ``agentic_reward/{correctness,aesthetics}`` is not
    # logged as a perpetual all-zero series under this scorer.
    for key in list(out):
        if key == "reward_correctness" or key == "reward_aesthetics" or key.startswith(
            ("reward_correctness_", "reward_aesthetics_")
        ):
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


def _reflection_reward(blob: str, *, gt: dict[str, Any]) -> tuple[float, float]:
    """``(R_reflect, quality)`` — satisfied-checkpoint ratio (+ reference coverage)."""
    hits = _iter_successful_judge_scores(blob)
    quality = 0.0
    if hits:
        for c, a, good_enough, _ in hits:
            if good_enough is True:
                quality = 0.5 * (c + a)
                break
        else:
            c, a, _, _ = hits[-1]
            quality = 0.5 * (c + a)
    reference = _reference_reflection_text(gt)
    if reference:
        coverage = _coverage(_reflection_text(blob), reference)
        return 0.5 * quality + 0.5 * coverage, quality
    feedback = _judge_feedback_text(blob)
    if feedback:
        return 0.5 * quality + 0.5 * _coverage(_reflection_text(blob), feedback), quality
    return quality, quality


def _plan_reward(blob: str, *, gt: dict[str, Any]) -> float:
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


def _weight_raw(gt: dict[str, Any], extra_info: dict[str, Any], dim: str) -> Any:
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
        try:
            weight = float(raw if raw is not None else 1.0)
        except (TypeError, ValueError):
            weight = 1.0
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

    rewards = {
        "reflect": _reflection_reward(blob, gt=gt)[0],
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
    score = sum(weights[dim] * rewards[dim] for dim in weights) / w_sum if w_sum > 0 else 0.0

    out.update(
        {
            "score": float(min(1.0, score)),
            **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
            "reward_done": float(f_done),
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
