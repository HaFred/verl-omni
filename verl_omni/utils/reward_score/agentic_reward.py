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
"""Scalar reward: ``generate_image`` + actor self-reflection protocol.

Protocol (gated):
  generate_image → (image obs attached)
  actor writes a short reflection, then either ``Done.`` OR rewrite +
  ``generate_image`` in the **same** assistant turn.

Frozen Qwen3-VL serves dual role: (1) in-turn ``judge_image`` agent tool
(structured VL feedback the agent reads before deciding Done / rewrite), and
(2) reward C/A for ``reward_correctness`` / ``reward_aesthetics``. Reward prefers
scores from the first ``good_enough=YES`` ``agentic_judge ok=1`` observation
(protocol: YES → Done); otherwise the last successful judge. This blocks
rewrite-after-YES roulette from replacing a good C/A with a failed last image.
If absent, it falls back to ``call_reflect_vlm`` via ``AGENTIC_VLLM_URL``
(OpenAI chat) or legacy ``AGENTIC_REFLECT_VLM_URL``.

``reward_reflection`` scores **agent prose** after generate (visual attributes /
rewrite / Done) — not frozen-tool markers.

``reward_tool_call`` is a per-rollout binary (1 if the trajectory contains at
least one parseable ``<tool_call>``, else 0), matching ``decode_has_tool_call``.

``reward_brevity`` scores assistant prose only (tool calls and tool
observations stripped). Target: ≤4 short sentences / ≤~280 chars of prose.

Tiers:
  0 generate_image                        → 0
  gen without reflection prose            → ~0.02–0.05 (starved)
  gen + reflection, open (no Done)        → mid
  closed loop (reflection + Done) + C/A   → high (protocol_ok)
"""

from __future__ import annotations

import json
import re
from typing import Any

from verl_omni.utils.reward_score.vl_reflect_client import call_reflect_vlm

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.IGNORECASE | re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL)
_BARE_JSON = re.compile(
    r'(?<!<tool_call>\s)\{\s*"name"\s*:\s*"generate_image"',
    re.IGNORECASE,
)
_TOOL_OK = re.compile(r"agentic_tool\s+ok=1", re.IGNORECASE)
_PATH_RE = re.compile(r"path=([^\s'\"]+)", re.IGNORECASE)
_DONE_RE = re.compile(r"\bDone\.?\b", re.IGNORECASE)
# Tool observations only — do NOT match agent prose that merely quotes "VL judge".
_TOOL_OBS_LINE = re.compile(
    r"(?im)^(?!.*\bReflection\s*:).*\b("
    r"agentic_tool|agentic_reflect|agentic_judge|"
    r"VL judge on the last generated image|"
    r"image_vis=|Frozen (?:diffusion|Qwen)|Image reflection vs user request"
    r")\b.*$"
)
_AGENT_REFLECTION_MARKER_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)
_CORRECTNESS_DIMENSIONS = (
    "subject_entities",
    "attributes",
    "relations_layout",
    "scene_context",
    "completeness",
)
_AESTHETICS_DIMENSIONS = (
    "composition",
    "lighting",
    "color",
    "fidelity",
    "appeal",
)
_VISUAL_ATTR_LEX = {
    "bright",
    "brighter",
    "dark",
    "color",
    "colors",
    "sharp",
    "sharper",
    "soft",
    "muted",
    "vivid",
    "lighting",
    "light",
    "edge",
    "edges",
    "composition",
    "contrast",
    "focus",
    "blur",
    "luma",
    "detail",
    "detailed",
    "rich",
    "richer",
    "match",
    "matches",
    "visible",
    "missing",
    "present",
    "figure",
    "figures",
    "rewrite",
    "rewritten",
    "reflection",
    "aesthetics",
    "correctness",
}
_REWRITE_LEX = {
    "rewrite",
    "rewritten",
    "brighter",
    "sharper",
    "richer",
    "increase",
    "add",
    "fix",
    "refine",
    "vivid",
    "detailed",
    "contrast",
}
_REFINE_LEX = {
    "detailed",
    "lighting",
    "composition",
    "focus",
    "texture",
    "color",
    "sharp",
    "richer",
    "coherent",
    "bright",
    "vivid",
    "contrast",
}


def _as_dict(ground_truth: Any) -> dict[str, Any]:
    if ground_truth is None:
        return {}
    if isinstance(ground_truth, dict):
        return dict(ground_truth)
    if isinstance(ground_truth, str):
        raw = ground_truth.strip()
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"user_request": raw}
    return {}


def _extract_tool_calls(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    """Parse Hermes JSON and/or Qwen3.5 XML tool calls inside ``<tool_call>`` blocks."""
    out: list[tuple[int, int, dict[str, Any]]] = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        body = (match.group(1) or "").strip()
        call = _parse_tool_call_body(body)
        if call is not None:
            out.append((match.start(), match.end(), call))
    return out


def _parse_tool_call_body(body: str) -> dict[str, Any] | None:
    if not body:
        return None
    # Hermes: {"name": "...", "arguments": {...}}
    if body.lstrip().startswith("{"):
        try:
            call = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(call, dict) and call.get("name"):
            return call
        return None
    # Qwen3.5 / Qwen3-Coder XML:
    # <function=name><parameter=k>\nvalue\n</parameter></function>
    fn = _FUNCTION_RE.search(body)
    if fn is None:
        return None
    name = (fn.group(1) or "").strip()
    if not name:
        return None
    args: dict[str, Any] = {}
    for pm in _PARAM_RE.finditer(fn.group(2) or ""):
        key = (pm.group(1) or "").strip()
        if not key:
            continue
        args[key] = (pm.group(2) or "").strip()
    return {"name": name, "arguments": args}


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


def _gen_image_prompts(calls: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    prompts: list[str] = []
    for _, _, call in calls:
        if str(call.get("name", "")).lower() != "generate_image":
            continue
        p = str(_call_args(call).get("prompt") or "").strip()
        if p:
            prompts.append(p)
    return prompts


def _ordered_tool_names(calls: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    return [str(c.get("name", "")).lower() for _, _, c in calls]


def _assistant_prose(text: str) -> str:
    """Strip tool_calls and tool-obs lines; keep private thinking as scored prose."""
    prose = _TOOL_CALL_RE.sub(" ", text or "")
    prose = _TOOL_OBS_LINE.sub(" ", prose)
    prose = re.sub(r"</?think>", " ", prose, flags=re.IGNORECASE)
    # Env-injected Reflection/Done (response_mask=0) must not earn Done credit.
    prose = re.sub(
        r"(?is)\bReflection\s*:.*?(?:agentic_forced_reflection=1|agentic_force_stop_max_passes=1)\S*",
        " ",
        prose,
    )
    return re.sub(r"\s+", " ", prose).strip()


def _prose_tokens(prose: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (prose or "").lower()))


def _has_agent_reflection_prose(prose: str) -> bool:
    """True only for explicit agent ``Reflection:`` prose.

    Visual-attribute lexicon matches alone are *not* enough: VL ``judge_image``
    observations also contain correctness/aesthetics wording and used to falsely
    promote open gen→judge loops into the mid reward tier (~0.4), starving the
    Done. learning signal.
    """
    if not prose:
        return False
    return bool(_AGENT_REFLECTION_MARKER_RE.search(prose))


def _last_successful_generate_image_path(text: str) -> str | None:
    """Prefer the last ``path=`` on an ``agentic_tool ok=1`` line; else last PNG path."""
    last_ok: str | None = None
    last_png: str | None = None
    for line in (text or "").splitlines():
        paths = _PATH_RE.findall(line)
        if not paths:
            continue
        path = paths[-1].strip()
        if path.lower().endswith(".png"):
            last_png = path
        if re.search(r"agentic_tool\s+ok=1", line, re.IGNORECASE):
            last_ok = path
    return last_ok or last_png


def _user_request_from_gt(gt: dict[str, Any], extra_info: dict[str, Any]) -> str:
    for key in ("user_request", "raw_prompt"):
        val = extra_info.get(key) or gt.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


_CORRECTNESS_EQ_RE = re.compile(r"\bcorrectness\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_AESTHETICS_EQ_RE = re.compile(r"\baesthetics\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_GOOD_ENOUGH_EQ_RE = re.compile(r"\bgood_enough\s*=\s*(YES|NO|1|0|true|false)\b", re.IGNORECASE)
_AGENTIC_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
_AGENTIC_JUDGE_PARSE_FAIL_RE = re.compile(r"\bagentic_judge\s+ok=0\b|\bparse_ok\s*=\s*0\b", re.IGNORECASE)


def _judge_parse_stats(text: str) -> tuple[int, int, float]:
    """Return ``(n_ok, n_fail, parse_ok_rate)`` from trajectory judge observations."""
    blob = text or ""
    n_ok = len(_AGENTIC_JUDGE_OK_RE.findall(blob))
    # Prefer explicit ok=0 marker (one per failed judge obs).
    n_fail = len(re.findall(r"\bagentic_judge\s+ok=0\b", blob, flags=re.IGNORECASE))
    if n_fail == 0:
        n_fail = len(_AGENTIC_JUDGE_PARSE_FAIL_RE.findall(blob))
    n_attempts = n_ok + n_fail
    rate = float(n_ok) / float(n_attempts) if n_attempts else 1.0
    return n_ok, n_fail, rate


def _good_enough_from_window(window: str) -> bool | None:
    m = _GOOD_ENOUGH_EQ_RE.search(window or "")
    if m is None:
        return None
    tok = m.group(1).strip().lower()
    if tok in {"yes", "1", "true"}:
        return True
    if tok in {"no", "0", "false"}:
        return False
    return None


def _iter_successful_judge_scores(text: str) -> list[tuple[float, float, bool | None, int]]:
    """Yield ``(c, a, good_enough, end_pos)`` for each successful judge obs."""
    blob = text or ""
    out: list[tuple[float, float, bool | None, int]] = []
    for match in _AGENTIC_JUDGE_OK_RE.finditer(blob):
        window = blob[max(0, match.start() - 1400) : match.end()]
        if "VL judge" not in window and "correctness" not in window.lower():
            continue
        if re.search(r"\bparse_ok\s*=\s*0\b", window, re.IGNORECASE):
            continue
        c_hits = list(_CORRECTNESS_EQ_RE.finditer(window))
        a_hits = list(_AESTHETICS_EQ_RE.finditer(window))
        if not c_hits or not a_hits:
            continue
        try:
            c = max(0.0, min(1.0, float(c_hits[-1].group(1))))
            a = max(0.0, min(1.0, float(a_hits[-1].group(1))))
        except (TypeError, ValueError):
            continue
        out.append((c, a, _good_enough_from_window(window), match.end()))
    return out


def _parse_last_agentic_judge_scores(
    text: str,
) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
    """Reuse C/A from judge obs: first ``good_enough=YES``, else last ok=1.

    Only trusts windows ending in ``agentic_judge ok=1`` (written by our tool),
    not bare ``correctness=`` markers the policy might hallucinate. Parse
    failures (``parse_ok=0`` / unparseable) never contribute C/A.
    """
    hits = _iter_successful_judge_scores(text)
    if not hits:
        return None, None, {}, {}
    for c, a, good_enough, _ in hits:
        if good_enough is True:
            return c, a, {}, {}
    c, a, _, _ = hits[-1]
    return c, a, {}, {}


def _num_generate_after_first_yes(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> int:
    """Count ``generate_image`` calls after the first ``good_enough=YES`` judge."""
    yes_pos: int | None = None
    for _, _, good_enough, end_pos in _iter_successful_judge_scores(text):
        if good_enough is True:
            yes_pos = end_pos
            break
    if yes_pos is None:
        return 0
    n = 0
    for start, _, call in calls:
        if start <= yes_pos:
            continue
        if str(call.get("name", "")).lower() == "generate_image":
            n += 1
    return n


def _delta_c_bonus(text: str, preferred_c: float) -> tuple[float, float | None, bool]:
    """Bonus for lifting C after a first-pass failure (``good_enough=NO``).

    Returns ``(f_delta_c in [0,1], first_c or None, first_was_no)``.
    """
    hits = _iter_successful_judge_scores(text)
    if not hits:
        return 0.0, None, False
    first_c, _, first_ge, _ = hits[0]
    first_was_no = first_ge is False
    if not first_was_no:
        return 0.0, float(first_c), False
    delta = max(0.0, min(1.0, float(preferred_c) - float(first_c)))
    return delta, float(first_c), True


def _vl_judge_correctness_aesthetics(
    text: str,
    *,
    user_request: str,
    image_prompt: str,
) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
    """Resolve C/A from trajectory judge obs, else re-call frozen VL on last PNG.

    Returns ``(None, None, {}, {})`` when neither source works — callers must
    treat C/A as 0.0 (no heuristic fallback for reward).
    """
    parsed = _parse_last_agentic_judge_scores(text)
    if parsed[0] is not None and parsed[1] is not None:
        return parsed

    image_path = _last_successful_generate_image_path(text)
    if not image_path:
        return None, None, {}, {}
    scored = call_reflect_vlm(
        user_request=user_request or "",
        image_prompt=image_prompt or "",
        notes="",
        image_path=image_path,
    )
    if scored is None:
        return None, None, {}, {}
    return (
        float(scored["correctness"]),
        float(scored["aesthetics"]),
        dict(scored.get("correctness_scores") or {}),
        dict(scored.get("aesthetics_scores") or {}),
    )


def _score_brevity(text: str) -> float:
    """Reward short assistant prose; pure tool-call trajectories score 1.0."""
    prose = _assistant_prose(text)
    if not prose:
        return 1.0
    sentences = [s for s in re.split(r"[.!?]+", prose) if s.strip()]
    n_sent = len(sentences)
    n_chars = len(prose)
    # Soft target matches the ≤4-sentence prompt reminder.
    sent_score = 1.0 if n_sent <= 4 else max(0.0, 1.0 - 0.15 * (n_sent - 4))
    char_score = 1.0 if n_chars <= 280 else max(0.0, 1.0 - (n_chars - 280) / 720.0)
    return float(min(1.0, 0.5 * sent_score + 0.5 * char_score))


def _score_format(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> float:
    if not calls:
        return 0.0
    valid = 0
    gen_calls = 0
    for _, _, call in calls:
        name = str(call.get("name", "")).lower()
        args = _call_args(call)
        if name == "generate_image":
            gen_calls += 1
            if "prompt" in args and str(args.get("prompt") or "").strip():
                valid += 1
        elif name == "judge_image":
            # judge_image is always well-formed (user_request + image_prompt);
            # don't penalize it in the format denominator.
            pass
    score = valid / max(1, gen_calls) if gen_calls else 0.0
    if _BARE_JSON.search(text or ""):
        score *= 0.5
    return float(min(1.0, score))


def _score_reflection(text: str, prompts: list[str]) -> float:
    """Score actor self-reflection prose (visual attrs / rewrite / Done)."""
    if not prompts:
        return 0.0
    prose = _assistant_prose(text)
    if not _has_agent_reflection_prose(prose):
        return 0.0

    score = 0.45
    tokens = _prose_tokens(prose)
    if "reflection:" in prose.lower():
        score += 0.10
    if tokens & _VISUAL_ATTR_LEX:
        score += 0.15
    if _DONE_RE.search(prose):
        score += 0.20
    if len(prompts) >= 2:
        distinct = prompts[0].lower().strip() != prompts[-1].lower().strip()
        if distinct and (tokens & _REWRITE_LEX or tokens & _REFINE_LEX):
            score += 0.10
        elif distinct:
            score += 0.05
    else:
        score += 0.05
    return float(min(1.0, score))


def _score_tool_usage(prompts: list[str], text: str) -> float:
    if len(prompts) == 0:
        return 0.0
    prose = _assistant_prose(text)
    has_refl = _has_agent_reflection_prose(prose)
    # Gen-only / Done-without-reflection is near-zero so GRPO cannot plateau.
    if not has_refl:
        return 0.05

    score = 0.55
    if _DONE_RE.search(prose):
        score = 0.75
    if len(prompts) >= 2:
        distinct = prompts[0].lower().strip() != prompts[-1].lower().strip()
        if distinct:
            t0 = set(re.findall(r"[a-z0-9]+", prompts[0].lower()))
            t1 = set(re.findall(r"[a-z0-9]+", prompts[-1].lower()))
            if len(t1) > len(t0) or (t1 & _REFINE_LEX):
                score = 1.0 if _DONE_RE.search(prose) else 0.90
            else:
                score = max(score, 0.85)
        else:
            score = min(score, 0.40)
    elif len(prompts) == 1:
        score = 0.90 if _DONE_RE.search(prose) else 0.60
    return float(min(1.0, score))


def _score_result(
    text: str,
    prompts: list[str],
    *,
    last_correctness: float | None,
    last_aesthetics: float | None,
) -> float:
    if not prompts:
        return 0.0
    ok = len(_TOOL_OK.findall(text or ""))
    prose = _assistant_prose(text)
    has_refl = _has_agent_reflection_prose(prose)
    if not has_refl:
        return 0.05 if (len(prompts) >= 1 and ok >= 1) else 0.0

    closed = bool(_DONE_RE.search(prose))
    ca = None
    if last_correctness is not None and last_aesthetics is not None:
        ca = 0.5 * (last_correctness + last_aesthetics)

    if ca is not None and closed:
        return float(min(1.0, 0.35 + 0.65 * ca))
    if ca is not None and ok >= 1:
        return float(min(1.0, 0.25 + 0.55 * ca))
    if closed:
        return 0.70
    if ok >= 1:
        return 0.45
    if len(prompts) >= 1 and ok >= 1:
        return 0.25
    return 0.10 if len(prompts) >= 1 else 0.0


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    result: dict[str, float | str | int | None] = {
        "score": 0.0,
        "reward_tool_call": 0.0,
        "reward_brevity": 0.0,
        "reward_format": 0.0,
        "reward_reflection": 0.0,
        "reward_tool_usage": 0.0,
        "reward_result": 0.0,
        "reward_correctness": 0.0,
        "reward_aesthetics": 0.0,
        "reward_done": 0.0,
        "num_hermes_tool_calls": 0,
        "num_generate_image_prompts": 0,
        "num_judge_image_calls": 0,
        "judge_parse_ok": 0,
        "judge_parse_fail": 0,
        "judge_parse_ok_rate": 1.0,
        "protocol_ok": 0,
        "rewrite_after_yes": 0,
        "reward_delta_c": 0.0,
        "first_correctness": 0.0,
        "first_judge_no": 0,
        "rollout_valid": 0,
        "method": method,
    }
    result.update({f"reward_correctness_{key}": 0.0 for key in _CORRECTNESS_DIMENSIONS})
    result.update({f"reward_aesthetics_{key}": 0.0 for key in _AESTHETICS_DIMENSIONS})
    return result


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str | int | None]:
    """Score an agentic image-generation trajectory for GRPO."""
    del data_source, kwargs
    extra_info = dict(extra_info or {})
    gt = _as_dict(ground_truth)

    blob = solution_str or ""
    if not blob.strip():
        return _zero_result(method="agentic_hermes_tool_calls")

    calls = _extract_tool_calls(blob)
    prompts = _gen_image_prompts(calls)
    names = _ordered_tool_names(calls)
    # Count judge_image tool calls alongside generate_image calls.
    n_reflect = sum(1 for n in names if n == "judge_image")
    f_tool_call = 1.0 if calls else 0.0
    f_brevity = _score_brevity(blob)

    if not prompts:
        # No generate_image → invalid rollout for GRPO (masked out of the update).
        out = _zero_result(method="agentic_no_generate")
        out["reward_tool_call"] = float(f_tool_call)
        out["reward_brevity"] = float(f_brevity)
        out["num_hermes_tool_calls"] = int(len(calls))
        out["num_judge_image_calls"] = int(n_reflect)
        out["rollout_valid"] = 0
        out["score"] = 0.0
        return out

    user_request = _user_request_from_gt(gt, extra_info)
    last_c, last_a, correctness_scores, aesthetics_scores = _vl_judge_correctness_aesthetics(
        blob,
        user_request=user_request,
        image_prompt=prompts[-1] if prompts else "",
    )
    n_judge_ok, n_judge_fail, judge_parse_rate = _judge_parse_stats(blob)
    # No successful parse anywhere → keep C/A at 0 (do not invent scores).
    if last_c is None and last_a is None and n_judge_fail > 0 and n_judge_ok == 0:
        last_c, last_a = 0.0, 0.0
        correctness_scores, aesthetics_scores = {}, {}

    f_format = _score_format(blob, calls)
    f_reflect = _score_reflection(blob, prompts)
    f_tool = _score_tool_usage(prompts, blob)
    f_result = _score_result(
        blob,
        prompts,
        last_correctness=last_c,
        last_aesthetics=last_a,
    )

    w_tool_call = float(extra_info.get("w_tool_call", gt.get("w_tool_call", 0.10)))
    w_correctness = float(extra_info.get("w_correctness", gt.get("w_correctness", 0.35)))
    w_aesthetics = float(extra_info.get("w_aesthetics", gt.get("w_aesthetics", 0.35)))
    # Closed-loop Done. must dominate open gen→judge loops (was only 0.10 and
    # got swamped by already-high C/A from the frozen judge).
    w_done = float(extra_info.get("w_done", gt.get("w_done", 0.20)))
    # Multiturn headroom: reward C lift after a failed first judge (NO → rewrite).
    w_delta_c = float(extra_info.get("w_delta_c", gt.get("w_delta_c", 0.15)))
    w_sum = w_tool_call + w_correctness + w_aesthetics + w_done
    if w_sum <= 0:
        w_tool_call, w_correctness, w_aesthetics, w_done, w_sum = 0.10, 0.35, 0.35, 0.20, 1.0

    prose = _assistant_prose(blob)
    has_refl = _has_agent_reflection_prose(prose)
    has_done = bool(_DONE_RE.search(prose))
    closed = has_refl and has_done
    distinct = len(prompts) >= 2 and prompts[0].lower().strip() != prompts[-1].lower().strip()
    f_correctness = float(last_c if last_c is not None else 0.0)
    f_aesthetics = float(last_a if last_a is not None else 0.0)
    # Mix terms: VL quality only fully counts after Reflection+Done. Open loops
    # keep a tiny fraction so C/A still appears in logs / weak ranking, but the
    # mean score cannot plateau near ~0.45 without learning Done.
    ca_mix_scale = 1.0 if closed else 0.05
    f_correctness_mix = ca_mix_scale * f_correctness
    f_aesthetics_mix = ca_mix_scale * f_aesthetics
    f_done = 1.0 if closed else (0.25 if has_done else 0.0)
    ca_ok = last_c is not None and last_a is not None and last_c >= 0.70 and last_a >= 0.70
    n_rewrite_after_yes = _num_generate_after_first_yes(blob, calls)
    f_delta_c, first_c, first_judge_no = _delta_c_bonus(blob, f_correctness)
    if not closed:
        # ΔC is a multiturn bonus on top of a closed (or rewrite) protocol.
        f_delta_c = 0.0

    # High tier only for protocol_ok. Gen without Reflection: is starved.
    # protocol_ok = closed loop (Reflection: + Done; single-pass or distinct rewrite).
    if closed and f_reflect >= 0.7 and (len(prompts) == 1 or distinct):
        protocol_ok = 1
        if ca_ok:
            base, scale = 0.10, 0.90
        else:
            # Closed loop but weak/missing C/A: protocol alone cannot dominate.
            base, scale = 0.05, 0.65
    elif has_refl and len(prompts) >= 1:
        # Reflection without Done — keep below closed tier.
        base, scale = 0.04, 0.30
        protocol_ok = 0
    else:
        # generate_image / judge without agent Reflection: — starve this plateau.
        base, scale = 0.02, 0.05
        protocol_ok = 0

    # good_enough=YES means stop. Extra generate_image after YES is protocol break
    # and was the main overfit failure mode (score/C/A drift via rewrite roulette).
    if n_rewrite_after_yes > 0:
        protocol_ok = 0
        base, scale = min(base, 0.05), min(scale, 0.35)
        f_done = min(float(f_done), 0.25)
        # No ΔC credit for gambling after YES.
        f_delta_c = 0.0

    # Scalar: tool_call gate + (gated) VL C/A + closed-loop Done + multiturn ΔC.
    total = base + scale * (
        (
            w_tool_call * f_tool_call
            + w_correctness * f_correctness_mix
            + w_aesthetics * f_aesthetics_mix
            + w_done * f_done
        )
        / w_sum
    )
    total = float(min(1.0, total + w_delta_c * f_delta_c))

    result: dict[str, float | str | int | None] = {
        "score": float(total),
        "reward_tool_call": float(f_tool_call),
        "reward_brevity": float(f_brevity),
        "reward_format": float(f_format),
        "reward_reflection": float(f_reflect),
        "reward_tool_usage": float(f_tool),
        "reward_result": float(f_result),
        # Log raw VL C/A (pre-gate) so WandB tracks image quality separately.
        "reward_correctness": f_correctness,
        "reward_aesthetics": f_aesthetics,
        "reward_done": float(f_done),
        "reward_delta_c": float(f_delta_c),
        "first_correctness": float(first_c if first_c is not None else 0.0),
        "first_judge_no": int(bool(first_judge_no)),
        "num_hermes_tool_calls": int(len(calls)),
        "num_generate_image_prompts": int(len(prompts)),
        "num_judge_image_calls": int(n_reflect),
        "judge_parse_ok": int(n_judge_ok),
        "judge_parse_fail": int(n_judge_fail),
        "judge_parse_ok_rate": float(judge_parse_rate),
        "protocol_ok": int(protocol_ok),
        "rewrite_after_yes": int(n_rewrite_after_yes),
        "rollout_valid": 1,
        "method": "agentic_hermes_tool_calls",
    }
    result.update(
        {f"reward_correctness_{key}": float(correctness_scores.get(key, 0.0)) for key in _CORRECTNESS_DIMENSIONS}
    )
    result.update(
        {f"reward_aesthetics_{key}": float(aesthetics_scores.get(key, 0.0)) for key in _AESTHETICS_DIMENSIONS}
    )
    # Always emit the full schema so Ray reward workers never KeyError on a
    # missing key when batching reward_extra_info (verl takes keys from sample 0).
    schema = _zero_result(method=str(result.get("method") or "agentic_hermes_tool_calls"))
    schema.update(result)
    return schema
