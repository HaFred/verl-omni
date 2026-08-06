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

Frozen Qwen3-VL at ``AGENTIC_REFLECT_VLM_URL`` is used **only** as the reward
judge for ``reward_correctness`` / ``reward_aesthetics`` (and per-dimension
rubric fields). It is not an agent tool.

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
_TOOL_OBS_LINE = re.compile(
    r"(?im)^.*\b(agentic_tool|agentic_reflect|image_vis=|Frozen (?:diffusion|Qwen)|"
    r"Image reflection vs user request)\b.*$"
)
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
    "apple",
    "red",
    "cabin",
    "sunset",
    "snowy",
    "mountains",
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
    return re.sub(r"\s+", " ", prose).strip()


def _prose_tokens(prose: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (prose or "").lower()))


def _has_agent_reflection_prose(prose: str) -> bool:
    """True when assistant prose mentions visual attributes / Reflection: / rewrite."""
    if not prose:
        return False
    lower = prose.lower()
    if "reflection:" in lower:
        return True
    tokens = _prose_tokens(prose)
    return bool(tokens & _VISUAL_ATTR_LEX)


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


def _vl_judge_correctness_aesthetics(
    text: str,
    *,
    user_request: str,
    image_prompt: str,
) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
    """Call frozen VL on the last successful generate_image PNG.

    Returns ``(None, None, {}, {})`` when URL unset, path missing, or call fails —
    callers must treat C/A as 0.0 (no heuristic fallback for reward).
    """
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
    for _, _, call in calls:
        name = str(call.get("name", "")).lower()
        args = _call_args(call)
        if name == "generate_image" and "prompt" in args and str(args.get("prompt") or "").strip():
            valid += 1
    score = valid / max(1, len(calls))
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
        "num_hermes_tool_calls": 0,
        "num_generate_image_prompts": 0,
        "num_reflect_image_calls": 0,
        "protocol_ok": 0,
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
    # Legacy metric: reflect_image is no longer an agent tool.
    n_reflect = sum(1 for n in names if n == "reflect_image")
    f_tool_call = 1.0 if calls else 0.0
    f_brevity = _score_brevity(blob)

    if not prompts:
        out = _zero_result(method="agentic_hermes_tool_calls")
        out["reward_tool_call"] = float(f_tool_call)
        out["reward_brevity"] = float(f_brevity)
        out["num_hermes_tool_calls"] = int(len(calls))
        out["num_reflect_image_calls"] = int(n_reflect)
        return out

    user_request = _user_request_from_gt(gt, extra_info)
    last_c, last_a, correctness_scores, aesthetics_scores = _vl_judge_correctness_aesthetics(
        blob,
        user_request=user_request,
        image_prompt=prompts[-1] if prompts else "",
    )

    f_format = _score_format(blob, calls)
    f_reflect = _score_reflection(blob, prompts)
    f_tool = _score_tool_usage(prompts, blob)
    f_result = _score_result(
        blob,
        prompts,
        last_correctness=last_c,
        last_aesthetics=last_a,
    )

    w_tool_call = float(extra_info.get("w_tool_call", gt.get("w_tool_call", 0.05)))
    w_brevity = float(extra_info.get("w_brevity", gt.get("w_brevity", 0.05)))
    w_format = float(extra_info.get("w_format", gt.get("w_format", 0.05)))
    w_reflect = float(extra_info.get("w_reflect", gt.get("w_reflect", 0.10)))
    w_tool = float(extra_info.get("w_tool", gt.get("w_tool", 0.10)))
    w_result = float(extra_info.get("w_result", gt.get("w_result", 0.05)))
    w_correctness = float(extra_info.get("w_correctness", gt.get("w_correctness", 0.30)))
    w_aesthetics = float(extra_info.get("w_aesthetics", gt.get("w_aesthetics", 0.30)))
    w_sum = w_tool_call + w_brevity + w_format + w_reflect + w_tool + w_result + w_correctness + w_aesthetics
    if w_sum <= 0:
        (
            w_tool_call,
            w_brevity,
            w_format,
            w_reflect,
            w_tool,
            w_result,
            w_correctness,
            w_aesthetics,
            w_sum,
        ) = (
            0.05,
            0.05,
            0.05,
            0.10,
            0.10,
            0.05,
            0.30,
            0.30,
            1.0,
        )

    prose = _assistant_prose(blob)
    has_refl = _has_agent_reflection_prose(prose)
    has_done = bool(_DONE_RE.search(prose))
    closed = has_refl and has_done
    distinct = len(prompts) >= 2 and prompts[0].lower().strip() != prompts[-1].lower().strip()
    f_correctness = float(last_c if last_c is not None else 0.0)
    f_aesthetics = float(last_a if last_a is not None else 0.0)
    ca_ok = last_c is not None and last_a is not None and last_c >= 0.70 and last_a >= 0.70

    # High tier only for protocol_ok. Gen without reflection prose is starved.
    # protocol_ok = closed loop (reflection + Done; single-pass or distinct rewrite).
    if closed and f_reflect >= 0.7 and (len(prompts) == 1 or distinct):
        protocol_ok = 1
        if ca_ok:
            base, scale = 0.10, 0.90
        else:
            # Closed loop but weak/missing C/A: protocol alone cannot dominate.
            base, scale = 0.05, 0.65
    elif has_refl and len(prompts) >= 1:
        base, scale = 0.05, 0.45
        protocol_ok = 0
    else:
        # generate_image without agent reflection — starve this plateau.
        base, scale = 0.02, 0.03
        protocol_ok = 0

    total = base + scale * (
        (
            w_tool_call * f_tool_call
            + w_brevity * f_brevity
            + w_format * f_format
            + w_reflect * f_reflect
            + w_tool * f_tool
            + w_result * f_result
            + w_correctness * f_correctness
            + w_aesthetics * f_aesthetics
        )
        / w_sum
    )

    result: dict[str, float | str | int | None] = {
        "score": float(total),
        "reward_tool_call": float(f_tool_call),
        "reward_brevity": float(f_brevity),
        "reward_format": float(f_format),
        "reward_reflection": float(f_reflect),
        "reward_tool_usage": float(f_tool),
        "reward_result": float(f_result),
        "reward_correctness": f_correctness,
        "reward_aesthetics": f_aesthetics,
        "num_hermes_tool_calls": int(len(calls)),
        "num_generate_image_prompts": int(len(prompts)),
        "num_reflect_image_calls": int(n_reflect),
        "protocol_ok": int(protocol_ok),
        "method": "agentic_hermes_tool_calls",
    }
    result.update(
        {f"reward_correctness_{key}": float(correctness_scores.get(key, 0.0)) for key in _CORRECTNESS_DIMENSIONS}
    )
    result.update(
        {f"reward_aesthetics_{key}": float(aesthetics_scores.get(key, 0.0)) for key in _AESTHETICS_DIMENSIONS}
    )
    return result
