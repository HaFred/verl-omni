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
"""Scalar reward: Hermes ``generate_image`` tool-calling first.

Overfit smoke without rollout force: score rises with valid Hermes tool calls.
The full protocol requires a grounded ``Reflection:`` between two distinct
calls. One call still earns bootstrap credit; incomplete spam / no
``<tool_call>`` stays at 0.

Protocol tiers:
  0 Hermes generate_image → 0
  1 valid Hermes call     → ~0.35–0.45 (learn to emit the tool)
  ≥2 distinct calls + reflection → ≥0.55

TODO (fred): multi-dimensional RPCO rewards (VLM reflection / plan / image
quality) — https://github.com/verl-project/verl-omni/issues/303.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_REFLECT_RE = re.compile(r"(?im)^\s*Reflection\s*:\s*(.+)$")
_BARE_JSON = re.compile(r'(?<!<tool_call>\s)\{\s*"name"\s*:\s*"generate_image"', re.IGNORECASE)
_TOOL_OK = re.compile(r"agentic_tool\s+ok=1", re.IGNORECASE)
_IMAGE_REFLECT_LEX = re.compile(
    r"\b(image_vis|mean_luma|edges?|detail|lighting|color|composition|focus|sharp|soft|muted|blur)\b",
    re.IGNORECASE,
)
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
    """Return (start, end, parsed_call) for each Hermes tool-call block."""
    out: list[tuple[int, int, dict[str, Any]]] = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        try:
            call = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(call, dict) and call.get("name"):
            out.append((match.start(), match.end(), call))
    return out


def _gen_image_prompts(calls: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    prompts: list[str] = []
    for _, _, call in calls:
        if str(call.get("name", "")).lower() != "generate_image":
            continue
        args = call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(args, dict):
            p = str(args.get("prompt") or "").strip()
            if p:
                prompts.append(p)
    return prompts


def _gen_image_spans(calls: list[tuple[int, int, dict[str, Any]]]) -> list[tuple[int, int, dict[str, Any]]]:
    return [(s, e, c) for s, e, c in calls if str(c.get("name", "")).lower() == "generate_image"]


def _reflection_between(text: str, gen: list[tuple[int, int, dict[str, Any]]]) -> str | None:
    if len(gen) < 2:
        return None
    between = (text or "")[gen[0][1] : gen[1][0]]
    m = _REFLECT_RE.search(between)
    if not m:
        return None
    body = m.group(1).strip()
    return body or None


def _score_format(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> float:
    if not calls:
        return 0.0
    valid = 0
    for _, _, call in calls:
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if isinstance(args, dict) and "prompt" in args:
            valid += 1
    score = valid / max(1, len(calls))
    if _BARE_JSON.search(text or ""):
        score *= 0.5
    return float(min(1.0, score))


def _score_reflection(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> float:
    gen = _gen_image_spans(calls)
    body = _reflection_between(text, gen)
    if not body:
        return 0.0
    score = 0.55
    if len(body) >= 12:
        score += 0.15
    if _IMAGE_REFLECT_LEX.search(body):
        score += 0.30
    return float(min(1.0, score))


def _score_tool_usage(prompts: list[str]) -> float:
    if len(prompts) == 0:
        return 0.0
    if len(prompts) == 1:
        return 0.55
    if prompts[0].lower().strip() == prompts[1].lower().strip():
        return 0.35
    t0 = set(re.findall(r"[a-z0-9]+", prompts[0].lower()))
    t1 = set(re.findall(r"[a-z0-9]+", prompts[1].lower()))
    if len(t1) > len(t0) or (t1 & _REFINE_LEX):
        return 1.0
    return 0.85


def _score_result(text: str, prompts: list[str], has_good_reflect: bool) -> float:
    if not prompts:
        return 0.0
    ok = len(_TOOL_OK.findall(text or ""))
    last_end = 0
    for m in _TOOL_CALL_RE.finditer(text or ""):
        last_end = max(last_end, m.end())
    after = (text or "")[last_end:].strip()
    has_final = len(after) > 5 and "<tool_call>" not in after.lower()
    if len(prompts) >= 2 and ok >= 2 and has_good_reflect:
        return 1.0
    if len(prompts) >= 2 and ok >= 1:
        return 0.7
    if len(prompts) >= 1 and ok >= 1:
        return 0.55
    if has_final and len(prompts) >= 1:
        return 0.35
    return 0.2 if len(prompts) >= 1 else 0.0


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    return {
        "score": 0.0,
        "reward_format": 0.0,
        "reward_reflection": 0.0,
        "reward_tool_usage": 0.0,
        "reward_result": 0.0,
        "num_hermes_tool_calls": 0,
        "num_generate_image_prompts": 0,
        "protocol_ok": 0,
        "method": method,
    }


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str | int | None]:
    """Score an agentic image-generation trajectory for GRPO (tool-call first)."""
    del data_source, kwargs
    extra_info = dict(extra_info or {})
    gt = _as_dict(ground_truth)

    blob = solution_str or ""
    if not blob.strip():
        return _zero_result(method="agentic_hermes_tool_calls")

    calls = _extract_tool_calls(blob)
    prompts = _gen_image_prompts(calls)
    if not prompts:
        # No Hermes generate_image → hard zero (bare JSON / prose / spam).
        out = _zero_result(method="agentic_hermes_tool_calls")
        out["num_hermes_tool_calls"] = int(len(calls))
        return out

    f_format = _score_format(blob, calls)
    f_reflect = _score_reflection(blob, calls)
    f_tool = _score_tool_usage(prompts)
    f_result = _score_result(blob, prompts, has_good_reflect=f_reflect >= 0.7)

    w_format = float(extra_info.get("w_format", gt.get("w_format", 0.35)))
    w_reflect = float(extra_info.get("w_reflect", gt.get("w_reflect", 0.10)))
    w_tool = float(extra_info.get("w_tool", gt.get("w_tool", 0.45)))
    w_result = float(extra_info.get("w_result", gt.get("w_result", 0.10)))
    w_sum = w_format + w_reflect + w_tool + w_result
    if w_sum <= 0:
        w_format, w_reflect, w_tool, w_result, w_sum = 0.35, 0.10, 0.45, 0.10, 1.0

    # Tiered floor: one call bootstraps discovery, but the overfit target is
    # image-grounded reflection followed by a materially different second call.
    distinct = len(prompts) >= 2 and prompts[0].lower().strip() != prompts[1].lower().strip()
    reflected_rewrite = distinct and f_reflect >= 0.7
    if reflected_rewrite:
        base, scale = 0.55, 0.45
        protocol_ok = 1
    elif distinct:
        base, scale = 0.45, 0.25
        protocol_ok = 0
    else:
        base, scale = 0.35, 0.20
        protocol_ok = 0

    total = base + scale * (
        (w_format * f_format + w_reflect * f_reflect + w_tool * f_tool + w_result * f_result) / w_sum
    )

    return {
        "score": float(total),
        "reward_format": float(f_format),
        "reward_reflection": float(f_reflect),
        "reward_tool_usage": float(f_tool),
        "reward_result": float(f_result),
        "num_hermes_tool_calls": int(len(prompts)),
        "num_generate_image_prompts": int(len(prompts)),
        "protocol_ok": int(protocol_ok),
        "method": "agentic_hermes_tool_calls",
    }
