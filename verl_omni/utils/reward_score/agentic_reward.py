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
"""Mode (2a) agentic reward: format + reflect + R_tool + R_result.

Aligned with RFC VisionCreator-R1 Stage-1 reflection focus (rule-based
``R_reflect`` proxy; full VLM judge is PR2). Credit favors **voluntary**
Hermes tool calls and **prompt rewriting** between turns so GRPO can train
reflection on Lance_3B_hf_und with a frozen diffusion tool.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_OK = re.compile(r"agentic_tool\s+ok=1", re.IGNORECASE)
_TOOL_FAIL = re.compile(r"agentic_tool\s+ok=0", re.IGNORECASE)
_PNG_PATH = re.compile(r"path=\S+\.png", re.IGNORECASE)
_STUB_PATH = re.compile(r"STUB_NO_IMAGE|tool_stubbed|stub diffusion", re.IGNORECASE)
_REFLECT_MARK = re.compile(r"\breflection\s*:", re.IGNORECASE)
_TOOL_ECHO = re.compile(
    r"lance frozen mot|agentic_tool\s+ok=|tool_response|review the returned image|path=/\S+\.png",
    re.IGNORECASE,
)
_BARE_JSON_TOOL = re.compile(
    r'^\s*\{\s*"name"\s*:\s*"generate_image"',
    re.IGNORECASE | re.MULTILINE,
)

_STOP = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "to",
    "and",
    "with",
    "for",
    "from",
    "generate",
    "image",
    "create",
    "draw",
    "please",
    "make",
    "reflection",
    "pass",
}


def _as_dict(ground_truth: Any) -> dict:
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
        return {"user_request": raw, "expected_num_images": 2}
    return {"expected_num_images": 2}


def _count_hermes_generate_image(text: str) -> int:
    n = 0
    for match in _TOOL_CALL_BLOCK.finditer(text or ""):
        if "generate_image" in match.group(1).lower():
            n += 1
    return n


def _count_successful_images(text: str) -> int:
    blob = text or ""
    ok_markers = len(_TOOL_OK.findall(blob))
    if ok_markers:
        return ok_markers
    return len(_PNG_PATH.findall(blob))


def _count_failed_tools(text: str) -> int:
    blob = text or ""
    fails = len(_TOOL_FAIL.findall(blob))
    if fails:
        return fails
    return len(_STUB_PATH.findall(blob))


def _keywords(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOP]


def _attr_coverage(user_request: str, trajectory: str) -> float:
    keys = _keywords(user_request)
    if not keys:
        return 1.0
    blob = (trajectory or "").lower()
    hit = sum(1 for k in keys if k in blob)
    return hit / len(keys)


def _looks_like_tool_echo_response(text: str) -> bool:
    """True when the policy mostly echoed tool obs / bare JSON instead of Hermes."""
    blob = text or ""
    if _count_hermes_generate_image(blob) >= 1:
        return False
    if _TOOL_ECHO.search(blob):
        return True
    if _BARE_JSON_TOOL.search(blob):
        return True
    return False


def compute_r_format(
    *,
    n_voluntary_hermes: int,
    has_reflect_mark: bool,
    n_hermes_text: int = 0,
) -> float:
    """Format reward: Hermes tool calls + Reflection: (teacher-forced text counts).

    Voluntary Hermes gets full credit. Teacher-forced ``Reflection:`` + ``<tool_call>``
    in ``solution_str`` still scores so early-curriculum PR1 curves are non-flat.
    """
    if n_voluntary_hermes >= 2:
        base = 1.0
    elif n_voluntary_hermes == 1:
        base = 0.85
    elif n_hermes_text >= 2:
        base = 0.7
    elif n_hermes_text == 1:
        base = 0.35
    else:
        base = 0.0
    if base > 0 and has_reflect_mark:
        base = min(1.0, base + 0.15)
    elif base == 0 and has_reflect_mark:
        base = 0.15
    return float(base)


def compute_r_reflect(
    *,
    diffusion_prompts: list[str],
    user_request: str,
    n_voluntary_hermes: int,
) -> float:
    """Rule-based R_reflect ∈ [0, 1] (RFC Stage-1 proxy without VLM judge).

    Components:
      - rewrite: second diffusion prompt differs from first (0.4)
      - enrichment: 2nd prompt has more task attrs or refine lexicon (0.3)
      - voluntary multi-call: model emitted ≥2 Hermes calls (0.3)
    """
    score = 0.0
    prompts = [p.strip() for p in (diffusion_prompts or []) if str(p).strip()]
    if len(prompts) >= 2 and prompts[0].lower() != prompts[1].lower():
        score += 0.4
        k0 = set(_keywords(prompts[0]))
        k1 = set(_keywords(prompts[1]))
        task_keys = set(_keywords(user_request))
        refine_lex = {"detailed", "lighting", "composition", "focus", "texture", "color", "sharp"}
        richer = len(k1) > len(k0) or len(k1 & task_keys) > len(k0 & task_keys) or bool(k1 & refine_lex)
        if richer:
            score += 0.3
    if n_voluntary_hermes >= 2:
        score += 0.3
    elif n_voluntary_hermes == 1 and len(prompts) >= 2:
        score += 0.1
    return float(min(1.0, score))


def compute_r_tool(*, n_success: int, n_fail: int, n_hermes: int) -> float:
    if n_success >= 2:
        return 1.0
    if n_success == 1:
        return 0.8
    if n_fail > 0 or n_hermes > 0:
        return 0.1
    return 0.0


def compute_r_result(*, n_success: int, expected_num_images: int, attr_coverage: float, attr_threshold: float) -> float:
    expected = max(1, int(expected_num_images))
    if n_success != expected:
        return 0.0
    if attr_coverage < float(attr_threshold):
        return 0.0
    return 1.0


def _tool_extra(kwargs: dict, extra_info: dict) -> dict | None:
    for key in ("tool_extra_fields", "extras"):
        cand = kwargs.get(key)
        if cand is None and extra_info:
            cand = extra_info.get(key)
        if isinstance(cand, dict):
            return cand
        if isinstance(cand, list | tuple) and cand and isinstance(cand[0], dict):
            return cand[0]
    return None


def compute_score_smoke(
    data_source: str,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """ST-1 merge-gate reward: response-length heuristic with within-group variance.

    Cold ``Lance_3B_hf_und`` rarely emits Hermes on a 1-step smoke, so the full
    Mode (2a) ``compute_score`` collapses to all-zero advantages and ``actor/loss=0``.
    This length proxy still exercises GRPO → actor update without claiming
    reflection / tool learning.
    """
    del data_source, ground_truth, extra_info, kwargs
    text = (solution_str or "").strip()
    score = 0.0 if not text else min(1.0, len(text) / 256.0)
    return {
        "score": float(score),
        "method": "response_length_heuristic",
    }


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """GRPO scalar: format + reflect + voluntary tool/result."""
    del data_source
    extra_info = dict(extra_info or {})
    gt = _as_dict(ground_truth)
    tool_extra = _tool_extra(kwargs, extra_info) or {}

    blob = solution_str or ""
    n_hermes_text = _count_hermes_generate_image(blob)
    n_success_text = _count_successful_images(blob)
    n_fail_text = _count_failed_tools(blob)
    has_reflect_mark = bool(_REFLECT_MARK.search(blob))

    n_vol_hermes = int(
        tool_extra.get("num_voluntary_hermes", extra_info.get("num_voluntary_hermes", n_hermes_text)) or 0
    )

    n_vol_success = tool_extra.get("num_voluntary_successful_images")
    if n_vol_success is None:
        n_vol_success = extra_info.get("num_voluntary_successful_images")
    if n_vol_success is None:
        n_forced = int(tool_extra.get("num_forced_tool_calls") or extra_info.get("num_forced_tool_calls") or 0)
        n_vol_success = 0 if n_forced > 0 else n_success_text
    n_vol_success = int(n_vol_success or 0)

    n_all_success = int(
        tool_extra.get("num_successful_images", extra_info.get("num_successful_images", n_success_text)) or 0
    )
    n_forced = int(tool_extra.get("num_forced_tool_calls") or extra_info.get("num_forced_tool_calls") or 0)

    n_fail = n_fail_text
    if tool_extra.get("num_tool_calls_executed") is not None:
        executed = int(tool_extra["num_tool_calls_executed"])
        n_fail = max(n_fail, max(0, executed - n_all_success))

    prompts = tool_extra.get("diffusion_prompts") or extra_info.get("diffusion_prompts") or []
    if not isinstance(prompts, list):
        prompts = []

    expected = int(gt.get("expected_num_images", extra_info.get("expected_num_images", 2)))
    user_request = str(gt.get("user_request") or extra_info.get("raw_prompt") or "")
    attr_threshold = float(gt.get("attr_threshold", extra_info.get("attr_threshold", 0.5)))
    # Prefer attribute coverage on diffusion prompts (what Lance actually saw).
    prompt_blob = " ".join(str(p) for p in prompts) if prompts else blob
    attr_cov = _attr_coverage(user_request, prompt_blob)

    r_format = compute_r_format(
        n_voluntary_hermes=n_vol_hermes,
        has_reflect_mark=has_reflect_mark,
        n_hermes_text=n_hermes_text,
    )
    echo_penalty = 0.0
    if _looks_like_tool_echo_response(blob) and n_hermes_text == 0 and n_vol_hermes == 0:
        # Strong negative so GRPO prefers teacher Hermes / voluntary format over echo prose.
        echo_penalty = 0.35
        r_format = 0.0

    r_reflect = compute_r_reflect(
        diffusion_prompts=[str(p) for p in prompts],
        user_request=user_request,
        n_voluntary_hermes=n_vol_hermes,
    )
    # Teacher-forced rewrite still earns reflect credit (prompt0 != prompt1).
    if r_reflect < 0.4 and len(prompts) >= 2 and str(prompts[0]).lower() != str(prompts[1]).lower():
        r_reflect = max(r_reflect, 0.55 if has_reflect_mark else 0.4)
    if has_reflect_mark:
        r_reflect = max(r_reflect, 0.25)

    r_tool = compute_r_tool(n_success=n_vol_success, n_fail=n_fail, n_hermes=n_vol_hermes)
    forced_consolation = float(gt.get("forced_consolation", extra_info.get("forced_consolation", 0.05)))
    if n_vol_success == 0 and n_all_success > 0 and n_forced > 0:
        # Successful forced tools: stronger consolation so score rises with 2 images.
        r_tool = max(r_tool, 0.35 if n_all_success >= 2 else forced_consolation)

    r_result = compute_r_result(
        n_success=n_vol_success,
        expected_num_images=expected,
        attr_coverage=attr_cov,
        attr_threshold=attr_threshold,
    )
    # When only forced tools ran, still allow R_result on all-success + attrs so
    # within-group prompt quality can create advantage during early curriculum.
    if n_vol_success == 0 and n_all_success == expected and attr_cov >= attr_threshold:
        r_result = max(r_result, 0.5)
    elif n_vol_success == 0 and n_all_success >= 1 and attr_cov >= attr_threshold:
        r_result = max(r_result, 0.25)
    w_format = float(extra_info.get("w_format", gt.get("w_format", 0.25)))
    w_reflect = float(extra_info.get("w_reflect", gt.get("w_reflect", 0.35)))
    w_tool = float(extra_info.get("w_tool", gt.get("w_tool", 0.2)))
    w_result = float(extra_info.get("w_result", gt.get("w_result", 0.2)))
    w_sum = w_format + w_reflect + w_tool + w_result
    if w_sum <= 0:
        w_format, w_reflect, w_tool, w_result, w_sum = 0.25, 0.35, 0.2, 0.2, 1.0
    score = (w_format * r_format + w_reflect * r_reflect + w_tool * r_tool + w_result * r_result) / w_sum
    score = max(0.0, score - echo_penalty)

    return {
        "score": float(score),
        "r_format": float(r_format),
        "r_reflect": float(r_reflect),
        "r_tool": float(r_tool),
        "r_result": float(r_result),
        "echo_penalty": float(echo_penalty),
        "attr_coverage": float(attr_cov),
        "num_successful_images": int(n_all_success),
        "num_voluntary_successful_images": int(n_vol_success),
        "num_voluntary_hermes": int(n_vol_hermes),
        "num_forced_tool_calls": int(n_forced),
        "num_diffusion_prompts": int(len(prompts)),
        "num_failed_tools": int(n_fail),
        "num_hermes_tool_calls": int(n_hermes_text),
        "expected_num_images": int(expected),
        "num_turns": extra_info.get("num_turns"),
        "method": "agentic_reflect_format_tool_result",
    }


def compute_agentic_reward_metrics(batch) -> dict[str, float]:
    """Pull agentic reward extras from ``non_tensor_batch`` into step metrics."""
    ntb = getattr(batch, "non_tensor_batch", None) or {}
    keys = (
        "r_format",
        "r_reflect",
        "r_tool",
        "r_result",
        "attr_coverage",
        "num_successful_images",
        "num_voluntary_successful_images",
        "num_voluntary_hermes",
        "num_forced_tool_calls",
        "num_diffusion_prompts",
    )
    out: dict[str, float] = {}
    for key in keys:
        if key not in ntb:
            continue
        vals = []
        for v in list(ntb[key]):
            try:
                if v is None:
                    continue
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        out[f"agentic/{key}/mean"] = float(mean)
        out[f"agentic/{key}/max"] = float(max(vals))
        out[f"agentic/{key}/min"] = float(min(vals))
    return out
