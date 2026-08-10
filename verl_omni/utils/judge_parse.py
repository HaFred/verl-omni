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
"""Shared VL judge JSON parse + prompt helpers (tool path and reward fallback)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

_CORRECTNESS_KEYS = (
    "subject_entities",
    "attributes",
    "relations_layout",
    "scene_context",
    "completeness",
)
_AESTHETICS_KEYS = (
    "composition",
    "lighting",
    "color",
    "fidelity",
    "appeal",
)

_CORRECTNESS_QUESTIONS = {
    "subject_entities": "Are the requested primary subjects/entities visibly present and recognizable?",
    "attributes": "Are requested attributes such as color, count, material, text, and identity correct?",
    "relations_layout": "Are requested actions, spatial relations, and layout/composition constraints correct?",
    "scene_context": "Does the environment, setting, style, and overall scene match the request?",
    "completeness": "Is the request fully satisfied without missing requested details or contradictory extras?",
}
_AESTHETICS_QUESTIONS = {
    "composition": "Is the composition balanced with a clear focal hierarchy and intentional framing?",
    "lighting": "Are lighting, exposure, contrast, and depth visually effective?",
    "color": "Are color harmony, saturation, and tonal relationships pleasing and coherent?",
    "fidelity": "Is the image sharp and spatially coherent, without obvious generation artifacts or distortions?",
    "appeal": "Does the image have strong overall visual appeal and professional finish?",
}


def good_enough_threshold() -> float:
    """Min C and A for good_enough=YES (client-side vLLM judge path).

    Default 0.85: first-pass often fails (headroom) but reachable after a solid
    rewrite so GRPO groups get enough high-reward rollouts. Override with
    ``AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD`` or ``AGENTIC_REFLECT_GOOD_ENOUGH``.
    """
    raw = os.getenv("AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD") or os.getenv("AGENTIC_REFLECT_GOOD_ENOUGH") or "0.85"
    try:
        return float(raw)
    except ValueError:
        return 0.85


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _mean_scores(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return sum(scores.values()) / max(1, len(scores))


def normalize_judge_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a parsed judge dict into the canonical scored shape."""
    if not isinstance(data, dict):
        return None

    c_scores_raw = data.get("correctness_scores")
    a_scores_raw = data.get("aesthetics_scores")
    c_scores: dict[str, float] = {}
    a_scores: dict[str, float] = {}
    if isinstance(c_scores_raw, dict) and c_scores_raw:
        for key, value in c_scores_raw.items():
            if isinstance(value, int | float):
                c_scores[str(key)] = _safe_float(value)
    if isinstance(a_scores_raw, dict) and a_scores_raw:
        for key, value in a_scores_raw.items():
            if isinstance(value, int | float):
                a_scores[str(key)] = _safe_float(value)

    if c_scores and a_scores:
        correctness = _mean_scores(c_scores)
        aesthetics = _mean_scores(a_scores)
    elif "correctness" in data or "aesthetics" in data:
        correctness = _safe_float(data.get("correctness", 0.0))
        aesthetics = _safe_float(data.get("aesthetics", 0.0))
    else:
        return None

    thr = good_enough_threshold()
    # Always derive YES/NO from scores × env threshold. Ignore any model-emitted
    # ``good_enough`` flag so AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD actually controls
    # rewrite pressure (VLM often returns true with A≈0.85).
    good_enough = correctness >= thr and aesthetics >= thr

    return {
        "correctness": correctness,
        "aesthetics": aesthetics,
        "correctness_scores": c_scores,
        "aesthetics_scores": a_scores,
        "findings": str(data.get("findings") or ""),
        "suggested_fixes": str(data.get("suggested_fixes") or ""),
        "good_enough": good_enough,
    }


def parse_judge_json(text: str) -> dict[str, Any] | None:
    """Extract C/A judge scores from VLM text (think blocks / fences / truncation)."""
    blob = (text or "").strip()
    blob = re.sub(r"<think>[\s\S]*?</think>", " ", blob, flags=re.IGNORECASE)
    blob = re.sub(r"```(?:json)?\s*", "", blob, flags=re.IGNORECASE).replace("```", "")
    blob = blob.strip()

    decoder = json.JSONDecoder()
    best: tuple[int, int, dict[str, Any]] | None = None
    for index, char in enumerate(blob):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(blob[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        normalized = normalize_judge_payload(data)
        if normalized is None:
            continue
        # Prefer full facet dicts over scalar-only parses.
        score = 2 if normalized["correctness_scores"] and normalized["aesthetics_scores"] else 1
        cand = (score, -index, normalized)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is not None:
        return best[2]

    # Last resort: aggregate scalars if JSON was truncated mid-object.
    c_m = re.search(r'"?correctness"?\s*[:=]\s*([0-9]*\.?[0-9]+)', blob, re.IGNORECASE)
    a_m = re.search(r'"?aesthetics"?\s*[:=]\s*([0-9]*\.?[0-9]+)', blob, re.IGNORECASE)
    if c_m and a_m:
        return normalize_judge_payload(
            {
                "correctness": float(c_m.group(1)),
                "aesthetics": float(a_m.group(1)),
                "findings": "parsed from truncated VLM text",
                "suggested_fixes": "none",
            }
        )
    return None


def build_judge_prompt(user_request: str, image_prompt: str, notes: str = "", *, strict_json: bool = False) -> str:
    """Build the VL judge prompt. ``strict_json`` is used on parse-failure retry."""
    c_schema = ",\n".join(f'    "{key}": 0.0' for key in _CORRECTNESS_KEYS)
    a_schema = ",\n".join(f'    "{key}": 0.0' for key in _AESTHETICS_KEYS)
    rubric = "\n".join(
        [
            "CORRECTNESS QUESTIONS:",
            *[f"- {key}: {q}" for key, q in _CORRECTNESS_QUESTIONS.items()],
            "AESTHETICS QUESTIONS:",
            *[f"- {key}: {q}" for key, q in _AESTHETICS_QUESTIONS.items()],
        ]
    )
    header = (
        "You are a strict, calibrated visual reward judge. Inspect the pixels, not merely the "
        "diffusion prompt. Independently answer all ten rubric questions.\n"
        f"User request: {user_request}\n"
        f"Diffusion prompt used (context only; never treat it as visual evidence): {image_prompt or '(none)'}\n"
        f"Notes: {notes or '(none)'}\n\n"
        f"{rubric}\n\n"
    )
    if strict_json:
        return (
            header + "CRITICAL RETRY: Your previous reply was not valid JSON.\n"
            "Reply with ONE JSON object only. No markdown, no <think>, no prose before/after.\n"
            "Use this exact shape (numbers in [0,1]):\n"
            "{\n"
            '  "correctness_scores": {\n'
            f"{c_schema}\n"
            "  },\n"
            '  "aesthetics_scores": {\n'
            f"{a_schema}\n"
            "  },\n"
            '  "findings": "short pixel evidence",\n'
            '  "suggested_fixes": "short rewrite hints"\n'
            "}\n"
        )
    return (
        header + "CALIBRATION (HARSH — default LOW):\n"
        "  0.00 = absent, broken, or completely wrong\n"
        "  0.20 = severely deficient, majority of requested elements missing or wrong\n"
        "  0.40 = several elements present but at least half are wrong, blurry, or misplaced\n"
        "  0.55 = mostly correct BUT one or two notable flaws\n"
        "  0.70 = clearly good, all key elements present, minor aesthetic issues only\n"
        "  0.85+ = near-flawless (RARELY given)\n"
        "Return ONLY this JSON shape (replace every 0.0 with an independently judged score):\n"
        "{\n"
        '  "correctness_scores": {\n'
        f"{c_schema}\n"
        "  },\n"
        '  "aesthetics_scores": {\n'
        f"{a_schema}\n"
        "  },\n"
        '  "findings": "specific visual evidence for the lowest scores",\n'
        '  "suggested_fixes": "specific prompt rewrite hints"\n'
        "}\n"
    )


def format_judge_observation(
    *,
    image_path: str,
    parsed: dict[str, Any],
    backend: str,
    parse_retries: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Format a successful judge obs (``agentic_judge ok=1 parse_ok=1``)."""
    correctness = float(parsed["correctness"])
    aesthetics = float(parsed["aesthetics"])
    good = bool(parsed.get("good_enough", False))
    findings_short = re.sub(r"\s+", " ", str(parsed.get("findings") or "no specific findings")).strip()[:220]
    fixes_short = re.sub(r"\s+", " ", str(parsed.get("suggested_fixes") or "none")).strip()[:160]
    text = (
        f"VL judge on the last generated image:\n"
        f"  path={image_path}\n"
        f"  correctness={correctness:.2f}\n"
        f"  aesthetics ={aesthetics:.2f}\n"
        f"  good_enough ={'YES' if good else 'NO'}\n"
        f"  findings: {findings_short}\n"
        f"  suggested_fixes: {fixes_short}\n"
        f"  agentic_judge ok=1 parse_ok=1 stub=0 backend={backend} parse_retries={parse_retries}"
    )
    meta = {
        "correctness": correctness,
        "aesthetics": aesthetics,
        "good_enough": good,
        "findings": str(parsed.get("findings") or ""),
        "suggested_fixes": str(parsed.get("suggested_fixes") or "none"),
        "image_path": image_path,
        "backend": backend,
        "parse_ok": 1,
        "parse_retries": int(parse_retries),
    }
    for key, value in (parsed.get("correctness_scores") or {}).items():
        if isinstance(value, int | float):
            meta[f"correctness_{key}"] = float(value)
    for key, value in (parsed.get("aesthetics_scores") or {}).items():
        if isinstance(value, int | float):
            meta[f"aesthetics_{key}"] = float(value)
    return text, meta


def format_judge_parse_error(
    *,
    image_path: str,
    raw_text: str = "",
    backend: str = "vllm",
    parse_retries: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Format a failed judge obs (``agentic_judge ok=0 parse_ok=0``) — no fake C/A."""
    text = (
        "[judge error] VLM returned unparseable response — do not invent scores. "
        "Retry judge_image or rewrite the diffusion prompt and generate again.\n"
        f"  path={image_path}\n"
        f"  agentic_judge ok=0 parse_ok=0 stub=0 backend={backend} parse_retries={parse_retries}"
    )
    meta = {
        "error": "unparseable",
        "raw": (raw_text or "")[:300],
        "image_path": image_path,
        "backend": backend,
        "parse_ok": 0,
        "parse_retries": int(parse_retries),
    }
    return text, meta
