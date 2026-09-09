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
"""HTTP client for the frozen agentic image-judge sidecar (reward C/A fallback).

Primary reward C/A comes from ``agentic_judge ok=1`` observations already in the
trajectory. This client is the fallback when those markers are missing.

Uses Hydra ``agentic_image_gen.vllm_url`` (OpenAI ``/v1/chat/completions``,
same as ``judge_image``). E2E runs require the vLLM judge sidecar; there is no
legacy ``/reflect`` path.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

from verl_omni.tools.trajectory.hydra_env import agentic_get_bool, agentic_get_float, agentic_get_int, agentic_get_str
from verl_omni.utils.agentic_image_judge_parse import build_judge_prompt, parse_judge_json

logger = logging.getLogger(__name__)


def judge_enable_thinking() -> bool:
    """Qwen3.5 defaults to long CoT; that burns ``max_tokens`` before JSON lands."""
    return agentic_get_bool("judge_enable_thinking", False)


def post_vllm_chat(
    *,
    vllm_url: str,
    image_b64: str,
    prompt_text: str,
    max_tokens: int,
    enable_thinking: bool | None = None,
) -> tuple[str | None, str | None]:
    """POST OpenAI ``/v1/chat/completions``. Returns ``(raw_text, error)``."""
    thinking = judge_enable_thinking() if enable_thinking is None else bool(enable_thinking)
    model = agentic_get_str("vllm_model")
    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if not payload["model"]:
        del payload["model"]
    timeout = agentic_get_float("reflect_vlm_timeout", 120.0)
    try:
        req = Request(
            f"{vllm_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    choices = data.get("choices") or []
    raw_text = ""
    if choices:
        raw_text = str(choices[0].get("message", {}).get("content", "") or "")
    if not raw_text:
        return None, "empty_response"
    return raw_text, None


def _normalize_scored(data: dict, *, backend: str) -> dict | None:
    try:
        correctness = float(data.get("correctness", 0.0))
        aesthetics = float(data.get("aesthetics", 0.0))
    except (TypeError, ValueError):
        return None
    match = float(data.get("match", 0.55 * correctness + 0.45 * aesthetics))
    correctness_scores = data.get("correctness_scores") or {}
    aesthetics_scores = data.get("aesthetics_scores") or {}
    if not isinstance(correctness_scores, dict):
        correctness_scores = {}
    if not isinstance(aesthetics_scores, dict):
        aesthetics_scores = {}
    return {
        "ok": True,
        "correctness": max(0.0, min(1.0, correctness)),
        "aesthetics": max(0.0, min(1.0, aesthetics)),
        "correctness_scores": {
            str(key): max(0.0, min(1.0, float(value)))
            for key, value in correctness_scores.items()
            if isinstance(value, int | float)
        },
        "aesthetics_scores": {
            str(key): max(0.0, min(1.0, float(value)))
            for key, value in aesthetics_scores.items()
            if isinstance(value, int | float)
        },
        "match": max(0.0, min(1.0, match)),
        "good_enough": bool(data.get("good_enough", False)),
        "findings": str(data.get("findings") or ""),
        "suggested_fixes": str(data.get("suggested_fixes") or "none"),
        "backend": str(data.get("backend") or backend),
        "missing_attrs": [],
        "fixes": [],
        "parse_ok": 1,
    }


def _call_vllm_openai(
    *,
    user_request: str,
    image_prompt: str,
    notes: str,
    image_path: str,
    vllm_url: str,
) -> dict | None:
    try:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("reflect VLM cannot read image %s: %s", image_path, exc)
        return None

    base_tokens = agentic_get_int("reflect_max_new_tokens", 1024)
    max_retries = max(0, agentic_get_int("judge_parse_retries", 1))

    for attempt in range(max_retries + 1):
        strict = attempt > 0
        tokens = base_tokens if attempt == 0 else max(base_tokens, 1536)
        raw_text, err = post_vllm_chat(
            vllm_url=vllm_url,
            image_b64=image_b64,
            prompt_text=build_judge_prompt(user_request, image_prompt, notes, strict_json=strict),
            max_tokens=tokens,
        )
        if err is not None:
            logger.warning("reflect VLM OpenAI call failed (%s); C/A will be zeroed", err)
            return None
        assert raw_text is not None
        parsed = parse_judge_json(raw_text)
        if parsed is not None:
            return _normalize_scored(parsed, backend="vllm")
        logger.warning("reflect VLM OpenAI unparseable (attempt=%d)", attempt)
    return None


def call_reflect_vlm(
    *,
    user_request: str,
    image_prompt: str,
    notes: str = "",
    image_path: str | None = None,
) -> dict | None:
    """Score an image via frozen VL on ``agentic_image_gen.vllm_url``; ``None`` on failure.

    Requires a running vLLM OpenAI chat sidecar. On unset URL, missing image, or
    any transport/parse error, returns ``None`` so the reward scorer can zero C/A
    (no heuristic / legacy ``/reflect`` fallback).
    """
    vllm_url = agentic_get_str("vllm_url")
    if not vllm_url:
        return None
    if not image_path or not Path(image_path).is_file():
        return None
    return _call_vllm_openai(
        user_request=user_request,
        image_prompt=image_prompt,
        notes=notes,
        image_path=image_path,
        vllm_url=vllm_url,
    )
