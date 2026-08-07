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
"""HTTP client for the frozen Qwen3-VL reflect sidecar (reward judge only).

Used by ``agentic_reward`` for ``reward_correctness`` / ``reward_aesthetics``.
The actor calls ``judge_image`` tool (in ``diffusion_tool.py``) during rollout;
this client is also used by ``agentic_reward`` for ``reward_correctness`` /
``reward_aesthetics`` at score time.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def call_reflect_vlm(
    *,
    user_request: str,
    image_prompt: str,
    notes: str = "",
    image_path: str | None = None,
) -> dict | None:
    """POST to ``AGENTIC_REFLECT_VLM_URL``; return scored dict or ``None`` on failure.

    On unset URL or any transport/parse error, returns ``None`` so the reward
    scorer can zero C/A (no heuristic fallback for reward).
    """
    endpoint = os.getenv("AGENTIC_REFLECT_VLM_URL", "").strip()
    if not endpoint:
        return None
    payload: dict = {
        "user_request": user_request,
        "image_prompt": image_prompt,
        "notes": notes or "",
    }
    if image_path and Path(image_path).is_file():
        payload["image_path"] = image_path
        # Also send base64 so the sidecar works across hosts / non-shared FS.
        try:
            payload["image_base64"] = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        except OSError:
            pass
    timeout = float(os.getenv("AGENTIC_REFLECT_VLM_TIMEOUT", "120"))
    try:
        req = Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        logger.warning("reflect VLM call failed (%s); C/A will be zeroed", exc)
        return None
    if not isinstance(data, dict):
        return None
    try:
        correctness = float(data.get("correctness", 0.0))
        aesthetics = float(data.get("aesthetics", 0.0))
    except (TypeError, ValueError):
        return None
    match = float(data.get("match", 0.55 * correctness + 0.45 * aesthetics))
    good_enough = bool(data.get("good_enough", False))
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
        "good_enough": good_enough,
        "findings": str(data.get("findings") or ""),
        "suggested_fixes": str(data.get("suggested_fixes") or "none"),
        "backend": str(data.get("backend") or "qwen3_vl"),
        "missing_attrs": [],
        "fixes": [],
    }
