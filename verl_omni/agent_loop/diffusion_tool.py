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

"""Frozen diffusion function tool for verl's stock ``ToolAgentLoop``."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from urllib.request import Request, urlopen

from PIL import Image
from verl.tools.function_tool import function_tool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__file__)

DIFFUSION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Generate an image with the frozen diffusion model. Review the returned "
            "image, then call this tool again with a refined prompt or finish."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The complete prompt to send to the diffusion model.",
                }
            },
            "required": ["prompt"],
        },
    },
}


def _decode_images(payload: dict) -> list[Image.Image]:
    """Decode the endpoint's optional base64 image fields."""
    encoded = payload.get("images_base64")
    if encoded is None and payload.get("image_base64") is not None:
        encoded = [payload["image_base64"]]
    if not encoded:
        return []
    if not isinstance(encoded, list):
        raise ValueError("diffusion tool response 'images_base64' must be a list")
    return [Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB") for item in encoded]


@function_tool("generate_image", schema=DIFFUSION_TOOL_SCHEMA)
def generate_image(prompt: str) -> tuple[ToolResponse, float, dict]:
    """Generate an image with a frozen external diffusion service.

    Args:
        prompt: Complete text prompt for the diffusion model.
    """
    endpoint = os.getenv("AGENTIC_DIFFUSION_TOOL_URL")
    if not endpoint:
        logger.warning(
            "AGENTIC_DIFFUSION_TOOL_URL is unset; using text-only stub diffusion tool (acceptance smoke only)"
        )
        response = ToolResponse(
            text=(f"[stub diffusion result] No image service is configured. The requested prompt was: {prompt}")
        )
        return response, 0.0, {"tool_stubbed": True}

    headers = {"Content-Type": "application/json"}
    token = os.getenv("AGENTIC_DIFFUSION_TOOL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        endpoint,
        data=json.dumps({"prompt": prompt}).encode(),
        headers=headers,
        method="POST",
    )
    timeout = float(os.getenv("AGENTIC_DIFFUSION_TOOL_TIMEOUT", "120"))
    with urlopen(request, timeout=timeout) as result:  # noqa: S310 - endpoint is operator-configured
        payload = json.loads(result.read())

    images = _decode_images(payload)
    text = payload.get("text") or "The frozen diffusion tool generated the requested image."
    reward = float(payload.get("reward", 0.0))
    return ToolResponse(text=text, image=images or None), reward, {"tool_stubbed": False}
