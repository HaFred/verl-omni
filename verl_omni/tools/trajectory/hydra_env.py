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

"""Bind Hydra ``agentic_image_gen`` knobs into ``AGENTIC_*`` process env.

FunctionTool sync bodies run under ``asyncio.to_thread`` and only see
``os.getenv``. OmniAgentLoopWorker/Manager call this so Hydra is the source of
truth while tools keep reading env (CPU tests may still monkeypatch env).
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["bind_agentic_image_gen_env"]

# Hydra field name → AGENTIC_* env var.
_AGENTIC_IMAGE_GEN_ENV: tuple[tuple[str, str], ...] = (
    ("vllm_omni_url", "AGENTIC_VLLM_OMNI_URL"),
    ("qwen_image_url", "AGENTIC_QWEN_IMAGE_URL"),
    ("diffusion_tool_url", "AGENTIC_DIFFUSION_TOOL_URL"),
    ("diffusion_tool_token", "AGENTIC_DIFFUSION_TOOL_TOKEN"),
    ("diffusion_tool_timeout", "AGENTIC_DIFFUSION_TOOL_TIMEOUT"),
    ("vllm_url", "AGENTIC_VLLM_URL"),
    ("vllm_model", "AGENTIC_VLLM_MODEL"),
    ("reflect_max_new_tokens", "AGENTIC_REFLECT_MAX_NEW_TOKENS"),
    ("judge_parse_retries", "AGENTIC_JUDGE_PARSE_RETRIES"),
    ("reflect_vlm_timeout", "AGENTIC_REFLECT_VLM_TIMEOUT"),
    ("judge_enable_thinking", "AGENTIC_JUDGE_ENABLE_THINKING"),
    ("block_generate_after_yes", "AGENTIC_BLOCK_GENERATE_AFTER_YES"),
    ("block_generate_after_max_passes", "AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES"),
    ("max_generate_image_passes", "AGENTIC_MAX_GENERATE_IMAGE_PASSES"),
)


def _as_env_str(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def bind_agentic_image_gen_env(config: Any) -> None:
    """Push non-null ``config.agentic_image_gen`` fields into ``AGENTIC_*`` env."""
    if config is None:
        return
    try:
        node = config.get("agentic_image_gen")
    except Exception:  # noqa: BLE001 — DictConfig / plain dict / missing
        node = getattr(config, "agentic_image_gen", None)
    if node is None:
        return

    for field, env_name in _AGENTIC_IMAGE_GEN_ENV:
        try:
            value = node.get(field)
        except Exception:  # noqa: BLE001
            value = getattr(node, field, None)
        if value is None:
            continue
        os.environ[env_name] = _as_env_str(value)
