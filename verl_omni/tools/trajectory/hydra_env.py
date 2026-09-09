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

"""Process-local store for Hydra ``agentic_image_gen`` knobs.

``OmniAgentLoopWorker`` / ``OmniAgentLoopManager`` call
``bind_agentic_image_gen`` so FunctionTool bodies (``asyncio.to_thread``) and
agent-loop helpers can read the same knobs without ``os.getenv`` / AGENTIC_*
env. Defaults mirror ``trainer/config/agentic/image_gen_tools.yaml``.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "agentic_get",
    "agentic_get_bool",
    "agentic_get_float",
    "agentic_get_int",
    "agentic_get_str",
    "bind_agentic_image_gen",
    "bind_agentic_image_gen_env",
    "clear_agentic_image_gen",
    "get_agentic_image_gen",
]

# Defaults mirror verl_omni/trainer/config/agentic/image_gen_tools.yaml.
_DEFAULTS: dict[str, Any] = {
    "vllm_omni_url": "",
    "qwen_image_url": "",
    "diffusion_tool_url": "",
    "diffusion_tool_token": None,
    "diffusion_tool_timeout": 900,
    "vllm_url": "",
    "vllm_model": "",
    "reflect_max_new_tokens": 1024,
    "judge_parse_retries": 1,
    "reflect_vlm_timeout": 120,
    "judge_enable_thinking": False,
    "block_generate_after_yes": True,
    "block_generate_after_max_passes": True,
    "max_generate_image_passes": 3,
    "force_first_generate": False,
    "force_first_warmup_steps": 10,
    "force_first_end_step": 20,
    "force_reflection_after_judge": True,
    "rewrite_judge_before_generate": True,
    "e2e_root": None,
}

_MISSING = object()

_cfg: dict[str, Any] = {}


def _node_to_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            raw = OmegaConf.to_container(node, resolve=True)
            return dict(raw) if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    if isinstance(node, Mapping):
        return dict(node)
    out: dict[str, Any] = {}
    for key in _DEFAULTS:
        if hasattr(node, key):
            out[key] = getattr(node, key)
    return out


def clear_agentic_image_gen() -> None:
    """Drop bound knobs so readers fall back to yaml defaults (tests)."""
    global _cfg
    _cfg = {}


def get_agentic_image_gen() -> dict[str, Any]:
    """Return bound knobs merged over defaults (shallow copy)."""
    merged = dict(_DEFAULTS)
    merged.update(_cfg)
    return merged


def bind_agentic_image_gen(config: Any) -> None:
    """Store ``config.agentic_image_gen`` for process-local readers."""
    global _cfg
    if config is None:
        return
    try:
        node = config.get("agentic_image_gen")
    except Exception:  # noqa: BLE001 — DictConfig / plain dict / missing
        node = getattr(config, "agentic_image_gen", None)
    if node is None:
        return
    _cfg = _node_to_dict(node)


def bind_agentic_image_gen_env(config: Any) -> None:
    """Alias for ``bind_agentic_image_gen`` (historical name; no longer sets env)."""
    bind_agentic_image_gen(config)


def agentic_get(key: str, default: Any = _MISSING) -> Any:
    """Read one ``agentic_image_gen`` field (bound value, else default, else yaml default)."""
    if key in _cfg:
        return _cfg[key]
    if default is not _MISSING:
        return default
    return _DEFAULTS.get(key)


def agentic_get_str(key: str, default: str = "") -> str:
    value = agentic_get(key, default)
    if value is None:
        return default
    return str(value).strip()


def agentic_get_bool(key: str, default: bool = False) -> bool:
    value = agentic_get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def agentic_get_int(key: str, default: int = 0) -> int:
    value = agentic_get(key, default)
    if value is None:
        return default
    return int(value)


def agentic_get_float(key: str, default: float = 0.0) -> float:
    value = agentic_get(key, default)
    if value is None:
        return default
    return float(value)
