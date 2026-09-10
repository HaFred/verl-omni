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
env.

YAML ``trainer/config/agentic/image_gen_tools.yaml`` is the single source of
truth for defaults. Readers fail loud when nothing has bound this process
(stock ``AgentLoopManager`` / reward workers must not silently look healthy).

Reward-side scorers should not call ``agentic_get``: pass knobs through
``extra_info`` via ``agentic_scorer_knobs_from_config`` / ``merge_agentic_scorer_knobs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "SCORER_KNOB_KEYS",
    "agentic_get",
    "agentic_get_bool",
    "agentic_get_float",
    "agentic_get_int",
    "agentic_get_str",
    "agentic_scorer_knobs_from_config",
    "bind_agentic_image_gen",
    "clear_agentic_image_gen",
    "get_agentic_image_gen",
    "is_agentic_image_gen_bound",
    "merge_agentic_scorer_knobs",
    "yaml_agentic_image_gen_defaults",
]

_MISSING = object()

# Scorer knobs that reward workers read from extra_info (not process-local bind).
SCORER_KNOB_KEYS: tuple[str, ...] = (
    "vllm_url",
    "vllm_model",
    "reflect_max_new_tokens",
    "judge_parse_retries",
    "reflect_vlm_timeout",
    "judge_enable_thinking",
    "good_enough_threshold",
)

_YAML_PATH = Path(__file__).resolve().parents[2] / "trainer" / "config" / "agentic" / "image_gen_tools.yaml"

_yaml_defaults: dict[str, Any] | None = None
_cfg: dict[str, Any] = {}
_bound: bool = False


def yaml_agentic_image_gen_defaults() -> dict[str, Any]:
    """Load ``image_gen_tools.yaml`` once (shallow copy)."""
    global _yaml_defaults
    if _yaml_defaults is None:
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(OmegaConf.load(_YAML_PATH), resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"expected mapping in {_YAML_PATH}, got {type(raw).__name__}")
        _yaml_defaults = dict(raw)
    return dict(_yaml_defaults)


def _defaults() -> dict[str, Any]:
    return yaml_agentic_image_gen_defaults()


def _node_to_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    from omegaconf import OmegaConf

    if OmegaConf.is_config(node):
        raw = OmegaConf.to_container(node, resolve=True)
        return dict(raw) if isinstance(raw, dict) else {}
    if isinstance(node, Mapping):
        return dict(node)
    out: dict[str, Any] = {}
    for key in _defaults():
        if hasattr(node, key):
            out[key] = getattr(node, key)
    return out


def is_agentic_image_gen_bound() -> bool:
    """Return whether ``bind_agentic_image_gen`` has run in this process."""
    return _bound


def clear_agentic_image_gen() -> None:
    """Unbind knobs so subsequent ``agentic_get`` without an explicit default raises."""
    global _cfg, _bound
    _cfg = {}
    _bound = False


def get_agentic_image_gen() -> dict[str, Any]:
    """Return bound knobs merged over yaml defaults (shallow copy). Requires bind."""
    if not _bound:
        raise RuntimeError(
            "agentic_image_gen is not bound; OmniAgentLoopWorker/Manager must call "
            "bind_agentic_image_gen before get_agentic_image_gen"
        )
    merged = _defaults()
    merged.update(_cfg)
    return merged


def bind_agentic_image_gen(config: Any) -> None:
    """Store ``config.agentic_image_gen`` for process-local readers.

    Missing ``config`` or missing ``agentic_image_gen`` node clears overrides so
    worker/test reuse cannot keep stale URLs or curriculum knobs, but still
    marks the process as bound (readers then see yaml defaults).
    """
    global _cfg, _bound
    _bound = True
    if config is None:
        _cfg = {}
        return
    try:
        node = config.get("agentic_image_gen")
    except (AttributeError, TypeError, KeyError):
        node = getattr(config, "agentic_image_gen", None)
    if node is None:
        _cfg = {}
        return
    _cfg = _node_to_dict(node)


def agentic_get(key: str, default: Any = _MISSING) -> Any:
    """Read one ``agentic_image_gen`` field.

    Priority when bound: ``_cfg`` → explicit ``default`` → yaml defaults.
    When unbound: explicit ``default`` only; otherwise raise (stock AgentLoopManager
    must not silently look configured).
    """
    if key in _cfg:
        return _cfg[key]
    if default is not _MISSING:
        return default
    if not _bound:
        raise RuntimeError(
            f"agentic_image_gen.{key} read while unbound; "
            "OmniAgentLoopWorker/Manager must call bind_agentic_image_gen "
            "(reward scorers should use extra_info / merge_agentic_scorer_knobs instead)"
        )
    defaults = _defaults()
    if key not in defaults:
        raise KeyError(f"unknown agentic_image_gen key: {key!r}")
    return defaults[key]


def agentic_get_str(key: str, default: Any = _MISSING) -> str:
    fallback = "" if default is _MISSING else default
    value = agentic_get(key) if default is _MISSING else agentic_get(key, default)
    if value is None:
        return str(fallback) if default is not _MISSING else ""
    return str(value).strip()


def agentic_get_bool(key: str, default: Any = _MISSING) -> bool:
    fallback = False if default is _MISSING else default
    value = agentic_get(key) if default is _MISSING else agentic_get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(fallback)
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def agentic_get_int(key: str, default: Any = _MISSING) -> int:
    fallback = 0 if default is _MISSING else default
    value = agentic_get(key) if default is _MISSING else agentic_get(key, default)
    if value is None:
        return int(fallback)
    return int(value)


def agentic_get_float(key: str, default: Any = _MISSING) -> float:
    fallback = 0.0 if default is _MISSING else default
    value = agentic_get(key) if default is _MISSING else agentic_get(key, default)
    if value is None:
        return float(fallback)
    return float(value)


def agentic_scorer_knobs_from_config(config: Any) -> dict[str, Any]:
    """Extract reward-side judge knobs from Hydra without binding this process."""
    defaults = _defaults()
    if config is None:
        node: Any = None
    else:
        try:
            node = config.get("agentic_image_gen")
        except (AttributeError, TypeError, KeyError):
            node = getattr(config, "agentic_image_gen", None)
    overrides = _node_to_dict(node) if node is not None else {}
    out: dict[str, Any] = {}
    for key in SCORER_KNOB_KEYS:
        out[key] = overrides[key] if key in overrides else defaults[key]
    return out


def merge_agentic_scorer_knobs(extra_info: dict[str, Any] | None, config: Any = None) -> dict[str, Any]:
    """Fill missing scorer knobs on ``extra_info`` from ``config`` (or yaml defaults).

    Existing ``extra_info`` values win (same precedence pattern as ``w_*``).
    """
    merged = dict(extra_info or {})
    knobs = agentic_scorer_knobs_from_config(config)
    for key, value in knobs.items():
        merged.setdefault(key, value)
    return merged
