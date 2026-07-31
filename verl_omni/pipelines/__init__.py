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
"""Auto-register optional pipeline adapters.

Each recipe adapter is imported independently so a missing optional
dependency (e.g. a newer vLLM-Omni symbol) does not block unrelated recipes
such as Lance agentic GRPO.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

_OPTIONAL_SUBMODULES = (
    "bagel_flow_grpo",
    "qwen3_omni",
    "qwen_image_diffusion_nft",
    "qwen_image_dpo",
    "qwen_image_edit_flow_grpo",
    "qwen_image_flow_grpo",
    "qwen_image_mix_grpo",
    "sd3_dpo",
    "sd3_flow_grpo",
    "wan22_dance_grpo",
)

__all__: list[str] = []


def _load_optional(name: str) -> None:
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        # Missing optional deps / shared libs (e.g. libcudart) — keep package importable.
        # Unexpected bugs (TypeError, RuntimeError, ...) still propagate.
        logger.warning("Skipping optional pipeline adapter %s: %s", name, exc)
        return
    exported = getattr(module, "__all__", None)
    if exported:
        globals().update({symbol: getattr(module, symbol) for symbol in exported})
        __all__.extend(exported)


for _name in _OPTIONAL_SUBMODULES:
    _load_optional(_name)

# Namespace hygiene after optional adapters load (do not del importlib: ruff F821).
del logging, pkgutil, _name, _load_optional
