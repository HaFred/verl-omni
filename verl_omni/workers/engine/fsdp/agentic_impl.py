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
"""FSDP2 engine for agentic LLM token-level GRPO training (Mode 2a).

Inherits verl's ``FSDPEngineWithLMHead`` to get micro-batch splitting,
global DP loss normalization, FSDP grad scaler, autocast, and checkpoint
management for free.  Overrides ``_build_module`` (the hook ``initialize``
actually calls) for selective freezing of the generation path.
"""

from __future__ import annotations

import logging
import os

from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="agentic_llm", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class AgenticLLMFSDPEngine(FSDPEngineWithLMHead):
    """FSDP2 engine for agentic LLM token-level GRPO (Mode 2a).

    Inherits all token-level PPO infrastructure from verl's
    ``FSDPEngineWithLMHead`` and only overrides :meth:`_build_module` to
    (1) load weights via the language-model HF path and (2) freeze
    generation-path weights listed in ``model_config.freeze``.
    """

    def __init__(
        self,
        model_config,  # OmniModelConfig (lazy-imported to avoid diffusers)
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _build_module(self):
        """Load HF weights (language_model path) then freeze gen-path params.

        Parent ``_build_module`` only accepts ``language_model`` / ``value_model``.
        Temporarily treat ``agentic_llm`` as ``language_model`` for the load.
        """
        orig_type = self.model_config.model_type
        if orig_type == "agentic_llm":
            self.model_config.model_type = "language_model"
        try:
            module = super()._build_module()
        finally:
            self.model_config.model_type = orig_type

        self.module = module
        self._freeze_generation_path()
        return module

    def _freeze_generation_path(self) -> None:
        freeze_prefixes = getattr(self.model_config, "freeze", []) or []
        if freeze_prefixes:
            for name, param in self.module.named_parameters():
                for prefix in freeze_prefixes:
                    if prefix in name:
                        param.requires_grad = False
                        break

        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.module.parameters())
        # warning: default VERL_LOGGING_LEVEL is WARN, so info would be invisible in smoke logs.
        logger.warning(
            "AgenticLLMFSDPEngine: %d/%d params trainable (freeze prefixes: %s)",
            trainable,
            total,
            freeze_prefixes,
        )
