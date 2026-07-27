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
management for free.  The only override is :meth:`build_module` for
selective freezing of the generation path.
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
    ``FSDPEngineWithLMHead`` (micro-batching, global DP norm, FSDP scaler,
    autocast, checkpointing) and only overrides :meth:`build_module` to
    freeze the generation-path weights.

    The trainer provides the loss function (e.g. standard ``ppo_loss`` with
    ``loss_mode="gspo"``) and the data batch carries ``response_mask`` set
    to the per-token ``loss_mask`` so that observation placeholder tokens
    receive zero gradient.
    """

    def __init__(
        self,
        model_config,  # OmniModelConfig (lazy-imported to avoid diffusers)
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def build_module(self):
        """Build the model and freeze generation-path weights.

        Calls the parent ``build_module`` first, then walks ``named_parameters``
        to set ``requires_grad=False`` on any parameter whose name contains a
        prefix listed in ``model_config.freeze``.
        """
        super().build_module()

        freeze_prefixes = getattr(self.model_config, "freeze", []) or []
        if not freeze_prefixes:
            return

        for name, param in self.module.named_parameters():
            for prefix in freeze_prefixes:
                if prefix in name:
                    param.requires_grad = False
                    break

        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.module.parameters())
        logger.info(
            "AgenticLLMFSDPEngine: %d/%d params trainable (freeze prefixes: %s)",
            trainable,
            total,
            freeze_prefixes,
        )
