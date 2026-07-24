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
"""FSDP2 engine for agentic LLM token-level GRPO training (Mode 2a)."""

from __future__ import annotations

import logging
import os
from typing import Callable

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.base import BaseEngine, EngineRegistry

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


@EngineRegistry.register(model_type="agentic_llm", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class AgenticLLMFSDPEngine(BaseEngine):
    """FSDP2 engine for agentic LLM token-level GRPO.

    Trains understanding-path weights with per-token loss masking.
    Observation placeholder positions (loss_mask=0) receive zero gradient.
    Diffusion tool path is frozen (requires_grad=False, set during model init).
    """

    def __init__(
        self,
        model_config: "DiffusionModelConfig",  # noqa: F821
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        # Lazy import to avoid triggering verl_omni.__init__ (which requires diffusers)
        # at module level.  The ``from __future__ import annotations`` at the top of
        # this file keeps the type-hint string annotations from being evaluated here.
        from verl_omni.workers.config import DiffusionModelConfig  # noqa: F811

        self.model_config: DiffusionModelConfig = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

    def build_module(self):
        """Load model, freeze GEN path params, keep UND path trainable."""
        from verl_omni.pipelines.model_base import DiffusionModelBase

        torch_dtype = self.engine_config.model_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        self.module = DiffusionModelBase.build_module(
            model_config=self.model_config,
            torch_dtype=torch_dtype,
        )

        freeze_prefixes = getattr(self.model_config, "freeze", [])
        for name, param in self.module.named_parameters():
            for prefix in freeze_prefixes:
                if prefix in name:
                    param.requires_grad = False
                    break

        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.module.parameters())
        logger.info("AgenticLLMFSDPEngine: %d/%d params trainable", trainable, total)

    def build_optimizer(self):
        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.optimizer_config.lr,
            betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
        )

    def forward_backward_batch(
        self, data: TensorDict, loss_function: Callable, forward_only: bool = False
    ) -> list[TensorDict]:
        """Token-level forward/backward with loss_mask.

        Expects in data:
            agent_tokens: [micro_bsz, max_total_tokens]
            agent_logprobs: [micro_bsz, max_total_tokens]  # old (rollout) logprobs
            loss_mask: [micro_bsz, max_total_tokens]
            advantages: [micro_bsz, max_total_tokens]
        """
        agent_tokens = data["agent_tokens"]
        old_logprobs = data["agent_logprobs"]
        loss_mask = data["loss_mask"]
        advantages = data["advantages"]

        # Forward
        outputs = self.module(input_ids=agent_tokens)
        logits = outputs.logits  # [bsz, seq, vocab]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = agent_tokens[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous()
        shift_old_logprobs = old_logprobs[:, 1:].contiguous()
        shift_advantages = advantages[:, 1:].contiguous()

        # Token logprobs under current policy
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # PPO clipped loss
        clip_ratio = 0.2  # TODO: make configurable from DiffusionLossConfig
        ratio = torch.exp(token_logprobs - shift_old_logprobs)
        pg_losses1 = -ratio * shift_advantages
        pg_losses2 = -torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * shift_advantages
        pg_loss = torch.max(pg_losses1, pg_losses2)

        # Apply loss mask
        active = shift_mask.sum().clamp(min=1)
        loss = (pg_loss * shift_mask).sum() / active

        if not forward_only:
            loss.backward()

        metrics = {
            "loss": loss.detach().item(),
            "approx_kl": ((ratio - 1) - (token_logprobs - shift_old_logprobs)).detach().mean().item(),
        }

        return [tu.get_tensordict({"loss": loss.detach(), "metrics": metrics})]

    def optimizer_step(self):
        assert self.optimizer_config.clip_grad is not None
        grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        if torch.isfinite(grad_norm):
            self.optimizer.step()
        else:
            logger.warning("grad_norm is not finite: %s", grad_norm)
            self.optimizer.zero_grad()
        return grad_norm.item()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def lr_scheduler_step(self):
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()[0]
