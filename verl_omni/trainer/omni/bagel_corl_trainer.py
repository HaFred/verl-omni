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
"""Bagel UND+GEN Co-RL trainer: one post-gather composite update + weight publish."""

from __future__ import annotations

import logging

from omegaconf import OmegaConf, open_dict
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)

# Fields accepted by DiffusionModelConfig (plus _target_). Omni YAML keys outside this
# set must be stripped before instantiate/omega_conf_to_dataclass.
_DIFFUSION_MODEL_KEYS = {
    "path",
    "architecture",
    "transformer_config",
    "algorithm",
    "local_path",
    "tokenizer_path",
    "local_tokenizer_path",
    "hf_config",
    "model_type",
    "composite_mode",
    "load_tokenizer",
    "tokenizer",
    "processor",
    "extra_tokenizers",
    "extra_tokenizer_map",
    "use_shm",
    "trust_remote_code",
    "custom_chat_template",
    "external_lib",
    "enable_gradient_checkpointing",
    "attn_backend",
    "lora_rank",
    "lora_alpha",
    "lora_init_weights",
    "target_modules",
    "target_parameters",
    "exclude_modules",
    "lora",
    "lora_adapter_path",
    "policy_state_adapters",
    "lora_dtype",
    "mtp",
    "pipeline",
    "algo",
    "fsdp_layer_prefixes",
    "config_path",
    "transformer_subfolder",
}


@register_trainer("bagel_corl_sync")
class OmniBagelCoRLTrainerSync(OmniPPOTrainerSync):
    """Synchronous Bagel Co-RL: serial J-episode gather, then one UND+GEN optimizer step.

    Weight sync runs once in ``on_step_end`` (inherited) to every replica. No mid-episode sync.
    Replay uses the sync ``ReplayBuffer`` even though ``trainer_mode`` is not the string ``sync``.

    ``main_omni`` defaults to ``OmniModelConfig`` / ``OmniActorConfig``. Bagel Co-RL needs the
    diffusion model surface (``algorithm``, ``pipeline``, ``algo``) for FlowGRPO GEN while
    keeping the omni actor for UND ``ppo_loss``. Rewrite model config and inject
    ``diffusion_loss`` before tokenizer / worker init.
    """

    def _rewrite_bagel_corl_configs(self) -> None:
        """Point model at ``DiffusionModelConfig`` and ensure actor has ``diffusion_loss``."""
        raw = OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True) or {}
        filtered = {k: v for k, v in raw.items() if k in _DIFFUSION_MODEL_KEYS}
        filtered["_target_"] = "verl_omni.workers.config.diffusion.DiffusionModelConfig"
        filtered.setdefault("algorithm", "flow_grpo")
        filtered["model_type"] = "diffusion_model"
        filtered.setdefault("architecture", "OmniBagelForConditionalGeneration")
        filtered.setdefault("composite_mode", "bagel_corl")
        filtered.setdefault("trust_remote_code", True)

        with open_dict(self.config):
            self.config.actor_rollout_ref.model = OmegaConf.create(filtered)
            actor = self.config.actor_rollout_ref.actor
            if actor.get("diffusion_loss") is None:
                actor.diffusion_loss = OmegaConf.create(
                    {
                        "_target_": "verl_omni.workers.config.diffusion.DiffusionLossConfig",
                        "loss_mode": filtered.get("algorithm", "flow_grpo"),
                    }
                )
            elif actor.diffusion_loss.get("loss_mode") is None:
                actor.diffusion_loss.loss_mode = filtered.get("algorithm", "flow_grpo")

            # Use verl-omni AgentLoopConfig subclass so Co-RL fields are legal without patching verl.
            agent = self.config.actor_rollout_ref.rollout.agent
            agent._target_ = "verl_omni.workers.config.omni.BagelCorlAgentLoopConfig"
            if agent.get("gen_samples_per_call") is None:
                agent.gen_samples_per_call = 4
            if agent.get("max_generate_passes") is None:
                agent.max_generate_passes = 1
            if agent.get("max_und_turns") is None:
                agent.max_und_turns = 8

    def _init_tokenizer(self):
        self._rewrite_bagel_corl_configs()
        model_config: DiffusionModelConfig = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.model, DiffusionModelConfig
        )
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _build_replay_buffer(self):
        original_mode = self.trainer_mode
        self.trainer_mode = "sync"
        try:
            return super()._build_replay_buffer()
        finally:
            self.trainer_mode = original_mode

    def _init_resource_pool_mgr(self):
        """Use VeRL-Omni ``ActorRolloutRefWorker`` so Bagel FSDP + composite loss are available."""
        import ray

        super()._init_resource_pool_mgr()
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker as OmniActorRolloutRefWorker

        remote_cls = ray.remote(OmniActorRolloutRefWorker)
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = remote_cls

    def _compute_advantage(self, batch, metrics: dict):
        """UND GRPO groups by ``und_group_uid``; GEN FlowGRPO grouping is recorded in extra_info."""
        extra = batch.extra_info if hasattr(batch, "extra_info") else {}
        if extra.get("und/no_image_credit") is not None:
            metrics["und/no_image_credit"] = extra["und/no_image_credit"]
        if extra.get("gen/dropped_incomplete_groups") is not None:
            metrics["gen/dropped_incomplete_groups"] = extra["gen/dropped_incomplete_groups"]
        if extra.get("has_complete_gen_groups") is False or extra.get("gen/skipped_no_groups"):
            metrics["gen/skipped_no_groups"] = extra.get("gen/skipped_no_groups", 1.0)
        return super()._compute_advantage(batch, metrics)

    def _update_actor(self, batch, metrics: dict):
        """Single composite ``update_actor`` after the J-gather. Skip GEN branch when no complete K-groups."""
        extra = batch.extra_info if hasattr(batch, "extra_info") else {}
        has_complete = bool(extra.get("has_complete_gen_groups", extra.get("gen/num_rows", 0)))
        if not has_complete:
            metrics["gen/skipped_no_groups"] = 1.0
            extra["skip_gen"] = True
            extra["has_complete_gen_groups"] = False
        else:
            extra["skip_gen"] = False
            extra["has_complete_gen_groups"] = True
        logger.info(
            "bagel_corl_sync update_actor skip_gen=%s policy_version=%s",
            extra.get("skip_gen"),
            getattr(self, "global_steps", None),
        )
        return super()._update_actor(batch, metrics)
