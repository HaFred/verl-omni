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
"""CPU tests for bagel_corl_sync registration, J=2K validation, and GEN skip loss."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.trainer.ppo.v1.trainer_base import get_trainer_cls
from verl.utils.tensordict_utils import assign_non_tensor_data

import verl_omni.trainer.omni  # noqa: F401  registers bagel_corl_sync
from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync
from verl_omni.utils.config import validate_bagel_corl_config, validate_config
from verl_omni.workers.utils.losses import bagel_composite_loss


def _corl_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "disable",
                "v1": {"trainer_mode": "bagel_corl_sync"},
            },
            "actor_rollout_ref": {
                "model": {"path": "/models/ByteDance-Seed/BAGEL-7B-MoT", "lora_rank": 64},
                "rollout": {
                    "n": 8,
                    "agent": {"gen_samples_per_call": 4, "max_generate_passes": 1},
                },
            },
        }
    )
    OmegaConf.set_struct(cfg, False)
    return OmegaConf.merge(cfg, overrides)


def test_register_bagel_corl_sync():
    assert get_trainer_cls("bagel_corl_sync") is OmniBagelCoRLTrainerSync


def test_rewrite_bagel_corl_configs_strips_omni_model_keys():
    from omegaconf import OmegaConf

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "_target_": "verl_omni.workers.config.omni.OmniModelConfig",
                    "path": "/tmp/bagel",
                    "model_type": "omni_model",
                    "algorithm": "flow_grpo",
                    "composite_mode": "bagel_corl",
                    "architecture": "OmniBagelForConditionalGeneration",
                    "override_config": {},
                    "model_stage": "thinker",
                    "use_remove_padding": True,
                },
                "actor": {
                    "_target_": "verl_omni.workers.config.omni.OmniActorConfig",
                    "trainer_type": "policy_gradient",
                },
                "rollout": {
                    "agent": {
                        "_target_": "verl.workers.config.AgentLoopConfig",
                        "default_agent_loop": "bagel_multiturn_agent",
                    }
                },
            }
        }
    )
    trainer._rewrite_bagel_corl_configs()
    model = trainer.config.actor_rollout_ref.model
    assert model._target_.endswith("DiffusionModelConfig")
    assert model.model_type == "diffusion_model"
    assert model.algorithm == "flow_grpo"
    assert "override_config" not in model
    assert "model_stage" not in model
    assert trainer.config.actor_rollout_ref.actor.diffusion_loss.loss_mode == "flow_grpo"
    agent = trainer.config.actor_rollout_ref.rollout.agent
    assert agent._target_.endswith("BagelCorlAgentLoopConfig")
    assert agent.gen_samples_per_call == 4
    assert agent.max_generate_passes == 1


def test_j_equals_two_k_fail_closed():
    with pytest.raises(ValueError, match="J=2K"):
        validate_bagel_corl_config(
            _corl_cfg(actor_rollout_ref={"rollout": {"n": 8, "agent": {"gen_samples_per_call": 3}}})
        )


def test_j_equals_two_k_ok():
    validate_config(_corl_cfg())


def test_qwen_und_forbidden():
    with pytest.raises(ValueError, match="Qwen"):
        validate_bagel_corl_config(
            _corl_cfg(actor_rollout_ref={"model": {"path": "Qwen/Qwen3-VL-8B-Instruct", "lora_rank": 8}})
        )


def test_composite_loss_skips_gen_without_complete_groups():
    data = TensorDict({}, batch_size=[])
    assign_non_tensor_data(data, "has_complete_gen_groups", False)
    assign_non_tensor_data(data, "skip_gen", True)
    assign_non_tensor_data(data, "num_gen_rows", 0)
    loss, metrics = bagel_composite_loss(config=None, model_output={}, data=data)
    assert float(loss.detach()) == 0.0
    assert "gen/skipped_no_groups" in metrics
