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
"""CPU tests for bagel_corl_sync registration, N/S validation, and GEN skip loss."""

from __future__ import annotations

import pytest
import torch
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
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "vllm_omni",
                    "response_length": 512,
                    "do_sample": True,
                    "free_cache_engine": None,
                    "trace": {
                        "_target_": "verl.workers.config.TraceConfig",
                        "backend": None,
                        "token2text": False,
                    },
                    "agent": {
                        "_target_": "verl.workers.config.AgentLoopConfig",
                        "default_agent_loop": "bagel_multiturn_agent",
                    },
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
    rollout = trainer.config.actor_rollout_ref.rollout
    assert rollout._target_.endswith("DiffusionRolloutConfig")
    assert rollout.rollout_adapter == "default"
    assert rollout.free_cache_engine is True
    assert rollout.response_length == 512
    assert rollout.trace.token2text is False
    assert "do_sample" not in rollout
    agent = rollout.agent
    assert agent._target_.endswith("BagelCorlAgentLoopConfig")
    assert agent.gen_samples_per_call == 4
    assert agent.max_generate_passes == 1


def test_seeds_s_must_be_at_least_two():
    with pytest.raises(ValueError, match="gen_samples_per_call >= 2"):
        validate_bagel_corl_config(
            _corl_cfg(actor_rollout_ref={"rollout": {"n": 8, "agent": {"gen_samples_per_call": 1}}})
        )


def test_sibling_n_need_not_equal_two_s():
    """N=rollout.n and S=gen_samples_per_call are independent; old J=2K gate is retired."""
    validate_bagel_corl_config(
        _corl_cfg(actor_rollout_ref={"rollout": {"n": 8, "agent": {"gen_samples_per_call": 4}}})
    )
    validate_bagel_corl_config(
        _corl_cfg(actor_rollout_ref={"rollout": {"n": 3, "agent": {"gen_samples_per_call": 2}}})
    )


def test_default_corl_config_validates():
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


def test_composite_loss_never_uses_und_token_adv_for_gen():
    """MoT GEN must not train on TQ token-GRPO advantages from the UND batch."""
    data = TensorDict({"advantages": torch.ones(2, 4), "old_log_probs": torch.zeros(2, 4)}, batch_size=[2])
    assign_non_tensor_data(data, "has_complete_gen_groups", True)
    assign_non_tensor_data(data, "skip_gen", False)
    assign_non_tensor_data(data, "num_gen_rows", 2)
    # Completeness flags say GEN should run, but bagel_corl_gen (FlowGRPO view) is absent.
    loss, metrics = bagel_composite_loss(
        config=None,
        model_output={"log_probs": torch.zeros(2)},
        data=data,
    )
    assert "gen/skipped_no_groups" in metrics
    assert "gen/missing_flowgrpo_view" in metrics
    assert float(loss.detach()) == 0.0


def test_gen_flowgrpo_advantage_groups_by_gen_group_uid():
    from types import SimpleNamespace

    from verl_omni.trainer.omni.bagel_corl_gen_adv import apply_gen_flowgrpo_advantage
    from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync

    rows = []
    for group, scores in (("callA", (1.0, 0.0)), ("callB", (0.5, 0.5))):
        for seed, score in enumerate(scores):
            rows.append(
                {
                    "gen_group_uid": group,
                    "gen_sample_uid": f"{group}:{seed}",
                    "rollout_log_probs": [0.1, 0.2, 0.3],
                    "rm_score": score,
                }
            )
    proto, metrics = apply_gen_flowgrpo_advantage(rows, adv_estimator="flow_grpo")
    assert proto is not None
    assert metrics["has_complete_gen_groups"] == 1.0
    assert metrics["gen/skipped_no_groups"] == 0.0
    uids = list(proto.non_tensor_batch["uid"])
    assert uids == ["callA", "callA", "callB", "callB"]
    adv = proto.batch["advantages"]
    assert adv[0, 0].item() > 0
    assert adv[1, 0].item() < 0
    assert abs(adv[2, 0].item()) < 1e-5
    assert abs(adv[3, 0].item()) < 1e-5

    skipped, skip_metrics = apply_gen_flowgrpo_advantage(
        [{"gen_group_uid": "g", "rm_score": 1.0}],
        adv_estimator="flow_grpo",
    )
    assert skipped is None
    assert skip_metrics["has_complete_gen_groups"] == 0.0

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    # UND keeps token grpo; GEN must take model.algorithm=flow_grpo (not algorithm.adv_estimator).
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": True,
                "global_std": True,
            },
            "actor_rollout_ref": {
                "model": {"algorithm": "flow_grpo"},
                "rollout": {"agent": {"gen_samples_per_call": 2}},
            },
        }
    )
    assert trainer._gen_adv_estimator() == "flow_grpo"
    order: list[str] = []

    def _und(self, batch, metrics):
        order.append("und")
        assert batch.extra_info["skip_gen"] is False
        assert batch.extra_info["bagel_corl_gen"] is not None
        assert "advantages" in batch.extra_info["bagel_corl_gen"].keys()
        return batch

    original = OmniPPOTrainerSync._compute_advantage
    OmniPPOTrainerSync._compute_advantage = _und
    try:
        batch = SimpleNamespace(extra_info={"gen_batch": rows})
        trainer._compute_advantage(batch, {})
    finally:
        OmniPPOTrainerSync._compute_advantage = original
    assert order == ["und"]


def test_gen_adv_estimator_rejects_token_grpo_collision():
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo"},
            "actor_rollout_ref": {"model": {"algorithm": "grpo"}, "rollout": {"agent": {}}},
        }
    )
    with pytest.raises(ValueError, match="collides"):
        trainer._compute_advantage(type("B", (), {"extra_info": {}})(), {})
