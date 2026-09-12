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
from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync, _normalize_tq_kv_get_result
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
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
                    "agent": {
                        "gen_samples_per_call": 4,
                        "max_generate_passes": 1,
                        # Unit tests exercise N/S/LoRA gates; dual-role serving is a separate spike.
                        "und_ar_serving_ready": True,
                        "und_deploy_config": "examples/agenticllmgrpo_trainer/bagel/bagel_corl_deploy_ar.yaml",
                    },
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
            "data": {"train_files": "/tmp/train.parquet"},
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
    assert rollout.calculate_log_probs is True
    assert float(rollout.algo.noise_level) > 0.0
    assert int(rollout.algo.sde_window_size) >= 1
    assert int(rollout.pipeline.num_inference_steps) >= 1
    assert trainer.config.data.continuous_token.enable is False


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


def test_und_ar_serving_ready_required():
    with pytest.raises(ValueError, match="dual-role UND AR serving is not ready"):
        validate_bagel_corl_config(
            _corl_cfg(actor_rollout_ref={"rollout": {"agent": {"und_ar_serving_ready": False}}})
        )


def test_und_deploy_config_required_when_ready():
    with pytest.raises(ValueError, match="und_deploy_config"):
        validate_bagel_corl_config(
            _corl_cfg(actor_rollout_ref={"rollout": {"agent": {"und_ar_serving_ready": True, "und_deploy_config": None}}})
        )


def test_output_mode_ar_alone_rejected():
    with pytest.raises(ValueError, match="output_mode=ar alone"):
        validate_bagel_corl_config(
            _corl_cfg(
                actor_rollout_ref={
                    "rollout": {
                        "engine_kwargs": {"vllm_omni": {"output_mode": "ar"}},
                        "agent": {
                            "und_ar_serving_ready": True,
                            "und_deploy_config": "examples/agenticllmgrpo_trainer/bagel/bagel_corl_deploy_ar.yaml",
                        },
                    }
                }
            )
        )


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


def test_build_gen_proto_packs_traj_for_diffusion_engine():
    """GEN actor view must carry latents/timesteps so diffusion V1 engine can train under AR outer loop."""
    from verl_omni.trainer.omni.bagel_corl_gen_adv import apply_gen_flowgrpo_advantage, build_gen_flowgrpo_proto
    from verl_omni.workers.engine.fsdp.bagel_corl_composite import gen_view_has_traj, materialize_gen_train_batch

    rows = []
    for seed in range(2):
        rows.append(
            {
                "gen_group_uid": "call0",
                "gen_sample_uid": f"call0:{seed}",
                "rollout_log_probs": [0.1, 0.2],
                "rm_score": float(seed),
                "all_latents": torch.randn(2, 4),
                "timesteps": torch.tensor([999.0, 500.0]),
                "prompt_token_ids": [1, 2, 3],
            }
        )
    proto = build_gen_flowgrpo_proto(rows)
    assert proto is not None
    assert "all_latents" in proto.batch.keys()
    assert "all_timesteps" in proto.batch.keys()
    assert proto.batch["all_latents"].shape[0] == 2
    assert proto.batch["all_timesteps"].shape == (2, 2)

    adv_proto, metrics = apply_gen_flowgrpo_advantage(rows, adv_estimator="flow_grpo")
    assert adv_proto is not None
    assert metrics["gen/has_traj"] == 1.0
    assert gen_view_has_traj(adv_proto.batch)
    gen_data = materialize_gen_train_batch(adv_proto.batch, {"num_gen_rows": 2})
    assert "all_latents" in gen_data.keys()
    assert "advantages" in gen_data.keys()


def test_response_aligned_und_log_probs():
    from verl_omni.workers.engine.fsdp.bagel_corl_composite import response_aligned_und_log_probs

    token_logp = torch.arange(12, dtype=torch.float32).view(2, 6)
    out = response_aligned_und_log_probs(token_logp, response_len=3)
    assert out.shape == (2, 3)
    assert torch.equal(out[0], token_logp[0, -3:])


def test_composite_loss_und_plus_gen_separate_views():
    """bagel_composite_loss sums UND ppo stub + GEN diffusion when both outputs present."""
    from types import SimpleNamespace
    from unittest.mock import patch

    und_data = TensorDict(
        {
            "response_mask": torch.ones(2, 3),
            "old_log_probs": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
        },
        batch_size=[2],
    )
    gen_data = TensorDict(
        {"advantages": torch.ones(2, 2), "old_log_probs": torch.zeros(2, 2)},
        batch_size=[2],
    )
    data = TensorDict({}, batch_size=[])
    assign_non_tensor_data(data, "bagel_corl_und", und_data)
    assign_non_tensor_data(data, "bagel_corl_gen", gen_data)
    assign_non_tensor_data(data, "has_complete_gen_groups", True)
    assign_non_tensor_data(data, "skip_gen", False)
    assign_non_tensor_data(data, "num_gen_rows", 2)

    model_output = {
        "und": {"log_probs": torch.zeros(2, 3, requires_grad=True)},
        "gen": {"log_probs": torch.zeros(2, requires_grad=True)},
    }

    def fake_ppo(config, model_output, data, dp_group=None):
        return torch.tensor(1.0, requires_grad=True), {}

    def fake_diff(config, model_output, data, dp_group=None):
        return torch.tensor(2.0, requires_grad=True), {}

    with (
        patch("verl.workers.utils.losses.ppo_loss", fake_ppo),
        patch("verl_omni.workers.utils.losses.diffusion_loss", fake_diff),
    ):
        loss, metrics = bagel_composite_loss(config=SimpleNamespace(), model_output=model_output, data=data)
    assert float(loss.detach()) == 3.0
    assert metrics.get("gen/skipped_no_groups") is not None
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
                    "all_latents": torch.randn(3, 4),
                    "timesteps": torch.tensor([900.0, 600.0, 300.0]),
                }
            )
    proto, metrics = apply_gen_flowgrpo_advantage(rows, adv_estimator="flow_grpo")
    assert proto is not None
    assert metrics["has_complete_gen_groups"] == 1.0
    assert metrics["gen/has_traj"] == 1.0
    assert metrics["gen/skipped_no_groups"] == 0.0
    uids = list(proto.non_tensor_batch["uid"])
    assert uids == ["callA", "callA", "callB", "callB"]
    adv = proto.batch["advantages"]
    assert adv[0, 0].item() > 0
    assert adv[1, 0].item() < 0
    assert abs(adv[2, 0].item()) < 1e-5
    assert abs(adv[3, 0].item()) < 1e-5

    with pytest.raises(RuntimeError, match="lacks all_latents/timesteps"):
        apply_gen_flowgrpo_advantage(
            [{"gen_group_uid": "g", "rm_score": 1.0, "rollout_log_probs": [0.1, 0.2]}],
            adv_estimator="flow_grpo",
        )

    skipped, skip_metrics = apply_gen_flowgrpo_advantage([], adv_estimator="flow_grpo")
    assert skipped is None
    assert skip_metrics["has_complete_gen_groups"] == 0.0
    assert skip_metrics["gen/has_traj"] == 0.0

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


def test_normalize_tq_kv_get_result_columnar_tensordict():
    """kv_batch_get returns a columnar TensorDict, not a keyed dict."""
    keys = ["k0::gen::c::0", "k0::gen::c::1"]
    td = TensorDict(
        {
            "rm_scores": torch.tensor([[0.9], [0.1]]),
            "all_latents": torch.zeros(2, 4),
        },
        batch_size=[2],
    )
    assign_non_tensor_data(td, "gen_group_uid", ["c", "c"])
    out = _normalize_tq_kv_get_result(td, keys)
    assert set(out.keys()) == set(keys)
    assert out[keys[0]]["fields"]["gen_group_uid"] == "c"
    assert out[keys[1]]["fields"]["rm_scores"] is not None


def test_normalize_tq_kv_get_result_alt_shapes():
    keys = ["k0::gen::c::0", "k0::gen::c::1"]
    # keyed dict
    out = _normalize_tq_kv_get_result({k: {"rm_score": 0.5} for k in keys}, keys)
    assert out[keys[0]]["fields"]["rm_score"] == 0.5
    # columnar dict with a keys column
    out = _normalize_tq_kv_get_result({"keys": keys, "rm_score": [0.5, 0.6]}, keys)
    assert out[keys[1]]["fields"]["rm_score"] == 0.6
    # positional list
    out = _normalize_tq_kv_get_result([{"rm_score": 0.1}, {"rm_score": 0.2}], keys)
    assert out[keys[1]]["fields"]["rm_score"] == 0.2


def test_normalize_tq_kv_get_result_fails_loud_on_unknown_shape():
    with pytest.raises(ValueError, match="unrecognized"):
        _normalize_tq_kv_get_result(object(), ["k0"])


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


def test_diffusion_v1_gen_lane_binds_v1_hooks():
    from verl_omni.trainer.omni.bagel_corl_diff_v1 import DiffusionV1GenLane, diffusion_v1_gen_hooks

    try:
        old, ref, adv = diffusion_v1_gen_hooks()
    except ImportError:
        pytest.skip("verl pin missing ReplayBufferAsync; GEN lane binds at runtime on a current pin")
    owner = type(
        "Owner",
        (),
        {"config": object(), "actor_rollout_wg": object(), "ref_in_actor": True, "ref_policy_wg": None},
    )()
    lane = DiffusionV1GenLane(owner)
    assert lane._compute_old_log_prob.__func__ is old
    assert lane._compute_advantage.__func__ is adv
    assert lane.actor_rollout_wg is owner.actor_rollout_wg


def test_composite_forward_mode_gen_only_vs_und_infer():
    from verl.utils.tensordict_utils import assign_non_tensor_data
    from verl_omni.workers.engine.fsdp.bagel_corl_composite import composite_forward_mode

    gen_only = TensorDict(
        {"all_latents": torch.zeros(2, 3, 4), "all_timesteps": torch.zeros(2, 3)},
        batch_size=[2],
    )
    assert composite_forward_mode(gen_only, forward_only=True) == "gen_only_diffusion"

    und_infer = TensorDict(
        {"input_ids": torch.ones(2, 5, dtype=torch.long), "response_mask": torch.ones(2, 2)},
        batch_size=[2],
    )
    assert composite_forward_mode(und_infer, forward_only=True) == "und_infer"

    empty = TensorDict({}, batch_size=[])
    assign_non_tensor_data(empty, "skip_gen", True)
    assign_non_tensor_data(empty, "has_complete_gen_groups", False)
    assert composite_forward_mode(empty, forward_only=False) == "empty"


def test_compute_advantage_without_worker_group_uses_helper():
    """No actor_rollout_wg → stash FlowGRPO helper (CPU), not a second trainer."""
    from unittest.mock import patch

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
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
    extra = {
        "gen_batch": [
            {
                "gen_group_uid": "callA",
                "rollout_log_probs": [0.1, 0.2],
                "rm_score": 1.0,
                "all_latents": torch.randn(2, 4),
                "timesteps": torch.tensor([900.0, 500.0]),
            },
            {
                "gen_group_uid": "callA",
                "rollout_log_probs": [0.1, 0.2],
                "rm_score": 0.0,
                "all_latents": torch.randn(2, 4),
                "timesteps": torch.tensor([900.0, 500.0]),
            },
        ]
    }
    batch = type("B", (), {"extra_info": extra})()

    def fake_super_adv(_batch, _metrics):
        return _batch

    with patch.object(OmniPPOTrainerSync, "_compute_advantage", lambda self, b, m: fake_super_adv(b, m)):
        out = OmniBagelCoRLTrainerSync._compute_advantage(trainer, batch, {})
    assert extra["has_complete_gen_groups"]
    assert extra["bagel_corl_gen"] is not None
    assert extra["bagel_corl_gen"]["advantages"][0, 0].item() > 0
    assert extra["bagel_corl_gen"]["advantages"][1, 0].item() < 0
    assert out is batch


def test_gen_batch_rejects_legacy_nested_meta():
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"gen_samples_per_call": 2}}}})
    extra = {"bagel_corl": {"gen_batch": [{"gen_group_uid": "x"}]}}
    batch = type("B", (), {"extra_info": extra, "meta_info": {}, "non_tensor_batch": {}})()
    with pytest.raises(RuntimeError, match="legacy nested path removed"):
        trainer._gen_batch_from_step(batch)


def test_fetch_gen_records_raises_without_partition_id():
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    extra = {}
    batch = type("B", (), {"extra_info": extra})()
    und_records = [{"fields": {"child_gen_keys": ["k0::gen::c::0"]}}]
    with pytest.raises(RuntimeError, match="no partition_id"):
        trainer._fetch_gen_records_by_keys(batch, und_records)


# ---------------------------------------------------------------------------
# PR2 (M2): lane weights, per-group LRs, real-shape TQ normalize, ragged edges.
# ---------------------------------------------------------------------------


def test_composite_loss_applies_lane_weights():
    """loss_weight_und / loss_weight_gen scale their branches (RFC §4.4)."""
    from types import SimpleNamespace
    from unittest.mock import patch

    und_data = TensorDict(
        {
            "response_mask": torch.ones(2, 3),
            "old_log_probs": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
        },
        batch_size=[2],
    )
    gen_data = TensorDict(
        {"advantages": torch.ones(2, 2), "old_log_probs": torch.zeros(2, 2)},
        batch_size=[2],
    )
    data = TensorDict({}, batch_size=[])
    assign_non_tensor_data(data, "bagel_corl_und", und_data)
    assign_non_tensor_data(data, "bagel_corl_gen", gen_data)
    assign_non_tensor_data(data, "has_complete_gen_groups", True)
    assign_non_tensor_data(data, "skip_gen", False)
    assign_non_tensor_data(data, "num_gen_rows", 2)

    model_output = {
        "und": {"log_probs": torch.zeros(2, 3, requires_grad=True)},
        "gen": {"log_probs": torch.zeros(2, requires_grad=True)},
    }
    config = SimpleNamespace(
        diffusion_loss=OmegaConf.create({"loss_weight_und": 2.0, "loss_weight_gen": 3.0})
    )

    def fake_ppo(config, model_output, data, dp_group=None):
        return torch.tensor(1.0, requires_grad=True), {}

    def fake_diff(config, model_output, data, dp_group=None):
        return torch.tensor(2.0, requires_grad=True), {}

    with (
        patch("verl.workers.utils.losses.ppo_loss", fake_ppo),
        patch("verl_omni.workers.utils.losses.diffusion_loss", fake_diff),
    ):
        loss, _metrics = bagel_composite_loss(config=config, model_output=model_output, data=data)
    # 1.0 (und stub) * 2.0 + 2.0 (gen stub) * 3.0
    assert float(loss.detach()) == pytest.approx(8.0)


def test_gen_regularizer_velocity_mse_fails_loud():
    """velocity_mse (UniGRPO Eq. 8) lands in Phase 2 — selecting it must fail loud."""
    from types import SimpleNamespace

    gen_data = TensorDict(
        {"advantages": torch.ones(2, 2), "old_log_probs": torch.zeros(2, 2)},
        batch_size=[2],
    )
    data = TensorDict({}, batch_size=[])
    assign_non_tensor_data(data, "bagel_corl_gen", gen_data)
    assign_non_tensor_data(data, "has_complete_gen_groups", True)
    assign_non_tensor_data(data, "skip_gen", False)
    assign_non_tensor_data(data, "num_gen_rows", 2)
    config = SimpleNamespace(diffusion_loss=OmegaConf.create({"gen_regularizer": "velocity_mse"}))
    model_output = {"gen": {"log_probs": torch.zeros(2, requires_grad=True)}}
    with pytest.raises(NotImplementedError, match="velocity_mse"):
        bagel_composite_loss(config=config, model_output=model_output, data=data)


def test_dual_lora_param_groups_lr_override():
    import torch.nn as nn

    from verl_omni.pipelines.bagel_flow_grpo.bagel_corl import dual_lora_param_groups

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Parameter(torch.zeros(2))
            self.mlp_moe_gen = nn.Parameter(torch.zeros(2))
            self.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)

    module = _Tiny()
    groups = dual_lora_param_groups(module)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"und_lora", "gen_lora"}
    assert "lr" not in by_name["und_lora"]
    assert "lr" not in by_name["gen_lora"]

    groups = dual_lora_param_groups(module, lr_gen=3e-5)
    by_name = {g["name"]: g for g in groups}
    assert by_name["gen_lora"]["lr"] == pytest.approx(3e-5)
    assert "lr" not in by_name["und_lora"]  # UND keeps the base actor.optim.lr


def test_rewrite_sets_lr_gen_and_cfg_free_pipeline():
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_files": "/tmp/train.parquet"},
            "actor_rollout_ref": {
                "model": {"path": "/tmp/bagel", "lora_rank": 64},
                "actor": {},
                "rollout": {
                    "name": "vllm_omni",
                    "response_length": 512,
                    "agent": {"default_agent_loop": "bagel_multiturn_agent"},
                },
            },
        }
    )
    trainer._rewrite_bagel_corl_configs()
    model = trainer.config.actor_rollout_ref.model
    assert model.lr_gen == pytest.approx(3e-5)
    pipeline = trainer.config.actor_rollout_ref.rollout.pipeline
    assert pipeline.cfg_text_scale == pytest.approx(1.0)  # UniGRPO CFG-free training


def _non_tensor_stack(values):
    """Version-robust NonTensorStack construction (API moved across tensordict pins)."""
    from tensordict import NonTensorData, NonTensorStack

    if hasattr(NonTensorStack, "from_list_positional_stack"):
        return NonTensorStack.from_list_positional_stack(values)
    return NonTensorStack([NonTensorData(v) for v in values])


def test_normalize_tq_kv_get_result_with_non_tensor_stack():
    """Real kv_batch_get shape: NonTensorStack columns must unwrap to plain values."""
    keys = ["und_k::gen::call0::0", "und_k::gen::call0::1"]
    td = TensorDict({}, batch_size=[2])
    td["gen_group_uid"] = _non_tensor_stack(["call0", "call0"])
    td["seed_index"] = _non_tensor_stack([0, 1])
    td["rm_score"] = _non_tensor_stack([0.5, 0.7])
    out = _normalize_tq_kv_get_result(td, keys)
    assert set(out.keys()) == set(keys)
    row0 = out[keys[0]]["fields"]
    assert row0["gen_group_uid"] == "call0"
    assert row0["rm_score"] == pytest.approx(0.5)
    row1 = out[keys[1]]["fields"]
    assert row1["seed_index"] == 1


def test_fetch_gen_records_by_keys_with_mocked_tq(monkeypatch):
    """The child_gen_keys → kv_batch_get gather must survive the real columnar shape."""
    import transfer_queue as tq_module

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    batch = type("B", (), {"extra_info": {}, "partition_id": "train"})()
    und_records = [{"fields": {"child_gen_keys": ["k0", "k1"]}}]

    td = TensorDict({}, batch_size=[2])
    td["gen_group_uid"] = _non_tensor_stack(["call0", "call0"])
    td["seed_index"] = _non_tensor_stack([0, 1])
    monkeypatch.setattr(tq_module, "kv_batch_get", lambda keys, partition_id: td)

    out = trainer._fetch_gen_records_by_keys(batch, und_records)
    assert set(out.keys()) == {"k0", "k1"}
    assert out["k0"]["fields"]["gen_group_uid"] == "call0"
    assert out["k1"]["fields"]["seed_index"] == 1


def test_gen_batch_from_step_aggregates_metrics_over_siblings(monkeypatch):
    """child_gen_keys path: GEN rows survive the gather and J/K are batch means."""
    import transfer_queue as tq_module

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"gen_samples_per_call": 2}}}})
    extra: dict = {}
    batch = type(
        "B",
        (),
        {
            "extra_info": extra,
            "partition_id": "train",
            "non_tensor_batch": {
                "child_gen_keys": [["k0"], ["k1"]],
                "episode_J": [2, 4],
                "episode_K": [1, 0],
                "bagel_corl_metrics": [
                    {
                        "episode/J": 2.0,
                        "episode/K": 1.0,
                        "gen/dropped_incomplete_groups": 0.0,
                        "und/no_image_credit": 0.0,
                        "gen/skipped_no_groups": 0.0,
                    },
                    {
                        "episode/J": 4.0,
                        "episode/K": 0.0,
                        "gen/dropped_incomplete_groups": 1.0,
                        "und/no_image_credit": 1.0,
                        "gen/skipped_no_groups": 1.0,
                    },
                ],
            },
        },
    )()
    monkeypatch.setattr(
        tq_module,
        "kv_batch_get",
        lambda keys, partition_id: {
            "k0": {"fields": {"gen_group_uid": "call0", "rm_score": 0.5, "rollout_log_probs": [0.1]}},
            "k1": {"fields": {"gen_group_uid": "call1", "rm_score": 0.6, "rollout_log_probs": [0.2]}},
        },
    )
    gen_batch = trainer._gen_batch_from_step(batch)
    assert [row["gen_group_uid"] for row in gen_batch] == ["call0", "call1"]
    assert extra["episode/J"] == pytest.approx(3.0)  # mean, not first-row-wins
    assert extra["episode/K"] == pytest.approx(0.5)
    assert extra["gen/dropped_incomplete_groups"] == pytest.approx(1.0)
    assert extra["und/no_image_credit"] == pytest.approx(0.5)


def test_stack_padded_ragged_timesteps():
    from verl_omni.trainer.omni.bagel_corl_gen_adv import _stack_padded

    out = _stack_padded([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([9.0])])
    assert out.shape == (2, 3)
    assert torch.equal(out[1], torch.tensor([9.0, 0.0, 0.0]))


def test_build_gen_flowgrpo_proto_resizes_log_probs_to_timesteps():
    from verl_omni.trainer.omni.bagel_corl_gen_adv import build_gen_flowgrpo_proto

    rows = [
        {
            "gen_group_uid": "g0",
            "rm_score": 1.0,
            "rollout_log_probs": [0.1, 0.2, 0.3],
            "all_latents": torch.randn(3, 4),
            "timesteps": [999.0, 500.0, 100.0],
        },
        {
            "gen_group_uid": "g1",
            "rm_score": 0.5,
            "rollout_log_probs": [0.4, 0.5],
            "all_latents": torch.randn(5, 4),
            "timesteps": [999.0, 700.0, 500.0, 300.0, 100.0],
        },
    ]
    proto = build_gen_flowgrpo_proto(rows)
    assert proto.batch["all_timesteps"].shape == (2, 5)
    # old_log_probs zero-padded from 3 → 5 timesteps to stay aligned.
    assert proto.batch["old_log_probs"].shape == (2, 5)
    assert torch.equal(proto.batch["old_log_probs"][0, 3:], torch.zeros(2))


def test_build_gen_flowgrpo_proto_empty_and_missing_traj():
    from verl_omni.trainer.omni.bagel_corl_gen_adv import build_gen_flowgrpo_proto

    assert build_gen_flowgrpo_proto([]) is None
    with pytest.raises(RuntimeError, match="lacks all_latents"):
        build_gen_flowgrpo_proto([{"gen_group_uid": "g", "rm_score": 1.0, "rollout_log_probs": [0.1]}])


def test_split_und_gen_empty_records():
    from verl_omni.agent_loop.bagel_corl_tq import split_und_gen_metas

    und_batch, gen_batch = split_und_gen_metas([], None)
    assert und_batch == [] and gen_batch == []
