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

import json
import pathlib
import types

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.trainer.ppo.v1.trainer_base import get_trainer_cls
from verl.utils import tensordict_utils as tu
from verl.utils.tensordict_utils import assign_non_tensor_data

import verl_omni.trainer.omni  # noqa: F401  registers bagel_corl_sync
from verl_omni.pipelines.bagel_flow_grpo.bagel_corl import DUAL_LORA_TARGET_MODULES, validate_disjoint_lora_targets
from verl_omni.trainer.omni.bagel_corl_trainer import (
    OmniBagelCoRLTrainerSync,
    _actor_merges_lora,
    _normalize_tq_kv_get_result,
)
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl_omni.utils.config import validate_bagel_corl_config, validate_config
from verl_omni.workers.config.diffusion import (
    DiffusionLossConfig,
    DiffusionPipelineConfig,
    DiffusionSamplingConfig,
)
from verl_omni.workers.utils.losses import bagel_composite_loss

# RFC §4.0.2 dual-lane LoRA: bagel_corl_sync requires the explicit list, so fixtures use
# the canonical UND+GEN split instead of the shared "all-linear" default.
_BAGEL_LORA_TARGETS = list(DUAL_LORA_TARGET_MODULES)

_AR_DEPLOY = (
    pathlib.Path(__file__).resolve().parents[3]
    / "examples"
    / "agenticllmgrpo_trainer"
    / "bagel"
    / "bagel_corl_deploy_ar.yaml"
)


def _corl_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "disable",
                "v1": {"trainer_mode": "bagel_corl_sync"},
            },
            "actor_rollout_ref": {
                "model": {
                    "path": "/models/ByteDance-Seed/BAGEL-7B-MoT",
                    "lora_rank": 64,
                    "target_modules": list(_BAGEL_LORA_TARGETS),
                },
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
                    "target_modules": list(_BAGEL_LORA_TARGETS),
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
                        # RFC §5 knob SoT: no code default for these three.
                        "gen_samples_per_call": 2,
                        "max_generate_passes": 1,
                        "max_und_turns": 8,
                    },
                },
            },
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
    # Passed through from the recipe, not invented by the rewrite (RFC §5).
    assert agent.gen_samples_per_call == 2
    assert agent.max_generate_passes == 1
    assert agent.max_und_turns == 8
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
    validate_bagel_corl_config(_corl_cfg(actor_rollout_ref={"rollout": {"n": 8, "agent": {"gen_samples_per_call": 4}}}))
    validate_bagel_corl_config(_corl_cfg(actor_rollout_ref={"rollout": {"n": 3, "agent": {"gen_samples_per_call": 2}}}))


def test_default_corl_config_validates():
    validate_config(_corl_cfg())


def test_und_ar_serving_ready_required():
    with pytest.raises(ValueError, match="dual-role UND AR serving is not ready"):
        validate_bagel_corl_config(_corl_cfg(actor_rollout_ref={"rollout": {"agent": {"und_ar_serving_ready": False}}}))


def test_und_deploy_config_required_when_ready():
    with pytest.raises(ValueError, match="und_deploy_config"):
        validate_bagel_corl_config(
            _corl_cfg(
                actor_rollout_ref={"rollout": {"agent": {"und_ar_serving_ready": True, "und_deploy_config": None}}}
            )
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


def _fake_bagel_snapshot(tmp_path):
    """Minimal Bagel-shaped snapshot: the root ``config.json`` is weights-only.

    This is the real checkpoint layout — ``config.json`` declares
    ``model_type: bagel`` with no ``auto_map`` and no modeling code, so
    ``AutoConfig.from_pretrained`` on the snapshot root raises. The LLM sub-config
    (``llm_config.json``) is what transformers *can* resolve.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "bagel", "architectures": ["OmniBagelForConditionalGeneration"]})
    )
    (tmp_path / "llm_config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 128,
                "tie_word_embeddings": False,
            }
        )
    )
    return tmp_path


def test_und_ar_model_node_pointing_at_the_snapshot_root_reproduces_the_reported_crash(tmp_path):
    """Pins the failure this branch had to fix: ``AutoConfig`` on the snapshot root."""
    snap = _fake_bagel_snapshot(tmp_path)
    with pytest.raises(Exception, match="model type `bagel`"):
        hydra.utils.instantiate(
            {
                "_target_": "verl_omni.workers.config.omni.OmniModelConfig",
                "path": str(snap),
                "tokenizer_path": str(snap),
                "model_type": "omni_model",
                "architecture": "OmniBagelForConditionalGeneration",
                "trust_remote_code": True,
                "composite_mode": "bagel_corl",
                "load_tokenizer": False,
                "lora_rank": 8,
                "lora_alpha": 16,
            }
        )


def test_und_ar_hf_config_path_resolves_to_the_llm_subconfig(tmp_path):
    """``OmniModelConfig.__post_init__`` falls back to ``path``, so the UND AR node
    must carry the snapshot's ``llm_config.json`` explicitly (or the trainer dies
    with the InstantiationException above, before any worker starts)."""
    snap = _fake_bagel_snapshot(tmp_path)
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)

    resolved = trainer._resolve_und_hf_config_path(OmegaConf.create({"path": str(snap)}), OmegaConf.create({}))
    assert resolved == str(snap / "llm_config.json")

    # Explicit ``agent.und_hf_config_path`` wins over the derived sub-config.
    explicit = str(snap / "llm_config.json")
    assert (
        trainer._resolve_und_hf_config_path(
            OmegaConf.create({"path": str(snap)}), OmegaConf.create({"und_hf_config_path": explicit})
        )
        == explicit
    )
    # No model path at all: nothing to derive (caller must set the knob).
    assert trainer._resolve_und_hf_config_path(OmegaConf.create({}), OmegaConf.create({})) is None


def test_und_ar_model_node_with_the_resolved_hf_config_instantiates(tmp_path):
    snap = _fake_bagel_snapshot(tmp_path)
    cfg = hydra.utils.instantiate(
        {
            "_target_": "verl_omni.workers.config.omni.OmniModelConfig",
            "path": str(snap),
            "tokenizer_path": str(snap),
            "model_type": "omni_model",
            "architecture": "OmniBagelForConditionalGeneration",
            "trust_remote_code": True,
            "composite_mode": "bagel_corl",
            "hf_config_path": str(snap / "llm_config.json"),
            "load_tokenizer": False,
            "lora_rank": 8,
            "lora_alpha": 16,
        }
    )
    assert cfg.hf_config.model_type == "qwen2"
    assert cfg.local_hf_config_path == str(snap / "llm_config.json")


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


def test_und_train_loss_data_satisfies_the_ppo_loss_contract(monkeypatch):
    """``ppo_loss`` must find the full-sequence grid plus per-row ``prompts``/``responses``.

    ``ppo_loss`` re-derives the response slice itself: its first act is
    ``no_padding_2_padding(model_output["log_probs"], data)``
    (``verl/workers/utils/losses.py:59`` -> ``workers/utils/padding.py:99``), which walks
    ``values[seq_offset - resp_len - 1 : seq_offset - 1]`` per row and so needs each grid row to span
    ``prompt + response`` -- and needs the lengths from ``data["prompts"]``/``data["responses"]``.
    The UND train pass used to hand it a response-only ``(B, R)`` grid and a three-field
    TensorDict, so the first *reachable* UND train step died with

        KeyError: 'key "prompts" not found in TensorDict with keys
        ['old_log_probs', 'response_mask', 'rollout_log_probs', 'uid']'

    (measured 2026-09-18 on hk01dgx012, devices 4-7, right after the UND contract, the reward
    column and the flag plumbing all went green). This runs the real ``no_padding_2_padding`` over
    what the pass builds, which also catches the padded-branch misread: with the two rows flattened
    to 2-D, row 1 (prompt 2, response 2) would report a prompt length of 3 -- ``attention_mask``'s
    first three columns already cover one of its response tokens -- and the slice would land on the
    wrong log-probs instead of failing.
    """
    composite, _ = _spy_on_und_split(monkeypatch)
    from verl.workers.utils.padding import no_padding_2_padding

    seen: dict = {}

    def _loss(model_output, data, **kwargs):
        seen["data"] = data
        seen["model_output"] = model_output
        seen["sliced"] = no_padding_2_padding(model_output["log_probs"], data)
        return torch.zeros((), requires_grad=True), {}

    engine = _stub_bagel_engine(types.SimpleNamespace(micro_batch_size_per_gpu=_UND_TRAIN_MICRO_BSZ))
    engine.module = _UndModelStub(torch.arange(1.0, 9.0).view(2, 4)[:, :3])
    composite.run_und_token_forward_backward(
        engine, _und_batch(tensor_keys=("old_log_probs", "advantages")), _loss, forward_only=False
    )

    loss_data = seen["data"]
    assert loss_data["prompts"].is_nested
    assert loss_data["responses"].is_nested
    assert tu.get_non_tensor_data(loss_data, key="max_response_len", default=None) == 2
    # Row 0's single response token is its third; row 1's two are its own third and fourth. The
    # values are the ``(B, L-1)`` grid's entries for those tokens (entry ``j`` = token ``j + 1``).
    assert torch.equal(seen["sliced"], torch.tensor([[3.0, 0.0], [6.0, 7.0]]))


def test_und_train_loss_refuses_a_padded_prompt_response_pair(monkeypatch):
    """A padded ``prompts``/``responses`` pair would mis-score silently; it must fail loud instead.

    The padded branch of ``no_padding_2_padding`` reads ``attention_mask[:, :prompts.shape[1]]`` as
    the prompt (``verl/verl/workers/utils/padding.py:130``), which is only valid when every row
    shares one prompt length. With the rows the guard tests use (3+1 and 2+2) the flattening turns
    row 1's response into ``[[3.0, 0.0], [7.0, 0.0]]`` -- one wrong log-prob and one dropped -- and
    the shapes still line up, so nothing downstream would notice.
    """
    composite, _ = _spy_on_und_split(monkeypatch)

    def _padded_spy(data, **kwargs):
        keys = {
            key: data[key]
            for key in ("input_ids", "attention_mask", "response_mask", "old_log_probs", "advantages")
            if key in data.keys()
        }
        keys["prompts"] = torch.tensor([[1, 2, 3], [4, 5, 0]])
        keys["responses"] = torch.tensor([[6, 0], [7, 8]])
        return [TensorDict(keys, batch_size=[2])], None

    monkeypatch.setattr(composite, "prepare_micro_batches", _padded_spy)
    engine = _stub_bagel_engine(types.SimpleNamespace(micro_batch_size_per_gpu=_UND_TRAIN_MICRO_BSZ))

    with pytest.raises(ValueError, match="arrived padded"):
        composite.run_und_token_forward_backward(
            engine,
            _und_batch(tensor_keys=("old_log_probs", "advantages")),
            _und_train_loss,
            forward_only=False,
        )


def test_und_train_loss_uses_the_micro_batch_mean_agg_loss_contract(monkeypatch):
    """The UND term must be the *micro-batch* mean, and ``agg_loss`` must accept it.

    RFC §4.4/§4.10: both branches normalize by their own micro-batch counts (GEN by
    ``len(micro_batches) * num_timesteps`` in ``diffusers_impl.py:893``, UND by
    ``len(micro_batches)`` in ``bagel_composite_loss``), so ``ppo_loss``'s term has to be a mean over
    the micro-batch. That is ``agg_loss``'s ``dp_size == 1`` / ``batch_num_tokens is None`` branch
    (``core_algos.py:1169-1173``); handing it the DP-reduced global count is a different
    normalization. Run9 died on the missing-count path:

        ValueError: (global) batch_num_tokens is required when dp_size > 1

    (measured 2026-09-18 on hk01dgx012, devices 4-7, the first step past
    ``no_padding_2_padding``). This drives the real ``agg_loss`` with whatever the pass put on the
    loss data, so it fails if the values drift back toward the global convention.
    """
    from verl.trainer.ppo.core_algos import agg_loss

    composite, _ = _spy_on_und_split(monkeypatch)
    # A 2-way DP group: the point is that its size must NOT reach ``agg_loss``.
    engine = _stub_bagel_engine(types.SimpleNamespace(micro_batch_size_per_gpu=_UND_TRAIN_MICRO_BSZ))
    engine.get_data_parallel_size = lambda: 2
    engine.module = _UndModelStub(torch.arange(1.0, 9.0).view(2, 4)[:, :3])
    seen: dict = {}

    def _loss(model_output, data, **kwargs):
        seen["data"] = data
        return torch.zeros((), requires_grad=True), {}

    composite.run_und_token_forward_backward(
        engine, _und_batch(tensor_keys=("old_log_probs", "advantages")), _loss, forward_only=False
    )

    loss_data = seen["data"]
    mask = loss_data["response_mask"].to(bool)
    loss = agg_loss(
        loss_mat=torch.ones_like(mask, dtype=torch.float32),
        loss_mask=mask,
        loss_agg_mode="token-mean",
        dp_size=tu.get_non_tensor_data(loss_data, key="dp_size", default=None),
        batch_num_tokens=tu.get_non_tensor_data(loss_data, key="batch_num_tokens", default=None),
        global_batch_size=tu.get_non_tensor_data(loss_data, key="global_batch_size", default=None),
    )
    # An all-ones loss matrix must come back as exactly 1.0: mean over this micro-batch, unscaled.
    assert loss.item() == pytest.approx(1.0)


def test_the_engine_dp_size_never_scales_the_und_lane(monkeypatch):
    """``dp_size`` on the loss data is a normalization choice, not the engine's world size."""
    composite, _ = _spy_on_und_split(monkeypatch)
    engine = _stub_bagel_engine(types.SimpleNamespace(micro_batch_size_per_gpu=_UND_TRAIN_MICRO_BSZ))
    engine.get_data_parallel_size = lambda: 4
    seen: dict = {}

    def _loss(model_output, data, **kwargs):
        seen["data"] = data
        return torch.zeros((), requires_grad=True), {}

    composite.run_und_token_forward_backward(
        engine, _und_batch(tensor_keys=("old_log_probs", "advantages")), _loss, forward_only=False
    )

    assert tu.get_non_tensor_data(seen["data"], key="dp_size", default=None) == 1
    assert tu.get_non_tensor_data(seen["data"], key="batch_num_tokens", default=None) is None


def test_und_grid_keeps_the_full_sequence_for_both_lanes(monkeypatch):
    """The train and infer lanes must publish the same ``prompt + response`` grid convention."""
    composite, _ = _spy_on_und_split(monkeypatch)
    seen: dict = {}

    def _loss(model_output, data, **kwargs):
        seen["grid"] = model_output["log_probs"]
        return torch.zeros((), requires_grad=True), {}

    engine = _stub_bagel_engine(types.SimpleNamespace(micro_batch_size_per_gpu=_UND_TRAIN_MICRO_BSZ))
    engine.module = _UndModelStub(torch.arange(1.0, 9.0).view(2, 4)[:, :3])
    composite.run_und_token_forward_backward(
        engine, _und_batch(tensor_keys=("old_log_probs", "advantages")), _loss, forward_only=False
    )

    grid = seen["grid"]
    assert grid.is_nested
    # One row per sequence, each spanning its own ``attention_mask`` length -- not the response.
    assert [row.numel() for row in grid.unbind()] == [4, 4]


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


def test_advantage_flags_survive_the_kv_batch_put_hand_off(monkeypatch):
    """``_compute_advantage`` must publish its lane flags onto the meta ``super()`` returns.

    The v1 base ends that method with ``batch = tq.kv_batch_put(...)``. ``kv_batch_put`` builds its
    result from a meta it creates internally (``extra_info=batch_meta.extra_info``), so the returned
    ``KVBatchMeta`` carries the dataclass default ``{}`` and never sees the incoming batch's dict.
    ``step()`` then rebinds ``batch`` to that new object, so any flag written to the *incoming*
    batch is dropped before ``_update_actor`` reads it back.

    Measured 2026-09-23 on devices 0,1,6,7: ``bagel_corl_sync advantage ... skip_gen=False`` was
    followed 35 ms later by ``bagel_corl_sync update_actor skip_gen=True`` on every step -- GEN
    advantages were computed and then never optimised. The stub below returns a *new* object with a
    fresh ``extra_info``, which is what the real parent does; a stub returning the same batch cannot
    catch this, which is why the existing view test above passed while the run was broken.
    """
    from types import SimpleNamespace

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

    def _rows(scores_by_group):
        rows = []
        for group, scores in scores_by_group:
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
        return rows

    def _fresh_meta(self, batch, metrics):
        # Mimic ``tq.kv_batch_put``: a brand-new carrier whose ``extra_info`` is the default ``{}``.
        return SimpleNamespace(extra_info={})

    monkeypatch.setattr(OmniPPOTrainerSync, "_compute_advantage", _fresh_meta)

    # Both GEN samples of each group present -> the lane is trainable.
    incoming = SimpleNamespace(
        extra_info={"gen_batch": _rows((("callA", (1.0, 0.0)), ("callB", (0.5, 0.5))))}
    )
    result = trainer._compute_advantage(incoming, {})

    assert result is not incoming, "the parent hands back a new carrier; the test must model that"
    assert result.extra_info["has_complete_gen_groups"] is True
    assert result.extra_info["skip_gen"] is False
    assert result.extra_info["num_gen_rows"] == 4
    assert result.extra_info["bagel_corl_gen"] is not None
    assert "advantages" in result.extra_info["bagel_corl_gen"].keys()

    # No GEN rows at all -> the flags must still reach the returned carrier, now skipping GEN.
    skipped = trainer._compute_advantage(SimpleNamespace(extra_info={"gen_batch": []}), {})
    assert skipped.extra_info["has_complete_gen_groups"] is False
    assert skipped.extra_info["skip_gen"] is True


def test_gen_prompt_token_ids_stay_a_1d_object_array():
    """Uniform prompt lengths must not collapse ``prompt_token_ids`` into a 2-D matrix.

    ``np.array(list_of_lists, dtype=object)`` returns a ``(B, L)`` **2-D** object array when
    every row happens to share one length, and a ``(B,)`` array of lists when they are ragged.
    Only the ragged case used to survive the consumer: for the 2-D case
    ``prompt_token_ids[i]`` is an ``ndarray``, and ``torch.as_tensor(ids, dtype=torch.long)``
    rejects any object-dtype array even when every element is an int:

        TypeError: can't convert np.ndarray of type numpy.object_.

    Measured 2026-09-24: run ``bagel_corl_rm1_20260923_234021`` died at step 9 this way after
    9 otherwise clean steps. Identical lengths are the trigger, so both cases are pinned here.
    """
    import numpy as np

    from verl_omni.trainer.omni.bagel_corl_gen_adv import build_gen_flowgrpo_proto

    def _rows(lengths):
        rows = []
        for index, length in enumerate(lengths):
            rows.append(
                {
                    "gen_group_uid": "callA",
                    "gen_sample_uid": f"callA:{index}",
                    "rollout_log_probs": [0.1, 0.2, 0.3],
                    "rm_score": 1.0 if index == 0 else 0.0,
                    "all_latents": torch.randn(3, 4),
                    "timesteps": torch.tensor([900.0, 600.0, 300.0]),
                    "prompt_token_ids": list(range(10, 10 + length)),
                }
            )
        return rows

    # The regression: identical lengths used to yield shape (2, 5) and crash the adapter.
    for label, lengths in (("identical", [5, 5]), ("ragged", [3, 5])):
        proto = build_gen_flowgrpo_proto(_rows(lengths))
        ids = proto.non_tensor_batch["prompt_token_ids"]
        assert isinstance(ids, np.ndarray), label
        assert ids.dtype == object, label
        assert ids.ndim == 1, f"{label}: expected 1-D, got shape {ids.shape}"
        # Each row must be the row's own flat sequence, not a column slice of a matrix.
        assert list(ids[0]) == list(range(10, 10 + lengths[0])), label


def test_as_token_id_list_normalises_dtypes_and_rejects_nesting():
    """The consumer-side guard: object dtype is normalised, real nesting is rejected."""
    import numpy as np

    from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import _as_token_id_list

    assert _as_token_id_list([1, 2, 3], index=0) == [1, 2, 3]
    # The regression itself: a 1-D object-dtype row. ``torch.as_tensor`` rejects this dtype
    # even though every element is an int, so the normalisation is what saves the adapter.
    assert _as_token_id_list(np.array([1, 2, 3], dtype=object), index=0) == [1, 2, 3]
    # Exactly the row the upstream collapse produced: ``arr[i]`` of a 2-D object matrix is a
    # 1-D object-dtype array, so shape was never the problem -- dtype was.
    assert _as_token_id_list(np.array([[1, 2, 3], [4, 5, 6]], dtype=object)[0], index=0) == [1, 2, 3]
    assert _as_token_id_list(np.array([1, 2, 3], dtype=np.int64), index=0) == [1, 2, 3]
    assert _as_token_id_list(torch.tensor([[1, 2, 3]]), index=0) == [1, 2, 3]

    # A genuinely nested payload (a batch of sequences handed in as one row) must raise a
    # pointed error rather than let torch emit its opaque dtype complaint.
    with pytest.raises(ValueError, match="expected one flat sequence"):
        _as_token_id_list(np.array([[1, 2], [3, 4]], dtype=object), index=0)
    with pytest.raises(ValueError, match="not a flat sequence of ints"):
        _as_token_id_list([{"not": "a token"}], index=0)


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
    config = SimpleNamespace(diffusion_loss=OmegaConf.create({"loss_weight_und": 2.0, "loss_weight_gen": 3.0}))

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


def test_lane_views_are_moved_onto_the_forward_device_before_dispatch():
    """``bagel_corl_{und,gen}`` views must be aligned with the model output device.

    Both views are parked as **non-tensor** data, so the diffusion engine's
    ``micro_batch.to(device)`` (``diffusers_impl.py:983``) never recurses into them -- ``TensorDict.to``
    leaves non-tensor payloads alone. Their tensors are allocated on the CPU by
    ``build_gen_flowgrpo_proto``, so the first elementwise op against a cuda model output died:

        RuntimeError: Expected all tensors to be on the same device, but found at least two devices,
        cuda:0 and cpu!   (``diffusion_algos.py:350``, ``log_ratio = log_prob - old_log_prob``)

    (measured 2026-09-23 18:50 on hk01dgx039, devices 0/1/6/7, ``bagel_corl_rm1_20260923_183457``).

    ``meta`` stands in for cuda: the views start on the CPU (as they really do) and the forward output
    starts on ``meta``, so the assertion below is a genuine cross-device move rather than a same-device
    no-op. (``meta`` -> cpu is not permitted by torch, which is why the stand-in runs this direction.)
    """
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
    # The engine stamps these on the *micro-batch* (``diffusers_impl.py:984``); the lane views are
    # non-tensor stashes and never receive them, so the loss has to inherit them. Without that,
    # ``gradient_accumulation_steps`` is None and ``diffusion_loss`` dies on the division.
    assign_non_tensor_data(data, "gradient_accumulation_steps", 3)
    assign_non_tensor_data(data, "sp_size", 1)

    # The forward "ran on cuda" (meta here), so that is the device the lane views have to match.
    model_output = {
        "und": {"log_probs": torch.zeros(2, 3, device="meta", requires_grad=True)},
        "gen": {"log_probs": torch.zeros(2, device="meta", requires_grad=True)},
    }
    config = SimpleNamespace(diffusion_loss=OmegaConf.create({"loss_weight_und": 1.0, "loss_weight_gen": 1.0}))

    seen: dict[str, str] = {}
    inherited: dict[str, object] = {}

    def fake_ppo(config, model_output, data, dp_group=None):
        seen["und"] = data["old_log_probs"].device.type
        return torch.tensor(1.0, requires_grad=True), {}

    def fake_diff(config, model_output, data, dp_group=None):
        seen["gen"] = data["old_log_probs"].device.type
        inherited["gradient_accumulation_steps"] = tu.get_non_tensor_data(
            data, "gradient_accumulation_steps", default=None
        )
        inherited["sp_size"] = tu.get_non_tensor_data(data, "sp_size", default=None)
        return torch.tensor(2.0, requires_grad=True), {}

    with (
        patch("verl.workers.utils.losses.ppo_loss", fake_ppo),
        patch("verl_omni.workers.utils.losses.diffusion_loss", fake_diff),
    ):
        bagel_composite_loss(config=config, model_output=model_output, data=data)

    assert seen == {"und": "meta", "gen": "meta"}, (
        f"lane views must be moved onto the forward device before dispatch, got {seen}"
    )
    assert inherited == {"gradient_accumulation_steps": 3, "sp_size": 1}, (
        f"lane views must inherit the micro-batch's non-tensor scalars, got {inherited}"
    )


def test_first_tensor_device_sees_through_nested_output_and_align_tolerates_plain_dicts():
    """The device probe must find a nested ``log_probs``, and alignment must not choke on odd views."""
    from types import SimpleNamespace

    from verl_omni.workers.utils.losses import _align_view_to_device, _first_tensor_device

    assert _first_tensor_device({"gen": {"log_probs": torch.zeros(1, device="meta")}}) == torch.device("meta")
    # A payload with no tensor at all reports None rather than raising, so the caller can fall back.
    assert _first_tensor_device({"a": {"b": 1.0}}) is None
    assert _first_tensor_device(None) is None

    # A plain mapping is handled as well as a TensorDict.
    plain = {"old_log_probs": torch.zeros(1)}
    _align_view_to_device(plain, torch.device("meta"))
    assert plain["old_log_probs"].device.type == "meta"

    # A matching device is a no-op, and a None device short-circuits.
    same = {"old_log_probs": torch.zeros(1)}
    _align_view_to_device(same, torch.device("cpu"))
    assert same["old_log_probs"].device.type == "cpu"
    assert _align_view_to_device(same, None) is None
    # A non-mapping payload is left alone rather than raising.
    _align_view_to_device(SimpleNamespace(old_log_probs=1), torch.device("meta"))


def test_diffusion_loss_tolerates_an_ar_config_without_the_distill_node():
    """The GEN lane's loss must not need the diffusion-only distill node to exist.

    ``use_distill_loss`` / ``distill_loss_mode`` / ``distill_loss_coef`` are declared on
    ``DiffusionActorConfig``, but the Co-RL composite hands ``diffusion_loss`` the outer AR trainer's
    ``OmniActorConfig``, which declares none of them. The unconditional read killed the GEN loss on
    the first step that reached it:

        AttributeError: 'OmniActorConfig' object has no attribute 'use_distill_loss'.

    (measured 2026-09-23 19:12 on hk01dgx039, devices 0/1/6/7, ``bagel_corl_rm1_20260923_185651``).

    ``rollout_correction`` on the same object was the identical failure one step earlier, so the
    assertion below pins the *class* of problem: every node the diffusion loss reads off the actor
    config has to be optional.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from verl_omni.workers.utils.losses import diffusion_loss

    # Only the fields ``OmniActorConfig`` really declares; no distill node, no rollout_correction.
    ar_config = SimpleNamespace(
        global_batch_info={},
        loss_scale_factor=1.0,
        diffusion_loss=OmegaConf.create({"loss_mode": "flow_grpo"}),
        use_kl_loss=False,
        rollout_correction=None,
    )
    data = TensorDict(
        {
            "advantages": torch.ones(2, 2),
            "old_log_probs": torch.zeros(2, 2),
            "return": torch.ones(2, 2),
            "response_mask": torch.ones(2, 2),
        },
        batch_size=[2],
    )
    # ``diffusion_loss`` divides by the accumulation count and scales by ``sp_size``; the engine
    # stamps both as non-tensor data, so the fixture has to as well.
    assign_non_tensor_data(data, "gradient_accumulation_steps", 1)
    assign_non_tensor_data(data, "sp_size", 1)
    model_output = {"log_probs": torch.zeros(2, requires_grad=True)}

    with patch("verl_omni.workers.utils.losses.get_diffusion_loss_fn") as get_fn:
        stub = get_fn.return_value
        stub.required_model_output_keys = ["log_probs"]
        stub.validate_inputs.return_value = None
        stub.return_value = SimpleNamespace(
            loss=torch.tensor(0.0, requires_grad=True) * 1.0, metrics={}, add_loss_metric=True
        )
        loss, _metrics = diffusion_loss(ar_config, model_output, data)

    assert float(loss) == pytest.approx(0.0)
    # The distill term must not have been dispatched at all.
    assert get_fn.call_count == 1
    assert get_fn.call_args.args[0] == "flow_grpo"


def test_diffusion_loss_still_activates_distillation_from_the_loss_mode():
    """Selecting distillation via ``diffusion_loss.loss_mode`` must still dispatch the distill term.

    ``diffusion_trainer_utils._validate_distill_config`` already treats "``loss_mode`` is a distill
    mode" as equivalent to ``use_distill_loss``, so the loss has to agree with it -- otherwise making
    the flag optional would silently drop a configured term.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from verl_omni.workers.utils.losses import diffusion_loss

    ar_config = SimpleNamespace(
        global_batch_info={},
        loss_scale_factor=1.0,
        diffusion_loss=OmegaConf.create({"loss_mode": "distill_kl"}),
        use_kl_loss=False,
    )
    data = TensorDict(
        {
            "advantages": torch.ones(2, 2),
            "old_log_probs": torch.zeros(2, 2),
            "return": torch.ones(2, 2),
            "response_mask": torch.ones(2, 2),
        },
        batch_size=[2],
    )
    # ``diffusion_loss`` divides by the accumulation count and scales by ``sp_size``; the engine
    # stamps both as non-tensor data, so the fixture has to as well.
    assign_non_tensor_data(data, "gradient_accumulation_steps", 1)
    assign_non_tensor_data(data, "sp_size", 1)
    model_output = {"log_probs": torch.zeros(2, requires_grad=True)}

    with patch("verl_omni.workers.utils.losses.get_diffusion_loss_fn") as get_fn:
        stub = get_fn.return_value
        stub.required_model_output_keys = ["log_probs"]
        stub.validate_inputs.return_value = None
        # A real loss returns a non-leaf tensor (``loss_value += ...`` is in-place), so the stub
        # must not hand back a leaf that requires grad.
        stub.return_value = SimpleNamespace(
            loss=torch.tensor(0.5, requires_grad=True) * 1.0, metrics={}, add_loss_metric=True
        )
        diffusion_loss(ar_config, model_output, data)

    dispatched = [call.args[0] for call in get_fn.call_args_list]
    # ``loss_mode`` itself is dispatched first, then the distill term -- both are "distill_kl" here.
    assert dispatched.count("distill_kl") >= 1, f"distill term was not dispatched: {dispatched}"


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
                "model": {"path": "/tmp/bagel", "lora_rank": 64, "target_modules": list(_BAGEL_LORA_TARGETS)},
                "actor": {},
                "rollout": {
                    "name": "vllm_omni",
                    "response_length": 512,
                    "agent": {
                        "default_agent_loop": "bagel_multiturn_agent",
                        # RFC §5 knob SoT: no code default for these three.
                        "gen_samples_per_call": 2,
                        "max_generate_passes": 1,
                        "max_und_turns": 8,
                    },
                },
            },
        }
    )
    trainer._rewrite_bagel_corl_configs()
    model = trainer.config.actor_rollout_ref.model
    assert model.lr_gen == pytest.approx(3e-5)
    pipeline = trainer.config.actor_rollout_ref.rollout.pipeline
    assert pipeline.cfg_text_scale == pytest.approx(1.0)  # UniGRPO CFG-free training
    # RFC §5: the train side reads model.pipeline.cfg_text_scale. Leaving it at the
    # Bagel CFG default (4.0) while rollout ran CFG-free silently biased the GEN ratio.
    assert model.pipeline.cfg_text_scale == pytest.approx(pipeline.cfg_text_scale)

    # The node must also be *typed*: ``omni_model.yaml`` declares no ``pipeline`` node at
    # all, so a bare mapping (even one carrying cfg_text_scale) satisfies every assert
    # above and still kills the GEN engine — ``diffusers_training_adapter.build_scheduler``
    # reads ``model_config.pipeline.num_inference_steps``, and ``height`` / ``width`` feed
    # the latent position ids, so an untyped node fails worker init with
    # ``'dict' object has no attribute 'num_inference_steps'``. Both the model and the
    # rollout node therefore need the ``_target_`` that ``diffusion_model.yaml`` and
    # ``diffusion_rollout.yaml`` set on theirs.
    def _plain(node):
        return {k: v for k, v in node.items() if not k.startswith("_")}

    assert _plain(model.pipeline) == _plain(pipeline)  # rollout is the knob SoT (RFC §5)
    for label, node in (("model", model.pipeline), ("rollout", pipeline)):
        instantiated = hydra.utils.instantiate(OmegaConf.to_container(node, resolve=True))
        assert isinstance(instantiated, DiffusionPipelineConfig), (
            f"{label}.pipeline must instantiate to DiffusionPipelineConfig, got {type(instantiated).__name__}"
        )
        for key, value in _plain(node).items():
            assert getattr(instantiated, key) == value


def test_rewrite_retargets_val_kwargs_to_diffusion_sampling():
    """The recipe's ``val_kwargs`` appends must survive as a typed diffusion node.

    The omni schema declares ``actor_rollout_ref.rollout.val_kwargs`` with the *AR*
    ``_target_`` (``verl.workers.config.SamplingConfig``), while
    ``DiffusionRolloutConfig`` types the field as ``DiffusionSamplingConfig``. Appending a
    diffusion-only subtree to that node composes fine and then kills the first
    ``init_model`` on every actor rank, because ``omega_conf_to_dataclass`` honours the
    nested ``_target_``:

        InstantiationException: Error in call to target
        'verl.workers.config.rollout.SamplingConfig':
        TypeError("SamplingConfig.__init__() got an unexpected keyword argument 'pipeline'")
        full_key: actor_rollout_ref.rollout.val_kwargs

    Measured 2026-09-21 on hk01dgx039 (devices 2-5), the first attempt to run the recipe
    with the val appends in place. The rewrite therefore has to filter *and* retarget, the
    same way it does for the parent node and for ``pipeline``: ``do_sample`` is AR-only and
    cannot ride along into ``DiffusionSamplingConfig``.
    """
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = _corl_cfg()
    trainer.config.actor_rollout_ref.rollout.agent.max_und_turns = 8
    # ``_rewrite_bagel_corl_configs`` also injects the actor ``diffusion_loss`` block, so that
    # node has to exist (``_corl_cfg`` omits it).
    trainer.config.actor_rollout_ref.actor = OmegaConf.create({})
    # Mirror the composed omni schema's val_kwargs node, then apply the recipe's two
    # ``+actor_rollout_ref.rollout.val_kwargs.*`` appends on top of it.
    trainer.config.actor_rollout_ref.rollout.val_kwargs = OmegaConf.create(
        {
            "_target_": "verl.workers.config.SamplingConfig",
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0,
            "n": 1,
            "do_sample": False,
            "pipeline": {"num_inference_steps": 50},
            "algo": {"noise_level": 0.0},
        }
    )

    trainer._rewrite_bagel_corl_configs()
    val_kwargs = trainer.config.actor_rollout_ref.rollout.val_kwargs

    # The node must instantiate, which is exactly what crashed before the fix.
    instantiated = hydra.utils.instantiate(OmegaConf.to_container(val_kwargs, resolve=True))
    assert isinstance(instantiated, DiffusionSamplingConfig), (
        f"rollout.val_kwargs must instantiate to DiffusionSamplingConfig, got {type(instantiated).__name__}"
    )
    # The recipe's intent is preserved...
    assert instantiated.pipeline.num_inference_steps == 50
    assert instantiated.algo.noise_level == pytest.approx(0.0)
    # ...and the AR-only key is gone rather than smuggled in.
    assert "do_sample" not in instantiated.__dict__

    # A val node without any appends is still retargeted, so the validate branch of
    # ``composite_agent_loop`` can read ``.pipeline`` / ``.algo`` / ``.seed`` -- none of
    # which exist on the AR SamplingConfig.
    trainer2 = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer2.config = _corl_cfg()
    trainer2.config.actor_rollout_ref.actor = OmegaConf.create({})
    trainer2.config.actor_rollout_ref.rollout.agent.max_und_turns = 8
    trainer2.config.actor_rollout_ref.rollout.val_kwargs = OmegaConf.create(
        {"_target_": "verl.workers.config.SamplingConfig", "top_k": -1, "do_sample": False}
    )
    trainer2._rewrite_bagel_corl_configs()
    bare = hydra.utils.instantiate(
        OmegaConf.to_container(trainer2.config.actor_rollout_ref.rollout.val_kwargs, resolve=True)
    )
    assert isinstance(bare, DiffusionSamplingConfig)
    assert hasattr(bare, "pipeline") and hasattr(bare, "algo") and hasattr(bare, "seed")


def test_rewrite_binds_a_recipe_supplied_diffusion_loss_node_to_the_schema():
    """A partial ``diffusion_loss`` dict from the recipe must still carry the schema defaults.

    The omni schema does not declare ``actor_rollout_ref.actor.diffusion_loss`` at all -- the
    Co-RL recipes create it with an append (``+actor_rollout_ref.actor.diffusion_loss.loss_mode=
    flow_grpo``), which produces a **plain dict**. The loss reads its knobs as attributes, so the
    first step that reached the GEN loss died on

        AttributeError: 'dict' object has no attribute 'adv_clip_max'.

    (measured 2026-09-23 18:16 on hk01dgx039, devices 0/1/6/7, ``bagel_corl_rm1_20260923_175819``,
    after the five earlier GEN-lane fixes finally let the loss run). The rewrite has to stamp
    ``_target_`` so ``omega_conf_to_dataclass`` fills in ``clip_ratio`` / ``adv_clip_max`` /
    ``dpo_beta`` rather than leaving a bare dict behind.
    """
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = _corl_cfg()
    trainer.config.actor_rollout_ref.rollout.agent.max_und_turns = 8
    # Exactly what the recipe's ``+...diffusion_loss.loss_mode=flow_grpo`` append composes to: a
    # dict with one meaningful key and no ``_target_``.
    trainer.config.actor_rollout_ref.actor = OmegaConf.create(
        {"diffusion_loss": {"loss_mode": "flow_grpo"}}
    )

    trainer._rewrite_bagel_corl_configs()

    node = trainer.config.actor_rollout_ref.actor.diffusion_loss
    assert node.get("_target_") == "verl_omni.workers.config.diffusion.DiffusionLossConfig"
    # The recipe's override survives the binding.
    assert node.get("loss_mode") == "flow_grpo"

    # Bind for real and confirm every attribute the loss reads is present with its declared default.
    instantiated = hydra.utils.instantiate(OmegaConf.to_container(node, resolve=True))
    assert isinstance(instantiated, DiffusionLossConfig)
    assert instantiated.loss_mode == "flow_grpo"
    assert instantiated.adv_clip_max == pytest.approx(5.0)
    assert instantiated.clip_ratio == pytest.approx(0.0001)

    # A node the recipe never touched keeps being created whole.
    trainer2 = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer2.config = _corl_cfg()
    trainer2.config.actor_rollout_ref.rollout.agent.max_und_turns = 8
    trainer2.config.actor_rollout_ref.actor = OmegaConf.create({})
    trainer2._rewrite_bagel_corl_configs()
    built = hydra.utils.instantiate(
        OmegaConf.to_container(trainer2.config.actor_rollout_ref.actor.diffusion_loss, resolve=True)
    )
    assert isinstance(built, DiffusionLossConfig)
    assert built.adv_clip_max == pytest.approx(5.0)


def test_loss_cfg_of_upgrades_a_plain_dict_with_the_declared_defaults():
    """The loss-side belt to the rewrite's braces (see ``diffusion_algos._loss_cfg_of``).

    Even if a ``diffusion_loss`` dict reaches the loss unbound, the attribute reads must resolve to
    the declared ``DiffusionLossConfig`` defaults instead of raising. A config node that is not a
    dict (a ``DictConfig``, or the instantiated dataclass) has to pass through untouched.
    """
    from types import SimpleNamespace

    from verl_omni.trainer.diffusion.diffusion_algos import _loss_cfg_of

    upgraded = _loss_cfg_of(SimpleNamespace(diffusion_loss={"loss_mode": "flow_grpo"}))
    assert upgraded.loss_mode == "flow_grpo"
    assert upgraded.adv_clip_max == pytest.approx(5.0)
    assert upgraded.clip_ratio == pytest.approx(0.0001)
    assert upgraded.dpo_beta == pytest.approx(2000.0)

    # An explicit value in the dict wins over the default.
    overridden = _loss_cfg_of(
        SimpleNamespace(diffusion_loss={"loss_mode": "flow_grpo", "adv_clip_max": 9.0})
    )
    assert overridden.adv_clip_max == pytest.approx(9.0)

    # A typed node is returned as-is, not rebuilt.
    typed = DiffusionLossConfig(loss_mode="flow_grpo", adv_clip_max=7.0)
    assert _loss_cfg_of(SimpleNamespace(diffusion_loss=typed)) is typed
    node = OmegaConf.create({"_target_": "x", "loss_mode": "dpo"})
    assert _loss_cfg_of(SimpleNamespace(diffusion_loss=node)) is node


def test_rewrite_fails_loud_on_non_list_lora_targets():
    """``target_modules`` must be an explicit list for the dual-lane LoRA split.

    The omni model yaml defaults the field to the *string* ``all-linear``; iterating it
    reports nonsense unknown names, so the driver refuses it instead.
    """
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = _corl_cfg()
    trainer.config.actor_rollout_ref.rollout.agent.max_und_turns = 8
    trainer.config.actor_rollout_ref.model.target_modules = "all-linear"
    with pytest.raises(ValueError, match="target_modules"):
        trainer._rewrite_bagel_corl_configs()


def test_disjoint_lora_targets_rejects_a_bare_string():
    """Locks the low-level guard: a string is iterable, so it must never reach the split."""
    with pytest.raises(ValueError, match="explicit list"):
        validate_disjoint_lora_targets("all-linear")
    # The real recipe list still validates and splits UND from GEN.
    und, gen = validate_disjoint_lora_targets(_BAGEL_LORA_TARGETS)
    assert und and gen and not (und & gen)


def _non_tensor_stack(values):
    """Version-robust NonTensorStack construction (API moved across tensordict pins).

    On tensordict 0.10 the only correct form is ``from_list``: the bare
    ``NonTensorStack([...])`` constructor collapses a list into a single element
    (batch_size ``[1]``), which then trips this file's failures with the lazy-TD
    "Received a new batch size torch.Size([2]) with an existing batch_size
    torch.Size([1])" error instead of exercising the code under test.
    """
    from tensordict import NonTensorData, NonTensorStack

    items = [v if isinstance(v, NonTensorData) else NonTensorData(v) for v in values]
    if hasattr(NonTensorStack, "from_list"):
        return NonTensorStack.from_list(items)
    if hasattr(NonTensorStack, "from_list_positional_stack"):
        return NonTensorStack.from_list_positional_stack(values)
    return NonTensorStack(items)


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


def test_extra_fields_from_tq_reads_the_columnar_column(monkeypatch):
    """``kv_batch_get`` hands back a columnar TensorDict, not a keyed dict.

    Returning that object unchanged made ``_und_records_from_batch`` iterate *field names* (a
    TensorDict's iterator yields keys) and build one empty record per field, so no record had
    ``child_gen_keys``, ``build_gen_flowgrpo_proto`` returned None and the GEN lane silently
    disappeared even though the pack had written the seed rows.
    """
    import transfer_queue as tq_module
    from transfer_queue import KVBatchMeta

    column = TensorDict({}, batch_size=[2])
    column["extra_fields"] = [{"child_gen_keys": ["u_0_0::gen::c::0"]}, {"child_gen_keys": []}]
    monkeypatch.setattr(tq_module, "kv_batch_get", lambda keys, partition_id, select_fields=None: column)

    batch = KVBatchMeta(
        partition_id="train",
        keys=["u_0_0", "u_1_0"],
        tags=[{"bagel_role": "und"}, {"bagel_role": "und"}],
    )
    fields = OmniBagelCoRLTrainerSync._extra_fields_from_tq(batch)
    assert [row["child_gen_keys"] for row in fields] == [["u_0_0::gen::c::0"], []]


def test_und_records_drop_padding_rows(monkeypatch):
    """``upsample_batch_to_divisible_size`` deep-copies its template's fields *and* tag.

    A synthetic row therefore still carries ``child_gen_keys``, ``extra_fields`` and J/K, so
    counting it as an episode would gather the template's GEN rows twice and fold its J/K into the
    batch mean a second time.
    """
    import transfer_queue as tq_module
    from transfer_queue import KVBatchMeta

    column = TensorDict({}, batch_size=[2])
    column["extra_fields"] = [
        {"child_gen_keys": ["k0"], "episode_J": 4},
        {"child_gen_keys": ["k0"], "episode_J": 4},  # padding copy of row 0
    ]
    monkeypatch.setattr(tq_module, "kv_batch_get", lambda keys, partition_id, select_fields=None: column)

    batch = KVBatchMeta(
        partition_id="train",
        keys=["u_0_0", "pad9_0_0"],
        tags=[{"bagel_role": "und"}, {"bagel_role": "und", "is_padding": True}],
    )
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    records = trainer._und_records_from_batch(batch)

    assert len(records) == 1
    assert records[0]["fields"]["episode_J"] == 4
    assert records[0]["fields"]["child_gen_keys"] == ["k0"]


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


def _und_ar_trainer(tmp_path, und_n_gpus: int):
    """Minimal trainer carrying just what ``_build_und_ar_entrypoint_config`` reads."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bagel"}))
    (tmp_path / "llm_config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 64}))
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 1024, "max_response_length": 1024},
            "actor_rollout_ref": {
                "model": {"path": str(tmp_path), "tokenizer_path": str(tmp_path), "lora_rank": 8},
                "rollout": {
                    "n": 2,
                    "engine_kwargs": {"vllm_omni": {"output_mode": "ar", "deploy_config": "gen.yaml"}},
                    "agent": {
                        "und_ar_serving_ready": True,
                        "und_deploy_config": "examples/agenticllmgrpo_trainer/bagel/bagel_corl_deploy_ar.yaml",
                        "und_n_gpus": und_n_gpus,
                        "und_gpu_memory_utilization": 0.40,
                    },
                },
            },
        }
    )
    OmegaConf.set_struct(trainer.config, False)
    return trainer


def test_und_ar_entrypoint_declares_replica_width_as_tp(tmp_path):
    """The AR clone must tell vLLM-Omni how wide the replica is, and pin the stages back.

    ``RolloutReplica`` builds the server's ``CUDA_VISIBLE_DEVICES`` from
    ``gpus_per_replica_node = min(n_gpus_per_node, world_size)`` (``world_size = TP*DP*PP``)
    and ``vLLMReplica.launch_servers`` asserts the worker count equals ``world_size``.
    A declared width of 1 against a 2-wide actor-pool slice is what put both ``bagel_think``
    stages on GPU 0 and later OOM'd the GEN wake-up. The width must ride on TP: the DP path
    also emits ``data_parallel_size_local``, which is not an ``OrchestratorArgs``/deploy
    field, so it is not filtered and would reach every stage against a DP=1 pin.
    """
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=2)
    rollout = trainer._build_und_ar_entrypoint_config().actor_rollout_ref.rollout

    assert int(rollout.tensor_model_parallel_size) == 2
    assert int(rollout.data_parallel_size) == 1
    assert int(rollout.pipeline_model_parallel_size) == 1
    assert int(rollout.n_gpus_per_node) == 2
    # world_size must equal the actor-pool slice the replica is handed.
    assert (
        int(rollout.tensor_model_parallel_size)
        * int(rollout.data_parallel_size)
        * int(rollout.pipeline_model_parallel_size)
    ) == 2

    # Every stage of the deploy yaml is pinned back to one rank, or a stage holding a single
    # ``devices`` entry is asked for two. Derived from the yaml, so it tracks stage changes.
    stage_overrides = rollout.engine_kwargs.vllm_omni.stage_overrides
    assert {int(stage_id): dict(cfg) for stage_id, cfg in stage_overrides.items()} == {
        0: {"tensor_parallel_size": 1},
        1: {"tensor_parallel_size": 1},
    }


def test_und_ar_engine_keeps_lora_off_when_the_actor_merges(tmp_path):
    """A merged-LoRA actor must not launch the AR engine with ``enable_lora``.

    ``vllm_async_server`` turns a nonzero ``model_config.lora_rank`` into
    ``enable_lora=True, max_loras=1``: it reads ``model.lora.rank``, falls back to the
    top-level ``lora_rank`` ("FIXME: fallback to lora_rank") and only zeroes that fallback
    when ``model.lora.merge`` is set. This builder copies ``lora_rank`` but not ``merge``,
    so a merged run (``actor_rollout_ref.model.lora.merge=True`` => the actor publishes full
    merged weights and never an adapter) still handed the AR engine ``enable_lora=True``.

    Measured 2026-09-20 19:44:45 on hk01dgx039 (devices 3,5,6,7): the first UND decode of
    step 0 never returned -- the AR worker's main thread sat in
    ``vllm/lora/punica_wrapper/punica_gpu.py:add_lora_linear`` with no adapter loaded -- and
    the whole step plus its validation produced no TQ rows.
    """
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=2)
    # Adapter mode (no merge) is unchanged: the AR engine legitimately gets a rank.
    assert int(trainer._build_und_ar_entrypoint_config().actor_rollout_ref.model.lora_rank) == 8

    trainer.config.actor_rollout_ref.model.lora = {"merge": True}
    model = trainer._build_und_ar_entrypoint_config().actor_rollout_ref.model
    assert int(model.lora_rank) == 0, "merged actor => the AR engine must not enable LoRA"


def test_actor_merges_lora_reads_both_config_shapes():
    """The helper accepts the resolved DictConfig and the converted dataclass."""
    assert _actor_merges_lora(OmegaConf.create({"lora": {"merge": True}}))
    assert not _actor_merges_lora(OmegaConf.create({"lora": {"merge": False}}))
    assert not _actor_merges_lora(OmegaConf.create({"lora_rank": 8}))
    assert _actor_merges_lora(types.SimpleNamespace(lora=types.SimpleNamespace(merge=True)))
    assert not _actor_merges_lora(types.SimpleNamespace(lora={"merge": False}))
    assert not _actor_merges_lora(types.SimpleNamespace(lora_rank=8))


def test_und_ar_width_must_match_the_actor_pool_slice(tmp_path):
    """The guard rejects a declared width that disagrees with ``und_n_gpus``.

    This is the mismatch that otherwise surfaces as ``assert len(self.workers) ==
    world_size`` inside ``vLLMReplica.launch_servers``.
    """
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=2)
    rollout = trainer._build_und_ar_entrypoint_config().actor_rollout_ref.rollout
    deploy = str(_AR_DEPLOY)

    # Matching width + a pool wide enough for the stages: accepted.
    OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 2, deploy)

    with pytest.raises(ValueError, match="declares world_size=2"):
        OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 1, deploy)


def test_und_ar_pool_narrower_than_stage_devices_is_rejected(tmp_path):
    """A 1-wide pool cannot resolve ``devices: "1"``.

    The 05:06 run used ``und_n_gpus=1`` against the two-device deploy yaml and failed as
    ``StageEngineCoreProc_stage0_replica0 ... ValueError: No available memory for the cache
    blocks`` -> ``Orchestrator initialization failed: ... Failed core proc(s): {}``.
    """
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=1)
    rollout = trainer._build_und_ar_entrypoint_config().actor_rollout_ref.rollout
    # A 1-wide declaration matches the pool, so this passes the first check and only the
    # stage-device invariant can reject it.
    with pytest.raises(ValueError, match="cannot cover the stage devices"):
        OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 1, str(_AR_DEPLOY))

    # The 1-GPU smoke-test layout (every stage on logical 0) is accepted at width 1.
    smoke = OmegaConf.load(_AR_DEPLOY)
    for stage in smoke.stages:
        stage.devices = "0"
    smoke_path = tmp_path / "smoke.yaml"
    smoke_path.write_text(OmegaConf.to_yaml(smoke))
    OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 1, str(smoke_path))


def test_und_ar_width_must_divide_the_actor_pool(tmp_path):
    """``und_n_gpus`` has to divide the actor pool, because that is how it is sliced.

    ``_ensure_dual_role_rollout`` carves the replica out with
    ``split_resource_pool(actor_pool, split_size=und_n_gpus)``, whose first act is
    ``assert resource_pool.world_size % split_size == 0``
    (``verl/single_controller/ray/base.py:289``). A width that does not divide therefore
    kills the launch *after* the engines have loaded -- measured 2026-09-18 10:36 on
    ``hk01dgx012`` (devices 4-7), where ``UND_N_GPUS=3`` against a 4-card pool died ~8
    minutes in as ``AssertionError: split_size must be a divisor of world_size``. On a
    4-card pool the valid widths are {1, 2, 4}.
    """
    # 3 divides 4? No -- and the message has to name the pool width, not the assert.
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=3)
    rollout = trainer._build_und_ar_entrypoint_config().actor_rollout_ref.rollout
    with pytest.raises(ValueError, match="does not divide the actor pool"):
        OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 3, str(_AR_DEPLOY), actor_world_size=4)

    # The divisors of a 4-card pool that the stage layout also allows stay accepted.
    # (1 is a divisor but cannot resolve the DiT stage's logical device 1, so the
    # stage-device invariant still owns that case -- see the sibling test.)
    for width in (2, 4):
        matching = _und_ar_trainer(tmp_path, und_n_gpus=width)
        matching_rollout = matching._build_und_ar_entrypoint_config().actor_rollout_ref.rollout
        OmniBagelCoRLTrainerSync._validate_und_ar_pool(matching_rollout, width, str(_AR_DEPLOY), actor_world_size=4)

    # A 3-card pool's own divisors are fine, so this is not "3 is always wrong".
    OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 3, str(_AR_DEPLOY), actor_world_size=3)

    # ``None`` (the pool could not be resolved) skips only this check, not the others:
    # a genuine width mismatch still raises.
    OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 3, str(_AR_DEPLOY), actor_world_size=None)
    with pytest.raises(ValueError, match="declares world_size=3"):
        OmniBagelCoRLTrainerSync._validate_und_ar_pool(rollout, 2, str(_AR_DEPLOY), actor_world_size=None)


def test_und_ar_context_comes_from_the_deploy_config_not_the_episode_length(tmp_path):
    """The AR engine's context must be the deploy yaml's budget, not ``max_prompt + max_resp``.

    ``_und_decode`` hands the engine ``prompt_ids + response_ids`` -- the *whole episode so
    far* -- so a context of exactly ``max_prompt + max_resp`` (1024+1024) leaves zero
    headroom for the turn being generated once the episode's response budget is spent, and
    the AR strategy refuses it outright:

        ValueError: Prompt length (2048) meets or exceeds the model's maximum context
        length (2048), leaving no space for generation.

    Measured 2026-09-17 09:30. It killed every UND tool-call decode, so the step ended with
    no materializable trajectories. The builder used to set that sum while the deploy config
    it ships declares 16384 -- contradicting both the yaml and the RFC.
    """
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=2)
    rollout = trainer._build_und_ar_entrypoint_config().actor_rollout_ref.rollout
    declared = OmniBagelCoRLTrainerSync._und_stage_max_model_len(str(_AR_DEPLOY))
    assert declared == 16384
    assert int(rollout.max_model_len) == declared
    # Strictly more than the episode budget, or no turn could ever emit a token.
    assert int(rollout.max_model_len) > int(rollout.prompt_length) + int(rollout.response_length)


def test_und_ar_context_smaller_than_the_episode_budget_is_rejected(tmp_path):
    """A too-small AR context is the measured failure; fail loud instead of at decode time."""
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=2)
    trainer.config.actor_rollout_ref.rollout.agent.und_max_model_len = 1024
    with pytest.raises(ValueError, match="smaller than the episode context"):
        trainer._build_und_ar_entrypoint_config()


def test_und_ar_context_honours_the_explicit_knob(tmp_path):
    trainer = _und_ar_trainer(tmp_path, und_n_gpus=2)
    trainer.config.actor_rollout_ref.rollout.agent.und_max_model_len = 4096
    rollout = trainer._build_und_ar_entrypoint_config().actor_rollout_ref.rollout
    assert int(rollout.max_model_len) == 4096


def test_get_reward_handles_is_none_when_the_pool_is_empty():
    """An empty handle list must never reach ``random.choice``.

    ``RewardLoopManager.reward_loop_worker_handles`` returns the (empty) worker list when
    ``reward_model.enable=False``, which the Co-RL recipe sets alongside
    ``reward.num_workers=0`` (the in-loop Bagel reward is the only scorer). The agent loop
    guards with ``is not None`` and then picks a worker at random, so any episode that ended
    with ``reward_score is None`` died with

        IndexError: Cannot choose from an empty sequence

    -- masking the real error. Measured 2026-09-17 09:30, 29 ms behind the AR context
    ValueError in the same worker.
    """
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.reward_loop_manager = types.SimpleNamespace(reward_loop_workers=[], reward_loop_worker_handles=[])
    assert trainer.get_reward_handles() is None

    handles = [object(), object()]
    trainer.reward_loop_manager = types.SimpleNamespace(reward_loop_workers=handles)
    assert trainer.get_reward_handles() == handles


def _trainer_with_stubbed_super_update_actor(monkeypatch):
    """Bare trainer whose inherited ``_update_actor`` just records the batch."""
    monkeypatch.setattr(OmniPPOTrainerSync, "_update_actor", lambda self, batch, metrics: batch)
    return OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)


def test_update_actor_flags_ride_the_kv_meta_extra_info(monkeypatch):
    """``_update_actor`` must not push non-tensor data onto a ``KVBatchMeta``.

    v1 hands the trainer a ``KVBatchMeta``. Its ``extra_info`` dict *is* the channel: the TQ
    dispatch layer copies it onto the actor's TensorDict as non-tensor data
    (``verl/utils/transferqueue_utils.py:160-177``), which is where the composite reads
    ``skip_gen``/``bagel_corl_gen`` back (``bagel_corl_composite.py:129,131``). Guarding the
    assignment with ``hasattr(batch, "keys")`` was not a TensorDict test -- ``KVBatchMeta.keys``
    is its list of TQ keys -- so the call asserted

        AssertionError: input dict must be a TensorDict

    at ``tensordict_utils.py:44``. Measured 2026-09-18 on `hk01dgx012` (devices 4-7), the first
    step that got past the reward and reached the actor.
    """
    from transfer_queue import KVBatchMeta

    trainer = _trainer_with_stubbed_super_update_actor(monkeypatch)
    gen_view = TensorDict({"rm_scores": torch.ones(2, 1)}, batch_size=[2])
    batch = KVBatchMeta(
        keys=["u0_0", "u0_1"],
        tags=[{"bagel_role": "und"}, {"bagel_role": "und"}],
        partition_id="p0",
        extra_info={
            "has_complete_gen_groups": True,
            "num_gen_rows": 2,
            "bagel_corl_gen": gen_view,
            "episode/J": 3.0,
        },
    )
    metrics: dict = {}

    assert trainer._update_actor(batch, metrics) is batch
    # The flags the actor reads must be on the channel the TQ layer forwards.
    assert batch.extra_info["skip_gen"] is False
    assert batch.extra_info["has_complete_gen_groups"] is True
    assert batch.extra_info["num_gen_rows"] == 2
    assert isinstance(batch.extra_info["bagel_corl_gen"], TensorDict)
    assert metrics["episode/J"] == pytest.approx(3.0)


def test_update_actor_flags_still_reach_a_plain_tensordict_carrier(monkeypatch):
    """The defensive branch stays honest: a real TensorDict still gets the non-tensor flags."""
    trainer = _trainer_with_stubbed_super_update_actor(monkeypatch)
    batch = TensorDict({"rm_scores": torch.zeros(2, 1)}, batch_size=[2])
    metrics: dict = {}

    trainer._update_actor(batch, metrics)

    assert tu.get_non_tensor_data(batch, "skip_gen", default=None) is True
    assert tu.get_non_tensor_data(batch, "has_complete_gen_groups", default=None) is False
    assert tu.get_non_tensor_data(batch, "num_gen_rows", default=None) == 0
    assert tu.get_non_tensor_data(batch, "num_gen_rows", default=None) == 0
    assert metrics["gen/skipped_no_groups"] == pytest.approx(1.0)


def test_update_actor_a_meta_without_gen_groups_is_skipped_not_scored(monkeypatch):
    """No complete S-group ⇒ the flags must say "skip GEN", never leave the actor guessing."""
    from transfer_queue import KVBatchMeta

    trainer = _trainer_with_stubbed_super_update_actor(monkeypatch)
    batch = KVBatchMeta(
        keys=["u0_0"],
        tags=[{"bagel_role": "und"}],
        partition_id="p0",
        extra_info={"has_complete_gen_groups": False},
    )
    metrics: dict = {}

    trainer._update_actor(batch, metrics)

    assert batch.extra_info["skip_gen"] is True
    assert metrics["gen/skipped_no_groups"] == pytest.approx(1.0)


# --- UND pass must hand the micro-batch splitter its batch size -------------------
#
# ``run_und_token_forward_backward`` forces ``use_dynamic_bsz=False``, and on that
# branch ``prepare_micro_batches`` reads ``micro_batch_size_per_gpu`` straight off the
# batch (verl/verl/workers/engine/utils.py:89). The worker injects that key onto the
# *parent* batch -- train from ``engine_config.micro_batch_size_per_gpu``, infer from
# ``infer_micro_batch_size_per_gpu`` (verl_omni/workers/engine_workers.py:412/473) --
# but the composite builds its batch with ``data.select(*_UND_SELECT)``, which drops
# non-tensor metadata. The first ``compute_log_prob`` after sampling therefore died
# with, on hk01dgx012 (devices 4-7), right after ``Training Progress: 0%``:
#
#   KeyError: 'key "micro_batch_size_per_gpu" not found in TensorDict with keys
#   ['input_ids', 'position_ids', 'prompts', 'response_mask', 'responses', 'sp_size',
#    'use_dynamic_bsz', ...]'
#
# These tests spy on ``prepare_micro_batches`` to assert what the UND pass hands it.

_UND_INFER_MICRO_BSZ = 5
_UND_TRAIN_MICRO_BSZ = 7


class _UndModelStub:
    """Stand-in for the Bagel module reached through ``compute_und_log_prob``."""

    def __init__(self, grid: torch.Tensor | None = None):
        # ``grid`` lets a test pin a distinguishable ``(B, L-1)`` log-prob grid; the default zeros
        # are enough for the plumbing tests that only care about shapes and attributes.
        self.grid = grid

    def compute_und_log_prob(self, input_ids, attention_mask, response_mask, *, with_entropy=False):
        if self.grid is not None:
            log_probs = self.grid
        else:
            log_probs = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1)
        if with_entropy:
            return log_probs, log_probs.clone()
        return log_probs


def _stub_bagel_engine(engine_config: object | None = None) -> object:
    """Minimal stand-in for the FSDP engine the UND pass touches."""
    engine = types.SimpleNamespace()
    engine.ulysses_sequence_parallel_size = 1
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1
    engine.module = _UndModelStub()
    engine.engine_config = engine_config
    engine.postprocess_batch_func = lambda output_lst, indices, data: {"output_lst": output_lst, "indices": indices}
    return engine


def _und_batch(*, tensor_keys: tuple[str, ...] = ()) -> TensorDict:
    """A batch shaped like the UND pass's input, non-tensor keys added by the caller."""
    keys = {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "response_mask": torch.ones(2, 2, dtype=torch.long),
    }
    for key in tensor_keys:
        keys[key] = torch.zeros(2, 4) if key != "old_log_probs" else torch.zeros(2, 2)
    return TensorDict(keys, batch_size=[2])


def _spy_on_und_split(monkeypatch: pytest.MonkeyPatch):
    """Swap ``prepare_micro_batches`` for a spy; return ``(module, seen)``.

    The spy hands back one real micro-batch instead of an empty list so the tests exercise the
    whole UND loop, including the postprocess that publishes log-probs. ``get_device_id`` is pinned
    to CPU so the suite stays device-free.

    ``prompts``/``responses`` go in as jagged rows, which is how the TQ fetch delivers them (the
    same shape ``response_from_nested`` consumes, ``verl/verl/workers/utils/padding.py:196``): the
    two rows below both span 4 tokens but split 3+1 and 2+2, which is the layout that tells the
    nested branch of ``no_padding_2_padding`` apart from the padded one.
    """
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    seen: dict = {}

    def _spy(data, **kwargs):
        seen["micro_batch_size_per_gpu"] = tu.get_non_tensor_data(
            data=data, key="micro_batch_size_per_gpu", default=None
        )
        keys = {
            key: data[key]
            for key in ("input_ids", "attention_mask", "response_mask", "old_log_probs", "advantages")
            if key in data.keys()
        }
        keys["prompts"] = _jagged_rows([[1, 2, 3], [4, 5]])
        keys["responses"] = _jagged_rows([[6], [7, 8]])
        return [TensorDict(keys, batch_size=[2])], None

    monkeypatch.setattr(composite, "prepare_micro_batches", _spy)
    monkeypatch.setattr(composite, "unwrap_bagel_module", lambda module: module)
    monkeypatch.setattr(composite, "get_device_id", lambda: "cpu")
    return composite, seen


def _jagged_rows(rows: list[list[int]]) -> torch.Tensor:
    """A jagged tensor in the shape the TQ fetch hands the UND pass."""
    return torch.nested.as_nested_tensor([torch.tensor(row, dtype=torch.long) for row in rows], layout=torch.jagged)


def test_und_pass_reattaches_the_micro_batch_size_the_splitter_needs(monkeypatch):
    """A batch size carried on the parent batch must survive the ``select``."""
    composite, seen = _spy_on_und_split(monkeypatch)
    data = _und_batch()
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=3)

    composite.run_und_token_forward_backward(_stub_bagel_engine(), data, None, forward_only=True)

    assert seen["micro_batch_size_per_gpu"] == 3


def test_und_pass_infer_falls_back_to_the_infer_micro_batch_size(monkeypatch):
    """Without a key on the batch, the infer pass reads the engine's infer knob."""
    composite, seen = _spy_on_und_split(monkeypatch)
    engine = _stub_bagel_engine(
        types.SimpleNamespace(
            infer_micro_batch_size_per_gpu=_UND_INFER_MICRO_BSZ,
            micro_batch_size_per_gpu=_UND_TRAIN_MICRO_BSZ,
        )
    )

    composite.run_und_token_forward_backward(engine, _und_batch(), None, forward_only=True)

    assert seen["micro_batch_size_per_gpu"] == _UND_INFER_MICRO_BSZ


def _und_train_loss(**kwargs):
    """Stand-in loss so the train path's ``loss.backward()`` has a graph to walk."""
    return torch.zeros((), requires_grad=True), {}


def test_und_pass_train_falls_back_to_the_train_micro_batch_size(monkeypatch):
    """The train pass must read the train knob, not the infer one."""
    composite, seen = _spy_on_und_split(monkeypatch)
    engine = _stub_bagel_engine(
        types.SimpleNamespace(
            infer_micro_batch_size_per_gpu=_UND_INFER_MICRO_BSZ,
            micro_batch_size_per_gpu=_UND_TRAIN_MICRO_BSZ,
        )
    )
    data = _und_batch(tensor_keys=("old_log_probs", "advantages"))

    composite.run_und_token_forward_backward(engine, data, _und_train_loss, forward_only=False)

    assert seen["micro_batch_size_per_gpu"] == _UND_TRAIN_MICRO_BSZ


def test_und_pass_fails_loud_when_no_micro_batch_size_is_declared(monkeypatch):
    """No batch key and no engine config must fail here, not deep inside tensordict."""
    composite, _ = _spy_on_und_split(monkeypatch)

    with pytest.raises(KeyError, match="micro_batch_size_per_gpu"):
        composite.run_und_token_forward_backward(_stub_bagel_engine(), _und_batch(), None, forward_only=True)


# --- GEN pass must hand the same splitter the same batch size ---------------------
#
# The sibling defect to the UND one above, and it is worth keeping the two tests adjacent
# because the fix is identical: ``DiffusersFSDPEngine._run_forward_backward_batch`` force-sets
# ``use_dynamic_bsz=False`` (verl_omni/workers/engine/fsdp/diffusers_impl.py) and then calls
# ``prepare_micro_batches``, which reads ``micro_batch_size_per_gpu`` off the batch
# (verl/verl/workers/engine/utils.py:89). The worker injects the key onto the *parent* batch
# (verl_omni/workers/engine_workers.py:413 train / :473 infer), but the Bagel Co-RL GEN sub-batch
# is materialized from the trajectory view by ``materialize_gen_train_batch``, so the key never
# propagates. It stayed invisible while ``ENABLE_RM=0`` because ``skip_gen=True`` retired the GEN
# loss before the splitter ran; the first RM-on step to reach it died with, on hk01dgx039
# (devices 0/1/6/7, ``bagel_corl_rm1_20260923_154117``):
#
#   KeyError: 'key "micro_batch_size_per_gpu" not found in TensorDict with keys
#   ['advantages', 'all_latents', 'all_timesteps', 'bagel_corl_gen', 'has_complete_gen_groups',
#    'num_gen_rows', 'old_log_probs', 'returns', 'rm_scores', 'sample_level_rewards',
#    'sample_level_scores', 'skip_gen', 'sp_size', 'use_dynamic_bsz']'
#
# These tests call the GEN entry point directly with ``prepare_micro_batches`` spied out. The spy
# returns no micro-batches, so ``forward_step``/``postprocess_batch_func`` never run and the test
# stays a pure plumbing check on what the splitter was handed.

_GEN_INFER_MICRO_BSZ = 5
_GEN_TRAIN_MICRO_BSZ = 7


def _diffusers_engine_stub(engine_config: object | None = None) -> object:
    """Minimal stand-in for ``DiffusersFSDPEngine`` around ``_run_forward_backward_batch``."""
    from verl_omni.workers.engine.fsdp import diffusers_impl

    engine = types.SimpleNamespace()
    engine.ulysses_sequence_parallel_size = 1
    engine.get_data_parallel_group = lambda: None
    engine.engine_config = engine_config
    # The GEN pass re-slices row-aligned non-tensors for the micro-batches; a stub has to carry the
    # real implementation or the call raises before the splitter is ever reached.
    engine._reslice_per_row_non_tensors = diffusers_impl.DiffusersFSDPEngine._reslice_per_row_non_tensors
    engine.postprocess_batch_func = lambda output_lst, indices, data: {
        "output_lst": output_lst,
        "indices": indices,
    }
    return engine


def _gen_batch() -> TensorDict:
    """A batch shaped like the materialized GEN sub-batch (latents over a timestep axis)."""
    return TensorDict(
        {
            "all_latents": torch.zeros(2, 4, 3, 8, 8),
            "all_timesteps": torch.zeros(2, 4),
            "old_log_probs": torch.zeros(2, 4),
            "advantages": torch.zeros(2, 4),
            "rm_scores": torch.zeros(2),
        },
        batch_size=[2],
    )


def _spy_on_gen_split(monkeypatch: pytest.MonkeyPatch):
    """Swap the GEN path's ``prepare_micro_batches`` for a spy; return ``(engine_mod, seen)``."""
    from verl_omni.workers.engine.fsdp import diffusers_impl

    seen: dict = {}

    def _spy(data, **kwargs):
        seen["micro_batch_size_per_gpu"] = tu.get_non_tensor_data(
            data=data, key="micro_batch_size_per_gpu", default=None
        )
        seen["use_dynamic_bsz"] = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=None)
        seen["sp_size"] = tu.get_non_tensor_data(data=data, key="sp_size", default=None)
        # No micro-batches: the timestep loop and the postprocess never run, so this test only
        # observes what the splitter was handed.
        return [], None

    monkeypatch.setattr(diffusers_impl, "prepare_micro_batches", _spy)
    return diffusers_impl, seen


def test_gen_pass_reattaches_the_micro_batch_size_the_splitter_needs(monkeypatch):
    """A batch size carried on the GEN sub-batch must reach the splitter."""
    diffusers_impl, seen = _spy_on_gen_split(monkeypatch)
    data = _gen_batch()
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=3)

    diffusers_impl.DiffusersFSDPEngine._run_forward_backward_batch(
        _diffusers_engine_stub(), data, None, False, timesteps_key="all_timesteps"
    )

    assert seen["micro_batch_size_per_gpu"] == 3


def test_gen_pass_train_falls_back_to_the_train_micro_batch_size(monkeypatch):
    """The materialized GEN train batch carries no key, so the engine's train knob must be used."""
    diffusers_impl, seen = _spy_on_gen_split(monkeypatch)
    engine = _diffusers_engine_stub(
        types.SimpleNamespace(
            infer_micro_batch_size_per_gpu=_GEN_INFER_MICRO_BSZ,
            micro_batch_size_per_gpu=_GEN_TRAIN_MICRO_BSZ,
        )
    )

    diffusers_impl.DiffusersFSDPEngine._run_forward_backward_batch(
        engine, _gen_batch(), None, False, timesteps_key="all_timesteps"
    )

    assert seen["micro_batch_size_per_gpu"] == _GEN_TRAIN_MICRO_BSZ
    # The forcing that makes the key mandatory in the first place must still be in place.
    assert seen["use_dynamic_bsz"] is False
    assert seen["sp_size"] == 1


def test_gen_pass_infer_falls_back_to_the_infer_micro_batch_size(monkeypatch):
    """A GEN infer pass must read the infer knob, not the train one."""
    diffusers_impl, seen = _spy_on_gen_split(monkeypatch)
    engine = _diffusers_engine_stub(
        types.SimpleNamespace(
            infer_micro_batch_size_per_gpu=_GEN_INFER_MICRO_BSZ,
            micro_batch_size_per_gpu=_GEN_TRAIN_MICRO_BSZ,
        )
    )

    diffusers_impl.DiffusersFSDPEngine._run_forward_backward_batch(
        engine, _gen_batch(), None, True, timesteps_key="all_timesteps"
    )

    assert seen["micro_batch_size_per_gpu"] == _GEN_INFER_MICRO_BSZ


def test_gen_pass_fails_loud_when_no_micro_batch_size_is_declared(monkeypatch):
    """No batch key and no engine config must fail here, not deep inside tensordict."""
    diffusers_impl, _ = _spy_on_gen_split(monkeypatch)

    with pytest.raises(KeyError, match="micro_batch_size_per_gpu"):
        diffusers_impl.DiffusersFSDPEngine._run_forward_backward_batch(
            _diffusers_engine_stub(), _gen_batch(), None, False, timesteps_key="all_timesteps"
        )


# --- FSDP2 must treat the UND entry point as a forward method ---------------------
#
# FSDP2 only all-gathers parameters and converts activations for ``nn.Module.forward``
# and for methods registered via ``register_fsdp_forward_method``. ``unwrap_bagel_module``
# reaches the model through a custom method, so without registration the call runs against
# sharded ``DTensor`` parameters with plain activations and dies on the first elementwise
# op, in the very same idiom that ``BagelForTraining.forward`` uses successfully:
#
#   RuntimeError: aten.mul.Tensor got mixed torch.Tensor and DTensor, need to convert all
#   torch.Tensor to DTensor before calling distributed operators!
#   ...bagel_corl.py:249: hidden[text_idx] = self.norm(sequence[text_idx])
#   ...bagel_model.py:153: return self.weight * x.to(input_dtype)
#
# Measured 2026-09-18 on hk01dgx012 (devices 4-7).


class _FakeFSDPModule:
    """Stand-in so ``isinstance(module, FSDPModule)`` passes without a real sharded module."""

    def compute_und_log_prob(self, *args, **kwargs):  # pragma: no cover - never executed
        raise AssertionError("the UND pass must not execute the stub module")


def _patch_fsdp_registration(monkeypatch: pytest.MonkeyPatch, fsdp_module_cls: type) -> list:
    """Swap ``register_fsdp_forward_method`` for a recorder; return the call list."""
    from torch.distributed import fsdp as fsdp_pkg

    calls: list = []
    monkeypatch.setattr(fsdp_pkg, "FSDPModule", fsdp_module_cls)
    monkeypatch.setattr(fsdp_pkg, "register_fsdp_forward_method", lambda module, name: calls.append((module, name)))
    return calls


def test_und_entry_point_is_registered_as_an_fsdp_forward_method(monkeypatch):
    """A managed module must get ``compute_und_log_prob`` registered."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    calls = _patch_fsdp_registration(monkeypatch, _FakeFSDPModule)
    module = _FakeFSDPModule()

    composite.register_und_forward_method(types.SimpleNamespace(module=module), module)

    assert calls == [(module, "compute_und_log_prob")]


def test_und_forward_registration_is_idempotent(monkeypatch):
    """Registering on every micro-batch must not re-register."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    calls = _patch_fsdp_registration(monkeypatch, _FakeFSDPModule)
    module = _FakeFSDPModule()
    engine = types.SimpleNamespace(module=module)

    composite.register_und_forward_method(engine, module)
    composite.register_und_forward_method(engine, module)

    assert calls == [(module, "compute_und_log_prob")]


def test_und_forward_registration_skips_unmanaged_modules(monkeypatch):
    """An unsharded module (plain FSDP off) must be left alone, not raise."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    calls = _patch_fsdp_registration(monkeypatch, _FakeFSDPModule)
    plain = types.SimpleNamespace()

    composite.register_und_forward_method(types.SimpleNamespace(module=plain), plain)

    assert calls == []


def test_und_pass_registers_the_entry_point_before_scoring(monkeypatch):
    """The registration must happen on the real ``run_und_token_forward_backward`` path."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    calls = _patch_fsdp_registration(monkeypatch, _FakeFSDPModule)
    monkeypatch.setattr(composite, "prepare_micro_batches", lambda data, **kwargs: ([], None))

    module = _FakeFSDPModule()
    engine = _stub_bagel_engine()
    engine.module = module
    data = _und_batch(tensor_keys=("old_log_probs", "advantages"))
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=1)

    # Train mode: registration has to happen before the first ``compute_und_log_prob`` on either
    # lane, and the fake module must still never be executed.
    composite.run_und_token_forward_backward(engine, data, None, forward_only=False)

    assert calls == [(module, "compute_und_log_prob")]


# --- the UND lane must not ride the GEN postprocessor ------------------------------
#
# ``OmniFSDPEngine.postprocess_batch_func`` (diffusers_impl.py:572) is the GEN contract: it walks a
# list of per-*timestep* flat dicts and stacks every key into ``(bsz, steps, ...)``. The UND payload
# is a single flat blob whose ``"und"`` value is itself a dict, so the first ``compute_log_prob``
# after sampling died inside that stack:
#
#   TypeError: expected Tensor as element 0 in argument 0, but got dict
#
# and the trainer reads the result back out of the TransferQueue with ``response_from_nested``,
# which only accepts a *jagged* tensor over ``prompt + response``.
# Measured 2026-09-18 on hk01dgx012 (devices 4-7).


class _PatternedUndModel:
    """``compute_und_log_prob`` returning ``10 * row + column`` so placement is unambiguous."""

    def compute_und_log_prob(self, input_ids, attention_mask, response_mask, *, with_entropy=False):
        rows = torch.arange(input_ids.shape[0], dtype=torch.float32).unsqueeze(-1)
        cols = torch.arange(input_ids.shape[1] - 1, dtype=torch.float32)
        grid = rows * 10 + cols
        if with_entropy:
            return grid, grid + _ENTROPY_OFFSET
        return grid


_ENTROPY_OFFSET = 1000


_ENTROPY_OFFSET = 1000


def _patterned_engine() -> object:
    engine = _stub_bagel_engine()
    engine.module = _PatternedUndModel()
    return engine


def test_the_gen_step_stacker_is_what_rejects_the_und_blob():
    """Mutation guard: keep proving the GEN-shaped stack cannot consume a UND blob."""
    blob = {"und": {"log_probs": torch.zeros(2, 2)}, "modality": "und", "log_probs": torch.zeros(2, 2)}

    with pytest.raises(TypeError, match="expected Tensor as element 0"):
        stacked: dict = {}
        for key, val in blob.items():
            stacked.setdefault(key, []).append(val)
        for _key, val in stacked.items():
            torch.stack(val, dim=1)


def test_und_infer_publishes_nested_log_probs_and_entropy(monkeypatch):
    """``compute_log_prob`` must hand back the shape ``_postprocess_output`` expects."""
    composite, _ = _spy_on_und_split(monkeypatch)
    data = _und_batch()
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=2)

    out = composite.run_und_token_forward_backward(_patterned_engine(), data, None, forward_only=True)

    # ``_postprocess_output`` pops these three (engine_workers.py:243,249,298); a bare TensorDict
    # of log_probs/entropy got as far as ``output.pop("metrics")`` and died with KeyError('metrics').
    assert set(out) >= {"model_output", "loss", "metrics"}
    log_probs = out["model_output"]["log_probs"]
    entropy = out["model_output"]["entropy"]
    for tensor in (log_probs, entropy):
        assert tensor.is_nested
    # L=4 with a 2-token response => each row is prompt(2) + response(2) wide, not response-only.
    assert log_probs.offsets().tolist() == [0, 4, 8]
    # The trailing unused slot is 0 on both grids, so compare the offset on the real entries.
    lp = log_probs.values().tolist()
    ent = entropy.values().tolist()
    assert [e - v for v, e in zip(lp, ent, strict=True)] == [1000.0, 1000.0, 1000.0, 0.0] * 2


def test_und_infer_rows_land_on_the_response_after_the_trainers_slice(monkeypatch):
    """The decisive alignment check: ``response_from_nested`` must recover the response tokens.

    Every row is ``[lp(1), lp(2), lp(3), unused]``; the trainer keeps ``values[L - R - 1 : L - 1]``,
    i.e. the last response tokens. Padding the row on the *left* instead would shift the window and
    silently score the last prompt token while dropping the last response token.
    """
    from verl.workers.utils.padding import response_from_nested

    composite, _ = _spy_on_und_split(monkeypatch)
    data = _und_batch()
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=2)

    out = composite.run_und_token_forward_backward(_patterned_engine(), data, None, forward_only=True)

    log_probs = out["model_output"]["log_probs"]
    assert log_probs.values().tolist() == [0.0, 1.0, 2.0, 0.0, 10.0, 11.0, 12.0, 0.0]

    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long)], layout=torch.jagged
    )
    aligned = response_from_nested(log_probs, response_mask)

    # Rows are prompt(2) + response(2): the response is the last two tokens, i.e. lp(2) and lp(3).
    assert aligned.values().tolist() == [1.0, 2.0, 11.0, 12.0]
    assert aligned.offsets().tolist() == [0, 2, 4]


def test_und_train_drops_model_output_like_the_transformer_engine(monkeypatch):
    """Train mode must not carry the nested blob into postprocessing; only loss/metrics survive."""
    composite, _ = _spy_on_und_split(monkeypatch)
    data = _und_batch(tensor_keys=("old_log_probs", "advantages"))
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=2)

    out = composite.run_und_token_forward_backward(_stub_bagel_engine(), data, _und_train_loss, forward_only=False)

    assert isinstance(out, dict)
    assert out["model_output"] == {}
    assert len(out["loss"]) == 1


def test_und_infer_without_a_single_scored_row_fails_loud(monkeypatch):
    """An empty publish must not look like a successful ``compute_log_prob``."""
    composite, _ = _spy_on_und_split(monkeypatch)
    monkeypatch.setattr(composite, "prepare_micro_batches", lambda data, **kwargs: ([], None))
    data = _und_batch()
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=2)

    with pytest.raises(RuntimeError, match="no per-row log-probs"):
        composite.run_und_token_forward_backward(_stub_bagel_engine(), data, None, forward_only=True)


def test_und_postprocess_refuses_dynamic_bsz_reordering():
    """Reordering micro-batches would scramble the published rows; refuse instead of guessing."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    data = _und_batch()
    tu.assign_non_tensor(data, use_dynamic_bsz=True)

    with pytest.raises(NotImplementedError, match="dynamic bsz"):
        composite.postprocess_und_batch([], None, data, forward_only=True)


def test_merge_composite_outputs_refuses_to_swallow_a_published_infer_output():
    """Keeping only ``parts[-1]``'s model_output would drop the published log-probs."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    published = {"model_output": {"log_probs": torch.zeros(2, 3)}, "loss": [0.0], "metrics": {}}

    with pytest.raises(RuntimeError, match="would drop"):
        composite.merge_composite_outputs([published, {"model_output": {}, "loss": [], "metrics": {}}])


def test_merge_composite_outputs_passes_a_single_part_through():
    """The solo UND infer part has to survive the merge untouched."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    published = {"model_output": {"log_probs": torch.zeros(2, 3)}, "loss": [0.0], "metrics": {}}

    assert composite.merge_composite_outputs([published]) is published


def test_merge_composite_outputs_is_silent_when_only_the_last_lane_has_model_output():
    """Train mode drops the UND blob, so the ordinary merge must not trip the guard."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    und = {"model_output": {}, "loss": [0.5], "metrics": {"gen/skipped_no_groups": [1.0]}}
    gen = {"model_output": {"latents": torch.zeros(2, 3)}, "loss": [0.25], "metrics": {}}

    merged = composite.merge_composite_outputs([und, gen])

    assert merged["model_output"] == gen["model_output"]
    assert merged["loss"] == [0.5, 0.25]
    assert merged["metrics"] == {"gen/skipped_no_groups": [1.0]}


def test_merge_composite_outputs_flattens_the_gen_per_timestep_losses():
    """Merging the two lanes must yield a rectangular ``loss`` the worker can tensorize.

    The lanes publish different shapes: UND is flat per-micro-batch scalars, GEN is the
    diffusion postprocessor's per-*timestep* list per micro-batch. Merged verbatim the result is
    ragged, and ``engine_workers._postprocess_output`` does
    ``torch.sum(torch.tensor(output.pop("loss")))``:

        TypeError: must be real number, not list

    (measured 2026-09-23 20:09 on hk01dgx039, devices 0/1/6/7, ``bagel_corl_rm1_20260923_195416``).
    The flatten has to be sum-preserving, since the only consumer sums the entries.
    """
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    und = {"model_output": {}, "loss": [0.5], "metrics": {}}
    # Two micro-batches, three denoise timesteps each -- exactly what the diffusion postprocessor
    # appends when GEN is the last lane.
    gen = {"model_output": {"latents": torch.zeros(2, 3)}, "loss": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "metrics": {}}

    merged = composite.merge_composite_outputs([und, gen])

    assert merged["loss"] == [0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (
        f"every entry must be a scalar, got {merged['loss']}"
    )
    # The one thing the worker does with it: sum. It has to be a valid tensor, and the total has to
    # match the ragged input's total exactly.
    total = float(torch.sum(torch.tensor(merged["loss"])))
    assert total == pytest.approx(0.5 + 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6)


def test_merge_composite_outputs_keeps_a_gen_only_merge_tensorizable():
    """A GEN-only merge (no UND part) must stay tensorizable too, not just the mixed case."""
    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    gen = {"model_output": {"latents": torch.zeros(2, 3)}, "loss": [[0.1, 0.2], [0.3, 0.4]], "metrics": {}}
    empty = {"model_output": {}, "loss": [], "metrics": {}}

    merged = composite.merge_composite_outputs([empty, gen])

    assert all(isinstance(entry, float) for entry in merged["loss"]), merged["loss"]
    assert float(torch.sum(torch.tensor(merged["loss"]))) == pytest.approx(1.0)


def test_extra_fields_rows_reads_the_tq_column_and_unwraps_it():
    """The ``agentic_rewards`` telemetry read must survive the TransferQueue's columnar shape.

    ``tq.kv_batch_get`` returns a *columnar* TensorDict whose columns are tensordict containers, and
    the old reader called ``.tolist()`` on it -- which no ``LinkedList``/``NonTensorStack`` has. The
    blanket ``except`` in ``_compute_metrics`` swallowed the raise, so every step logged

        Failed to compute agentic_rewards metrics: 'LinkedList' object has no attribute 'tolist'

    and the whole ``agentic_rewards/*`` block was dead for the Bagel lanes (measured 2026-09-23 21:12
    on hk01dgx039, devices 0/1/6/7, ``bagel_corl_rm1_20260923_205640``). The unwrap matters just as
    much: ``list(NonTensorStack)`` yields ``NonTensorData`` wrappers, and the caller's
    ``isinstance(ef, dict)`` is False for those -- an all-zero block that reads like "no tool rewards
    were collected" rather than an error.
    """
    from tensordict import NonTensorStack
    from tensordict.tensorclass import NonTensorData

    from verl.trainer.ppo.v1.trainer_base import _extra_fields_rows

    rows = [{"tool_rewards": [1.0, 0.5]}, {"reward_extra_info": {"calc_gsm8k_reward_sum": 2.0}}]
    ef_data = TensorDict({"extra_fields": NonTensorStack(*[NonTensorData(r) for r in rows])}, batch_size=[])

    # The shape the helper must tolerate really has no ``tolist``.
    assert not hasattr(ef_data["extra_fields"], "tolist")

    assert _extra_fields_rows(ef_data) == rows


def test_extra_fields_rows_also_handles_a_plain_list_column():
    """A backend that already returns plain rows must keep working unchanged."""
    from verl.trainer.ppo.v1.trainer_base import _extra_fields_rows

    rows = [{"tool_rewards": [0.25]}, {"tool_rewards": []}]
    ef_data = TensorDict({"extra_fields": rows}, batch_size=[])

    assert _extra_fields_rows(ef_data) == rows


def test_merge_composite_outputs_keeps_metrics_aggregatable():
    """A merged metric must stay a single ``Metric``, not a list of them.

    Both lanes publish ``Metric`` objects, and the hand-rolled merge wrapped each lane's single
    ``Metric`` in a list, so the key became ``[Metric, Metric]`` and the driver's legacy reducer did
    ``np.mean`` over it:

        TypeError: unsupported operand type(s) for +: 'Metric' and 'Metric'
        (``reduce_metrics`` in ``verl/verl/utils/metric/utils.py``)

    (measured 2026-09-23 20:33 on hk01dgx039, devices 0/1/6/7, ``bagel_corl_rm1_20260923_201744``).
    ``append_to_dict`` creates ``val.init_list()`` for a ``Metric`` and flattens nested metrics, so
    the merged key stays aggregatable.
    """
    from verl.utils.metric import Metric, reduce_metrics

    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    und = {
        "model_output": {},
        "loss": [0.5],
        "metrics": Metric.from_dict({"actor/loss": 0.5}, aggregation="mean"),
    }
    gen = {
        "model_output": {"latents": torch.zeros(2, 3)},
        "loss": [[0.25]],
        "metrics": Metric.from_dict({"actor/loss": 0.25, "actor/kl": 0.1}, aggregation="mean"),
    }

    merged = composite.merge_composite_outputs([und, gen])

    # ``reduce_metrics`` is exactly what the driver runs; it must not raise.
    reduced = reduce_metrics(dict(merged["metrics"]))
    assert reduced["actor/loss"] == pytest.approx(0.375)  # mean of both lanes' contributions
    assert reduced["actor/kl"] == pytest.approx(0.1)


def test_merge_composite_outputs_keeps_plain_scalar_metrics_working():
    """Plain scalar metrics must still accumulate into a list that ``reduce_metrics`` can mean."""
    from verl.utils.metric import reduce_metrics

    from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite

    und = {"model_output": {}, "loss": [0.5], "metrics": {"gen/skipped_no_groups": [1.0]}}
    gen = {"model_output": {"latents": torch.zeros(2, 3)}, "loss": [[0.25]], "metrics": {"gen/skipped_no_groups": 0.0}}

    merged = composite.merge_composite_outputs([und, gen])
    reduced = reduce_metrics(dict(merged["metrics"]))

    assert reduced["gen/skipped_no_groups"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# UND AR sleep/wake bridge (RFC §4.11).
#
# ``checkpoint_manager.sleep_replicas()`` sleeps the UND AR replica along with GEN, but the
# naive sync path only wakes the actor's own colocated server, so the AR engine stayed asleep
# and the next rollout died at the first ``_und_decode`` with
# ``RuntimeError: Generation rejected: Engine is partially or fully asleep``.
# ---------------------------------------------------------------------------


class _FakeWakeUp:
    """Stand-in for the Ray actor method handle ``server.wake_up``."""

    def __init__(self, log):
        self._log = log

    async def remote(self, **kwargs):
        self._log.append(("wake_up", tuple(kwargs.get("tags") or ())))


class _FakeServer:
    def __init__(self, log):
        self.wake_up = _FakeWakeUp(log)


class _RecordingReplica:
    """Rollout replica whose servers record every ``wake_up.remote(tags=...)``."""

    def __init__(self, log, n_servers=1):
        self.servers = [_FakeServer(log) for _ in range(n_servers)]


class _FakeCheckpointManager:
    def __init__(self, log, replicas=()):
        self._log = log
        self.replicas = list(replicas)

    def update_weights(self, global_steps=None):
        self._log.append(("update_weights", global_steps))

    def sleep_replicas(self):
        self._log.append(("sleep_replicas", None))

    def resume_generation_replicas(self):
        self._log.append(("resume_generation_replicas", None))


def _dual_role_step_trainer(log, replicas):
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.timing_raw = {}
    trainer.global_steps = 3
    trainer.und_rollout_replicas = list(replicas)
    trainer.checkpoint_manager = _FakeCheckpointManager(log, replicas)
    return trainer


def test_on_step_end_wakes_the_und_ar_replica_after_the_gen_publish():
    """The AR wake must follow the GEN publish, one call per server, with both tags.

    Both tags are required: ``vLLMOmniHttpServer.wake_up`` defaults to ``["weights"]`` and
    ``AsyncOmni`` keeps rejecting generation while ``kv_cache`` is still sleeping.
    """
    log: list = []
    replicas = [_RecordingReplica(log, n_servers=2)]
    trainer = _dual_role_step_trainer(log, replicas)

    trainer.on_step_end()

    assert log == [
        ("update_weights", 3),
        ("resume_generation_replicas", None),
        ("wake_up", ("weights", "kv_cache")),
        ("wake_up", ("weights", "kv_cache")),
    ]


def test_on_init_end_wakes_und_after_the_parent_publish(monkeypatch):
    """``on_init_end`` must leave both pools awake, not just the GEN one ``_setup`` slept."""
    log: list = []
    replicas = [_RecordingReplica(log)]
    trainer = _dual_role_step_trainer(log, replicas)
    monkeypatch.setattr(OmniBagelCoRLTrainerSync, "_ensure_dual_role_rollout", lambda self: None)
    monkeypatch.setattr(OmniBagelCoRLTrainerSync, "_bagel_rm_enabled", lambda self: False)

    trainer.on_init_end()

    assert log == [
        ("update_weights", 3),
        ("resume_generation_replicas", None),
        ("wake_up", ("weights", "kv_cache")),
    ]


def test_on_validate_end_rewakes_und_after_the_colocated_reward_sleep():
    """``_validate`` sleeps *all* replicas for a colocated RM; only GEN gets woken back.

    Without this hook the AR engine would be asleep for the step after validation, i.e. the
    ``ENABLE_RM=1`` (and ``val_before_train=True``) variants of the same bug.
    """
    log: list = []
    replicas = [_RecordingReplica(log)]
    trainer = _dual_role_step_trainer(log, replicas)
    # What ``_validate`` left behind: everything slept, GEN resumed by the naive publish.
    trainer.checkpoint_manager.sleep_replicas()
    trainer.checkpoint_manager.update_weights(3)

    trainer.on_validate_end()

    assert log == [
        ("sleep_replicas", None),
        ("update_weights", 3),
        ("wake_up", ("weights", "kv_cache")),
    ]


def test_wake_rollout_replicas_wakes_the_gen_pool_too():
    """``on_step_end`` must clear BOTH pools' tags, not just the UND AR one.

    ``checkpoint_manager.sleep_replicas()`` sleeps every registered replica, so the GEN pool
    is asleep too. The parent's naive publish resumes only the actor's own GEN server and
    passes no tags -- ``vLLMOmniHttpServer.wake_up`` then falls back to
    ``_get_wake_up_tags() == ["weights"]`` while ``AsyncOmni`` keeps rejecting generation
    until every tag is cleared. Measured 2026-09-24, ``bagel_corl_rm1_20260923_213235``
    (200-step, ``test_freq=30``): 30 clean steps, then the validation boundary died with
    ``Generation rejected: Engine is partially or fully asleep ... ['weights', 'kv_cache']``.
    """
    log: list = []
    # The real topology: the parent registered the GEN replica, we added the UND AR one on top.
    gen_replica = _RecordingReplica(log)
    und_replica = _RecordingReplica(log)
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.timing_raw = {}
    trainer.global_steps = 7
    trainer.und_rollout_replicas = [und_replica]
    trainer.checkpoint_manager = _FakeCheckpointManager(log, [gen_replica, und_replica])

    trainer._wake_rollout_replicas()

    assert log == [
        ("wake_up", ("weights", "kv_cache")),  # UND AR pool
        ("wake_up", ("weights", "kv_cache")),  # GEN pool, previously left on ['weights']
    ]

    # The GEN pool is exactly "registered minus the UND AR pool", by identity.
    assert trainer._gen_rollout_replicas() == [gen_replica]


def test_validate_rewakes_both_pools_between_the_rm_sleep_and_the_next_batch(monkeypatch):
    """The val loop's colocated-RM sleep must be undone *before* the next val batch rolls out.

    The parent loop (``verl/trainer/ppo/v1/trainer_base.py:1034-1037``) sleeps every replica to
    hand the GPU to the colocated RM, scores, then publishes through
    ``checkpoint_manager.update_weights()`` -- whose naive path resumes only the actor's own GEN
    server and requests no tags. It then rolls out again, so every batch after the first starts
    against a sleeping engine. ``on_validate_end`` runs after the whole loop, too late.

    Measured 2026-09-24 (``bagel_corl_rm1_20260924_034359``, ``ENABLE_RM=1``): 4 validation
    batches (``VAL_MAX_SAMPLES=8`` / ``val_batch_size=2``) and the run died at the ``test_freq=30``
    boundary with "Generation rejected: Engine is partially or fully asleep". This pins the wake
    landing *between* the publish and the next rollout.
    """
    log: list = []
    gen_replica = _RecordingReplica(log)
    und_replica = _RecordingReplica(log)

    def _fake_parent_validate(self):
        """Stand in for one pass of the parent's per-batch val loop."""
        log.append(("generate_sequences", 1))
        self.checkpoint_manager.sleep_replicas()
        self.checkpoint_manager.update_weights()
        log.append(("generate_sequences", 2))
        return {"episode/K": 1.0}

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.global_steps = 30
    trainer.und_rollout_replicas = [und_replica]
    trainer.checkpoint_manager = _FakeCheckpointManager(log, [gen_replica, und_replica])

    monkeypatch.setattr(OmniPPOTrainerSync, "_validate", _fake_parent_validate)
    trainer._validate()

    assert log == [
        ("generate_sequences", 1),
        ("sleep_replicas", None),
        ("update_weights", None),
        # Both pools fully awake again, and in that order: UND AR then GEN.
        ("wake_up", ("weights", "kv_cache")),
        ("wake_up", ("weights", "kv_cache")),
        ("generate_sequences", 2),
    ]
    # The wrap must not outlive the call, or the training path would double-wake.
    assert trainer.checkpoint_manager.update_weights.__func__ is _FakeCheckpointManager.update_weights


def test_wake_und_rollout_replicas_is_a_noop_without_a_dual_role_replica():
    """No colocated AR replica means no Ray traffic, so the hook is safe to call unguarded."""
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer._wake_und_rollout_replicas()  # attribute never set
    trainer.und_rollout_replicas = []
    trainer._wake_und_rollout_replicas()


# --- The colocated RM must be handed the fields its scorer keys off -----------------
#
# ``_compute_reward_colocate`` is the *only* scorer once ``reward.reward_model.enable=1``:
# ``RewardLoopManager.reward_loop_worker_handles`` returns None without a separate resource
# pool (verl/verl/experimental/reward_loop/reward_loop.py:300), so ``AgentLoopWorker._compute_score``
# never runs the RM inline and ``_postprocess``'s ``input_non_tensor_batch`` forwarding is not
# the path that feeds it. The colocated path then fetched only
# ``prompts``/``responses``/``raw_prompt``, so the episode row reached ``VisualRewardManager``
# with ``data_source=''`` and no ground truth, and every episode scored a structural zero.
# Measured 2026-09-23 on hk01dgx039 (devices 0/1/6/7, ``bagel_corl_rm1_20260923_154117``),
# once per UND row:
#
#   VisualRewardManager: episode row carries no reward_model.ground_truth
#   (data_source=''); the episode scorer will return a zero reward for it.
#
# The non-zero ``und_reward`` in that run's trajectory dumps came from the agent loop's own
# tool-call-presence term, not from this scorer, so the zero was invisible in ``und_reward``.


def _colocate_reward_harness(monkeypatch, *, tq_fields: dict):
    """Call ``_compute_reward_colocate`` against a fake TQ and a capturing reward manager."""
    from verl.trainer.ppo.v1 import trainer_base as v1_trainer_base
    from verl import DataProto

    seen: dict = {}

    def _kv_batch_get(keys=None, partition_id=None, select_fields=None):
        seen["select_fields"] = list(select_fields or [])
        return dict(tq_fields)

    def _kv_batch_put(keys=None, partition_id=None, fields=None):
        seen["write_back"] = fields

    monkeypatch.setattr(v1_trainer_base.tq, "kv_batch_get", _kv_batch_get)
    monkeypatch.setattr(v1_trainer_base.tq, "kv_batch_put", _kv_batch_put)

    class _RewardManager:
        def compute_rm_score(self, rm_input):
            seen["rm_input"] = rm_input
            return DataProto.from_tensordict(
                tu.get_tensordict({"rm_scores": torch.zeros(2, 3)}), meta_info={"reward_extra_keys": []}
            )

    trainer = types.SimpleNamespace()
    trainer.reward_loop_manager = _RewardManager()
    trainer.tokenizer = types.SimpleNamespace(pad_token_id=0)
    trainer._lengths_to_mask = v1_trainer_base.PPOTrainer._lengths_to_mask

    class _Batch(dict):
        """Stand-in for ``KVBatchMeta``: the real object supports ``len()`` and ``.keys``."""

        partition_id = "train"

    batch = _Batch({"a": None, "b": None})
    v1_trainer_base.PPOTrainer._compute_reward_colocate(trainer, batch)
    return seen


def _nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=torch.long) for r in rows], layout=torch.jagged)


def _tq_episode_fields(*, with_dataset_fields: bool) -> dict:
    fields = {"prompts": _nested([[1, 2, 3], [4, 5]]), "responses": _nested([[6], [7, 8]])}
    fields["raw_prompt"] = ["drop me", "drop me too"]
    if with_dataset_fields:
        fields["data_source"] = ["unicot_reflection", "unicot_breakdown"]
        fields["reward_model"] = [
            {"ground_truth": {"task_type": "reflect", "expected_num_images": 3}},
            {"ground_truth": {"task_type": "plan", "expected_num_images": 2}},
        ]
    return fields


def test_colocate_reward_requests_the_dataset_fields(monkeypatch):
    """The scorer cannot dispatch without them, so they must be in the TQ projection."""
    seen = _colocate_reward_harness(monkeypatch, tq_fields=_tq_episode_fields(with_dataset_fields=True))

    assert "data_source" in seen["select_fields"]
    assert "reward_model" in seen["select_fields"]


def test_colocate_reward_forwards_data_source_and_ground_truth(monkeypatch):
    """``VisualRewardManager`` must actually see them on the row, not just in the fetch."""
    seen = _colocate_reward_harness(monkeypatch, tq_fields=_tq_episode_fields(with_dataset_fields=True))
    non_tensor = seen["rm_input"].non_tensor_batch

    assert list(non_tensor["data_source"]) == ["unicot_reflection", "unicot_breakdown"]
    assert [gt["ground_truth"]["task_type"] for gt in non_tensor["reward_model"]] == ["reflect", "plan"]
    # The pre-existing contract must survive: ``raw_prompt`` is still forwarded.
    assert list(non_tensor["raw_prompt"]) == ["drop me", "drop me too"]


def test_colocate_reward_unwraps_non_tensor_data_wrappers(monkeypatch):
    """The TQ hands these back wrapped; the scorer needs plain values, not wrappers.

    ``VisualRewardManager`` reads ``data_source`` as a string and
    ``reward_model["ground_truth"]`` as a dict. A ``NonTensorData`` wrapper satisfies neither, and
    the failure is silent: the wrapper is truthy where a default would have been used, so the
    episode still scores a structural zero.
    """
    from tensordict.tensorclass import NonTensorData

    fields = _tq_episode_fields(with_dataset_fields=False)
    fields["data_source"] = [NonTensorData("unicot_reflection"), NonTensorData("unicot_breakdown")]
    fields["reward_model"] = [
        NonTensorData({"ground_truth": {"task_type": "reflect"}}),
        NonTensorData({"ground_truth": {"task_type": "plan"}}),
    ]
    seen = _colocate_reward_harness(monkeypatch, tq_fields=fields)
    non_tensor = seen["rm_input"].non_tensor_batch

    assert list(non_tensor["data_source"]) == ["unicot_reflection", "unicot_breakdown"]
    assert [gt["ground_truth"]["task_type"] for gt in non_tensor["reward_model"]] == ["reflect", "plan"]


def test_colocate_reward_tolerates_rows_without_the_dataset_fields(monkeypatch):
    """Padding rows / non-dataset partitions must degrade to a default, never to a crash."""
    seen = _colocate_reward_harness(monkeypatch, tq_fields=_tq_episode_fields(with_dataset_fields=False))
    non_tensor = seen["rm_input"].non_tensor_batch

    assert "data_source" not in non_tensor
    assert "reward_model" not in non_tensor
    assert list(non_tensor["raw_prompt"]) == ["drop me", "drop me too"]


# --- The GEN prompt ids must survive onto the batch the diffusion step sees -------
#
# ``DiffusionTrainingAdapter._prompt_token_ids_to_batch``
# (``bagel_flow_grpo/diffusers_training_adapter.py:152``) reads ``prompt_token_ids`` off the
# micro-batch, and ``prepare_micro_batches`` only propagates non-tensor entries that sit on the
# batch it is handed -- the sibling ``skip_gen`` / ``has_complete_gen_groups`` / ``num_gen_rows``
# flags ride through for exactly that reason. Ids parked in ``non_tensor_batch`` therefore never
# reach the diffusion step:
#
#   KeyError: 'key "prompt_token_ids" not found in TensorDict with keys
#   ['advantages', 'all_latents', 'all_timesteps', 'bagel_corl_gen', 'gradient_accumulation_steps',
#    'has_complete_gen_groups', 'micro_batch_size_per_gpu', 'num_gen_rows', 'old_log_probs',
#    'returns', 'rm_scores', 'sample_level_rewards', 'sample_level_scores', 'skip_gen', 'sp_size',
#    'use_dynamic_bsz']'
#
# Measured 2026-09-23 16:58 on hk01dgx039 (devices 0/1/6/7, ``bagel_corl_rm1_20260923_164059``):
# the first step to reach the GEN diffusion loss at all, dying after rollout + reward + advantage
# had all been paid for. The fold had been inlined in ``apply_gen_flowgrpo_advantage`` -- the
# CPU-test fallback branch -- while the live path goes through
# ``_diffusion_v1_gen_lane()._compute_advantage(proto)``, so it never ran where it mattered.


def _gen_proto_with_prompt_ids(prompt_ids=None):
    from verl import DataProto
    from verl_omni.trainer.omni.bagel_corl_gen_adv import build_gen_flowgrpo_proto

    rows = [
        {
            "gen_group_uid": "g0",
            "all_latents": torch.zeros(3, 4),
            "timesteps": torch.zeros(3),
            "rollout_log_probs": torch.zeros(3),
            "rm_score": 1.0,
            "seed_index": i,
            "prompt_token_ids": ids,
        }
        for i, ids in enumerate(prompt_ids if prompt_ids is not None else ([11, 12], [21]))
    ]
    return build_gen_flowgrpo_proto(rows)


def test_gen_prompt_token_ids_fold_puts_the_ids_on_the_batch():
    """``fold_gen_prompt_token_ids`` must move them off ``non_tensor_batch``."""
    from verl.utils import tensordict_utils as tu
    from verl_omni.trainer.omni.bagel_corl_gen_adv import fold_gen_prompt_token_ids

    proto = _gen_proto_with_prompt_ids()
    assert "prompt_token_ids" in proto.non_tensor_batch
    assert "prompt_token_ids" not in proto.batch.keys()

    fold_gen_prompt_token_ids(proto)

    folded = tu.get_non_tensor_data(proto.batch, "prompt_token_ids", default=None)
    assert folded is not None
    assert [list(ids) for ids in folded] == [[11, 12], [21]]


def test_gen_prompt_token_ids_fold_is_a_noop_without_the_field():
    """A proto that never carried the ids must not gain an empty one."""
    from verl_omni.trainer.omni.bagel_corl_gen_adv import fold_gen_prompt_token_ids

    proto = _gen_proto_with_prompt_ids()
    proto.non_tensor_batch.pop("prompt_token_ids", None)

    fold_gen_prompt_token_ids(proto)

    assert "prompt_token_ids" not in proto.batch.keys()


def test_apply_gen_flowgrpo_advantage_still_folds_the_ids():
    """The fallback branch must keep folding, now via the shared helper."""
    from verl.utils import tensordict_utils as tu
    from verl_omni.trainer.omni.bagel_corl_gen_adv import apply_gen_flowgrpo_advantage

    rows = [
        {
            "gen_group_uid": "g0",
            "all_latents": torch.zeros(3, 4),
            "timesteps": torch.zeros(3),
            "rollout_log_probs": torch.zeros(3),
            "rm_score": float(i),
            "seed_index": i,
            "prompt_token_ids": [i, i + 1],
        }
        for i in range(2)
    ]
    proto, _ = apply_gen_flowgrpo_advantage(rows)

    folded = tu.get_non_tensor_data(proto.batch, "prompt_token_ids", default=None)
    assert folded is not None
    assert [list(ids) for ids in folded] == [[0, 1], [1, 2]]


# --- The GEN prompt ids must survive onto the batch the diffusion step sees -------
#
# ``DiffusionTrainingAdapter._prompt_token_ids_to_batch``
# (``bagel_flow_grpo/diffusers_training_adapter.py:152``) reads ``prompt_token_ids`` off the
# micro-batch, and ``prepare_micro_batches`` only propagates non-tensor entries that sit on the
# batch it is handed -- the sibling ``skip_gen`` / ``has_complete_gen_groups`` / ``num_gen_rows``
# flags ride through for exactly that reason. Ids parked in ``non_tensor_batch`` therefore never
# reach the diffusion step:
#
#   KeyError: 'key "prompt_token_ids" not found in TensorDict with keys
#   ['advantages', 'all_latents', 'all_timesteps', 'bagel_corl_gen', 'gradient_accumulation_steps',
#    'has_complete_gen_groups', 'micro_batch_size_per_gpu', 'num_gen_rows', 'old_log_probs',
#    'returns', 'rm_scores', 'sample_level_rewards', 'sample_level_scores', 'skip_gen', 'sp_size',
#    'use_dynamic_bsz']'
#
# Measured 2026-09-23 16:58 on hk01dgx039 (devices 0/1/6/7, ``bagel_corl_rm1_20260923_164059``):
# the first step to reach the GEN diffusion loss at all, dying after rollout + reward + advantage
# had all been paid for. The fold had been inlined in ``apply_gen_flowgrpo_advantage`` -- the
# CPU-test fallback branch -- while the live path goes through
# ``_diffusion_v1_gen_lane()._compute_advantage(proto)``, so it never ran where it mattered.


def _gen_proto_with_prompt_ids(prompt_ids=None):
    from verl import DataProto
    from verl_omni.trainer.omni.bagel_corl_gen_adv import build_gen_flowgrpo_proto

    rows = [
        {
            "gen_group_uid": "g0",
            "all_latents": torch.zeros(3, 4),
            "timesteps": torch.zeros(3),
            "rollout_log_probs": torch.zeros(3),
            "rm_score": 1.0,
            "seed_index": i,
            "prompt_token_ids": ids,
        }
        for i, ids in enumerate(prompt_ids if prompt_ids is not None else ([11, 12], [21]))
    ]
    return build_gen_flowgrpo_proto(rows)


def test_gen_prompt_token_ids_fold_puts_the_ids_on_the_batch():
    """``fold_gen_prompt_token_ids`` must move them off ``non_tensor_batch``."""
    from verl.utils import tensordict_utils as tu
    from verl_omni.trainer.omni.bagel_corl_gen_adv import fold_gen_prompt_token_ids

    proto = _gen_proto_with_prompt_ids()
    assert "prompt_token_ids" in proto.non_tensor_batch
    assert "prompt_token_ids" not in proto.batch.keys()

    fold_gen_prompt_token_ids(proto)

    folded = tu.get_non_tensor_data(proto.batch, "prompt_token_ids", default=None)
    assert folded is not None
    assert [list(ids) for ids in folded] == [[11, 12], [21]]


def test_gen_prompt_token_ids_fold_is_a_noop_without_the_field():
    """A proto that never carried the ids must not gain an empty one."""
    from verl_omni.trainer.omni.bagel_corl_gen_adv import fold_gen_prompt_token_ids

    proto = _gen_proto_with_prompt_ids()
    proto.non_tensor_batch.pop("prompt_token_ids", None)

    fold_gen_prompt_token_ids(proto)

    assert "prompt_token_ids" not in proto.batch.keys()


def test_apply_gen_flowgrpo_advantage_still_folds_the_ids():
    """The fallback branch must keep folding, now via the shared helper."""
    from verl.utils import tensordict_utils as tu
    from verl_omni.trainer.omni.bagel_corl_gen_adv import apply_gen_flowgrpo_advantage

    rows = [
        {
            "gen_group_uid": "g0",
            "all_latents": torch.zeros(3, 4),
            "timesteps": torch.zeros(3),
            "rollout_log_probs": torch.zeros(3),
            "rm_score": float(i),
            "seed_index": i,
            "prompt_token_ids": [i, i + 1],
        }
        for i in range(2)
    ]
    proto, _ = apply_gen_flowgrpo_advantage(rows)

    folded = tu.get_non_tensor_data(proto.batch, "prompt_token_ids", default=None)
    assert folded is not None
    assert [list(ids) for ids in folded] == [[0, 1], [1, 2]]


def test_gen_pass_reslices_per_row_non_tensors_for_the_diffusion_step(monkeypatch):
    """A per-row non-tensor must be sliced per micro-batch, not replicated whole.

    ``chunk_tensordict`` rebuilds each micro-batch from ``td.items()`` and copies non-tensor entries
    through verbatim, so an 8-row ``prompt_token_ids`` used to reach **every** micro-batch intact.
    The Bagel adapter reads those ids per row (``diffusers_training_adapter.py:152``), so
    ``bagel_model`` then concatenated 8 text rows against 1 latent row:

        RuntimeError: Sizes of tensors must match except in dimension 1.
        Expected size 8 but got size 1 for tensor number 1 in the list.

    Measured 2026-09-23 17:23 on hk01dgx039 (devices 0/1/6/7, ``bagel_corl_rm1_20260923_170726``).
    """
    from verl.utils import tensordict_utils as tu
    from verl_omni.workers.engine.fsdp import diffusers_impl

    rows = 4
    data = TensorDict({"all_latents": torch.zeros(rows, 3, 2, 2, 2)}, batch_size=[rows])
    tu.assign_non_tensor(data, prompt_token_ids=np.array([[i, i + 1] for i in range(rows)], dtype=object))
    tu.assign_non_tensor(data, sp_size=1)

    micro_batches = tu.chunk_tensordict(data, rows)

    # The defect, before the fix: every micro-batch still sees all rows.
    assert len(tu.get_non_tensor_data(micro_batches[0], "prompt_token_ids", default=None)) == rows

    diffusers_impl.DiffusersFSDPEngine._reslice_per_row_non_tensors(data, micro_batches)

    for index, micro_batch in enumerate(micro_batches):
        ids = tu.get_non_tensor_data(micro_batch, "prompt_token_ids", default=None)
        assert [list(row) for row in ids] == [[index, index + 1]]
        # Scalars are untouched: they are not row-aligned and must keep reaching every micro-batch.
        assert tu.get_non_tensor_data(micro_batch, "sp_size", default=None) == 1


def test_gen_pass_rereslice_leaves_scalars_and_blobs_alone():
    """Only row-aligned list-like keys are re-sliced; a whole-batch stash must survive intact."""
    from verl.utils import tensordict_utils as tu
    from verl_omni.workers.engine.fsdp import diffusers_impl

    rows = 4
    data = TensorDict({"all_latents": torch.zeros(rows, 3, 2, 2, 2)}, batch_size=[rows])
    tu.assign_non_tensor(data, skip_gen=False)
    stash = TensorDict({"all_latents": torch.zeros(rows, 3, 2, 2, 2)}, batch_size=[rows])
    tu.assign_non_tensor(data, bagel_corl_gen=stash)

    micro_batches = tu.chunk_tensordict(data, rows)

    diffusers_impl.DiffusersFSDPEngine._reslice_per_row_non_tensors(data, micro_batches)

    for micro_batch in micro_batches:
        assert tu.get_non_tensor_data(micro_batch, "skip_gen", default=None) is False
        # ``bagel_corl_gen`` reports a length too, but it is one blob, not a column.
        assert len(tu.get_non_tensor_data(micro_batch, "bagel_corl_gen", default=None)) == rows
