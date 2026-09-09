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
"""CPU tests for dual-lane Bagel Co-RL TQ packing (patterns 1–3, child gather)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils.tensordict_utils import assign_non_tensor_data

from verl_omni.agent_loop.bagel_corl_lib import EpisodeRollout, GenSample, gen_sample_uid
from verl_omni.agent_loop.bagel_corl_tq import (
    classify_episode_pattern,
    pack_dual_lane_episode,
    split_und_gen_metas,
)
from verl_omni.trainer.omni.bagel_corl_gen_adv import select_gen_advantages_for_step
from verl_omni.utils.config import validate_bagel_corl_config


def _sample(call: str, seed: int, *, valid: bool = True, score: float = 1.0) -> GenSample:
    return GenSample(
        gen_sample_uid=gen_sample_uid(call, seed),
        gen_group_uid=call,
        seed_index=seed,
        valid=valid,
        prompt_token_ids=[1, 2],
        all_latents=torch.zeros(2, 4),
        timesteps=torch.zeros(2),
        rollout_log_probs=[0.1, 0.2],
        rm_score=score,
    )


def _ep(*, uid: str, turns: int, k: int, samples, used: bool = True):
    return EpisodeRollout(
        und_group_uid=uid,
        episode_uid=f"{uid}-ep",
        policy_version=1,
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        turns=turns,
        gen_samples=list(samples),
        used_image_credit=used,
        und_reward=1.0 if samples else 0.0,
        num_gen_calls=k,
    )


def test_pattern_paired_j_eq_k():
    samples = [_sample("c0", 0), _sample("c0", 1, score=0.0)]
    pack = pack_dual_lane_episode(
        _ep(uid="task", turns=1, k=1, samples=samples),
        und_key="task_0_0",
        expected_s=2,
    )
    assert pack.pattern == "paired"
    assert pack.j == pack.k == 1
    assert len(pack.child_gen_keys) == 2
    assert all(k.startswith("task_0_0::gen::c0::") for k in pack.child_gen_keys)
    assert pack.und_record["fields"]["uid"] == "task"
    assert pack.gen_records[0]["fields"]["uid"] == "c0"


def test_pattern_mixed_reflection_no_gen_keys_for_surplus_und():
    samples = [_sample("c0", 0), _sample("c0", 1)]
    pack = pack_dual_lane_episode(
        _ep(uid="task", turns=2, k=1, samples=samples),
        und_key="task_0_0",
        expected_s=2,
    )
    assert pack.pattern == "mixed"
    assert pack.j == 2 and pack.k == 1
    assert len(pack.child_gen_keys) == 2


def test_pattern_gen_off_k_zero():
    pack = pack_dual_lane_episode(
        _ep(uid="task", turns=2, k=0, samples=[], used=False),
        und_key="task_0_0",
        expected_s=2,
    )
    assert pack.pattern == "gen_off"
    assert pack.child_gen_keys == []
    assert pack.metrics["und/no_image_credit"] == 1.0
    assert pack.metrics["gen/skipped_no_groups"] == 1.0


def test_gate_rejects_j_lt_k():
    with pytest.raises(ValueError, match="J >= K"):
        classify_episode_pattern(1, 2)


def test_flowgrpo_uid_is_gen_group():
    samples = [_sample("callA", 0), _sample("callA", 1)]
    pack = pack_dual_lane_episode(
        _ep(uid="dataset_task", turns=1, k=1, samples=samples),
        und_key="dataset_task_0_0",
        expected_s=2,
    )
    for rec in pack.gen_records:
        assert rec["fields"]["uid"] == "callA"
        assert rec["fields"]["uid"] != "dataset_task"


def test_token_grpo_uid_is_und_group():
    packs = [
        pack_dual_lane_episode(
            _ep(uid="taskA", turns=1, k=0, samples=[], used=False),
            und_key=f"taskA_{session}_0",
            expected_s=2,
        )
        for session in (0, 1)
    ]
    assert packs[0].und_record["fields"]["uid"] == packs[1].und_record["fields"]["uid"] == "taskA"


def test_split_und_gen_metas_n_siblings_mixed_patterns():
    """N=2: one pattern-3 sibling + one pattern-1 sibling."""
    off = pack_dual_lane_episode(
        _ep(uid="task", turns=1, k=0, samples=[], used=False),
        und_key="task_0_0",
        expected_s=2,
    )
    paired = pack_dual_lane_episode(
        _ep(uid="task", turns=1, k=1, samples=[_sample("c1", 0), _sample("c1", 1)]),
        und_key="task_1_0",
        expected_s=2,
    )
    gen_by_key = {r["key"]: r for r in paired.gen_records}
    und_batch, gen_batch = split_und_gen_metas(
        [off.und_record, paired.und_record],
        gen_by_key,
    )
    assert len(und_batch) == 2
    assert und_batch[0]["child_gen_keys"] == []
    assert len(gen_batch) == 2
    assert all(row["gen_group_uid"] == "c1" for row in gen_batch)


def test_engine_gen_step_reads_bagel_corl_gen_advantages():
    poisoned = torch.full((2, 3), 99.0)
    flow = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    micro = TensorDict({"advantages": poisoned, "old_log_probs": torch.zeros(2, 3)}, batch_size=[2])
    assign_non_tensor_data(micro, "bagel_corl_gen", TensorDict({"advantages": flow}, batch_size=[2]))
    got = select_gen_advantages_for_step(micro, 1, require_bagel_corl_gen=True)
    assert torch.allclose(got, flow[:, 1])
    assert not torch.allclose(got, poisoned[:, 1])

    bare = TensorDict({"advantages": poisoned}, batch_size=[2])
    with pytest.raises(ValueError, match="refusing UND"):
        select_gen_advantages_for_step(bare, 0, require_bagel_corl_gen=True)


def test_retire_n_eq_two_s_config_gate():
    cfg = OmegaConf.create(
        {
            "trainer": {"v1": {"trainer_mode": "bagel_corl_sync"}},
            "actor_rollout_ref": {
                "model": {"path": "/models/ByteDance-Seed/BAGEL-7B-MoT", "lora_rank": 64},
                "rollout": {"n": 3, "agent": {"gen_samples_per_call": 2, "max_generate_passes": 1}},
            },
        }
    )
    validate_bagel_corl_config(cfg)


def test_rewrite_sets_dual_lane_manager():
    from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "/tmp/bagel",
                    "algorithm": "flow_grpo",
                    "composite_mode": "bagel_corl",
                    "architecture": "OmniBagelForConditionalGeneration",
                },
                "actor": {},
                "rollout": {
                    "name": "vllm_omni",
                    "response_length": 512,
                    "trace": {"backend": None, "token2text": False},
                    "agent": {"default_agent_loop": "bagel_multiturn_agent"},
                },
            }
        }
    )
    trainer._rewrite_bagel_corl_configs()
    agent = trainer.config.actor_rollout_ref.rollout.agent
    assert agent.agent_loop_manager_class.endswith("BagelCorlAgentLoopManagerTQ")
