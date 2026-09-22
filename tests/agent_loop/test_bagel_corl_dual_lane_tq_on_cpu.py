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
"""CPU tests for dual-lane Bagel Co-RL (Joint-Training) TQ packing (patterns 1–3, child gather)."""

from __future__ import annotations

import asyncio

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils.tensordict_utils import assign_non_tensor_data, list_of_dict_to_tensordict

from verl_omni.agent_loop.bagel_corl_lib import (
    EpisodeRollout,
    GenSample,
    aggregate_episode_metrics,
    cond_reuse_metrics,
    conditioning_uid,
    flatten_multiturn_rollouts,
    gen_sample_uid,
)
from verl_omni.agent_loop.bagel_corl_tq import (
    classify_episode_pattern,
    episode_from_agent_extra,
    pack_dual_lane_episode,
    put_dual_lane_rows,
    split_und_gen_metas,
    und_row_reward,
)
from verl_omni.pipelines.bagel_flow_grpo.bagel_corl import DUAL_LORA_TARGET_MODULES
from verl_omni.trainer.omni.bagel_corl_gen_adv import select_gen_advantages_for_step
from verl_omni.utils.config import validate_bagel_corl_config

# RFC §4.0.2 dual-lane LoRA: the rewrite requires the explicit list these fixtures model.
_BAGEL_LORA_TARGETS = list(DUAL_LORA_TARGET_MODULES)


def _sample(
    call: str,
    seed: int,
    *,
    valid: bool = True,
    score: float = 1.0,
    cond: str | None = None,
    cond_len: int = 0,
) -> GenSample:
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
        cond_uid=cond,
        cond_len=cond_len,
    )


def _ep(*, uid: str, turns: int, k: int, samples, used: bool = True, r2: dict | None = None):
    return EpisodeRollout(
        und_group_uid=uid,
        episode_uid=f"{uid}-ep",
        policy_version=1,
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        rollout_log_probs=[-0.25, -0.5],
        turns=turns,
        gen_samples=list(samples),
        used_image_credit=used,
        und_reward=1.0 if samples else 0.0,
        num_gen_calls=k,
        r2_metrics=dict(r2 or {}),
    )


def test_und_record_and_flatten_row_publish_the_rollout_log_probs():
    """Both UND record builders must carry π_rollout: the trainer reads it per response token.

    The recipe runs ``rollout.calculate_log_probs=True``, and ``_compute_old_log_prob`` then
    selects ``rollout_log_probs`` off the TQ next to our recomputed ``old_log_probs``
    (``trainer_base.py:1506``). A builder that drops the field reproduces the measured
    ``KeyError: 'rollout_log_probs'``.
    """
    episode = _ep(uid="task", turns=1, k=0, samples=[])
    pack = pack_dual_lane_episode(episode, und_key="task_0_0", expected_s=2)
    assert pack.und_record["fields"]["rollout_log_probs"] == pytest.approx([-0.25, -0.5])

    flat = flatten_multiturn_rollouts([episode], expected_s=2)
    assert flat.und_batch[0]["rollout_log_probs"] == pytest.approx([-0.25, -0.5])


def test_episode_rebuilt_from_agent_fields_keeps_the_rollout_log_probs():
    """The TQ worker rebuilds the episode before packing; π_rollout rides ``response_logprobs``."""
    episode = episode_from_agent_extra(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        extra={"und_group_uid": "task"},
        und_group_uid="task",
        rollout_log_probs=[-0.25, -0.5],
    )
    assert episode.rollout_log_probs == pytest.approx([-0.25, -0.5])

    # ``response_logprobs`` in ``extra`` is the fallback for an extra-only rebuild.
    from_extra = episode_from_agent_extra(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        extra={"und_group_uid": "task", "response_logprobs": [-0.75, -1.0]},
        und_group_uid="task",
    )
    assert from_extra.rollout_log_probs == pytest.approx([-0.75, -1.0])


class _RecordingTQ:
    """Records each ``kv_batch_put`` call so the lane split is assertable on CPU."""

    def __init__(self) -> None:
        self.puts: list[dict] = []

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.puts.append({"keys": list(keys), "fields": fields, "tags": list(tags), "partition_id": partition_id})


def _und_row() -> dict:
    return {"prompts": torch.arange(2), "responses": torch.arange(3), "input_ids": torch.arange(5)}


def _gen_row() -> dict:
    return {"prompt_token_ids": [1, 2], "all_latents": torch.zeros(2, 4), "timesteps": torch.zeros(2)}


def test_the_two_lanes_are_written_in_separate_puts():
    """UND and GEN rows have disjoint schemas, so they cannot share one ``kv_batch_put``.

    ``list_of_dict_to_tensordict`` takes the column set from the first row
    (``tensordict_utils.py:929``), so packing a UND row (prompts/responses/input_ids) with a GEN
    row (prompt_token_ids/all_latents/timesteps) raises ``KeyError: 'prompts'``. That fires the
    first time an episode calls ``generate_image`` (K >= 1); pattern-3 episodes hid it because
    then there is exactly one UND row and the schema is trivially uniform.
    """
    tq_client = _RecordingTQ()
    asyncio.run(
        put_dual_lane_rows(
            tq_client,
            keys=["task_0_0", "task_0_0_gen0", "task_0_0_gen1"],
            field_dicts=[_und_row(), _gen_row(), _gen_row()],
            tags=[{"bagel_role": "und"}, {"bagel_role": "gen"}, {"bagel_role": "gen"}],
            partition_id="train",
        )
    )

    assert [put["keys"] for put in tq_client.puts] == [["task_0_0"], ["task_0_0_gen0", "task_0_0_gen1"]]
    # Each put sees one homogeneous schema, which is what ``list_of_dict_to_tensordict`` needs.
    assert set(tq_client.puts[0]["fields"].keys()) == {"prompts", "responses", "input_ids"}
    assert set(tq_client.puts[1]["fields"].keys()) == {"prompt_token_ids", "all_latents", "timesteps"}
    assert all(put["partition_id"] == "train" for put in tq_client.puts)


def test_packing_both_lanes_into_one_put_is_what_fails():
    """Mutation guard: the single-put version is exactly the bug being avoided."""
    with pytest.raises(KeyError, match="prompts"):
        list_of_dict_to_tensordict([_und_row(), _gen_row()])


def test_pattern3_writes_a_single_und_put():
    """K=0 has no GEN rows: one put, and no empty ``kv_batch_put`` call."""
    tq_client = _RecordingTQ()
    asyncio.run(
        put_dual_lane_rows(
            tq_client,
            keys=["task_0_0"],
            field_dicts=[_und_row()],
            tags=[{"bagel_role": "und"}],
            partition_id="val",
        )
    )
    assert [put["keys"] for put in tq_client.puts] == [["task_0_0"]]
    assert tq_client.puts[0]["partition_id"] == "val"


def test_unaligned_dual_lane_rows_are_rejected():
    """A missing tag/field must fail loud, not silently write the wrong row for a key."""
    with pytest.raises(ValueError, match="aligned keys/fields/tags"):
        asyncio.run(
            put_dual_lane_rows(
                _RecordingTQ(),
                keys=["a", "b"],
                field_dicts=[_und_row()],
                tags=[{}, {}],
                partition_id="train",
            )
        )


# --------------------------------------------------------------------------- #
# RFC §4.4.2a — conditioning identity (``cond_uid`` / ``cond_len``)
# --------------------------------------------------------------------------- #
def test_conditioning_uid_is_content_addressed_not_a_counter():
    """Same tokens -> same uid (that is the reuse group); anything else -> different."""
    assert conditioning_uid([1, 2, 3]) == conditioning_uid([1, 2, 3])
    assert conditioning_uid([1, 2, 3]) != conditioning_uid([1, 2, 4])
    # Order matters: a reordered slice is a different prefill, so it must not collide.
    assert conditioning_uid([1, 2, 3]) != conditioning_uid([3, 2, 1])
    # Different lengths must not collide via a shared prefix.
    assert conditioning_uid([1, 2]) != conditioning_uid([1, 2, 0])
    assert conditioning_uid([]) == conditioning_uid([])


def test_cond_identity_rides_each_gen_row():
    samples = [_sample("c0", 0, cond="h0", cond_len=7), _sample("c0", 1, cond="h0", cond_len=7)]
    pack = pack_dual_lane_episode(
        _ep(uid="task", turns=1, k=1, samples=samples),
        und_key="task_0_0",
        expected_s=2,
    )
    assert [r["fields"]["cond_uid"] for r in pack.gen_records] == ["h0", "h0"]
    assert [r["fields"]["cond_len"] for r in pack.gen_records] == [7, 7]


def test_und_row_carries_distinct_conditionings_not_one_scalar():
    """K>1 episodes have K conditionings, so the UND row records the *set*."""
    samples = [
        _sample("c0", 0, cond="h0", cond_len=5),
        _sample("c0", 1, cond="h0", cond_len=5),
        _sample("c1", 0, cond="h1", cond_len=9),
        _sample("c1", 1, cond="h1", cond_len=9),
    ]
    pack = pack_dual_lane_episode(
        _ep(uid="task", turns=2, k=2, samples=samples),
        und_key="task_0_0",
        expected_s=2,
    )
    assert pack.und_record["fields"]["gen_cond_uids"] == ["h0", "h1"]
    assert pack.und_record["fields"]["gen_cond_len"] == 9


def test_und_row_ignores_invalid_samples_and_tolerates_a_missing_uid():
    samples = [
        _sample("c0", 0, cond="h0", cond_len=5),
        _sample("c0", 1, cond="h0", cond_len=5),
        # dropped seed: its (large) length must not become the episode's answer
        _sample("c1", 0, valid=False, cond="h2", cond_len=99),
        # a serving path that reported no uid contributes no identity
        _sample("c1", 1, cond=None, cond_len=0),
    ]
    pack = pack_dual_lane_episode(
        _ep(uid="task", turns=2, k=2, samples=samples),
        und_key="task_0_0",
        expected_s=2,
    )
    assert pack.und_record["fields"]["gen_cond_uids"] == ["h0"]
    assert pack.und_record["fields"]["gen_cond_len"] == 5


# --------------------------------------------------------------------------- #
# RFC §4.4.4 — R2 metrics come from the engine's counters
# --------------------------------------------------------------------------- #
def test_cond_reuse_metrics_amortize_one_encode_over_the_seed_group():
    m = cond_reuse_metrics(calls=4, hits=3, misses=1, bypassed=0)
    assert m["gen/cond_cache_hits"] == 3.0
    assert m["gen/cond_cache_misses"] == 1.0
    # target 1/S: one prefill serves the whole group.
    assert m["gen/cond_recompute_ratio"] == pytest.approx(0.25)
    assert m["gen/prompt_embed_cache_hit_rate"] == pytest.approx(0.75)
    assert m["gen/cond_amortization"] == pytest.approx(4.0)


def test_cond_reuse_metrics_counts_a_bypass_as_a_prefill_but_not_a_cache_decision():
    m = cond_reuse_metrics(calls=4, hits=3, misses=0, bypassed=1)
    assert m["gen/cond_recompute_ratio"] == pytest.approx(0.25)
    # 3 hits / (3 hits + 0 misses): the bypass is excluded from the denominator.
    assert m["gen/prompt_embed_cache_hit_rate"] == pytest.approx(1.0)


def test_cond_reuse_metrics_never_fabricates_a_zero():
    """No counters or no GEN call -> publish nothing, not 0.0 (RFC §8.7)."""
    assert cond_reuse_metrics(calls=0, hits=1, misses=1) == {}
    assert cond_reuse_metrics(calls=1, hits=None, misses=None) == {}
    assert "gen/prompt_embed_cache_hit_rate" not in cond_reuse_metrics(calls=2, hits=0, misses=0)


def test_pack_metrics_carry_exactly_the_measured_r2_names():
    """§4.4.4: the four measured keys, and nothing fabricated when the engine is mute."""
    pack = pack_dual_lane_episode(
        _ep(
            uid="taskA",
            turns=1,
            k=1,
            samples=[_sample("c0", 0, cond="h0", cond_len=3), _sample("c0", 1, cond="h0", cond_len=3)],
            r2=cond_reuse_metrics(calls=2, hits=1, misses=1, bypassed=0),
        ),
        und_key="taskA_0_0",
        expected_s=2,
    )
    r2_keys = {key for key in pack.metrics if key.startswith("gen/cond_") or "prompt_embed_cache" in key}
    assert r2_keys == {
        "gen/cond_cache_hits",
        "gen/cond_cache_misses",
        "gen/cond_cache_bypassed",
        "gen/cond_recompute_ratio",
        "gen/prompt_embed_cache_hit_rate",
        "gen/cond_amortization",
    }
    assert pack.metrics["gen/cond_recompute_ratio"] == pytest.approx(0.5)
    assert pack.metrics["gen/prompt_embed_cache_hit_rate"] == pytest.approx(0.5)
    assert pack.metrics["gen/cond_amortization"] == pytest.approx(2.0)


def _as_step_record(pack):
    """The shape ``aggregate_episode_metrics`` actually receives.

    ``pack_dual_lane_episode`` returns metrics as a sibling of ``und_record``; the
    worker writes them into ``extra_fields["bagel_corl_metrics"]`` on the wire and
    ``OmniBagelCoRLTrainerSync._und_records_from_batch`` flattens ``extra_fields``
    onto the record before aggregating. Mirror that flattening here so this test
    fails if either end stops agreeing on where the blob lives.
    """
    fields = dict(pack.und_record["fields"])
    fields["bagel_corl_metrics"] = dict(pack.metrics)
    return {"fields": fields}


def test_r2_metrics_flow_from_the_episode_into_the_und_row_and_the_step_aggregate():
    pack = pack_dual_lane_episode(
        _ep(
            uid="taskA",
            turns=1,
            k=1,
            samples=[_sample("c0", 0, cond="h0", cond_len=3), _sample("c0", 1, cond="h0", cond_len=3)],
            r2=cond_reuse_metrics(calls=2, hits=1, misses=1, bypassed=0),
        ),
        und_key="taskA_0_0",
        expected_s=2,
    )
    pack2 = pack_dual_lane_episode(
        _ep(
            uid="taskB",
            turns=1,
            k=1,
            samples=[_sample("c1", 0, cond="h1", cond_len=3), _sample("c1", 1, cond="h1", cond_len=3)],
            r2=cond_reuse_metrics(calls=2, hits=2, misses=0, bypassed=0),
        ),
        und_key="taskB_0_0",
        expected_s=2,
    )

    agg = aggregate_episode_metrics([_as_step_record(pack), _as_step_record(pack2)])
    # counts are additive over the batch ...
    assert agg["gen/cond_cache_hits"] == pytest.approx(3.0)
    assert agg["gen/cond_cache_misses"] == pytest.approx(1.0)
    # ... ratios are batch means of the per-episode values, not re-derived from the
    # summed counts (which would read 1/4 = 0.25 for recompute, not 0.25 here only
    # by coincidence of equal weights).
    assert agg["gen/cond_recompute_ratio"] == pytest.approx((0.5 + 0.0) / 2)
    assert agg["gen/prompt_embed_cache_hit_rate"] == pytest.approx((0.5 + 1.0) / 2)


def test_step_aggregate_omits_r2_keys_when_no_episode_measured_any():
    pack = pack_dual_lane_episode(
        _ep(uid="task", turns=1, k=1, samples=[_sample("c0", 0), _sample("c0", 1)]),
        und_key="task_0_0",
        expected_s=2,
    )
    agg = aggregate_episode_metrics([_as_step_record(pack)])
    assert not any("cond_" in key or "prompt_embed_cache" in key for key in agg)


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


def test_gen_rows_are_tagged_auxiliary_so_the_sampler_skips_them():
    """A GEN seed key shares its UND episode's uid prefix but is not a trajectory of it.

    ``ReplayBuffer._materialize_batch`` selects every key whose prefix matches a sampled prompt
    uid, so without ``is_auxiliary`` the seed row lands in the training batch and the UND pass --
    which has no token sequence for it -- returns fewer rows than the batch
    (``assert len(output) == len(batch)``).
    """
    samples = [_sample("c0", 0), _sample("c0", 1)]
    pack = pack_dual_lane_episode(_ep(uid="task", turns=1, k=1, samples=samples), und_key="task_0_0", expected_s=2)

    assert pack.gen_records
    assert all(rec["tag"]["is_auxiliary"] is True for rec in pack.gen_records)
    # The UND episode is the trainable row: it must stay sampleable.
    assert "is_auxiliary" not in pack.und_record["tag"]


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
                "rollout": {"n": 3, "agent": {"gen_samples_per_call": 2, "max_generate_passes": 1, "und_ar_serving_ready": True, "und_deploy_config": "examples/agenticllmgrpo_trainer/bagel/bagel_corl_deploy_ar.yaml"}},
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
                    # RFC §4.0.2 dual-lane LoRA: the rewrite requires the explicit list.
                    "target_modules": list(_BAGEL_LORA_TARGETS),
                },
                "actor": {},
                "rollout": {
                    "name": "vllm_omni",
                    "response_length": 512,
                    "trace": {"backend": None, "token2text": False},
                    "agent": {
                        "default_agent_loop": "bagel_multiturn_agent",
                        # RFC §5 knob SoT: these three have no code default, so the
                        # recipe/yaml must carry them or _rewrite_bagel_corl_configs
                        # fails loud.
                        "gen_samples_per_call": 2,
                        "max_generate_passes": 1,
                        "max_und_turns": 8,
                    },
                },
            }
        }
    )
    trainer._rewrite_bagel_corl_configs()
    agent = trainer.config.actor_rollout_ref.rollout.agent
    assert agent.agent_loop_manager_class.endswith("BagelCorlAgentLoopManagerTQ")


def test_rewrite_fails_loud_on_missing_agent_knobs():
    """RFC §5: gen_samples_per_call / max_generate_passes / max_und_turns have a
    single SoT and no code fallback; a missing knob must raise, not default."""
    import pytest

    from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync

    for missing in ("gen_samples_per_call", "max_generate_passes", "max_und_turns"):
        agent_cfg = {
            "default_agent_loop": "bagel_multiturn_agent",
            "gen_samples_per_call": 2,
            "max_generate_passes": 1,
            "max_und_turns": 8,
        }
        del agent_cfg[missing]
        trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {
                        "path": "/tmp/bagel",
                        "algorithm": "flow_grpo",
                        "composite_mode": "bagel_corl",
                        "architecture": "OmniBagelForConditionalGeneration",
                        # Valid so the failure under test is the missing agent knob.
                        "target_modules": list(_BAGEL_LORA_TARGETS),
                    },
                    "actor": {},
                    "rollout": {
                        "name": "vllm_omni",
                        "response_length": 512,
                        "trace": {"backend": None, "token2text": False},
                        "agent": agent_cfg,
                    },
                }
            }
        )
        with pytest.raises(ValueError, match=missing):
            trainer._rewrite_bagel_corl_configs()


def _reward_episode(*, mask, und_reward=0.25, rm_scores=None):
    """Minimal episode whose reward contract is the thing under test."""
    samples = []
    for index, score in enumerate(rm_scores or []):
        samples.append(
            GenSample(
                gen_group_uid="g0",
                gen_sample_uid=gen_sample_uid("g0", index),
                seed_index=index,
                prompt_token_ids=[1, 2],
                all_latents=torch.zeros(2, 2),
                timesteps=torch.zeros(2),
                rollout_log_probs=torch.zeros(2),
                valid=True,
                rm_score=score,
            )
        )
    return EpisodeRollout(
        und_group_uid="u0",
        episode_uid="u0-ep",
        policy_version=0,
        prompt_ids=[1, 2],
        response_ids=[3] * len(mask),
        response_mask=list(mask),
        rollout_log_probs=[0.0] * len(mask),
        turns=1,
        gen_samples=samples,
        und_reward=und_reward,
    )


def test_und_row_publishes_rm_scores_for_the_v1_advantage():
    """The v1 advantage copies ``rm_scores`` into ``token_level_scores`` unconditionally.

    Measured 2026-09-18 (`hk01dgx012`, devices 4-7), first step after the agent loop finally
    instantiated::

        File "verl/trainer/ppo/v1/trainer_base.py", line 1595, in _compute_advantage
            data.batch["token_level_scores"] = data.batch["rm_scores"]
        KeyError: 'key "rm_scores" not found in TensorDict with keys
        [\'old_log_probs\', \'response_mask\', \'rollout_log_probs\', \'uid\']'

    ``AgentLoopOutput.as_dict`` only emits the column when ``reward_score`` is set, and the UND
    ingest never set it (the loop leaves it None so the reward loop owns it). So the ingest must
    publish the column itself.
    """
    episode = _reward_episode(mask=[1, 1, 1])
    scores = und_row_reward(episode, torch.tensor([1, 1, 1]))
    assert scores.shape == (3,)
    assert float(scores.sum()) == pytest.approx(0.25)


def test_the_und_reward_lands_on_the_last_trainable_token_not_a_masked_tail():
    """A forced turn ends on ``mask_value=0`` tokens; ``as_dict`` parks the score on ``[-1]``.

    That position is masked, so ``compute_advantage_for_multi_trajectories`` drops it and the
    episode silently trains on zero signal. The pinned reward manager uses the last *valid*
    token (``reward_manager/base.py:78-80``); mirror it.
    """
    mask = torch.tensor([1, 1, 0, 0])
    scores = und_row_reward(_reward_episode(mask=[1, 1, 0, 0], und_reward=0.75), mask)
    assert int(torch.nonzero(scores).flatten()[-1]) == 1
    assert float(scores[1]) == pytest.approx(0.75)
    assert float(scores[-1]) == 0.0

    # Mutation: the pinned ``as_dict`` placement is what this replaces.
    as_dict_style = torch.zeros_like(mask, dtype=torch.float32)
    as_dict_style[-1] = 0.75
    assert float((as_dict_style * mask).sum()) == 0.0


def test_the_und_reward_survives_the_flatten_contract():
    """TQ row and flatten row must carry the same number, or UND and GEN see different rewards.

    Both consumers go through ``episode_und_reward``; deriving the mean-of-rated-seeds in two
    places is what let them drift.
    """
    episode = _reward_episode(mask=[1, 1, 1], und_reward=0.1, rm_scores=[0.4, 0.6])
    flattened = flatten_multiturn_rollouts([episode], expected_s=2)
    row = flattened.und_batch[0]
    scores = und_row_reward(episode, torch.tensor(row["response_mask"]))
    assert float(scores.sum()) == float(row["token_level_scores"]) == pytest.approx(0.5)


def test_the_non_image_und_scalar_reaches_both_consumers():
    """Pattern 3 (K=0) has no seed score: the loop's scalar must be what both rows publish."""
    episode = _reward_episode(mask=[1, 1], und_reward=0.3, rm_scores=None)
    flattened = flatten_multiturn_rollouts([episode], expected_s=2)
    row = flattened.und_batch[0]
    scores = und_row_reward(episode, torch.tensor(row["response_mask"]))
    # ``scores`` is float32; compare each side to the exact target rather than to each other.
    assert float(scores.sum()) == pytest.approx(0.3)
    assert float(row["token_level_scores"]) == pytest.approx(0.3)
    assert flattened.metrics["und/no_image_credit"] == pytest.approx(1.0)


def test_an_unrated_and_unscored_episode_publishes_a_zero_but_the_column_exists():
    """``ENABLE_RM=0`` leaves every reward at 0; the column must still be published.

    The step has to reach the optimizer to prove the pipeline works, and the v1 advantage reads
    ``rm_scores`` unconditionally -- a missing column is a crash, a zero column is the documented
    "no scorer wired" state.
    """
    episode = _reward_episode(mask=[1, 1], und_reward=0.0, rm_scores=None)
    scores = und_row_reward(episode, torch.tensor([1, 1]))
    assert scores.shape == (2,)
    assert float(scores.sum()) == 0.0


def test_an_all_masked_und_row_refuses_to_publish_an_invisible_reward():
    episode = _reward_episode(mask=[0, 0])
    with pytest.raises(ValueError, match="no trainable position"):
        und_row_reward(episode, torch.tensor([0, 0]))
