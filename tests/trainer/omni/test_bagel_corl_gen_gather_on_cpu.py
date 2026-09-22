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
"""CPU tests for the per-step dual-lane TransferQueue gather inside ``bagel_corl_sync``.

The trainer never receives GEN seed rows in its batch: it addresses them through the UND episode
row's ``child_gen_keys``. That gather starts from ``transfer_queue.kv_batch_get``, which returns a
**columnar TensorDict** -- one entry per requested key, in request order, keyed by field name. It is
neither a dict nor a keyed mapping, and iterating it yields *field names*. An earlier
``_extra_fields_from_tq`` returned that object as-is, so ``_und_records_from_batch`` built one empty
record per field name, found no ``child_gen_keys`` anywhere, ``build_gen_flowgrpo_proto`` returned
None, and the GEN lane silently never trained while the pack log still reported ``gen/num_rows`` > 0.

These tests pin the gather against a **real** TransferQueue (mocked-return tests cannot catch the
column shape, which is the whole bug) and end at the same ``build_gen_flowgrpo_proto`` the step uses.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl_omni.agent_loop.bagel_corl_tq import put_dual_lane_rows
from verl_omni.trainer.omni.bagel_corl_gen_adv import build_gen_flowgrpo_proto
from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync


@pytest.fixture(scope="module")
def tq_client():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    """A partition per test, so one test's episode rows cannot be seen by the next."""
    return f"bagel-gen-gather-{uuid.uuid4().hex}"


def _und_row(child_gen_keys: list[str]) -> dict:
    """The UND episode row: token sequence to score plus the GEN keys it points at."""
    return {
        "prompts": torch.arange(2),
        "responses": torch.arange(3),
        "input_ids": torch.arange(5),
        "child_gen_keys": list(child_gen_keys),
        "extra_fields": {
            "bagel_role": "und",
            "child_gen_keys": list(child_gen_keys),
            "episode_J": 1,
            "episode_K": 1,
            "bagel_corl_metrics": {
                "episode/J": 1.0,
                "episode/K": 1.0,
                "gen/num_rows": float(len(child_gen_keys)),
                "gen/dropped_incomplete_groups": 0.0,
                "und/no_image_credit": 0.0,
                "gen/skipped_no_groups": 0.0,
            },
        },
    }


def _gen_row(seed: int) -> dict:
    """A rate-diffusion seed row: latents/timesteps, no token sequence for the UND pass."""
    return {
        "gen_group_uid": "call0",
        "gen_sample_uid": f"call0:{seed}",
        "seed_index": seed,
        "rm_score": 0.5,
        "rollout_log_probs": [0.1, 0.2],
        "all_latents": torch.full((2, 4), float(seed)),
        "timesteps": torch.tensor([999.0, 500.0]),
        "prompt_token_ids": [1, 2, 3],
        "extra_fields": {"bagel_role": "gen", "parent_und_key": "task_0_0"},
    }


def _write_episode(partition_id: str, und_key: str, gen_keys: list[str]) -> None:
    """Land one episode through the real dual-lane writer (so tags/schemas are the run's)."""
    asyncio.run(
        put_dual_lane_rows(
            tq,
            keys=[und_key, *gen_keys],
            field_dicts=[_und_row(gen_keys), *(_gen_row(seed) for seed in range(len(gen_keys)))],
            tags=[
                {"status": "success", "bagel_role": "und", "seq_len": 5, "global_steps": 0},
                *(
                    {"status": "success", "bagel_role": "gen", "is_auxiliary": True, "seq_len": 0, "global_steps": 0}
                    for _ in gen_keys
                ),
            ],
            partition_id=partition_id,
        )
    )


def _batch(partition_id: str, und_key: str) -> KVBatchMeta:
    """A sampled batch after ``ReplayBuffer`` dropped the auxiliary lane, as in a real step."""
    return KVBatchMeta(
        partition_id=partition_id,
        keys=[und_key],
        tags=[{"status": "success", "bagel_role": "und", "seq_len": 5, "global_steps": 0}],
    )


def _trainer() -> OmniBagelCoRLTrainerSync:
    return OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)


def test_the_und_gather_reads_the_columnar_extra_fields(tq_client, partition_id):
    und_key = "task_0_0"
    gen_keys = [f"{und_key}::gen::call0::{seed}" for seed in range(2)]
    _write_episode(partition_id, und_key, gen_keys)

    records = _trainer()._und_records_from_batch(_batch(partition_id, und_key))

    assert len(records) == 1
    assert records[0]["fields"]["child_gen_keys"] == gen_keys
    assert records[0]["fields"]["episode_J"] == 1
    assert records[0]["fields"]["bagel_corl_metrics"]["gen/num_rows"] == 2.0


def test_the_gen_lane_reaches_a_flowgrpo_proto(tq_client, partition_id):
    """The step's GEN lane is only real once the gather ends at a non-None proto."""
    und_key = "task_0_0"
    gen_keys = [f"{und_key}::gen::call0::{seed}" for seed in range(2)]
    _write_episode(partition_id, und_key, gen_keys)

    trainer = _trainer()
    batch = _batch(partition_id, und_key)
    gen_rows = trainer._gen_batch_from_step(batch)

    assert [row["seed_index"] for row in gen_rows] == [0, 1]
    assert [row["gen_sample_uid"] for row in gen_rows] == ["call0:0", "call0:1"]
    proto = build_gen_flowgrpo_proto(gen_rows)
    assert proto is not None
    assert "all_latents" in proto.batch.keys()


def test_padding_rows_are_not_counted_as_episodes(tq_client, partition_id):
    """The real padding helper deep-copies a sample's fields *and* tag, ``child_gen_keys`` included.

    Counting a synthetic row as an episode would gather the template's GEN rows twice and fold its
    J/K into the batch mean a second time, so the UND gather has to drop rows tagged ``is_padding``.
    """
    from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size

    und_key = "task_0_0"
    gen_keys = [f"{und_key}::gen::call0::0"]
    _write_episode(partition_id, und_key, gen_keys)

    padded = upsample_batch_to_divisible_size(_batch(partition_id, und_key), 2, eos_token_id=0)

    assert len(padded) == 2
    assert padded.tags[1]["is_padding"] is True
    # Why the filter has to exist: the synthetic row inherits its template's ``child_gen_keys``.
    pad_fields = tq.kv_batch_get(
        keys=[padded.keys[1]], partition_id=partition_id, select_fields=["extra_fields", "child_gen_keys"]
    )
    assert list(pad_fields["extra_fields"])[0]["child_gen_keys"] == gen_keys
    assert list(pad_fields["child_gen_keys"])[0] == gen_keys

    records = _trainer()._und_records_from_batch(padded)
    assert len(records) == 1
    assert records[0]["fields"]["child_gen_keys"] == gen_keys
    # ... so counting it would gather the same seed row twice.
    assert [row["gen_sample_uid"] for row in _trainer()._gen_batch_from_step(padded)] == ["call0:0"]


def test_a_gen_row_shares_its_und_uid_so_it_is_selectable(tq_client, partition_id):
    """Mutation guard for the sampler-side filter: the GEN row is a *TQ row under the same uid*.

    ``ReplayBuffer._materialize_batch`` matches by ``key.split("_")[0]``, so the seed key is
    selectable; scoring it needs a token sequence it does not have. Assert the shape that makes
    ``assert len(output) == len(batch)`` fail -- i.e. the batch would carry a row with no
    ``responses`` -- so the ``is_auxiliary`` tag in the writer is load-bearing.
    """
    und_key = "task_0_0"
    gen_keys = [f"{und_key}::gen::call0::0"]
    _write_episode(partition_id, und_key, gen_keys)

    assert gen_keys[0].split("_")[0] == und_key.split("_")[0]
    gen_row = tq.kv_batch_get(keys=gen_keys, partition_id=partition_id, select_fields=["all_latents"])
    assert "responses" not in gen_row.keys()


class _ColumnCarrier:
    """The TensorDict shape the v1 advantage phase can be handed instead of a ``KVBatchMeta``.

    It has no ``partition_id``, so ``_extra_fields_from_tq`` returns ``None`` for it and the TQ
    re-fetch is impossible -- the dual-lane bookkeeping has to be read off ``non_tensor_batch``.
    """

    def __init__(self, extra_fields: list[dict], child_gen_keys=None, tags=None):
        self.non_tensor_batch = {"extra_fields": extra_fields}
        if child_gen_keys is not None:
            self.non_tensor_batch["child_gen_keys"] = child_gen_keys
        self.tags = tags or []


def test_a_tensordict_carrier_is_read_from_its_extra_fields_column():
    """No TQ handle (no ``partition_id``) must not mean "no GEN lane"."""
    gen_keys = ["task_0_0::gen::call0::0", "task_0_0::gen::call0::1"]
    fields = _und_row(gen_keys)["extra_fields"]
    records = _trainer()._und_records_from_batch(_ColumnCarrier([fields]))

    assert len(records) == 1
    assert records[0]["fields"]["child_gen_keys"] == gen_keys
    assert records[0]["fields"]["episode_K"] == 1


def test_the_child_gen_keys_column_is_reattached_when_the_blob_omits_it():
    """``extra_fields`` and ``child_gen_keys`` are sibling columns; the column must survive.

    The dual-lane writer stores the keys twice (inside the blob and as its own field). A reader that
    returns only the first column loses the lane for any row whose blob does not repeat them, which
    is exactly what a partial/legacy blob looks like.
    """
    gen_keys = ["task_0_0::gen::call0::0", "task_0_0::gen::call0::1"]
    blob = dict(_und_row(gen_keys)["extra_fields"])
    blob.pop("child_gen_keys")

    records = _trainer()._und_records_from_batch(_ColumnCarrier([blob], child_gen_keys=[gen_keys]))

    assert records[0]["fields"]["child_gen_keys"] == gen_keys

    assert OmniBagelCoRLTrainerSync._merge_child_gen_keys({"episode_K": 1}, gen_keys) == {
        "episode_K": 1,
        "child_gen_keys": gen_keys,
    }
    # A row that already carries the keys keeps its own value (no cross-row bleed).
    assert OmniBagelCoRLTrainerSync._merge_child_gen_keys({"child_gen_keys": ["mine"]}, gen_keys) == {
        "child_gen_keys": ["mine"]
    }
