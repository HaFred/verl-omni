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
"""CPU tests for RFC §4.4.2b/§4.4.4 in the GEN path of ``BagelMultiturnAgentLoop``.

Drives the real ``_generate_image`` with a fake server manager that emulates
``PromptEmbedCache`` (first request for a conditioning misses, the rest hit), so
the tests cover the two things that are easy to get subtly wrong:

* the **sticky key** — one request id for the whole S-group, because the client
  acquires its replica from the load balancer with that id and a per-seed
  ``uuid4()`` would scatter the group across replicas that hold no entry;
* the **denominator** of ``gen/cond_recompute_ratio`` — the engine counts one
  hit/miss per *generate request*, i.e. ``S`` per ``generate_image`` turn. Counting
  one call per turn would report ``1`` ("no reuse") on a perfectly warm cache.
"""

from __future__ import annotations

import asyncio

import pytest
from omegaconf import OmegaConf

from verl_omni.agent_loop import bagel_corl as loop_mod
from verl_omni.agent_loop.bagel_corl import BagelMultiturnAgentLoop
from verl_omni.agent_loop.bagel_corl_lib import EpisodeRollout, cond_reuse_metrics


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        _ = add_special_tokens
        return [7, 8, 9]


class _FakeServerManager:
    """Emulates the engine's conditioning cache, at request granularity."""

    def __init__(self) -> None:
        self.request_ids: list[str] = []
        self.prompts: list[list[int]] = []
        self.roles: list[str] = []
        self._counters = {"hits": 0, "misses": 0, "bypassed": 0}
        self._warm: set[tuple[int, ...]] = set()

    async def generate(self, *, request_id: str, prompt_ids, sampling_params):
        self.request_ids.append(request_id)
        self.prompts.append(list(prompt_ids))
        self.roles.append(str(sampling_params.get("bagel_role", "")))
        key = tuple(int(t) for t in prompt_ids)
        if key in self._warm:
            self._counters["hits"] += 1
        else:
            self._warm.add(key)
            self._counters["misses"] += 1
        return {"stub": True}

    async def prompt_embed_cache_stats(self, routing_key: str):
        # The real client routes by key; the fake serves one replica, so the key is
        # only recorded for the stickiness assertions above.
        _ = routing_key
        return dict(self._counters)


def _loop(*, affinity: bool, server: _FakeServerManager) -> BagelMultiturnAgentLoop:
    loop = BagelMultiturnAgentLoop.__new__(BagelMultiturnAgentLoop)
    loop.tokenizer = _FakeTokenizer()
    loop.config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"calculate_log_probs": True}}},
    )
    loop.server_manager = server
    loop._sampling_params = {"global_steps": 3}
    # ``None`` disables the timing accumulator (it must be pre-seeded with gen_s).
    loop._bagel_timing = None
    loop._bagel_cond_affinity = affinity
    # Mirrors production: ``None`` = unmeasured until a probe answers.
    loop._bagel_r2 = {"calls": 0, "hits": None, "misses": None, "bypassed": None}
    return loop


def _stub_gen_serve(monkeypatch) -> None:
    monkeypatch.setattr(loop_mod, "normalize_token_ids", lambda ids: [int(t) for t in ids])
    monkeypatch.setattr(
        loop_mod,
        "build_gen_sampling_params",
        lambda rollout, *, base=None, seed=None: {"seed": seed, "bagel_role": "gen"},
    )
    monkeypatch.setattr(
        loop_mod,
        "stash_gen_row_from_diffusion_output",
        lambda output, *, seed: {"scene_stub": output, "seed": seed},
    )


def test_generate_image_pins_the_whole_s_group_to_one_replica(monkeypatch):
    """R2's premise: S seeds share a conditioning, so they must share a replica."""
    _stub_gen_serve(monkeypatch)
    server = _FakeServerManager()
    loop = _loop(affinity=True, server=server)

    rows = asyncio.run(loop._generate_image(prompt="a cat", seeds=[0, 1, 2, 3], gen_call_id="call7"))

    assert len(rows) == 4
    assert len(set(server.request_ids)) == 1, "one sticky id for the S-group, not a uuid per seed"
    assert server.request_ids[0].startswith("bagel_cond::")
    assert "call7" in server.request_ids[0]
    # Same conditioning slice submitted four times -> that is the reuse group.
    assert all(prompt == [7, 8, 9] for prompt in server.prompts)
    assert all(role == "gen" for role in server.roles)


def test_generate_image_counts_requests_not_turns_in_the_ratio_denominator(monkeypatch):
    """A warm cache must read ``1/S``, not ``1``: the denominator is generate requests."""
    _stub_gen_serve(monkeypatch)
    server = _FakeServerManager()
    loop = _loop(affinity=True, server=server)

    asyncio.run(loop._generate_image(prompt="a cat", seeds=[0, 1, 2, 3], gen_call_id="call7"))

    # 4 requests, 1 cold prefill (the miss) + 3 hits — exactly what the engine counts.
    assert loop._bagel_r2 == {"calls": 4, "hits": 3, "misses": 1, "bypassed": 0}
    metrics = cond_reuse_metrics(**loop._bagel_r2)
    assert metrics["gen/cond_recompute_ratio"] == 0.25
    assert metrics["gen/prompt_embed_cache_hit_rate"] == 0.75
    assert metrics["gen/cond_amortization"] == 4.0


def test_generate_image_tags_every_row_with_the_same_conditioning_identity(monkeypatch):
    _stub_gen_serve(monkeypatch)
    loop = _loop(affinity=True, server=_FakeServerManager())

    rows = asyncio.run(loop._generate_image(prompt="a cat", seeds=[0, 1], gen_call_id="call7"))

    assert {row["cond_uid"] for row in rows} == {rows[0]["cond_uid"]}
    assert rows[0]["cond_uid"]
    assert [row["cond_len"] for row in rows] == [3, 3]


def test_two_calls_with_the_same_prompt_share_a_conditioning_uid(monkeypatch):
    """The key is content-derived, so it is stable across episodes/calls."""
    _stub_gen_serve(monkeypatch)
    server = _FakeServerManager()
    loop = _loop(affinity=True, server=server)

    first = asyncio.run(loop._generate_image(prompt="a cat", seeds=[0], gen_call_id="a"))
    second = asyncio.run(loop._generate_image(prompt="a cat", seeds=[0], gen_call_id="b"))

    assert first[0]["cond_uid"] == second[0]["cond_uid"]
    # ... but the sticky key is scoped per call, so call b does not land on call a's pin.
    assert server.request_ids[0] != server.request_ids[1]


def test_affinity_off_keeps_per_seed_ids_but_still_counts_calls(monkeypatch):
    """The counter is independent of measurement: ``calls`` is requests either way."""
    _stub_gen_serve(monkeypatch)
    server = _FakeServerManager()
    loop = _loop(affinity=False, server=server)

    asyncio.run(loop._generate_image(prompt="a cat", seeds=[0, 1, 2], gen_call_id="call7"))

    assert len(set(server.request_ids)) == 3
    assert loop._bagel_r2["calls"] == 3
    # No pin -> no replica to probe -> no counters -> no fabricated metrics.
    assert cond_reuse_metrics(**loop._bagel_r2) == {}


def test_a_zero_hit_rate_is_still_published_when_it_was_actually_measured(monkeypatch):
    """The inverse of the trap: a real 0% hit rate is a measurement and must show up."""
    _stub_gen_serve(monkeypatch)

    class _NeverHits(_FakeServerManager):
        async def generate(self, *, request_id, prompt_ids, sampling_params):
            self.request_ids.append(request_id)
            self._counters["misses"] += 1
            return {"stub": True}

    server = _NeverHits()
    loop = _loop(affinity=True, server=server)

    asyncio.run(loop._generate_image(prompt="a cat", seeds=[0, 1], gen_call_id="call7"))

    metrics = cond_reuse_metrics(**loop._bagel_r2)
    assert metrics["gen/cond_recompute_ratio"] == 1.0
    assert metrics["gen/prompt_embed_cache_hit_rate"] == 0.0
    assert "gen/cond_amortization" not in metrics or metrics["gen/cond_amortization"] == 1.0


def test_cache_probe_failure_degrades_to_no_measurement(monkeypatch):
    """Metrics must never break a rollout — and a dead probe must not read as perfect."""
    _stub_gen_serve(monkeypatch)
    server = _FakeServerManager()

    async def _boom(routing_key: str):
        raise RuntimeError("replica went away")

    server.prompt_embed_cache_stats = _boom
    loop = _loop(affinity=True, server=server)

    rows = asyncio.run(loop._generate_image(prompt="a cat", seeds=[0, 1], gen_call_id="call7"))

    assert len(rows) == 2
    assert loop._bagel_r2 == {"calls": 2, "hits": None, "misses": None, "bypassed": None}
    # The trap: 0/0 counters would publish cond_recompute_ratio == 0.0, the *best*
    # possible score, for an episode that measured nothing (§8.7 silent pass).
    assert cond_reuse_metrics(**loop._bagel_r2) == {}


def test_generate_image_without_a_prompt_or_tokens_fails_loud(monkeypatch):
    """No silent empty conditioning: an empty slice has no identity to key on."""
    _stub_gen_serve(monkeypatch)
    loop = _loop(affinity=True, server=_FakeServerManager())

    with pytest.raises(ValueError, match="non-empty diffusion prompt"):
        asyncio.run(loop._generate_image(prompt="", seeds=[0], gen_call_id="call7"))

    monkeypatch.setattr(loop_mod, "normalize_token_ids", lambda ids: [])
    with pytest.raises(ValueError, match="non-empty tokenized diffusion prompt"):
        asyncio.run(loop._generate_image(prompt="a cat", seeds=[0], gen_call_id="call7"))


def _episode(**overrides) -> EpisodeRollout:
    kwargs = {
        "und_group_uid": "task",
        "episode_uid": "task-ep",
        "policy_version": 1,
        "prompt_ids": [1],
        "response_ids": [2, 3],
        "response_mask": [1, 1],
        "turns": 1,
    }
    kwargs.update(overrides)
    return EpisodeRollout(**kwargs)


def test_und_rollout_log_probs_reach_the_tq_field_the_trainer_reads():
    """π_rollout has to end up on the UND row the trainer pairs with ``old_log_probs``.

    The recipe runs ``rollout.calculate_log_probs=True``, so ``_compute_old_log_prob`` selects
    ``rollout_log_probs`` off the TQ (``trainer_base.py:1506``). ``AgentLoopOutput.as_dict`` is
    what the TQ worker writes the UND row from, and it maps ``response_logprobs`` onto that
    field (``agent_loop.py:124``) — the measured failure was

        KeyError: 'rollout_log_probs'

    because the loop hard-coded ``response_logprobs=None``.
    """
    output = loop_mod.episode_to_agent_output(_episode(rollout_log_probs=[-0.25, -0.5]))

    assert output.response_logprobs == pytest.approx([-0.25, -0.5])
    published = output.as_dict()
    assert published["rollout_log_probs"].tolist() == pytest.approx([-0.25, -0.5])
    assert published["rollout_log_probs"].shape[0] == published["responses"].shape[0]


def test_an_episode_that_sampled_nothing_publishes_no_rollout_log_probs():
    """No measurement must stay absent, not become a zero log-prob the trainer trusts."""
    published = loop_mod.episode_to_agent_output(_episode()).as_dict()
    assert "rollout_log_probs" not in published


def test_the_bagel_agent_loop_is_what_gets_registered():
    """The registry must point at the *class*, not at whatever sits under ``@register``.

    ``register`` builds its target from the decorated object's ``__qualname__``
    (``verl/experimental/agent_loop/agent_loop.py:486-491``), so inserting a module-level helper
    between the decorator and the class silently registers that helper and leaves the class
    unregistered. Measured 2026-09-18 on `hk01dgx012` (devices 4-7): every prompt then died in the
    worker with

        hydra.errors.InstantiationException: Error in call to target
        'verl_omni.agent_loop.bagel_corl.episode_to_agent_output':
        TypeError: episode_to_agent_output() got an unexpected keyword argument 'trainer_config'

    (the worker passes ``trainer_config``/``server_manager``/``tokenizer``/... into whatever the
    registry names), and the step surfaced as ``Sync replay buffer selected terminal groups with
    no materializable trajectories`` because no episode ever ran.
    """
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

    target = _agent_loop_registry["bagel_multiturn_agent"]["_target_"]
    assert target == "verl_omni.agent_loop.bagel_corl.BagelMultiturnAgentLoop"
    assert target.rsplit(".", 1)[-1] == BagelMultiturnAgentLoop.__qualname__
