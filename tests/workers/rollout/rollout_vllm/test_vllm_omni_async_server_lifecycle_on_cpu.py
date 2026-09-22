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
"""Async rollout server lifecycle contract.

Driven by a mocked ``AsyncOmni`` matching the pinned engine API: ``abort``
takes EXTERNAL ids in one batched acked call; ``pause_generation`` never
delivers tokens; diffusion sleep/wake ACKs are not engine-checked.
"""

import asyncio
from types import SimpleNamespace

import pytest

server_module = pytest.importorskip("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server")

from verl.workers.rollout.replica import RolloutMode  # noqa: E402

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy  # noqa: E402
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer  # noqa: E402
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy  # noqa: E402
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base import OmniStrategyBase  # noqa: E402

_SUCCESS_ACK = SimpleNamespace(status="SUCCESS")


class _FakeRequestState:
    def __init__(self, request_id: str, external_request_id: str):
        self.request_id = request_id
        self.external_request_id = external_request_id
        self.queue: asyncio.Queue = asyncio.Queue()


def _terminal(request_id: str, token_ids: list[int]):
    """Engine-side abort terminal: real cumulative partial tokens."""
    output = SimpleNamespace(finish_reason="abort", token_ids=list(token_ids))
    return SimpleNamespace(request_id=request_id, engine_outputs=SimpleNamespace(outputs=[output]), finished=True)


class _FakeAsyncOmni:
    def __init__(self, *, states=None, fail_abort=False, sleep_acks=None, wake_acks=None):
        self.request_states: dict[str, _FakeRequestState] = dict(states or {})
        self.fail_abort = fail_abort
        self.sleep_acks = sleep_acks if sleep_acks is not None else [_SUCCESS_ACK]
        self.wake_acks = wake_acks if wake_acks is not None else [_SUCCESS_ACK]
        self.calls: list[str] = []
        self.abort_calls: list = []
        self.pause_calls: list[dict] = []
        self.sleep_calls: list[dict] = []
        self.wake_calls: list[dict] = []
        self.resumed = 0
        # Real-shaped handle: the frontend multimodal cache lives on the
        # renderer, so the fake must expose clear_mm_cache_async there.
        self.mm_clears = 0

        async def _clear_mm_cache_async():
            self.mm_clears += 1

        self.renderer = SimpleNamespace(clear_mm_cache_async=_clear_mm_cache_async)

    async def abort(self, request_ids):
        self.calls.append("abort")
        self.abort_calls.append(request_ids if isinstance(request_ids, list) else [request_ids])
        if self.fail_abort:
            raise RuntimeError("abort rpc failed")
        ids = request_ids if isinstance(request_ids, list) else [request_ids]
        for state in self.request_states.values():
            if state.external_request_id in ids:
                state.queue.put_nowait(_terminal(state.request_id, token_ids=[7, 8, 9]))

    async def pause_generation(self, **kwargs):
        self.calls.append("pause")
        self.pause_calls.append(kwargs)

    async def sleep(self, stage_ids=None, level=2, mode="abort"):
        self.calls.append("sleep")
        self.sleep_calls.append({"stage_ids": stage_ids, "level": level, "mode": mode})
        return self.sleep_acks

    async def wake_up(self, stage_ids=None, tags=None):
        self.calls.append("wake_up")
        self.wake_calls.append({"stage_ids": stage_ids, "tags": tags})
        return self.wake_acks

    async def resume_generation(self, stage_ids=None):
        self.resumed += 1


def _make_server(engine, rollout_mode=RolloutMode.HYBRID, node_rank=0, free_cache_engine=True):
    # HYBRID default: release/resume_kv_cache are skipped in COLOCATED mode
    # (parent semantics), so their delegation is exercised via HYBRID.
    server = object.__new__(vLLMOmniHttpServer)
    server.engine = engine
    server.node_rank = node_rank
    server.rollout_mode = rollout_mode
    server.config = SimpleNamespace(free_cache_engine=free_cache_engine)
    server._lora_request_cache = None  # a valid cached value
    return server


# ---------------------------------------------------------------------------
# abort-then-pause ordering; single batched abort; no empty-list abort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_runs_before_pause_with_one_batched_call():
    states = {
        "ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1"),
        "ext-1-def": _FakeRequestState("ext-1-def", "ext-1"),  # same external id
        "ext-2-xyz": _FakeRequestState("ext-2-xyz", "ext-2"),
    }
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    result = await server.abort_all_requests()

    # Deduped EXTERNAL ids (request_states keys are internal ids).
    assert engine.abort_calls == [["ext-1", "ext-2"]]
    assert engine.calls == ["abort", "pause"], "abort must run while generate is live, pause after"
    assert engine.pause_calls == [{"mode": "abort", "wait_for_inflight_requests": False, "clear_cache": True}]
    assert result == {"aborted_count": 2, "request_ids": ["ext-1", "ext-2"]}


@pytest.mark.asyncio
async def test_abort_with_no_in_flight_requests_still_pauses():
    engine = _FakeAsyncOmni(states={})
    server = _make_server(engine)

    result = await server.abort_all_requests()

    # The engine returns immediately on an empty id list.
    assert engine.abort_calls == [[]]
    assert engine.calls == ["abort", "pause"]
    assert result == {"aborted_count": 0, "request_ids": []}


# ---------------------------------------------------------------------------
# abort outputs arrive via the request queues; mapper untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_outputs_come_from_engine_queues_with_non_empty_tokens():
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    await server.abort_all_requests()

    terminal = states["ext-1-abc"].queue.get_nowait()
    output = terminal.engine_outputs.outputs[0]
    assert output.token_ids == [7, 8, 9], "cumulative partial tokens must survive the abort"
    assert output.finish_reason == "abort"
    assert OmniStrategyBase._map_stop_reason(output.finish_reason) == "aborted"
    # The engine owns success-path terminals; the server synthesizes only on failure.
    assert states["ext-1-abc"].queue.qsize() == 0


# ---------------------------------------------------------------------------
# all four lifecycle methods delegate, validate ACKs, level=1, keyword tags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sleep_and_wake_delegate_with_level_one_and_keyword_tags():
    for mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
        engine = _FakeAsyncOmni()
        server = _make_server(engine, rollout_mode=mode)
        server._lora_request_cache = None

        await server.sleep()
        await server.wake_up()

        # Uniform delegation in BOTH modes (the old code split hybrid vs
        # colocated between raw RPC and engine.sleep).
        assert [c for c in engine.calls if c in ("sleep", "wake_up")] == ["sleep", "wake_up"]
        assert engine.sleep_calls == [{"stage_ids": None, "level": 1, "mode": "abort"}]
        # Keyword tags only: a positional call would have bound to stage_ids.
        assert engine.wake_calls == [{"stage_ids": None, "tags": ["weights"]}]
        assert server._lora_request_cache is server_module._LORA_REQUEST_CACHE_MISS


@pytest.mark.asyncio
async def test_release_and_resume_kv_cache_route_both_halves_through_delegation():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)

    await server.release_kv_cache()
    await server.resume_kv_cache()

    assert [c for c in engine.calls if c in ("sleep", "wake_up")] == ["sleep", "wake_up", "wake_up"]
    assert engine.sleep_calls[0]["level"] == 1
    assert engine.wake_calls == [
        {"stage_ids": None, "tags": ["weights"]},  # release: keep weights awake
        {"stage_ids": None, "tags": ["kv_cache"]},  # resume: restore the cache
    ]


@pytest.mark.asyncio
async def test_lifecycle_guards_skip_standalone_and_non_driver_ranks():
    engine = _FakeAsyncOmni()
    server = _make_server(engine, rollout_mode=RolloutMode.STANDALONE)
    await server.sleep()
    await server.wake_up()
    assert engine.calls == []

    engine = _FakeAsyncOmni()
    server = _make_server(engine, node_rank=1)
    await server.sleep()
    await server.wake_up()
    await server.release_kv_cache()
    await server.resume_kv_cache()
    assert engine.calls == []


# ---------------------------------------------------------------------------
# mm cache drift: EngineCore.sleep wipes only the engine-side multimodal
# cache, so every successful sleep must also clear the frontend copy through
# the renderer (the engine's reset_mm_cache no-ops on a missing attribute)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sleep_and_release_kv_cache_clear_frontend_mm_sender_cache():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)

    await server.sleep()
    assert engine.sleep_calls[0]["level"] == 1
    assert engine.mm_clears == 1

    await server.release_kv_cache()
    assert engine.mm_clears == 2


@pytest.mark.asyncio
async def test_frontend_mm_clear_skipped_when_sleep_acks_fail():
    engine = _FakeAsyncOmni(sleep_acks=[SimpleNamespace(status="FAILED", error_msg="boom")])
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="sleep failed"):
        await server.sleep()

    assert engine.mm_clears == 0


# A healthy ``sleep`` comes back as the platform-audit *dict* below, not a typed ack.
# Rejecting every dict aborted the first weight sync after sampling even though the
# stage had freed 49.45 GiB. Shape copied from the measured run: hk01dgx012,
# 2026-09-18 03:35 (devices 4-7), stage 0, rank 0, just after ``Training Progress: 0%``.
_AUDIT_SUCCESS_ACK = {
    "task_id": "8bd58260-0826-4bcb-bcc3-08152896f7c1",
    "status": "SUCCESS",
    "stage_id": 0,
    "rank": 0,
    "freed_bytes": 53097791488,
    "metadata": {"source": "omni_platform_audit", "total_freed_gib": "49.45", "rank_residual_gib": "6.35"},
    "error_msg": None,
}


@pytest.mark.asyncio
async def test_dict_success_ack_is_accepted():
    """A dict ack reporting SUCCESS must not be rejected merely for being a dict."""
    engine = _FakeAsyncOmni(sleep_acks=[dict(_AUDIT_SUCCESS_ACK)])
    server = _make_server(engine)

    await server.sleep()
    assert engine.mm_clears == 1


@pytest.mark.asyncio
async def test_dict_ack_carrying_an_error_still_fails_closed():
    """Removing the blanket dict rejection must not weaken fail-closed."""
    for bad in (
        {**_AUDIT_SUCCESS_ACK, "error_msg": "cumem pool corrupted"},
        {**_AUDIT_SUCCESS_ACK, "status": "FAILED"},
    ):
        engine = _FakeAsyncOmni(sleep_acks=[bad])
        server = _make_server(engine)

        with pytest.raises(RuntimeError, match="sleep failed on a stage"):
            await server.sleep()
        assert engine.mm_clears == 0


@pytest.mark.asyncio
async def test_sleep_skips_frontend_mm_clear_when_renderer_is_none():
    # Diffusion-only engines build no InputProcessor, so renderer is None.
    engine = _FakeAsyncOmni()
    engine.renderer = None
    server = _make_server(engine)

    await server.sleep()
    await server.release_kv_cache()
    assert engine.mm_clears == 0
    assert engine.sleep_calls[0]["level"] == 1


@pytest.mark.asyncio
async def test_abort_pause_clears_frontend_mm_sender_cache():
    engine = _FakeAsyncOmni(states={"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")})
    server = _make_server(engine)

    await server.abort_all_requests()

    # pause_generation(clear_cache=True) wipes the engine-side mm cache; the
    # frontend copy must go with it or hash-only follow-ups finish empty.
    assert engine.pause_calls[0]["clear_cache"] is True
    assert engine.mm_clears == 1


@pytest.mark.asyncio
async def test_abort_skips_frontend_mm_clear_without_cache_reset():
    engine = _FakeAsyncOmni(states={"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")})
    server = _make_server(engine)

    await server.abort_all_requests(reset_prefix_cache=False)

    assert engine.pause_calls[0]["clear_cache"] is False
    assert engine.mm_clears == 0


@pytest.mark.asyncio
async def test_frontend_mm_clear_skipped_when_pause_fails():
    class _PauseFailsEngine(_FakeAsyncOmni):
        async def pause_generation(self, **kwargs):
            self.calls.append("pause")
            self.pause_calls.append(kwargs)
            raise RuntimeError("pause rpc failed")

    engine = _PauseFailsEngine()
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="pause rpc failed"):
        await server.abort_all_requests()

    assert engine.mm_clears == 0


@pytest.mark.asyncio
async def test_ack_validation_fails_closed():
    bad_acks = [
        # diffusion worker error dict (collective_rpc error shape)
        [{"supported": False, "error": "diffusion stage failed"}, _SUCCESS_ACK],
        # dict without an error key is still a failure
        [{"supported": False}],
        # diffusion OmniACK failure
        [SimpleNamespace(status="ERROR", error_msg="camem pool corrupted")],
        # mixed-stage: AR raise happens engine-side; the dict/ACK mix must still raise here
        [_SUCCESS_ACK, {"error": "late diffusion failure"}],
    ]
    for acks in bad_acks:
        engine = _FakeAsyncOmni(sleep_acks=acks)
        server = _make_server(engine)
        with pytest.raises(RuntimeError, match="failed on a stage"):
            await server.sleep()

    # Empty ACK list is success ("already warm").
    engine = _FakeAsyncOmni(wake_acks=[])
    server = _make_server(engine)
    await server.wake_up()
    assert engine.wake_calls == [{"stage_ids": None, "tags": ["weights"]}]


@pytest.mark.asyncio
async def test_engine_side_rpc_failure_propagates_from_all_four_methods():
    class _RaisingEngine(_FakeAsyncOmni):
        async def sleep(self, **kwargs):
            raise RuntimeError("AR EngineCore control methods re-raise")

        async def wake_up(self, **kwargs):
            raise RuntimeError("AR EngineCore control methods re-raise")

    engine = _RaisingEngine()
    server = _make_server(engine)
    for method in (server.sleep, server.wake_up, server.release_kv_cache, server.resume_kv_cache):
        with pytest.raises(RuntimeError, match="re-raise"):
            await method()


# ---------------------------------------------------------------------------
# fail-closed abort: enqueue terminals first, then raise; timeout raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_failure_enqueues_terminals_then_raises():
    states = {
        "ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1"),
        "ext-2-xyz": _FakeRequestState("ext-2-xyz", "ext-2"),
    }
    engine = _FakeAsyncOmni(states=states, fail_abort=True)
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="abort rpc failed"):
        await server.abort_all_requests()

    for state in states.values():
        terminal = state.queue.get_nowait()
        assert terminal.finished is True
        assert terminal.engine_outputs.outputs[0].finish_reason == "abort"


@pytest.mark.asyncio
async def test_pause_failure_after_successful_abort_does_not_double_enqueue():
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)

    async def failing_pause(**kwargs):
        raise RuntimeError("pause rpc failed")

    engine.pause_generation = failing_pause
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="pause rpc failed"):
        await server.abort_all_requests()

    # Exactly the engine's real terminal — no synthetic appended after it.
    assert states["ext-1-abc"].queue.qsize() == 1
    assert states["ext-1-abc"].queue.get_nowait().engine_outputs.outputs[0].token_ids == [7, 8, 9]


@pytest.mark.asyncio
async def test_abort_ack_timeout_raises(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "0.05")

    class _HangingEngine(_FakeAsyncOmni):
        async def abort(self, request_ids):
            await asyncio.sleep(999)

    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    server = _make_server(_HangingEngine(states=states))

    with pytest.raises(asyncio.TimeoutError):
        await server.abort_all_requests()

    for state in states.values():
        assert state.queue.get_nowait().finished is True  # terminal still enqueued


# ---------------------------------------------------------------------------
# single-request abort shares the contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_request_aborts_single_id_without_pausing():
    """Per-request abort must not pause the whole engine (pause_scheduler
    finishes ALL in-flight requests); pausing belongs to abort_all_requests."""
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    result = await server.abort_request("ext-1")

    assert engine.abort_calls == [["ext-1"]]  # external id, not internal
    assert engine.calls == ["abort"], "single-request abort must not pause the engine"
    assert states["ext-1-abc"].queue.qsize() == 1  # engine terminal only
    assert result == {"aborted": True, "request_id": "ext-1"}


@pytest.mark.asyncio
async def test_abort_request_unknown_id_is_a_noop():
    engine = _FakeAsyncOmni(states={})
    server = _make_server(engine)

    result = await server.abort_request("missing")

    assert engine.abort_calls == [["missing"]]
    assert engine.calls == ["abort"]
    assert result == {"aborted": True, "request_id": "missing"}


# ---------------------------------------------------------------------------
# the manager-level gather observes the server's raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_manager_gather_propagates_server_abort_raise():
    class _Replica:
        def __init__(self, server):
            self._server = server

        async def abort_all_requests(self):
            # Same seam CheckpointEngineManager.abort_replicas drives:
            # await asyncio.gather(*[r.abort_all_requests() for r in replicas])
            return await self._server.abort_all_requests()

    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    server = _make_server(_FakeAsyncOmni(states=states, fail_abort=True))

    with pytest.raises(RuntimeError, match="abort rpc failed"):
        await asyncio.gather(_Replica(server).abort_all_requests())


# ---------------------------------------------------------------------------
# diffusion-only sleep/wake contract — why diffusion v1 sync needs no
# resume bridge (drives the REAL AsyncOmni state machine, not a mock)
# ---------------------------------------------------------------------------


def _build_async_omni(stage_type: str):
    """Drive the real AsyncOmni state machine; only RPC plumbing is stubbed.

    ``stage_type`` selects a pure-diffusion ("diffusion") or pure-AR ("llm")
    single-stage engine.
    """
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    ack = SimpleNamespace(status="SUCCESS")
    engine_rpc: list[str] = []

    async def _engine_core_rpc(method, stage_ids=None, args=(), kwargs=None):
        engine_rpc.append(method)
        return []

    async def _acks(*args, **kwargs):
        return [ack]

    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(stage_clients=[SimpleNamespace(stage_type=stage_type)])
    omni._pause_cond = asyncio.Condition()
    omni._admitting = 0
    omni._paused = False
    omni._hold_admission_until_resume = False
    omni._level2_sleeping = False
    omni._sleeping_tags = {"weights"}
    omni._stage_sleeping_tags = {}
    # RPC plumbing stubs — not the contract under test.
    omni.reset_mm_cache = _acks
    omni._final_output_handler = lambda: None
    omni._sleep_diffusion = _acks
    omni._wake_diffusion = _acks
    omni._record_stage_sleep = lambda stage_ids, tags: None
    omni._sleeping_tags_for_stages = lambda stage_ids: {"weights"}
    omni._clear_stage_sleep = lambda stage_ids, tags: None
    omni._engine_core_rpc = _engine_core_rpc
    omni._engine_rpc_log = engine_rpc
    return omni


# ``AsyncOmni``-side control plane moved on: the pinned engine now sizes
# its ACK fan-out from ``engine.stage_vllm_configs[sid].parallel_config``, gates
# on ``event_resolver``, and reaches the engine through
# ``engine.collective_rpc_async`` — none of which this fake models (it still stubs
# the superseded ``_engine_core_rpc``). Both cases were unmarked ``async def``, so
# they never actually ran and the drift went unnoticed. Left unmarked-drift rather
# than half-fixed: they assert on upstream ``AsyncOmni.sleep``/``wake_up``
# admission semantics, not on ``verl_omni``, so the faithful fix is to re-derive
# the fake against the pinned engine, not to paper over it with more stubs.
_ASYNC_OMNI_CONTROL_PLANE_DRIFT = pytest.mark.skip(
    reason="fake AsyncOmni predates the pinned control plane (stage_vllm_configs / "
    "event_resolver / collective_rpc_async); re-derive the fake to re-enable"
)


@pytest.mark.asyncio
@_ASYNC_OMNI_CONTROL_PLANE_DRIFT
async def test_diffusion_only_sleep_wake_needs_no_resume_generation():
    """Pure-diffusion sleep must not set the admission hold.

    Diffusion v1 sync has no resume bridge; if the engine ever holds
    admission on diffusion sleep, that trainer hangs silently — this
    fails first.
    """
    omni = _build_async_omni("diffusion")

    await omni.sleep(level=1)

    assert omni._paused is True, "race guard set during sleep"
    assert omni._hold_admission_until_resume is False, "no AR stages — the hold must NOT be set"
    assert omni._engine_rpc_log == [], "diffusion sleep must not touch EngineCore RPCs"

    await omni.wake_up()

    assert omni._paused is False, "diffusion-only wake must restore admission by itself"


@pytest.mark.asyncio
@_ASYNC_OMNI_CONTROL_PLANE_DRIFT
async def test_ar_sleep_wake_requires_resume_generation_contrast():
    """The AR-side contrast that makes the omni_sync bridge load-bearing."""
    omni = _build_async_omni("llm")

    await omni.sleep(level=1)
    assert omni._hold_admission_until_resume is True

    await omni.wake_up()
    assert omni._paused is True, "wake must keep the hold for AR stages"
    assert "sleep" in omni._engine_rpc_log and "wake_up" in omni._engine_rpc_log

    await omni.resume_generation()
    assert omni._paused is False
    assert omni._hold_admission_until_resume is False


# ---------------------------------------------------------------------------
# admission resume rides every successful server-side wake — no per-trainer
# bridge needed (the engine keeps the sleep hold through its own wake_up)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_up_resumes_engine_admission():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)

    await server.wake_up()

    assert engine.wake_calls == [{"stage_ids": None, "tags": ["weights"]}]
    assert engine.resumed == 1


@pytest.mark.asyncio
async def test_release_and_resume_kv_cache_resume_admission():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)

    await server.release_kv_cache()
    assert engine.resumed == 1

    await server.resume_kv_cache()
    assert engine.resumed == 2


@pytest.mark.asyncio
async def test_failed_wake_skips_admission_resume():
    engine = _FakeAsyncOmni(wake_acks=[SimpleNamespace(status="FAILED", error_msg="boom")])
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="wake_up failed"):
        await server.wake_up()

    assert engine.resumed == 0


# ---------------------------------------------------------------------------
# RFC §4.4.0(b) — caches keyed on the served weights are flushed at publish.
#
# Deliberately sync ``asyncio.run`` tests. ``pytest-asyncio`` is declared in
# ``pyproject.toml`` and is the right way to drive the async cases above, but a
# bare ``async def test_`` with no marker is collected and never awaited (an
# invisible no-op for a regression test, which is worse than no test), so these
# five drive the coroutine by hand and stay runnable in any environment.
# ---------------------------------------------------------------------------


class _RpcRecordingEngine(_FakeAsyncOmni):
    """Adds the control-RPC surface the prompt-embed-cache calls reach for.

    The method is ``collective_rpc``, mirroring the real ``AsyncOmni`` client
    (``entrypoints/async_omni.py``), which exposes only ``collective_rpc`` and reaches
    the engine's ``AsyncOmniEngine.collective_rpc_async`` internally. An earlier
    version of this fake defined ``collective_rpc_async`` here, which is the *inner*
    engine's name -- so it mirrored the bug instead of the API, the tests passed, and
    production logged ``AttributeError: 'AsyncOmni' object has no attribute
    'collective_rpc_async'`` 888 times in one run. Keeping the fake faithful to the
    client is what makes these tests able to catch that.
    """

    def __init__(self, *, rpc_error: Exception | None = None, **kwargs):
        super().__init__(**kwargs)
        self.rpc_methods: list[str] = []
        self._rpc_error = rpc_error

    async def collective_rpc(self, method, stage_ids=None, args=(), kwargs=None):
        self.rpc_methods.append(method)
        if self._rpc_error is not None:
            raise self._rpc_error
        return [{"hits": 0, "misses": 0, "bypassed": 0, "size": 0}]


def _server_with_strategy(engine, strategy_cls):
    server = _make_server(engine)
    server._generate_strategy = object.__new__(strategy_cls)
    server.global_steps = 0
    return server


def test_weight_publish_flushes_both_policy_caches_on_a_diffusion_replica():
    engine = _RpcRecordingEngine()
    server = _server_with_strategy(engine, DiffusionStrategy)
    cached_before = server._lora_request_cache

    asyncio.run(server.set_global_steps(1))

    assert engine.rpc_methods == ["clear_prompt_embed_cache"], (
        "a conditioning entry is a function of (tokens, weights); new merged weights "
        "make it stale, since it would condition on the previous policy"
    )
    assert server._lora_request_cache is not cached_before
    assert server.global_steps == 1


def test_setting_the_same_global_step_does_not_flush():
    """Otherwise a no-op timer tick would cold-start the cache every step."""
    engine = _RpcRecordingEngine()
    server = _server_with_strategy(engine, DiffusionStrategy)

    asyncio.run(server.set_global_steps(0))

    assert engine.rpc_methods == []
    assert server.global_steps == 0


def test_an_ar_replica_has_no_conditioning_cache_to_flush():
    """The AR strategy owns no diffusion runner, so there is no such RPC to make."""
    engine = _RpcRecordingEngine()
    server = _server_with_strategy(engine, ARStrategy)

    asyncio.run(server.set_global_steps(1))

    assert engine.rpc_methods == []
    assert server.global_steps == 1


def test_every_weight_sync_transition_flushes_the_conditioning_cache():
    """sleep/release/resume/wake are all publish-adjacent, so all must flush."""
    engine = _RpcRecordingEngine()
    server = _server_with_strategy(engine, DiffusionStrategy)

    async def _run():
        await server.sleep()
        await server.release_kv_cache()
        await server.resume_kv_cache()
        await server.wake_up()

    asyncio.run(_run())

    # ``release_kv_cache`` spans a sleep + a weights wake, so it flushes on both
    # sides; the total is 5, not one per method.
    assert engine.rpc_methods == ["clear_prompt_embed_cache"] * 5


def test_a_failed_flush_does_not_break_the_weight_sync():
    """The flush is a performance device; it must not abort a publish."""
    engine = _RpcRecordingEngine(rpc_error=RuntimeError("runner has no such method"))
    server = _server_with_strategy(engine, DiffusionStrategy)

    asyncio.run(server.set_global_steps(2))

    assert engine.rpc_methods == ["clear_prompt_embed_cache"]
    assert server.global_steps == 2, "the publish itself still completed"


# ---------------------------------------------------------------------------
# a missing engine method is latched off after the first attempt.
#
# ``clear_prompt_embed_cache`` / ``get_prompt_embed_cache_stats`` live on
# ``DiffusionModelRunner``, but the pinned engine resolves RPCs against the *worker*
# and never reaches ``model_runner``. The stage pool logs the traceback itself and
# returns ``{"supported": False, "error": "… has no attribute …"}``, so this cannot be
# caught by the client try/except -- retrying it on every weight sync floods the log.
# Measured 2026-09-22 on hk01dgx039: one run emitted the worker traceback for every
# flush, and the early-latch case below is what bounds it to one per server.
# ---------------------------------------------------------------------------

def _missing_method_rpc(engine):
    """Install an ``collective_rpc`` that reports the called method as absent."""

    async def _rpc(method, stage_ids=None, args=(), kwargs=None):
        engine.rpc_methods.append(method)
        return [
            {
                "supported": False,
                "error": (
                    f"RPC '{method}' failed on worker rank(s): rank 0: AttributeError: "
                    f"'DiffusionWorkerWithvLLMOmniColocateWorkerExtension' object has no "
                    f"attribute '{method}'"
                ),
            }
        ]

    engine.collective_rpc = _rpc


def test_an_unserviceable_flush_is_attempted_only_once():
    engine = _RpcRecordingEngine()
    server = _server_with_strategy(engine, DiffusionStrategy)
    _missing_method_rpc(engine)

    async def _run():
        # Four publish-adjacent transitions; the first issues the RPC, the rest skip it.
        await server.sleep()
        await server.release_kv_cache()
        await server.resume_kv_cache()
        await server.wake_up()

    asyncio.run(_run())

    assert engine.rpc_methods == ["clear_prompt_embed_cache"], (
        "the engine logs a traceback per attempt, so a known-missing method must be "
        "latched off after the first failure"
    )


def test_a_transient_stage_failure_does_not_latch_the_method_off():
    """Only a *missing method* latches. A detached replica or a sleeping stage returns
    the same ``supported: False`` shape and must keep being retried."""
    engine = _RpcRecordingEngine()
    server = _server_with_strategy(engine, DiffusionStrategy)

    async def _engine_reports_transient(method, stage_ids=None, args=(), kwargs=None):
        engine.rpc_methods.append(method)
        return [{"supported": False, "error": "stage 0 replica 0 is not attached"}]

    engine.collective_rpc = _engine_reports_transient

    async def _run():
        await server.sleep()
        await server.release_kv_cache()

    asyncio.run(_run())

    assert engine.rpc_methods == ["clear_prompt_embed_cache"] * 3, (
        "a transient failure must not permanently disable the flush"
    )


def test_unserviceable_stats_return_no_counters_and_latch_once():
    engine = _RpcRecordingEngine()
    server = _make_server(engine)
    _missing_method_rpc(engine)

    first = asyncio.run(server.prompt_embed_cache_stats())
    second = asyncio.run(server.prompt_embed_cache_stats())

    assert first == [] and second == [], "no counters must be reported, not an error dict"
    assert engine.rpc_methods == ["get_prompt_embed_cache_stats"], "latched after the first attempt"


def test_stats_latch_does_not_disable_the_flush():
    """Independent latches: the two RPCs are separate methods and fail separately."""
    server = _make_server(_FakeAsyncOmni())
    server._mark_prompt_embed_cache_rpc_unsupported("get_prompt_embed_cache_stats")

    assert server._prompt_embed_cache_rpc_unsupported("get_prompt_embed_cache_stats") is True
    assert server._prompt_embed_cache_rpc_unsupported("clear_prompt_embed_cache") is False


def test_prompt_embed_cache_stats_read_through_the_client_rpc():
    """The RFC §4.4.4 counters must actually be fetched, not raise and vanish.

    This call site is *not* inside a try/except, so the wrong method name surfaces as
    an ``AttributeError`` per poll rather than a silent no-op.
    """
    engine = _RpcRecordingEngine()
    server = _make_server(engine)

    stats = asyncio.run(server.prompt_embed_cache_stats())

    assert engine.rpc_methods == ["get_prompt_embed_cache_stats"]
    assert stats == [{"hits": 0, "misses": 0, "bypassed": 0, "size": 0}]


def test_prompt_embed_cache_stats_are_none_off_the_head_rank():
    """A non-head rank holds no engine client; it must not attempt the RPC at all."""
    engine = _RpcRecordingEngine()
    server = _make_server(engine, node_rank=1)

    assert asyncio.run(server.prompt_embed_cache_stats()) is None
    assert engine.rpc_methods == []


def test_the_async_omni_client_exposes_collective_rpc_not_the_engine_internal_name():
    """Pin the upstream API shape the two call sites depend on.

    ``AsyncOmni`` (the client held in ``server.engine``) exposes ``collective_rpc`` as a
    coroutine and delegates to ``AsyncOmniEngine.collective_rpc_async`` internally. Only
    the engine has the ``_async`` name. Asserting this here means a future engine bump
    that renames the client method fails loudly in CI rather than at runtime on a GPU
    node, which is how the original mismatch survived 888 logged occurrences.
    """
    async_omni = pytest.importorskip("vllm_omni.entrypoints.async_omni")

    assert hasattr(async_omni.AsyncOmni, "collective_rpc"), (
        "verl_omni calls engine.collective_rpc(...); the pinned client must keep exposing it"
    )
    assert asyncio.iscoroutinefunction(async_omni.AsyncOmni.collective_rpc), (
        "the call sites await it, so the client method must stay async"
    )
    assert not hasattr(async_omni.AsyncOmni, "collective_rpc_async"), (
        "collective_rpc_async belongs to AsyncOmniEngine; if it appears on the client, "
        "re-check which object verl_omni holds before changing the call sites back"
    )
