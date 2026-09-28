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
"""CPU regression tests for the rollout wake's VRAM re-map retry.

The failure these pin was measured 2026-09-25 00:21:42 on hk01dgx039 (devices 0/1/6/7,
``bagel_corl_rm1_20260925_000612``): the first ``update_weights`` reached
``engine_workers.update_weights`` -> ``resume_weights_task`` -> ``vLLMOmniHttpServer.wake_up``
and the engine answered

    Wake-up failed on Rank 0: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:163

with ~41GiB nominally free on device 0 against a ~14GiB TP=2 re-map, because a *neighbouring*
job (pid 3240057) held 28.66GiB of that card. The re-map is a ``cuMemCreate``, which needs a
large physically contiguous region, so it can fail transiently on a shared card; a retry that
first returns the actor's cached blocks to the driver is what recovers the window.
"""

import asyncio
import inspect
from pathlib import Path

import pytest

from verl_omni.workers import engine_workers as ew

_ALLOC_FAILURE = (
    "CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:163"
)


class _RayStyleTaskError(Exception):
    """Mimics ``ray.exceptions.RayTaskError``: the real error lives on ``.cause``.

    This shape is the whole reason the classifier exists. ``RayTaskError.__str__`` renders the
    *task* repr -- ``ray::vLLMOmniHttpServer.wake_up() (pid=..., ip=...)`` -- so a check against
    ``str(exc)`` would never see the allocator message that is nested one level down.
    """

    def __init__(self, cause: BaseException):
        super().__init__("ray::vLLMOmniHttpServer.wake_up() (pid=3045736, ip=10.248.12.145)")
        self.cause = cause


class _FakeRollout:
    """A rollout whose first ``failures`` wake attempts blow up, then succeeds."""

    def __init__(self, failures: int, error_factory):
        self.failures = failures
        self.error_factory = error_factory
        self.calls = 0
        self.tags_seen: list[list[str]] = []

    async def resume(self, tags):
        self.calls += 1
        self.tags_seen.append(list(tags))
        if self.calls <= self.failures:
            raise self.error_factory()
        return [f"ack:{','.join(tags)}"]


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """Keep the retry schedule's *shape* but drop its wall-clock cost."""
    monkeypatch.setattr(ew, "WAKE_OOM_BACKOFF_S", (0.0, 0.0, 0.0, 0.0))


@pytest.fixture
def flushed(monkeypatch):
    """Record every actor-cache flush the retry performs."""
    calls = []

    def _flush(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(ew, "aggressive_empty_cache", _flush)
    return calls


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------------------


def test_is_cuda_oom_reads_the_allocator_message():
    assert ew._is_cuda_oom(RuntimeError(_ALLOC_FAILURE)) is True


def test_is_cuda_oom_reads_the_plain_torch_message():
    assert ew._is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) is True


def test_is_cuda_oom_reads_through_a_ray_style_wrapper():
    """The measured shape: RayTaskError's own text says nothing about memory."""
    wrapped = _RayStyleTaskError(RuntimeError(_ALLOC_FAILURE))
    assert "out of memory" not in str(wrapped)
    assert ew._is_cuda_oom(wrapped) is True


def test_is_cuda_oom_follows_a_raise_from_chain():
    inner = RuntimeError(_ALLOC_FAILURE)
    outer = RuntimeError("collective_rpc failed")
    outer.__cause__ = inner
    assert ew._is_cuda_oom(outer) is True


def test_is_cuda_oom_ignores_unrelated_failures():
    assert ew._is_cuda_oom(RuntimeError("Engine is partially or fully asleep")) is False
    assert ew._is_cuda_oom(AssertionError("len(output) == len(batch)")) is False


def test_is_cuda_oom_is_safe_on_a_self_referential_chain():
    loop = RuntimeError("wrapped")
    loop.__cause__ = loop
    assert ew._is_cuda_oom(loop) is False


# --------------------------------------------------------------------------------------
# the retry
# --------------------------------------------------------------------------------------


def test_resume_retries_a_transient_oom_and_then_succeeds(flushed):
    rollout = _FakeRollout(failures=2, error_factory=lambda: _RayStyleTaskError(RuntimeError(_ALLOC_FAILURE)))
    timings: dict = {}

    acks = _run(ew._resume_rollout_vram_safe(rollout, ["weights"], timings, "resume_weights"))

    assert acks == ["ack:weights"]
    assert rollout.calls == 3, "two failed attempts plus the successful one"
    # The flush is what makes a later attempt likelier to win: it hands the driver back the
    # actor's reserved-but-unallocated blocks that the engine's cuMemCreate wants.
    assert len(flushed) == 2, "one flush before each retry, none before the first attempt"
    assert "resume_weights" in timings, "timing is recorded across the whole retry schedule"


def test_resume_flushes_before_every_retry_not_before_the_first_attempt(flushed):
    rollout = _FakeRollout(failures=1, error_factory=lambda: RuntimeError(_ALLOC_FAILURE))
    _run(ew._resume_rollout_vram_safe(rollout, ["kv_cache"], {}, "resume_kv_cache"))
    assert len(flushed) == 1
    assert flushed[0][1].get("force_sync") is True


def test_resume_does_not_retry_a_real_failure(flushed):
    """A non-OOM error must surface on the first attempt, unchanged."""
    rollout = _FakeRollout(failures=1, error_factory=lambda: RuntimeError("Engine is partially or fully asleep"))
    with pytest.raises(RuntimeError, match="asleep"):
        _run(ew._resume_rollout_vram_safe(rollout, ["weights"], {}, "resume_weights"))
    assert rollout.calls == 1
    assert flushed == []


def test_resume_gives_up_and_reraise_once_the_schedule_is_exhausted(monkeypatch, flushed):
    """A genuine shortage still fails loud -- the retries only buy back a transient window."""
    monkeypatch.setattr(ew, "WAKE_OOM_RETRIES", 2)
    rollout = _FakeRollout(failures=99, error_factory=lambda: RuntimeError(_ALLOC_FAILURE))

    with pytest.raises(RuntimeError, match="cumem_allocator"):
        _run(ew._resume_rollout_vram_safe(rollout, ["weights"], {}, "resume_weights"))

    assert rollout.calls == 3, "the first attempt plus 2 retries"
    assert len(flushed) == 2


def test_resume_issues_the_same_tags_on_every_attempt(flushed):
    """Re-issuing must be the *same* request: a failed wake samples nothing, so it is idempotent."""
    rollout = _FakeRollout(failures=1, error_factory=lambda: RuntimeError(_ALLOC_FAILURE))
    _run(ew._resume_rollout_vram_safe(rollout, ["weights", "kv_cache"], {}, "resume_weights"))
    assert rollout.tags_seen == [["weights", "kv_cache"], ["weights", "kv_cache"]]


# --------------------------------------------------------------------------------------
# the call sites
# --------------------------------------------------------------------------------------


def test_update_weights_routes_both_resumes_through_the_retry():
    """Guard the wiring: a bare ``self.rollout.resume`` would silently lose the retry."""
    source = inspect.getsource(ew)
    assert "self.rollout.resume(tags=" not in source, "every wake must go through the VRAM-safe wrapper"
    assert source.count("_resume_rollout_vram_safe(self.rollout,") == 2, "weights and kv_cache"


def test_recipe_guard_test_still_reads_this_file():
    """``_resume_rollout_vram_safe`` must live in the module the trainer actually imports."""
    assert Path(ew.__file__).name == "engine_workers.py"
    assert inspect.iscoroutinefunction(ew._resume_rollout_vram_safe)


# --------------------------------------------------------------------------------------
# the shared leaf, and the server-side call site the 2026-09-27 run died on
# --------------------------------------------------------------------------------------


def test_the_classifier_is_one_shared_leaf_module():
    """``engine_workers`` and the rollout server must judge the same way.

    They cannot import each other (``engine_workers`` -> rollout -> server would cycle), which is
    why the classifier lives in a leaf. Asserting the identity -- not just equality -- is what
    keeps a future edit from growing a second copy that drifts.
    """
    from verl_omni.utils import rollout_wake as rw

    assert ew._is_cuda_oom is rw.is_cuda_oom


def test_wake_with_oom_retry_succeeds_after_transient_failures_and_flushes():
    from verl_omni.utils import rollout_wake as rw

    flushes: list[float] = []
    attempts: list[int] = []
    state = {"calls": 0}

    async def _wake_once():
        state["calls"] += 1
        attempts.append(state["calls"])
        if state["calls"] <= 2:
            raise _RayStyleTaskError(RuntimeError(_ALLOC_FAILURE))
        return ["ack"]

    async def _no_wait(delay):  # keep the schedule's shape, drop its wall-clock cost
        flushes.append(delay)

    acks = _run(
        rw.wake_with_oom_retry(
            _wake_once,
            flush=lambda: flushes.append(-1.0),
            backoff=lambda attempt: 3.0 * attempt,
            sleep=_no_wait,
            retries=4,
        )
    )
    assert acks == ["ack"]
    assert attempts == [1, 2, 3], "two failed attempts plus the successful one"
    # Flush before every retry, never before the first attempt: the first attempt has nothing to
    # reclaim, and flushing costs a device sync.
    assert flushes == [-1.0, 3.0, -1.0, 6.0]


def test_wake_with_oom_retry_reraises_a_non_oom_immediately():
    from verl_omni.utils import rollout_wake as rw

    async def _wake_once():
        raise RuntimeError("Engine is partially or fully asleep")

    with pytest.raises(RuntimeError, match="partially or fully asleep"):
        _run(rw.wake_with_oom_retry(_wake_once, retries=4))


def test_wake_with_oom_retry_gives_up_and_reraises_once_the_schedule_is_exhausted():
    from verl_omni.utils import rollout_wake as rw

    state = {"calls": 0}

    async def _wake_once():
        state["calls"] += 1
        raise _RayStyleTaskError(RuntimeError(_ALLOC_FAILURE))

    async def _no_wait(_delay):
        return None

    with pytest.raises(_RayStyleTaskError):
        _run(rw.wake_with_oom_retry(_wake_once, retries=2, sleep=_no_wait, backoff=lambda _a: 0.0))
    assert state["calls"] == 3, "the first attempt plus two retries"


def test_the_rollout_server_wakes_through_the_vram_safe_retry():
    """Guard the wiring that killed ``bagel_corl_rm1_20260927_165227`` at step 41.

    The trainer calls ``server.wake_up.remote(...)`` directly (``_wake_replicas``), so a bare
    ``self.engine.wake_up`` inside this server has *no* retry anywhere up the stack. Measured
    2026-09-27 18:11: ``RuntimeError: wake_up failed on a stage: 'CUDA Error: out of memory at
    /workspace/csrc/cumem_allocator.cpp:163'``, raised out of ``_validate_acks`` and fatal 41
    steps into a healthy run.

    Read as source (not imported): this file needs vLLM at import time and the property under
    test is purely structural.
    """
    source = Path(__file__).resolve().parents[2] / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"
    text = source.read_text()
    # Every ``engine.wake_up`` in the module must sit inside the VRAM-safe wrapper, and every
    # public wake path must route through it (the wrapper itself is the third caller).
    assert text.count("await self.engine.wake_up(tags=") == 1, "the only bare wake is inside the wrapper"
    assert text.count("await self._wake_engine_vram_safe(") == 3, "wake_up, release_kv_cache, resume_kv_cache"
    # The public ``wake_up`` must not talk to the engine directly.
    start = text.index("    async def wake_up(")
    body = text[start : text.index("\n    async def ", start + 1)]
    assert "engine.wake_up(" not in body
    assert "await self._wake_engine_vram_safe(" in body

    wrapper = text[text.index("    async def _wake_engine_vram_safe(") : text.index("    async def wake_up(")]
    assert "wake_with_oom_retry(" in wrapper
    assert "aggressive_empty_cache(force_sync=True)" in wrapper, "each retry must reclaim the cached gap"
    # The failure rides the ack, not the await (``_validate_acks`` raises), so both must be inside
    # the retried unit -- retrying only ``engine.wake_up`` would never see this OOM at all.
    assert 'self._validate_acks("wake_up", acks)' in wrapper


def test_the_trainer_wakes_replicas_sequentially():
    """The wake is a ``cuMemCreate`` re-map; asking all replicas at once races itself for the
    contiguous window. ``bagel_corl_rm1_20260927_165227`` died on exactly that at step 41 while
    ``asyncio.gather``-ing 3 replicas on 4 GPUs that also carry the colocated RM pool.
    """
    source = Path(__file__).resolve().parents[2] / "verl_omni/trainer/omni/bagel_corl_trainer.py"
    text = source.read_text()
    start = text.index("def _wake_replicas(")
    body = text[start : text.index("\n    def ", start + 1)]
    assert "asyncio.gather(" not in body, "sequential wake is the point"
    assert "await server.wake_up.remote(" in body
