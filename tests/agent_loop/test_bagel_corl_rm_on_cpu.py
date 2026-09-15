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
"""CPU tests for the Bagel Co-RL (Joint-Training) in-loop RM adapter (torch-free)."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _ensure_pkg(name: str, path: Path) -> None:
    """Register a transparent package stub so submodule imports skip heavy ``__init__``.

    A bare stub leaks for the rest of the pytest session: any later test file asking
    for a name the stub lacks (``from verl_omni.tools.trajectory import ...``, or a
    ``monkeypatch.setattr`` on ``verl_omni.tools.trajectory.hydra_env``) then dies with
    ImportError or AttributeError, depending on collection order. ``_missing`` answers
    those from the real package without making the heavy import eager.
    """
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")

    def _missing(attr: str):
        # PEP 562 hook, invoked only for names this stub lacks, so anything a test file
        # registered here still wins. Resolution mirrors a real package in two steps,
        # only the second of which executes an ``__init__``:
        #   1. a submodule of that name — ``getattr(verl_omni, "tools")``, and the
        #      ``verl_omni.tools.trajectory.hydra_env`` that dotted-path patching needs;
        #   2. otherwise the real ``__init__.py``, lazily — a re-export such as
        #      ``from verl_omni.tools.trajectory import active_trajectory_relpath``.
        if attr.startswith("__"):
            raise AttributeError(attr)
        try:
            child = importlib.import_module(f"{pkg.__name__}.{attr}")
        except ImportError:
            pass
        else:
            pkg.__dict__[attr] = child  # real packages expose submodules as attributes
            return child
        if path.is_dir() and not pkg.__dict__.get("_real_init_loaded"):
            try:
                spec = importlib.util.spec_from_file_location(
                    pkg.__name__, path / "__init__.py", submodule_search_locations=[str(path)]
                )
                real = importlib.util.module_from_spec(spec)
                # Execute the real ``__init__`` under the package name — that is what
                # makes its relative ``from .x import y`` resolve — while this stub stays
                # the canonical ``sys.modules`` entry, so the hook above keeps working.
                real.__package__ = pkg.__name__
                spec.loader.exec_module(real)
            except Exception:  # optional/heavy deps absent: stay a stub
                pass
            else:
                pkg.__dict__["_real_init_loaded"] = True
                pkg.__dict__.update(
                    {k: v for k, v in vars(real).items() if k not in {"__getattr__", "__dict__"}}
                )
        try:
            return pkg.__dict__[attr]
        except KeyError:
            raise AttributeError(f"module {pkg.__name__!r} has no attribute {attr!r}") from None

    pkg.__getattr__ = _missing
    sys.modules[name] = pkg


def _load_by_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_modules():
    root = Path(__file__).resolve().parents[2]
    omni = root / "verl_omni"
    _ensure_pkg("verl_omni", omni)
    _ensure_pkg("verl_omni.agent_loop", omni / "agent_loop")
    _ensure_pkg("verl_omni.tools", omni / "tools")
    _ensure_pkg("verl_omni.tools.trajectory", omni / "tools" / "trajectory")
    _load_by_path("verl_omni.tools.trajectory.hydra_env", omni / "tools" / "trajectory" / "hydra_env.py")
    _load_by_path("verl_omni.tools.trajectory.paths", omni / "tools" / "trajectory" / "paths.py")
    _load_by_path("verl_omni.tools.trajectory.context", omni / "tools" / "trajectory" / "context.py")
    _load_by_path("verl_omni.tools.trajectory.artifacts", omni / "tools" / "trajectory" / "artifacts.py")
    _load_by_path("verl_omni.tools.trajectory.judge_latch", omni / "tools" / "trajectory" / "judge_latch.py")
    _load_by_path("verl_omni.agent_loop.rpco_turn_protocol", omni / "agent_loop" / "rpco_turn_protocol.py")
    lib = _load_by_path("bagel_corl_lib_isolated_rm", omni / "agent_loop" / "bagel_corl_lib.py")
    rm = _load_by_path("bagel_corl_rm_isolated", omni / "agent_loop" / "bagel_corl_rm.py")
    return lib, rm


lib, rm = _load_modules()


def _sample(seed_index: int, *, path: str | None = "/tmp/img.png", valid: bool = True) -> lib.GenSample:
    return lib.GenSample(
        gen_sample_uid=f"g:{seed_index}",
        gen_group_uid="g",
        seed_index=seed_index,
        valid=valid,
        prompt_token_ids=[1],
        image_path=path,
    )


class _FakeAsyncHandle:
    """Duck-typed stand-in for an async ray reward-loop worker."""

    def __init__(self, result=None, *, fail: bool = False):
        self.calls: list = []
        self._result = result if result is not None else {"reward_score": 0.8, "reward_extra_info": {}}
        self._fail = fail

    async def compute_score(self, data):
        self.calls.append(data)
        if self._fail:
            raise RuntimeError("rm exploded")
        return self._result


def test_bind_get_roundtrip_and_fail_loud():
    rm.bind_bagel_rm_handles(None)
    assert rm.get_bagel_rm_gen_handle() is None
    h1, h2 = object(), object()
    rm.bind_bagel_rm_handles([h1, h2])
    assert rm.get_bagel_rm_gen_handle() is h1  # [0] is the GEN pool
    with pytest.raises(rm.RMScoringError, match="at least 1"):
        rm.bind_bagel_rm_handles([])
    with pytest.raises(rm.RMScoringError, match="must be a list"):
        rm.bind_bagel_rm_handles(h1)
    rm.bind_bagel_rm_handles(None)
    assert rm.get_bagel_rm_gen_handle() is None


def test_build_payload_requires_image_paths():
    payload = rm.build_rm_score_payload(["/a.png"], ["ref.png"], {"task_type": "plan"}, {"good_enough_threshold": 0.8})
    assert payload["image_paths"] == ["/a.png"]
    assert payload["reference_paths"] == ["ref.png"]
    assert payload["extra_info"]["task_type"] == "plan"
    assert payload["scorer_knobs"]["good_enough_threshold"] == 0.8
    with pytest.raises(rm.RMScoringError, match="at least one image"):
        rm.build_rm_score_payload([])


def test_parse_rm_result_scalar_broadcast_and_per_sample():
    scores, flags = rm.parse_rm_result({"reward_score": 0.8, "reward_extra_info": {"good_enough": True}}, ["/a", "/b"])
    assert scores == [0.8, 0.8]
    assert flags == [True, True]
    scores, flags = rm.parse_rm_result(
        {"reward_score": 0.0, "reward_extra_info": {"sample_scores": [0.1, 0.9], "good_enough": False}},
        ["/a", "/b"],
    )
    assert scores == [0.1, 0.9]
    assert flags == [False, False]
    # Per-image flags win over the broadcast scalar.
    scores, flags = rm.parse_rm_result(
        {
            "reward_score": 0.0,
            "reward_extra_info": {
                "sample_scores": [0.1, 0.9],
                "good_enough": False,
                "sample_good_enough": [True, False],
            },
        },
        ["/a", "/b"],
    )
    assert flags == [True, False]
    scores, flags = rm.parse_rm_result({"reward_score": 0.5}, ["/a"])
    assert scores == [0.5]
    assert flags == [None]
    with pytest.raises(rm.RMScoringError, match="length"):
        rm.parse_rm_result({"reward_score": 0.0, "reward_extra_info": {"sample_scores": [0.1]}}, ["/a", "/b"])
    with pytest.raises(rm.RMScoringError, match="length"):
        rm.parse_rm_result(
            {"reward_score": 0.0, "reward_extra_info": {"sample_good_enough": [True]}}, ["/a", "/b"]
        )
    with pytest.raises(rm.RMScoringError, match="missing 'reward_score'"):
        rm.parse_rm_result({"nope": 1}, ["/a"])


def test_adapter_forwards_image_prompt_and_per_sample_flags():
    handle = _FakeAsyncHandle(
        {
            "reward_score": 0.5,
            "reward_extra_info": {"sample_scores": [0.3, 0.7], "sample_good_enough": [False, True]},
        }
    )
    prompts: list[str] = []

    def _image_prompt() -> str:
        prompt = "a castle"
        prompts.append(prompt)
        return prompt

    score_fn = rm.make_rm_score_fn(handle, get_image_prompt=_image_prompt, data_builder=lambda payload: payload)
    samples = [_sample(0, path="/a.png"), _sample(1, path="/b.png")]
    out = asyncio.run(score_fn(samples))
    assert prompts == ["a castle"]
    assert handle.calls[0]["image_prompt"] == "a castle"
    assert out[0].rm_score == pytest.approx(0.3)
    assert out[0].good_enough is False
    assert out[1].good_enough is True


def test_make_rm_score_fn_requires_handle():
    with pytest.raises(rm.RMScoringError, match="requires a reward-loop handle"):
        rm.make_rm_score_fn(None)


def test_adapter_scores_valid_samples_and_skips_invalid():
    handle = _FakeAsyncHandle({"reward_score": 0.8, "reward_extra_info": {"good_enough": True}})
    score_fn = rm.make_rm_score_fn(
        handle, get_reference_paths=lambda: ["ref0.png"], data_builder=lambda payload: payload
    )
    samples = [
        _sample(0, path="/a.png"),
        _sample(1, path=None),
        _sample(2, path="/b.png", valid=False),
    ]
    out = asyncio.run(score_fn(samples))
    assert len(handle.calls) == 1
    # Injected builder passes the payload dict through: only valid samples with a
    # path enter it (the real DataProto builder wraps it as one object row).
    row = handle.calls[0]
    assert row["image_paths"] == ["/a.png"]
    assert row["reference_paths"] == ["ref0.png"]
    assert out[0].rm_score == pytest.approx(0.8)
    assert out[0].good_enough is True
    assert out[1].rm_score is None  # no path → unscored
    assert out[2].rm_score is None  # invalid → unscored


def test_adapter_without_path_scores_returns_samples_untouched():
    handle = _FakeAsyncHandle()
    score_fn = rm.make_rm_score_fn(handle, data_builder=lambda payload: payload)
    samples = [_sample(0, path=None)]
    out = asyncio.run(score_fn(samples))
    assert handle.calls == []
    assert out[0].rm_score is None


def test_adapter_fail_loud_on_bad_handle_and_bad_result():
    class _NoCompute:
        pass

    with pytest.raises(rm.RMScoringError, match="no compute_score"):
        asyncio.run(rm.make_rm_score_fn(_NoCompute(), data_builder=lambda payload: payload)([_sample(0)]))

    handle = _FakeAsyncHandle({"wrong": 1})
    with pytest.raises(rm.RMScoringError, match="missing 'reward_score'"):
        asyncio.run(rm.make_rm_score_fn(handle, data_builder=lambda payload: payload)([_sample(0)]))

    handle = _FakeAsyncHandle(fail=True)
    with pytest.raises(RuntimeError, match="rm exploded"):
        asyncio.run(rm.make_rm_score_fn(handle, data_builder=lambda payload: payload)([_sample(0)]))


def test_adapter_maps_per_sample_scores_by_path():
    handle = _FakeAsyncHandle({"reward_score": 0.0, "reward_extra_info": {"sample_scores": [0.3, 0.7]}})
    score_fn = rm.make_rm_score_fn(handle, data_builder=lambda payload: payload)
    samples = [_sample(0, path="/a.png"), _sample(1, path="/b.png")]
    out = asyncio.run(score_fn(samples))
    assert out[0].rm_score == pytest.approx(0.3)
    assert out[1].rm_score == pytest.approx(0.7)
