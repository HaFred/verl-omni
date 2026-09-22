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
"""CPU tests for the Bagel Co-RL (Joint-Training) in-loop RM payload consumer (torch-free)."""

from __future__ import annotations

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
    root = Path(__file__).resolve().parents[3]
    omni = root / "verl_omni"
    _ensure_pkg("verl_omni", omni)
    _ensure_pkg("verl_omni.utils", omni / "utils")
    _ensure_pkg("verl_omni.utils.reward_score", omni / "utils" / "reward_score")
    _ensure_pkg("verl_omni.utils.agentic", omni / "utils" / "agentic")
    _load_by_path("verl_omni.utils.agentic.vllm_chat", omni / "utils" / "agentic" / "vllm_chat.py")
    _load_by_path(
        "verl_omni.utils.agentic_image_judge_parse", omni / "utils" / "agentic_image_judge_parse.py"
    )
    client = _load_by_path(
        "verl_omni.utils.reward_score.agentic_image_judge_client",
        omni / "utils" / "reward_score" / "agentic_image_judge_client.py",
    )
    scorer = _load_by_path(
        "verl_omni.utils.reward_score.bagel_rm_image_scorer_isolated",
        omni / "utils" / "reward_score" / "bagel_rm_image_scorer.py",
    )
    return client, scorer


client, scorer = _load_modules()


def _kwargs(payload=None, **extra):
    """Build the exact call ``NaiveRewardManager`` makes into ``compute_score``.

    The manager passes four keywords — ``data_source``, ``solution_str``,
    ``ground_truth``, ``extra_info`` — and never a single positional ``data``.
    The in-loop payload must ride ``extra_info["bagel_corl"]`` (L3/RFC §4.2).
    """
    extra_info = {"bagel_corl": payload} if payload is not None else {}
    return {
        "data_source": extra.pop("data_source", "bagel_corl_mid_loop_rm"),
        "solution_str": extra.pop("solution_str", ""),
        "ground_truth": extra.pop("ground_truth", ""),
        "extra_info": extra.pop("extra_info", extra_info),
        **extra,
    }


def _payload(**overrides):
    payload = {
        "image_paths": ["/a.png", "/b.png"],
        "reference_paths": ["ref.png"],
        "extra_info": {"user_prompt": "draw a castle", "good_enough_threshold": 0.8},
        "scorer_knobs": {"vllm_url": "http://rm:8000", "good_enough_threshold": 0.8},
        "image_prompt": "a castle at dusk",
    }
    payload.update(overrides)
    return payload


def test_mid_loop_payload_and_fail_loud_shapes():
    payload = _payload()
    assert scorer._mid_loop_payload({"bagel_corl": payload}) is payload
    # extra_info without the wire key → episode path, not an error.
    assert scorer._mid_loop_payload({"other": 1}) is None
    assert scorer._mid_loop_payload(None) is None
    assert scorer._mid_loop_payload("not-a-dict") is None
    # payload present but not a dict → fail loud.
    with pytest.raises(ValueError, match="must be a dict"):
        scorer._mid_loop_payload({"bagel_corl": "not-a-dict"})


def test_wire_key_matches_the_producer():
    """The producer (agent_loop.bagel_corl_rm) and this consumer must agree.

    Asserted as a literal so this module stays torch-free; the producer side is
    covered by ``tests/agent_loop/test_bagel_corl_rm_on_cpu.py``.
    """
    assert scorer.BAGEL_RM_EXTRA_INFO_KEY == "bagel_corl"


def test_compute_score_happy_path_aligns_per_image_outputs(monkeypatch):
    knobs_seen: list[dict] = []

    def fake_judge(**kwargs):
        knobs_seen.append(dict(kwargs["extra_info"]))
        # Second image slightly worse: exercises per-image alignment.
        if len(knobs_seen) == 1:
            return {"ok": True, "correctness": 0.8, "aesthetics": 0.6, "good_enough": True}
        return {"ok": True, "correctness": 0.4, "aesthetics": 0.2, "good_enough": False}

    monkeypatch.setattr(client, "call_reflect_vlm", fake_judge)
    result = scorer.compute_score(**_kwargs(_payload()))
    # Flat, manager-shaped: NaiveRewardManager reads result["score"] and merges
    # every other key into reward_extra_info (that is where parse_rm_result looks).
    assert result["sample_scores"] == [pytest.approx(0.7), pytest.approx(0.3)]
    assert result["sample_good_enough"] == [True, False]
    assert result["good_enough"] is False
    assert result["score"] == pytest.approx(0.5)
    assert "reward_extra_info" not in result
    assert "reward_score" not in result
    assert [row["image_path"] for row in result["per_image"]] == ["/a.png", "/b.png"]
    # Threshold knob reached the judge, fail-loud contract satisfied.
    assert knobs_seen[0]["good_enough_threshold"] == pytest.approx(0.8)
    assert knobs_seen[0]["vllm_url"] == "http://rm:8000"


def test_compute_score_fails_loud_when_any_image_unscored(monkeypatch):
    def fake_judge(**kwargs):
        if kwargs["image_path"] == "/b.png":
            return None  # missing file / empty URL / parse failure
        return {"ok": True, "correctness": 0.9, "aesthetics": 0.9, "good_enough": True}

    monkeypatch.setattr(client, "call_reflect_vlm", fake_judge)
    with pytest.raises(ValueError, match="refusing zero-fill"):
        scorer.compute_score(**_kwargs(_payload()))


def test_the_reward_pool_router_wins_over_the_configured_judge_url(monkeypatch):
    """``ENABLE_RM=1`` scores through the reward pool, not a second judge process.

    ``VisualRewardManager`` hands our function the router it built in front of the RM
    replicas. That router is the whole point of the internal design: the reward model
    itself produces the C/A numbers and the Yes/No flag, so the frozen sidecar (and its
    extra ~26 GiB process) is not needed.
    """
    knobs_seen: list[dict] = []

    def fake_judge(**kwargs):
        knobs_seen.append(dict(kwargs["extra_info"]))
        return {"ok": True, "correctness": 0.9, "aesthetics": 0.9, "good_enough": True}

    monkeypatch.setattr(client, "call_reflect_vlm", fake_judge)
    result = scorer.compute_score(
        **_kwargs(_payload(), reward_router_address="10.0.0.5:9000", model_name="/ckpt/qwen3-vl-2b")
    )
    # The pool's router replaced the configured external judge...
    assert knobs_seen[0]["vllm_url"] == "http://10.0.0.5:9000"
    # ...and carries the RM's own model id, which the OpenAI route requires.
    assert knobs_seen[0]["vllm_model"] == "/ckpt/qwen3-vl-2b"
    # Yes/No turn signals still come back through the identical parse.
    assert result["sample_good_enough"] == [True, True]
    assert result["good_enough"] is True


def test_an_http_router_address_is_not_double_prefixed(monkeypatch):
    knobs_seen: list[dict] = []

    def fake_judge(**kwargs):
        knobs_seen.append(dict(kwargs["extra_info"]))
        return {"ok": True, "correctness": 0.5, "aesthetics": 0.5, "good_enough": False}

    monkeypatch.setattr(client, "call_reflect_vlm", fake_judge)
    scorer.compute_score(**_kwargs(_payload(), reward_router_address="http://10.0.0.7:1234"))
    assert knobs_seen[0]["vllm_url"] == "http://10.0.0.7:1234"


def test_without_a_router_the_configured_judge_url_is_untouched(monkeypatch):
    """No reward model deployed → the configured judge stays the fallback (no regression)."""
    knobs_seen: list[dict] = []

    def fake_judge(**kwargs):
        knobs_seen.append(dict(kwargs["extra_info"]))
        return {"ok": True, "correctness": 0.5, "aesthetics": 0.5, "good_enough": False}

    monkeypatch.setattr(client, "call_reflect_vlm", fake_judge)
    scorer.compute_score(**_kwargs(_payload()))
    assert knobs_seen[0]["vllm_url"] == "http://rm:8000"
    assert "vllm_model" not in knobs_seen[0]


def test_a_blank_router_does_not_disable_a_configured_judge(monkeypatch):
    """An empty/whitespace ``reward_router_address`` must not shadow the configured URL."""
    knobs_seen: list[dict] = []

    def fake_judge(**kwargs):
        knobs_seen.append(dict(kwargs["extra_info"]))
        return {"ok": True, "correctness": 0.5, "aesthetics": 0.5, "good_enough": False}

    monkeypatch.setattr(client, "call_reflect_vlm", fake_judge)
    scorer.compute_score(**_kwargs(_payload(), reward_router_address="   "))
    assert knobs_seen[0]["vllm_url"] == "http://rm:8000"


def test_compute_score_requires_threshold(monkeypatch):
    payload = _payload(scorer_knobs={"vllm_url": "http://rm:8000"})
    payload["extra_info"].pop("good_enough_threshold", None)
    with pytest.raises(ValueError, match="good_enough_threshold missing"):
        scorer.compute_score(**_kwargs(payload))


def test_compute_score_requires_image_paths():
    with pytest.raises(ValueError, match="no image_paths"):
        scorer.compute_score(**_kwargs(_payload(image_paths=[])))


def test_compute_score_dispatches_to_episode_scorer(monkeypatch):
    """No payload → post-hoc episode scoring (agentic_multidim delegate)."""
    seen: dict = {}
    fake = types.ModuleType("verl_omni.utils.reward_score.agentic_multidim_reward")

    def fake_compute(data_source, solution_str, ground_truth, extra_info, **kwargs):
        seen.update(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return {"score": 0.42}

    fake.compute_score = fake_compute
    monkeypatch.setitem(sys.modules, "verl_omni.utils.reward_score.agentic_multidim_reward", fake)
    result = scorer.compute_score(**_kwargs(None, ground_truth="a castle"))
    assert result == {"score": 0.42}
    assert seen["ground_truth"] == "a castle"
    assert seen["data_source"] == "bagel_corl_mid_loop_rm"
