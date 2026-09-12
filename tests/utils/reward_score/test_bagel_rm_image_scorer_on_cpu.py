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
"""CPU tests for the Bagel Co-RL in-loop RM payload consumer (torch-free)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")
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


def _data(payload):
    return types.SimpleNamespace(non_tensor_batch={"bagel_rm_payload": np.array([payload], dtype=object)})


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


def test_extract_payload_and_fail_loud_shapes():
    payload = _payload()
    assert scorer.extract_bagel_rm_payload(_data(payload)) is payload
    # non_tensor_batch without the payload key
    with pytest.raises(ValueError, match="missing non_tensor_batch"):
        scorer.extract_bagel_rm_payload(types.SimpleNamespace(non_tensor_batch={}))
    # payload row present but not a dict
    bad_row = types.SimpleNamespace(
        non_tensor_batch={"bagel_rm_payload": np.array(["not-a-dict"], dtype=object)}
    )
    with pytest.raises(ValueError, match="must be a dict"):
        scorer.extract_bagel_rm_payload(bad_row)
    two = types.SimpleNamespace(non_tensor_batch={"bagel_rm_payload": np.array([payload, payload], dtype=object)})
    with pytest.raises(ValueError, match="exactly 1 payload row"):
        scorer.extract_bagel_rm_payload(two)


def test_compute_score_happy_path_aligns_per_image_outputs(monkeypatch):
    knobs_seen: list[dict] = []

    def fake_judge(**kwargs):
        knobs_seen.append(dict(kwargs["extra_info"]))
        # Second image slightly worse: exercises per-image alignment.
        if len(knobs_seen) == 1:
            return {"ok": True, "correctness": 0.8, "aesthetics": 0.6, "good_enough": True}
        return {"ok": True, "correctness": 0.4, "aesthetics": 0.2, "good_enough": False}

    monkeypatch.setattr(client, "call_reflect_vlm", fake_judge)
    result = scorer.compute_score(_data(_payload()))
    assert result["reward_extra_info"]["sample_scores"] == [pytest.approx(0.7), pytest.approx(0.3)]
    assert result["reward_extra_info"]["sample_good_enough"] == [True, False]
    assert result["reward_extra_info"]["good_enough"] is False
    assert result["reward_score"] == pytest.approx(0.5)
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
        scorer.compute_score(_data(_payload()))


def test_compute_score_requires_threshold(monkeypatch):
    payload = _payload(scorer_knobs={"vllm_url": "http://rm:8000"})
    payload["extra_info"].pop("good_enough_threshold", None)
    with pytest.raises((KeyError, TypeError, ValueError)):
        scorer.compute_score(_data(payload))


def test_compute_score_requires_image_paths():
    with pytest.raises(ValueError, match="no image_paths"):
        scorer.compute_score(_data(_payload(image_paths=[])))


def test_compute_score_dispatches_to_episode_scorer(monkeypatch):
    """No payload row → post-hoc episode scoring (agentic_multidim delegate)."""
    seen: dict = {}
    fake = types.ModuleType("verl_omni.utils.reward_score.agentic_multidim_reward")

    def fake_compute(data):
        seen["data"] = data
        return {"reward_score": 0.42}

    fake.compute_score = fake_compute
    monkeypatch.setitem(sys.modules, "verl_omni.utils.reward_score.agentic_multidim_reward", fake)
    data = types.SimpleNamespace(non_tensor_batch={})
    result = scorer.compute_score(data)
    assert result == {"reward_score": 0.42}
    assert seen["data"] is data
