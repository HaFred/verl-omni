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
"""End-to-end L3 regression: the in-loop RM payload must survive the real reward manager.

RFC §4.2 wires the SAME handles for mid-episode GEN scoring and post-hoc episode
scoring, and ``RewardLoopWorker.compute_score`` hands the built ``DataProto``
straight to ``reward_manager.run_single``. This module pins the contract the
scorer has to match against the REAL manager class rather than a stub:

- the manager calls the reward function with exactly four keywords
  (``data_source`` / ``solution_str`` / ``ground_truth`` / ``extra_info``);
- ``extra_info`` is the only field forwarded verbatim, so the payload rides
  ``extra_info["bagel_corl"]``;
- the function must return ``{"score": ...}`` flat; the manager does the nesting
  into ``reward_extra_info``, which is where ``parse_rm_result`` looks.

A scorer taking a single positional ``data``, or returning ``reward_score``,
fails at episode time with ``TypeError``/``KeyError`` — the L3 latent blocker
this test exists to catch.
"""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.protocol import DataProto

from verl_omni.agent_loop import bagel_corl_rm as rm
from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager, _validate_visual_response
from verl_omni.utils.reward_score import agentic_image_judge_client as judge_client
from verl_omni.utils.reward_score import bagel_rm_image_scorer as scorer

THRESHOLD = 0.8


class _DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return ""


def _manager(compute_score, **kwargs):
    return NaiveRewardManager(
        config=OmegaConf.create({"reward": {}}),
        tokenizer=_DummyTokenizer(),
        compute_score=compute_score,
        **kwargs,
    )


def _run_single(build_manager, data):
    """Construct the manager and run ``run_single`` on ONE loop.

    ``RewardManagerBase`` captures ``self.loop`` at construction time and then
    ``run_in_executor``s the tokenizer decode on it. Constructing the manager
    outside the loop it is later driven on fails with "attached to a different
    loop". pytest-asyncio is not installed here, so drive the loop by hand.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(build_manager().run_single(data))
    finally:
        asyncio.set_event_loop(None)
        loop.close()

def _judge(**kwargs):
    """Per-image judge stub: ``/a.png`` good, everything else not."""
    if kwargs["image_path"] == "/a.png":
        return {"ok": True, "correctness": 0.8, "aesthetics": 0.6, "good_enough": True}
    return {"ok": True, "correctness": 0.4, "aesthetics": 0.2, "good_enough": False}


def _mid_loop_payload(image_paths=("/a.png", "/b.png"), *, backend="vlm_judge"):
    # ``backend`` is pinned here, not left to default: these tests monkeypatch
    # ``call_reflect_vlm`` and assert on the facet-grid arithmetic (0.8+0.6)/2 == 0.7,
    # so they exercise the ``vlm_judge`` backend specifically. The shipped default is
    # ``unified_reward`` (continuous 1-5 axes); see
    # ``test_unified_backend_scores_are_graded_and_fail_loud`` for that path.
    return rm.build_rm_score_payload(
        list(image_paths),
        extra_info={"user_prompt": "a castle", "good_enough_threshold": THRESHOLD},
        scorer_knobs={"good_enough_threshold": THRESHOLD, "score_backend": backend},
    )


def test_builder_row_is_consumable_by_the_real_reward_manager():
    payload = _mid_loop_payload()
    data = rm._default_data_builder(payload)

    # The manager indexes batch_size [1] and reads responses / attention_mask,
    # plus non_tensor_batch["data_source"] and ["reward_model"]["ground_truth"].
    assert isinstance(data, DataProto)
    assert data.batch["responses"].shape[-1] >= 1
    assert data.batch["attention_mask"].shape[-1] >= data.batch["responses"].shape[-1]
    assert data.non_tensor_batch["data_source"][0] == "bagel_corl_mid_loop_rm"
    assert data.non_tensor_batch["reward_model"][0]["ground_truth"] == ""
    # And the payload rides the one key the manager forwards verbatim.
    assert data.non_tensor_batch["extra_info"][0][rm.BAGEL_RM_EXTRA_INFO_KEY] is payload


def test_payload_survives_the_manager_and_parse_rm_result(monkeypatch):
    """Full mid-loop path: builder → real manager → scorer → parse_rm_result."""
    monkeypatch.setattr(judge_client, "call_reflect_vlm", _judge)
    seen: dict = {}
    real_compute_score = scorer.compute_score

    def spy(**kwargs):
        seen.update(kwargs)
        return real_compute_score(**kwargs)

    payload = _mid_loop_payload()
    result = _run_single(lambda: _manager(spy), rm._default_data_builder(payload))

    # 4-kwarg contract: no positional ``data``.
    assert set(seen) >= {"data_source", "solution_str", "ground_truth", "extra_info"}
    # Payload arrived under extra_info["bagel_corl"], untouched.
    assert seen["extra_info"][rm.BAGEL_RM_EXTRA_INFO_KEY]["image_paths"] == ["/a.png", "/b.png"]

    # Manager shape: scalar under reward_score, scorer keys nested for parse_rm_result.
    assert result["reward_score"] == pytest.approx(0.5)
    extra = result["reward_extra_info"]
    assert extra["sample_scores"] == [pytest.approx(0.7), pytest.approx(0.3)]
    assert extra["sample_good_enough"] == [True, False]

    scores, flags = rm.parse_rm_result(result, payload["image_paths"])
    assert scores == [pytest.approx(0.7), pytest.approx(0.3)]
    assert flags == [True, False]


def test_scorer_returns_flat_score_not_reward_score(monkeypatch):
    """Flat ``score`` is required: the manager itself does the nesting."""
    monkeypatch.setattr(judge_client, "call_reflect_vlm", _judge)
    out = scorer.compute_score(
        data_source="bagel_corl_mid_loop_rm",
        solution_str="",
        ground_truth="",
        extra_info={rm.BAGEL_RM_EXTRA_INFO_KEY: _mid_loop_payload(("/a.png",))},
    )
    assert out["score"] == pytest.approx(0.7)
    assert "reward_score" not in out
    assert "reward_extra_info" not in out


def test_default_data_builder_dtype_matches_pipeline_output_type():
    """The RM worker's ``VisualRewardManager`` dtype-checks the row before scoring.

    ``_validate_visual_response`` demands uint8 pixels, or a floating-point tensor
    when ``rollout.pipeline.output_type == "latent"``. A payload row built with the
    wrong dtype raises a ValueError inside the RM worker, mid-episode — after the
    images were already generated.
    """
    image_row = rm._default_data_builder(_mid_loop_payload(), output_type="image")
    assert image_row.batch["responses"].dtype == torch.uint8
    latent_row = rm._default_data_builder(_mid_loop_payload(), output_type="latent")
    assert latent_row.batch["responses"].dtype.is_floating_point
    # And the real validator accepts both.
    for output_type, row in (("image", image_row), ("latent", latent_row)):
        _validate_visual_response(
            row.batch["responses"],
            OmegaConf.create({"actor_rollout_ref": {"rollout": {"pipeline": {"output_type": output_type}}}}),
            is_validate=False,
        )


def test_default_data_builder_rejects_wrong_dtype_for_image_output():
    """Guards the guard: the manager really does reject a long ``responses``."""
    row = rm._default_data_builder(_mid_loop_payload(), output_type="latent")
    with pytest.raises(ValueError, match="uint8"):
        _validate_visual_response(
            row.batch["responses"],
            OmegaConf.create({"actor_rollout_ref": {"rollout": {"pipeline": {"output_type": "image"}}}}),
            is_validate=False,
        )


def test_visual_reward_manager_consumes_the_in_loop_payload(monkeypatch):
    """Row 9/10: ``VisualRewardManager`` is the repo default for omni.

    It passes ``solution_image`` (not ``solution_str``) and validates the response
    dtype, so the mid-loop builder must satisfy that manager — not only the
    ``NaiveRewardManager`` the earlier tests use.
    """
    monkeypatch.setattr(judge_client, "call_reflect_vlm", _judge)
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"pipeline": {"output_type": "image"}}},
            "reward": {"reward_model": {"model_path": ""}},
        }
    )
    payload = _mid_loop_payload()

    def build_manager():
        return VisualRewardManager(config=cfg, tokenizer=_DummyTokenizer(), compute_score=scorer.compute_score)

    result = _run_single(build_manager, rm._default_data_builder(payload))
    assert result["reward_score"] == pytest.approx(0.5)
    assert result["reward_extra_info"]["sample_scores"] == [pytest.approx(0.7), pytest.approx(0.3)]

    scores, flags = rm.parse_rm_result(result, payload["image_paths"])
    assert scores == [pytest.approx(0.7), pytest.approx(0.3)]
    assert flags == [True, False]


def test_unified_backend_scores_are_graded_and_fail_loud(monkeypatch, tmp_path):
    """The continuous backend must (a) keep within-group variance and (b) never zero-fill.

    Measured 2026-09-22: the facet judge put every GEN image on the 0.0 floor, so a whole
    FlowGRPO group scored identically and ``actor/loss`` was 0.0 on every step -- a reward
    with no variance cannot train a policy. UnifiedReward's graded 1-5 axes exist to keep
    that variance, so a regression to a constant score is the failure this test guards.
    """
    from verl_omni.utils.agentic import vllm_chat
    from verl_omni.utils.reward_score import bagel_rm_image_scorer as scorer_mod

    # Two images, two different quality levels -> 1-5 axes that must survive as 0.5 vs 0.0.
    replies = {
        "good.png": "Alignment Score (1-5): 4\nCoherence Score (1-5): 3\nStyle Score (1-5): 3\n",
        "bad.png": "Alignment Score (1-5): 1\nCoherence Score (1-5): 1\nStyle Score (1-5): 1\n",
    }
    seen: list[str] = []

    def fake_post(*, vllm_url, image_b64, prompt_text, max_tokens, model="", timeout=120.0,
                  enable_thinking=False):
        # Identify which image by its encoded payload.
        import base64

        name = base64.b64decode(image_b64).decode()
        seen.append(name)
        # The canonical prompt must be the one UnifiedReward expects.
        assert "Alignment Score (1-5)" in prompt_text and "Text Caption" in prompt_text
        return replies[name], None

    monkeypatch.setattr(vllm_chat, "post_vllm_chat", fake_post)

    paths = []
    for name in ("good.png", "bad.png"):
        p = tmp_path / name
        # Contents must differ per file: the stub identifies the image by what it read.
        p.write_bytes(name.encode())
        paths.append(str(p))

    knobs = {"good_enough_threshold": THRESHOLD, "score_backend": "unified_reward",
             "vllm_url": "http://rm-router:1", "unified_good_enough_threshold": 0.6}
    payload = rm.build_rm_score_payload(paths, extra_info={"user_prompt": "a castle"},
                                        scorer_knobs=knobs)
    payload["image_prompt"] = "a castle at dusk"
    out = scorer_mod.compute_score(
        data_source="bagel_corl_mid_loop_rm", solution_str="", ground_truth="",
        extra_info={rm.BAGEL_RM_EXTRA_INFO_KEY: payload},
    )

    # (4+3+3)/3 = 3.333 -> (3.333-1)/4 = 0.5833 ; (1+1+1)/3 = 1 -> 0.0
    assert out["sample_scores"] == [pytest.approx(7 / 12), pytest.approx(0.0)]
    assert out["sample_scores"][0] != out["sample_scores"][1], "reward collapsed: no GRPO gradient"
    assert out["sample_good_enough"] == [False, False]  # 0.5833 < 0.6, and 0.0 < 0.6
    assert out["score"] == pytest.approx(7 / 24)
    assert [os.path.basename(s) for s in seen] == ["good.png", "bad.png"]

    # Fail loud: an unparseable reply must raise, not silently score 0.0.
    monkeypatch.setattr(vllm_chat, "post_vllm_chat",
                        lambda **kw: ("I cannot help with that.", None))
    with pytest.raises(ValueError, match="failed to score"):
        scorer_mod.compute_score(
            data_source="bagel_corl_mid_loop_rm", solution_str="", ground_truth="",
            extra_info={rm.BAGEL_RM_EXTRA_INFO_KEY: payload},
        )


def test_episode_row_without_payload_dispatches_to_episode_scorer(monkeypatch):
    """Post-hoc episode rows carry no payload; dispatch must fall through."""
    calls: dict = {}

    def fake_episode_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        calls.update(data_source=data_source, ground_truth=ground_truth)
        return {"score": 0.25, "acc": 1.0}

    monkeypatch.setitem(
        sys.modules,
        "verl_omni.utils.reward_score.agentic_multidim_reward",
        type("M", (), {"compute_score": staticmethod(fake_episode_compute_score)}),
    )
    data = DataProto(
        batch=TensorDict(
            {
                "responses": torch.zeros((1, 1), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "data_source": np.array(["unicot"], dtype=object),
            "reward_model": np.array([{"ground_truth": "a castle"}], dtype=object),
        },
    )
    result = _run_single(lambda: _manager(scorer.compute_score), data)
    assert result["reward_score"] == pytest.approx(0.25)
    assert calls == {"data_source": "unicot", "ground_truth": "a castle"}
