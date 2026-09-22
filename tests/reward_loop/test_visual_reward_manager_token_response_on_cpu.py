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
"""CPU tests for the token-trajectory branch of ``VisualRewardManager``.

The manager serves two response kinds in the omni tree: pixels (image/latent lanes) and
token ids (the Bagel Joint-Training lane, whose episode is a Hermes text trajectory). The
pixel dtype guard used to run for both, so a Bagel episode raised
``Expected uint8 pixel responses for output_type='image', got torch.int64`` inside the
reward worker -- failing every episode and taking the run down through the sync replay
buffer (measured 2026-09-22, ``ENABLE_RM=1``).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.protocol import DataProto

from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager, _is_token_response


class _FakeTokenizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.seen: list = []

    def decode(self, ids, skip_special_tokens=False):
        self.seen.append(list(ids))
        return self.mapping.get(tuple(ids), "decoded-text")


def _config(output_type="image"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "pipeline": {"output_type": output_type},
                    "val_kwargs": {"pipeline": {"output_type": output_type}},
                }
            },
            "reward": {"reward_model": {"model_path": "/ckpt/rm"}},
        }
    )


def _data(responses, *, extra_info=None, data_source="bagel_corl"):
    return DataProto(
        batch=TensorDict(
            {
                "responses": responses,
                "attention_mask": torch.ones((1, responses.shape[-1]), dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "data_source": np.array([data_source], dtype=object),
            "reward_model": np.array([{"ground_truth": "gt"}], dtype=object),
            "extra_info": np.array([extra_info if extra_info is not None else {}], dtype=object),
        },
    )


def _manager(compute_score, tokenizer, *, config=None, router=None):
    return VisualRewardManager(
        config if config is not None else _config(),
        tokenizer,
        compute_score,
        reward_router_address=router,
    )


def test_token_ids_are_detected_and_pixels_are_not():
    assert _is_token_response(torch.zeros(1, 4, dtype=torch.int64))
    assert _is_token_response(torch.zeros(1, 4, dtype=torch.int32))
    # Pixels and latents keep riding the image branch.
    assert not _is_token_response(torch.zeros(1, 3, 8, 8, dtype=torch.uint8))
    assert not _is_token_response(torch.zeros(1, 3, 8, 8, dtype=torch.float32))
    assert not _is_token_response(None)


def test_a_token_trajectory_is_decoded_and_scored_as_text():
    seen: dict = {}

    async def compute_score(**kwargs):
        seen.update(kwargs)
        return {"score": 0.75, "acc": 0.75}

    tokenizer = _FakeTokenizer()
    manager = _manager(compute_score, tokenizer)
    data = _data(torch.tensor([[11, 22, 33]], dtype=torch.int64))

    out = asyncio.run(manager.run_single(data))

    # The id tensor became text; it was NOT handed over as `solution_image`.
    assert "solution_str" in seen
    assert "solution_image" not in seen
    assert tokenizer.seen == [[11, 22, 33]]
    assert out["reward_score"] == pytest.approx(0.75)


def test_token_trajectory_does_not_raise_the_pixel_dtype_guard():
    """The exact failure that killed the run: int64 ids into an image-output config."""

    async def compute_score(**kwargs):
        return {"score": 0.1}

    manager = _manager(compute_score, _FakeTokenizer())
    data = _data(torch.randint(0, 100, (1, 512), dtype=torch.int64))
    out = asyncio.run(manager.run_single(data))
    assert out["reward_score"] == pytest.approx(0.1)


def test_pixels_still_take_the_image_branch_and_are_still_guarded():
    seen: dict = {}

    async def compute_score(**kwargs):
        seen.update(kwargs)
        return {"score": 0.5}

    manager = _manager(compute_score, _FakeTokenizer())
    pixels = torch.randint(0, 255, (1, 3, 8, 8), dtype=torch.uint8)
    asyncio.run(manager.run_single(_data(pixels)))
    assert "solution_image" in seen
    assert "solution_str" not in seen

    # A float tensor under an image config is still a hard error (the guard's real job).
    with pytest.raises(ValueError, match="Expected uint8 pixel responses"):
        asyncio.run(manager.run_single(_data(torch.zeros(1, 3, 8, 8, dtype=torch.float32))))


def test_a_latent_config_still_requires_a_float_response():
    async def compute_score(**kwargs):
        return {"score": 0.5}

    manager = _manager(compute_score, _FakeTokenizer(), config=_config("latent"))
    asyncio.run(manager.run_single(_data(torch.zeros(1, 4, dtype=torch.float32))))
    with pytest.raises(ValueError, match="Expected floating-point latent responses"):
        asyncio.run(manager.run_single(_data(torch.zeros(1, 4, dtype=torch.uint8))))


def test_the_reward_pool_router_reaches_the_scorer_on_the_token_branch():
    """The internal RM route is handed to the scorer on both branches."""
    seen: dict = {}

    async def compute_score(**kwargs):
        seen.update(kwargs)
        return {"score": 0.2}

    manager = _manager(compute_score, _FakeTokenizer(), router="10.0.0.9:9000")
    asyncio.run(manager.run_single(_data(torch.tensor([[7]], dtype=torch.int64))))
    assert seen["reward_router_address"] == "10.0.0.9:9000"
    assert seen["model_name"] == "/ckpt/rm"


def test_no_router_leaves_the_scorer_extras_untouched():
    seen: dict = {}

    async def compute_score(**kwargs):
        seen.update(kwargs)
        return {"score": 0.2}

    manager = _manager(compute_score, _FakeTokenizer())
    asyncio.run(manager.run_single(_data(torch.tensor([[7]], dtype=torch.int64))))
    assert "reward_router_address" not in seen
    assert "model_name" not in seen
