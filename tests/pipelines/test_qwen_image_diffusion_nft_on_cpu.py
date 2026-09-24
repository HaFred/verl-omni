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
"""CPU tests for the Qwen-Image DiffusionNFT training adapter."""

import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.qwen_image_diffusion_nft.diffusers_training_adapter import QwenImageDiffusionNFT
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


def _write_checkpoint(tmp_path, config: dict) -> str:
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


def _model_config(local_path: str) -> DiffusionModelConfig:
    config = object.__new__(DiffusionModelConfig)
    object.__setattr__(config, "architecture", "QwenImagePipeline")
    object.__setattr__(config, "algorithm", "diffusion_nft")
    object.__setattr__(config, "external_lib", None)
    object.__setattr__(config, "local_path", local_path)
    object.__setattr__(config, "pipeline", DiffusionPipelineConfig(guidance_scale=1.0))
    return config


def _prepare(local_path: str, timesteps: torch.Tensor):
    micro_batch = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor_data(micro_batch, "height", 64)
    tu.assign_non_tensor_data(micro_batch, "width", 64)
    tu.assign_non_tensor_data(micro_batch, "vae_scale_factor", 8)
    module = SimpleNamespace(config=SimpleNamespace(guidance_embeds=False))
    return QwenImageDiffusionNFT.prepare_model_inputs(
        module=module,
        model_config=_model_config(local_path),
        latents=torch.zeros(1, 3, 8, 8),
        timesteps=timesteps,
        prompt_embeds=torch.ones(1, 4, 8),
        prompt_embeds_mask=torch.ones(1, 4, dtype=torch.bool),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )


@pytest.mark.parametrize(
    ("num_train_timesteps", "timestep", "expected"),
    [
        (1000, 500.0, 0.5),
        (2000, 500.0, 0.25),
        (2500, 1250.0, 0.5),
    ],
)
def test_timestep_uses_the_checkpoint_scale_not_a_literal(tmp_path, num_train_timesteps, timestep, expected):
    """The DiT must be conditioned at the scale the engine used to mix `xt`.

    `NFTDiffusersFSDPEngine.prepare_model_inputs` recovers flow time as
    `train_timesteps / scheduler.config.num_train_timesteps`. Dividing by a literal
    `1000.0` here instead fed the DiT a timestep that disagreed with its own noisy
    input for any checkpoint whose scheduler ships a different `num_train_timesteps`.
    The literal agreed only because every in-tree checkpoint happens to ship 1000.
    """
    local_path = _write_checkpoint(tmp_path, {"num_train_timesteps": num_train_timesteps})

    model_inputs, negative_model_inputs = _prepare(local_path, torch.tensor([timestep]))

    assert negative_model_inputs is None
    torch.testing.assert_close(model_inputs["timestep"], torch.tensor([expected]))


def test_checkpoint_without_a_timestep_scale_fails_loudly(tmp_path):
    """A missing scale is an error, never a silent `1000` default."""
    local_path = _write_checkpoint(tmp_path, {"shift": 3.0})

    with pytest.raises(ValueError, match="num_train_timesteps"):
        _prepare(local_path, torch.tensor([500.0]))
