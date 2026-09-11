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
"""CPU tests for Bagel Co-RL live GEN traj stash helpers (path-loaded, no CUDA init)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_gen_serve():
    root = Path(__file__).resolve().parents[2]
    omni = root / "verl_omni"
    if "verl_omni" not in sys.modules:
        pkg = types.ModuleType("verl_omni")
        pkg.__path__ = [str(omni)]
        sys.modules["verl_omni"] = pkg
    if "verl_omni.agent_loop" not in sys.modules:
        loop = types.ModuleType("verl_omni.agent_loop")
        loop.__path__ = [str(omni / "agent_loop")]
        sys.modules["verl_omni.agent_loop"] = loop
    name = "bagel_corl_gen_serve_isolated"
    if name in sys.modules:
        return sys.modules[name]
    path = omni / "agent_loop" / "bagel_corl_gen_serve.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


serve = _load_gen_serve()


def test_build_gen_sampling_params_merges_pipeline_algo():
    rollout = SimpleNamespace(
        pipeline=SimpleNamespace(height=256, width=256, num_inference_steps=4, output_type="image"),
        algo=SimpleNamespace(noise_level=0.7, sde_window_size=2, sde_window_range=[0, 7]),
        calculate_log_probs=True,
    )
    params = serve.build_gen_sampling_params(
        rollout,
        base={"temperature": 0.7, "top_p": 0.9, "global_steps": 3},
        seed=11,
    )
    assert params["num_inference_steps"] == 4
    assert params["noise_level"] == 0.7
    assert params["sde_window_size"] == 2
    assert params["logprobs"] is True
    assert params["seed"] == 11
    assert params["global_steps"] == 3
    assert "temperature" not in params


def test_build_gen_sampling_params_rejects_zero_noise():
    rollout = SimpleNamespace(
        pipeline={"num_inference_steps": 4},
        algo={"noise_level": 0.0},
        calculate_log_probs=True,
    )
    with pytest.raises(ValueError, match="noise_level"):
        serve.build_gen_sampling_params(rollout)


def test_build_gen_sampling_params_rejects_missing_calculate_log_probs():
    rollout = SimpleNamespace(
        pipeline={"num_inference_steps": 4},
        algo={"noise_level": 0.7},
    )
    with pytest.raises(ValueError, match="calculate_log_probs"):
        serve.build_gen_sampling_params(rollout)


def test_stash_gen_row_fail_closed_without_traj(tmp_path):
    output = SimpleNamespace(
        diffusion_output=torch.zeros(3, 8, 8, dtype=torch.uint8),
        log_probs=None,
        stop_reason="stop",
        extra_fields={"global_steps": 1},
    )
    with pytest.raises(RuntimeError, match="traj stash incomplete"):
        serve.stash_gen_row_from_diffusion_output(output, seed=0, image_root=str(tmp_path))


def test_stash_gen_row_extracts_traj_and_saves_png(tmp_path):
    latents = torch.randn(1, 2, 4)  # leftover batch axis from pipeline
    timesteps = torch.tensor([[900.0, 500.0]])
    log_probs = torch.tensor([[0.1, 0.2]])
    pixels = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
    output = SimpleNamespace(
        diffusion_output=pixels,
        log_probs=log_probs,
        stop_reason="stop",
        extra_fields={"all_latents": latents, "all_timesteps": timesteps},
    )
    row = serve.stash_gen_row_from_diffusion_output(output, seed=7, image_root=str(tmp_path))
    assert row["valid"] is True
    assert torch.equal(row["all_latents"], latents[0])
    assert torch.equal(row["timesteps"], timesteps[0])
    assert torch.equal(row["rollout_log_probs"], log_probs[0])
    assert Path(row["image_path"]).is_file()


def test_extract_gen_traj_prefers_extra_fields():
    output = SimpleNamespace(
        log_probs=torch.tensor([1.0, 2.0]),
        extra_fields={
            "all_latents": torch.ones(3, 2),
            "all_timesteps": torch.tensor([1.0, 2.0, 3.0]),
        },
    )
    latents, timesteps, log_probs = serve.extract_gen_traj_from_diffusion_output(output)
    assert latents.shape == (3, 2)
    assert timesteps.tolist() == [1.0, 2.0, 3.0]
    assert log_probs.tolist() == [1.0, 2.0]
