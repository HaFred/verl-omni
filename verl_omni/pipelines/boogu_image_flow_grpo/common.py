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

"""Shared utilities for Boogu-Image FlowGRPO training and rollout pipelines."""

import torch

BOOGU_IMAGE_VAE_SCALE_FACTOR = 8


def coalesce_not_none(value, default):
    """Return *default* when *value* is ``None``, otherwise *value*."""
    return default if value is None else value


def build_img_shapes(
    height: int, width: int, batch_size: int, vae_scale_factor: int
) -> list[list[tuple[int, int, int]]]:
    """Build per-sample image shapes for the Boogu transformer.

    Boogu uses patch_size=2, so latent dimensions are divided by vae_scale_factor
    and then again by 2 (patch_size).
    """
    latent_height = height // vae_scale_factor // 2
    latent_width = width // vae_scale_factor // 2
    return [[(1, latent_height, latent_width)]] * batch_size


def apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Apply True-CFG guidance with norm-preserving scaling."""
    comb_pred = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)
