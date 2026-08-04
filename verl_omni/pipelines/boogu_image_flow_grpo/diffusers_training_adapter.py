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

"""
Boogu-Image training-side adapter for diffusers-based diffusion RL (FlowGRPO).
"""

from typing import Optional

import numpy as np
import torch
from diffusers.models.transformers.transformer_boogu import BooguImageTransformer2DModel, get_freqs_cis
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import BOOGU_IMAGE_VAE_SCALE_FACTOR, apply_true_cfg, build_img_shapes

__all__ = ["BooguImage"]


# ---------------------------------------------------------------------------
# Boogu-specific 0→1 sigma schedule helpers
# ---------------------------------------------------------------------------

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute the resolution-dependent time shift *mu* for Boogu."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _build_boogu_sigmas(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_inference_steps: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Build Boogu's training-aligned 0→1 sigma schedule.

    Boogu trains with an **inverted** schedule where sigmas run 0 → 1
    (standard flow-matching runs 1 → 0).  The returned sigmas are installed
    through ``scheduler.set_timesteps(sigmas=...)``.
    """
    seq_len = (height // BOOGU_IMAGE_VAE_SCALE_FACTOR) * (width // BOOGU_IMAGE_VAE_SCALE_FACTOR)
    mu = _calculate_shift(
        seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    t = np.linspace(0.0, 1.0, num_inference_steps + 1, dtype=np.float32)[:-1]
    sigmas = (
        1.0 - scheduler._time_shift_exponential(mu, 1.0, 1.0 - torch.from_numpy(t))
    ).numpy()
    return sigmas


def _build_boogu_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    """Create an SDE scheduler loaded from the Boogu checkpoint."""
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


def _configure_boogu_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    device: str,
) -> None:
    sigmas = _build_boogu_sigmas(scheduler, num_inference_steps, height, width)
    scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
    # set_timesteps appends a terminal sigma of 0.0; Boogu's 0→1 schedule ends at 1.0.
    scheduler.sigmas = torch.cat(
        [scheduler.sigmas[:-1], torch.ones(1, device=scheduler.sigmas.device)]
    )


# ---------------------------------------------------------------------------
# Training adapter
# ---------------------------------------------------------------------------


@DiffusionModelBase.register("BooguImagePipeline", algorithm="flow_grpo")
class BooguImage(DiffusionModelBase):
    """Training adapter for the Boogu-Image diffusion model.

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for the ``BooguImagePipeline`` architecture, providing scheduler
    configuration (Boogu-specific 0→1 sigma schedule), model-input construction,
    and the forward/sampling step used during FlowGRPO training.

    Registered under ``"BooguImagePipeline"`` so it is automatically selected
    when ``DiffusionModelConfig.architecture`` matches that name.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = _build_boogu_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ):
        _configure_boogu_scheduler(
            scheduler,
            height=model_config.pipeline.height,
            width=model_config.pipeline.width,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module: BooguImageTransformer2DModel,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, dict]:
        """Build Boogu-Image-specific inputs for the transformer forward pass.

        Key differences from Qwen-Image:
        - Timestep is the raw 0→1 sigma (not timestep/1000).
        - ``hidden_states`` is a **list** of tensors (one per sample) for
          variable-resolution support.
        - ``freqs_cis`` (3-axis RoPE frequencies) are built from the
          transformer config and passed to every forward.
        - ``ref_image_hidden_states`` is ``None`` for pure T2I.
        - ``instruction_hidden_states`` = prompt_embeds,
          ``instruction_attention_mask`` = prompt_embeds_mask.
        """
        height = tu.get_non_tensor_data(data=micro_batch, key="height", default=None)
        width = tu.get_non_tensor_data(data=micro_batch, key="width", default=None)
        vae_scale_factor = tu.get_non_tensor_data(data=micro_batch, key="vae_scale_factor", default=None)

        # Boogu timestep: convert scheduler timestep → raw sigma [0, 1].
        num_train_timesteps = 1000  # FlowMatchEulerDiscreteScheduler default
        timestep_sigma = (timesteps[:, step].float() / num_train_timesteps).to(latents.device)

        # Boogu transformer expects hidden_states as list of per-sample tensors.
        hidden_states_batch = latents[:, step]  # [B, C, H, W]
        hidden_states_list = [hidden_states_batch[i] for i in range(hidden_states_batch.shape[0])]

        # Build 3-axis RoPE frequency tables from transformer config.
        freqs_cis = get_freqs_cis(
            axes_dim=module.config.axes_dim_rope,
            axes_lens=module.config.axes_lens,
            theta=10000,
        )

        model_inputs_base = {
            "hidden_states": hidden_states_list,
            "timestep": timestep_sigma,
            "instruction_hidden_states": prompt_embeds,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": prompt_embeds_mask,
            "ref_image_hidden_states": None,  # T2I only; TI2I added in follow-up
            "return_dict": False,
        }

        model_inputs = dict(model_inputs_base)
        negative_model_inputs = {
            **model_inputs_base,
            "instruction_hidden_states": negative_prompt_embeds,
            "instruction_attention_mask": negative_prompt_embeds_mask,
        }

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: BooguImageTransformer2DModel,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Run the Boogu transformer and sample the previous denoising step.

        Uses CPS (Consistency-Preserving Sampling) as the SDE type because
        Boogu's 0→1 inverted sigma schedule makes the standard "sde" variant's
        ``sqrt(-dt)`` term NaN.

        Args:
            module: The Boogu-Image transformer module.
            scheduler: SDE scheduler with Boogu's 0→1 sigmas set.
            model_config: Configuration providing ``true_cfg_scale``,
                ``algo.noise_level``, and ``algo.sde_type``.
            model_inputs: Positive-prompt inputs for transformer forward.
            negative_model_inputs: Negative-prompt inputs for True-CFG.
            scheduler_inputs: Must contain ``"all_latents"`` and
                ``"all_timesteps"`` tensors.
            step: Current denoising step index.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0:
            assert negative_model_inputs is not None
            neg_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

        (
            _,
            log_prob,
            prev_sample_mean,
            std_dev_t,
            sqrt_dt,
        ) = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
