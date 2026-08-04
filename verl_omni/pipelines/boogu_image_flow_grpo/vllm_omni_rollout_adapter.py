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

"""Boogu-Image rollout pipeline for vLLM-Omni with SDE-based log-prob collection."""

import copy
import os
from dataclasses import replace
from typing import Any, Literal

import torch
from diffusers.models.transformers.transformer_boogu import get_freqs_cis
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.request_batch import (
    collate_prompt_mask as _collate_prompt_mask,
)
from verl_omni.pipelines.request_batch import (
    collate_prompt_rows as _collate_prompt_rows,
)
from verl_omni.pipelines.request_batch import (
    sample_per_sample_sde_windows as _sample_per_sample_sde_windows,
)
from verl_omni.pipelines.request_batch import (
    split_diffusion_output_by_request as _split_diffusion_output_by_request,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import BOOGU_IMAGE_VAE_SCALE_FACTOR, apply_true_cfg, build_img_shapes, coalesce_not_none

__all__ = ["BooguImagePipelineWithLogProb"]


# ---------------------------------------------------------------------------
# Boogu-specific sigma schedule (0→1, inverted vs standard flow-matching)
# ---------------------------------------------------------------------------


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _set_boogu_sigmas_on_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_inference_steps: int,
    height: int,
    width: int,
    device: str,
) -> None:
    import numpy as np

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
    scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
    scheduler.sigmas = torch.cat(
        [scheduler.sigmas[:-1], torch.ones(1, device=scheduler.sigmas.device)]
    )


# ---------------------------------------------------------------------------
# Rollout pipeline
# ---------------------------------------------------------------------------


@VllmOmniPipelineBase.register("BooguImagePipeline", algorithm="flow_grpo")
class BooguImagePipelineWithLogProb(VllmOmniPipelineBase):
    """Rollout pipeline for Boogu-Image that captures per-step log-probabilities.

    Uses CPS (Consistency-Preserving Sampling) SDE to compute log-probs on
    Boogu's 0→1 inverted sigma schedule.  The pipeline encodes text prompts
    through the Qwen3-VL MLLM and runs the Boogu transformer for denoising.

    Registered under ``"BooguImagePipeline"`` for vLLM-Omni rollout dispatch.
    """

    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model_path = od_config.model
        local_files_only = os.path.exists(model_path)

        # --- Load pipeline components ---
        from diffusers import AutoencoderKL
        from diffusers.pipelines.boogu.pipeline_boogu import BooguImagePipeline
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        self.transformer = BooguImagePipeline.from_pretrained(
            model_path, subfolder="transformer", local_files_only=local_files_only
        ).transformer
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", local_files_only=local_files_only
        ).to(self.device).eval()
        self.mllm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, subfolder="mllm", local_files_only=local_files_only
        ).to(self.device).eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path, subfolder="processor", local_files_only=local_files_only
        )

        self.vae_scale_factor = BOOGU_IMAGE_VAE_SCALE_FACTOR
        self.default_sample_size = 128

        # --- SDE scheduler ---
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # --- System prompts (from Boogu pipeline) ---
        self.SYSTEM_PROMPT_4_T2I = (
            "You are a helpful assistant that generates high-quality images "
            "based on user instructions. The instructions are as follows."
        )

    # ------------------------------------------------------------------
    # Prompt encoding (Qwen3-VL MLLM)
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1280,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt(s) into instruction embeddings via the Qwen3-VL MLLM.

        Returns:
            (prompt_embeds [B*N, L, D], prompt_embeds_mask [B*N, L])
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # Build chat prompts (T2I system prompt + user instruction).
        chat_prompts = []
        for p in prompt:
            chat_prompts.append([
                {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT_4_T2I}]},
                {"role": "user", "content": [{"type": "text", "text": p}]},
            ])

        vlm_inputs = self.processor.apply_chat_template(
            chat_prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        for k in vlm_inputs:
            if isinstance(vlm_inputs[k], torch.Tensor):
                vlm_inputs[k] = vlm_inputs[k].to(self.device)

        instruction_mask = vlm_inputs["attention_mask"]

        with torch.no_grad():
            instruction_feats = self.mllm(**vlm_inputs).last_hidden_state

        instruction_feats = instruction_feats.to(dtype=self.transformer.dtype)
        instruction_mask = instruction_mask.to(device=self.device)

        if num_images_per_prompt > 1:
            instruction_feats = instruction_feats.repeat_interleave(num_images_per_prompt, dim=0)
            instruction_mask = instruction_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return instruction_feats, instruction_mask

    def encode_negative_prompt(
        self,
        batch_size: int,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 1280,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode empty/negative prompt for CFG."""
        negative_prompt = [""] * batch_size
        return self.encode_prompt(negative_prompt, num_images_per_prompt, max_sequence_length)

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            from diffusers.utils.torch_utils import randn_tensor
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    # ------------------------------------------------------------------
    # Diffuse (full denoising loop with SDE log-prob collection)
    # ------------------------------------------------------------------

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        latents: torch.Tensor,
        height: int,
        width: int,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        true_cfg_scale: float,
        noise_level: float,
        sde_window: tuple[int, int] | list[tuple[int, int]],
        sde_type: str,
        generator: torch.Generator | None,
        logprobs: bool,
    ):
        """Run the CPS-SDE diffusion loop and collect per-step rollout data."""
        batch_size = latents.shape[0]
        windows = [sde_window] * batch_size if isinstance(sde_window, tuple) else list(sde_window)
        if len(windows) != batch_size:
            raise ValueError(f"Expected {batch_size} SDE windows, got {len(windows)}.")
        if len({end - start for start, end in windows}) != 1:
            raise ValueError("Packed SDE windows must share the same size.")

        all_latents: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        all_log_probs: list[list[Any]] = [[] for _ in range(batch_size)]
        all_timesteps_list: list[list[Any]] = [[] for _ in range(batch_size)]

        # Build RoPE frequency tables once.
        freqs_cis = get_freqs_cis(
            axes_dim=self.transformer.config.axes_dim_rope,
            axes_lens=self.transformer.config.axes_lens,
            theta=10000,
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for i, timestep_value in enumerate(timesteps):
            for batch_idx, (start, end) in enumerate(windows):
                if i == start:
                    all_latents[batch_idx].append(latents[batch_idx].detach().float().clone())

            levels = [float(noise_level) if start <= i < end else 0.0 for start, end in windows]
            cur_noise_level: float | torch.Tensor = (
                levels[0]
                if all(level == levels[0] for level in levels)
                else torch.tensor(levels, device=latents.device, dtype=torch.float32).view(
                    batch_size, *([1] * (latents.ndim - 1))
                )
            )

            # Convert latents to list for transformer.
            latents_list = [latents[b] for b in range(batch_size)]

            # Timestep sigma [0, 1].
            timestep_sigma = timestep_value.float() / num_train_timesteps
            timestep_batch = timestep_sigma.expand(batch_size).to(device=latents.device, dtype=latents.dtype)

            # Positive forward.
            noise_pred = self.transformer(
                latents_list,
                timestep_batch,
                prompt_embeds,
                freqs_cis,
                prompt_embeds_mask,
                ref_image_hidden_states=None,
                return_dict=False,
            )

            if do_true_cfg and negative_prompt_embeds is not None:
                neg_noise_pred = self.transformer(
                    latents_list,
                    timestep_batch,
                    negative_prompt_embeds,
                    freqs_cis,
                    negative_prompt_embeds_mask,
                    ref_image_hidden_states=None,
                    return_dict=False,
                )
                noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

            # SDE step (CPS type for Boogu's 0→1 schedule).
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32),
                timestep_value,
                latents.to(torch.float32),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            for batch_idx, (start, end) in enumerate(windows):
                if start <= i < end:
                    all_latents[batch_idx].append(latents[batch_idx].detach().to(torch.float32).clone())
                    all_log_probs[batch_idx].append(None if log_prob is None else log_prob[batch_idx])
                    all_timesteps_list[batch_idx].append(timestep_value)

        all_latents_t = torch.stack([torch.stack(traj, dim=0) for traj in all_latents], dim=0)
        if all_log_probs and all_log_probs[0] and all_log_probs[0][0] is not None:
            all_log_probs_t = torch.stack([torch.stack(traj, dim=0) for traj in all_log_probs], dim=0)
        else:
            all_log_probs_t = None
        all_timesteps_t = torch.stack(
            [torch.stack(traj, dim=0) for traj in all_timesteps_list], dim=0
        )
        return latents, all_latents_t, all_log_probs_t, all_timesteps_t

    # ------------------------------------------------------------------
    # vLLM-Omni step-execution interface
    # ------------------------------------------------------------------

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Populate *state* with encoded prompts, latents, timesteps, and SDE config."""
        sampling = state.sampling
        prompt = state.prompt

        if prompt is None:
            raise ValueError(
                f"{self.__class__.__name__}.prepare_encode requires a prompt on state."
            )

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling.num_inference_steps or 50
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        true_cfg_scale = sampling.true_cfg_scale or 4.0
        max_sequence_length = sampling.max_sequence_length or 1280

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

        # Encode prompt.
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            str(prompt),
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        has_neg = getattr(prompt, "negative_prompt", None) or getattr(sampling, "negative_prompt", None)
        do_true_cfg = true_cfg_scale > 1 and has_neg
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_negative_prompt(
                1, num_images_per_prompt, max_sequence_length
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        # Prepare latents.
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            1 * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            self.device,
            generator,
            None,
        )

        # Configure scheduler with Boogu sigmas.
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)
        _set_boogu_sigmas_on_scheduler(req_scheduler, num_inference_steps, height, width, self.device)

        timesteps = req_scheduler.timesteps

        # SDE knobs.
        extra = sampling.extra_args or {}
        noise_level = coalesce_not_none(extra.get("noise_level"), 0.7)
        sde_window_size = coalesce_not_none(extra.get("sde_window_size"), None)
        sde_window_range = coalesce_not_none(extra.get("sde_window_range"), (0, 5))
        sde_type = coalesce_not_none(extra.get("sde_type"), "cps")
        logprobs = coalesce_not_none(extra.get("logprobs"), True)

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            sde_window = (start, start + sde_window_size)
        else:
            sde_window = (0, len(timesteps) - 1)

        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_embeds_mask
        state.negative_prompt_embeds = negative_prompt_embeds
        state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = do_true_cfg
        state.true_cfg_scale = true_cfg_scale
        state.height = height
        state.width = width
        state.sde_window = sde_window
        state.noise_level = noise_level
        state.sde_type = sde_type
        state.logprobs = logprobs
        state.all_latents = []
        state.all_log_probs = []
        state.all_timesteps_list = []
        state.sampling.generator = generator
        state.sampling.cfg_normalize = True
        state.sampling.num_images_per_prompt = num_images_per_prompt
        state.sampling.max_sequence_length = max_sequence_length

        return state

    def denoise_step(self, input_batch, **kwargs):
        """Single transformer forward for step-execution mode."""
        del kwargs
        if self._interrupt if hasattr(self, "_interrupt") else False:
            return None

        t = input_batch.timesteps
        batch_size = input_batch.latents.shape[0]

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        timestep_sigma = t.float() / num_train_timesteps
        timestep_batch = timestep_sigma.expand(batch_size).to(device=input_batch.latents.device)

        latents_list = [input_batch.latents[b] for b in range(batch_size)]

        freqs_cis = get_freqs_cis(
            axes_dim=self.transformer.config.axes_dim_rope,
            axes_lens=self.transformer.config.axes_lens,
            theta=10000,
        )

        noise_pred = self.transformer(
            latents_list,
            timestep_batch,
            input_batch.prompt_embeds,
            freqs_cis,
            input_batch.prompt_embeds_mask,
            ref_image_hidden_states=None,
            return_dict=False,
        )

        if input_batch.do_true_cfg and input_batch.negative_prompt_embeds is not None:
            neg_noise_pred = self.transformer(
                latents_list,
                timestep_batch,
                input_batch.negative_prompt_embeds,
                freqs_cis,
                input_batch.negative_prompt_embeds_mask,
                ref_image_hidden_states=None,
                return_dict=False,
            )
            noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, input_batch.true_cfg_scale)

        return noise_pred.float()

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One CPS-SDE scheduler step for step-execution mode."""
        del kwargs
        if getattr(self, "_interrupt", False):
            return

        i = state.step_index
        timestep_value = state.timesteps[i]
        sde_window = state.sde_window

        if i < sde_window[0]:
            cur_noise_level = 0.0
        elif i == sde_window[0]:
            cur_noise_level = state.noise_level
            state.all_latents.append(state.latents.to(torch.float32))
        elif sde_window[0] < i < sde_window[1]:
            cur_noise_level = state.noise_level
        else:
            cur_noise_level = 0.0

        new_latents, log_prob, _, _ = state.scheduler.step(
            noise_pred.to(torch.float32),
            timestep_value,
            state.latents.to(torch.float32),
            generator=state.sampling.generator,
            noise_level=cur_noise_level,
            sde_type=state.sde_type,
            return_logprobs=state.logprobs,
            return_dict=False,
        )

        if sde_window[0] <= i < sde_window[1]:
            state.all_latents.append(new_latents.to(torch.float32))
            state.all_log_probs.append(log_prob)
            state.all_timesteps_list.append(timestep_value)

        state.latents = new_latents.to(torch.float32)
        state.step_index += 1

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents and package rollout trajectory for training."""
        del kwargs
        latents_to_decode = state.latents.to(self.vae.dtype)

        if self.vae.config.scaling_factor is not None:
            latents_to_decode = latents_to_decode / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents_to_decode = latents_to_decode + self.vae.config.shift_factor

        # Unpack from [B, C, T, H, W] if needed.
        if latents_to_decode.ndim == 5:
            latents_to_decode = latents_to_decode[:, :, 0]

        image = self.vae.decode(latents_to_decode, return_dict=False)[0][:, :, 0]

        all_latents = state.all_latents
        all_log_probs = state.all_log_probs
        all_timesteps_list = state.all_timesteps_list

        stacked_latents = torch.stack(all_latents, dim=1) if all_latents else None
        stacked_log_probs = (
            torch.stack(all_log_probs, dim=1)
            if all_log_probs and all_log_probs[0] is not None
            else None
        )
        stacked_timesteps = (
            torch.stack(all_timesteps_list).unsqueeze(0).expand(state.latents.shape[0], -1)
            if all_timesteps_list
            else None
        )

        return DiffusionOutput(
            output=image,
            custom_output={
                "all_latents": stacked_latents,
                "all_log_probs": stacked_log_probs,
                "all_timesteps": stacked_timesteps,
                "prompt_embeds": state.prompt_embeds,
                "prompt_embeds_mask": state.prompt_embeds_mask,
                "negative_prompt_embeds": state.negative_prompt_embeds,
                "negative_prompt_embeds_mask": state.negative_prompt_embeds_mask,
            },
            to_cpu=True,
        )

    # ------------------------------------------------------------------
    # Main forward (request-mode)
    # ------------------------------------------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        prompt_token_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 1280,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["cps"] = "cps",
        logprobs: bool = True,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """End-to-end image generation with SDE rollout data collection.

        Args:
            req: One rollout request or a request batch.
            true_cfg_scale: CFG scale for True-CFG.
            height, width: Output image dimensions.
            num_inference_steps: Number of denoising steps.
            num_images_per_prompt: Images per prompt.
            noise_level: SDE noise injection magnitude.
            sde_window_size: Number of SDE steps.
            sde_window_range: ``(start, end)`` range for random window placement.
            sde_type: SDE variant; ``"cps"`` is required for Boogu's 0→1 schedule.
            logprobs: Whether to compute per-step log-probabilities.

        Returns:
            DiffusionOutput with generated image and trajectory data.
        """
        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)

        sampling_params = request_batch.sampling_params_list[0]
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        noise_level = coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = coalesce_not_none(sampling_params.extra_args.get("sde_window_range", None), sde_window_range)
        sde_type = coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)
        true_cfg_scale = coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)

        for request in request_batch.requests:
            rs = request.sampling_params
            if rs.generator is None and rs.seed is not None:
                rs.generator = torch.Generator(device=self.device).manual_seed(rs.seed)

        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs
        generator = request_batch.collate_request_generators(num_images_per_prompt, generator)
        latents = request_batch.collate_request_tensors("latents", latents)

        # Get prompts from request batch.
        prompts = request_batch.prompts
        prompt_texts = []
        for p in prompts:
            if isinstance(p, dict):
                prompt_texts.append(p.get("prompt", ""))
            elif isinstance(p, str):
                prompt_texts.append(p)
            else:
                prompt_texts.append(str(p))

        batch_size = len(prompt_texts)

        # Encode prompts.
        all_prompt_embeds = []
        all_prompt_embeds_masks = []
        for pt in prompt_texts:
            pe, pem = self.encode_prompt(pt, num_images_per_prompt, max_sequence_length)
            all_prompt_embeds.append(pe)
            all_prompt_embeds_masks.append(pem)
        prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
        prompt_embeds_mask = torch.cat(all_prompt_embeds_masks, dim=0)

        do_true_cfg = true_cfg_scale > 1
        if do_true_cfg:
            neg_embeds, neg_masks = self.encode_negative_prompt(batch_size, num_images_per_prompt, max_sequence_length)
            negative_prompt_embeds = neg_embeds
            negative_prompt_embeds_mask = neg_masks
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        # Prepare latents.
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            self.device,
            generator,
            latents,
        )

        # Configure scheduler.
        self.scheduler.set_begin_index(0)
        _set_boogu_sigmas_on_scheduler(self.scheduler, num_inference_steps, height, width, self.device)
        timesteps = self.scheduler.timesteps

        # SDE window.
        sde_window = _sample_per_sample_sde_windows(
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
            num_timesteps=len(timesteps),
            batch_size=latents.shape[0],
            generator=generator,
            device=self.device,
        )

        # Run diffuse.
        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            height,
            width,
            timesteps,
            do_true_cfg,
            true_cfg_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        # Decode.
        if output_type != "latent":
            latents_decode = latents.to(self.vae.dtype)
            if self.vae.config.scaling_factor is not None:
                latents_decode = latents_decode / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                latents_decode = latents_decode + self.vae.config.shift_factor
            image = self.vae.decode(latents_decode, return_dict=False)[0][:, :, 0]
        else:
            image = latents

        result = DiffusionOutput(
            output=image,
            custom_output={
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "all_timesteps": all_timesteps,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            },
            to_cpu=True,
        )
        outputs = _split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_images_per_prompt,
        )
        return outputs if return_batch else outputs[0]
