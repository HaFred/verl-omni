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

"""Shared helpers for the Boogu-Image FlowGRPO adapters.

The rollout adapter (vllm-omni) and the training adapter (diffusers) must
handle Boogu-Image with identical conventions — otherwise the rollout
trajectory and the training-time log-probs diverge and RL silently breaks.

Convention differences
----------------------

Boogu-Image runs flow matching "backwards" relative to diffusers:

                        Boogu-Image            diffusers (FlowMatchSDEDiscreteScheduler)
  timestep direction    t: 0 -> 1 (0 = noise)  sigma: 1 -> 0 (1 = noise)
  velocity target       v_boogu = x0 - noise   v_diffusers = noise - x0

Both adapters therefore apply the same two fix-ups:

1. Timestep mapping: t = 1 - sigma, i.e. t = 1 - timestep / num_train_timesteps
   for the scheduler-native timestep values that ride the rollout trajectory
   (see boogu_timestep_from_scheduler).
2. Velocity negation: the transformer output is negated before it enters the
   scheduler. Under this mapping Boogu's Euler update x + (t_next - t) * v_boogu
   is identical to the diffusers update x + (sigma_prev - sigma) * v_diffusers.

Boogu's native scheduler remains the source of truth for time shifting. Its
ascending timesteps are bridged to the SDE scheduler with ``sigma = 1 - t``;
the latter then only supplies stochastic sampling and log probabilities.
"""

from typing import Any, Optional

import torch


def build_boogu_native_scheduler(model_path: str):
    """Load the checkpoint's native scheduler, including Boogu shift config."""
    from vllm_omni.diffusion.models.boogu_image.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )

    return FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")


def configure_boogu_sde_timesteps(
    scheduler,
    *,
    native_scheduler,
    num_inference_steps: int,
    num_tokens: int,
    device,
) -> None:
    """Bridge native Boogu timesteps into the log-probability SDE scheduler."""
    native_scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        device=device,
        num_tokens=num_tokens,
    )
    sigmas = (1.0 - native_scheduler.timesteps).detach().cpu().numpy()

    # Native Boogu already applied the checkpoint's v1/v2 shift. Disable both
    # diffusers shift paths so the supplied sigmas are not transformed again.
    scheduler.register_to_config(use_dynamic_shifting=False, shift=1.0)
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=sigmas,
    )


def boogu_timestep_from_scheduler(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Map scheduler-native timestep values (sigma * N, descending) to Boogu t (1 - sigma)."""
    return 1.0 - timestep.float() / num_train_timesteps


def apply_boogu_text_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Boogu's sequential text CFG (standard formula, no renormalisation)."""
    return noise_pred + (guidance_scale - 1.0) * (noise_pred - negative_noise_pred)


_FREQS_CIS_CACHE: dict[tuple, Any] = {}


def get_boogu_freqs_cis(axes_dim_rope, axes_lens, theta: int = 10000):
    """Build (and cache) the rotary tables the Boogu transformer consumes.

    Prefers the canonical implementation from the installed boogu-image
    package (the training-side transformer is the canonical class, so its
    rope tables must come from the same code); falls back to the verbatim
    vllm-omni port, which the rollout-side transformer uses.
    """
    key = (tuple(axes_dim_rope), tuple(axes_lens), theta)
    if key in _FREQS_CIS_CACHE:
        return _FREQS_CIS_CACHE[key]

    try:
        from boogu.models.transformers.rope import BooguImageRotaryPosEmbed as _Rope
    except ImportError:
        from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
            BooguImageDoubleStreamRotaryPosEmbed as _Rope,
        )

    freqs_cis = _Rope.get_freqs_cis(list(axes_dim_rope), list(axes_lens), theta=theta)
    _FREQS_CIS_CACHE[key] = freqs_cis
    return freqs_cis


def resolve_text_guidance_scale(guidance_scale: Optional[float]) -> float:
    """Map a possibly-unset config guidance scale to Boogu's default (4.0)."""
    return 4.0 if guidance_scale is None else float(guidance_scale)


# ---------------------------------------------------------------------------
# LoRA name translation (diffusers -> vllm-omni)
# ---------------------------------------------------------------------------

# The actor trains against the diffusers naming, where the attention output
# projection lives inside an ``nn.Sequential`` (``attn.to_out.0``); the
# vllm-omni Boogu transformer exposes a direct ``attn.to_out``. That single
# target is the *only* divergence: the q/k/v projections, the joint
# attention's per-stream outputs and both feed-forward stacks all match
# verbatim. Because they match, the adapter is never dropped wholesale --
# vllm-omni only warns when *nothing* binds -- so the o-proj delta used to be
# exported, bound to zero rollout modules and silently ignored, leaving the
# rollout policy divergent in exactly the subspace the actor keeps training.
#
# Both halves of the mismatch have to be translated: the vLLM manager matches
# ``target_modules`` against the model's module names independently of the
# tensor keys, so renaming the keys alone would still wrap no layer.
# See https://github.com/verl-project/verl-omni/issues/658.
_BOOGU_LORA_NAME_RENAMES: tuple[tuple[str, str], ...] = (("to_out.0", "to_out"),)

#: LoRA targets the vllm-omni Boogu transformer can actually bind, mirroring
#: ``boogu_image_transformer.py``: the self-attention projections, the joint
#: attention's per-stream and merge projections, and the GEGLU halves of both
#: feed-forward stacks (``LuminaFeedForward`` exposes ``linear_1``/``linear_3``
#: as the gate/input halves and ``linear_2`` as the output).
BOOGU_LORA_TARGETS: frozenset[str] = frozenset(
    {
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "img_to_q",
        "img_to_k",
        "img_to_v",
        "img_out",
        "instruct_to_q",
        "instruct_to_k",
        "instruct_to_v",
        "instruct_out",
        "feed_forward.linear_1",
        "feed_forward.linear_2",
        "feed_forward.linear_3",
        "img_feed_forward.linear_1",
        "img_feed_forward.linear_2",
        "img_feed_forward.linear_3",
    }
)


def rename_boogu_lora_name(name: str) -> str:
    """Translate a diffusers Boogu LoRA tensor name or target to the vllm-omni layout."""
    for diffusers_name, vllm_name in _BOOGU_LORA_NAME_RENAMES:
        name = name.replace(diffusers_name, vllm_name)
    return name


def _boogu_lora_target_is_supported(target: str) -> bool:
    return any(target == known or target.endswith("." + known) for known in BOOGU_LORA_TARGETS)


def validate_boogu_lora_targets(target_modules) -> list[str]:
    """Return the translated, validated Boogu LoRA target list.

    Mirrors the MiniMax H3 whitelist so an unbindable target fails loudly here
    instead of vanishing during binding. Accepting one would reproduce the very
    bug this guards: a partial miss stays silent because vllm-omni only raises
    when *no* target binds.
    """
    if isinstance(target_modules, str):
        requested = [target_modules]
    elif isinstance(target_modules, list | tuple | set | frozenset):
        requested = [str(target) for target in target_modules]
    else:
        raise ValueError(f"Boogu-Image LoRA requires an explicit target_modules list; got {target_modules!r}.")

    translated = [rename_boogu_lora_name(target) for target in requested]
    if not translated:
        raise ValueError("Boogu-Image LoRA requires a non-empty target_modules list.")

    unsupported = sorted(target for target in translated if not _boogu_lora_target_is_supported(target))
    if unsupported:
        raise ValueError(
            "Boogu-Image LoRA supports only attention projections and feed-forward halves "
            f"{sorted(BOOGU_LORA_TARGETS)}; unsupported targets: {unsupported}. "
            "`all-linear` and other top-level modules are not synced to rollout "
            "(FSDP layered-summon does not transport them)."
        )
    return translated
