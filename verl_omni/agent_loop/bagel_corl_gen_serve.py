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

"""Live GEN serving helpers: FlowGRPO sampling params + traj stash from vLLM-Omni."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_AR_ONLY_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "min_p",
        "n",
    }
)


def config_to_sampling_dict(config: Any) -> dict[str, Any]:
    """Flatten OmegaConf / BaseConfig / dict / namespace into sampling kwargs."""
    if config is None:
        return {}
    items = None
    if isinstance(config, dict):
        items = config.items()
    elif hasattr(config, "items"):
        try:
            items = config.items()
        except (TypeError, AttributeError):
            items = None
    if items is not None:
        return {k: v for k, v in items if not str(k).startswith("_") and v is not None}
    if hasattr(config, "__dict__"):
        return {
            k: v
            for k, v in vars(config).items()
            if not str(k).startswith("_") and v is not None and not callable(v)
        }
    return {}


def build_gen_sampling_params(
    rollout_config: Any,
    *,
    base: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Merge diffusion ``pipeline`` + ``algo`` into GEN request sampling params.

    AgentLoopWorkerTQ builds AR-only params (temperature / top_p / logprobs). GEN
    must also carry ``num_inference_steps``, ``noise_level``, ``sde_window_*``, or
    Bagel FlowGRPO will not stash ``trajectory_latents`` / logprobs for training.
    """
    params: dict[str, Any] = {}
    if base:
        # Keep non-AR keys (e.g. global_steps) from the worker; drop text-decoding knobs.
        params.update({k: v for k, v in base.items() if k not in _AR_ONLY_SAMPLING_KEYS})

    pipeline = getattr(rollout_config, "pipeline", None)
    if pipeline is None and isinstance(rollout_config, dict):
        pipeline = rollout_config.get("pipeline")
    algo = getattr(rollout_config, "algo", None)
    if algo is None and isinstance(rollout_config, dict):
        algo = rollout_config.get("algo")

    params.update(config_to_sampling_dict(pipeline))
    params.update(config_to_sampling_dict(algo))

    calculate = getattr(rollout_config, "calculate_log_probs", None)
    if calculate is None and isinstance(rollout_config, dict):
        calculate = rollout_config.get("calculate_log_probs")
    if calculate is None:
        raise ValueError(
            "bagel_corl GEN requires actor_rollout_ref.rollout.calculate_log_probs=True "
            "(do not default it on; missing config hides ODE rollouts)."
        )
    params["logprobs"] = bool(calculate)
    if not params["logprobs"]:
        raise ValueError(
            "bagel_corl GEN requires actor_rollout_ref.rollout.calculate_log_probs=True "
            "(refuse soft-skip of GEN diffusion_loss)."
        )

    noise_level = float(params.get("noise_level", 0.0) or 0.0)
    if noise_level <= 0.0:
        raise ValueError(
            "bagel_corl GEN requires rollout.algo.noise_level > 0 so the SDE window "
            f"records latents/logprobs (got noise_level={noise_level})."
        )
    if params.get("num_inference_steps") is None:
        raise ValueError("bagel_corl GEN requires rollout.pipeline.num_inference_steps")
    if seed is not None:
        params["seed"] = int(seed)
    return params


def _maybe_unbatch_vector(value: Any) -> Any:
    """Unbatch (1, T) → (T,) for logprobs / timesteps; leave (T,) alone."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() == 2 and int(value.shape[0]) == 1:
            return value[0]
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _maybe_unbatch_latents(value: Any) -> Any:
    """Unbatch (1, T, ...) → (T, ...); never squeeze a lone timestep on dim0."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        # Strategy already peels B; only peel leftover batch when clearly B,T,* (≥3D with B=1).
        if value.dim() >= 3 and int(value.shape[0]) == 1:
            return value[0]
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def extract_gen_traj_from_diffusion_output(output: Any) -> tuple[Any, Any, Any]:
    """Pull ``all_latents`` / timesteps / log_probs from verl_omni ``DiffusionOutput``."""
    extra = dict(getattr(output, "extra_fields", None) or {})
    latents = _maybe_unbatch_latents(extra.get("all_latents"))
    timesteps = _maybe_unbatch_vector(extra.get("all_timesteps"))
    log_probs = _maybe_unbatch_vector(getattr(output, "log_probs", None))
    return latents, timesteps, log_probs


def save_diffusion_image(diffusion_output: Any, *, seed: int, root: Optional[str] = None) -> str:
    """Persist uint8 CHW/HWC image for RM / UND observation; return absolute path."""
    from PIL import Image

    image_root = Path(root or os.environ.get("BAGEL_CORL_GEN_IMAGE_DIR") or "/tmp/bagel_corl_gen")
    image_root.mkdir(parents=True, exist_ok=True)
    path = image_root / f"gen_{uuid.uuid4().hex}_s{int(seed)}.png"

    tensor = diffusion_output
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != torch.uint8:
            arr = arr.float().clamp(0, 1).mul(255).round().to(torch.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.permute(1, 2, 0)
        pil = Image.fromarray(arr.numpy())
    else:
        pil = Image.fromarray(tensor) if not hasattr(tensor, "save") else tensor
    if getattr(pil, "mode", None) not in (None, "RGB"):
        pil = pil.convert("RGB")
    pil.save(path)
    return str(path.resolve())


def stash_gen_row_from_diffusion_output(
    output: Any,
    *,
    seed: int,
    image_root: Optional[str] = None,
) -> dict[str, Any]:
    """Build one GEN seed row; fail closed when latents / timesteps / logprobs are missing."""
    if not hasattr(output, "diffusion_output"):
        raise RuntimeError(
            "Bagel Co-RL generate_image returned a non-diffusion output: dual-role GEN "
            "serving is not wired (bagel_single_stage is GEN-only / the AR replica is not "
            "diffusion-capable). Do not fall back to a Qwen sidecar."
        )

    stop_reason = getattr(output, "stop_reason", None)
    if stop_reason in ("aborted", "abort", "error"):
        raise RuntimeError(f"Bagel Co-RL GEN aborted (stop_reason={stop_reason!r}); refuse soft-skip.")

    latents, timesteps, log_probs = extract_gen_traj_from_diffusion_output(output)
    if latents is None or timesteps is None or log_probs is None:
        raise RuntimeError(
            "Bagel Co-RL GEN traj stash incomplete from vLLM-Omni "
            f"(all_latents={'ok' if latents is not None else 'missing'}, "
            f"timesteps={'ok' if timesteps is not None else 'missing'}, "
            f"log_probs={'ok' if log_probs is not None else 'missing'}). "
            "Require calculate_log_probs + algo.noise_level>0 + SDE window; "
            "refuse soft-fallback to stashed-only / skip GEN loss."
        )

    extra = dict(getattr(output, "extra_fields", None) or {})
    image_path = extra.get("image_path") or extra.get("path")
    if not image_path:
        image_path = save_diffusion_image(output.diffusion_output, seed=seed, root=image_root)

    return {
        "valid": True,
        "all_latents": latents,
        "timesteps": timesteps,
        "rollout_log_probs": log_probs,
        "image_path": image_path,
    }
