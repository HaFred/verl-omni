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
"""Bagel Co-RL composite actor: UND token path + GEN diffusion path on one FSDP module.

RFC: one ``update_actor`` / one ``optimizer.step``. Outer owner is
``verl.trainer.main_ppo.TaskRunnerV1`` via ``OmniBagelCoRLTrainerSync``. GEN reuses
diffusion V1 engine math (``PPODiffusersFSDPEngine`` timestep loop + ``diffusion_loss``)
and bound ``PolicyGradientDiffusionTrainerV1`` hooks, not ``fit()``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.workers.engine.utils import prepare_micro_batches

logger = logging.getLogger(__name__)

_UND_SELECT = (
    "input_ids",
    "attention_mask",
    "response_mask",
    "old_log_probs",
    "advantages",
    "responses",
    "prompts",
    "position_ids",
    "ref_log_prob",
    "rollout_is_weights",
)


def is_bagel_corl_composite(model_config) -> bool:
    if model_config is None:
        return False
    return model_config.get("composite_mode") == "bagel_corl"


def unwrap_bagel_module(module: torch.nn.Module) -> torch.nn.Module:
    """Peel FSDP / PEFT wrappers until ``compute_und_log_prob`` is reachable."""
    current = module
    for _ in range(8):
        if hasattr(current, "compute_und_log_prob"):
            return current
        nxt = getattr(current, "_fsdp_wrapped_module", None)
        if nxt is None:
            nxt = getattr(current, "module", None)
        if nxt is None:
            nxt = getattr(current, "base_model", None)
            if nxt is not None and hasattr(nxt, "model"):
                nxt = nxt.model
        if nxt is None or nxt is current:
            break
        current = nxt
    raise AttributeError(
        "Bagel Co-RL composite UND path requires BagelForCoRL.compute_und_log_prob; "
        f"got {type(module).__name__}"
    )


def composite_forward_mode(data: TensorDict, *, forward_only: bool) -> str:
    """Select UND infer / GEN-only diffusion / composite train / empty.

    GEN-only (``all_latents``, no ``input_ids``) is the diffusion V1
    ``infer_actor_batch`` old-logprob path and must use the vanilla timestep loop.
    """
    has_und = "input_ids" in data.keys()
    has_latents = "all_latents" in data.keys() and (
        "all_timesteps" in data.keys() or "timesteps" in data.keys()
    )
    gen_view = tu.get_non_tensor_data(data, "bagel_corl_gen", default=None)
    has_complete = bool(tu.get_non_tensor_data(data, "has_complete_gen_groups", default=False))
    skip_gen = bool(tu.get_non_tensor_data(data, "skip_gen", default=not has_complete))
    run_gen_view = (not skip_gen) and has_complete and gen_view_has_traj(gen_view)

    if has_latents and not has_und:
        return "gen_only_diffusion"
    if forward_only and has_und and not run_gen_view:
        return "und_infer"
    if has_und or run_gen_view:
        return "composite_train"
    return "empty"


def gen_view_has_traj(gen_view) -> bool:
    if gen_view is None:
        return False
    keys = getattr(gen_view, "keys", None)
    if keys is None:
        return False
    keyset = set(keys()) if callable(keys) else set(keys)
    return "all_latents" in keyset and ("all_timesteps" in keyset or "timesteps" in keyset)


def materialize_gen_train_batch(gen_view: TensorDict, flags: dict[str, Any]) -> TensorDict:
    """Build a diffusion train TensorDict from ``bagel_corl_gen`` (FlowGRPO + traj)."""
    if isinstance(gen_view, TensorDict):
        gen_data = gen_view.clone()
    else:
        gen_data = TensorDict(dict(gen_view), batch_size=getattr(gen_view, "batch_size", []))

    if "all_timesteps" not in gen_data.keys() and "timesteps" in gen_data.keys():
        gen_data["all_timesteps"] = gen_data["timesteps"]

    for key, val in flags.items():
        if val is not None:
            tu.assign_non_tensor(gen_data, **{key: val})
    # GEN phase: loss must use FlowGRPO view, not UND token advantages on this TensorDict.
    tu.assign_non_tensor(gen_data, bagel_corl_gen=gen_view if isinstance(gen_view, TensorDict) else gen_data)
    tu.assign_non_tensor(gen_data, skip_gen=False)
    tu.assign_non_tensor(gen_data, has_complete_gen_groups=True)
    return gen_data


def _pad_left_response_mask(response_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Pad response-only mask on the left to full sequence length (prompt | response)."""
    if response_mask.shape[-1] == seq_len:
        return response_mask
    if response_mask.shape[-1] > seq_len:
        raise ValueError(f"response_mask length {response_mask.shape[-1]} > seq_len {seq_len}")
    pad = seq_len - response_mask.shape[-1]
    return F.pad(response_mask, (pad, 0))


def response_aligned_und_log_probs(
    token_logp: torch.Tensor,
    *,
    response_len: int,
) -> torch.Tensor:
    """Slice ``compute_und_log_prob`` ``(B, L-1)`` down to response-length ``(B, R)``."""
    if response_len <= 0:
        raise ValueError("response_len must be positive")
    if token_logp.shape[-1] < response_len:
        raise ValueError(
            f"UND log_probs width {token_logp.shape[-1]} < response_len {response_len}"
        )
    return token_logp[:, -response_len:]


def run_und_token_forward_backward(
    engine,
    data: TensorDict,
    loss_function: Callable,
    forward_only: bool,
) -> dict:
    """AR-style UND pass on Bagel MoT text path; one backward per micro-batch when training."""
    from contextlib import nullcontext

    und = data.select(*[k for k in _UND_SELECT if k in data.keys()], strict=False)
    if "input_ids" not in und.keys():
        raise KeyError("Bagel Co-RL UND phase requires input_ids on the actor batch")
    if not forward_only and ("old_log_probs" not in und.keys() or "advantages" not in und.keys()):
        raise KeyError("Bagel Co-RL UND train phase requires old_log_probs and advantages (token GRPO)")

    tu.assign_non_tensor(und, sp_size=engine.ulysses_sequence_parallel_size)
    tu.assign_non_tensor(und, use_dynamic_bsz=False)

    micro_batches, indices = prepare_micro_batches(
        data=und, dp_group=engine.get_data_parallel_group(), same_micro_num_in_dp=True
    )
    gradient_accumulation_steps = max(len(micro_batches), 1)
    output_lst = []
    ctx = torch.no_grad() if forward_only else nullcontext()
    module = unwrap_bagel_module(engine.module)

    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())
        tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
        tu.assign_non_tensor(micro_batch, skip_gen=True)
        tu.assign_non_tensor(micro_batch, has_complete_gen_groups=False)
        tu.assign_non_tensor(micro_batch, num_gen_rows=0)

        padded = micro_batch
        pad_keys = [k for k in ("input_ids", "attention_mask", "response_mask", "old_log_probs", "advantages") if k in micro_batch.keys()]
        if hasattr(micro_batch, "to_padded_tensor"):
            padded = micro_batch.select(*pad_keys).to_padded_tensor()

        input_ids = padded["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = padded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        response_mask = padded["response_mask"]
        if response_mask.ndim == 1:
            response_mask = response_mask.unsqueeze(0)
        full_response_mask = _pad_left_response_mask(response_mask, input_ids.shape[1])
        if "old_log_probs" in padded.keys():
            resp_len = int(padded["old_log_probs"].shape[-1])
        else:
            resp_len = int(response_mask.shape[-1])

        with ctx:
            token_logp = module.compute_und_log_prob(input_ids, attention_mask, full_response_mask)
            log_probs = response_aligned_und_log_probs(token_logp, response_len=resp_len)
            model_output = {"und": {"log_probs": log_probs}, "modality": "und", "log_probs": log_probs}

            # Infer (old log-prob): no token-GRPO fields yet — return log_probs only.
            infer_only = forward_only and (
                "old_log_probs" not in padded.keys() or "advantages" not in padded.keys()
            )
            if infer_only or loss_function is None:
                loss = torch.tensor(1.0, device=get_device_id())
                metrics = {}
            else:
                resp_mask_loss = (
                    response_mask if response_mask.shape[-1] == resp_len else response_mask[:, -resp_len:]
                )
                loss_data = TensorDict(
                    {
                        "response_mask": resp_mask_loss,
                        "old_log_probs": padded["old_log_probs"],
                        "advantages": padded["advantages"],
                    },
                    batch_size=padded["old_log_probs"].shape[:1],
                )
                for opt in ("ref_log_prob", "rollout_is_weights"):
                    if opt in padded.keys():
                        loss_data[opt] = padded[opt]
                for key in ("dp_size", "batch_num_tokens", "global_batch_size"):
                    if key in micro_batch.keys():
                        loss_data[key] = micro_batch[key]
                    elif key in data.keys():
                        loss_data[key] = data[key]
                if "dp_size" not in loss_data.keys():
                    loss_data["dp_size"] = engine.get_data_parallel_size()
                if "batch_num_tokens" not in loss_data.keys():
                    loss_data["batch_num_tokens"] = None
                if "global_batch_size" not in loss_data.keys():
                    loss_data["global_batch_size"] = None

                tu.assign_non_tensor(loss_data, bagel_corl_und=loss_data)
                tu.assign_non_tensor(loss_data, skip_gen=True)
                tu.assign_non_tensor(loss_data, has_complete_gen_groups=False)
                tu.assign_non_tensor(loss_data, num_gen_rows=0)
                tu.assign_non_tensor(
                    loss_data,
                    gradient_accumulation_steps=tu.get_non_tensor_data(
                        micro_batch, "gradient_accumulation_steps", default=gradient_accumulation_steps
                    ),
                    sp_size=tu.get_non_tensor_data(micro_batch, "sp_size", default=1),
                )
                loss, metrics = loss_function(
                    model_output=model_output,
                    data=loss_data,
                    dp_group=engine.get_data_parallel_group(),
                )

            if not forward_only:
                loss.backward()

        meta_info_lst = {
            "model_output": [model_output],
            "loss": [loss.detach().item()],
            "metrics": [metrics],
        }
        output_lst.append(meta_info_lst)

    return engine.postprocess_batch_func(output_lst=output_lst, indices=indices, data=und)


def merge_composite_outputs(parts: list[dict]) -> dict:
    """Merge UND + GEN ``postprocess_batch_func`` dicts for one train_batch metrics blob."""
    if not parts:
        return {"model_output": {}, "loss": [], "metrics": {}}
    if len(parts) == 1:
        return parts[0]

    merged_loss: list = []
    merged_metrics: dict = {}
    for part in parts:
        merged_loss.extend(part.get("loss") or [])
        for key, val in (part.get("metrics") or {}).items():
            if key in merged_metrics and isinstance(merged_metrics[key], list) and isinstance(val, list):
                merged_metrics[key].extend(val)
            elif key in merged_metrics and isinstance(merged_metrics[key], list):
                merged_metrics[key].append(val)
            else:
                merged_metrics[key] = val if isinstance(val, list) else [val]
    return {
        "model_output": parts[-1].get("model_output") or {},
        "loss": merged_loss,
        "metrics": merged_metrics,
    }
