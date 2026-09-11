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
"""GEN FlowGRPO advantage for Bagel Co-RL (RFC flatten → GenAdv → composite update).

UND token GRPO stays on verl V1 ``_compute_advantage``. This module is the GEN
slice: same ``compute_advantage`` helper the diffusion trainers use, grouped by
``gen_group_uid`` (K samples from one ``generate_image`` call).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from verl_omni.trainer.diffusion.ray_diffusion_trainer import compute_advantage

__all__ = ["apply_gen_flowgrpo_advantage", "build_gen_flowgrpo_proto", "select_gen_advantages_for_step"]


def select_gen_advantages_for_step(micro_batch, step: int, *, require_bagel_corl_gen: bool = False):
    """GEN timestep advantages: ``bagel_corl_gen['advantages'][:, step]`` only.

    When ``require_bagel_corl_gen`` is True (Bagel Co-RL GEN engine path), refuse to
    fall back to UND / poisoned ``micro_batch['advantages']``.
    """
    from verl.utils import tensordict_utils as tu

    gen_view = tu.get_non_tensor_data(micro_batch, "bagel_corl_gen", default=None)
    if gen_view is not None and hasattr(gen_view, "keys") and "advantages" in gen_view.keys():
        return gen_view["advantages"][:, step]
    if require_bagel_corl_gen:
        raise ValueError(
            "bagel_corl GEN step requires bagel_corl_gen['advantages']; refusing UND token advantages"
        )
    return micro_batch["advantages"][:, step]


def _logprob_1d(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        return None
    return tensor


def _as_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _stack_padded(tensors: list[torch.Tensor], *, pad_value: float = 0.0) -> torch.Tensor:
    """Stack tensors that may differ on dim0 (timesteps); pad to max length."""
    if not tensors:
        raise ValueError("empty tensor list")
    max_t = max(int(t.shape[0]) for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.shape[0] == max_t:
            padded.append(tensor)
            continue
        pad_shape = (max_t - tensor.shape[0],) + tuple(tensor.shape[1:])
        pad = tensor.new_full(pad_shape, pad_value)
        padded.append(torch.cat([tensor, pad], dim=0))
    return torch.stack(padded, dim=0)


def build_gen_flowgrpo_proto(gen_batch: list[Mapping[str, Any]]) -> DataProto | None:
    """Stack flattened GEN rows into a diffusion ``DataProto`` for FlowGRPO + actor.

    Requires live traj stash: ``all_latents`` + ``timesteps`` + ``rollout_log_probs``
    (plus ``gen_group_uid`` / ``rm_score``). Refuse logprobs-only soft path that would
    skip GEN ``diffusion_loss``.

    Empty ``gen_batch`` (pattern 3 / ``K=0``) returns ``None`` so UND still trains.
    """
    usable: list[tuple[Mapping[str, Any], torch.Tensor, float, str, torch.Tensor, torch.Tensor]] = []
    missing_traj = 0
    for row in gen_batch:
        logprob = _logprob_1d(row.get("rollout_log_probs"))
        score = row.get("rm_score")
        uid = row.get("gen_group_uid")
        latents = _as_tensor(row.get("all_latents"))
        timesteps = _as_tensor(row.get("all_timesteps") if row.get("all_timesteps") is not None else row.get("timesteps"))
        if latents is None or timesteps is None:
            if uid is not None or logprob is not None:
                missing_traj += 1
            continue
        if logprob is None or score is None or uid is None:
            continue
        usable.append((row, logprob, float(score), str(uid), latents, timesteps))
    if not usable:
        if missing_traj:
            raise RuntimeError(
                "bagel_corl: GEN batch lacks all_latents/timesteps for FlowGRPO "
                f"({missing_traj} row(s)); refuse soft-fallback to rollout_log_probs / skip GEN loss. "
                "Fix live GEN traj stash (calculate_log_probs + algo.noise_level>0)."
            )
        return None

    max_steps = max(item[1].numel() for item in usable)
    old_log_probs = torch.zeros(len(usable), max_steps, dtype=torch.float32)
    for index, (_, logprob, _, _, _, _) in enumerate(usable):
        old_log_probs[index, : logprob.numel()] = logprob
    scores = torch.tensor([[item[2]] for item in usable], dtype=torch.float32)
    uids = np.array([item[3] for item in usable], dtype=object)
    batch_tensors: dict[str, torch.Tensor] = {
        "old_log_probs": old_log_probs,
        "rm_scores": scores,
        "sample_level_scores": scores,
    }

    latent_list: list[torch.Tensor] = []
    timestep_list: list[torch.Tensor] = []
    prompt_token_ids: list[Any] = []
    for row, _, _, _, latents, timesteps in usable:
        # Engine indexes latents[:, step]; accept (T, ...) or (T+1, ...).
        latent_list.append(latents.float())
        timestep_list.append(timesteps.reshape(-1).float())
        prompt_token_ids.append(row.get("prompt_token_ids"))

    batch_tensors["all_latents"] = _stack_padded(latent_list)
    batch_tensors["all_timesteps"] = _stack_padded(timestep_list)
    num_t = int(batch_tensors["all_timesteps"].shape[1])
    if old_log_probs.shape[1] != num_t:
        resized = torch.zeros(len(usable), num_t, dtype=torch.float32)
        copy_t = min(old_log_probs.shape[1], num_t)
        resized[:, :copy_t] = old_log_probs[:, :copy_t]
        batch_tensors["old_log_probs"] = resized

    batch = TensorDict(batch_tensors, batch_size=[len(usable)])
    non_tensor: dict[str, Any] = {
        "uid": uids,
        "gen_group_uid": uids,
        "prompt_token_ids": np.array(prompt_token_ids, dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor)


def apply_gen_flowgrpo_advantage(
    gen_batch: list[Mapping[str, Any]] | None,
    *,
    adv_estimator: str = "flow_grpo",
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    algo_config: Any = None,
) -> tuple[DataProto | None, dict[str, float]]:
    """RFC GenAdv: FlowGRPO within each K-sample ``gen_group_uid``.

    Returns ``(None, skip metrics)`` when there is no GEN view (``K=0``). Rows that
    claim GEN without latents raise — refuse soft-skip of ``diffusion_loss``.
    """
    metrics: dict[str, float] = {
        "gen/skipped_no_groups": 1.0,
        "gen/num_rows": 0.0,
        "has_complete_gen_groups": 0.0,
        "gen/has_traj": 0.0,
    }
    if not gen_batch:
        return None, metrics

    proto = build_gen_flowgrpo_proto(list(gen_batch))
    metrics["gen/num_rows"] = float(len(gen_batch))
    if proto is None:
        return None, metrics

    num_timesteps = proto.batch["old_log_probs"].shape[1]
    proto.batch["sample_level_rewards"] = proto.batch["sample_level_scores"].expand(-1, num_timesteps)
    proto = compute_advantage(
        proto,
        adv_estimator=adv_estimator,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        global_std=global_std,
        config=algo_config,
    )
    # Fold non-tensor traj metadata onto the batch so update_actor can materialize GEN.
    ntb = getattr(proto, "non_tensor_batch", None) or {}
    if "prompt_token_ids" in ntb:
        from verl.utils import tensordict_utils as tu

        tu.assign_non_tensor(proto.batch, prompt_token_ids=ntb["prompt_token_ids"])
    metrics["gen/skipped_no_groups"] = 0.0
    metrics["gen/num_usable_rows"] = float(len(proto))
    metrics["has_complete_gen_groups"] = 1.0
    metrics["gen/has_traj"] = 1.0
    return proto, metrics
