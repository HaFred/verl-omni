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


def build_gen_flowgrpo_proto(gen_batch: list[Mapping[str, Any]]) -> DataProto | None:
    """Stack flattened GEN rows into a diffusion ``DataProto`` for FlowGRPO.

    Drops rows missing ``gen_group_uid``, ``rm_score``, or ``rollout_log_probs``
    (no trajectory → cannot expand rewards over timesteps).
    """
    usable: list[tuple[Mapping[str, Any], torch.Tensor, float, str]] = []
    for row in gen_batch:
        logprob = _logprob_1d(row.get("rollout_log_probs"))
        score = row.get("rm_score")
        uid = row.get("gen_group_uid")
        if logprob is None or score is None or uid is None:
            continue
        usable.append((row, logprob, float(score), str(uid)))
    if not usable:
        return None

    max_steps = max(item[1].numel() for item in usable)
    old_log_probs = torch.zeros(len(usable), max_steps, dtype=torch.float32)
    for index, (_, logprob, _, _) in enumerate(usable):
        old_log_probs[index, : logprob.numel()] = logprob
    scores = torch.tensor([[item[2]] for item in usable], dtype=torch.float32)
    uids = np.array([item[3] for item in usable], dtype=object)
    batch = TensorDict(
        {
            "old_log_probs": old_log_probs,
            "rm_scores": scores,
            "sample_level_scores": scores,
        },
        batch_size=[len(usable)],
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={"uid": uids, "gen_group_uid": uids},
    )


def apply_gen_flowgrpo_advantage(
    gen_batch: list[Mapping[str, Any]] | None,
    *,
    adv_estimator: str = "flow_grpo",
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    algo_config: Any = None,
) -> tuple[DataProto | None, dict[str, float]]:
    """RFC GenAdv: FlowGRPO within each K-sample ``gen_group_uid``.

    Returns ``(None, skip metrics)`` when there is no complete GEN view with
    stashed traj log-probs — UND token GRPO still runs.
    """
    metrics: dict[str, float] = {
        "gen/skipped_no_groups": 1.0,
        "gen/num_rows": 0.0,
        "has_complete_gen_groups": 0.0,
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
    metrics["gen/skipped_no_groups"] = 0.0
    metrics["gen/num_usable_rows"] = float(len(proto))
    metrics["has_complete_gen_groups"] = 1.0
    return proto, metrics
