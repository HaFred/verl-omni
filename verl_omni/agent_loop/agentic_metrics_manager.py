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

"""Stock AgentLoopManager with W&B logging for agentic reward components."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from verl.experimental.agent_loop import AgentLoopManager

logger = logging.getLogger(__name__)

REWARD_COMPONENTS = (
    "reward_format",
    "reward_reflection",
    "reward_tool_usage",
    "reward_result",
)


def aggregate_agentic_reward_metrics(non_tensor_batch: dict[str, Any]) -> dict[str, float]:
    """Aggregate numeric reward extras already returned by verl's reward manager."""
    metrics: dict[str, float] = {}
    for key in REWARD_COMPONENTS:
        if key not in non_tensor_batch:
            continue
        values = np.asarray(non_tensor_batch[key], dtype=np.float64)
        if values.size == 0:
            continue
        prefix = f"agentic_reward/{key.removeprefix('reward_')}"
        metrics[f"{prefix}/mean"] = float(np.mean(values))
        metrics[f"{prefix}/min"] = float(np.min(values))
        metrics[f"{prefix}/max"] = float(np.max(values))
    return metrics


class AgenticMetricsAgentLoopManager(AgentLoopManager):
    """Use stock rollout management and publish reward-component time series."""

    def generate_sequences(self, prompts):
        step = prompts.meta_info.get("global_steps")
        output = super().generate_sequences(prompts)
        metrics = aggregate_agentic_reward_metrics(output.non_tensor_batch)
        if metrics:
            try:
                import wandb

                if wandb.run is not None:
                    # The trainer's Tracking.log call commits this same global step.
                    wandb.log(metrics, step=int(step) if step is not None else None, commit=False)
            except Exception as exc:  # noqa: BLE001
                # Logging must never fail or alter rollout generation.
                logger.warning("Failed to log agentic reward metrics to W&B: %s", exc)
        return output
