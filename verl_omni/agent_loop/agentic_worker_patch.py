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

"""Ray-safe AgentLoopWorker patches for agentic GRPO artifacts (PR1 ST-1).

``diffusion_tool.py`` is loaded via ``function_tool_path`` under a synthetic
module name. Closures defined there cannot be cloudpickled onto Ray actors.
Keep patched callables in this real package module instead.

Force-tool / teacher scaffolding lives on the PR2 working branch with the
multi-step Lance e2e example; PR1 only needs traj kwargs + reward metrics.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__file__)


async def _run_agent_loop_with_traj_kwargs(
    self,
    sampling_params: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    agent_name: str,
    trace: bool = True,
    **kwargs,
):
    """Bound onto ``AgentLoopWorker``; must live in this importable module for Ray."""
    orig = getattr(type(self), "_agentic_orig_run_agent_loop", None)
    if orig is None:
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

        orig = AgentLoopWorker._agentic_orig_run_agent_loop
    kwargs = {
        **kwargs,
        "_agentic_step": trajectory.get("step", -1),
        "_agentic_sample_index": trajectory.get("sample_index", 0),
        "_agentic_rollout_n": trajectory.get("rollout_n", 0),
        "_agentic_validate": bool(trajectory.get("validate", False)),
    }
    return await orig(self, sampling_params, trajectory, agent_name=agent_name, trace=trace, **kwargs)


def install_agentic_worker_patches() -> None:
    """Idempotently patch ``AgentLoopWorker`` for step/g*/n* layout + reward metrics."""
    install_agentic_reward_metrics_patch()
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    except Exception:  # noqa: BLE001
        return
    if getattr(AgentLoopWorker, "_agentic_traj_kwargs_patch", False):
        return
    AgentLoopWorker._agentic_orig_run_agent_loop = AgentLoopWorker._run_agent_loop
    AgentLoopWorker._run_agent_loop = _run_agent_loop_with_traj_kwargs
    AgentLoopWorker._agentic_traj_kwargs_patch = True
    logger.info("Installed AgentLoopWorker kwargs patch for step/g*/n* layout")


def install_agentic_reward_metrics_patch() -> None:
    """Fold ``r_format``/``r_tool``/``r_result`` into step metrics (wandb/console)."""
    try:
        import verl.trainer.ppo.metric_utils as metric_utils
        import verl.trainer.ppo.ray_trainer as ray_trainer
    except Exception:  # noqa: BLE001
        return
    if getattr(metric_utils, "_agentic_reward_metrics_patched", False):
        return

    from verl_omni.utils.reward_score.agentic_reward import compute_agentic_reward_metrics

    orig = metric_utils.compute_data_metrics

    def _compute_data_metrics_with_agentic(batch, use_critic: bool = True):
        metrics = orig(batch, use_critic=use_critic)
        metrics.update(compute_agentic_reward_metrics(batch))
        return metrics

    metric_utils.compute_data_metrics = _compute_data_metrics_with_agentic
    if getattr(ray_trainer, "compute_data_metrics", None) is orig:
        ray_trainer.compute_data_metrics = _compute_data_metrics_with_agentic
    metric_utils._agentic_reward_metrics_patched = True
    logger.info("Installed agentic reward metrics patch (r_format/r_tool/r_result → wandb)")
