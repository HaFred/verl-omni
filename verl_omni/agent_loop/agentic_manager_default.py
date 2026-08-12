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

"""Hard-bind AgenticMetricsAgentLoopManager when using agentic_tool_agent.

Launch recipes only need ``default_agent_loop=agentic_tool_agent``; they do not
need a Hydra ``agent_loop_manager_class`` override.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

AGENTIC_MANAGER_FQN = "verl_omni.agent_loop.agentic_metrics_manager.AgenticMetricsAgentLoopManager"
_AGENTIC_LOOP_NAME = "agentic_tool_agent"
_INSTALLED = False


def _maybe_bind_manager(config) -> None:
    from omegaconf import open_dict

    agent = config.actor_rollout_ref.rollout.get("agent", {})
    if agent.get("default_agent_loop") != _AGENTIC_LOOP_NAME:
        return
    if agent.get("agent_loop_manager_class"):
        return
    with open_dict(config.actor_rollout_ref.rollout.agent):
        config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = AGENTIC_MANAGER_FQN
    logger.info("Bound agent_loop_manager_class=%s for %s", AGENTIC_MANAGER_FQN, _AGENTIC_LOOP_NAME)


def install_agentic_manager_default() -> None:
    """Patch PPO worker init so agentic_tool_agent gets the metrics manager."""
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        _orig_init_workers = RayPPOTrainer.init_workers

        def init_workers(self, *args, **kwargs):
            _maybe_bind_manager(self.config)
            return _orig_init_workers(self, *args, **kwargs)

        RayPPOTrainer.init_workers = init_workers  # type: ignore[method-assign]
    except ImportError:
        pass

    try:
        from verl.trainer.main_ppo import TaskRunnerV1

        if hasattr(TaskRunnerV1, "init_agent_loop_manager"):
            _orig_init_mgr = TaskRunnerV1.init_agent_loop_manager

            def init_agent_loop_manager(self, *args, **kwargs):
                if self.config is not None:
                    _maybe_bind_manager(self.config)
                return _orig_init_mgr(self, *args, **kwargs)

            TaskRunnerV1.init_agent_loop_manager = init_agent_loop_manager  # type: ignore[method-assign]
    except ImportError:
        pass

    _INSTALLED = True
