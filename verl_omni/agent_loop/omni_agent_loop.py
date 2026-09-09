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

"""Agent-loop worker wiring, rollout monitoring, and invalid-rollout masking."""

from __future__ import annotations

import logging
from pathlib import Path

import ray
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils import hf_tokenizer

from verl_omni.tools.trajectory import (
    bind_run_artifact_env,
    build_trajectory_relpath,
    clear_good_enough_yes_reached,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)
from verl_omni.utils.agentic.image_gen_rollout_dump import discard_invalid_rollouts, dump_raw_rollouts
from verl_omni.utils.agentic.image_gen_rollout_parse import (
    last_user_prompt,
    split_assistant_rollouts,
    split_rollout_turns,
)
from verl_omni.utils.metrics_utils import AgenticRewardMetrics

# Register ``image_gen_tool_agent`` when this module is loaded.
from . import tool_agent_loop as image_gen_tool_agent_loop  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "OmniAgentLoopWorker",
    "OmniAgentLoopManager",
    "split_assistant_rollouts",
    "split_rollout_turns",
]


class OmniAgentLoopWorker(AgentLoopWorker):
    """Worker-side hooks: trajectory bind + step kwargs for force-first curriculum.

    ``AgentLoopManager.generate_sequences`` dispatches to Ray ``AgentLoopWorker``s.
    Overrides on the Manager class never run per-rollout — they must live here.

    Also hard-binds agentic multi-turn defaults (Hermes + ``verl_omni/tools``)
    when ``default_agent_loop == image_gen_tool_agent`` (only fills unset keys).
    """

    _AGENTIC_TOOL_FORMAT = "hermes"
    _AGENTIC_FUNCTION_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "image_gen.py"

    def __init__(self, config, *args, **kwargs):
        from omegaconf import open_dict

        # Bind by path string only — importing image_gen.py would double-register tools.
        bind_run_artifact_env(config)
        default_loop = None
        try:
            default_loop = config.actor_rollout_ref.rollout.agent.get("default_agent_loop")
        except Exception:  # noqa: BLE001
            default_loop = None
        if default_loop == "image_gen_tool_agent":
            tool_path = self._AGENTIC_FUNCTION_TOOLS
            if not tool_path.is_file():
                raise FileNotFoundError(
                    f"agentic function tools not found at {tool_path}. Expected verl_omni/tools/image_gen.py"
                )
            with open_dict(config.actor_rollout_ref.rollout.multi_turn):
                mt = config.actor_rollout_ref.rollout.multi_turn
                # Only fill unset keys so explicit Hydra overrides still win.
                if not mt.get("function_tool_path"):
                    mt.function_tool_path = str(tool_path)
                if not mt.get("format"):
                    mt.format = self._AGENTIC_TOOL_FORMAT
        super().__init__(config, *args, **kwargs)

    async def _run_agent_loop(
        self,
        sampling_params,
        trajectory,
        *,
        agent_name,
        trace=True,
        **kwargs,
    ):
        relpath = build_trajectory_relpath(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
        )
        raw_prompt = kwargs.get("raw_prompt")
        user_prompt = last_user_prompt(raw_prompt) if raw_prompt is not None else ""
        path_token = set_active_trajectory_relpath(relpath)
        prompt_token = set_active_user_prompt(user_prompt)
        clear_good_enough_yes_reached()
        kwargs["_agentic_step"] = trajectory["step"]
        kwargs["_agentic_validate"] = trajectory["validate"]
        kwargs["_agentic_trajectory_relpath"] = relpath
        try:
            return await super()._run_agent_loop(
                sampling_params,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                **kwargs,
            )
        finally:
            reset_active_user_prompt(prompt_token)
            reset_active_trajectory_relpath(path_token)


class OmniAgentLoopManager(AgentLoopManager):
    """Use stock rollout management, dump outputs, and mask invalid rollouts."""

    def __init__(self, *args, **kwargs):
        # Must set before AgentLoopManager.__init__ creates Ray workers.
        self.agent_loop_workers_class = ray.remote(OmniAgentLoopWorker)
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is not None:
            bind_run_artifact_env(config)
        super().__init__(*args, **kwargs)
        model_path = self.model_config.get("tokenizer_path") or self.model_config.get("path")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self._monitor_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    def generate_sequences(self, prompts):
        step = prompts.meta_info.get("global_steps")
        output = super().generate_sequences(prompts)
        # Dump before discard: discard zeros response_mask and hides tool-less prose.
        dump_raw_rollouts(tokenizer=self._monitor_tokenizer, output=output, step=step)
        discard_invalid_rollouts(output)
        metrics = AgenticRewardMetrics.aggregate(output.non_tensor_batch)
        if metrics:
            # Stash for trainers / Tracking; avoid bare wandb.log (drops tensorboard).
            meta = getattr(output, "meta_info", None)
            if not isinstance(meta, dict):
                output.meta_info = {}
                meta = output.meta_info
            existing = meta.get("agentic_metrics")
            if isinstance(existing, dict):
                existing.update(metrics)
            else:
                meta["agentic_metrics"] = dict(metrics)
            # Fold into timing so stock PPO's timing_raw.update carries them into
            # compute_timing_metrics → logger.log for every configured backend.
            # Keys keep the agentic_reward/ prefix (logged as timing_s/... only if
            # left in timing; prefer a parallel meta key + explicit emit below).
            self._emit_agentic_metrics(metrics, step=step)
        return output

    def _emit_agentic_metrics(self, metrics: dict[str, float], *, step) -> None:
        """Emit rollout metrics without assuming W&B is the only backend."""
        backends: list[str] = []
        try:
            raw = self.config.trainer.get("logger", ["console"])
            if isinstance(raw, str):
                backends = [raw]
            else:
                backends = [str(b) for b in list(raw)]
        except Exception:  # noqa: BLE001
            backends = ["console"]

        step_i = int(step) if step is not None else None
        if "wandb" in backends or "tracking" in backends:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=step_i, commit=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log agentic metrics to W&B: %s", exc)
        # Console / file backends: always leave a structured breadcrumb.
        logger.info("agentic_metrics step=%s %s", step_i, metrics)
