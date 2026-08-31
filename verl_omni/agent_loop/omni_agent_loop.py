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

import json
import logging
from pathlib import Path
from typing import Any

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
    resolve_rollout_images_root,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)
from verl_omni.agent_loop.rl_insight_profiler import init_rl_insight
from verl_omni.utils.agentic_val_viz import resolve_agentic_val_viz_provider
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


def _init_rl_insight_from_config(config) -> None:
    experiment_name = None
    try:
        trainer = config.get("trainer", {}) if config is not None else {}
        experiment_name = trainer.get("experiment_name")
    except Exception:  # noqa: BLE001
        pass
    init_rl_insight(project="verl_omni_agentic", experiment_name=experiment_name)


class OmniAgentLoopWorker(AgentLoopWorker):
    """Worker-side hooks: trajectory bind + step kwargs for force-first curriculum.

    ``AgentLoopManager.generate_sequences`` dispatches to Ray ``AgentLoopWorker``s.
    Overrides on the Manager class never run per-rollout — they must live here.

    Also hard-binds agentic multi-turn defaults (Hermes + ``verl_omni/tools``)
    so launch recipes need not pass ``function_tool_path`` / ``format`` Hydra overrides.
    """

    _AGENTIC_TOOL_FORMAT = "hermes"
    _AGENTIC_FUNCTION_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "image_gen.py"

    def __init__(self, config, *args, **kwargs):
        from omegaconf import open_dict

        # Bind by path string only — importing image_gen.py would double-register tools.
        bind_run_artifact_env(config)
        _init_rl_insight_from_config(config)
        tool_path = self._AGENTIC_FUNCTION_TOOLS
        if not tool_path.is_file():
            raise FileNotFoundError(
                f"agentic function tools not found at {tool_path}. Expected verl_omni/tools/image_gen.py"
            )
        with open_dict(config.actor_rollout_ref.rollout.multi_turn):
            config.actor_rollout_ref.rollout.multi_turn.function_tool_path = str(tool_path)
            config.actor_rollout_ref.rollout.multi_turn.format = self._AGENTIC_TOOL_FORMAT
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
            validate=bool(trajectory.get("validate")),
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
            _init_rl_insight_from_config(config)
        super().__init__(*args, **kwargs)
        model_path = self.model_config.get("tokenizer_path") or self.model_config.get("path")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self._monitor_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        self._val_viz_provider = resolve_agentic_val_viz_provider()
        self._val_viz_logged_steps: set[int] = set()

    def generate_sequences(self, prompts):
        step = prompts.meta_info.get("global_steps")
        is_val = bool(prompts.meta_info.get("validate", False))
        if is_val:
            self._maybe_run_val_viz(step)
        output = super().generate_sequences(prompts)
        if is_val:
            self._log_reward_metrics(output, step, prefix="val_")
            self._log_tool_latency_metrics(output, step, prefix="val_")
            return output
        # Dump before discard: masking hides tool-less prose.
        dump_raw_rollouts(tokenizer=self._monitor_tokenizer, output=output, step=step)
        self._log_tool_latency_metrics(output, step)
        discard_invalid_rollouts(output)
        self._log_reward_metrics(output, step)
        return output

    @staticmethod
    def _log_reward_metrics(output: Any, step: Any, *, prefix: str = "") -> None:
        metrics = AgenticRewardMetrics.aggregate(output.non_tensor_batch)
        if metrics:
            try:
                import wandb

                if wandb.run is not None:
                    if prefix:
                        metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
                    wandb.log(metrics, step=int(step) if step is not None else None, commit=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log agentic reward metrics to W&B: %s", exc)

    @staticmethod
    def _log_tool_latency_metrics(output: Any, step: Any, *, prefix: str = "") -> None:
        try:
            relpaths = output.non_tensor_batch.get("trajectory_relpath")
            if relpaths is None:
                return
            samples: dict[str, list[float]] = {"generate_image": [], "judge_image": []}
            seen: set[str] = set()
            images_root = resolve_rollout_images_root()
            for raw in relpaths:
                relpath = str(raw or "")
                if not relpath or relpath in seen:
                    continue
                seen.add(relpath)
                timing_path = images_root / relpath / "timing.jsonl"
                if not timing_path.is_file():
                    continue
                for line in timing_path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tool = str(row.get("tool", ""))
                    latency = row.get("latency_s")
                    if tool in samples and isinstance(latency, (int, float)):
                        samples[tool].append(float(latency))

            import wandb

            if wandb.run is None:
                return
            metric_prefix = f"{prefix}agentic_tool/"
            for tool, values in samples.items():
                if not values:
                    continue
                values.sort()
                n = len(values)
                wandb.log(
                    {
                        f"{metric_prefix}{tool}_latency_s/mean": sum(values) / n,
                        f"{metric_prefix}{tool}_latency_s/p50": (values[n // 2] + values[(n - 1) // 2]) / 2,
                        f"{metric_prefix}{tool}_latency_s/p95": values[min(n - 1, int(round(n * 0.95)) - 1)],
                        f"{metric_prefix}{tool}_latency_s/max": values[-1],
                        f"{metric_prefix}{tool}_latency_s/count": n,
                    },
                    step=int(step) if step is not None else None,
                    commit=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to aggregate tool-latency metrics: %s", exc)

    def _maybe_run_val_viz(self, step: Any) -> None:
        provider = self._val_viz_provider
        if provider is None:
            return
        try:
            step_i = int(step) if step is not None else -1
        except (TypeError, ValueError):
            step_i = -1
        if step_i in self._val_viz_logged_steps:
            return
        try:
            batch = provider.build_batch(
                step,
                eos_token_id=getattr(self._monitor_tokenizer, "eos_token_id", None),
                pad_token_id=getattr(self._monitor_tokenizer, "pad_token_id", None),
            )
            output = super().generate_sequences(batch)
            dump_raw_rollouts(
                tokenizer=self._monitor_tokenizer,
                output=output,
                step=step,
                write_monitor=False,
                validate=True,
            )
            self._val_viz_logged_steps.add(step_i)
            logger.info("Val holdout viz dumped for step=%s", step_i)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to generate val viz rollouts: %s", exc)
