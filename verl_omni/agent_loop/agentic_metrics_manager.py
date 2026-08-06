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

"""Stock AgentLoopManager with reward metrics and raw-rollout monitoring.

The agent implementation remains verl's registered ``tool_agent``. Monitoring
is done here, after generation, so it cannot alter, force, or replace tokens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from verl.experimental.agent_loop import AgentLoopManager
from verl.utils import hf_tokenizer

from verl_omni.agent_loop.agentic_trajectory_context import build_trajectory_relpath

logger = logging.getLogger(__name__)

REWARD_COMPONENTS = (
    "reward_format",
    "reward_reflection",
    "reward_tool_usage",
    "reward_result",
)
_HERMES_GENERATE_IMAGE_RE = re.compile(
    r"<tool_call>\s*\{.*?\"name\"\s*:\s*\"generate_image\".*?\}\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_ARTIFACT_PATH_RE = re.compile(r"\bpath=([^\s<>]+)")


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


def split_assistant_rollouts(token_ids, response_mask, tokenizer) -> list[str]:
    """Decode contiguous model-token spans; tool observations have mask 0."""
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    mask = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
    turns: list[str] = []
    current: list[int] = []
    for token_id, is_model_token in zip(ids, mask, strict=True):
        if int(is_model_token) == 1:
            current.append(int(token_id))
        elif current:
            turns.append(tokenizer.decode(current, skip_special_tokens=False))
            current = []
    if current:
        turns.append(tokenizer.decode(current, skip_special_tokens=False))
    return turns


def _last_user_prompt(raw_prompt: Any) -> str:
    messages = list(raw_prompt) if raw_prompt is not None else []
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content)
    return ""


def _run_dir() -> Path:
    image_dir = os.getenv("AGENTIC_DIFFUSION_IMAGE_DIR", "").strip()
    if image_dir:
        return Path(image_dir).parent
    root = Path(os.getenv("AGENTIC_E2E_ROOT", "outputs/e2e"))
    return root / os.getenv("AGENTIC_E2E_RUN_NAME", "agentic_run")


def _materialize_rollout_images(
    *,
    decoded_response: str,
    run_dir: Path,
    relpath: str,
    user_prompt: str,
) -> list[str]:
    """Copy stock function-tool artifacts into the stable step/sample layout."""
    target_dir = run_dir / "rollout_images" / relpath
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    seen: set[Path] = set()
    image_n = 0
    call_n = 0
    for raw_path in _TOOL_ARTIFACT_PATH_RE.findall(decoded_response):
        source = Path(raw_path.rstrip("',\".;"))
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            destination = target_dir / f"image_{image_n:02d}{source.suffix.lower()}"
            image_n += 1
        elif source.name == "STUB_NO_IMAGE.txt":
            destination = target_dir / f"STUB_NO_IMAGE_{call_n:02d}.txt"
        else:
            continue
        shutil.copy2(source, destination)
        copied.append(str(destination))
        source_meta = source.parent / "meta.json"
        if source_meta.is_file():
            shutil.copy2(source_meta, target_dir / f"call_{call_n:02d}_meta.json")
        call_n += 1
    if copied:
        (target_dir / "meta.json").write_text(
            json.dumps(
                {
                    "trajectory_relpath": relpath,
                    "user_prompt": user_prompt,
                    "image_paths": copied,
                    "source": "stock_tool_agent_manager_copy",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    return copied


class AgenticMetricsAgentLoopManager(AgentLoopManager):
    """Use stock rollout management; observe outputs without changing them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = self.model_config.get("tokenizer_path") or self.model_config.get("path")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self._monitor_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    def _dump_raw_rollouts(self, prompts, output, step) -> None:
        """Write user prompt + raw assistant turns only."""
        try:
            responses = output.batch["responses"]
            response_masks = output.batch["response_mask"]
            raw_prompts = output.non_tensor_batch.get("raw_prompt")
            indices = output.non_tensor_batch.get("index", np.arange(len(responses)))
            step_i = int(step) if step is not None else -1
            step_tag = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
            run_dir = _run_dir()
            monitor_dir = run_dir / "hermes_actions"
            trajectory_dir = run_dir / "rollout_trajectories" / step_tag
            monitor_dir.mkdir(parents=True, exist_ok=True)
            trajectory_dir.mkdir(parents=True, exist_ok=True)

            rollout_counts: dict[str, int] = {}
            step_text: list[str] = []
            jsonl_rows: list[str] = []
            for i in range(len(responses)):
                sample_index = indices[i]
                sample_key = str(int(sample_index)) if str(sample_index).lstrip("-").isdigit() else str(sample_index)
                rollout_n = rollout_counts.get(sample_key, 0)
                rollout_counts[sample_key] = rollout_n + 1
                relpath = build_trajectory_relpath(
                    step=step_i,
                    sample_index=sample_index,
                    rollout_n=rollout_n,
                )
                user_prompt = _last_user_prompt(raw_prompts[i]) if raw_prompts is not None else ""
                turns = split_assistant_rollouts(
                    responses[i],
                    response_masks[i],
                    self._monitor_tokenizer,
                )
                rollout_turns = [
                    {
                        "turn": turn_i,
                        "decode": decode,
                        "decode_has_tool_call": "<tool_call>" in decode.lower(),
                    }
                    for turn_i, decode in enumerate(turns, start=1)
                ]
                decoded_response = self._monitor_tokenizer.decode(
                    responses[i].tolist(),
                    skip_special_tokens=False,
                )
                image_paths = _materialize_rollout_images(
                    decoded_response=decoded_response,
                    run_dir=run_dir,
                    relpath=relpath,
                    user_prompt=user_prompt,
                )
                payload = {
                    "trajectory_relpath": relpath,
                    "step": step_i,
                    "sample_index": int(sample_index) if str(sample_index).lstrip("-").isdigit() else str(sample_index),
                    "rollout_n": rollout_n,
                    "user_prompt": user_prompt,
                    "rollout_turns": rollout_turns,
                    "image_paths": image_paths,
                    "num_tool_calls_executed": sum(
                        len(_HERMES_GENERATE_IMAGE_RE.findall(turn["decode"])) for turn in rollout_turns
                    ),
                    "num_forced_tool_calls": 0,
                    "num_voluntary_hermes": sum(
                        len(_HERMES_GENERATE_IMAGE_RE.findall(turn["decode"])) for turn in rollout_turns
                    ),
                }

                name = Path(relpath).name
                (trajectory_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
                trajectory_text = [
                    f"relpath={relpath}",
                    f"user_prompt: {user_prompt}",
                    "assistant_rollout:",
                ]
                step_text.extend(
                    [
                        f"=== {name}  sample={sample_index} rollout_n={rollout_n} ===",
                        f"user_prompt: {user_prompt}",
                        "assistant_rollout:",
                    ]
                )
                for turn in rollout_turns:
                    header = f"  turn={turn['turn']} decode_has_tool_call={turn['decode_has_tool_call']}"
                    trajectory_text.extend([header, "    decode:"])
                    step_text.extend([header, "    decode:"])
                    decode_lines = turn["decode"].splitlines() or [""]
                    trajectory_text.extend(f"      {line}" for line in decode_lines)
                    step_text.extend(f"      {line}" for line in decode_lines)
                trajectory_text.append("")
                step_text.append("")
                (trajectory_dir / f"{name}.txt").write_text("\n".join(trajectory_text) + "\n")
                jsonl_rows.append(json.dumps(payload, ensure_ascii=False))

            (monitor_dir / f"{step_tag}.txt").write_text("\n".join(step_text) + "\n")
            (monitor_dir / f"{step_tag}.jsonl").write_text("\n".join(jsonl_rows) + "\n")
        except Exception as exc:  # noqa: BLE001
            # Monitoring must never fail or alter rollout generation.
            logger.warning("Failed to dump raw agent rollouts: %s", exc)

    def generate_sequences(self, prompts):
        step = prompts.meta_info.get("global_steps")
        output = super().generate_sequences(prompts)
        self._dump_raw_rollouts(prompts, output, step)
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
