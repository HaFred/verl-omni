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

from verl_omni.agent_loop.agentic_trajectory_context import (
    build_trajectory_relpath,
    claim_tool_artifacts_for_prompts,
)

logger = logging.getLogger(__name__)

REWARD_COMPONENTS = (
    "reward_tool_call",
    "reward_brevity",
    "reward_format",
    "reward_reflection",
    "reward_tool_usage",
    "reward_result",
    "reward_correctness",
    "reward_aesthetics",
    "reward_correctness_subject_entities",
    "reward_correctness_attributes",
    "reward_correctness_relations_layout",
    "reward_correctness_scene_context",
    "reward_correctness_completeness",
    "reward_aesthetics_composition",
    "reward_aesthetics_lighting",
    "reward_aesthetics_color",
    "reward_aesthetics_fidelity",
    "reward_aesthetics_appeal",
)
REWARD_ARTIFACT_FIELDS = (
    *REWARD_COMPONENTS,
    "num_hermes_tool_calls",
    "num_generate_image_prompts",
    "num_reflect_image_calls",
    "protocol_ok",
)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?:\{.*?\"name\"\s*:\s*\"[^\"]+\".*?\}|"
    r"<function=[^>\s]+\s*>.*?</function>)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_EXECUTED_TOOL_RESPONSE_RE = re.compile(r"\bagentic_(?:tool|reflect)\s+ok=[01]\b", re.IGNORECASE)
_TOOL_ARTIFACT_PATH_RE = re.compile(r"\bpath=([^\s<>]+)")
_HERMES_PROMPT_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_QWEN_XML_PROMPT_RE = re.compile(
    r"<tool_call>\s*<function=generate_image\s*>.*?"
    r"<parameter=prompt\s*>\s*(.*?)\s*</parameter>.*?"
    r"</function>\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)


def _extract_generate_image_prompts(decoded_response: str) -> list[str]:
    """Ordered prompts from Hermes JSON or Qwen3.5 XML tool calls."""
    found: list[tuple[int, str]] = []
    for match in _HERMES_PROMPT_RE.finditer(decoded_response or ""):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("name", "")).strip() != "generate_image":
            continue
        args = payload.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            continue
        prompt = args.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            found.append((match.start(), prompt.strip()))
    for match in _QWEN_XML_PROMPT_RE.finditer(decoded_response or ""):
        prompt = match.group(1).strip()
        if prompt:
            found.append((match.start(), prompt))
    return [prompt for _, prompt in sorted(found)]


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


def _artifact_reward_metrics(output: Any, index: int) -> dict[str, float | int]:
    """Per-rollout scorer outputs for compact ``hermes_actions`` JSONL rows."""
    metrics: dict[str, float | int] = {}
    rm_scores = output.batch.get("rm_scores")
    if rm_scores is not None:
        # AgentLoopManager writes the scalar reward on the final valid response
        # token; the first token is normally zero. Sum the token-level tensor.
        value = np.asarray(rm_scores[index].detach().cpu()).sum()
        metrics["score"] = float(value)

    integer_fields = {
        "num_hermes_tool_calls",
        "num_generate_image_prompts",
        "num_reflect_image_calls",
        "protocol_ok",
    }
    for key in REWARD_ARTIFACT_FIELDS:
        values = output.non_tensor_batch.get(key)
        if values is None:
            continue
        value = np.asarray(values[index]).reshape(-1)[0]
        metrics[key] = int(value) if key in integer_fields else float(value)
    return metrics


def split_assistant_rollouts(token_ids, response_mask, tokenizer) -> list[str]:
    """Decode contiguous model-token spans; tool observations have mask 0."""
    return [turn["decode"] for turn in split_rollout_turns(token_ids, response_mask, tokenizer)]


def split_rollout_turns(token_ids, response_mask, tokenizer) -> list[dict[str, Any]]:
    """Split response into assistant turns with the tool-obs that preceded each turn.

    ``response_mask==1`` → model tokens (assistant decode).
    ``response_mask==0`` → tool / env tokens (recorded as the next turn's ``turn_prompt``).

    Turn 1 has no prior tool obs in the response tensor; the caller should set
    ``turn_prompt`` to the dataset ``user_prompt`` when empty.
    """
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    mask = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
    turns: list[dict[str, Any]] = []
    current_model: list[int] = []
    current_tool: list[int] = []
    pending_tool_prompt = ""

    def _flush_tool() -> None:
        nonlocal pending_tool_prompt, current_tool
        if not current_tool:
            return
        # Drop vision/special pads so monitors keep path= / image_vis / agentic_tool.
        pending_tool_prompt = tokenizer.decode(current_tool, skip_special_tokens=True).strip()
        current_tool = []

    def _flush_model() -> None:
        nonlocal current_model, pending_tool_prompt
        if not current_model:
            return
        decode = tokenizer.decode(current_model, skip_special_tokens=False)
        turns.append(
            {
                "turn": len(turns) + 1,
                "turn_prompt": pending_tool_prompt,
                "decode": decode,
                "decode_has_tool_call": "<tool_call>" in decode.lower(),
            }
        )
        pending_tool_prompt = ""
        current_model = []

    for token_id, is_model_token in zip(ids, mask, strict=True):
        if int(is_model_token) == 1:
            if current_tool:
                _flush_tool()
            current_model.append(int(token_id))
        else:
            if current_model:
                _flush_model()
            current_tool.append(int(token_id))
    if current_model:
        _flush_model()
    # Trailing tool obs (after final assistant tool_call) is kept on a synthetic note.
    if current_tool:
        _flush_tool()
        if turns:
            turns[-1]["tool_response_after"] = pending_tool_prompt
        pending_tool_prompt = ""
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
    """Move live tool artifacts into ``step_*/sample_*/image_00.png``, ``image_01.png``, …

    Stock ``ToolAgentLoop`` does not bind trajectory contextvars, so live saves land
    under ``call_<ts>_<uuid>/``. Attached vision tokens also often truncate later
    ``path=`` markers from the logged response. We therefore:

    1. Claim registry rows by ordered Hermes ``generate_image`` prompts (preferred)
    2. Fall back to any remaining ``path=`` markers in the decoded response
    3. Remove the temporary ``call_*`` staging directory after all its files move
    """
    target_dir = run_dir / "rollout_images" / relpath
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    seen: set[Path] = set()
    staging_dirs: set[Path] = set()
    image_n = 0
    call_n = 0

    def _copy_source(source: Path) -> None:
        nonlocal image_n, call_n
        if source in seen or not source.is_file():
            return
        seen.add(source)
        if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            destination = target_dir / f"image_{image_n:02d}{source.suffix.lower()}"
            image_n += 1
        elif source.name.startswith("STUB_NO_IMAGE"):
            destination = target_dir / f"STUB_NO_IMAGE_{call_n:02d}.txt"
        else:
            return
        source_meta = source.parent / "meta.json"
        if source_meta.is_file():
            shutil.copy2(source_meta, target_dir / f"call_{call_n:02d}_meta.json")
        # The call_* path is only a live staging location. Move instead of copy
        # so every persisted generated image has one canonical step/sample path.
        shutil.move(str(source), str(destination))
        copied.append(str(destination))
        if source.parent.name.startswith("call_"):
            staging_dirs.add(source.parent)
        call_n += 1

    prompts = _extract_generate_image_prompts(decoded_response)
    for entry in claim_tool_artifacts_for_prompts(prompts):
        for raw in entry.get("paths") or []:
            _copy_source(Path(raw))

    # Fallback / fill-in when registry was empty (e.g. manager-only dump of old run).
    for raw_path in _TOOL_ARTIFACT_PATH_RE.findall(decoded_response):
        source = Path(raw_path.rstrip("',\".;"))
        # path= may point at the PNG or a stub file.
        _copy_source(source)

    for staging_dir in staging_dirs:
        # Preserve per-call metadata above, then remove the staging envelope.
        source_meta = staging_dir / "meta.json"
        if source_meta.is_file():
            source_meta.unlink()
        try:
            staging_dir.rmdir()
        except OSError:
            # A multi-image/concurrent call may still have an unclaimed file.
            logger.debug("Keeping non-empty tool staging directory: %s", staging_dir)

    if copied:
        (target_dir / "meta.json").write_text(
            json.dumps(
                {
                    "trajectory_relpath": relpath,
                    "user_prompt": user_prompt,
                    "image_paths": copied,
                    "tool_prompts": prompts,
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
                rollout_turns = split_rollout_turns(
                    responses[i],
                    response_masks[i],
                    self._monitor_tokenizer,
                )
                if rollout_turns and not rollout_turns[0].get("turn_prompt"):
                    rollout_turns[0]["turn_prompt"] = user_prompt
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
                        len(
                            _EXECUTED_TOOL_RESPONSE_RE.findall(
                                f"{turn.get('turn_prompt', '')}\n{turn.get('tool_response_after', '')}"
                            )
                        )
                        for turn in rollout_turns
                    ),
                    "num_forced_tool_calls": 0,
                    "num_voluntary_tool_calls": sum(
                        len(_TOOL_CALL_RE.findall(turn["decode"])) for turn in rollout_turns
                    ),
                    # Legacy field name retained for downstream dashboards.
                    "num_voluntary_hermes": sum(len(_TOOL_CALL_RE.findall(turn["decode"])) for turn in rollout_turns),
                }
                reward_metrics = _artifact_reward_metrics(output, i)

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
                    t = int(turn["turn"])
                    header = f"  turn={t} decode_has_tool_call={turn['decode_has_tool_call']}"
                    turn_prompt = turn.get("turn_prompt") or ""
                    trajectory_text.extend(
                        [
                            header,
                            f"    turn_{t}_prompt:",
                            *[f"      {line}" for line in (turn_prompt.splitlines() or [""])],
                            "    decode:",
                        ]
                    )
                    step_text.extend(
                        [
                            header,
                            f"    turn_{t}_prompt:",
                            *[f"      {line}" for line in (turn_prompt.splitlines() or [""])],
                            "    decode:",
                        ]
                    )
                    decode_lines = turn["decode"].splitlines() or [""]
                    trajectory_text.extend(f"      {line}" for line in decode_lines)
                    step_text.extend(f"      {line}" for line in decode_lines)
                    after = turn.get("tool_response_after") or ""
                    if after:
                        trajectory_text.extend(
                            [
                                f"    turn_{t}_tool_response:",
                                *[f"      {line}" for line in after.splitlines() or [""]],
                            ]
                        )
                        step_text.extend(
                            [
                                f"    turn_{t}_tool_response:",
                                *[f"      {line}" for line in after.splitlines() or [""]],
                            ]
                        )
                trajectory_text.append("")
                step_text.append("")
                (trajectory_dir / f"{name}.txt").write_text("\n".join(trajectory_text) + "\n")
                # ``rollout_trajectories`` is the canonical home of raw
                # decodes. Keep hermes_actions compact and focused on action
                # metadata plus the exact per-rollout reward outputs.
                monitor_payload = {
                    **payload,
                    "rollout_turns": [
                        {key: value for key, value in turn.items() if key != "decode"} for turn in rollout_turns
                    ],
                    "reward_metrics": reward_metrics,
                }
                jsonl_rows.append(json.dumps(monitor_payload, ensure_ascii=False))

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
