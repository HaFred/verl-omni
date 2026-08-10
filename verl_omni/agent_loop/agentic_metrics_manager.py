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
from pathlib import Path
from typing import Any

import numpy as np
from verl.experimental.agent_loop import AgentLoopManager
from verl.utils import hf_tokenizer

from verl_omni.agent_loop.agentic_trajectory_context import (
    build_trajectory_relpath,
    clear_good_enough_yes_reached,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)

logger = logging.getLogger(__name__)

# WandB ``agentic_reward/*`` — only the scalar mix terms used in compute_score.
REWARD_COMPONENTS = (
    "reward_tool_call",
    "reward_correctness",
    "reward_aesthetics",
    "reward_done",
)
REWARD_ARTIFACT_FIELDS = (
    *REWARD_COMPONENTS,
    "num_hermes_tool_calls",
    "num_generate_image_prompts",
    "num_judge_image_calls",
    "judge_parse_ok",
    "judge_parse_fail",
    "judge_parse_ok_rate",
    "protocol_ok",
    "rewrite_after_yes",
    "reward_delta_c",
    "first_correctness",
    "first_judge_no",
    "rollout_valid",
)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?:\{.*?\"name\"\s*:\s*\"[^\"]+\".*?\}|"
    r"<function=[^>\s]+\s*>.*?</function>)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_EXECUTED_TOOL_RESPONSE_RE = re.compile(r"\bagentic_(?:tool|reflect|judge)\s+ok=[01]\b", re.IGNORECASE)
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
_JUDGE_CALL_RE = re.compile(r"<function=judge_image\b|\"name\"\s*:\s*\"judge_image\"", re.IGNORECASE)
_GEN_CALL_RE = re.compile(r"<function=generate_image\b|\"name\"\s*:\s*\"generate_image\"", re.IGNORECASE)
_AGENT_REFLECTION_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)
_VL_JUDGE_OBS_RE = re.compile(r"\b(?:VL judge|agentic_judge)\b", re.IGNORECASE)
_FORCED_REFLECTION_RE = re.compile(r"\bagentic_forced_reflection=1\b", re.IGNORECASE)


def _turn_kind(decode: str, turn_prompt: str, response: str = "") -> str:
    """Label turns so trajectory dumps make protocol stages grep-able."""
    resp = response or ""
    if _JUDGE_CALL_RE.search(decode or ""):
        return "call_judge_image"
    if _GEN_CALL_RE.search(decode or ""):
        if _FORCED_REFLECTION_RE.search(resp) or _FORCED_REFLECTION_RE.search(turn_prompt or ""):
            return "agent_rewrite_after_forced_reflection"
        return "call_generate_image"
    if re.search(r"\bagentic_force_stop_max_passes=1\b", resp) or (
        not (decode or "").strip() and re.search(r"\bagentic_force_stop_max_passes=1\b", turn_prompt or "")
    ):
        return "forced_reflection_max_passes_done"
    if _FORCED_REFLECTION_RE.search(resp):
        if re.search(r"\bDone\.", resp):
            return "forced_reflection_done"
        return "forced_reflection_continue"
    if not (decode or "").strip() and _FORCED_REFLECTION_RE.search(turn_prompt or ""):
        if re.search(r"\bDone\.", turn_prompt or ""):
            return "forced_reflection_done"
        return "forced_reflection_continue"
    if _AGENT_REFLECTION_RE.search(decode or ""):
        if _GEN_CALL_RE.search(decode or ""):
            return "agent_reflection_rewrite"
        return "agent_reflection_done"
    if _VL_JUDGE_OBS_RE.search(turn_prompt or ""):
        return "after_judge_feedback"
    if "path=" in (turn_prompt or "") and "agentic_tool" in (turn_prompt or ""):
        return "after_generate_image"
    return "other"


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
        "num_judge_image_calls",
        "protocol_ok",
        "rollout_valid",
    }
    for key in REWARD_ARTIFACT_FIELDS:
        values = output.non_tensor_batch.get(key)
        if values is None:
            continue
        value = np.asarray(values[index]).reshape(-1)[0]
        metrics[key] = int(value) if key in integer_fields else float(value)
    return metrics


def _split_env_blob(blob: str) -> tuple[str, str]:
    """Split mask=0 env text into ``(turn_prompt, response)``.

    ``turn_prompt`` = tool / user obs the policy reads.
    ``response`` = injected assistant text (forced ``Reflection:…``), if any.
    These can diverge from the previous turn's ``decode`` because the agent loop
    may inject Reflection after ``judge_image`` without sampling it from the policy.
    """
    text = (blob or "").strip()
    if not text:
        return "", ""
    force_idx = -1
    for match in re.finditer(r"(?:^|\n)\s*Reflection\s*:", text, re.IGNORECASE):
        # Prefer the forced marker when present.
        window = text[match.start() : match.start() + 400]
        if "agentic_forced_reflection=1" in window or force_idx < 0:
            force_idx = match.start()
            if "agentic_forced_reflection=1" in window:
                break
    if force_idx < 0 or not _FORCED_REFLECTION_RE.search(text):
        # No injected assistant response — entire blob is the next-turn prompt.
        return text, ""
    # Include any chat-template role tags immediately before Reflection.
    cut = force_idx
    preamble = text[:force_idx]
    # If the decode left a trailing bare ``assistant`` / think block before Reflection,
    # keep tool_response in turn_prompt and put Reflection(+trailing) in response.
    tool_end = preamble.rfind("</tool_response>")
    if tool_end >= 0:
        turn_prompt = text[: tool_end + len("</tool_response>")].strip()
        response = text[tool_end + len("</tool_response>") :].strip()
        # Drop leading role/think scaffolding noise from response but keep Reflection.
        refl = re.search(r"Reflection\s*:", response, re.IGNORECASE)
        if refl:
            response = response[refl.start() :].strip()
        return turn_prompt, response
    return text[:cut].strip(), text[cut:].strip()


def _turn_record(
    *,
    turn: int,
    turn_prompt: str,
    response: str,
    decode: str,
) -> dict[str, Any]:
    return {
        "turn": turn,
        "turn_prompt": turn_prompt or "",
        "decode": decode or "",
        "response": response or "",
        "decode_has_tool_call": "<tool_call>" in (decode or "").lower(),
    }


def split_assistant_rollouts(token_ids, response_mask, tokenizer) -> list[str]:
    """Decode contiguous model-token spans; tool observations have mask 0."""
    return [turn["decode"] for turn in split_rollout_turns(token_ids, response_mask, tokenizer)]


def split_rollout_turns(token_ids, response_mask, tokenizer) -> list[dict[str, Any]]:
    """Split response into turns with explicit prompt / response / decode fields.

    ``response_mask==1`` → model tokens (``decode``).
    ``response_mask==0`` → env tokens, split into:
      - ``turn_prompt``: tool obs the policy conditions on
      - ``response``: forced ``Reflection:…`` (injected, not policy-sampled)

    Keys per turn: ``turn``, ``turn_prompt``, ``decode``, ``response``,
    ``decode_has_tool_call``.
    """
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    mask = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
    turns: list[dict[str, Any]] = []
    current_model: list[int] = []
    current_tool: list[int] = []
    pending_prompt = ""
    pending_response = ""

    def _flush_tool() -> None:
        nonlocal pending_prompt, pending_response, current_tool
        if not current_tool:
            return
        blob = tokenizer.decode(current_tool, skip_special_tokens=True).strip()
        current_tool = []
        prompt, response = _split_env_blob(blob)
        pending_prompt = prompt
        pending_response = response

    def _flush_model() -> None:
        nonlocal current_model, pending_prompt, pending_response
        if not current_model:
            return
        decode = tokenizer.decode(current_model, skip_special_tokens=False)
        turns.append(
            _turn_record(
                turn=len(turns) + 1,
                turn_prompt=pending_prompt,
                response=pending_response,
                decode=decode,
            )
        )
        pending_prompt = ""
        pending_response = ""
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
    # Trailing env (e.g. final judge + forced Done with no further decode).
    if current_tool:
        _flush_tool()
        if pending_prompt or pending_response:
            turns.append(
                _turn_record(
                    turn=len(turns) + 1,
                    turn_prompt=pending_prompt,
                    response=pending_response,
                    decode="",
                )
            )
        pending_prompt = ""
        pending_response = ""
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
    """Index images that the tool wrote directly into this rollout directory."""
    target_dir = run_dir / "rollout_images" / relpath
    target_dir.mkdir(parents=True, exist_ok=True)
    prompts = _extract_generate_image_prompts(decoded_response)
    image_paths = [
        str(path)
        for path in sorted(target_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    meta_path = target_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    except json.JSONDecodeError:
        meta = {}
    meta.update(
        {
            "trajectory_relpath": relpath,
            "user_prompt": user_prompt,
            "image_paths": image_paths,
            "tool_prompts": prompts,
            "source": "direct_tool_write",
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return image_paths


class AgenticMetricsAgentLoopManager(AgentLoopManager):
    """Use stock rollout management; observe outputs without changing them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = self.model_config.get("tokenizer_path") or self.model_config.get("path")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self._monitor_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    async def _run_agent_loop(
        self,
        sampling_params,
        trajectory,
        *,
        agent_name,
        trace=True,
        **kwargs,
    ):
        """Bind the final step/sample path before this rollout executes tools."""
        relpath = build_trajectory_relpath(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
        )
        raw_prompt = kwargs.get("raw_prompt")
        user_prompt = _last_user_prompt(raw_prompt) if raw_prompt is not None else ""
        path_token = set_active_trajectory_relpath(relpath)
        prompt_token = set_active_user_prompt(user_prompt)
        # Fresh trajectory: allow generate_image until the first good_enough=YES.
        clear_good_enough_yes_reached()
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
                        len(_EXECUTED_TOOL_RESPONSE_RE.findall(turn.get("turn_prompt", "") or ""))
                        for turn in rollout_turns
                    ),
                    "num_forced_tool_calls": 0,
                    "num_voluntary_tool_calls": sum(
                        len(_TOOL_CALL_RE.findall(turn.get("decode") or "")) for turn in rollout_turns
                    ),
                    # Legacy field name retained for downstream dashboards.
                    "num_voluntary_hermes": sum(
                        len(_TOOL_CALL_RE.findall(turn.get("decode") or "")) for turn in rollout_turns
                    ),
                }
                reward_metrics = _artifact_reward_metrics(output, i)

                # Stable key order for trajectory JSON.
                for turn in rollout_turns:
                    turn["turn_kind"] = _turn_kind(
                        turn.get("decode") or "",
                        turn.get("turn_prompt") or "",
                        turn.get("response") or "",
                    )
                ordered_turns = [
                    {
                        "turn": t.get("turn"),
                        "turn_kind": t.get("turn_kind"),
                        "turn_prompt": t.get("turn_prompt") or "",
                        "decode": t.get("decode") or "",
                        "response": t.get("response") or "",
                        "decode_has_tool_call": bool(t.get("decode_has_tool_call")),
                    }
                    for t in rollout_turns
                ]
                payload["rollout_turns"] = ordered_turns

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
                for turn in ordered_turns:
                    t = int(turn["turn"])
                    turn_prompt = turn.get("turn_prompt") or ""
                    response = turn.get("response") or ""
                    decode = turn.get("decode") or ""
                    kind = turn.get("turn_kind") or "other"
                    header = f"  turn={t} kind={kind} decode_has_tool_call={turn['decode_has_tool_call']}"
                    block = [
                        header,
                        f"    turn_{t}_prompt:",
                        *[f"      {line}" for line in (turn_prompt.splitlines() or [""])],
                        f"    turn_{t}_response:",
                        *[f"      {line}" for line in (response.splitlines() or [""])],
                        "    decode:",
                        *[f"      {line}" for line in (decode.splitlines() or [""])],
                    ]
                    trajectory_text.extend(block)
                    step_text.extend(block)
                trajectory_text.append("")
                step_text.append("")
                (trajectory_dir / f"{name}.txt").write_text("\n".join(trajectory_text) + "\n")
                # ``rollout_trajectories`` is the canonical home of raw
                # decodes. Keep hermes_actions compact and focused on action
                # metadata plus the exact per-rollout reward outputs.
                monitor_payload = {
                    **payload,
                    "rollout_turns": [
                        {key: value for key, value in turn.items() if key != "decode"} for turn in ordered_turns
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
        self._discard_invalid_rollouts(output)
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

    @staticmethod
    def _discard_invalid_rollouts(output: Any) -> None:
        """Drop no-``generate_image`` rollouts from the policy update.

        Sets ``response_mask`` to 0 so GRPO/PPO give them no gradient. Their
        scalar reward is already 0 with ``rollout_valid=0`` from
        ``agentic_reward``; they can still slightly affect the GRPO group mean,
        which is acceptable (penalizes skip-gen relative to siblings).
        """
        valid = output.non_tensor_batch.get("rollout_valid")
        n_gen = output.non_tensor_batch.get("num_generate_image_prompts")
        response_mask = output.batch.get("response_mask")
        if response_mask is None:
            return
        n = int(response_mask.shape[0])
        dropped = 0
        for i in range(n):
            is_valid = True
            if valid is not None:
                try:
                    is_valid = int(np.asarray(valid[i]).reshape(-1)[0]) == 1
                except (TypeError, ValueError, IndexError):
                    is_valid = True
            elif n_gen is not None:
                try:
                    is_valid = int(np.asarray(n_gen[i]).reshape(-1)[0]) >= 1
                except (TypeError, ValueError, IndexError):
                    is_valid = True
            if is_valid:
                continue
            response_mask[i].zero_()
            dropped += 1
        if dropped:
            logger.info(
                "Discarded %d/%d rollouts with no generate_image (response_mask=0, rollout_valid=0)",
                dropped,
                n,
            )
