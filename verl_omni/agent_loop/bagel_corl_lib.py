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

"""Hermes tool protocol, J×K IDs, serial episode, and flatten for Bagel Co-RL."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from verl_omni.agent_loop.image_gen_trajectory_context import (
    build_generate_call_meta,
    clear_good_enough_yes_reached,
    get_active_user_prompt,
    get_good_enough_yes_reached,
    register_tool_artifact,
    set_good_enough_yes_reached,
)
from verl_omni.agent_loop.rpco_turn_protocol import (
    build_forced_reflection,
    derive_good_enough_from_scores,
    format_rm_scores_as_judge_text,
)


logger = logging.getLogger(__name__)

GENERATE_IMAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image from a text prompt using the Bagel GEN pathway.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt for image generation."},
            },
            "required": ["prompt"],
        },
    },
}

HERMES_SPECIAL_TOKENS: tuple[str, ...] = (
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_DONE_RE = re.compile(r"\bDone\.\s*$", re.IGNORECASE)


class GenerateImageCapError(RuntimeError):
    """Raised when a second GEN call is attempted under ``max_generate_passes=1``."""


def parse_hermes_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a Hermes ``<tool_call>{...}</tool_call>`` span.

    Returns:
        Parsed JSON object, or ``None`` if no tool call is present.
    """
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    payload = json.loads(match.group(1))
    if "name" not in payload and "function" in payload:
        payload = payload["function"]
    return payload


def und_turn_kind(text: str) -> str:
    """Classify an UND decode: ``generate_image``, ``done``, or ``continue``."""
    call = parse_hermes_tool_call(text)
    if call is not None:
        name = str(call.get("name", ""))
        if name == "generate_image":
            return "generate_image"
        raise ValueError(f"Bagel CoRL UND emitted unsupported tool {name!r}; Qwen/other tools are fail-closed")
    if _DONE_RE.search(text.strip()):
        return "done"
    return "continue"


def bind_episode_ids(
    *,
    dataset_task_uid: str,
    episode_uid: str | None = None,
    policy_version: int = 0,
    gen_call_id: str | None = None,
) -> dict[str, str | int]:
    """RFC component-1 IDs. FlowGRPO groups by ``gen_group_uid``, never by semantic slot."""
    episode_uid = episode_uid or str(uuid.uuid4())
    gen_call_id = gen_call_id or str(uuid.uuid4())
    return {
        "dataset_task_uid": dataset_task_uid,
        "und_group_uid": dataset_task_uid,
        "episode_uid": episode_uid,
        "policy_version": int(policy_version),
        "gen_call_id": gen_call_id,
        "gen_group_uid": gen_call_id,
    }


def gen_sample_uid(gen_call_id: str, seed_index: int) -> str:
    return f"{gen_call_id}:{seed_index}"


@dataclass
class GenSample:
    """One of S FlowGRPO seeds for a single ``generate_image`` turn."""

    gen_sample_uid: str
    gen_group_uid: str
    seed_index: int
    valid: bool
    prompt_token_ids: list[int]
    all_latents: Any | None = None
    timesteps: Any | None = None
    rollout_log_probs: Any | None = None
    rm_score: float | None = None
    image_path: str | None = None
    call_role: str = "initial"
    good_enough: bool | None = None


@dataclass
class EpisodeRollout:
    """One serial UND episode (zero or more GEN calls).

    Runtime counters (RFC): ``turns`` is in-episode ``J`` (UND policy turns);
    ``num_gen_calls`` is in-episode ``K`` (``generate_image`` invocations). Invariant ``J >= K``.
    """

    und_group_uid: str
    episode_uid: str
    policy_version: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    turns: int
    gen_samples: list[GenSample] = field(default_factory=list)
    used_image_credit: bool = False
    forced_reflection: bool = False
    und_reward: float = 0.0
    judge_text: str | None = None
    stop_required: bool = False
    num_gen_calls: int = 0


class BagelGenerateImageTool:
    """One GEN turn per call (RFC ``K`` += 1).

    ``gen_samples_per_call`` is **S** (FlowGRPO seeds under the same conditioning),
    not the episode GEN-turn count **K**. One UND ``generate_image`` tool verdict
    invokes this once → one GEN turn; the seed fan-out is group sampling for FlowGRPO.
    ``max_generate_passes`` bounds how many such GEN turns an episode may enqueue.
    """

    def __init__(
        self,
        *,
        gen_samples_per_call: int,
        max_generate_passes: int = 1,
        generate_fn: Callable[..., Any] | None = None,
    ):
        if gen_samples_per_call < 1:
            raise ValueError("gen_samples_per_call must be >= 1")
        if max_generate_passes < 1:
            raise ValueError("max_generate_passes must be >= 1")
        self.s = int(gen_samples_per_call)
        self.max_generate_passes = int(max_generate_passes)
        self._passes = 0
        self._generate_fn = generate_fn

    def remaining_passes(self) -> int:
        return max(0, self.max_generate_passes - self._passes)

    async def __call__(
        self,
        *,
        prompt: str,
        prompt_token_ids: list[int],
        gen_call_id: str,
        seeds: list[int] | None = None,
    ) -> list[GenSample]:
        if self._passes >= self.max_generate_passes:
            raise GenerateImageCapError(
                f"max_generate_passes={self.max_generate_passes} refuses a second generate_image call"
            )
        self._passes += 1
        if seeds is None:
            seeds = list(range(self.s))
        if len(seeds) != self.s:
            raise ValueError(f"expected S={self.s} FlowGRPO seeds for one GEN turn, got {len(seeds)}")
        raw_rows: list[dict[str, Any]]
        if self._generate_fn is None:
            raw_rows = [{"valid": True} for _ in seeds]
        else:
            maybe = self._generate_fn(
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                seeds=seeds,
                gen_call_id=gen_call_id,
            )
            raw_rows = await maybe if asyncio.iscoroutine(maybe) else maybe
        samples: list[GenSample] = []
        for seed_index, (seed, row) in enumerate(zip(seeds, raw_rows, strict=True)):
            valid = bool(row.get("valid", True))
            samples.append(
                GenSample(
                    gen_sample_uid=gen_sample_uid(gen_call_id, seed_index),
                    gen_group_uid=gen_call_id,
                    seed_index=seed_index,
                    valid=valid,
                    prompt_token_ids=list(prompt_token_ids),
                    all_latents=row.get("all_latents"),
                    timesteps=row.get("timesteps"),
                    rollout_log_probs=row.get("rollout_log_probs"),
                    image_path=row.get("image_path"),
                    call_role="initial",
                )
            )
            _ = seed
        return samples


def compact_image_observation(path: str) -> str:
    """UND-facing observation: path only; S seed trajectories stay on the GEN batch."""
    return f"path={path}"


def judge_text_from_gen_samples(samples: list[GenSample]) -> str | None:
    """Average RM scores into Mode-2a judge text for ``build_forced_reflection``."""
    scores = [float(s.rm_score) for s in samples if s.valid and s.rm_score is not None]
    if not scores:
        return None
    mean = float(sum(scores) / len(scores))
    explicit = [bool(s.good_enough) for s in samples if s.good_enough is not None]
    if explicit:
        good_enough = any(explicit)
    else:
        good_enough = derive_good_enough_from_scores(
            correctness=mean,
            aesthetics=mean,
            similarity=None,
        )
    for sample in samples:
        if sample.good_enough is None:
            sample.good_enough = good_enough
    return format_rm_scores_as_judge_text(
        correctness=mean,
        aesthetics=mean,
        good_enough=good_enough,
        similarity=None,
    )


def _encode_text(tokenizer: Any | None, text: str, *, fallback_ids: list[int] | None = None) -> list[int]:
    if tokenizer is not None:
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(fallback_ids or [])


def turn_histogram(turns: list[int]) -> dict[str, float]:
    if not turns:
        return {
            "turns_per_episode/mean": 0.0,
            "turns_per_episode/p50": 0.0,
            "turns_per_episode/p95": 0.0,
            "turns_per_episode/max": 0.0,
            "tail_frac": 0.0,
        }
    arr = np.asarray(turns, dtype=np.float64)
    p95 = float(np.percentile(arr, 95))
    max_t = float(arr.max())
    tail = float(np.mean(arr >= max(p95, 1.0))) if max_t > 0 else 0.0
    return {
        "turns_per_episode/mean": float(arr.mean()),
        "turns_per_episode/p50": float(np.percentile(arr, 50)),
        "turns_per_episode/p95": p95,
        "turns_per_episode/max": max_t,
        "tail_frac": tail,
    }


async def run_serial_episode(
    *,
    dataset_task_uid: str,
    policy_version: int,
    prompt_ids: list[int],
    und_decode: Callable[..., Any],
    generate_tool: BagelGenerateImageTool,
    score_fn: Callable[[list[GenSample]], Any] | None = None,
    build_judge_text_fn: Callable[[list[GenSample]], str | None] | None = None,
    tokenizer: Any | None = None,
    user_prompt: str = "",
    max_und_turns: int = 8,
    forced_reflection_text: str = "Done.",
    episode_uid: str | None = None,
    non_image_reward: float | None = None,
) -> EpisodeRollout:
    """Serial UND turn(s) → optional one GEN turn (S FlowGRPO seeds) → RM → reflection / Done.

    Episode shape follows the RFC: ``UND (+ GEN) (+ UND…)* + Done``, where each
    ``generate_image`` verdict enqueues exactly one GEN turn (``K += 1``). ``S`` is
    seed fan-out inside that turn for FlowGRPO — not additional GEN turns.

    ``non_image_reward`` is the RFC "non-image UND scalar" used when the episode
    makes zero ``generate_image`` calls (pattern 3, ``K = 0``). When it is ``None``
    and no image was scored, the episode reward is 0.0 and the caller should let the
    reward model fill it post-hoc.
    """
    ids = bind_episode_ids(
        dataset_task_uid=dataset_task_uid,
        episode_uid=episode_uid,
        policy_version=policy_version,
    )
    response_ids: list[int] = []
    response_mask: list[int] = []
    gen_samples: list[GenSample] = []
    used_image_credit = False
    forced = False
    turns = 0
    num_gen_calls = 0
    judge_text: str | None = None
    stop_required = False
    clear_good_enough_yes_reached()
    active_user_prompt = user_prompt or get_active_user_prompt() or ""

    for _ in range(max_und_turns):
        turns += 1
        decode = und_decode(prompt_ids=prompt_ids, response_ids=response_ids)
        step = await decode if asyncio.iscoroutine(decode) else decode
        token_ids: list[int] = list(step["token_ids"])
        text: str = str(step["text"])
        kind = und_turn_kind(text)
        response_ids.extend(token_ids)
        response_mask.extend([1] * len(token_ids))
        if kind == "done":
            break
        if kind != "generate_image":
            continue

        if get_good_enough_yes_reached():
            # Env hard-stop: prior YES latch blocks further generate_image.
            break

        call = parse_hermes_tool_call(text) or {}
        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        prompt = str(arguments.get("prompt", ""))
        meta = build_generate_call_meta(prompt=prompt, user_prompt=active_user_prompt)
        gen_call_id = str(ids["gen_call_id"]) if generate_tool._passes == 0 else str(uuid.uuid4())
        num_gen_calls += 1
        call_samples = await generate_tool(
            prompt=prompt,
            prompt_token_ids=list(prompt_ids) + list(response_ids),
            gen_call_id=gen_call_id,
        )
        call_role = str(meta.get("call_role") or "initial")
        for sample in call_samples:
            sample.call_role = call_role

        valid_paths = [s.image_path for s in call_samples if s.valid and s.image_path]
        if valid_paths:
            register_tool_artifact(prompt=prompt, paths=[str(p) for p in valid_paths])

        if score_fn is not None:
            scored = score_fn(call_samples)
            if asyncio.iscoroutine(scored):
                scored = await scored
            call_samples = scored

        gen_samples = call_samples

        if valid_paths:
            used_image_credit = True
            obs = compact_image_observation(valid_paths[0])
            obs_ids = _encode_text(
                tokenizer,
                obs,
                fallback_ids=list(step.get("obs_token_ids") or [0]),
            )
            if not obs_ids:
                obs_ids = [0]
            response_ids.extend(obs_ids)
            response_mask.extend([0] * len(obs_ids))

        if build_judge_text_fn is not None:
            maybe_text = build_judge_text_fn(call_samples)
            if asyncio.iscoroutine(maybe_text):
                maybe_text = await maybe_text
            judge_text = maybe_text
        if not judge_text:
            judge_text = judge_text_from_gen_samples(call_samples)

        if judge_text:
            remaining = generate_tool.remaining_passes()
            force_done = remaining == 0
            reflection = build_forced_reflection(
                judge_text,
                force_done=force_done,
                generate_pass=generate_tool.max_generate_passes - remaining,
                max_passes=generate_tool.max_generate_passes,
            )
            if reflection is not None:
                reflection_text, stop_required = reflection
                forced_ids = _encode_text(
                    tokenizer,
                    reflection_text,
                    fallback_ids=list(step.get("forced_token_ids") or []),
                )
                if forced_ids:
                    forced = True
                    response_ids.extend(forced_ids)
                    response_mask.extend([0] * len(forced_ids))
                if any(s.good_enough for s in call_samples if s.good_enough is not None):
                    set_good_enough_yes_reached(True)
                elif "good_enough=YES" in judge_text:
                    set_good_enough_yes_reached(True)

                if stop_required:
                    done_decode = und_decode(prompt_ids=prompt_ids, response_ids=response_ids)
                    done_step = await done_decode if asyncio.iscoroutine(done_decode) else done_decode
                    done_ids = list(done_step.get("token_ids") or [])
                    done_text = str(done_step.get("text") or "")
                    if done_ids and und_turn_kind(done_text) == "done":
                        response_ids.extend(done_ids)
                        response_mask.extend([1] * len(done_ids))
                    else:
                        fallback = _encode_text(
                            tokenizer,
                            "Done.",
                            fallback_ids=list(done_step.get("done_token_ids") or step.get("done_token_ids") or []),
                        )
                        if fallback:
                            response_ids.extend(fallback)
                            response_mask.extend([1] * len(fallback))
                    break

                if remaining > 0:
                    continue
                # remaining == 0 should have set stop_required via force_done
                break

        # No judge text: keep legacy forced/done token fallbacks for unit tests.
        forced_ids = list(step.get("forced_token_ids") or [])
        if forced_ids:
            forced = True
            response_ids.extend(forced_ids)
            response_mask.extend([0] * len(forced_ids))
        else:
            done_ids = list(step.get("done_token_ids") or [])
            if done_ids:
                response_ids.extend(done_ids)
                response_mask.extend([1] * len(done_ids))
        break
    else:
        forced = True
        _ = forced_reflection_text

    image_scores = [float(s.rm_score) for s in gen_samples if s.valid and s.rm_score is not None]
    if image_scores:
        und_reward = float(np.mean(image_scores))
    elif non_image_reward is not None:
        # Pattern 3 (K=0): use the non-image UND scalar (RFC und/no_image_credit),
        # never a hard-coded 0 that would zero out token GRPO signal.
        und_reward = float(non_image_reward)
    else:
        und_reward = 0.0

    return EpisodeRollout(
        und_group_uid=str(ids["und_group_uid"]),
        episode_uid=str(ids["episode_uid"]),
        policy_version=int(ids["policy_version"]),
        prompt_ids=list(prompt_ids),
        response_ids=response_ids,
        response_mask=response_mask,
        turns=turns,
        gen_samples=gen_samples,
        used_image_credit=used_image_credit,
        forced_reflection=forced,
        und_reward=und_reward,
        judge_text=judge_text,
        stop_required=stop_required,
        num_gen_calls=num_gen_calls,
    )


@dataclass
class FlattenResult:
    und_batch: list[dict[str, Any]]
    gen_batch: list[dict[str, Any]]
    gen_episode_map: list[dict[str, Any]]
    metrics: dict[str, float]


def flatten_multiturn_rollouts(
    episodes: list[EpisodeRollout],
    *,
    expected_s: int,
) -> FlattenResult:
    """Split token UND rows from latent GEN rows. Never concatenate the two.

    Incomplete S-seed groups are dropped. Reflection-only episodes contribute zero GEN rows.
    One finished ``generate_image`` tool call is one GEN turn (RFC K); ``expected_s`` is
    FlowGRPO seeds under that single turn.
    """
    if expected_s < 1:
        raise ValueError("expected_s must be >= 1")
    und_batch: list[dict[str, Any]] = []
    gen_batch: list[dict[str, Any]] = []
    gen_episode_map: list[dict[str, Any]] = []
    dropped_incomplete = 0
    no_image_credit = 0

    for und_index, episode in enumerate(episodes):
        image_scores = [float(s.rm_score) for s in episode.gen_samples if s.valid and s.rm_score is not None]
        if image_scores:
            und_reward = float(np.mean(image_scores))
        else:
            # Pattern 3 (K=0): propagate the non-image UND scalar instead of zeroing
            # token GRPO signal (RFC und/no_image_credit).
            und_reward = float(episode.und_reward)
            no_image_credit += 1
        und_batch.append(
            {
                "und_group_uid": episode.und_group_uid,
                "episode_uid": episode.episode_uid,
                "policy_version": episode.policy_version,
                "prompt_ids": list(episode.prompt_ids),
                "response_ids": list(episode.response_ids),
                "response_mask": list(episode.response_mask),
                "token_level_scores": und_reward,
                "used_image_credit": episode.used_image_credit,
            }
        )
        valid = [s for s in episode.gen_samples if s.valid]
        if not valid:
            continue
        if len(valid) != expected_s:
            dropped_incomplete += 1
            continue
        for sample in valid:
            gen_batch.append(
                {
                    "gen_group_uid": sample.gen_group_uid,
                    "gen_sample_uid": sample.gen_sample_uid,
                    "seed_index": sample.seed_index,
                    "prompt_token_ids": list(sample.prompt_token_ids),
                    "all_latents": sample.all_latents,
                    "timesteps": sample.timesteps,
                    "rollout_log_probs": sample.rollout_log_probs,
                    "rm_score": sample.rm_score,
                    "call_role": sample.call_role,
                }
            )
            gen_episode_map.append(
                {
                    "und_index": und_index,
                    "episode_uid": episode.episode_uid,
                    "gen_sample_uid": sample.gen_sample_uid,
                    "gen_group_uid": sample.gen_group_uid,
                }
            )

    metrics = {
        "gen/dropped_incomplete_groups": float(dropped_incomplete),
        "und/no_image_credit": float(no_image_credit),
        "gen/num_rows": float(len(gen_batch)),
        "und/num_rows": float(len(und_batch)),
        "gen/skipped_no_groups": 1.0 if not gen_batch else 0.0,
    }
    return FlattenResult(
        und_batch=und_batch,
        gen_batch=gen_batch,
        gen_episode_map=gen_episode_map,
        metrics=metrics,
    )


def _as_gen_sample(item: Any) -> GenSample | None:
    if isinstance(item, GenSample):
        return item
    if not isinstance(item, dict):
        return None
    return GenSample(
        gen_sample_uid=str(item.get("gen_sample_uid", "")),
        gen_group_uid=str(item.get("gen_group_uid", "")),
        seed_index=int(item.get("seed_index", 0)),
        valid=bool(item.get("valid", False)),
        prompt_token_ids=list(item.get("prompt_token_ids") or []),
        all_latents=item.get("all_latents"),
        timesteps=item.get("timesteps"),
        rollout_log_probs=item.get("rollout_log_probs"),
        rm_score=item.get("rm_score"),
        image_path=item.get("image_path"),
        call_role=str(item.get("call_role", "initial")),
        good_enough=item.get("good_enough"),
    )


def _ntb_from_extra_fields(extras: Any) -> dict[str, list[Any]]:
    rows = list(extras or [])
    ntb: dict[str, list[Any]] = {
        "gen_samples": [],
        "und_group_uid": [],
        "episode_uid": [],
        "prompt_ids": [],
        "response_ids": [],
        "response_mask": [],
        "turns": [],
        "used_image_credit": [],
    }
    for extra in rows:
        mapping = extra if isinstance(extra, dict) else {}
        ntb["gen_samples"].append(mapping.get("gen_samples") or [])
        ntb["und_group_uid"].append(mapping.get("und_group_uid", "missing_task"))
        ntb["episode_uid"].append(mapping.get("episode_uid"))
        ntb["prompt_ids"].append(mapping.get("prompt_ids") or [])
        ntb["response_ids"].append(mapping.get("response_ids") or [])
        ntb["response_mask"].append(mapping.get("response_mask") or [])
        ntb["turns"].append(mapping.get("turns", 1))
        ntb["used_image_credit"].append(mapping.get("used_image_credit", False))
    return ntb


def flatten_from_agent_output(output: Any, *, expected_s: int) -> FlattenResult:
    """RFC Flatten after ``generate_sequences``: UND vs GEN rows from extra_fields."""
    ntb = dict(getattr(output, "non_tensor_batch", None) or {})
    if ntb.get("gen_samples") is None and ntb.get("extra_fields") is not None:
        ntb.update(_ntb_from_extra_fields(ntb["extra_fields"]))
    gen_col = ntb.get("gen_samples")
    if gen_col is None:
        return flatten_multiturn_rollouts([], expected_s=expected_s)

    length = len(gen_col)

    def _col(name: str, default: Any) -> list[Any]:
        arr = ntb.get(name)
        if arr is None:
            return [default] * length
        values = list(arr)
        if len(values) < length:
            values.extend([default] * (length - len(values)))
        return values[:length]

    episodes: list[EpisodeRollout] = []
    und_groups = _col("und_group_uid", "missing_task")
    episode_uids = _col("episode_uid", None)
    prompt_ids = _col("prompt_ids", [])
    response_ids = _col("response_ids", [])
    response_mask = _col("response_mask", [])
    turns = _col("turns", 1)
    used = _col("used_image_credit", False)
    for index in range(length):
        raw_samples = gen_col[index] or []
        samples = []
        for item in raw_samples:
            sample = _as_gen_sample(item)
            if sample is not None:
                samples.append(sample)
        episodes.append(
            EpisodeRollout(
                und_group_uid=str(und_groups[index]),
                episode_uid=str(episode_uids[index] or f"ep{index}"),
                policy_version=0,
                prompt_ids=list(prompt_ids[index] or []),
                response_ids=list(response_ids[index] or []),
                response_mask=list(response_mask[index] or []),
                turns=int(turns[index] or 1),
                gen_samples=samples,
                used_image_credit=bool(used[index]),
                num_gen_calls=len({str(s.gen_group_uid) for s in samples if s.valid}),
            )
        )
    return flatten_multiturn_rollouts(episodes, expected_s=expected_s)


def strip_pixels_for_actor(row: dict[str, Any]) -> dict[str, Any]:
    """Bagel GEN trains on ``prompt_token_ids``, not pixels or ``prompt_embeds``."""
    cleaned = dict(row)
    for key in ("images", "pixel_values", "prompt_embeds", "negative_prompt_embeds"):
        cleaned.pop(key, None)
    return cleaned

