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
    """One of K GEN seeds from a single ``generate_image`` call."""

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


@dataclass
class EpisodeRollout:
    """One serial UND episode (zero or one GEN call in PR1)."""

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


class BagelGenerateImageTool:
    """K-seed GEN tool. Refuses a second call when ``max_generate_passes=1``."""

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
        self.k = int(gen_samples_per_call)
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
            seeds = list(range(self.k))
        if len(seeds) != self.k:
            raise ValueError(f"expected K={self.k} seeds, got {len(seeds)}")
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
    """UND-facing observation: path only; K trajectories stay on the GEN batch."""
    return f"path={path}"


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
    max_und_turns: int = 8,
    forced_reflection_text: str = "Done.",
    episode_uid: str | None = None,
) -> EpisodeRollout:
    """Serial UND decode → optional GEN (K seeds) → RM → forced reflection / Done."""
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
        if kind == "generate_image":
            call = parse_hermes_tool_call(text) or {}
            arguments = call.get("arguments", call.get("parameters", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            prompt = str(arguments.get("prompt", ""))
            gen_samples = await generate_tool(
                prompt=prompt,
                prompt_token_ids=list(prompt_ids) + list(response_ids),
                gen_call_id=str(ids["gen_call_id"]),
            )
            if score_fn is not None:
                scored = score_fn(gen_samples)
                if asyncio.iscoroutine(scored):
                    scored = await scored
                gen_samples = scored
            valid_paths = [s.image_path for s in gen_samples if s.valid and s.image_path]
            if valid_paths:
                used_image_credit = True
                obs = compact_image_observation(valid_paths[0])
                obs_ids: list[int] = list(step.get("obs_token_ids", []))
                if not obs_ids:
                    obs_ids = [0]
                _ = obs
                response_ids.extend(obs_ids)
                response_mask.extend([0] * len(obs_ids))
            forced_ids: list[int] = list(step.get("forced_token_ids", []))
            if forced_ids:
                forced = True
                response_ids.extend(forced_ids)
                response_mask.extend([0] * len(forced_ids))
            else:
                done_ids: list[int] = list(step.get("done_token_ids", []))
                if done_ids:
                    response_ids.extend(done_ids)
                    response_mask.extend([1] * len(done_ids))
            break
    else:
        forced = True
        _ = forced_reflection_text

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
    expected_k: int,
) -> FlattenResult:
    """Split token UND rows from latent GEN rows. Never concatenate the two.

    Incomplete K-groups are dropped. Reflection-only episodes contribute zero GEN rows.
    """
    if expected_k < 1:
        raise ValueError("expected_k must be >= 1")
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
            und_reward = 0.0
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
        if len(valid) != expected_k:
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


def strip_pixels_for_actor(row: dict[str, Any]) -> dict[str, Any]:
    """Bagel GEN trains on ``prompt_token_ids``, not pixels or ``prompt_embeds``."""
    cleaned = dict(row)
    for key in ("images", "pixel_values", "prompt_embeds", "negative_prompt_embeds"):
        cleaned.pop(key, None)
    return cleaned

