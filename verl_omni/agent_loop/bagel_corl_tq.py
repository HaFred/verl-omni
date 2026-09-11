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
"""Dual-lane TransferQueue packing for Bagel Co-RL (UND episode key vs GEN seed keys).

In-episode counters (RFC):
  J = UND policy turns; K = generate_image calls; J >= K.
  S = seeds per call (FlowGRPO group size).
  N = sibling episodes (rollout.n) — not packed here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import ray
import torch
import transfer_queue as tq
from verl.experimental.agent_loop import AgentLoopManager, AgentLoopOutput
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from verl_omni.agent_loop.bagel_corl_lib import (
    EpisodeRollout,
    GenSample,
    strip_pixels_for_actor,
    _as_gen_sample,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Ray forbids subclassing an already-@ray.remote actor class. Inherit the
# underlying AgentLoopWorkerTQ implementation, then re-wrap with @ray.remote.
_AgentLoopWorkerTQBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class

__all__ = [
    "BagelCorlAgentLoopManagerTQ",
    "BagelCorlAgentLoopWorkerTQ",
    "DualLanePack",
    "classify_episode_pattern",
    "episode_from_agent_extra",
    "gate_episode_jk",
    "gen_tq_key",
    "pack_dual_lane_episode",
    "split_und_gen_from_und_records",
    "split_und_gen_metas",
    "und_tq_key",
]


def und_tq_key(*, dataset_task_uid: str, session_id: int | str, episode_index: int | str) -> str:
    return f"{dataset_task_uid}_{session_id}_{episode_index}"


def gen_tq_key(und_key: str, *, gen_call_id: str, seed_index: int) -> str:
    return f"{und_key}::gen::{gen_call_id}::{seed_index}"


def gate_episode_jk(j: int, k: int) -> None:
    if j < 0 or k < 0:
        raise ValueError(f"J and K must be non-negative, got J={j} K={k}")
    if j < k:
        raise ValueError(f"in-episode invariant J >= K violated: J={j} K={k}")


def classify_episode_pattern(j: int, k: int) -> str:
    """Return paired | mixed | gen_off. Raises if J < K."""
    gate_episode_jk(j, k)
    if k == 0:
        return "gen_off"
    if j == k:
        return "paired"
    return "mixed"


@dataclass
class DualLanePack:
    """One episode → one UND record + zero-or-more GEN seed records."""

    und_key: str
    und_record: dict[str, Any]
    gen_records: list[dict[str, Any]] = field(default_factory=list)
    child_gen_keys: list[str] = field(default_factory=list)
    j: int = 0
    k: int = 0
    pattern: str = "gen_off"
    metrics: dict[str, float] = field(default_factory=dict)


def _count_gen_calls(samples: list[GenSample]) -> int:
    groups: set[str] = set()
    for sample in samples:
        if sample.valid:
            groups.add(str(sample.gen_group_uid))
    return len(groups)


def _sample_has_gen_traj(sample: GenSample) -> bool:
    """Complete FlowGRPO seed requires latents + timesteps + rollout logprobs."""
    return (
        sample.valid
        and sample.all_latents is not None
        and sample.timesteps is not None
        and sample.rollout_log_probs is not None
    )


def _complete_groups(samples: list[GenSample], *, expected_s: int) -> dict[str, list[GenSample]]:
    by_group: dict[str, list[GenSample]] = {}
    for sample in samples:
        if not _sample_has_gen_traj(sample):
            continue
        by_group.setdefault(str(sample.gen_group_uid), []).append(sample)
    complete: dict[str, list[GenSample]] = {}
    for gid, rows in by_group.items():
        if len(rows) == expected_s:
            complete[gid] = sorted(rows, key=lambda s: int(s.seed_index))
    return complete


def pack_dual_lane_episode(
    episode: EpisodeRollout,
    *,
    und_key: str,
    expected_s: int,
    dataset_task_uid: str | None = None,
) -> DualLanePack:
    """Pack UND + GEN TQ records. Pattern 3 (K=0) yields zero GEN keys."""
    if expected_s < 1:
        raise ValueError("expected_s must be >= 1")

    j = int(episode.turns)
    if getattr(episode, "num_gen_calls", None) is not None:
        k = int(episode.num_gen_calls)
    else:
        k = _count_gen_calls(episode.gen_samples)
    pattern = classify_episode_pattern(j, k)

    complete = _complete_groups(episode.gen_samples, expected_s=expected_s)
    dropped = max(0, k - len(complete)) if k > 0 else 0
    k_complete = len(complete)

    child_keys: list[str] = []
    gen_records: list[dict[str, Any]] = []
    for gen_call_id, rows in complete.items():
        for sample in rows:
            key = gen_tq_key(und_key, gen_call_id=gen_call_id, seed_index=int(sample.seed_index))
            child_keys.append(key)
            row = strip_pixels_for_actor(
                {
                    "bagel_role": "gen",
                    "episode_uid": episode.episode_uid,
                    "und_group_uid": episode.und_group_uid,
                    "gen_group_uid": sample.gen_group_uid,
                    "gen_sample_uid": sample.gen_sample_uid,
                    "seed_index": sample.seed_index,
                    "parent_und_key": und_key,
                    "uid": sample.gen_group_uid,
                    "prompt_token_ids": list(sample.prompt_token_ids),
                    "all_latents": sample.all_latents,
                    "timesteps": sample.timesteps,
                    "rollout_log_probs": sample.rollout_log_probs,
                    "rm_score": sample.rm_score,
                    "call_role": sample.call_role,
                    "good_enough": sample.good_enough,
                    "image_path": sample.image_path,
                }
            )
            gen_records.append({"key": key, "fields": row, "tag": {"bagel_role": "gen", "status": "success"}})

    task_uid = dataset_task_uid or episode.und_group_uid
    und_fields = {
        "bagel_role": "und",
        "episode_uid": episode.episode_uid,
        "und_group_uid": episode.und_group_uid,
        "dataset_task_uid": task_uid,
        "uid": episode.und_group_uid,
        "parent_und_key": und_key,
        "prompt_ids": list(episode.prompt_ids),
        "response_ids": list(episode.response_ids),
        "response_mask": list(episode.response_mask),
        "child_gen_keys": list(child_keys),
        "episode_J": j,
        "episode_K": k,
        "episode_pattern": pattern,
        "used_image_credit": bool(episode.used_image_credit),
        "token_level_scores": float(episode.und_reward),
        "num_gen_calls": k,
        "policy_version": int(episode.policy_version),
    }
    metrics = {
        "episode/J": float(j),
        "episode/K": float(k),
        "episode/K_complete": float(k_complete),
        "gen/num_rows": float(len(gen_records)),
        "gen/dropped_incomplete_groups": float(dropped),
        "und/no_image_credit": 1.0 if k_complete == 0 else 0.0,
        "gen/skipped_no_groups": 1.0 if k_complete == 0 else 0.0,
        "episode/pattern_paired": 1.0 if pattern == "paired" else 0.0,
        "episode/pattern_mixed": 1.0 if pattern == "mixed" else 0.0,
        "episode/pattern_gen_off": 1.0 if pattern == "gen_off" else 0.0,
    }
    return DualLanePack(
        und_key=und_key,
        und_record={"key": und_key, "fields": und_fields, "tag": {"bagel_role": "und", "status": "success"}},
        gen_records=gen_records,
        child_gen_keys=child_keys,
        j=j,
        k=k,
        pattern=pattern,
        metrics=metrics,
    )


def split_und_gen_from_und_records(
    und_records: list[Mapping[str, Any]],
    gen_by_key: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """P1 gather: from UND records (+ optional GEN key map) produce und_batch / gen_batch field lists."""
    und_batch: list[dict[str, Any]] = []
    gen_batch: list[dict[str, Any]] = []
    gen_by_key = gen_by_key or {}
    for rec in und_records:
        fields = dict(rec.get("fields") or rec)
        und_batch.append(fields)
        for key in list(fields.get("child_gen_keys") or []):
            gen_rec = gen_by_key.get(key)
            if gen_rec is None:
                continue
            gen_fields = dict(gen_rec.get("fields") or gen_rec)
            gen_batch.append(gen_fields)
    return und_batch, gen_batch


# Plan name alias.
split_und_gen_metas = split_und_gen_from_und_records


def episode_from_agent_extra(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    extra: Mapping[str, Any],
    und_group_uid: str,
    reward_score: float | None = None,
) -> EpisodeRollout:
    """Rebuild ``EpisodeRollout`` from ``AgentLoopOutput`` fields / extra_fields."""
    raw_samples = extra.get("gen_samples") or []
    samples: list[GenSample] = []
    for item in raw_samples:
        sample = _as_gen_sample(item)
        if sample is not None:
            samples.append(sample)
    turns = int(extra.get("turns") or extra.get("num_turns") or 1)
    num_gen = extra.get("num_gen_calls")
    if num_gen is None:
        num_gen = _count_gen_calls(samples)
    return EpisodeRollout(
        und_group_uid=str(extra.get("und_group_uid") or und_group_uid),
        episode_uid=str(extra.get("episode_uid") or f"{und_group_uid}-ep"),
        policy_version=int(extra.get("policy_version") or 0),
        prompt_ids=list(prompt_ids),
        response_ids=list(response_ids),
        response_mask=list(response_mask),
        turns=turns,
        gen_samples=samples,
        used_image_credit=bool(extra.get("used_image_credit", False)),
        und_reward=float(reward_score if reward_score is not None else extra.get("und_reward") or 0.0),
        num_gen_calls=int(num_gen),
    )


@ray.remote
class BagelCorlAgentLoopWorkerTQ(_AgentLoopWorkerTQBase):
    """TransferQueue worker that writes dual-lane UND + GEN keys for Bagel Co-RL."""

    def _expected_s(self) -> int:
        agent = self.config.actor_rollout_ref.rollout.agent
        return int(getattr(agent, "gen_samples_per_call", None) or agent.get("gen_samples_per_call") or 4)

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put one UND episode key plus zero-or-more GEN seed keys into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            raise RuntimeError(f"Empty bagel Co-RL agent output for prompt {uid}_{session_id}")

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for out in outputs[:-1]:
                out.reward_score = final_output.reward_score
                out.extra_fields["reward_extra_info"] = final_output.extra_fields.get("reward_extra_info")

        # Bagel Co-RL: one serial episode per session → single UND index 0.
        if len(outputs) != 1:
            raise RuntimeError(
                f"Bagel Co-RL expected one AgentLoopOutput per session, got {len(outputs)}"
            )
        episode_output = outputs[0]
        und_key = und_tq_key(dataset_task_uid=uid, session_id=session_id, episode_index=0)
        episode = episode_from_agent_extra(
            prompt_ids=list(episode_output.prompt_ids),
            response_ids=list(episode_output.response_ids),
            response_mask=list(episode_output.response_mask),
            extra=episode_output.extra_fields or {},
            und_group_uid=str(uid),
            reward_score=episode_output.reward_score,
        )
        pack = pack_dual_lane_episode(
            episode,
            und_key=und_key,
            expected_s=self._expected_s(),
            dataset_task_uid=str(uid),
        )
        logger.info(
            "bagel_corl dual-lane pack und=%s pattern=%s J=%s K=%s gen_keys=%s metrics=%s",
            und_key,
            pack.pattern,
            pack.j,
            pack.k,
            len(pack.child_gen_keys),
            pack.metrics,
        )

        # UND token row: reuse AgentLoopOutput.as_dict for TQ training tensors.
        und_field = episode_output.as_dict()
        und_field.update({k: v for k, v in kwargs.items() if k not in und_field})
        und_field.pop("multi_modal_data", None)
        und_field["loss_mask"] = und_field["response_mask"]
        prompts = und_field["prompts"]
        responses = und_field["responses"]
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        multi_modal_inputs = self._compute_multi_modal_inputs(episode_output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
        ).squeeze(0)
        und_field["input_ids"] = input_ids
        und_field["position_ids"] = position_ids
        und_field["multi_modal_inputs"] = multi_modal_inputs
        # Dual-lane identity on UND row (no nested gen_samples as train source of truth).
        und_field["uid"] = pack.und_record["fields"]["uid"]
        und_extra = dict(und_field.get("extra_fields") or {})
        und_extra.pop("gen_samples", None)
        und_extra.update(
            {
                "bagel_role": "und",
                "child_gen_keys": list(pack.child_gen_keys),
                "episode_J": pack.j,
                "episode_K": pack.k,
                "episode_pattern": pack.pattern,
                "und_group_uid": pack.und_record["fields"]["und_group_uid"],
                "episode_uid": pack.und_record["fields"]["episode_uid"],
                "bagel_corl_metrics": dict(pack.metrics),
                "min_global_steps": und_extra.get("min_global_steps", kwargs.get("global_steps")),
                "max_global_steps": und_extra.get("max_global_steps", kwargs.get("global_steps")),
            }
        )
        und_field["extra_fields"] = und_extra
        und_field["child_gen_keys"] = list(pack.child_gen_keys)
        und_field["bagel_role"] = "und"

        keys = [und_key]
        fields = [und_field]
        tags = [
            {
                "status": "success",
                "bagel_role": "und",
                "prompt_len": int(prompts.size(0)),
                "response_len": int(responses.size(0)),
                "seq_len": int(prompts.size(0) + responses.size(0)),
                "global_steps": kwargs["global_steps"],
                "min_global_steps": und_extra.get("min_global_steps"),
                "max_global_steps": und_extra.get("max_global_steps"),
                "episode_J": pack.j,
                "episode_K": pack.k,
                "episode_pattern": pack.pattern,
            }
        ]

        for gen_rec in pack.gen_records:
            gen_fields = dict(gen_rec["fields"])
            gen_fields["extra_fields"] = {
                "bagel_role": "gen",
                "parent_und_key": und_key,
                "min_global_steps": kwargs.get("global_steps"),
                "max_global_steps": kwargs.get("global_steps"),
            }
            # RM score as a 1-length tensor for TQ consumers that expect rm_scores.
            if gen_fields.get("rm_score") is not None:
                gen_fields["rm_scores"] = torch.tensor([float(gen_fields["rm_score"])], dtype=torch.float32)
            keys.append(gen_rec["key"])
            fields.append(gen_fields)
            tags.append(
                {
                    "status": "success",
                    "bagel_role": "gen",
                    "parent_und_key": und_key,
                    "global_steps": kwargs["global_steps"],
                    "min_global_steps": kwargs.get("global_steps"),
                    "max_global_steps": kwargs.get("global_steps"),
                    "prompt_len": 0,
                    "response_len": 0,
                    "seq_len": 0,
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class BagelCorlAgentLoopManagerTQ(AgentLoopManager):
    """AgentLoopManager wired to ``BagelCorlAgentLoopWorkerTQ`` (dual-lane ingest)."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = BagelCorlAgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts):
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
