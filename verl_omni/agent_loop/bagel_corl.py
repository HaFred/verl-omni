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

"""Serial Bagel UND→GEN Co-RL agent loop (RFC phase A / PR1)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.bagel_corl_lib import (  # noqa: F401
    BagelGenerateImageTool,
    GenSample,
    run_serial_episode,
    turn_histogram,
)
from verl_omni.agent_loop.composite_agent_loop import CompositeAgentLoopWorker
from verl_omni.agent_loop.rpco_turn_protocol import derive_good_enough_from_scores

logger = logging.getLogger(__name__)


@register("bagel_multiturn_agent")
class BagelMultiturnAgentLoop(AgentLoopBase):
    """Per-episode serial UND→GEN→RM loop. Outer gather stays on the worker."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Stash for the GEN tool (``_generate_image_k``) so it can build diffusion
        # sampling params (seed / logprobs / num_inference_steps) per seed.
        self._sampling_params = sampling_params
        dataset_task_uid = str(
            kwargs.get("dataset_task_uid")
            or kwargs.get("uid")
            or (kwargs.get("extra_info") or {}).get("dataset_task_uid")
            or "missing_task"
        )
        policy_version = int(sampling_params.get("global_steps", kwargs.get("policy_version", 0)) or 0)
        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("bagel_multiturn_agent requires raw_prompt")
        prompt_ids = kwargs.get("prompt_ids")
        if prompt_ids is None:
            encoded = self.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=True)
            prompt_ids = list(encoded)

        agent_cfg = self.config.actor_rollout_ref.rollout.agent
        k = int(agent_cfg.get("gen_samples_per_call", 4))
        max_passes = int(agent_cfg.get("max_generate_passes", 1))
        max_und_turns = int(agent_cfg.get("max_und_turns", 8))

        async def _und_decode(**_decode_kwargs):
            with simple_timer("und_decode", {}):
                output = await self.server_manager.generate(
                    request_id=str(uuid.uuid4()),
                    prompt_ids=list(_decode_kwargs["prompt_ids"]) + list(_decode_kwargs["response_ids"]),
                    sampling_params=sampling_params,
                )
            token_ids = list(output.token_ids if hasattr(output, "token_ids") else output.get("token_ids", []))
            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            return {"token_ids": token_ids, "text": text}

        tool = BagelGenerateImageTool(
            gen_samples_per_call=k,
            max_generate_passes=max_passes,
            generate_fn=self._generate_image_k,
        )
        # Non-image UND scalar for pattern-3 (K=0) episodes. The reward model computes
        # the authoritative token-GRPO reward post-hoc (via the worker's ``_compute_score``);
        # this is only a fallback used when no RM is attached.
        non_image_reward = kwargs.get("non_image_reward")
        episode = await run_serial_episode(
            dataset_task_uid=dataset_task_uid,
            policy_version=policy_version,
            prompt_ids=list(prompt_ids),
            und_decode=_und_decode,
            generate_tool=tool,
            score_fn=self._score_gen_samples,
            tokenizer=self.tokenizer,
            max_und_turns=max_und_turns,
            non_image_reward=float(non_image_reward) if non_image_reward is not None else None,
        )
        extra = {
            "text_encoder_responses": "",
            "prompt_ids": episode.prompt_ids,
            "response_ids": episode.response_ids,
            "response_mask": episode.response_mask,
            "gen_samples": episode.gen_samples,
            "und_group_uid": episode.und_group_uid,
            "episode_uid": episode.episode_uid,
            "used_image_credit": episode.used_image_credit,
            "turns": episode.turns,
            "num_gen_calls": episode.num_gen_calls,
            "und_reward": episode.und_reward,
            "policy_version": episode.policy_version,
            "llm_all_log_probs": None,
        }
        # V1 TQ path expects AgentLoopOutput (token trajectory). Dual-lane packing
        # in BagelCorlAgentLoopWorkerTQ splits GEN seeds onto separate TQ keys.
        # reward_score must stay None so the worker's reward loop computes the real
        # reward (image-grounded when K>=1, text-only when K=0); pre-setting 0.0 here
        # would block the RM and zero out token-GRPO signal for pattern-3 episodes.
        return AgentLoopOutput(
            prompt_ids=episode.prompt_ids,
            response_ids=episode.response_ids,
            response_mask=episode.response_mask,
            response_logprobs=None,
            num_turns=episode.turns,
            metrics=AgentLoopMetrics(),
            reward_score=None,
            extra_fields=extra,
        )

    async def _generate_image_k(self, **kwargs) -> list[dict[str, Any]]:
        """Call the GEN vLLM-Omni replica ``S`` times (one per seed) and stash trajectories.

        Each returned row carries ``valid`` / ``all_latents`` / ``timesteps`` /
        ``rollout_log_probs`` / ``image_path``. Fails closed when the server returns a
        token (not diffusion) output, i.e. dual-role serving is not yet active — do not
        fall back to a Qwen image sidecar (RFC stop-gate).

        Incomplete / invalid S-groups are dropped at dual-lane ingest (no dummy pads).
        Pattern 3 (K=0) never calls this.
        """
        prompt = str(kwargs.get("prompt", ""))
        seeds = list(kwargs.get("seeds") or [])
        if not prompt:
            raise ValueError("_generate_image_k requires a non-empty diffusion prompt")

        sampling = dict(getattr(self, "_sampling_params", None) or {})
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            request_params = dict(sampling)
            request_params["seed"] = int(seed)
            request_params["logprobs"] = True
            # BagelPipeline reads text from the request prompt; encode the diffusion
            # prompt directly (not the whole UND conversation) so the GEN replica
            # denoises the right conditioning.
            gen_prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
            output = await self.server_manager.generate(
                request_id=str(uuid.uuid4()),
                prompt_ids=gen_prompt_ids,
                sampling_params=request_params,
            )
            if not self._is_diffusion_output(output):
                raise RuntimeError(
                    "Bagel Co-RL generate_image returned a token output: dual-role GEN "
                    "serving is not wired (bagel_single_stage is GEN-only / the AR replica "
                    "is not diffusion-capable). Do not fall back to a Qwen sidecar."
                )
            extra = dict(getattr(output, "extra_fields", None) or {})
            rows.append(
                {
                    "valid": getattr(output, "stop_reason", None) not in ("aborted", "abort", "error"),
                    "all_latents": extra.get("all_latents"),
                    "timesteps": extra.get("all_timesteps"),
                    "rollout_log_probs": getattr(output, "log_probs", None),
                    "image_path": extra.get("image_path") or extra.get("path"),
                }
            )
        return rows

    @staticmethod
    def _is_diffusion_output(output: Any) -> bool:
        """verl_omni ``DiffusionOutput`` carries ``diffusion_output``; ``TokenOutput`` does not."""
        return hasattr(output, "diffusion_output")

    async def _score_gen_samples(self, samples: list[GenSample]) -> list[GenSample]:
        """Score GEN samples via an injected RM hook (C/A, UniCoT similarity, good_enough).

        When no hook is wired (no mid-episode RM), derive ``good_enough`` from any
        pre-populated ``rm_score`` and otherwise leave the samples unscored so the
        post-episode reward loop supplies the authoritative scalar. Never hardcode a
        zero score.
        """
        rm_fn = getattr(self, "_rm_score_fn", None)
        if rm_fn is None:
            for sample in samples:
                if sample.rm_score is not None and sample.good_enough is None:
                    sample.good_enough = derive_good_enough_from_scores(
                        correctness=float(sample.rm_score),
                        aesthetics=float(sample.rm_score),
                    )
            return samples
        result = rm_fn(samples)
        if asyncio.iscoroutine(result):
            result = await result
        return result


class MultiturnAgentLoopWorker(CompositeAgentLoopWorker):
    """Composite worker that gathers J serial episodes, then flattens UND vs GEN.

    Prefer ``BagelCorlAgentLoopWorkerTQ`` for bagel_corl_sync (dual-lane TQ).
    """

    async def generate_sequences(self, batch):
        output = await super().generate_sequences(batch)
        turns = []
        if "metrics" in output.meta_info:
            for row in output.meta_info["metrics"]:
                if isinstance(row, dict) and "turns" in row:
                    turns.append(int(row["turns"]))
        hist = turn_histogram(turns)
        extra = output.meta_info.setdefault("bagel_corl", {})
        extra.update(hist)
        logger.info("bagel_corl turn histogram: %s", hist)
        agent_cfg = getattr(self.config.actor_rollout_ref.rollout, "agent", None)
        expected_k = int(getattr(agent_cfg, "gen_samples_per_call", None) or 4)
        try:
            from verl_omni.agent_loop.bagel_corl_lib import flatten_from_agent_output, strip_pixels_for_actor

            flat = flatten_from_agent_output(output, expected_k=expected_k)
            extra.update(flat.metrics)
            extra["gen_batch"] = [strip_pixels_for_actor(row) for row in flat.gen_batch]
            extra["und_batch"] = flat.und_batch
            extra["gen_episode_map"] = flat.gen_episode_map
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            logger.warning("bagel_corl flatten skipped: %s", exc)
        try:
            from verl_omni.utils.agentic.image_gen_rollout_dump import (
                dump_bagel_corl_episode_images,
                dump_raw_rollouts,
            )

            step = None
            if hasattr(batch, "meta_info") and isinstance(batch.meta_info, dict):
                step = batch.meta_info.get("global_steps") or batch.meta_info.get("step")
            if step is None and isinstance(getattr(output, "meta_info", None), dict):
                step = output.meta_info.get("global_steps") or output.meta_info.get("step")
            try:
                dump_raw_rollouts(
                    tokenizer=getattr(self, "tokenizer", None),
                    output=output,
                    step=step,
                )
            except (TypeError, KeyError, AttributeError) as exc:
                logger.debug("dump_raw_rollouts skipped for Bagel Co-RL batch shape: %s", exc)
                dump_bagel_corl_episode_images(output, step=step)
        except ImportError as exc:
            logger.debug("rollout dump unavailable: %s", exc)
        return output
