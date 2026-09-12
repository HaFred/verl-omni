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
import time
import uuid
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.bagel_corl_gen_serve import build_gen_sampling_params, stash_gen_row_from_diffusion_output
from verl_omni.agent_loop.bagel_corl_lib import (  # noqa: F401
    BagelGenerateImageTool,
    GenSample,
    run_serial_episode,
    turn_histogram,
)
from verl_omni.agent_loop.bagel_corl_rm import get_bagel_rm_dit_handle, make_rm_score_fn
from verl_omni.agent_loop.composite_agent_loop import CompositeAgentLoopWorker
from verl_omni.agent_loop.rpco_turn_protocol import derive_good_enough_from_scores
from verl_omni.tools.trajectory.artifacts import get_latest_generate_prompt_for_active_rollout
from verl_omni.tools.trajectory.context import (
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)
from verl_omni.tools.trajectory.hydra_env import agentic_get, agentic_scorer_knobs_from_config
from verl_omni.tools.trajectory.paths import build_trajectory_relpath
from verl_omni.utils.agentic.image_gen_rollout_parse import last_user_prompt

logger = logging.getLogger(__name__)


@register("bagel_multiturn_agent")
class BagelMultiturnAgentLoop(AgentLoopBase):
    """Per-episode serial UND→GEN→RM loop. Outer gather stays on the worker."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Stash for the GEN tool (``_generate_image``) so it can build diffusion
        # sampling params (seed / logprobs / num_inference_steps) per FlowGRPO seed.
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
        # Transformers 4/5: tokenize=True may return BatchEncoding; list(enc) would
        # iterate keys ('input_ids', ...) and crash normalize_token_ids on the server.
        prompt_ids = kwargs.get("prompt_ids")
        if prompt_ids is None:
            encoded = self.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=True)
            prompt_ids = normalize_token_ids(encoded)
        else:
            prompt_ids = normalize_token_ids(prompt_ids)

        agent_cfg = self.config.actor_rollout_ref.rollout.agent
        # S = FlowGRPO seeds under one generate_image turn. Not episode GEN-turn count K.
        s = int(agent_cfg.get("gen_samples_per_call", 4))
        max_passes = int(agent_cfg.get("max_generate_passes", 1))
        max_und_turns = int(agent_cfg.get("max_und_turns", 8))
        # good_enough SoT: agentic_image_gen.good_enough_threshold (yaml, bound at
        # worker init). Reduction over per-seed flags defaults to best-of-S ("any").
        good_enough_threshold = float(agentic_get("good_enough_threshold"))
        good_enough_reduction = str(agent_cfg.get("good_enough_reduction", "any"))
        # Role timing (RFC §8 KPIs): per-episode wall-clock split by lane.
        timing = {"und_decode_s": 0.0, "gen_s": 0.0, "rm_s": 0.0}
        self._bagel_timing = timing
        # In-loop RM (RFC §4.2): score each generate_image call mid-episode via the
        # DiT-side reward handle bound by the worker. UniCoT reference paths ride on
        # the row's extra_info (stamped by the dataset builder / driver).
        rm_handle = get_bagel_rm_dit_handle()
        if rm_handle is not None:
            extra_info = kwargs.get("extra_info") if isinstance(kwargs.get("extra_info"), dict) else {}
            ref_paths = extra_info.get("reference_image_path")
            self._rm_reference_paths = (
                [str(p) for p in ref_paths]
                if isinstance(ref_paths, (list, tuple))
                else ([str(ref_paths)] if ref_paths else [])
            )
            self._rm_extra_info = dict(extra_info)
            self._rm_score_fn = make_rm_score_fn(
                rm_handle,
                get_reference_paths=lambda: list(self._rm_reference_paths),
                get_extra_info=lambda: dict(self._rm_extra_info),
                get_scorer_knobs=lambda: dict(agentic_scorer_knobs_from_config(self.config)),
                get_image_prompt=lambda: get_latest_generate_prompt_for_active_rollout() or "",
            )
        else:
            self._rm_score_fn = None

        async def _und_decode(**_decode_kwargs):
            t0 = time.perf_counter()
            try:
                und_params = dict(sampling_params)
                und_params["bagel_role"] = "und"
                # UND must hit the AR replica (BagelDualRoleLLMServerClient routes on role /
                # absence of diffusion keys). Never send num_inference_steps here.
                for _k in (
                    "num_inference_steps",
                    "noise_level",
                    "sde_window_size",
                    "sde_window_range",
                    "sde_type",
                    "height",
                    "width",
                ):
                    und_params.pop(_k, None)
                try:
                    output = await self.server_manager.generate(
                        request_id=str(uuid.uuid4()),
                        prompt_ids=list(_decode_kwargs["prompt_ids"]) + list(_decode_kwargs["response_ids"]),
                        sampling_params=und_params,
                    )
                except Exception as exc:  # noqa: BLE001 — map diffusion-replica errors to UND dual-role failure
                    err = str(exc)
                    if "num_inference_steps" in err or "Diffusion" in type(exc).__name__:
                        raise RuntimeError(
                            "Bagel Co-RL UND decode hit the GEN diffusion replica "
                            "(bagel_single_stage / DiffusionStrategy). UND needs AR TokenOutput "
                            "(Hermes tool-call); refuse soft-empty TQ. "
                            "Prove dual-role serving, then set agent.und_ar_serving_ready=True. "
                            f"Underlying error: {err}"
                        ) from exc
                    raise
                if hasattr(output, "diffusion_output"):
                    raise RuntimeError(
                        "Bagel Co-RL UND decode received DiffusionOutput: rollout is on DiffusionStrategy "
                        "(bagel_single_stage / output_mode≠ar). UND needs AR token generation; "
                        "refuse empty token_ids that would leave TQ with no materializable trajectories."
                    )
                if not hasattr(output, "token_ids"):
                    raise RuntimeError(
                        f"Bagel Co-RL UND decode expected TokenOutput with token_ids, got {type(output)!r}"
                    )
                token_ids = list(output.token_ids)
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                return {"token_ids": token_ids, "text": text}
            finally:
                timing["und_decode_s"] += time.perf_counter() - t0

        tool = BagelGenerateImageTool(
            gen_samples_per_call=s,
            max_generate_passes=max_passes,
            generate_fn=self._generate_image,
        )
        # Non-image UND scalar for pattern-3 (K=0) episodes. The reward model computes
        # the authoritative token-GRPO reward post-hoc (via the worker's ``_compute_score``);
        # this is only a fallback used when no RM is attached.
        non_image_reward = kwargs.get("non_image_reward")
        # Rollout-scoped artifact context: registrations inside the episode land under
        # this trajectory in the unified tools.trajectory registry (audit T1.6).
        global_steps = sampling_params.get("global_steps", kwargs.get("global_steps"))
        session_id = kwargs.get("session_id", kwargs.get("index", 0))
        relpath = build_trajectory_relpath(step=global_steps, sample_index=session_id, rollout_n=1)
        path_tokens = set_active_trajectory_relpath(relpath)
        prompt_token = set_active_user_prompt(last_user_prompt(raw_prompt))
        try:
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
                good_enough_threshold=good_enough_threshold,
                good_enough_reduction=good_enough_reduction,
            )
        finally:
            reset_active_trajectory_relpath(path_tokens)
            reset_active_user_prompt(prompt_token)
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
            "bagel_role_timing": dict(self._bagel_timing),
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

    async def _generate_image(self, **kwargs) -> list[dict[str, Any]]:
        """Run **one** GEN turn: S same-conditioning FlowGRPO seeds, then return.

        Episode pattern (RFC): each UND ``generate_image`` verdict enqueues exactly
        one GEN turn (``K += 1``). This method is that turn — not ``K`` GEN turns
        attached to one UND decode. The seed loop is ``S = gen_samples_per_call``
        (group size for FlowGRPO), e.g. ``UND → GEN(S seeds) → UND → Done``.

        Each returned row carries ``valid`` / ``all_latents`` / ``timesteps`` /
        ``rollout_log_probs`` / ``image_path``. Fails closed when traj stash is
        missing (no soft-fallback to logprobs-only / skip GEN loss) and when the
        server returns a token output — do not fall back to a Qwen image sidecar.

        Incomplete / invalid S-groups are dropped at dual-lane ingest (no dummy pads).
        Pattern 3 (``K = 0``) never calls this.
        """
        prompt = str(kwargs.get("prompt", ""))
        seeds = list(kwargs.get("seeds") or [])
        if not prompt:
            raise ValueError("_generate_image requires a non-empty diffusion prompt")

        rollout = self.config.actor_rollout_ref.rollout
        base = dict(getattr(self, "_sampling_params", None) or {})
        rows: list[dict[str, Any]] = []
        timing = getattr(self, "_bagel_timing", None)
        for seed in seeds:
            t0 = time.perf_counter()
            try:
                request_params = build_gen_sampling_params(rollout, base=base, seed=int(seed))
                request_params["bagel_role"] = "gen"
                # BagelPipeline reads text from the request prompt; encode the diffusion
                # prompt directly (not the whole UND conversation) so the GEN replica
                # denoises the right conditioning.
                gen_prompt_ids = normalize_token_ids(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )
                output = await self.server_manager.generate(
                    request_id=str(uuid.uuid4()),
                    prompt_ids=gen_prompt_ids,
                    sampling_params=request_params,
                )
                rows.append(stash_gen_row_from_diffusion_output(output, seed=int(seed)))
            finally:
                if timing is not None:
                    timing["gen_s"] += time.perf_counter() - t0
        return rows
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
        t0 = time.perf_counter()
        try:
            result = rm_fn(samples)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            timing = getattr(self, "_bagel_timing", None)
            if timing is not None:
                timing["rm_s"] += time.perf_counter() - t0


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
        expected_s = int(getattr(agent_cfg, "gen_samples_per_call", None) or 4)
        from verl_omni.agent_loop.bagel_corl_lib import flatten_from_agent_output, strip_pixels_for_actor

        flat = flatten_from_agent_output(output, expected_s=expected_s)
        extra.update(flat.metrics)
        extra["gen_batch"] = [strip_pixels_for_actor(row) for row in flat.gen_batch]
        extra["und_batch"] = flat.und_batch
        extra["gen_episode_map"] = flat.gen_episode_map
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
