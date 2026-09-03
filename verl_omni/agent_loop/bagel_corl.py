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

import logging
import uuid
from typing import Any

import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, register
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.bagel_corl_lib import (  # noqa: F401
    BagelGenerateImageTool,
    GenSample,
    run_serial_episode,
    turn_histogram,
)
from verl_omni.agent_loop.composite_agent_loop import CompositeAgentLoopOutput, CompositeAgentLoopWorker

logger = logging.getLogger(__name__)


@register("bagel_multiturn_agent")
class BagelMultiturnAgentLoop(AgentLoopBase):
    """Per-episode serial UND→GEN→RM loop. Outer gather stays on the worker."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> CompositeAgentLoopOutput:
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
        episode = await run_serial_episode(
            dataset_task_uid=dataset_task_uid,
            policy_version=policy_version,
            prompt_ids=list(prompt_ids),
            und_decode=_und_decode,
            generate_tool=tool,
            score_fn=self._score_gen_samples,
            max_und_turns=max_und_turns,
        )
        extra = {
            "text_encoder_responses": "",
            "response_ids": episode.response_ids,
            "response_mask": episode.response_mask,
            "gen_samples": episode.gen_samples,
            "und_group_uid": episode.und_group_uid,
            "turns": episode.turns,
            "llm_all_log_probs": None,
        }
        return CompositeAgentLoopOutput(
            prompt_ids=episode.prompt_ids,
            response_diffusion_output=torch.zeros(3, 8, 8),
            response_logprobs=None,
            num_turns=episode.turns,
            metrics=AgentLoopMetrics(),
            extra_fields=extra,
        )

    async def _generate_image_k(self, **kwargs) -> list[dict[str, Any]]:
        """GEN replica: K seeds, stash latents / logprobs / prompt_token_ids (not prompt_embeds)."""
        raise NotImplementedError(
            "BagelGenerateImageTool.generate_fn must be wired to the GEN vLLM-Omni replica in serving"
        )

    async def _score_gen_samples(self, samples: list[GenSample]) -> list[GenSample]:
        """Score via OmniRewardLoopManager handles (C/A, UniCoT similarity, good_enough)."""
        return samples


class MultiturnAgentLoopWorker(CompositeAgentLoopWorker):
    """Composite worker that gathers J serial episodes, then flattens UND vs GEN."""

    async def generate_sequences(self, batch):
        output = await super().generate_sequences(batch)
        turns = []
        if "metrics" in output.meta_info:
            for row in output.meta_info["metrics"]:
                if isinstance(row, dict) and "turns" in row:
                    turns.append(int(row["turns"]))
        hist = turn_histogram(turns)
        output.meta_info.setdefault("bagel_corl", {}).update(hist)
        logger.info("bagel_corl turn histogram: %s", hist)
        return output
