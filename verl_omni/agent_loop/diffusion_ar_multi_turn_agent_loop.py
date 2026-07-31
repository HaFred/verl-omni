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

"""Multi-turn agentic agent loop with prompt rewriting for Mode (2a) RL."""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.agent_output_parser import AGENT_SYSTEM_PROMPT, parse_agent_output
from verl_omni.agent_loop.agentic_trajectory import (
    AgenticMetadata,
    AgenticTrajectory,
    AgenticTurn,
    ToolCall,
    ToolOutput,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _user_text_from_raw_prompt(raw_prompt: Any) -> str:
    """Extract the seed user request from RLHFDataset ``raw_prompt``.

    Online agentic rollout stores a string prompt on ``AgenticTrajectory``.
    Datasets usually provide chat messages
    ``[{"role": "user", "content": "..."}]``; accept either form.
    """
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, list):
        for message in reversed(raw_prompt):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                ]
                joined = " ".join(text for text in texts if text).strip()
                if joined:
                    return joined
        return ""
    return "" if raw_prompt is None else str(raw_prompt)


@register("diffusion_ar_multi_turn_agent")
class DiffusionARMultiTurnAgentLoop(AgentLoopBase):
    """Agent loop for multi-turn agentic RL (Mode 2a).

    Executes: agent reasons -> rewrites prompt -> frozen tool generates image ->
    agent reflects -> continue/stop -> repeat.
    Each turn's rewritten prompt is captured explicitly in AgenticTrajectory.

    Returns ``AgentLoopOutput`` so stock ``AgentLoopWorker`` postprocess can pad
    ``response_ids`` / ``response_mask``. Agent tokens are mask=1; tool image
    observations are not tokenized into the response (und-only / stub-image path).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _decode_agent_text(self, text_output: Any) -> tuple[str, list[int]]:
        """Decode agent text from AR ``TokenOutput`` or diffusion string payload."""
        token_ids = getattr(text_output, "token_ids", None)
        if token_ids is not None:
            ids = list(token_ids)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            return text, ids

        payload = getattr(text_output, "diffusion_output", None)
        if isinstance(payload, str):
            return payload, []
        if payload is not None and not isinstance(payload, torch.Tensor | bytes):
            return str(payload), []
        return "", []

    def _extract_tool_image(self, img_output: Any) -> Any:
        """Return a tool image tensor; synthesize a stub when AR-only rollout has none."""
        diffusion_output = getattr(img_output, "diffusion_output", None)
        if isinstance(diffusion_output, torch.Tensor):
            return diffusion_output
        logger.warning("No diffusion_output from tool generate; using stub image tensor")
        return torch.zeros(3, 64, 64)

    def _normalize_logprobs(self, logprobs: Any, n_tokens: int) -> list[float]:
        if logprobs is None:
            return [0.0] * n_tokens
        if hasattr(logprobs, "tolist"):
            logprobs = logprobs.tolist()
        values = list(logprobs)
        if len(values) < n_tokens:
            values = values + [0.0] * (n_tokens - len(values))
        return values[:n_tokens]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = _user_text_from_raw_prompt(kwargs["raw_prompt"])
        max_turns = sampling_params.get("max_turns", 5)

        vision_source = kwargs["raw_prompt"]
        multi_modal_data = await self.process_vision_info(vision_source)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        chat_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": raw_prompt},
        ]

        turns: list[AgenticTurn] = []
        metrics: dict[str, Any] = {}
        decision = "stop"

        initial_prompt_ids = await self.apply_chat_template(chat_messages, images=images, videos=videos)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        for turn_idx in range(max_turns):
            prompt_ids = await self.apply_chat_template(chat_messages, images=images, videos=videos)

            with simple_timer(f"agent_text_gen_turn_{turn_idx}", metrics):
                gen_params = {**sampling_params, "logprobs": True}
                text_output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=gen_params,
                    image_data=images,
                    video_data=videos,
                )

            agent_text, agent_token_ids = self._decode_agent_text(text_output)
            if not agent_token_ids and agent_text:
                agent_token_ids = self.tokenizer.encode(agent_text, add_special_tokens=False)

            agent_logprobs = self._normalize_logprobs(getattr(text_output, "log_probs", None), len(agent_token_ids))

            response_ids.extend(agent_token_ids)
            response_mask.extend([1] * len(agent_token_ids))
            response_logprobs.extend(agent_logprobs)

            parsed = parse_agent_output(agent_text)
            decision = parsed["decision"]
            rewritten_prompt = parsed["prompt"] or raw_prompt

            if decision == "stop":
                turns.append(
                    AgenticTurn(
                        turn_idx=turn_idx,
                        agent_tokens=agent_token_ids,
                        agent_logprobs=agent_logprobs,
                        agent_text=agent_text,
                        tool_call=None,
                        tool_output=None,
                        decision="stop",
                    )
                )
                break

            with simple_timer(f"tool_call_turn_{turn_idx}", metrics):
                tool_params = {**sampling_params, "logprobs": False, "prompt": rewritten_prompt}
                img_output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids + agent_token_ids,
                    sampling_params=tool_params,
                    image_data=images,
                    video_data=videos,
                )

            diffusion_output = self._extract_tool_image(img_output)

            turns.append(
                AgenticTurn(
                    turn_idx=turn_idx,
                    agent_tokens=agent_token_ids,
                    agent_logprobs=agent_logprobs,
                    agent_text=agent_text,
                    tool_call=ToolCall(tool_name="t2i", params={"prompt": rewritten_prompt}),
                    tool_output=ToolOutput(output_type="image", output_data=diffusion_output),
                    decision="continue",
                )
            )

            chat_messages.append({"role": "assistant", "content": agent_text})
            chat_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": diffusion_output},
                        {
                            "type": "text",
                            "text": "Here is the generated image. Review it and decide whether to continue.",
                        },
                    ],
                }
            )
            images = [diffusion_output]

            if len(response_mask) >= self.response_length:
                break

        meta = AgenticMetadata(
            num_turns=len(turns),
            terminated=decision == "stop",
            termination_reason="agent_stop" if decision == "stop" else "max_turns",
        )
        trajectory = AgenticTrajectory(prompt=raw_prompt, turns=turns, metadata=meta)

        from verl_omni.agent_loop.agentic_trajectory import agentic_trajectory_to_dict

        # JSON-serializable trajectory for reward expand + rollout dumps.
        traj_dict = agentic_trajectory_to_dict(trajectory)

        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            num_turns=len(turns) + 1,
            metrics=metrics,
            extra_fields={
                "agentic_trajectory": traj_dict,
            },
        )
