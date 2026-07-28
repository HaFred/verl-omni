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

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.utils.profiler import simple_timer

from verl_omni.agent_loop.agent_output_parser import AGENT_SYSTEM_PROMPT, parse_agent_output
from verl_omni.agent_loop.agentic_trajectory import (
    AgenticMetadata,
    AgenticTrajectory,
    AgenticTurn,
    ToolCall,
    ToolOutput,
)
from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput

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
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
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
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> DiffusionAgentLoopOutput:
        raw_prompt = _user_text_from_raw_prompt(kwargs["raw_prompt"])
        max_turns = sampling_params.get("max_turns", 5)

        # Vision extraction still sees the original dataset messages when present.
        vision_source = kwargs["raw_prompt"]
        multi_modal_data = await self.process_vision_info(vision_source)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # Build initial chat with system prompt
        chat_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": raw_prompt},
        ]

        turns: list[AgenticTurn] = []
        metrics = {}
        final_diffusion_output = None
        final_logprobs = None
        decision = "stop"

        for turn_idx in range(max_turns):
            # 1. Tokenize multimodal chat
            prompt_ids = await self.apply_chat_template(chat_messages, images=images, videos=videos)

            # 2. Generate agent text (understanding path, with logprobs)
            with simple_timer(f"agent_text_gen_turn_{turn_idx}", metrics):
                gen_params = {**sampling_params, "logprobs": True}
                text_output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=gen_params,
                    image_data=images,
                    video_data=videos,
                )

            agent_text = text_output.diffusion_output
            agent_tokens = prompt_ids
            agent_logprobs = text_output.log_probs
            if agent_logprobs is None:
                agent_logprobs = []
            elif hasattr(agent_logprobs, "tolist"):
                agent_logprobs = agent_logprobs.tolist()

            # 3. Parse agent output
            parsed = parse_agent_output(agent_text)
            decision = parsed["decision"]
            rewritten_prompt = parsed["prompt"] or raw_prompt

            if decision == "stop":
                turn = AgenticTurn(
                    turn_idx=turn_idx, agent_tokens=agent_tokens,
                    agent_logprobs=agent_logprobs, agent_text=agent_text,
                    tool_call=None, tool_output=None, decision="stop",
                )
                turns.append(turn)
                break

            # 4. Call frozen diffusion tool (generation path) with rewritten prompt
            with simple_timer(f"tool_call_turn_{turn_idx}", metrics):
                tool_params = {**sampling_params, "logprobs": False, "prompt": rewritten_prompt}
                img_output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=tool_params,
                    image_data=images,
                    video_data=videos,
                )

            diffusion_output = img_output.diffusion_output
            final_diffusion_output = diffusion_output
            final_logprobs = img_output.log_probs

            tool_call = ToolCall(tool_name="t2i", params={"prompt": rewritten_prompt})
            tool_output = ToolOutput(output_type="image", output_data=diffusion_output)

            # 5. Capture this turn
            turn = AgenticTurn(
                turn_idx=turn_idx, agent_tokens=agent_tokens,
                agent_logprobs=agent_logprobs, agent_text=agent_text,
                tool_call=tool_call, tool_output=tool_output,
                decision="continue",
            )
            turns.append(turn)

            # 6. Append to chat history for next turn
            chat_messages.append({"role": "assistant", "content": agent_text})
            chat_messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": diffusion_output},
                    {"type": "text", "text": "Here is the generated image. Review it and decide whether to continue."},
                ],
            })

        # Build trajectory
        meta = AgenticMetadata(
            num_turns=len(turns),
            terminated=decision == "stop",
            termination_reason="agent_stop" if decision == "stop" else "max_turns",
        )
        trajectory = AgenticTrajectory(prompt=raw_prompt, turns=turns, metadata=meta)

        output = DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=final_diffusion_output,
            response_logprobs=final_logprobs,
            num_turns=len(turns) + 1,
            metrics=metrics,
            extra_fields={"agentic_trajectory": trajectory},
        )
        return output
