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
from omegaconf import OmegaConf
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

_TURN2PLUS_MASK_WARNED = False


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

    Executes: agent reasons -> rewrites prompt -> external tool generates image ->
    agent reflects -> continue/stop -> repeat.
    Each turn's rewritten prompt is captured explicitly in AgenticTrajectory.

    Returns ``AgentLoopOutput`` so stock ``AgentLoopWorker`` postprocess can pad
    ``response_ids`` / ``response_mask``.

    PR1 training contract (pragmatic):
      - Flat concat of agent tokens only (tool images stay in chat history).
      - ``response_mask=1`` for turn-0 agent tokens; turn ≥1 masked to 0 until
        full chat-template retokenize lands in PR2 (train↔rollout parity).
      - Und-only tool path may synthesize a stub image (marked ``is_stub`` /
        ``tool_stubbed``); not a real diffusion tool claim.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _agentic_config(self) -> dict[str, Any]:
        """Read custom loop knobs without extending the upstream model schema."""
        if OmegaConf.is_config(self.rollout_config):
            agentic_cfg = OmegaConf.select(self.rollout_config, "custom.agentic")
        elif isinstance(self.rollout_config, dict):
            agentic_cfg = (self.rollout_config.get("custom") or {}).get("agentic")
        else:
            custom_cfg = getattr(self.rollout_config, "custom", None) or {}
            agentic_cfg = custom_cfg.get("agentic") if isinstance(custom_cfg, dict) else None
        if agentic_cfg is None:
            return {}
        if OmegaConf.is_config(agentic_cfg):
            return OmegaConf.to_container(agentic_cfg, resolve=True) or {}
        if isinstance(agentic_cfg, dict):
            return agentic_cfg
        return {
            "max_turns": getattr(agentic_cfg, "max_turns", 5),
            "early_termination": getattr(agentic_cfg, "early_termination", True),
        }

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

    def _extract_tool_image(self, img_output: Any) -> tuple[Any, bool]:
        """Return ``(image_tensor, is_stub)``. Stub when AR-only rollout has no diffusion_output."""
        diffusion_output = getattr(img_output, "diffusion_output", None)
        if isinstance(diffusion_output, torch.Tensor):
            return diffusion_output, False
        logger.warning(
            "No diffusion_output from tool generate; using stub image tensor "
            "(und-only smoke — not a real diffusion tool)"
        )
        return torch.zeros(3, 64, 64), True

    def _require_logprobs(self, logprobs: Any, n_tokens: int, *, reencoded: bool) -> list[float]:
        """Align logprobs to token length; never silently zero-fill when re-encoding."""
        if logprobs is None:
            if reencoded and n_tokens > 0:
                raise RuntimeError(
                    "Agent text was re-encoded from string output but log_probs are missing. "
                    "Refusing to invent zero old-logprobs for PPO/GRPO. "
                    "Ensure rollout returns token_ids + log_probs (calculate_log_probs=true)."
                )
            return [0.0] * n_tokens
        if hasattr(logprobs, "tolist"):
            logprobs = logprobs.tolist()
        values = list(logprobs)
        if len(values) < n_tokens:
            if reencoded:
                raise RuntimeError(
                    f"Re-encoded agent text has {n_tokens} tokens but only {len(values)} log_probs. "
                    "Refusing to zero-pad old logprobs for PPO/GRPO."
                )
            values = values + [0.0] * (n_tokens - len(values))
        return values[:n_tokens]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        global _TURN2PLUS_MASK_WARNED

        raw_prompt = _user_text_from_raw_prompt(kwargs["raw_prompt"])
        agentic_cfg = self._agentic_config()
        max_turns = int(sampling_params.get("max_turns", agentic_cfg.get("max_turns", 5)))
        early_termination = bool(agentic_cfg.get("early_termination", True))

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
        tool_stubbed = False
        termination_reason = "max_turns"

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
            reencoded = False
            if not agent_token_ids and agent_text:
                agent_token_ids = self.tokenizer.encode(agent_text, add_special_tokens=False)
                reencoded = True

            agent_logprobs = self._require_logprobs(
                getattr(text_output, "log_probs", None),
                len(agent_token_ids),
                reencoded=reencoded,
            )

            # PR1: only turn-0 agent tokens enter the policy gradient. Later turns
            # were generated under a different chat-template context than
            # initial_prompt + flat response; full retokenize is PR2.
            train_mask = 1 if turn_idx == 0 else 0
            if turn_idx > 0 and not _TURN2PLUS_MASK_WARNED:
                logger.warning(
                    "Multi-turn agentic PR1: masking turn>=1 agent tokens out of "
                    "response_mask (train turn-0 only). Full train↔rollout retokenize "
                    "deferred to PR2."
                )
                _TURN2PLUS_MASK_WARNED = True

            response_ids.extend(agent_token_ids)
            response_mask.extend([train_mask] * len(agent_token_ids))
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
                termination_reason = "agent_stop"
                if early_termination:
                    break
                # early_termination=False: keep looping until max_turns even after stop.
                continue

            with simple_timer(f"tool_call_turn_{turn_idx}", metrics):
                tool_params = {**sampling_params, "logprobs": False, "prompt": rewritten_prompt}
                img_output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids + agent_token_ids,
                    sampling_params=tool_params,
                    image_data=images,
                    video_data=videos,
                )

            diffusion_output, is_stub = self._extract_tool_image(img_output)
            tool_stubbed = tool_stubbed or is_stub

            turns.append(
                AgenticTurn(
                    turn_idx=turn_idx,
                    agent_tokens=agent_token_ids,
                    agent_logprobs=agent_logprobs,
                    agent_text=agent_text,
                    tool_call=ToolCall(tool_name="t2i", params={"prompt": rewritten_prompt}),
                    tool_output=ToolOutput(
                        output_type="image",
                        output_data=diffusion_output,
                        is_stub=is_stub,
                    ),
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
                termination_reason = "response_truncated"
                break
        else:
            if decision != "stop":
                termination_reason = "max_turns"

        meta = AgenticMetadata(
            num_turns=len(turns),
            terminated=decision == "stop",
            termination_reason=termination_reason,
            tool_stubbed=tool_stubbed,
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
            # Align with AgenticMetadata.num_turns (agent decision turns, not +1 user).
            num_turns=len(turns),
            metrics=metrics,
            extra_fields={
                "agentic_trajectory": traj_dict,
                "tool_stubbed": tool_stubbed,
            },
        )
