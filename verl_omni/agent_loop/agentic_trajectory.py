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
"""Agentic multi-turn trajectory data structures for Mode (2a) agentic RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass
class ToolCall:
    """A tool invocation by the agent, containing a (possibly rewritten) prompt."""

    tool_name: str
    params: dict[str, Any]  # key "prompt" holds the prompt sent to diffusion


@dataclass
class ToolOutput:
    """Observation returned by a frozen diffusion tool call."""

    output_type: Literal["image", "video"]
    output_data: torch.Tensor  # image tensor [C,H,W] or video [T,C,H,W]


@dataclass
class AgenticTurn:
    """One turn of the multi-turn agentic interaction.

    Prompt rewriting is captured via:
      - turn[i].tool_call.params["prompt"]  — prompt at turn i
      - turn[i+1].tool_call.params["prompt"] — rewritten prompt at turn i+1
    """

    turn_idx: int
    agent_tokens: list[int]      # full agent text tokens (loss_mask=1)
    agent_logprobs: list[float]  # per-token logprobs from rollout
    agent_text: str              # decoded text: reasoning + prompt + decision
    tool_call: ToolCall | None = None      # None on termination turn
    tool_output: ToolOutput | None = None  # None on termination turn
    decision: Literal["continue", "terminate"] = "terminate"


@dataclass
class AgenticMetadata:
    """Trajectory-level metadata."""

    num_turns: int
    terminated: bool
    termination_reason: str  # "agent_stop" | "max_turns"


@dataclass
class AgenticTrajectory:
    """Full multi-turn agentic trajectory with prompt rewriting captured per turn."""

    prompt: str
    turns: list[AgenticTurn]
    reward_score: float | None = None
    metadata: AgenticMetadata = field(default_factory=lambda: AgenticMetadata(0, False, ""))


def agentic_trajectory_to_dict(traj: AgenticTrajectory) -> dict[str, Any]:
    """Serialize an AgenticTrajectory to a JSON-serializable dict for non_tensor_batch."""
    return {
        "prompt": traj.prompt,
        "turns": [
            {
                "turn_idx": t.turn_idx,
                "agent_tokens": t.agent_tokens,
                "agent_logprobs": t.agent_logprobs,
                "agent_text": t.agent_text,
                "tool_call": (
                    {"tool_name": t.tool_call.tool_name, "params": t.tool_call.params}
                    if t.tool_call else None
                ),
                "tool_output": (
                    {"output_type": t.tool_output.output_type,
                     "output_data_shape": list(t.tool_output.output_data.shape)}
                    if t.tool_output else None
                ),
                "decision": t.decision,
            }
            for t in traj.turns
        ],
        "reward_score": traj.reward_score,
        "metadata": {
            "num_turns": traj.metadata.num_turns,
            "terminated": traj.metadata.terminated,
            "termination_reason": traj.metadata.termination_reason,
        },
    }


def agentic_trajectory_from_dict(d: dict[str, Any]) -> AgenticTrajectory:
    """Deserialize an AgenticTrajectory from a dict."""
    turns = []
    for t in d["turns"]:
        tool_call = None
        if t.get("tool_call") is not None:
            tool_call = ToolCall(**t["tool_call"])
        tool_output = None
        if t.get("tool_output") is not None:
            to_dict = dict(t["tool_output"])
            if "output_data_shape" in to_dict:
                shape = to_dict.pop("output_data_shape")
                to_dict["output_data"] = torch.zeros(shape)
            tool_output = ToolOutput(**to_dict)
        turns.append(AgenticTurn(
            turn_idx=t["turn_idx"],
            agent_tokens=t["agent_tokens"],
            agent_logprobs=t["agent_logprobs"],
            agent_text=t["agent_text"],
            tool_call=tool_call,
            tool_output=tool_output,
            decision=t["decision"],
        ))
    return AgenticTrajectory(
        prompt=d["prompt"],
        turns=turns,
        reward_score=d.get("reward_score"),
        metadata=AgenticMetadata(**d["metadata"]),
    )
