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
"""Agentic multi-turn trajectory data structures for Mode (2a) agentic RL.

Defines two layers:

1. **SFT layer** (aligned with #295): ``VisualReflectionTrajectory`` and its
   constituent types (``ImageRef``, ``ReflectionStep``).  These are the
   canonical format for multi-turn visual-reflection data produced by SFT
   cold-start training (#295) and by offline dataset adapters.

2. **RL layer**: ``AgenticTrajectory`` and its constituent types
   (``AgenticTurn``, ``ToolCall``, ``ToolOutput``, ``AgenticMetadata``).
   This is a superset of the SFT format that adds rollout-specific fields
   (token IDs, logprobs, loss masks, tool outputs, reward scores).

Conversion: ``visual_reflection_to_agentic()`` lifts a
``VisualReflectionTrajectory`` to an ``AgenticTrajectory``, filling RL
fields with sensible defaults so that SFT checkpoints can serve as
cold-start initialization for agentic RL training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


# ---------------------------------------------------------------------------
# SFT layer — aligned with #295 (Multi-Turn Visual Reflection SFT for BAGEL)
# ---------------------------------------------------------------------------

@dataclass
class ImageRef:
    """Reference to an image by URI and content hash."""

    uri: str
    sha256: str = ""


@dataclass
class ReflectionStep:
    """One step of the visual-reflection loop.

    Decision vocabulary matches #295: ``"continue"`` or ``"stop"``.
    """

    reflection: str
    action: Literal["continue", "stop"]
    edit: str  # non-empty for "continue", empty for "stop"


@dataclass
class VisualReflectionTrajectory:
    """Canonical SFT trajectory format — aligned with #295.

    For a trajectory with *N* images:
    - ``len(images) == len(steps) == N``
    - the first *N-1* steps are ``"continue"`` with non-empty ``edit``
    - the final step is ``"stop"`` with empty ``edit``
    """

    trajectory_id: str
    prompt: str
    images: list[ImageRef]
    steps: list[ReflectionStep]
    # provenance
    manifest_id: str = ""
    source_dataset: str = ""
    source_record_id: str = ""
    pipeline_variant: Literal["direct_parse", "pair_0_1_turn", "prompt_k_turn"] = "direct_parse"


# ---------------------------------------------------------------------------
# RL layer — superset of the SFT format
# ---------------------------------------------------------------------------

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

    Decision vocabulary aligned with #295: ``"continue"`` or ``"stop"``.

    Prompt rewriting is captured via:
      - turn[i].tool_call.params["prompt"]  — prompt at turn i
      - turn[i+1].tool_call.params["prompt"] — rewritten prompt at turn i+1
    """

    turn_idx: int
    agent_tokens: list[int]      # full agent text tokens (loss_mask=1)
    agent_logprobs: list[float]  # per-token logprobs from rollout
    agent_text: str              # decoded text: reasoning + prompt + decision
    tool_call: ToolCall | None = None      # None on stop turn
    tool_output: ToolOutput | None = None  # None on stop turn
    decision: Literal["continue", "stop"] = "stop"


@dataclass
class AgenticMetadata:
    """Trajectory-level metadata."""

    num_turns: int
    terminated: bool
    termination_reason: str  # "agent_stop" | "max_turns"


@dataclass
class AgenticTrajectory:
    """Full multi-turn agentic trajectory — RL superset of VisualReflectionTrajectory.

    RL-specific fields (``turns``, ``reward_score``, ``metadata``) are added
    on top of the SFT provenance fields shared with #295.
    """

    prompt: str
    turns: list[AgenticTurn]
    reward_score: float | None = None
    metadata: AgenticMetadata = field(default_factory=lambda: AgenticMetadata(0, False, ""))
    # provenance — aligned with VisualReflectionTrajectory
    trajectory_id: str = ""
    source_dataset: str = ""


# ---------------------------------------------------------------------------
# SFT → RL conversion
# ---------------------------------------------------------------------------

def visual_reflection_to_agentic(
    sft_traj: VisualReflectionTrajectory,
) -> AgenticTrajectory:
    """Convert a #295 VisualReflectionTrajectory to an RL AgenticTrajectory.

    RL-specific fields (agent_tokens, agent_logprobs, tool_output tensors)
    are initialised to sensible defaults.  The resulting trajectory is
    suitable as cold-start input for agentic RL training.
    """
    turns: list[AgenticTurn] = []
    for i, step in enumerate(sft_traj.steps):
        has_image = (i < len(sft_traj.images))
        tc = ToolCall("t2i", {"prompt": step.edit}) if step.action == "continue" and step.edit else None
        to = ToolOutput("image", torch.empty(0)) if has_image else None

        turns.append(AgenticTurn(
            turn_idx=i,
            agent_tokens=[],
            agent_logprobs=[],
            agent_text=(
                f"Reflection: {step.reflection}\n"
                f"Action: {step.action}\n"
                f"Edit: {step.edit}"
            ),
            tool_call=tc,
            tool_output=to,
            decision=step.action,
        ))

    metadata = AgenticMetadata(
        num_turns=len(turns),
        terminated=(sft_traj.steps[-1].action == "stop") if sft_traj.steps else False,
        termination_reason="agent_stop" if (sft_traj.steps and sft_traj.steps[-1].action == "stop") else "max_turns",
    )

    return AgenticTrajectory(
        prompt=sft_traj.prompt,
        turns=turns,
        reward_score=None,
        metadata=metadata,
        trajectory_id=sft_traj.trajectory_id,
        source_dataset=sft_traj.source_dataset,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def agentic_trajectory_to_dict(traj: AgenticTrajectory) -> dict[str, Any]:
    """Serialize an AgenticTrajectory to a JSON-serializable dict for non_tensor_batch."""
    return {
        "prompt": traj.prompt,
        "trajectory_id": traj.trajectory_id,
        "source_dataset": traj.source_dataset,
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
        trajectory_id=d.get("trajectory_id", ""),
        source_dataset=d.get("source_dataset", ""),
    )
