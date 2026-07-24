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

"""Tests for AgenticTrajectory: round-trip, loss mask, parser, backward compat."""

import torch

from verl_omni.agent_loop.agent_output_parser import parse_agent_output
from verl_omni.agent_loop.agentic_trajectory import (
    AgenticMetadata,
    AgenticTrajectory,
    AgenticTurn,
    ToolCall,
    ToolOutput,
    agentic_trajectory_from_dict,
    agentic_trajectory_to_dict,
)
from verl_omni.agent_loop.trajectory_serializer import serialize_trajectories


def _make_turn(idx, tokens, logprobs, text, decision, tc=None, to=None):
    return AgenticTurn(idx, tokens, logprobs, text, tc, to, decision)


class TestTrajectoryRoundTrip:
    def test_round_trip_full(self):
        t0 = _make_turn(0, [1, 2, 3], [0.1, 0.2, 0.3], "text0", "continue",
                        ToolCall("t2i", {"prompt": "p0"}),
                        ToolOutput("image", torch.zeros(3, 64, 64)))
        t1 = _make_turn(1, [4, 5], [0.4, 0.5], "text1", "terminate",
                        ToolCall("t2i", {"prompt": "p1 rewritten"}),
                        ToolOutput("image", torch.ones(3, 64, 64)))
        traj = AgenticTrajectory("test", [t0, t1], 0.8, AgenticMetadata(2, True, "agent_stop"))

        d = agentic_trajectory_to_dict(traj)
        restored = agentic_trajectory_from_dict(d)

        assert restored.prompt == "test"
        assert restored.reward_score == 0.8
        assert restored.metadata.num_turns == 2
        assert restored.turns[0].tool_call.params["prompt"] == "p0"
        assert restored.turns[1].tool_call.params["prompt"] == "p1 rewritten"

    def test_round_trip_minimal(self):
        t0 = _make_turn(0, [1], [0.1], "text", "terminate")
        traj = AgenticTrajectory("minimal", [t0])
        d = agentic_trajectory_to_dict(traj)
        restored = agentic_trajectory_from_dict(d)
        assert restored.prompt == "minimal"
        assert len(restored.turns) == 1


class TestLossMasking:
    def test_loss_mask_agent_tokens_only(self):
        t0 = _make_turn(0, [1, 2, 3], [0.1, 0.2, 0.3], "t0", "continue",
                        ToolCall("t2i", {"prompt": "p0"}),
                        ToolOutput("image", torch.zeros(3, 512, 512)))
        t1 = _make_turn(1, [4, 5, 6], [0.4, 0.5, 0.6], "t1", "terminate",
                        None, ToolOutput("image", torch.ones(3, 512, 512)))
        traj = AgenticTrajectory("p", [t0, t1], metadata=AgenticMetadata(2, True, "agent_stop"))

        result = serialize_trajectories([traj], 256, 128, 1024)
        loss_mask = result["loss_mask"][0]
        assert loss_mask.sum().item() == 6, f"Expected 6, got {loss_mask.sum()}"

    def test_loss_mask_single_turn(self):
        t0 = _make_turn(0, [1, 2], [0.1, 0.2], "t", "terminate")
        traj = AgenticTrajectory("p", [t0])
        result = serialize_trajectories([traj], 256, 128, 1024)
        mask = result["loss_mask"][0]
        assert mask[:2].sum() == 2
        assert mask[2:].sum() == 0

    def test_prompt_rewriting_captured(self):
        t0 = _make_turn(0, [1, 2], [0.1, 0.2], "t0", "continue",
                        ToolCall("t2i", {"prompt": "original prompt"}),
                        ToolOutput("image", torch.zeros(3, 64, 64)))
        t1 = _make_turn(1, [3, 4], [0.3, 0.4], "t1", "terminate",
                        ToolCall("t2i", {"prompt": "rewritten prompt v2"}),
                        ToolOutput("image", torch.ones(3, 64, 64)))
        traj = AgenticTrajectory("p", [t0, t1])
        assert traj.turns[0].tool_call.params["prompt"] == "original prompt"
        assert traj.turns[1].tool_call.params["prompt"] == "rewritten prompt v2"
        assert traj.turns[0].tool_call.params["prompt"] != traj.turns[1].tool_call.params["prompt"]


class TestAgentOutputParser:
    def test_full_format(self):
        text = "<reasoning>analyze</reasoning>\n<prompt>A test</prompt>\n<decision>continue</decision>"
        r = parse_agent_output(text)
        assert r["reasoning"] == "analyze"
        assert r["prompt"] == "A test"
        assert r["decision"] == "continue"

    def test_terminate(self):
        text = "<reasoning>done</reasoning>\n<prompt></prompt>\n<decision>terminate</decision>"
        r = parse_agent_output(text)
        assert r["decision"] == "terminate"

    def test_malformed(self):
        r = parse_agent_output("garbage text")
        assert r["decision"] == "terminate"
        assert r["reasoning"] is None


class TestBackwardCompatibility:
    def test_diffusion_config(self):
        from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
        c = DiffusionAlgoConfig()
        assert c.adv_estimator == "flow_grpo"

    def test_flow_grpo_adv_estimator(self):
        from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_adv_estimator_fn
        assert get_diffusion_adv_estimator_fn("flow_grpo") is not None

    def test_agentic_grpo_adv_estimator(self):
        from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_adv_estimator_fn
        assert get_diffusion_adv_estimator_fn("agentic_grpo") is not None

    def test_single_turn_agent(self):
        from verl_omni.agent_loop import DiffusionSingleTurnAgentLoop
        assert DiffusionSingleTurnAgentLoop is not None
