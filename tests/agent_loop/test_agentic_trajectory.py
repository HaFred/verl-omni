# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the isolated agent-loop extension and trajectory metadata."""

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
from verl_omni.agent_loop.diffusion_ar_multi_turn_agent_loop import (
    DiffusionARMultiTurnAgentLoop,
    _user_text_from_raw_prompt,
)


def _make_turn(idx, tokens, logprobs, text, decision, tc=None, to=None):
    return AgenticTurn(idx, tokens, logprobs, text, tc, to, decision)


class TestRawPromptNormalization:
    def test_string_passthrough(self):
        assert _user_text_from_raw_prompt("a blue hat") == "a blue hat"

    def test_chat_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Generate a cat wearing a blue hat"},
        ]
        assert _user_text_from_raw_prompt(messages) == "Generate a cat wearing a blue hat"


class TestTrajectory:
    def test_round_trip_full(self):
        t0 = _make_turn(
            0,
            [1, 2, 3],
            [0.1, 0.2, 0.3],
            "text0",
            "continue",
            ToolCall("t2i", {"prompt": "p0"}),
            ToolOutput("image", torch.zeros(3, 64, 64), is_stub=True),
        )
        t1 = _make_turn(1, [4, 5], [0.4, 0.5], "text1", "stop")
        trajectory = AgenticTrajectory(
            "test",
            [t0, t1],
            0.8,
            AgenticMetadata(2, True, "agent_stop", tool_stubbed=True),
        )

        restored = agentic_trajectory_from_dict(agentic_trajectory_to_dict(trajectory))

        assert restored.prompt == "test"
        assert restored.reward_score == 0.8
        assert restored.metadata.num_turns == 2
        assert restored.metadata.tool_stubbed is True
        assert restored.turns[0].tool_call.params["prompt"] == "p0"
        assert restored.turns[0].tool_output.is_stub is True

    def test_round_trip_minimal(self):
        trajectory = AgenticTrajectory("minimal", [_make_turn(0, [1], [0.1], "text", "stop")])
        restored = agentic_trajectory_from_dict(agentic_trajectory_to_dict(trajectory))
        assert restored.prompt == "minimal"
        assert len(restored.turns) == 1

    def test_prompt_rewriting_captured(self):
        t0 = _make_turn(
            0,
            [1],
            [0.1],
            "t0",
            "continue",
            ToolCall("t2i", {"prompt": "original"}),
            ToolOutput("image", torch.zeros(3, 8, 8)),
        )
        t1 = _make_turn(
            1,
            [2],
            [0.2],
            "t1",
            "stop",
            ToolCall("t2i", {"prompt": "rewritten"}),
        )
        trajectory = AgenticTrajectory("p", [t0, t1])
        assert trajectory.turns[0].tool_call.params["prompt"] == "original"
        assert trajectory.turns[1].tool_call.params["prompt"] == "rewritten"


class TestAgentOutputParser:
    def test_full_format(self):
        text = "<reasoning>analyze</reasoning>\n<prompt>A test</prompt>\n<decision>continue</decision>"
        parsed = parse_agent_output(text)
        assert parsed["reasoning"] == "analyze"
        assert parsed["prompt"] == "A test"
        assert parsed["decision"] == "continue"

    def test_non_exact_continue_stops(self):
        assert parse_agent_output("<decision>discontinue</decision>")["decision"] == "stop"
        assert parse_agent_output("<decision>continuous</decision>")["decision"] == "stop"

    def test_malformed_stops(self):
        parsed = parse_agent_output("garbage text")
        assert parsed["decision"] == "stop"
        assert parsed["reasoning"] is None


class TestPr1TrainMaskContract:
    def test_turn0_only_mask_pattern(self):
        turns = [
            _make_turn(0, [1, 2, 3], [0.1, 0.2, 0.3], "t0", "continue"),
            _make_turn(1, [4, 5], [0.4, 0.5], "t1", "stop"),
        ]
        response_mask = [
            train_mask for turn in turns for train_mask in [1 if turn.turn_idx == 0 else 0] * len(turn.agent_tokens)
        ]
        assert response_mask == [1, 1, 1, 0, 0]


class TestLogprobs:
    def test_reencode_without_logprobs_raises(self):
        loop = DiffusionARMultiTurnAgentLoop.__new__(DiffusionARMultiTurnAgentLoop)
        try:
            loop._require_logprobs(None, 3, reencoded=True)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "log_probs are missing" in str(exc)

    def test_native_tokens_allow_missing_logprobs(self):
        loop = DiffusionARMultiTurnAgentLoop.__new__(DiffusionARMultiTurnAgentLoop)
        assert loop._require_logprobs(None, 2, reencoded=False) == [0.0, 0.0]


class TestAgenticConfig:
    def test_reads_rollout_custom_extension(self):
        loop = DiffusionARMultiTurnAgentLoop.__new__(DiffusionARMultiTurnAgentLoop)
        loop.rollout_config = {
            "custom": {
                "agentic": {
                    "max_turns": 3,
                    "early_termination": False,
                }
            }
        }
        assert loop._agentic_config() == {"max_turns": 3, "early_termination": False}


class TestAgenticReward:
    def test_format_heuristic_from_trajectory(self):
        from verl_omni.utils.reward_score.agentic_reward import compute_score

        text = "<reasoning>r</reasoning>\n<prompt>a cat</prompt>\n<decision>stop</decision>"
        trajectory = {
            "turns": [{"agent_text": text, "decision": "stop"}],
            "metadata": {"tool_stubbed": True},
        }
        result = compute_score(
            "jpeg_compressibility",
            solution_str="ignored",
            extra_info={"agentic_trajectory": trajectory, "tool_stubbed": True},
        )
        assert result["method"] == "agentic_format_heuristic"
        assert result["score"] == 1.0
        assert result["tool_stubbed"] is True

    def test_format_heuristic_from_stock_dapo_response(self):
        from verl_omni.utils.reward_score.agentic_reward import compute_score

        text = "<reasoning>r</reasoning>\n<prompt>a cat</prompt>\n<decision>stop</decision>"
        result = compute_score("jpeg_compressibility", solution_str=text)
        assert result["method"] == "agentic_format_heuristic"
        assert result["score"] == 1.0

    def test_length_fallback(self):
        from verl_omni.utils.reward_score.agentic_reward import compute_score

        result = compute_score("x", solution_str="a" * 128)
        assert result["method"] == "response_length_heuristic"
        assert abs(result["score"] - 0.5) < 1e-6


class TestBackwardCompatibility:
    def test_single_turn_agent(self):
        from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop

        assert DiffusionSingleTurnAgentLoop is not None
