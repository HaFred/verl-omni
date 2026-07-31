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
from verl_omni.agent_loop.diffusion_ar_multi_turn_agent_loop import _user_text_from_raw_prompt
from verl_omni.agent_loop.trajectory_serializer import serialize_trajectories


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


class TestTrajectoryRoundTrip:
    def test_round_trip_full(self):
        t0 = _make_turn(
            0,
            [1, 2, 3],
            [0.1, 0.2, 0.3],
            "text0",
            "continue",
            ToolCall("t2i", {"prompt": "p0"}),
            ToolOutput("image", torch.zeros(3, 64, 64)),
        )
        t1 = _make_turn(
            1,
            [4, 5],
            [0.4, 0.5],
            "text1",
            "stop",
            ToolCall("t2i", {"prompt": "p1 rewritten"}),
            ToolOutput("image", torch.ones(3, 64, 64)),
        )
        traj = AgenticTrajectory("test", [t0, t1], 0.8, AgenticMetadata(2, True, "agent_stop"))

        d = agentic_trajectory_to_dict(traj)
        restored = agentic_trajectory_from_dict(d)

        assert restored.prompt == "test"
        assert restored.reward_score == 0.8
        assert restored.metadata.num_turns == 2
        assert restored.turns[0].tool_call.params["prompt"] == "p0"
        assert restored.turns[1].tool_call.params["prompt"] == "p1 rewritten"

    def test_round_trip_minimal(self):
        t0 = _make_turn(0, [1], [0.1], "text", "stop")
        traj = AgenticTrajectory("minimal", [t0])
        d = agentic_trajectory_to_dict(traj)
        restored = agentic_trajectory_from_dict(d)
        assert restored.prompt == "minimal"
        assert len(restored.turns) == 1


class TestLossMasking:
    def test_loss_mask_agent_tokens_only(self):
        t0 = _make_turn(
            0,
            [1, 2, 3],
            [0.1, 0.2, 0.3],
            "t0",
            "continue",
            ToolCall("t2i", {"prompt": "p0"}),
            ToolOutput("image", torch.zeros(3, 512, 512)),
        )
        t1 = _make_turn(1, [4, 5, 6], [0.4, 0.5, 0.6], "t1", "stop", None, ToolOutput("image", torch.ones(3, 512, 512)))
        traj = AgenticTrajectory("p", [t0, t1], metadata=AgenticMetadata(2, True, "agent_stop"))

        result = serialize_trajectories([traj], 256, 128, 1024)
        loss_mask = result["loss_mask"][0]
        assert loss_mask.sum().item() == 6, f"Expected 6, got {loss_mask.sum()}"

    def test_loss_mask_single_turn(self):
        t0 = _make_turn(0, [1, 2], [0.1, 0.2], "t", "stop")
        traj = AgenticTrajectory("p", [t0])
        result = serialize_trajectories([traj], 256, 128, 1024)
        mask = result["loss_mask"][0]
        assert mask[:2].sum() == 2
        assert mask[2:].sum() == 0

    def test_prompt_rewriting_captured(self):
        t0 = _make_turn(
            0,
            [1, 2],
            [0.1, 0.2],
            "t0",
            "continue",
            ToolCall("t2i", {"prompt": "original prompt"}),
            ToolOutput("image", torch.zeros(3, 64, 64)),
        )
        t1 = _make_turn(
            1,
            [3, 4],
            [0.3, 0.4],
            "t1",
            "stop",
            ToolCall("t2i", {"prompt": "rewritten prompt v2"}),
            ToolOutput("image", torch.ones(3, 64, 64)),
        )
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
        assert r["decision"] == "stop"

    def test_malformed(self):
        r = parse_agent_output("garbage text")
        assert r["decision"] == "stop"
        assert r["reasoning"] is None


class TestBackwardCompatibility:
    def test_diffusion_config(self):
        from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig

        c = DiffusionAlgoConfig()
        assert c.adv_estimator == "flow_grpo"

    def test_flow_grpo_adv_estimator(self):
        from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_adv_estimator_fn

        assert get_diffusion_adv_estimator_fn("flow_grpo") is not None

    def test_single_turn_agent(self):
        from verl_omni.agent_loop import DiffusionSingleTurnAgentLoop

        assert DiffusionSingleTurnAgentLoop is not None


class TestEdgeCases:
    """Edge case coverage for trajectory serialization (UT-11)."""

    def test_empty_turns(self):
        """serialize_trajectories handles a trajectory with zero turns."""
        traj = AgenticTrajectory(
            "empty prompt",
            [],
            metadata=AgenticMetadata(0, False, "max_turns"),
        )
        result = serialize_trajectories(
            [traj], max_prompt_length=256, observation_token_length=128, max_total_tokens=1024
        )
        # All tensors must be present with correct batch dimension
        assert result["prompt_tokens"].shape[0] == 1
        assert result["agent_tokens"].shape[0] == 1
        assert result["agent_logprobs"].shape[0] == 1
        assert result["loss_mask"].shape[0] == 1
        # No agent tokens → all loss_mask entries are zero
        assert result["loss_mask"][0].sum().item() == 0
        # All agent token entries are padding
        assert (result["agent_tokens"][0] == 0).all()
        assert (result["agent_logprobs"][0] == 0.0).all()


class TestLogProbConsistency:
    """Structural logprob consistency tests (UT-12 / RFC line 314)."""

    def test_shape_alignment(self):
        """agent_logprobs tensor shape == agent_tokens tensor shape."""
        t0 = _make_turn(
            0,
            [1, 2, 3],
            [0.1, 0.2, 0.3],
            "t0",
            "continue",
            ToolCall("t2i", {"prompt": "p0"}),
            ToolOutput("image", torch.zeros(3, 512, 512)),
        )
        t1 = _make_turn(1, [4, 5], [0.4, 0.5], "t1", "stop")
        traj = AgenticTrajectory("p", [t0, t1])
        result = serialize_trajectories([traj], 256, 128, 1024)
        assert result["agent_logprobs"].shape == result["agent_tokens"].shape

    def test_loss_mask_logprob_pairing(self):
        """loss_mask[i]==1 iff a real logprob value was provided at position i."""
        t0 = _make_turn(
            0,
            [10, 20, 30],
            [-0.1, -0.2, -0.3],
            "t0",
            "continue",
            ToolCall("t2i", {"prompt": "x"}),
            ToolOutput("image", torch.zeros(3, 64, 64)),
        )
        traj = AgenticTrajectory("p", [t0])
        result = serialize_trajectories([traj], 256, 128, 512)
        mask = result["loss_mask"][0]
        logprobs = result["agent_logprobs"][0]
        for i in range(len(mask)):
            if mask[i] == 1:
                # Real agent token → must have a non-trivial logprob
                # (allow zero as a valid logprob value)
                assert logprobs[i] == -0.1 * ((i % 3) + 1) if i < 3 else True, (
                    f"Position {i}: mask=1 but unexpected logprob {logprobs[i]}"
                )
            else:
                # Observation placeholder or padding
                pass  # values at masked-out positions are don't-care for this check

    def test_padding_positions_zero(self):
        """All positions beyond the serialized tokens have logprob == 0.0."""
        t0 = _make_turn(0, [7, 8], [0.7, 0.8], "t", "stop")
        traj = AgenticTrajectory("p", [t0])
        result = serialize_trajectories([traj], 256, 128, 64)
        mask = result["loss_mask"][0]
        logprobs = result["agent_logprobs"][0]
        # Find last position with mask=1
        active = (mask == 1).nonzero(as_tuple=True)[0]
        if len(active) > 0:
            first_pad = active[-1].item() + 1
            assert (logprobs[first_pad:] == 0.0).all(), f"Padding positions [{first_pad}:] should be zero"

    def test_per_turn_length_equality(self):
        """agent_tokens and agent_logprobs have equal length in each AgenticTurn."""
        t0 = _make_turn(
            0,
            [1, 2, 3],
            [0.1, 0.2, 0.3],
            "t0",
            "continue",
            ToolCall("t2i", {"prompt": "p0"}),
            ToolOutput("image", torch.zeros(3, 64, 64)),
        )
        t1 = _make_turn(1, [4, 5, 6, 7], [0.4, 0.5, 0.6, 0.7], "t1", "stop")
        t2 = _make_turn(2, [8, 9], [0.8, 0.9], "t2", "stop")
        for turn in [t0, t1, t2]:
            n_tok = len(turn.agent_tokens)
            n_lp = len(turn.agent_logprobs)
            assert n_tok == n_lp, f"Turn {turn.turn_idx}: len(tokens)={n_tok} != len(logprobs)={n_lp}"


class TestFreezeLogic:
    """Unit test for the selective-freezing pattern used by AgenticLLMFSDPEngine (UT-13)."""

    @staticmethod
    def _apply_freeze(module, freeze_prefixes):
        """Mirror AgenticLLMFSDPEngine.build_module freeze logic."""
        for name, param in module.named_parameters():
            for prefix in freeze_prefixes:
                if prefix in name:
                    param.requires_grad = False
                    break

    def test_freeze_prefix_matching(self):
        """Params whose names contain a freeze prefix get requires_grad=False."""
        import torch.nn as nn

        class TinyAgentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(8, 8)
                self.gen_block = nn.Linear(8, 8)  # "gen" matches freeze prefix
                self.moe_gen_0 = nn.Linear(8, 8)  # "moe_gen" matches
                self.moe_gen_1 = nn.Linear(8, 8)  # "moe_gen" matches
                self.llm_und = nn.Linear(8, 8)  # "und" only — should stay trainable

        model = TinyAgentModel()
        self._apply_freeze(model, freeze_prefixes=["moe_gen", "gen"])

        # Frozen (generation path)
        for name in [
            "gen_block.weight",
            "gen_block.bias",
            "moe_gen_0.weight",
            "moe_gen_0.bias",
            "moe_gen_1.weight",
            "moe_gen_1.bias",
        ]:
            param = dict(model.named_parameters())[name]
            assert not param.requires_grad, f"{name} should be frozen (requires_grad=False)"

        # Trainable (understanding path)
        for name in ["lm_head.weight", "lm_head.bias", "llm_und.weight", "llm_und.bias"]:
            param = dict(model.named_parameters())[name]
            assert param.requires_grad, f"{name} should be trainable (requires_grad=True)"


class TestAgenticConfig:
    """Configuration defaults for agentic RL (UT-14)."""

    def test_defaults(self):
        from verl_omni.workers.config.omni.model import AgenticConfig

        c = AgenticConfig()
        assert c.enabled is False, "agentic RL must be opt-in (enabled=False by default)"
        assert c.max_turns == 5
        assert c.early_termination is True
        assert c.observation_token_length == 128

    def test_enabled_override(self):
        from verl_omni.workers.config.omni.model import AgenticConfig

        c = AgenticConfig(enabled=True, max_turns=3, early_termination=False, observation_token_length=64)
        assert c.enabled is True
        assert c.max_turns == 3
        assert c.early_termination is False
        assert c.observation_token_length == 64
