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

from pathlib import Path

import numpy as np
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.function_tool import get_function_tool

from verl_omni.agent_loop.agentic_force_tool_agent_loop import AgenticForceToolAgentLoop
from verl_omni.agent_loop.agentic_metrics_manager import aggregate_agentic_reward_metrics
from verl_omni.agent_loop.diffusion_tool import DIFFUSION_TOOL_SCHEMA, generate_image


class TestTrainMaskContract:
    def test_all_agent_turns_train_and_tool_observations_do_not(self):
        # Stock ToolAgentLoop contract:
        # assistant turn 0 | tool observation | assistant turn 1
        response_mask = [1, 1, 1] + [0, 0] + [1, 1]
        assert response_mask == [1, 1, 1, 0, 0, 1, 1]


class TestStockToolAgentWiring:
    def test_recipe_uses_stock_manager_with_tool_loop_subclass(self):
        root = Path(__file__).resolve().parents[2]
        recipe = (root / "examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh").read_text()
        assert "default_agent_loop=agentic_force_tool_agent" in recipe
        assert (
            "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager."
            in recipe
            or "agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager." in recipe
        )
        assert "multi_turn.enable=true" in recipe
        assert "function_tool_path=verl_omni/agent_loop/diffusion_tool.py" in recipe

    def test_agentic_loop_only_specializes_upstream_tool_loop(self):
        assert ToolAgentLoop.__module__ == "verl.experimental.agent_loop.tool_agent_loop"
        assert issubclass(AgenticForceToolAgentLoop, ToolAgentLoop)

    def test_diffusion_function_tool_registered(self):
        tool = get_function_tool("generate_image")
        assert tool.fn is generate_image
        assert tool.tool_schema.function.name == DIFFUSION_TOOL_SCHEMA["function"]["name"]

    def test_diffusion_tool_stub_without_endpoint(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENTIC_DIFFUSION_TOOL_URL", raising=False)
        monkeypatch.delenv("AGENTIC_LANCE_SERVER_URL", raising=False)
        monkeypatch.setenv("AGENTIC_DIFFUSION_IMAGE_DIR", str(tmp_path / "rollout_images"))
        response, reward, metrics = generate_image("a blue hat")
        assert response.image is None
        assert "stub diffusion result" in response.text
        assert reward == 0.0
        assert metrics["tool_stubbed"] is True
        assert metrics["diffusion_backend"] == "stub"
        assert metrics["num_images"] == 0
        assert metrics["image_paths"]
        assert Path(metrics["image_paths"][0]).name == "STUB_NO_IMAGE.txt"


def test_agentic_reward_components_are_aggregated_for_wandb():
    metrics = aggregate_agentic_reward_metrics(
        {
            "reward_format": np.array([0.0, 1.0]),
            "reward_reflection": np.array([0.4, 0.8]),
            "reward_tool_usage": np.array([0.5, 1.0]),
            "reward_result": np.array([0.0, 0.5]),
            "method": np.array(["a", "b"]),
        }
    )
    assert metrics["agentic_reward/format/mean"] == 0.5
    assert metrics["agentic_reward/reflection/min"] == 0.4
    assert metrics["agentic_reward/tool_usage/max"] == 1.0
    assert metrics["agentic_reward/result/mean"] == 0.25
    assert all("method" not in key for key in metrics)


class TestOverfitStableTeacher:
    def test_stable_prompts_match_fewshot_shape(self):
        from verl_omni.agent_loop.agentic_force_tool_agent_loop import (
            _hermes_generate_image_text,
            _stable_overfit_prompt_and_reflection,
        )

        p0, r0, m0 = _stable_overfit_prompt_and_reflection(
            user_task="Generate an image of a cat wearing a blue hat",
            executed=0,
            prev_prompts=[],
        )
        assert r0 == ""
        assert "cat wearing a blue hat" in p0
        assert m0["stage"] == "overfit_initial"
        t0 = _hermes_generate_image_text(p0, reflect=None)
        assert t0.startswith("<tool_call>")
        assert "Reflection:" not in t0

        p1, r1, m1 = _stable_overfit_prompt_and_reflection(
            user_task="Generate an image of a cat wearing a blue hat",
            executed=1,
            prev_prompts=[p0],
        )
        assert r1.lower().startswith("looking at the generated image")
        assert p1 != p0
        assert "highly detailed" in p1
        assert m1["stage"] == "overfit_reflect_rewrite"
        t1 = _hermes_generate_image_text(p1, reflect=r1)
        assert t1.startswith("Reflection:")
        assert "<tool_call>" in t1
        assert p0 not in t1 or p1 in t1
