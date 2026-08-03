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

"""CPU unit checks formerly embedded as ST-2 / ST-3 in the GPU smoke shell.

ST-1 (1-step Lance GRPO) remains the GPU e2e in
``tests/special_e2e/run_agentic_grpo_lance.sh``.

AC2 / former ST-2: Mode (2a) keeps diffusion outside the actor optimizer via
stock ``ToolAgentLoop`` + function tool (not a selective MoT freeze in-ckpt).
AC3 / former ST-3: single-turn FlowGRPO entrypoints stay importable and
``ray_diffusion_trainer.py`` has no agentic branches.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMode2aDiffusionOutsideActor:
    """AC2 — diffusion is an external tool, not an actor FSDP submodule."""

    def test_recipe_wires_external_function_tool(self):
        recipe = (REPO_ROOT / "examples/agenticrpco_trainer/lance/agentic_grpo_overrides.sh").read_text()
        assert "default_agent_loop=tool_agent" in recipe
        assert "function_tool_path=verl_omni/agent_loop/diffusion_tool.py" in recipe
        assert "agent_loop_config_path=null" in recipe
        assert "algorithm.adv_estimator=grpo" in recipe
        assert "multi_turn.enable=true" in recipe

    def test_no_custom_agentic_fsdp_engine_module(self):
        # PR1 removed AgenticLLMFSDPEngine; stock HF FSDP path only.
        engine_root = REPO_ROOT / "verl_omni/workers/engine"
        hits = [p for p in engine_root.rglob("*agentic*.py") if p.is_file()]
        assert hits == [], f"unexpected agentic engine files: {hits}"


class TestFlowGrpoBackwardCompat:
    """AC3 — former ST-3: FlowGRPO single-turn path unaffected by Mode (2a)."""

    def test_main_diffusion_importable(self):
        from verl_omni.trainer import main_diffusion  # noqa: F401

    def test_diffusion_algo_config_defaults_flow_grpo(self):
        from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig

        assert DiffusionAlgoConfig().adv_estimator == "flow_grpo"

    def test_single_turn_agent_loop_importable(self):
        from verl_omni.agent_loop import DiffusionSingleTurnAgentLoop

        assert DiffusionSingleTurnAgentLoop is not None

    def test_flow_grpo_adv_estimator_registered(self):
        from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_adv_estimator_fn

        assert get_diffusion_adv_estimator_fn("flow_grpo") is not None

    def test_ray_diffusion_trainer_has_no_agentic_branches(self):
        path = REPO_ROOT / "verl_omni/trainer/diffusion/ray_diffusion_trainer.py"
        source = path.read_text()
        # Parse ensures the file is valid Python; string scan matches former ST-3e.
        ast.parse(source)
        assert "is_agentic" not in source
        assert "agentic_grpo" not in source
