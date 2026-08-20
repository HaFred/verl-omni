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
"""CPU tests for binding judge_image to the original user request."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _load_source_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_tools_without_gpu_pipelines(monkeypatch):
    """Load the recipe tool directly without executing ``verl_omni.__init__``."""
    for package in ("verl", "verl.tools", "verl_omni", "verl_omni.agent_loop", "verl_omni.utils"):
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))

    function_tool_module = types.ModuleType("verl.tools.function_tool")
    function_tool_module.function_tool = lambda *_args, **_kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "verl.tools.function_tool", function_tool_module)

    schemas_module = types.ModuleType("verl.tools.schemas")

    class ToolResponse:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    schemas_module.ToolResponse = ToolResponse
    monkeypatch.setitem(sys.modules, "verl.tools.schemas", schemas_module)

    context = _load_source_module(
        monkeypatch,
        "verl_omni.agent_loop.agentic_trajectory_context",
        _ROOT / "verl_omni/agent_loop/agentic_trajectory_context.py",
    )
    _load_source_module(
        monkeypatch,
        "verl_omni.utils.agentic_image_judge_parse",
        _ROOT / "verl_omni/utils/agentic_image_judge_parse.py",
    )
    tools = _load_source_module(
        monkeypatch,
        "agentic_function_tools_under_test",
        _ROOT / "examples/agenticllmgrpo_trainer/function_tools/tools.py",
    )
    return tools, context


def test_live_judge_uses_original_request_for_both_prompt_fields(monkeypatch):
    tools, context = _load_tools_without_gpu_pipelines(monkeypatch)
    original = "A mythological scene from Greek antiquity in ancient vase painting style."
    dataset_prompt = f"{original} Keep any private thinking to AT MOST one short paragraph (≤4 sentences)."
    rewritten = (
        "Zeus emerges from a thunderbolt above mountains, wearing a winged helmet, "
        "with red blue and yellow lines."
    )
    captured = {}

    def fake_call(user_request, image_prompt, vllm_url):
        captured.update(
            user_request=user_request,
            image_prompt=image_prompt,
            vllm_url=vllm_url,
        )
        return "ok", {}

    monkeypatch.setenv("AGENTIC_VLLM_URL", "http://judge.test")
    monkeypatch.setattr(tools, "_call_judge_vllm", fake_call)
    token = context.set_active_user_prompt(dataset_prompt)
    try:
        tools._call_judge_vlm("same as user message", rewritten)
    finally:
        context.reset_active_user_prompt(token)

    assert captured == {
        "user_request": original,
        "image_prompt": original,
        "vllm_url": "http://judge.test",
    }


def test_unbound_judge_retains_explicit_prompt(monkeypatch):
    tools, _ = _load_tools_without_gpu_pipelines(monkeypatch)
    rewritten = "A direct-tool diffusion prompt."

    assert tools._expand_judge_user_request("explicit user request") == "explicit user request"
    assert tools._expand_judge_image_prompt(rewritten) == rewritten
