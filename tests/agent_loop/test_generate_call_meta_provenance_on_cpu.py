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
"""CPU tests for generate_image meta.json reflection provenance."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_context(monkeypatch):
    """Load trajectory context without executing ``verl_omni.__init__`` (CUDA)."""
    for package in ("verl_omni", "verl_omni.agent_loop"):
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    spec = importlib.util.spec_from_file_location(
        "verl_omni.agent_loop.image_gen_trajectory_context",
        _ROOT / "verl_omni/agent_loop/image_gen_trajectory_context.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "verl_omni.agent_loop.image_gen_trajectory_context", module)
    spec.loader.exec_module(module)
    return module


def _reset(ctx) -> None:
    ctx.clear_pending_generate_provenance()
    ctx.clear_latest_tool_image_for_active_rollout()
    ctx.clear_good_enough_yes_reached()
    with ctx._artifact_registry_lock:
        ctx._artifact_registry.clear()
        ctx._artifact_by_id.clear()
        ctx._latest_image_by_rollout.clear()


def test_first_generate_meta_is_initial(monkeypatch) -> None:
    ctx = _load_context(monkeypatch)
    _reset(ctx)
    path_tokens = ctx.set_active_trajectory_relpath("step_000001/sample_1")
    prompt_token = ctx.set_active_user_prompt("一张海报")
    try:
        ctx.merge_pending_generate_provenance(
            {
                "model_decode": '<tool_call>{"name":"generate_image"}</tool_call>',
                "llm_prompt": "a blue poster",
            }
        )
        meta = ctx.build_generate_call_meta(prompt="a blue poster", user_prompt="一张海报")
        assert meta["call_role"] == "initial"
        assert meta["controlled_by_reflection"] is False
        assert meta["rewritten_prompt"] == ""
        assert meta["prev_tool_prompt"] == ""
        assert meta["source_image_for_reflection"] == ""
        assert meta["content_source"] == "initial"
        assert meta["tool_prompt"] == "a blue poster"
        assert meta["llm_prompt"] == "a blue poster"
        assert "generate_image" in meta["model_decode"]
        assert meta["user_prompt"] == "一张海报"
    finally:
        ctx.reset_active_user_prompt(prompt_token)
        ctx.reset_active_trajectory_relpath(path_tokens)
        _reset(ctx)


def test_second_generate_meta_records_rewrite(monkeypatch, tmp_path) -> None:
    ctx = _load_context(monkeypatch)
    _reset(ctx)
    relpath = "step_000001/sample_2"
    path_tokens = ctx.set_active_trajectory_relpath(relpath)
    prompt_token = ctx.set_active_user_prompt("一张海报")
    png = tmp_path / "image_00_abc.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        ctx.register_tool_artifact(
            prompt="prompt v0",
            paths=[str(png)],
            backend="vllm_omni",
            artifact_id="aaaaaaaaaaaa",
            trajectory_relpath=relpath,
        )
        ctx.set_latest_tool_image_path(str(png))
        reflection = "Reflection: good_enough=NO. Rewriting next. agentic_forced_reflection=1"
        ctx.set_pending_generate_provenance(
            {
                "reflection": reflection,
                "llm_reflection": reflection,
                "controlled_by_reflection": True,
                "call_role": "rewrite",
                "model_decode": "I'll revise.\n<tool_call>..</tool_call>",
                "llm_prompt": "prompt v1 legible",
            }
        )
        meta = ctx.build_generate_call_meta(prompt="prompt v1 legible", user_prompt="一张海报")
        assert meta["call_role"] == "rewrite"
        assert meta["controlled_by_reflection"] is True
        assert meta["prev_tool_prompt"] == "prompt v0"
        assert meta["source_image_for_reflection"] == str(png)
        assert meta["rewritten_prompt"] == "prompt v1 legible"
        assert meta["image_generated_from_reflected_prompt"] is True
        assert meta["tool_prompt_equals_rewritten_prompt"] is True
        assert meta["content_source"] == "reflection"
        assert meta["reflection"] == reflection
        assert meta["llm_reflection"] == reflection
        assert meta["llm_prompt"] == "prompt v1 legible"
        assert meta["model_decode"].startswith("I'll revise")
        assert meta["tool_prompt"] == "prompt v1 legible"
        assert ctx.get_pending_generate_provenance() is None
    finally:
        ctx.reset_active_user_prompt(prompt_token)
        ctx.reset_active_trajectory_relpath(path_tokens)
        _reset(ctx)
