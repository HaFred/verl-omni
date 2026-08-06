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

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.tools.function_tool import get_function_tool

from verl_omni.agent_loop.agentic_metrics_manager import (
    AgenticMetricsAgentLoopManager,
    _materialize_rollout_images,
    aggregate_agentic_reward_metrics,
    split_assistant_rollouts,
)
from verl_omni.agent_loop.diffusion_tool import DIFFUSION_TOOL_SCHEMA, generate_image


class TestTrainMaskContract:
    def test_all_agent_turns_train_and_tool_observations_do_not(self):
        # Stock ToolAgentLoop contract:
        # assistant turn 0 | tool observation | assistant turn 1
        response_mask = [1, 1, 1] + [0, 0] + [1, 1]
        assert response_mask == [1, 1, 1, 0, 0, 1, 1]


class TestStockToolAgentWiring:
    def test_recipe_uses_stock_tool_agent(self):
        root = Path(__file__).resolve().parents[2]
        recipe = (root / "examples/agenticrpco_trainer/agent_llm/run_agentic_grpo.sh").read_text()
        assert "default_agent_loop=tool_agent" in recipe
        assert "agentic_force_tool_agent" not in recipe
        assert "AGENTIC_FORCE_" not in recipe
        assert "TEACHER_FORCE_HERMES" not in recipe
        assert (
            "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager."
            in recipe
            or "agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager." in recipe
        )
        assert "multi_turn.enable=true" in recipe
        assert "Qwen/Qwen3-VL-2B-Thinking" in recipe
        assert "multi_turn.format=hermes" in recipe
        assert 'AGENTIC_DIFFUSION_ATTACH_IMAGE="${AGENTIC_DIFFUSION_ATTACH_IMAGE:-1}"' in recipe
        assert "Installed tool-aware chat template" not in recipe
        assert "function_tool_path=verl_omni/agent_loop/diffusion_tool.py" in recipe

    def test_stock_tool_agent_is_upstream(self):
        assert ToolAgentLoop.__module__ == "verl.experimental.agent_loop.tool_agent_loop"

    def test_stock_hermes_parser_accepts_thinking_prefix(self):
        text = (
            "<think>\nI should use the image tool before answering.\n</think>\n"
            '<tool_call>{"name":"generate_image","arguments":{"prompt":"a blue cat"}}</tool_call>'
        )

        class Tokenizer:
            @staticmethod
            def decode(_):
                return text

        _, calls = asyncio.run(HermesToolParser(Tokenizer()).extract_tool_calls([1, 2, 3]))
        assert len(calls) == 1
        assert calls[0].name == "generate_image"
        assert json.loads(calls[0].arguments) == {"prompt": "a blue cat"}

    def test_diffusion_function_tool_registered(self):
        tool = get_function_tool("generate_image")
        assert tool.fn is generate_image
        assert tool.tool_schema.function.name == DIFFUSION_TOOL_SCHEMA["function"]["name"]

    def test_diffusion_tool_stub_without_endpoint(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENTIC_DIFFUSION_TOOL_URL", raising=False)
        monkeypatch.delenv("AGENTIC_QWEN_IMAGE_URL", raising=False)
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

    def test_qwen_image_backend_has_priority(self, monkeypatch):
        import verl_omni.agent_loop.diffusion_tool as module

        seen = {}

        def fake_http(prompt, endpoint, *, backend="http"):
            seen.update(prompt=prompt, endpoint=endpoint, backend=backend)
            return SimpleNamespace(text="ok", image=None), 0.0, {"diffusion_backend": backend}

        monkeypatch.setattr(module, "_call_generic_http", fake_http)
        monkeypatch.setenv("AGENTIC_QWEN_IMAGE_URL", "http://127.0.0.1:8092/generate")
        monkeypatch.setenv("AGENTIC_DIFFUSION_TOOL_URL", "http://wrong.example/generate")
        _, _, metrics = generate_image("a reflective silver robot")

        assert seen == {
            "prompt": "a reflective silver robot",
            "endpoint": "http://127.0.0.1:8092/generate",
            "backend": "qwen_image",
        }
        assert metrics["diffusion_backend"] == "qwen_image"

    def test_qwen_image_pixels_are_attached_for_vlm(self, monkeypatch, tmp_path):
        import verl_omni.agent_loop.diffusion_tool as module

        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (20, 80, 200)).save(buffer, format="PNG")
        payload = {
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "text": "candidate generated",
        }

        class FakeHTTPResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            @staticmethod
            def read():
                return json.dumps(payload).encode()

        monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: FakeHTTPResponse())
        monkeypatch.setenv("AGENTIC_QWEN_IMAGE_URL", "http://127.0.0.1:8092/generate")
        monkeypatch.setenv("AGENTIC_DIFFUSION_ATTACH_IMAGE", "1")
        monkeypatch.setenv("AGENTIC_DIFFUSION_IMAGE_DIR", str(tmp_path / "rollout_images"))

        response, _, metrics = generate_image("a blue square")

        assert response.image and response.image[0].size == (32, 32)
        assert metrics["diffusion_backend"] == "qwen_image"
        assert metrics["num_images"] == 1
        assert Path(metrics["image_paths"][0]).is_file()


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


def test_rollout_monitor_splits_model_turns_around_tool_observation():
    class Tokenizer:
        @staticmethod
        def decode(ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(chr(i) for i in ids)

    token_ids = [ord(c) for c in "callOBSreflect"]
    response_mask = [1] * 4 + [0] * 3 + [1] * 7
    assert split_assistant_rollouts(token_ids, response_mask, Tokenizer()) == ["call", "reflect"]


def test_rollout_monitor_writes_only_prompt_and_raw_decodes(monkeypatch, tmp_path):
    class Tokenizer:
        @staticmethod
        def decode(ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(chr(i) for i in ids if i)

    text = "firstOBStool"
    output = SimpleNamespace(
        batch={
            "responses": torch.tensor([[ord(c) for c in text]]),
            "response_mask": torch.tensor([[1] * 5 + [0] * 3 + [1] * 4]),
        },
        non_tensor_batch={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "Generate a cat"}]],
                dtype=object,
            ),
            "index": np.array([3]),
        },
    )
    manager = AgenticMetricsAgentLoopManager.__new__(AgenticMetricsAgentLoopManager)
    manager._monitor_tokenizer = Tokenizer()
    monkeypatch.setenv("AGENTIC_DIFFUSION_IMAGE_DIR", str(tmp_path / "rollout_images"))

    manager._dump_raw_rollouts(None, output, 1)

    monitor = (tmp_path / "hermes_actions" / "step_000001.txt").read_text()
    assert "user_prompt: Generate a cat" in monitor
    assert "first" in monitor and "tool" in monitor
    assert "Action:" not in monitor
    assert "Used:" not in monitor
    assert "forced=" not in monitor
    assert "mode=" not in monitor


def test_stock_tool_artifacts_are_copied_to_step_sample_layout(tmp_path):
    source_dir = tmp_path / "rollout_images" / "call_abc"
    source_dir.mkdir(parents=True)
    source = source_dir / "image_00.png"
    source.write_bytes(b"png")
    (source_dir / "meta.json").write_text("{}")

    copied = _materialize_rollout_images(
        decoded_response=f"tool result path={source} agentic_tool ok=1",
        run_dir=tmp_path,
        relpath="step_000001/sample_3.00",
        user_prompt="Generate a cat",
    )

    assert copied == [str(tmp_path / "rollout_images/step_000001/sample_3.00/image_00.png")]
    assert Path(copied[0]).read_bytes() == b"png"
