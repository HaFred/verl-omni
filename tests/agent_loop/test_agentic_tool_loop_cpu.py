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
import pytest
import torch
from PIL import Image
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.tools.function_tool import get_function_tool

from verl_omni.agent_loop.agentic_metrics_manager import (
    AgenticMetricsAgentLoopManager,
    _extract_generate_image_prompts,
    _materialize_rollout_images,
    aggregate_agentic_reward_metrics,
    split_assistant_rollouts,
    split_rollout_turns,
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
        assert "Qwen/Qwen3.5-2B" in recipe
        assert "multi_turn.format=${TOOL_PARSER_FORMAT}" in recipe
        assert '"qwen3_coder"' in recipe and '"hermes"' in recipe
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
        from verl_omni.agent_loop.agentic_trajectory_context import clear_good_enough_yes_reached

        clear_good_enough_yes_reached()
        monkeypatch.delenv("AGENTIC_DIFFUSION_TOOL_URL", raising=False)
        monkeypatch.delenv("AGENTIC_QWEN_IMAGE_URL", raising=False)
        monkeypatch.delenv("AGENTIC_VLLM_OMNI_URL", raising=False)
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

    def test_generate_blocked_after_good_enough_yes(self, monkeypatch, tmp_path):
        from verl_omni.agent_loop.agentic_trajectory_context import (
            clear_good_enough_yes_reached,
            set_good_enough_yes_reached,
        )

        clear_good_enough_yes_reached()
        monkeypatch.setenv("AGENTIC_BLOCK_GENERATE_AFTER_YES", "1")
        monkeypatch.delenv("AGENTIC_VLLM_OMNI_URL", raising=False)
        monkeypatch.delenv("AGENTIC_QWEN_IMAGE_URL", raising=False)
        monkeypatch.delenv("AGENTIC_DIFFUSION_TOOL_URL", raising=False)
        monkeypatch.delenv("AGENTIC_LANCE_SERVER_URL", raising=False)
        monkeypatch.setenv("AGENTIC_DIFFUSION_IMAGE_DIR", str(tmp_path / "rollout_images"))
        set_good_enough_yes_reached(True)
        response, reward, metrics = generate_image("should not run diffusion")
        assert "agentic_block_generate_after_yes=1" in response.text
        assert metrics["blocked_after_yes"] == 1
        assert metrics["diffusion_backend"] == "blocked_after_yes"
        assert metrics["num_images"] == 0
        assert reward == 0.0

    def test_qwen_image_backend_has_priority(self, monkeypatch):
        import verl_omni.agent_loop.diffusion_tool as module
        from verl_omni.agent_loop.agentic_trajectory_context import clear_good_enough_yes_reached

        clear_good_enough_yes_reached()
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
            "reward_tool_call": np.array([0.0, 1.0]),
            "reward_done": np.array([0.0, 1.0]),
            "reward_correctness": np.array([0.72, 0.88]),
            "reward_aesthetics": np.array([0.65, 0.95]),
            # Facet keys must not appear under agentic_reward/* (scalar mix only).
            "reward_correctness_subject_entities": np.array([0.8, 0.9]),
            "reward_aesthetics_composition": np.array([0.7, 0.85]),
            "method": np.array(["a", "b"]),
        }
    )
    assert metrics["agentic_reward/tool_call/mean"] == 0.5
    assert metrics["agentic_reward/done/mean"] == 0.5
    assert metrics["agentic_reward/correctness/mean"] == 0.80
    assert metrics["agentic_reward/aesthetics/min"] == 0.65
    assert "agentic_reward/correctness_subject_entities/max" not in metrics
    assert "agentic_reward/aesthetics_composition/mean" not in metrics
    assert len(metrics) == 12  # 4 components × mean/min/max
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
    turns = split_rollout_turns(token_ids, response_mask, Tokenizer())
    assert turns[0]["turn"] == 1 and turns[0]["turn_prompt"] == ""
    assert turns[1]["turn"] == 2 and turns[1]["turn_prompt"] == "OBS"
    assert turns[1]["decode"] == "reflect"


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
            # verl stores the scalar on the final valid response token.
            "rm_scores": torch.tensor([[0.0, 0.0, 0.75]]),
        },
        non_tensor_batch={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "Generate a cat"}]],
                dtype=object,
            ),
            "index": np.array([3]),
            "reward_tool_call": np.array([1.0]),
            "reward_done": np.array([0.0]),
            "reward_correctness": np.array([0.0]),
            "reward_aesthetics": np.array([0.0]),
            "num_hermes_tool_calls": np.array([1]),
            "num_generate_image_prompts": np.array([1]),
            "num_judge_image_calls": np.array([0]),
            "protocol_ok": np.array([0]),
            "rollout_valid": np.array([1]),
        },
    )
    manager = AgenticMetricsAgentLoopManager.__new__(AgenticMetricsAgentLoopManager)
    manager._monitor_tokenizer = Tokenizer()
    monkeypatch.setenv("AGENTIC_DIFFUSION_IMAGE_DIR", str(tmp_path / "rollout_images"))

    manager._dump_raw_rollouts(None, output, 1)

    monitor = (tmp_path / "hermes_actions" / "step_000001.txt").read_text()
    assert "user_prompt: Generate a cat" in monitor
    assert "first" in monitor and "tool" in monitor
    assert "turn_1_prompt:" in monitor
    assert "turn_2_prompt:" in monitor
    assert "OBS" in monitor
    assert "Action:" not in monitor
    assert "Used:" not in monitor
    assert "forced=" not in monitor
    assert "mode=" not in monitor

    traj = json.loads((tmp_path / "rollout_trajectories" / "step_000001" / "sample_3.00.json").read_text())
    assert traj["rollout_turns"][0]["turn_prompt"] == "Generate a cat"
    assert traj["rollout_turns"][1]["turn_prompt"] == "OBS"
    assert traj["rollout_turns"][0]["decode"] == "first"

    action = json.loads((tmp_path / "hermes_actions" / "step_000001.jsonl").read_text())
    assert all("decode" not in turn for turn in action["rollout_turns"])
    assert action["reward_metrics"] == {
        "score": 0.75,
        "reward_tool_call": 1.0,
        "reward_done": 0.0,
        "reward_correctness": 0.0,
        "reward_aesthetics": 0.0,
        "num_hermes_tool_calls": 1,
        "num_generate_image_prompts": 1,
        "num_judge_image_calls": 0,
        "protocol_ok": 0,
        "rollout_valid": 1,
    }


def test_qwen35_xml_monitor_counts_tools_and_sums_token_reward(monkeypatch, tmp_path):
    class Tokenizer:
        @staticmethod
        def decode(ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(chr(i) for i in ids if i)

    generate_call = """<tool_call>
<function=generate_image>
<parameter=prompt>
a cat
</parameter>
</function>
</tool_call>"""
    generate_obs = "agentic_tool ok=1 path=/tmp/cat.png"
    rewrite_call = """Reflection: soft edges; rewrite sharper.
<tool_call>
<function=generate_image>
<parameter=prompt>
a detailed cat, sharp focus
</parameter>
</function>
</tool_call>"""
    rewrite_obs = "agentic_tool ok=1 path=/tmp/cat2.png"
    final = "Reflection: sharp detailed cat matches. Done."
    parts = [
        (generate_call, 1),
        (generate_obs, 0),
        (rewrite_call, 1),
        (rewrite_obs, 0),
        (final, 1),
    ]
    text = "".join(part for part, _ in parts)
    ids = [ord(c) for c in text]
    response_mask = [mask for part, mask in parts for _ in part]
    rm_scores = torch.zeros((1, len(ids)))
    rm_scores[0, -1] = 0.8
    output = SimpleNamespace(
        batch={
            "responses": torch.tensor([ids]),
            "response_mask": torch.tensor([response_mask], dtype=torch.long),
            "rm_scores": rm_scores,
        },
        non_tensor_batch={
            "raw_prompt": np.array([[{"role": "user", "content": "Generate a cat"}]], dtype=object),
            "index": np.array([0]),
        },
    )
    manager = AgenticMetricsAgentLoopManager.__new__(AgenticMetricsAgentLoopManager)
    manager._monitor_tokenizer = Tokenizer()
    monkeypatch.setenv("AGENTIC_DIFFUSION_IMAGE_DIR", str(tmp_path / "rollout_images"))

    manager._dump_raw_rollouts(None, output, 1)

    action = json.loads((tmp_path / "hermes_actions" / "step_000001.jsonl").read_text())
    assert action["num_tool_calls_executed"] == 2
    assert action["num_voluntary_hermes"] == 2
    assert action["reward_metrics"]["score"] == pytest.approx(0.8)
    assert _extract_generate_image_prompts(text) == ["a cat", "a detailed cat, sharp focus"]


def test_tool_artifacts_are_indexed_in_direct_step_sample_layout(tmp_path):
    target_dir = tmp_path / "rollout_images" / "step_000001/sample_3.00"
    target_dir.mkdir(parents=True)
    image = target_dir / "image_00.png"
    image.write_bytes(b"png")
    (target_dir / "meta.json").write_text('{"calls": [{"file": "image_00.png"}]}')

    image_paths = _materialize_rollout_images(
        decoded_response="tool result agentic_tool ok=1",
        run_dir=tmp_path,
        relpath="step_000001/sample_3.00",
        user_prompt="Generate a cat",
    )

    assert image_paths == [str(image)]
    assert image.read_bytes() == b"png"
    meta = json.loads((target_dir / "meta.json").read_text())
    assert meta["calls"] == [{"file": "image_00.png"}]
    assert meta["source"] == "direct_tool_write"


def test_two_turn_tool_images_are_indexed_without_posthoc_move(tmp_path):
    target_dir = tmp_path / "rollout_images" / "step_000001/sample_0.00"
    target_dir.mkdir(parents=True)
    img0 = target_dir / "image_00.png"
    img1 = target_dir / "image_01.png"
    img0.write_bytes(b"png0")
    img1.write_bytes(b"png1")
    prompt0 = "a fluffy white cat wearing a blue knitted hat"
    prompt1 = "a fluffy white cat wearing a bright blue knitted hat with pom-pom"

    decoded = (
        f"turn1 <tool_call>\n"
        f'{{"name": "generate_image", "arguments": {{"prompt": "{prompt0}"}}}}\n'
        f"</tool_call>\n"
        f"turn2 Reflection: darker\n"
        f"<tool_call>\n"
        f'{{"name": "generate_image", "arguments": {{"prompt": "{prompt1}"}}}}\n'
        f"</tool_call>\n"
    )
    image_paths = _materialize_rollout_images(
        decoded_response=decoded,
        run_dir=tmp_path,
        relpath="step_000001/sample_0.00",
        user_prompt="Generate an image of a cat wearing a blue hat",
    )
    assert image_paths == [str(img0), str(img1)]
    assert img0.read_bytes() == b"png0"
    assert img1.read_bytes() == b"png1"
    meta = json.loads((target_dir / "meta.json").read_text())
    assert meta["tool_prompts"] == [prompt0, prompt1]
    assert meta["source"] == "direct_tool_write"
