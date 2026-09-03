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
"""CPU tests for Bagel Co-RL IDs, flatten, GEN cap, and serial episode."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_lib():
    path = Path(__file__).resolve().parents[2] / "verl_omni" / "agent_loop" / "bagel_corl_lib.py"
    spec = importlib.util.spec_from_file_location("bagel_corl_lib_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


lib = _load_lib()


def test_hermes_generate_image_is_bagel_not_qwen():
    text = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "a cat"}}</tool_call>'
    assert lib.und_turn_kind(text) == "generate_image"
    assert lib.parse_hermes_tool_call(text)["name"] == "generate_image"
    assert lib.GENERATE_IMAGE_TOOL_SCHEMA["function"]["name"] == "generate_image"


def test_unsupported_tool_is_fail_closed():
    text = '<tool_call>{"name": "judge_image", "arguments": {}}</tool_call>'
    with pytest.raises(ValueError, match="unsupported tool"):
        lib.und_turn_kind(text)


def test_jxk_ids_never_group_flowgrpo_across_und_prompts():
    a = lib.bind_episode_ids(dataset_task_uid="taskA", gen_call_id="callA")
    b = lib.bind_episode_ids(dataset_task_uid="taskB", gen_call_id="callB")
    assert a["und_group_uid"] == "taskA"
    assert b["und_group_uid"] == "taskB"
    assert a["gen_group_uid"] == "callA"
    assert a["gen_group_uid"] != b["gen_group_uid"]
    assert lib.gen_sample_uid("callA", 0) == "callA:0"


def _episode(*, uid: str, gens, used_image: bool):
    return lib.EpisodeRollout(
        und_group_uid=uid,
        episode_uid=uid + "-ep",
        policy_version=1,
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        turns=2,
        gen_samples=gens,
        used_image_credit=used_image,
    )


def test_flatten_reflection_only_has_zero_gen_rows():
    ep = _episode(uid="t0", gens=[], used_image=False)
    result = lib.flatten_multiturn_rollouts([ep], expected_k=2)
    assert len(result.und_batch) == 1
    assert result.gen_batch == []
    assert result.metrics["und/no_image_credit"] == 1.0
    assert result.metrics["gen/skipped_no_groups"] == 1.0


def test_flatten_drops_incomplete_k_groups():
    call = "g1"
    samples = [
        lib.GenSample(
            gen_sample_uid=lib.gen_sample_uid(call, 0),
            gen_group_uid=call,
            seed_index=0,
            valid=True,
            prompt_token_ids=[1],
            rm_score=1.0,
        )
    ]
    result = lib.flatten_multiturn_rollouts([_episode(uid="t0", gens=samples, used_image=True)], expected_k=2)
    assert result.gen_batch == []
    assert result.metrics["gen/dropped_incomplete_groups"] == 1.0


def test_flatten_complete_k_group_keeps_prompt_token_ids():
    call = "g1"
    samples = [
        lib.GenSample(
            gen_sample_uid=lib.gen_sample_uid(call, i),
            gen_group_uid=call,
            seed_index=i,
            valid=True,
            prompt_token_ids=[9, 8],
            rm_score=float(i + 1),
            all_latents="latents",
        )
        for i in range(2)
    ]
    result = lib.flatten_multiturn_rollouts([_episode(uid="t0", gens=samples, used_image=True)], expected_k=2)
    assert len(result.gen_batch) == 2
    assert result.gen_batch[0]["prompt_token_ids"] == [9, 8]
    stripped = lib.strip_pixels_for_actor({"prompt_embeds": 1, "prompt_token_ids": [1], "images": []})
    assert "prompt_embeds" not in stripped


def test_max_generate_passes_one_refuses_second_gen():
    tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=1)

    async def _run():
        await tool(prompt="a", prompt_token_ids=[1], gen_call_id="c1", seeds=[0, 1])
        await tool(prompt="b", prompt_token_ids=[1], gen_call_id="c2", seeds=[0, 1])

    with pytest.raises(lib.GenerateImageCapError, match="second"):
        asyncio.run(_run())


def test_serial_episode_await_und_then_gen():
    order: list[str] = []

    async def und_decode(**kwargs):
        order.append("und")
        return {
            "token_ids": [7],
            "text": '<tool_call>{"name": "generate_image", "arguments": {"prompt": "x"}}</tool_call>',
            "done_token_ids": [8],
        }

    async def gen_fn(**kwargs):
        order.append("gen")
        return [{"valid": True, "image_path": "/tmp/a.png"} for _ in kwargs["seeds"]]

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, generate_fn=gen_fn)
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=3,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
        )

    episode = asyncio.run(_run())
    assert order == ["und", "gen"]
    assert len(episode.gen_samples) == 2
    assert episode.response_mask[-1] == 1


def _load_config_mod():
    path = Path(__file__).resolve().parents[2] / "verl_omni" / "utils" / "config.py"
    spec = importlib.util.spec_from_file_location("omni_config_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_j_equals_two_k_config_fail_closed():
    from omegaconf import OmegaConf

    cfg_mod = _load_config_mod()
    cfg = OmegaConf.create(
        {
            "trainer": {"resume_mode": "disable", "v1": {"trainer_mode": "bagel_corl_sync"}},
            "actor_rollout_ref": {
                "model": {"path": "/models/ByteDance-Seed/BAGEL-7B-MoT", "lora_rank": 64},
                "rollout": {"n": 8, "agent": {"gen_samples_per_call": 3, "max_generate_passes": 1}},
            },
        }
    )
    with pytest.raises(ValueError, match="J=2K"):
        cfg_mod.validate_bagel_corl_config(cfg)
