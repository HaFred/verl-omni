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
import types
from pathlib import Path

import pytest


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")
    sys.modules[name] = pkg


def _load_by_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_lib():
    root = Path(__file__).resolve().parents[2]
    omni = root / "verl_omni"
    _ensure_pkg("verl_omni", omni)
    _ensure_pkg("verl_omni.agent_loop", omni / "agent_loop")
    _load_by_path(
        "verl_omni.agent_loop.rpco_turn_protocol",
        omni / "agent_loop" / "rpco_turn_protocol.py",
    )
    _load_by_path(
        "verl_omni.agent_loop.image_gen_trajectory_context",
        omni / "agent_loop" / "image_gen_trajectory_context.py",
    )
    return _load_by_path(
        "bagel_corl_lib_isolated",
        omni / "agent_loop" / "bagel_corl_lib.py",
    )


lib = _load_lib()
ctx = sys.modules["verl_omni.agent_loop.image_gen_trajectory_context"]


class _FakeTokenizer:
    """Minimal encode/decode for observation / reflection mask tests."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) + 1000 for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(chr(int(t) - 1000) for t in token_ids)


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
    assert result.und_batch[0]["token_level_scores"] == pytest.approx(1.5)
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


def test_forced_reflection_masks_and_observation_encoding():
    tok = _FakeTokenizer()
    gen_calls = {"n": 0}

    async def und_decode(**kwargs):
        if "Reflection:" in tok.decode(kwargs["response_ids"]):
            return {"token_ids": tok.encode("Done."), "text": "Done."}
        return {
            "token_ids": tok.encode('<tool_call>{"name": "generate_image", "arguments": {"prompt": "cat"}}</tool_call>'),
            "text": '<tool_call>{"name": "generate_image", "arguments": {"prompt": "cat"}}</tool_call>',
        }

    async def gen_fn(**kwargs):
        gen_calls["n"] += 1
        return [{"valid": True, "image_path": "/tmp/obs.png"} for _ in kwargs["seeds"]]

    def score_fn(samples):
        for s in samples:
            s.rm_score = 0.9
            s.good_enough = True
        return samples

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=1, generate_fn=gen_fn)
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
            score_fn=score_fn,
            tokenizer=tok,
        )

    episode = asyncio.run(_run())
    assert gen_calls["n"] == 1
    assert episode.forced_reflection is True
    assert episode.stop_required is True
    assert episode.judge_text is not None
    assert "good_enough=YES" in episode.judge_text
    assert 0 in episode.response_mask
    assert episode.response_mask[-1] == 1
    obs_ids = tok.encode("path=/tmp/obs.png")
    joined = episode.response_ids
    obs_start = None
    for i in range(len(joined) - len(obs_ids) + 1):
        if joined[i : i + len(obs_ids)] == obs_ids:
            obs_start = i
            break
    assert obs_start is not None
    assert episode.response_mask[obs_start : obs_start + len(obs_ids)] == [0] * len(obs_ids)
    done_ids = tok.encode("Done.")
    assert joined[-len(done_ids) :] == done_ids
    assert episode.response_mask[-len(done_ids) :] == [1] * len(done_ids)
    assert episode.response_mask[-len(done_ids) - 1] == 0
    assert episode.und_reward == pytest.approx(0.9)
    flat = lib.flatten_multiturn_rollouts([episode], expected_k=2)
    assert flat.und_batch[0]["token_level_scores"] == pytest.approx(0.9)


def test_good_enough_latch_blocks_second_generate():
    """Latch set after first GEN must block a second generate_image (max_passes=2)."""
    ctx.clear_good_enough_yes_reached()
    tok = _FakeTokenizer()
    gen_calls = {"n": 0}

    async def und_decode(**kwargs):
        return {
            "token_ids": tok.encode('<tool_call>{"name": "generate_image", "arguments": {"prompt": "p"}}</tool_call>'),
            "text": '<tool_call>{"name": "generate_image", "arguments": {"prompt": "p"}}</tool_call>',
        }

    async def gen_fn(**kwargs):
        gen_calls["n"] += 1
        return [{"valid": True, "image_path": f"/tmp/g{gen_calls['n']}.png"} for _ in kwargs["seeds"]]

    def score_fn(samples):
        for s in samples:
            s.rm_score = 0.95
            s.good_enough = True
        return samples

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=2, generate_fn=gen_fn)
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
            score_fn=score_fn,
            tokenizer=tok,
            max_und_turns=6,
        )

    episode = asyncio.run(_run())
    assert gen_calls["n"] == 1
    assert ctx.get_good_enough_yes_reached() is True
    assert episode.stop_required is True

    # Explicit latch-skip: with latch pre-set and clear disabled, GEN must not run.
    ctx.clear_good_enough_yes_reached()
    ctx.set_good_enough_yes_reached(True)
    blocked = {"n": 0}

    async def gen_blocked(**kwargs):
        blocked["n"] += 1
        return [{"valid": True, "image_path": "/tmp/blocked.png"} for _ in kwargs["seeds"]]

    original_clear = lib.clear_good_enough_yes_reached
    lib.clear_good_enough_yes_reached = lambda: None
    try:

        async def _blocked():
            tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=2, generate_fn=gen_blocked)
            return await lib.run_serial_episode(
                dataset_task_uid="blocked",
                policy_version=1,
                prompt_ids=[1],
                und_decode=und_decode,
                generate_tool=tool,
                tokenizer=tok,
                max_und_turns=3,
            )

        blocked_ep = asyncio.run(_blocked())
    finally:
        lib.clear_good_enough_yes_reached = original_clear
        ctx.clear_good_enough_yes_reached()

    assert blocked["n"] == 0
    assert blocked_ep.gen_samples == []


def _load_config_mod():
    root = Path(__file__).resolve().parents[2]
    _ensure_pkg("verl_omni", root / "verl_omni")
    _ensure_pkg("verl_omni.utils", root / "verl_omni" / "utils")
    return _load_by_path(
        "omni_config_isolated",
        root / "verl_omni" / "utils" / "config.py",
    )


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
