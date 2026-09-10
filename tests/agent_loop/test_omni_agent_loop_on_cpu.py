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
"""CPU tests for OmniAgentLoop wiring and the dump helpers it delegates to."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

import verl_omni  # noqa: F401
from verl_omni.agent_loop import omni_agent_loop
from verl_omni.agent_loop.omni_agent_loop import OmniAgentLoopManager, OmniAgentLoopWorker
from verl_omni.tools.trajectory import (
    active_trajectory_relpath,
    active_user_prompt,
    reset_active_trajectory_relpath,
    set_active_trajectory_relpath,
)
from verl_omni.utils.agentic.image_gen_rollout_dump import discard_invalid_rollouts
from verl_omni.utils.agentic.image_gen_rollout_parse import (
    extract_generate_image_prompts,
    split_env_blob,
    split_rollout_turns,
    turn_kind,
)


def test_worker_stamps_rollout_kwargs_and_resets_context(monkeypatch):
    captured: dict = {"kwargs": None}

    async def _parent_run(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        del sampling_params, trajectory, agent_name, trace
        captured["kwargs"] = dict(kwargs)
        captured["relpath_during_run"] = active_trajectory_relpath.get()
        captured["user_prompt_during_run"] = active_user_prompt.get()
        return "ok"

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", _parent_run)
    worker = OmniAgentLoopWorker.__new__(OmniAgentLoopWorker)
    assert OmniAgentLoopWorker._AGENTIC_FUNCTION_TOOLS.is_file()

    prior_path = set_active_trajectory_relpath("prior/path")
    prior_prompt = active_user_prompt.set("prior prompt")

    async def _run_then_read_context():
        result = await OmniAgentLoopWorker._run_agent_loop(
            worker,
            {},
            {"step": 7, "sample_index": 3, "rollout_n": 1, "validate": False},
            agent_name="image_gen_tool_agent",
            raw_prompt=[{"role": "user", "content": "draw a cafe poster"}],
        )
        return result, active_trajectory_relpath.get(), active_user_prompt.get()

    try:
        result, path_after, prompt_after = asyncio.run(_run_then_read_context())
    finally:
        active_user_prompt.reset(prior_prompt)
        reset_active_trajectory_relpath(prior_path)

    assert result == "ok"
    assert captured["kwargs"]["_agentic_step"] == 7
    assert captured["kwargs"]["_agentic_validate"] is False
    assert captured["kwargs"]["_agentic_trajectory_relpath"] == "step_000007/sample_3.01"
    assert captured["relpath_during_run"] == "step_000007/sample_3.01"
    assert captured["user_prompt_during_run"] == "draw a cafe poster"
    assert path_after == "prior/path"
    assert prompt_after == "prior prompt"


def test_manager_dumps_before_discarding_invalid_rollouts(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(AgentLoopManager, "generate_sequences", lambda self, prompts: prompts.output)
    monkeypatch.setattr(omni_agent_loop, "dump_raw_rollouts", lambda **kwargs: order.append("dump") or kwargs)
    monkeypatch.setattr(omni_agent_loop, "discard_invalid_rollouts", lambda output: order.append("discard") or output)
    monkeypatch.setattr(omni_agent_loop, "AgenticRewardMetrics", SimpleNamespace(aggregate=lambda batch: {}))

    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager._monitor_tokenizer = object()
    output = SimpleNamespace(non_tensor_batch={})
    prompts = SimpleNamespace(meta_info={"global_steps": 4}, output=output)
    assert OmniAgentLoopManager.generate_sequences(manager, prompts) is output
    assert order == ["dump", "discard"]


def test_turn_kind_stop_rewrite_and_continue():
    judge_no = "VL judge on the last generated image:\n  good_enough =NO\n  agentic_judge ok=1"
    judge_yes = judge_no.replace("good_enough =NO", "good_enough =YES")
    continue_cue = "Reflection: rewrite next. agentic_forced_reflection=1"
    stop_cue = "Reflection: Stop. agentic_stop_decision_required=1 agentic_forced_reflection=1"
    rewrite = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "lion"}}\n</tool_call>'
    done = "Reflection: The image meets the original request. Done.<|im_end|>"
    assert turn_kind(done, judge_yes, stop_cue) == "agent_done_after_forced_reflection"
    assert turn_kind(rewrite, judge_no, continue_cue) == "agent_rewrite_after_forced_reflection"
    assert turn_kind("", judge_yes, stop_cue) == "forced_reflection_stop_cue"
    assert turn_kind(done, judge_no, "") == "agent_reflection_done"


def test_extract_generate_image_prompts_hermes_and_qwen():
    hermes = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a cat"}}\n</tool_call>'
    qwen = "<tool_call>\n<function=generate_image>\n<parameter=prompt>\na dog\n</parameter>\n</function>\n</tool_call>"
    assert extract_generate_image_prompts(hermes) == ["a cat"]
    assert extract_generate_image_prompts(qwen) == ["a dog"]


def test_split_env_blob_and_rollout_turns():
    blob = (
        "<tool_response>\nagentic_tool ok=1 path=/tmp/x.png\n</tool_response>\n"
        "Reflection: rewrite next. agentic_forced_reflection=1"
    )
    prompt, response = split_env_blob(blob)
    assert "agentic_tool ok=1" in prompt
    assert response.startswith("Reflection:")

    class _Tok:
        pad_token_id = 0

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(chr(64 + int(x)) for x in ids)

    turns = split_rollout_turns([1, 2, 3, 4], [1, 1, 0, 0], _Tok())
    assert [turn["decode"] for turn in turns] == ["AB", ""]
    assert turns[1]["turn_prompt"] == "CD"


def test_discard_invalid_rollouts_zeros_mask_but_restores_if_all_invalid():
    class _MaskRow:
        def __init__(self, vals):
            self.vals = list(vals)

        def zero_(self):
            self.vals = [0] * len(self.vals)

        def any(self):
            return any(self.vals)

        def copy_(self, other):
            self.vals = list(other.vals)

    class _Mask:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows),)

        def __getitem__(self, i):
            return self.rows[i]

        def clone(self):
            return _Mask([_MaskRow(row.vals) for row in self.rows])

        def any(self):
            return any(row.any() for row in self.rows)

        def copy_(self, other):
            for dst, src in zip(self.rows, other.rows, strict=True):
                dst.copy_(src)

    mask = _Mask([_MaskRow([1, 1]), _MaskRow([1, 1])])
    discard_invalid_rollouts(
        SimpleNamespace(batch={"response_mask": mask}, non_tensor_batch={"rollout_valid": np.array([1, 0])})
    )
    assert mask.rows[0].vals == [1, 1]
    assert mask.rows[1].vals == [0, 0]

    all_invalid = _Mask([_MaskRow([1, 1]), _MaskRow([1, 0])])
    discard_invalid_rollouts(
        SimpleNamespace(batch={"response_mask": all_invalid}, non_tensor_batch={"rollout_valid": np.array([0, 0])})
    )
    assert all_invalid.rows[0].vals == [1, 1]
    assert all_invalid.rows[1].vals == [1, 0]
