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
"""CPU tests for the e2e agentic artifact layout.

Pins the baseline hierarchy that the PR #409/#411/#412 rebase regressed:

    <e2e_root>/<experiment_name>/            (single level, no double nesting)
    <run>/hermes_actions/step_XXXXXX.{jsonl,txt}
    <run>/rollout_trajectories/step_XXXXXX/sample_<index>[.<nn>].{json,txt}
    <run>/rollout_images/step_XXXXXX/sample_<index>[.<nn>]/
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import verl_omni  # noqa: F401
from verl_omni.agent_loop import omni_agent_loop
from verl_omni.agent_loop.omni_agent_loop import OmniAgentLoopManager
from verl_omni.tools import trajectory
from verl_omni.tools.trajectory import build_trajectory_relpath, resolve_run_dir
from verl_omni.tools.trajectory import locking as traj_locking
from verl_omni.utils.agentic.image_gen_rollout_dump import dump_rollout_artifacts
from verl_omni.utils.agentic_val_viz import resolve_agentic_val_viz_provider


class _Tok:
    """Minimal tokenizer: ``split_rollout_turns`` only needs ``decode``."""

    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return " ".join(str(int(token)) for token in ids)


def _bind(root, run_name="layout_test"):
    cfg = OmegaConf.create(
        {
            "trainer": {"experiment_name": run_name},
            "agentic_image_gen": {"e2e_root": str(root)},
        }
    )
    trajectory.bind_run_artifacts(cfg)


class _PieceTok:
    """Token id -> text piece, so a dumped turn decodes into real tool-call JSON."""

    pad_token_id = 0

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return "".join(self.pieces.get(int(token), "") for token in ids)


def _output(response_ids):
    return SimpleNamespace(
        prompt_ids=[1, 2, 3],
        response_ids=list(response_ids),
        response_mask=[1] * len(response_ids),
        reward_score=0.5,
        extra_fields={"reward_extra_info": {"num_generate_image_prompts": 2, "reward_tool_call": 0.25}},
    )


def test_train_and_val_relpaths_match_baseline_layout():
    assert build_trajectory_relpath(step=5, sample_index=42, rollout_n=0) == "step_000005/sample_42.00"
    # Val omits the group suffix so it can never collide with a train ``.00`` dir.
    assert build_trajectory_relpath(step=5, sample_index=9001, rollout_n=0, validate=True) == "step_000005/sample_9001"
    assert build_trajectory_relpath(step=None, sample_index=None, rollout_n=0) == "step_unknown/sample_unknown.00"


def test_run_dir_is_single_level_under_e2e_root(tmp_path):
    """``e2e_root`` is the parent: paths.py appends ``experiment_name`` itself."""
    _bind(tmp_path, run_name="agentic_rpco_demo")
    assert resolve_run_dir() == tmp_path.resolve() / "agentic_rpco_demo"


def test_dump_rollout_artifacts_writes_trajectory_and_appends_monitor(tmp_path):
    _bind(tmp_path)
    dump_rollout_artifacts(
        tokenizer=_Tok(),
        step=0,
        relpath="step_000000/sample_9001",
        sample_index=9001,
        raw_prompt=[{"role": "user", "content": "a cafe poster"}],
        outputs=_output([10, 11, 12]),
    )

    step_dir = tmp_path / "layout_test" / "rollout_trajectories" / "step_000000"
    payload = json.loads((step_dir / "sample_9001.json").read_text())
    assert payload["trajectory_relpath"] == "step_000000/sample_9001"
    assert payload["user_prompt"] == "a cafe poster"
    assert (step_dir / "sample_9001.txt").is_file()

    jsonl = tmp_path / "layout_test" / "hermes_actions" / "step_000000.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["sample_index"] == 9001
    assert rows[0]["reward_metrics"]["score"] == 0.5
    assert rows[0]["reward_metrics"]["num_generate_image_prompts"] == 2
    assert (tmp_path / "layout_test" / "hermes_actions" / "step_000000.txt").is_file()


def test_concurrent_appends_keep_each_jsonl_row_parseable(tmp_path):
    """V1 workers append to one step monitor; no row may be spliced."""
    _bind(tmp_path)

    def _dump(index):
        dump_rollout_artifacts(
            tokenizer=_Tok(),
            step=3,
            relpath=f"step_000003/sample_{index}.00",
            sample_index=index,
            raw_prompt=[{"role": "user", "content": f"poster {index}"}],
            outputs=_output([1, 2]),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_dump, range(24)))

    lines = [
        line
        for line in (tmp_path / "layout_test" / "hermes_actions" / "step_000003.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 24
    assert sorted(json.loads(line)["sample_index"] for line in lines) == list(range(24))
    assert len(list((tmp_path / "layout_test" / "rollout_trajectories" / "step_000003").glob("*.json"))) == 24


def test_dump_rollout_artifacts_preserves_live_image_meta(tmp_path):
    """Live tool PNGs are indexed into the trajectory payload, not overwritten."""
    _bind(tmp_path)
    relpath = "step_000000/sample_9002"
    image_dir = tmp_path / "layout_test" / "rollout_images" / relpath
    image_dir.mkdir(parents=True)
    (image_dir / "image_00_abcdef012345.png").write_bytes(b"not-a-real-png")

    dump_rollout_artifacts(
        tokenizer=_Tok(),
        step=0,
        relpath=relpath,
        sample_index=9002,
        raw_prompt=[{"role": "user", "content": "poster"}],
        outputs=_output([7, 8]),
    )

    payload = json.loads(
        (tmp_path / "layout_test" / "rollout_trajectories" / "step_000000" / "sample_9002.json").read_text()
    )
    assert payload["image_dir"].endswith(relpath.replace("/", str(Path("/"))))
    assert len(payload["image_paths"]) == 1
    meta = json.loads((image_dir / "meta.json").read_text())
    assert meta["trajectory_relpath"] == relpath
    assert meta["source"] == "direct_tool_write"


def test_dumped_turns_expose_the_rewritten_diffusion_prompt(tmp_path):
    """``tool_prompt`` makes a rewrite chain readable without unescaping ``decode``.

    ``turn_prompt`` is the whole chat template (identical on every turn) and
    ``turn_obs`` only the judge observation, so the rewritten diffusion prompt that
    the harness accepted existed only inside the escaped JSON tool call.
    """
    _bind(tmp_path)
    gen = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "PROMPT_V1"}}\n</tool_call>'
    judge_call = (
        '<tool_call>\n{"name": "judge_image", "arguments": '
        '{"user_request": "same as user message", "image_prompt": "PROMPT_V1"}}\n</tool_call>'
    )
    judge_obs = "<tool_response>\nagentic_judge ok=1 good_enough =NO\n</tool_response>"
    rewrite = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "PROMPT_V2"}}\n</tool_call>'
    tokenizer = _PieceTok({10: gen, 11: judge_obs, 12: judge_call, 13: judge_obs, 14: rewrite})

    dump_rollout_artifacts(
        tokenizer=tokenizer,
        step=0,
        relpath="step_000000/sample_9003",
        sample_index=9003,
        raw_prompt=[{"role": "user", "content": "a poster"}],
        outputs=SimpleNamespace(
            prompt_ids=[1, 2, 3],
            response_ids=[10, 11, 12, 13, 14],
            response_mask=[1, 0, 1, 0, 1],
            reward_score=0.1,
            extra_fields={},
        ),
    )

    run_dir = tmp_path / "layout_test"
    payload = json.loads((run_dir / "rollout_trajectories" / "step_000000" / "sample_9003.json").read_text())
    assert [(turn["tool_name"], turn["tool_prompt"]) for turn in payload["rollout_turns"]] == [
        ("generate_image", "PROMPT_V1"),
        ("judge_image", "PROMPT_V1"),
        ("generate_image", "PROMPT_V2"),
    ]
    # The judge echoes the prompt it inspected, so the pair reads as
    # submit v1 -> echo v1 -> submit v2 without unescaping ``decode``.

    row = json.loads((run_dir / "hermes_actions" / "step_000000.jsonl").read_text().splitlines()[0])
    assert [turn["tool_prompt"] for turn in row["rollout_turns"]] == ["PROMPT_V1", "PROMPT_V1", "PROMPT_V2"]
    text_block = (run_dir / "rollout_trajectories" / "step_000000" / "sample_9003.txt").read_text()
    assert "turn_3_tool_prompt:" in text_block
    assert "PROMPT_V2" in text_block


def test_val_set_rows_never_touch_rollout_trajectories_or_monitor(monkeypatch):
    """A val-set batch is neither dumped nor mask-discarded (baseline contract).

    Only the fixed 9001-9004 holdout populates a validation step, so a val batch
    of hundreds of rows must not append to ``hermes_actions`` or create
    ``sample_<index>`` trajectory files.
    """
    dumped: list = []
    discarded: list = []
    monkeypatch.setattr(omni_agent_loop, "dump_raw_rollouts", lambda **kwargs: dumped.append(kwargs))
    monkeypatch.setattr(omni_agent_loop, "discard_invalid_rollouts", lambda output: discarded.append(output))
    monkeypatch.setattr(omni_agent_loop, "_stamp_scorer_knobs", lambda batch, config: None)
    monkeypatch.setattr(omni_agent_loop, "AgenticRewardMetrics", SimpleNamespace(aggregate=lambda ntb: {}))
    monkeypatch.setattr(
        omni_agent_loop.OmniAgentLoopManager, "_maybe_run_val_viz", lambda self, step, *, dedicated_manager: None
    )
    monkeypatch.setattr(
        omni_agent_loop.AgentLoopManager,
        "generate_sequences",
        lambda self, batch: SimpleNamespace(non_tensor_batch={}, meta_info={}),
    )

    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager.config = OmegaConf.create({})
    manager._monitor_tokenizer = _Tok()

    def _run(validate):
        batch = SimpleNamespace(meta_info={"validate": validate, "global_steps": 7})
        OmniAgentLoopManager.generate_sequences(manager, batch)

    _run(False)
    assert len(dumped) == 1
    assert len(discarded) == 1

    _run(True)
    # Unchanged: the val pass skipped both, the train pass already ran them.
    assert len(dumped) == 1
    assert len(discarded) == 1


def test_tq_worker_skips_dump_for_validation(monkeypatch):
    """``rollout_valid``/val workers must not write the step monitor."""
    import asyncio

    dumped: list = []
    monkeypatch.setattr(omni_agent_loop, "dump_rollout_artifacts", lambda **kwargs: dumped.append(kwargs))

    async def _noop_postprocess(self, output, validate, **kwargs):
        return None

    monkeypatch.setattr(omni_agent_loop._AgentLoopWorkerTQImpl, "_agent_loop_postprocess", _noop_postprocess)

    worker = object.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    worker.tokenizer = _Tok()
    kwargs = {
        "_agentic_trajectory_relpath": "step_000070/sample_12",
        "global_steps": 70,
        "index": 12,
        "raw_prompt": [{"role": "user", "content": "hi"}],
    }

    asyncio.run(omni_agent_loop.OmniAgentLoopWorkerTQImpl._agent_loop_postprocess(worker, "out", True, **kwargs))
    assert dumped == []

    asyncio.run(omni_agent_loop.OmniAgentLoopWorkerTQImpl._agent_loop_postprocess(worker, "out", False, **kwargs))
    assert len(dumped) == 1
    assert dumped[0]["relpath"] == "step_000070/sample_12"
    assert dumped[0]["step"] == 70


def test_val_viz_provider_is_env_gated_and_uses_holdout_indices(monkeypatch):
    monkeypatch.delenv("AGENTIC_VAL_VIZ", raising=False)
    assert resolve_agentic_val_viz_provider() is None

    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")
    provider = resolve_agentic_val_viz_provider()
    assert provider is not None
    batch = provider.build_batch(0, eos_token_id=2, pad_token_id=0)
    assert list(batch.non_tensor_batch["index"]) == [9001, 9002, 9003, 9004]
    assert set(batch.non_tensor_batch["data_source"]) == {"agentic_val_viz"}
    assert batch.meta_info["validate"] is True


def test_traj_dir_lock_is_shared_across_modules(tmp_path):
    """``image_gen`` and the dump module must serialise on one lock object.

    ``locking`` is the single lock domain: importing it from either module
    yields the same reentrant lock, so the live tool cannot allocate an
    ``image_NN`` index (or rewrite ``meta.json``) mid-way through a
    post-processing rewrite of the same trajectory folder.
    """
    import verl_omni.tools.image_gen as image_gen

    traj_dir = tmp_path / "rollout_images" / "step_000001" / "sample_3.00"
    assert traj_locking.traj_dir_lock(traj_dir) is traj_locking.traj_dir_lock(traj_dir)
    # ``image_gen`` no longer defines its own lock domain.
    assert not hasattr(image_gen, "_traj_dir_lock")
    assert not hasattr(image_gen, "_traj_dir_exclusive")


def test_traj_dir_exclusive_blocks_a_second_writer(tmp_path):
    """A writer holding the folder lock keeps another writer out until it exits."""
    traj_dir = tmp_path / "rollout_images" / "step_000001" / "sample_4.00"
    traj_dir.mkdir(parents=True)
    order: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def _holder():
        with traj_locking.traj_dir_exclusive(traj_dir):
            order.append("first-in")
            entered.set()
            release.wait(timeout=5)
            order.append("first-out")

    def _waiter():
        entered.wait(timeout=5)
        with traj_locking.traj_dir_exclusive(traj_dir):
            order.append("second-in")

    first = threading.Thread(target=_holder)
    second = threading.Thread(target=_waiter)
    first.start()
    second.start()
    assert entered.wait(timeout=5)
    # The second writer must still be blocked while the first holds the lock.
    time.sleep(0.2)
    assert "second-in" not in order
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert order == ["first-in", "first-out", "second-in"]


def test_manager_runs_val_holdout_once_per_step(monkeypatch):
    """Holdouts must not repeat within a step, and never touch the val partition."""
    monkeypatch.setattr(omni_agent_loop, "dump_raw_rollouts", lambda **kwargs: None)
    monkeypatch.setattr(omni_agent_loop, "_stamp_scorer_knobs", lambda batch, config: None)
    dispatched: list = []
    monkeypatch.setattr(
        omni_agent_loop.AgentLoopManager, "generate_sequences", lambda self, batch: dispatched.append(batch)
    )

    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager._val_viz_provider = SimpleNamespace(build_batch=lambda *args, **kwargs: SimpleNamespace())
    manager._val_viz_logged_steps = set()
    manager._val_viz_manager = SimpleNamespace()
    manager._monitor_tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    manager.config = OmegaConf.create({})

    OmniAgentLoopManager._maybe_run_val_viz(manager, 5, dedicated_manager=False)
    OmniAgentLoopManager._maybe_run_val_viz(manager, 5, dedicated_manager=False)
    OmniAgentLoopManager._maybe_run_val_viz(manager, 6, dedicated_manager=False)

    assert len(dispatched) == 2
    assert manager._val_viz_logged_steps == {5, 6}
