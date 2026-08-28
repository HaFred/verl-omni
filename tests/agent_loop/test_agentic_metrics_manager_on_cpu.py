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
"""CPU tests for agentic reward metric aggregation (train + val prefixes)."""

import json

import numpy as np
import pytest

import verl_omni.agent_loop.agentic_metrics_manager as metrics_module
from verl_omni.agent_loop.agentic_metrics_manager import (
    AgenticMetricsAgentLoopManager,
    _turn_kind,
    aggregate_agentic_reward_metrics,
)
from verl_omni.agent_loop.agentic_trajectory_context import build_trajectory_relpath
from verl_omni.utils.tracking import AgenticValidationGenerationsLogger


def _val_prefixed(metrics: dict[str, float]) -> dict[str, float]:
    """The val transform used by ``_log_val_reward_metrics`` (mirrors it)."""
    return {f"val_{key}": value for key, value in metrics.items()}


def test_aggregate_includes_rpco_dimensions_and_pr1_fields():
    batch = {
        "reward_reflect": np.array([0.8, 0.9]),
        "reward_plan": np.array([0.5, 0.6]),
        "reward_format": np.array([1.0, 1.0]),
        "reward_result": np.array([1.0, 0.0]),
        "reward_done": np.array([1.0, 0.0]),
        "reward_tool_call": np.array([1.0, 1.0]),
    }
    metrics = aggregate_agentic_reward_metrics(batch)

    assert metrics["agentic_reward/reflect/mean"] == pytest.approx(0.85)
    assert metrics["agentic_reward/plan/max"] == pytest.approx(0.6)
    assert metrics["agentic_reward/result/min"] == pytest.approx(0.0)
    assert metrics["agentic_reward/tool_call/mean"] == pytest.approx(1.0)
    assert metrics["agentic_reward/done/mean"] == pytest.approx(0.5)
    # RPCO batches omit C/A stubs — they must not appear as perpetual zeros.
    assert "agentic_reward/correctness/mean" not in metrics
    assert "agentic_reward/aesthetics/mean" not in metrics
    # Only the named components are aggregated.
    assert all(key.startswith("agentic_reward/") for key in metrics)


def test_aggregate_pr1_correctness_aesthetics_when_present():
    metrics = aggregate_agentic_reward_metrics(
        {
            "reward_tool_call": np.array([1.0]),
            "reward_correctness": np.array([0.8]),
            "reward_aesthetics": np.array([0.7]),
            "reward_done": np.array([1.0]),
        }
    )
    assert metrics["agentic_reward/correctness/mean"] == pytest.approx(0.8)
    assert metrics["agentic_reward/aesthetics/max"] == pytest.approx(0.7)
    assert "agentic_reward/reflect/mean" not in metrics


def test_val_prefix_transform():
    metrics = aggregate_agentic_reward_metrics(
        {
            "reward_reflect": np.array([0.8, 0.9]),
            "reward_tool_call": np.array([1.0, 1.0]),
        }
    )
    val_metrics = _val_prefixed(metrics)

    assert val_metrics["val_agentic_reward/reflect/mean"] == pytest.approx(0.85)
    assert val_metrics["val_agentic_reward/tool_call/mean"] == pytest.approx(1.0)
    assert "val_agentic_reward/correctness/mean" not in val_metrics


def test_absent_keys_are_skipped():
    metrics = aggregate_agentic_reward_metrics({"reward_plan": np.array([0.4])})

    assert set(metrics) == {
        "agentic_reward/plan/mean",
        "agentic_reward/plan/min",
        "agentic_reward/plan/max",
    }


def test_empty_arrays_are_skipped():
    metrics = aggregate_agentic_reward_metrics({"reward_plan": np.array([])})

    assert metrics == {}


def test_pair_generate_image_turns_zips_prompts_and_paths():
    pairs = AgenticValidationGenerationsLogger.pair_turns(
        ["first rewrite", "second rewrite"], ["/tmp/a.png", "/tmp/b.png"]
    )
    assert pairs == [("first rewrite", "/tmp/a.png"), ("second rewrite", "/tmp/b.png")]


def test_pair_generate_image_turns_pads_missing_side():
    assert AgenticValidationGenerationsLogger.pair_turns(["only"], []) == [("only", None)]
    assert AgenticValidationGenerationsLogger.pair_turns([], ["/tmp/a.png"]) == [("", "/tmp/a.png")]


def test_val_generations_table_accumulates_all_steps(tmp_path, monkeypatch):
    """Each val step appends a row; commit=True + summary dual-write has all rows."""
    import types

    logged: list[dict] = []

    class _FakeImage:
        def __init__(self, path):
            self.path = str(path)

    class _FakeTable:
        def __init__(self, columns, data=None, log_mode="IMMUTABLE"):
            self.columns = columns
            self.data = [list(row) for row in (data or [])]
            self.log_mode = log_mode
            self.logged = False

        def add_data(self, *row):
            if self.logged:
                raise RuntimeError("cannot mutate an uploaded W&B table")
            self.data.append(list(row))

    class _FakeRun:
        step = 0

        def __init__(self):
            self.summary = {}

    def _log(payload, step=None, commit=True):
        for table in payload.values():
            table.logged = True
        logged.append({"payload": payload, "step": step, "commit": commit})

    fake_run = _FakeRun()
    fake_wandb = types.SimpleNamespace(
        run=fake_run,
        Image=_FakeImage,
        Table=_FakeTable,
        log=_log,
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "2")

    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    png_a.write_bytes(b"fake")
    png_b.write_bytes(b"fake")

    tracker = AgenticValidationGenerationsLogger(tmp_path, max_turns=2)
    tracker.log(0, {"val/generations": [("p0", str(png_a))]})
    tracker.log(10, {"val/generations": [("p10", str(png_b))]})
    tracker.log(20, {"val/generations": [("p20", None)]})

    assert len(logged) == 3
    # Ray-worker flush: must commit so Media is not stuck on the first row.
    assert all(entry["commit"] is True for entry in logged)
    assert [entry["step"] for entry in logged] == [0, 10, 20]
    # Committed payload and summary dual-write must carry all prior val steps.
    latest = logged[-1]["payload"]["val/generations"]
    assert fake_run.summary["val/generations"] is latest
    assert [row[0] for row in latest.data] == [0, 10, 20]
    assert latest.data[0][1] == "p0"
    assert isinstance(latest.data[0][2], _FakeImage)
    assert latest.data[1][1] == "p10"
    assert latest.data[2][2] == ""
    assert len(tracker.history["val/generations"]) == 3
    # Default (IMMUTABLE) table objects; never mutate after log — FlowGRPO copy.
    assert latest.log_mode == "IMMUTABLE"
    tables = [entry["payload"]["val/generations"] for entry in logged]
    assert len({id(table) for table in tables}) == 3
    assert [len(table.data) for table in tables] == [1, 2, 3]
    # Later steps reuse Image objects from prior ``table.data`` (FlowGRPO).
    assert tables[1].data[0][2] is tables[0].data[0][2]
    assert tables[2].data[0][2] is tables[1].data[0][2]
    assert tables[2].data[1][2] is tables[1].data[1][2]


def test_wandb_effective_log_step_keeps_trainer_global_steps(monkeypatch):
    """Mid-validate soft logs must use exact global_steps (never tip+1).

    Bumping past ``run.step`` before the trainer's ``Tracking.log`` made
    ``val-core`` at step N drop with ``Tried to log to step N < current N+1``.
    """
    import types

    fake_wandb = types.SimpleNamespace(run=types.SimpleNamespace(step=33))
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    assert AgenticValidationGenerationsLogger.effective_wandb_step(30) == 30
    assert AgenticValidationGenerationsLogger.effective_wandb_step(33) == 33
    assert AgenticValidationGenerationsLogger.effective_wandb_step(40) == 40
    fake_wandb.run = None
    assert AgenticValidationGenerationsLogger.effective_wandb_step(30) == 30
    assert AgenticValidationGenerationsLogger.effective_wandb_step(None) is None
    assert AgenticValidationGenerationsLogger.effective_wandb_step("bad") is None


def test_val_generations_commits_and_dual_writes_summary(tmp_path, monkeypatch):
    """Holdout commits at exact N and dual-writes run.summary; never tip+1."""
    import types

    logged: list[dict] = []

    class _FakeTable:
        def __init__(self, columns, data=None, log_mode="IMMUTABLE"):
            self.columns = columns
            self.data = [list(row) for row in (data or [])]
            self.log_mode = log_mode

        def add_data(self, *row):
            self.data.append(list(row))

    class _FakeRun:
        step = 10

        def __init__(self):
            self.summary = {}

    fake_run = _FakeRun()
    fake_wandb = types.SimpleNamespace(
        run=fake_run,
        Image=lambda path: str(path),
        Table=_FakeTable,
        log=lambda payload, step=None, commit=True: logged.append(
            {"payload": payload, "step": step, "commit": commit}
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    png = tmp_path / "cafe.png"
    png.write_bytes(b"fake")
    tracker = AgenticValidationGenerationsLogger(tmp_path, max_turns=1)
    tracker.log(10, {"val/generations": [("p10", str(png))]})

    assert len(logged) == 1
    assert logged[0]["step"] == 10
    assert logged[0]["commit"] is True
    assert fake_run.summary["val/generations"] is logged[0]["payload"]["val/generations"]
    assert [row[0] for row in fake_run.summary["val/generations"].data] == [10]


def test_val_generations_history_restores_from_image_metadata(tmp_path, monkeypatch):
    """A resumed run republishes all cafe validation steps, not only new ones."""
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "2")
    for step in (0, 10, 20):
        for sample in (9001, 9002):
            image_dir = tmp_path / "rollout_images" / f"step_{step:06d}" / f"sample_{sample}"
            image_dir.mkdir(parents=True)
            paths = [str(image_dir / f"image_{i:02d}.png") for i in range(2)]
            (image_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "tool_prompts": [f"prompt-{sample}-{step}-0", f"prompt-{sample}-{step}-1"],
                        "image_paths": paths,
                    }
                )
            )

    history = AgenticValidationGenerationsLogger(tmp_path, max_turns=2).history

    assert [row[0] for row in history["val/generations"]] == [0, 10, 20]
    assert [row[0] for row in history["val/generations_plan"]] == [0, 10, 20]
    assert history["val/generations"][1][1] == "prompt-9001-10-0"
    assert history["val/generations_plan"][2][3] == "prompt-9002-20-1"


def test_val_generations_reflect_and_plan_accumulate_independently(tmp_path, monkeypatch):
    """The 9001 reflect and 9002 plan protocols retain every validation step."""
    import types

    class _FakeTable:
        def __init__(self, columns, data=None, log_mode="IMMUTABLE"):
            self.columns = columns
            self.data = [list(row) for row in (data or [])]
            self.log_mode = log_mode

        def add_data(self, *row):
            self.data.append(list(row))

    fake_wandb = types.SimpleNamespace(
        run=types.SimpleNamespace(step=0),
        Image=lambda path: str(path),
        Table=_FakeTable,
        log=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "1")

    png = tmp_path / "cafe.png"
    png.write_bytes(b"fake")
    tracker = AgenticValidationGenerationsLogger(tmp_path, max_turns=1)

    for step in (0, 10, 20):
        tracker.log(
            step,
            {
                "val/generations": [(f"reflect-{step}", str(png))],
                "val/generations_plan": [(f"plan-{step}", str(png))],
            },
        )

    reflect = tracker.tables["val/generations"]
    plan = tracker.tables["val/generations_plan"]
    assert [row[0] for row in reflect.data] == [0, 10, 20]
    assert [row[0] for row in plan.data] == [0, 10, 20]
    assert [row[1] for row in reflect.data] == ["reflect-0", "reflect-10", "reflect-20"]
    assert [row[1] for row in plan.data] == ["plan-0", "plan-10", "plan-20"]
    assert reflect.log_mode == plan.log_mode == "IMMUTABLE"


def test_cafe_trajectory_dump_writes_samples_without_overwriting_monitor(tmp_path, monkeypatch):
    """9001/9002 get trajectory files without replacing regular hermes_actions."""
    import types

    class _Ids:
        def tolist(self):
            return [1, 2, 3]

    class _Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            del ids, skip_special_tokens
            return "decoded cafe rollout"

    output = types.SimpleNamespace(
        batch={
            "responses": [_Ids(), _Ids()],
            "response_mask": [object(), object()],
        },
        non_tensor_batch={
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": "reflect cafe"}],
                    [{"role": "user", "content": "plan cafe"}],
                ],
                dtype=object,
            ),
            "index": np.array([9001, 9002]),
            "trajectory_relpath": np.array(
                ["step_000010/sample_9001", "step_000010/sample_9002"],
                dtype=object,
            ),
        },
    )
    manager = types.SimpleNamespace(_monitor_tokenizer=_Tokenizer())
    monkeypatch.setattr(metrics_module, "_run_dir", lambda: tmp_path)
    monkeypatch.setattr(metrics_module, "_materialize_rollout_images", lambda **kwargs: [])
    monkeypatch.setattr(metrics_module, "_artifact_reward_metrics", lambda output, i: {})
    monkeypatch.setattr(
        metrics_module,
        "split_rollout_turns",
        lambda *args, **kwargs: [
            {
                "turn": 1,
                "turn_prompt": "cafe prompt",
                "turn_input": "full cafe input",
                "decode": "assistant decode",
                "response": "assistant response",
                "decode_has_tool_call": True,
            }
        ],
    )

    AgenticMetricsAgentLoopManager._dump_raw_rollouts(
        manager, None, output, 10, write_monitor=False, validate=True
    )

    trajectory_dir = tmp_path / "rollout_trajectories" / "step_000010"
    assert (trajectory_dir / "sample_9001.json").is_file()
    assert (trajectory_dir / "sample_9001.txt").is_file()
    assert (trajectory_dir / "sample_9002.json").is_file()
    assert (trajectory_dir / "sample_9002.txt").is_file()
    assert not (tmp_path / "hermes_actions").exists()


_JUDGE_NO = (
    "VL judge on the last generated image:\n"
    "  correctness=0.80\n"
    "  aesthetics =0.80\n"
    "  good_enough =NO\n"
    "  findings: fully rendered\n"
    "  suggested_fixes: No fixes needed\n"
    "  agentic_judge ok=1 parse_ok=1"
)
_JUDGE_YES = _JUDGE_NO.replace("good_enough =NO", "good_enough =YES")
_CONTINUE_CUE = (
    "Reflection: VL judge reports correctness=0.80, aesthetics=0.80, "
    "good_enough=NO. Rewriting the diffusion prompt next. agentic_forced_reflection=1"
)
_STOP_CUE = (
    "Reflection: VL judge reports good_enough=YES. Stop now. "
    "Your next and only action must be exactly Done. "
    "agentic_stop_decision_required=1 agentic_forced_reflection=1"
)
_REWRITE_DECODE = (
    '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "lion"}}\n</tool_call>'
)


def test_turn_kind_uses_good_enough_and_policy_decode_not_injected_cue():
    """Injected continue-cue must not hide a sampled Done / rewrite."""
    done = "Reflection: The image meets the original request. Done.<|im_end|>"
    assert _turn_kind(done, _JUDGE_NO, _CONTINUE_CUE) == "agent_reflection_done"
    assert _turn_kind(done, _JUDGE_YES, _STOP_CUE) == "agent_done_after_forced_reflection"
    assert _turn_kind(_REWRITE_DECODE, _JUDGE_NO, _CONTINUE_CUE) == (
        "agent_rewrite_after_forced_reflection"
    )


def test_turn_kind_forced_reflection_continue_only_when_decode_empty():
    assert _turn_kind("", _JUDGE_NO, _CONTINUE_CUE) == "forced_reflection_continue"
    assert _turn_kind("", _JUDGE_YES, _STOP_CUE) == "forced_reflection_stop_cue"


def test_val_holdout_runs_before_main_validate_generate():
    """Cafe 9001/9002 + W&B table commit must precede the UniCoT val set."""
    import inspect

    src = inspect.getsource(AgenticMetricsAgentLoopManager.generate_sequences)
    # Strip the holdout body call inside ``_maybe_run_val_viz`` by looking only
    # at the outer method: viz gate, then parent generate for the val batch.
    viz_gate = src.index("self._maybe_run_val_viz(step)")
    main_generate = src.index("output = super().generate_sequences(prompts)")
    assert viz_gate < main_generate
    assert "if is_val:" in src[:viz_gate]


def test_build_trajectory_relpath_val_omits_rollout_n_suffix():
    """Train group member ``.00`` must not share a folder with val ``n=1``."""
    train = build_trajectory_relpath(step=140, sample_index=145, rollout_n=0, validate=False)
    val = build_trajectory_relpath(step=140, sample_index=145, rollout_n=0, validate=True)
    assert train == "step_000140/sample_145.00"
    assert val == "step_000140/sample_145"
    assert train != val
