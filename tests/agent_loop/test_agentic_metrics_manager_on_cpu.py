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

import numpy as np
import pytest

import verl_omni.agent_loop.agentic_metrics_manager as metrics_module
from verl_omni.agent_loop.agentic_metrics_manager import (
    AgenticMetricsAgentLoopManager,
    _pair_generate_image_turns,
    aggregate_agentic_reward_metrics,
)
from verl_omni.agent_loop.agentic_trajectory_context import build_trajectory_relpath


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
    decoded = (
        '<tool_call>{"name": "generate_image", "arguments": {"prompt": "first rewrite"}}</tool_call>\n'
        '<tool_call>{"name": "generate_image", "arguments": {"prompt": "second rewrite"}}</tool_call>'
    )
    pairs = _pair_generate_image_turns(decoded, ["/tmp/a.png", "/tmp/b.png"])
    assert pairs == [("first rewrite", "/tmp/a.png"), ("second rewrite", "/tmp/b.png")]


def test_pair_generate_image_turns_pads_missing_side():
    decoded = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "only"}}</tool_call>'
    assert _pair_generate_image_turns(decoded, []) == [("only", None)]
    assert _pair_generate_image_turns("", ["/tmp/a.png"]) == [("", "/tmp/a.png")]


def test_val_generations_table_accumulates_all_steps(tmp_path, monkeypatch):
    """FlowGRPO-style: each val step appends a row; summary table has all rows."""
    import types

    logged: list[dict] = []

    class _FakeImage:
        def __init__(self, path):
            self.path = str(path)

    class _FakeTable:
        def __init__(self, columns, data=None, log_mode="IMMUTABLE"):
            self.columns = columns
            self.data = list(data or [])
            self.log_mode = log_mode

        def add_data(self, *row):
            self.data.append(list(row))

    class _FakeRun:
        pass

    fake_wandb = types.SimpleNamespace(
        run=_FakeRun(),
        Image=_FakeImage,
        Table=_FakeTable,
        log=lambda payload, step=None, commit=True: logged.append({"payload": payload, "step": step, "commit": commit}),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "2")

    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    png_a.write_bytes(b"fake")
    png_b.write_bytes(b"fake")

    manager = types.SimpleNamespace(
        _val_generations_history={},
        _val_generations_tables={},
        _log_val_generations_table=AgenticMetricsAgentLoopManager._log_val_generations_table,
    )

    AgenticMetricsAgentLoopManager._log_val_generations_table(
        manager, 0, table_key="val/generations", turn_pairs=[("p0", str(png_a))]
    )
    AgenticMetricsAgentLoopManager._log_val_generations_table(
        manager, 10, table_key="val/generations", turn_pairs=[("p10", str(png_b))]
    )
    AgenticMetricsAgentLoopManager._log_val_generations_table(
        manager, 20, table_key="val/generations", turn_pairs=[("p20", None)]
    )

    assert len(logged) == 3
    assert all(entry["commit"] is True for entry in logged)
    # Latest summary payload must contain all prior val steps as rows.
    latest = logged[-1]["payload"]["val/generations"]
    assert [row[0] for row in latest.data] == [0, 10, 20]
    assert latest.data[0][1] == "p0"
    assert isinstance(latest.data[0][2], _FakeImage)
    assert latest.data[1][1] == "p10"
    assert latest.data[2][2] == ""
    assert len(manager._val_generations_history["val/generations"]) == 3
    assert latest.log_mode == "MUTABLE"
    # All logs reuse one mutable object instead of creating immutable
    # one-row artifacts that leave runs.summary stuck at step 0.
    assert all(entry["payload"]["val/generations"] is latest for entry in logged)


def test_val_generations_reflect_and_plan_accumulate_independently(tmp_path, monkeypatch):
    """The 9001 reflect and 9002 plan protocols retain every validation step."""
    import types

    class _FakeTable:
        def __init__(self, columns, log_mode="IMMUTABLE"):
            self.columns = columns
            self.data = []
            self.log_mode = log_mode

        def add_data(self, *row):
            self.data.append(list(row))

    fake_wandb = types.SimpleNamespace(
        run=object(),
        Image=lambda path: str(path),
        Table=_FakeTable,
        log=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "1")

    png = tmp_path / "cafe.png"
    png.write_bytes(b"fake")
    manager = types.SimpleNamespace(_val_generations_history={}, _val_generations_tables={})

    for step in (0, 10, 20):
        AgenticMetricsAgentLoopManager._log_val_generations_table(
            manager, step, table_key="val/generations", turn_pairs=[(f"reflect-{step}", str(png))]
        )
        AgenticMetricsAgentLoopManager._log_val_generations_table(
            manager, step, table_key="val/generations_plan", turn_pairs=[(f"plan-{step}", str(png))]
        )

    reflect = manager._val_generations_tables["val/generations"]
    plan = manager._val_generations_tables["val/generations_plan"]
    assert [row[0] for row in reflect.data] == [0, 10, 20]
    assert [row[0] for row in plan.data] == [0, 10, 20]
    assert [row[1] for row in reflect.data] == ["reflect-0", "reflect-10", "reflect-20"]
    assert [row[1] for row in plan.data] == ["plan-0", "plan-10", "plan-20"]
    assert reflect.log_mode == plan.log_mode == "MUTABLE"


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


def test_build_trajectory_relpath_val_omits_rollout_n_suffix():
    """Train group member ``.00`` must not share a folder with val ``n=1``."""
    train = build_trajectory_relpath(step=140, sample_index=145, rollout_n=0, validate=False)
    val = build_trajectory_relpath(step=140, sample_index=145, rollout_n=0, validate=True)
    assert train == "step_000140/sample_145.00"
    assert val == "step_000140/sample_145"
    assert train != val
