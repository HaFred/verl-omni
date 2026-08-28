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
"""CPU tests for the UniCoT → agentic RL parquet builder."""

import json
from pathlib import Path

import pandas as pd
import pytest

from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl as builder


def _write_metadata(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    """Write a HF-hub-style snapshot dir (snapshots/<hash>/metadata.json)."""
    snapshot = tmp_path / "datasets" / name / "snapshots" / "0000000000000000000000000000000000000000"
    snapshot.mkdir(parents=True)
    (snapshot / "metadata.json").write_text(json.dumps(rows))
    return snapshot.parent.parent


def _reflection_row(data_id: str, states: int) -> dict:
    """Synthetic Self-Reflection row with ``states`` reflection states."""
    inputs = [f"./images/source/{data_id}_{i}.png" for i in range(states)]
    outputs: list[str | None] = [f"./images/source/{data_id}_{i + 1}.png" for i in range(states - 1)] + [None]
    edits = ["Make the scene brighter."] * (states - 1) + ["Everything is good. No editing needed."]
    return {
        "data_id": data_id,
        "prompt": f"Visual prompt {data_id}.",
        "eval": [f"Evaluation state {i}." for i in range(states)],
        "eval_summary": [f"Summary state {i}." for i in range(states)],
        "edit": edits,
        "input_image": inputs,
        "output_image": outputs,
    }


def _breakdown_row(data_id: str, subtask_count: int) -> dict:
    subtasks: list[str | None] = [f"Subtask {i}." for i in range(subtask_count)] + [None] * (3 - subtask_count)
    images: list[str | None] = [f"./images/{data_id}_{i}.png" for i in range(subtask_count)] + [None] * (
        3 - subtask_count
    )
    return {
        "data_id": data_id,
        "prompt": f"Complex prompt {data_id}.",
        "subtasks": subtasks,
        "subtask_images": images,
    }


def _build(
    tmp_path: Path,
    *,
    reflection_rows,
    breakdown_rows,
    train_size: int | None = None,
    val_size: int | None = None,
    val_ratio: float = 0.05,
    **kwargs,
) -> Path:
    save_dir = tmp_path / "out"
    reflection_dir = _write_metadata(tmp_path, "refl", reflection_rows) if reflection_rows else ""
    breakdown_dir = _write_metadata(tmp_path, "brk", breakdown_rows) if breakdown_rows else ""
    builder.main_cli(
        reflection_dir=str(reflection_dir),
        breakdown_dir=str(breakdown_dir),
        local_save_dir=str(save_dir),
        train_size=train_size,
        val_size=val_size,
        val_ratio=val_ratio,
        **kwargs,
    )
    return save_dir


def _load(save_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(save_dir / f"{split}.parquet")


def test_build_mixed_parquet_schema(tmp_path):
    save_dir = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}", 1 + i % 3) for i in range(40)],
        breakdown_rows=[_breakdown_row(f"b{i}", 1 + i % 3) for i in range(40)],
        train_size=20,
        val_size=8,
        mix_ratio=0.5,
        seed=7,
    )
    train = _load(save_dir, "train")
    val = _load(save_dir, "val")

    assert set(train.columns) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert 0 < len(train) <= 20 and 0 < len(val) <= 8
    assert {"unicot_reflection", "unicot_breakdown"}.issubset(set(train["data_source"]))

    # Prompt is system + user only; UniCoT references must not leak into the prompt.
    for messages in train["prompt"]:
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "Summary state" not in messages[0]["content"]
        assert "Subtask 0." not in messages[0]["content"]
        assert messages[1]["content"].endswith("(≤4 sentences).")

    # Ground truth carries the reward references and weight set.
    gt0 = train["reward_model"].iloc[0]["ground_truth"]
    assert {"user_request", "task_type", "expected_num_images"}.issubset(set(gt0))
    assert all(f"w_{dim}" in gt0 for dim in builder.DIMS)
    assert gt0[f"w_{builder.DIMS[0]}"] == 1.0  # paper default: all weights 1

    extra0 = train["extra_info"].iloc[0]
    assert {"split", "index", "data_id", "task_type", "expected_num_images", "raw_prompt", "unicot_source"}.issubset(
        set(extra0)
    )
    assert (save_dir / "build_report.json").is_file()


def test_plan_rows_carry_reference_subtasks_and_plan_prompt(tmp_path):
    save_dir = _build(
        tmp_path,
        reflection_rows=[],
        breakdown_rows=[_breakdown_row(f"b{i}", 3) for i in range(8)],
        train_size=8,
        val_size=2,
        mix_ratio=0.5,
        seed=7,
    )
    train = _load(save_dir, "train")
    plan_rows = train[train["extra_info"].apply(lambda info: info["task_type"] == "plan")]
    assert len(plan_rows) == len(train)
    for reward_model in plan_rows["reward_model"]:
        gt = reward_model["ground_truth"]
        assert gt["plan_expected"] is True
        assert len(gt["reference_subtasks"]) == 3
        assert gt["expected_num_images"] == 3
    for messages in plan_rows["prompt"]:
        assert "plan" in messages[0]["content"].lower()
        assert "subtask" in messages[0]["content"].lower()


def test_no_breakdown_rows_become_reflect_tasks(tmp_path):
    no_breakdown = {
        "data_id": "single",
        "prompt": "A simple prompt.",
        "subtasks": ["No breakdown needed.", None, None],
        "subtask_images": [None, None, None],
    }
    save_dir = _build(
        tmp_path,
        reflection_rows=[_reflection_row("r0", 1)],
        breakdown_rows=[no_breakdown],
        train_size=4,
        val_size=2,
        mix_ratio=0.5,
        seed=7,
    )
    train = _load(save_dir, "train")
    breakdown_rows = train[train["data_source"] == "unicot_breakdown"]
    assert len(breakdown_rows) == 1
    gt = breakdown_rows["reward_model"].iloc[0]["ground_truth"]
    assert gt["task_type"] == "reflect"
    assert gt["expected_num_images"] == 1
    assert gt["plan_expected"] is False


def test_reflect_rows_keep_reference_steps(tmp_path):
    save_dir = _build(
        tmp_path,
        reflection_rows=[_reflection_row("r0", 2)],
        breakdown_rows=[],
        train_size=2,
        val_size=2,
        mix_ratio=0.5,
        seed=7,
    )
    train = _load(save_dir, "train")
    gt = train["reward_model"].iloc[0]["ground_truth"]
    assert gt["task_type"] == "reflect"
    assert gt["expected_num_images"] == 2
    assert [step["action"] for step in gt["reference_steps"]] == ["continue", "stop"]


def test_rejections_are_recorded_and_dropped(tmp_path):
    bad_row = _reflection_row("bad", 2)
    bad_row["edit"][0] = ""  # continue transition with an empty edit → fail closed
    save_dir = _build(
        tmp_path,
        reflection_rows=[_reflection_row("good", 1), bad_row],
        breakdown_rows=[],
        train_size=2,
        val_size=2,
        mix_ratio=0.5,
        seed=7,
    )
    report = json.loads((save_dir / "build_report.json").read_text())
    assert report["rejection_count"] == 1
    assert report["rejections"][0]["data_id"] == "bad"
    train = _load(save_dir, "train")
    assert all(info["data_id"] != "bad" for info in train["extra_info"])


def test_splits_are_disjoint_and_deterministic(tmp_path):
    reflection_rows = [_reflection_row(f"r{i}", 1 + i % 3) for i in range(40)]
    breakdown_rows = [_breakdown_row(f"b{i}", 1 + i % 3) for i in range(40)]
    save_a = _build(
        tmp_path / "a",
        reflection_rows=reflection_rows,
        breakdown_rows=breakdown_rows,
        train_size=24,
        val_size=8,
        mix_ratio=0.5,
        seed=7,
    )
    save_b = _build(
        tmp_path / "b",
        reflection_rows=reflection_rows,
        breakdown_rows=breakdown_rows,
        train_size=24,
        val_size=8,
        mix_ratio=0.5,
        seed=7,
    )
    train_ids = set(_load(save_a, "train")["extra_info"].apply(lambda info: info["data_id"]))
    val_ids = set(_load(save_a, "val")["extra_info"].apply(lambda info: info["data_id"]))
    assert not (train_ids & val_ids)
    assert list(_load(save_a, "train")["extra_info"]) == list(_load(save_b, "train")["extra_info"])


def test_mix_ratio_controls_reflect_fraction(tmp_path):
    rows = {"reflection": [_reflection_row(f"r{i}", 1) for i in range(30)]}
    rows["breakdown"] = [_breakdown_row(f"b{i}", 2) for i in range(30)]
    save_dir = _build(
        tmp_path,
        reflection_rows=rows["reflection"],
        breakdown_rows=rows["breakdown"],
        train_size=20,
        val_size=4,
        mix_ratio=0.25,
        seed=7,
    )
    train = _load(save_dir, "train")
    reflect_count = sum(info["task_type"] == "reflect" for info in train["extra_info"])
    assert 0 < reflect_count < len(train)  # plan-majority at mix_ratio=0.25


def test_full_mode_uses_all_rows_when_no_sizes(tmp_path):
    reflection_rows = [_reflection_row(f"r{i}", 1 + i % 3) for i in range(40)]
    breakdown_rows = [_breakdown_row(f"b{i}", 1 + i % 3) for i in range(40)]
    save_dir = _build(
        tmp_path,
        reflection_rows=reflection_rows,
        breakdown_rows=breakdown_rows,
        mix_ratio=0.5,  # ignored without sizes
        seed=7,
    )
    train = _load(save_dir, "train")
    val = _load(save_dir, "val")

    assert len(train) + len(val) == 80  # every parsed row lands in a split
    assert 0 < len(val) < len(train)  # ~95/5 hash split
    train_ids = {info["data_id"] for info in train["extra_info"]}
    val_ids = {info["data_id"] for info in val["extra_info"]}
    assert not (train_ids & val_ids)


def test_requires_at_least_one_dataset(tmp_path):
    with pytest.raises(SystemExit):
        builder.main_cli(
            reflection_dir="",
            breakdown_dir="",
            local_save_dir=str(tmp_path / "out"),
            train_size=None,
            val_size=None,
            mix_ratio=0.5,
            seed=7,
            val_ratio=0.05,
        )
