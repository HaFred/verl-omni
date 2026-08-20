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

from verl_omni.agent_loop.agentic_metrics_manager import (
    _pair_generate_image_turns,
    aggregate_agentic_reward_metrics,
)


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
