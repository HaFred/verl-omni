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

"""CPU tests for agentic validation-visualization providers."""

import pytest

from verl_omni.utils.agentic_val_viz import (
    AgenticValVizProvider,
    ValVizCase,
    resolve_agentic_val_viz_provider,
)


def test_resolve_disabled_without_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_VAL_VIZ", raising=False)
    assert resolve_agentic_val_viz_provider() is None


def test_resolve_cafe_poster_provider(monkeypatch):
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")
    monkeypatch.setenv("AGENTIC_VAL_VIZ_PROVIDER", "cafe_poster")
    provider = resolve_agentic_val_viz_provider()
    assert provider is not None
    assert [case.sample_index for case in provider.cases] == [9001, 9002, 9003, 9004]
    assert provider.sample_table_keys == {
        "sample_9001": "val/generations",
        "sample_9002": "val/generations_plan",
        "sample_9003": "val/generations_cn",
        "sample_9004": "val/generations_plan_cn",
    }
    assert provider.cases[0].task_type == "reflect"
    assert provider.cases[1].task_type == "plan"
    assert provider.cases[2].task_type == "reflect"
    assert provider.cases[3].task_type == "plan"
    assert "ARTISAN ROAST" in provider.cases[0].user_request
    assert "沙发山打呼节" in provider.cases[2].user_request
    assert provider.cases[2].user_request == provider.cases[3].user_request


def test_provider_build_batch_shapes(monkeypatch):
    pytest.importorskip("verl")
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")
    provider = resolve_agentic_val_viz_provider()
    assert provider is not None
    batch = provider.build_batch(40, eos_token_id=1, pad_token_id=0)
    assert len(batch) == 4
    assert list(batch.non_tensor_batch["index"]) == [9001, 9002, 9003, 9004]
    assert batch.meta_info["validate"] is True
    assert batch.meta_info["global_steps"] == 40
    assert batch.non_tensor_batch["reward_model"][1]["ground_truth"]["task_type"] == "plan"


def test_custom_provider_cases_do_not_require_cafe():
    provider = AgenticValVizProvider(
        [
            ValVizCase(
                sample_index=42,
                table_key="val/generations_custom",
                viz_id="custom",
                task_type="reflect",
                system_prompt="sys",
                user_request="draw a cat",
            )
        ]
    )
    assert provider.sample_table_keys == {"sample_42": "val/generations_custom"}
