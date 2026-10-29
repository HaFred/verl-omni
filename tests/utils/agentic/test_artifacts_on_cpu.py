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
"""Artifact binding must not inherit a previous run's configuration."""

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.agentic import artifacts, config


@pytest.mark.parametrize(
    "incoming, expected",
    [
        ({}, "outputs/e2e/new"),
        ({"agentic_image_gen": {}}, "outputs/e2e/new"),
        ({"agentic_image_gen": {"run_dir": None}}, "outputs/e2e/new"),
        ({"agentic_image_gen": {"e2e_root": "new-root"}}, "new-root/new"),
        ({"agentic_image_gen": {"run_dir": "new-run"}}, "new-run"),
    ],
)
def test_rebinding_artifacts_uses_only_the_incoming_config(tmp_path, monkeypatch, incoming, expected):
    monkeypatch.setenv("VERLOMNI_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    previous = OmegaConf.create(
        {
            "trainer": {"experiment_name": "old"},
            "agentic_image_gen": {"run_dir": str(tmp_path / "old-run"), "e2e_root": str(tmp_path / "old-root")},
        }
    )
    artifacts.clear_run_artifacts()
    config.clear_agentic_image_gen()
    try:
        config.bind_agentic_image_gen(previous)
        artifacts.bind_run_artifacts(previous)
        assert artifacts.resolve_run_dir() == tmp_path / "old-run"

        current = OmegaConf.create({"trainer": {"experiment_name": "new"}, **incoming})
        artifacts.bind_run_artifacts(current)

        assert artifacts.resolve_run_dir() == tmp_path / expected
        assert artifacts.resolve_rollout_images_root() == tmp_path / expected / "rollout_images"
    finally:
        artifacts.clear_run_artifacts()
        config.clear_agentic_image_gen()
