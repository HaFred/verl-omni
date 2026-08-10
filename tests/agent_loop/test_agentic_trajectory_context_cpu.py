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

from __future__ import annotations

from pathlib import Path

from verl_omni.agent_loop.agentic_trajectory_context import (
    build_artifact_id,
    clear_tool_artifact_registry,
    get_active_rollout_id,
    register_tool_artifact,
    resolve_tool_image_path,
    rollout_id_from_relpath,
    set_active_trajectory_relpath,
    set_latest_tool_image_path,
)


def test_artifact_id_is_identity_not_pixels():
    a = build_artifact_id(relpath="step_000001/sample_0.00", index=0, prompt="soldier letter")
    b = build_artifact_id(relpath="step_000001/sample_0.01", index=0, prompt="soldier letter")
    c = build_artifact_id(relpath="step_000001/sample_0.00", index=1, prompt="soldier letter")
    assert a != b  # different rollouts, same prompt text
    assert a != c  # different call index
    assert len(a) == 12


def test_resolve_never_crosses_rollouts(tmp_path: Path):
    clear_tool_artifact_registry()
    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    png_a.write_bytes(b"\x89PNG\r\n\x1a\na")
    png_b.write_bytes(b"\x89PNG\r\n\x1a\nb")

    set_active_trajectory_relpath("step_000005/sample_2.00")
    rid_a = get_active_rollout_id()
    assert rid_a == rollout_id_from_relpath("step_000005/sample_2.00")
    register_tool_artifact(
        prompt="soldier letter by lamplight",
        paths=[str(png_a)],
        artifact_id="aaaaaaaaaaaa",
        trajectory_relpath="step_000005/sample_2.00",
        rollout_id=rid_a,
    )
    set_latest_tool_image_path(str(png_a))

    # Switch to a later-step concurrent-style rollout with the SAME prompt.
    set_active_trajectory_relpath("step_000006/sample_8.00")
    rid_b = get_active_rollout_id()
    assert rid_b != rid_a
    register_tool_artifact(
        prompt="soldier letter by lamplight",
        paths=[str(png_b)],
        artifact_id="bbbbbbbbbbbb",
        trajectory_relpath="step_000006/sample_8.00",
        rollout_id=rid_b,
    )
    set_latest_tool_image_path(str(png_b))

    hit = resolve_tool_image_path(image_prompt="soldier letter by lamplight")
    assert hit == str(png_b)

    # Explicit artifact id from the other rollout must still resolve (direct map),
    # but prompt-only lookup must stay on the active rollout.
    assert resolve_tool_image_path(artifact_id="aaaaaaaaaaaa") == str(png_a)
    assert resolve_tool_image_path(image_prompt="soldier letter by lamplight") == str(png_b)

    clear_tool_artifact_registry()


def test_count_and_latest_prompt_are_rollout_scoped(tmp_path: Path):
    from verl_omni.agent_loop.agentic_trajectory_context import (
        count_live_generate_artifacts_for_active_rollout,
        get_latest_generate_prompt_for_active_rollout,
    )

    clear_tool_artifact_registry()
    png = tmp_path / "c.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nc")
    set_active_trajectory_relpath("step_000007/sample_0.00")
    register_tool_artifact(
        prompt="first prompt",
        paths=[str(png)],
        backend="vllm_omni",
        trajectory_relpath="step_000007/sample_0.00",
    )
    png2 = tmp_path / "d.png"
    png2.write_bytes(b"\x89PNG\r\n\x1a\nd")
    register_tool_artifact(
        prompt="second prompt",
        paths=[str(png2)],
        backend="vllm_omni",
        trajectory_relpath="step_000007/sample_0.00",
    )
    assert count_live_generate_artifacts_for_active_rollout() == 2
    assert get_latest_generate_prompt_for_active_rollout() == "second prompt"
    clear_tool_artifact_registry()
