# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for client-side image-judge normalization controls."""

from verl_omni.utils.agentic_image_judge_parse import normalize_judge_payload


def _flat_high_payload() -> dict:
    return {
        "correctness_scores": {"subject": 0.95, "attributes": 0.95},
        "aesthetics_scores": {"composition": 0.95, "appeal": 0.95},
        "findings": "The image fully satisfies the request.",
        "suggested_fixes": "None required.",
    }


def test_rubber_stamp_guard_enabled_forces_no(monkeypatch):
    monkeypatch.setenv("AGENTIC_JUDGE_RUBBER_STAMP_GUARD", "1")
    parsed = normalize_judge_payload(_flat_high_payload())

    assert parsed is not None
    assert parsed["rubber_stamp"] is True
    assert parsed["good_enough"] is False
    assert parsed["correctness"] == 0.8
    assert parsed["aesthetics"] == 0.8


def test_rubber_stamp_guard_disabled_uses_scores_for_verdict(monkeypatch):
    monkeypatch.setenv("AGENTIC_JUDGE_RUBBER_STAMP_GUARD", "0")
    parsed = normalize_judge_payload(_flat_high_payload())

    assert parsed is not None
    assert parsed["rubber_stamp"] is False
    assert parsed["good_enough"] is True
    # snap_score maps raw [0.9, 1.0) to the 0.8 band; disabling the guard
    # removes the forced-NO behavior, not score-grid normalization.
    assert parsed["correctness"] == 0.8
    assert parsed["aesthetics"] == 0.8
