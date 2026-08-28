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
"""CPU tests for fail-closed UniCoT-Breakdown direct parsing."""

import copy

import pytest

from verl_omni.utils.dataset.visual_reflection import RejectionReason, VisualReflectionDataError
from verl_omni.utils.dataset.visual_reflection.unicot_breakdown import parse_unicot_breakdown_record


def make_breakdown_row(*, subtask_count: int, data_id: str = "fixture") -> dict:
    """Create a valid row in the public UniCoT-Breakdown field shape."""
    subtasks: list[str | None] = [f"Subtask {index} detail." for index in range(subtask_count)]
    subtasks += [None] * (3 - subtask_count)
    images: list[str | None] = [
        f"./images/{data_id}_breakdown_subtask_{index + 1}.png" for index in range(subtask_count)
    ]
    images += [None] * (3 - subtask_count)
    return {
        "data_id": data_id,
        "prompt": "A complex visual prompt.",
        "subtasks": subtasks,
        "subtask_images": images,
    }


def make_no_breakdown_row(data_id: str = "fixture") -> dict:
    return {
        "data_id": data_id,
        "prompt": "A simple visual prompt.",
        "subtasks": ["No breakdown needed.", None, None],
        "subtask_images": [None, None, None],
    }


@pytest.mark.parametrize("subtask_count", [1, 2, 3])
def test_parse_valid_plan_rows(subtask_count):
    record = parse_unicot_breakdown_record(make_breakdown_row(subtask_count=subtask_count), manifest_id="m")

    assert record.task_type == "plan"
    assert record.plan_expected is True
    assert record.expected_num_images == subtask_count
    assert len(record.subtasks) == subtask_count
    assert record.subtasks == tuple(f"Subtask {i} detail." for i in range(subtask_count))
    assert len(record.subtask_images) == subtask_count
    assert record.prompt == "A complex visual prompt."


def test_parse_no_breakdown_row_is_a_reflect_task_with_one_image():
    record = parse_unicot_breakdown_record(make_no_breakdown_row(), manifest_id="m")

    assert record.task_type == "reflect"
    assert record.plan_expected is False
    assert record.expected_num_images == 1
    assert record.subtasks == ()
    assert record.subtask_images == ()


def test_no_breakdown_sentinel_is_whitespace_case_normalized():
    row = make_no_breakdown_row()
    row["subtasks"][0] = "  NO  breakdown\nneeded.  "

    record = parse_unicot_breakdown_record(row, manifest_id="m")

    assert record.task_type == "reflect"


def test_parser_does_not_mutate_source_record():
    row = make_breakdown_row(subtask_count=2)
    before = copy.deepcopy(row)

    parse_unicot_breakdown_record(row, manifest_id="m")

    assert row == before


def test_source_record_override_must_match_public_data_id():
    row = make_breakdown_row(subtask_count=1, data_id="source-id")

    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_breakdown_record(row, manifest_id="m", source_record_id="wrong-id")

    assert error.value.reason is RejectionReason.DUPLICATE_SOURCE_RECORD


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.__setitem__("subtasks", None), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row: row["subtasks"].__setitem__(0, None), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row: row["subtasks"].__setitem__(1, ""), RejectionReason.INVALID_FIELD_TYPE),
        # Gap: null slot (index 1) before a non-null subtask breaks the prefix rule.
        (lambda row: row["subtasks"].__setitem__(2, "Sneaky third subtask."), RejectionReason.CONTRADICTORY_TERMINAL),
        (lambda row: row["subtasks"].pop(), RejectionReason.LENGTH_MISMATCH),
        (lambda row: row["subtasks"].append(None), RejectionReason.LENGTH_MISMATCH),
        (lambda row: row.__setitem__("prompt", ""), RejectionReason.EMPTY_PROMPT),
        (lambda row: row.pop("prompt"), RejectionReason.MISSING_FIELD),
        (lambda row: row.__setitem__("subtask_images", "nope"), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row: row["subtask_images"].pop(), RejectionReason.LENGTH_MISMATCH),
        (lambda row: row["subtask_images"].__setitem__(0, 42), RejectionReason.INVALID_FIELD_TYPE),
        # Image present for a null subtask slot (plan row with one subtask).
        (
            lambda row: row["subtask_images"].__setitem__(1, "./images/ghost.png"),
            RejectionReason.CONTRADICTORY_TERMINAL,
        ),
    ],
)
def test_malformed_rows_fail_closed(mutation, reason):
    row = make_breakdown_row(subtask_count=1)
    mutation(row)

    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_breakdown_record(row, manifest_id="m")

    assert error.value.reason is reason


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        # A no-breakdown row cannot carry plan text in later slots.
        (lambda row: row["subtasks"].__setitem__(1, "Extra plan."), RejectionReason.CONTRADICTORY_TERMINAL),
        (lambda row: row["subtask_images"].__setitem__(0, "./images/x.png"), RejectionReason.CONTRADICTORY_TERMINAL),
        (lambda row: row.__setitem__("prompt", None), RejectionReason.INVALID_FIELD_TYPE),
    ],
)
def test_malformed_no_breakdown_rows_fail_closed(mutation, reason):
    row = make_no_breakdown_row()
    mutation(row)

    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_breakdown_record(row, manifest_id="m")

    assert error.value.reason is reason


def test_parser_rejects_non_mapping_records():
    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_breakdown_record("not-a-row", manifest_id="m")  # type: ignore[arg-type]

    assert error.value.reason is RejectionReason.INVALID_FIELD_TYPE


def test_real_dataset_shape_parses():
    """A row with the exact public snapshot shape (list length 3) must parse."""
    row = {
        "data_id": "00061139_detailed_prompt",
        "prompt": "A high-speed bullet train speeding along a straight railroad track.",
        "subtasks": ["Subtask one.", "Subtask two.", None],
        "subtask_images": ["./images/a_1.png", "./images/a_2.png", None],
    }

    record = parse_unicot_breakdown_record(row, manifest_id="m")

    assert record.task_type == "plan"
    assert record.expected_num_images == 2
