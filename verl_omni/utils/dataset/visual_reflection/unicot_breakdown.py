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
"""Fail-closed direct parser for public UniCoT-Breakdown rows.

A breakdown row is either:
- ``plan``: ``prompt`` decomposed into 1..3 non-None ``subtasks`` (progressive
  multi-image composition), ``subtask_images`` aligned per subtask; or
- ``reflect``: ``subtasks[0] == "No breakdown needed."`` — a single-image task
  with no planning demands (``expected_num_images=1``).

Actions/task types are derived only from the subtask list structure, never from
natural-language phrases. The adapter is text-only: it validates and
canonicalizes records but never touches pixels (reward scoring for the agentic
RL run is text-grounded; image resolution is optional and lives in ``images.py``).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import (
    RejectionReason,
    VisualReflectionDataError,
    require_nonempty_text,
)

UNICOT_BREAKDOWN_DATASET_ID = "Fr0zencr4nE/UniCoT-Breakdown-3K"
UNICOT_BREAKDOWN_PARSER_VERSION = "subtask_structure_v1"
DEFAULT_NO_BREAKDOWN_SENTINEL = "No breakdown needed."
# Public rows keep exactly three subtask slots; a valid plan uses a non-None prefix.
MAX_SUBTASK_SLOTS = 3


@dataclass(frozen=True)
class UniCoTBreakdownRecord:
    """Canonical validated row: what the agentic RL run consumes."""

    data_id: str
    prompt: str
    task_type: str  # "plan" | "reflect"
    subtasks: tuple[str, ...]  # empty for reflect rows
    subtask_images: tuple[str | None, ...]  # aligned with subtasks; empty for reflect rows
    plan_expected: bool
    expected_num_images: int


def breakdown_converter_config() -> dict[str, Any]:
    """Return the exact UniCoT-Breakdown filter settings for a manifest."""
    return {
        "unicot_breakdown_parser": UNICOT_BREAKDOWN_PARSER_VERSION,
        "unicot_breakdown_no_breakdown_sentinel": _normalize_sentinel(DEFAULT_NO_BREAKDOWN_SENTINEL),
        "unicot_breakdown_max_subtask_slots": MAX_SUBTASK_SLOTS,
    }


def parse_unicot_breakdown_record(
    record: Mapping[str, Any],
    *,
    manifest_id: str,
    source_record_id: str | None = None,
) -> UniCoTBreakdownRecord:
    """Parse one source row into a validated canonical breakdown record."""
    del manifest_id  # reserved for manifest provenance parity with unicot.py
    if not isinstance(record, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "UniCoT-Breakdown record must be a mapping",
        )
    dataset_record_id = require_nonempty_text(
        _required_field(record, "data_id", source_record_id or "<unknown>"),
        field="source_record_id",
        source_record_id=source_record_id,
    )
    if source_record_id is not None:
        override_record_id = require_nonempty_text(source_record_id, field="source_record_id")
        if override_record_id != dataset_record_id:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                "source_record_id override does not match UniCoT-Breakdown data_id",
                field="source_record_id",
                source_record_id=override_record_id,
            )
    record_id = dataset_record_id
    prompt = require_nonempty_text(
        _required_field(record, "prompt", record_id),
        field="prompt",
        reason=RejectionReason.EMPTY_PROMPT,
        source_record_id=record_id,
    )

    subtasks = _required_list(record, "subtasks", record_id)
    if not subtasks:
        raise VisualReflectionDataError(
            RejectionReason.LENGTH_MISMATCH,
            "subtasks must contain at least one slot",
            field="subtasks",
            source_record_id=record_id,
        )
    if len(subtasks) > MAX_SUBTASK_SLOTS:
        raise VisualReflectionDataError(
            RejectionReason.LENGTH_MISMATCH,
            f"subtasks has {len(subtasks)} slots but the public schema keeps {MAX_SUBTASK_SLOTS}",
            field="subtasks",
            source_record_id=record_id,
        )
    subtask_images = _optional_image_list(record, "subtask_images", record_id, expected=len(subtasks))

    if _is_no_breakdown(subtasks[0]):
        # Single-image row: the source itself says no breakdown is needed.
        for index, value in enumerate(subtasks[1:], start=1):
            if value is not None:
                raise VisualReflectionDataError(
                    RejectionReason.CONTRADICTORY_TERMINAL,
                    f"subtasks[{index}] is non-null on a no-breakdown row",
                    field=f"subtasks[{index}]",
                    source_record_id=record_id,
                )
        for index, image in enumerate(subtask_images):
            if image is not None:
                raise VisualReflectionDataError(
                    RejectionReason.CONTRADICTORY_TERMINAL,
                    f"subtask_images[{index}] is non-null on a no-breakdown row",
                    field=f"subtask_images[{index}]",
                    source_record_id=record_id,
                )
        return UniCoTBreakdownRecord(
            data_id=record_id,
            prompt=prompt,
            task_type="reflect",
            subtasks=(),
            subtask_images=(),
            plan_expected=False,
            expected_num_images=1,
        )

    # Plan row: a non-None prefix of subtasks, then None slots only.
    first_none = next((index for index, value in enumerate(subtasks) if value is None), len(subtasks))
    if first_none == 0:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "subtasks[0] is empty (expected the no-breakdown sentinel or plan text)",
            field="subtasks[0]",
            source_record_id=record_id,
        )
    for index in range(first_none):
        require_nonempty_text(
            subtasks[index],
            field=f"subtasks[{index}]",
            source_record_id=record_id,
        )
    for index in range(first_none, len(subtasks)):
        if subtasks[index] is not None:
            raise VisualReflectionDataError(
                RejectionReason.CONTRADICTORY_TERMINAL,
                f"subtasks[{index}] is non-null after a null slot (plan subtasks must form a prefix)",
                field=f"subtasks[{index}]",
                source_record_id=record_id,
            )
    plan_subtasks = [
        require_nonempty_text(value, field=f"subtasks[{i}]", source_record_id=record_id)
        for i, value in enumerate(subtasks[:first_none])
    ]
    for index, image in enumerate(subtask_images[:first_none]):
        if image is not None:
            require_nonempty_text(image, field=f"subtask_images[{index}]", source_record_id=record_id)
    for index in range(first_none, len(subtask_images)):
        if subtask_images[index] is not None:
            raise VisualReflectionDataError(
                RejectionReason.CONTRADICTORY_TERMINAL,
                f"subtask_images[{index}] is non-null for a null subtask slot",
                field=f"subtask_images[{index}]",
                source_record_id=record_id,
            )
    return UniCoTBreakdownRecord(
        data_id=record_id,
        prompt=prompt,
        task_type="plan",
        subtasks=tuple(plan_subtasks),
        subtask_images=tuple(subtask_images[:first_none]),
        plan_expected=True,
        expected_num_images=len(plan_subtasks),
    )


def _required_field(record: Mapping[str, Any], field: str, source_record_id: str) -> Any:
    if field not in record:
        raise VisualReflectionDataError(
            RejectionReason.MISSING_FIELD,
            f"UniCoT-Breakdown record is missing {field}",
            field=field,
            source_record_id=source_record_id,
        )
    return record[field]


def _required_list(record: Mapping[str, Any], field: str, source_record_id: str) -> list[Any]:
    value = _required_field(record, field, source_record_id)
    if not isinstance(value, list):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a list",
            field=field,
            source_record_id=source_record_id,
        )
    return value


def _optional_image_list(
    record: Mapping[str, Any],
    field: str,
    source_record_id: str,
    *,
    expected: int,
) -> list[str | None]:
    if field not in record:
        return [None] * expected
    value = record[field]
    if not isinstance(value, list):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a list",
            field=field,
            source_record_id=source_record_id,
        )
    if len(value) != expected:
        raise VisualReflectionDataError(
            RejectionReason.LENGTH_MISMATCH,
            f"{field} has length {len(value)} but expected {expected}",
            field=field,
            source_record_id=source_record_id,
        )
    for index, item in enumerate(value):
        if item is not None and not isinstance(item, str):
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                f"{field}[{index}] must be a string or null",
                field=f"{field}[{index}]",
                source_record_id=source_record_id,
            )
    return value


def _is_no_breakdown(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return _normalize_sentinel(value) == _normalize_sentinel(DEFAULT_NO_BREAKDOWN_SENTINEL)


def _normalize_sentinel(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
