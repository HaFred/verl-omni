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
"""Plan-mode protocol primitives shared by the reward, the builder, and the loop.

Two unrelated problems live here because they are two halves of the same protocol.

**Reading a plan.** ``plan`` rows ask the policy to write a numbered list of subtask
prompts, and the ``plan`` reward dim grades those lines against reference subtasks. The
loop also needs the same predicate to tell "the model wrote its plan" from "the model
skipped straight to a tool call", so the line grammar has one definition here instead of
one per consumer.

**Writing the reference.** UniCoT-Breakdown subtasks were authored for a *image edit*
tool: from subtask 1 onward every row opens by asserting the previous render is preserved
("Keep the outline of the image unchanged and edit with the following details. …"). This
harness has no edit tool — ``generate_image`` is stateless text-to-image — so a subtask
carried forward to step *i* must restate everything steps 0..*i* asked for, and the
edit-tool lead-in is an instruction the tool cannot act on. :func:`cumulative_subtasks`
therefore drops the lead-in and accumulates, which is what the reward should be measuring
against and what the system prompt asks the policy to produce.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = [
    "MIN_PLAN_LINE_TOKENS",
    "cumulative_subtasks",
    "plan_lines_from_prose",
    "strip_edit_lead_in",
]

#: A plan line needs enough distinct words to be a subtask prompt rather than a heading.
MIN_PLAN_LINE_TOKENS = 4

#: Word-ish tokens, matching ``agentic_multidim_reward._tokens`` so the "≥ 4 tokens"
#: filter here and the coverage metric there agree on what counts as content.
_TOKEN_RE = re.compile(r"[a-z0-9_']+")
_PLAN_HEADER_RE = re.compile(r"\bPlan\s*:", re.IGNORECASE)
_PLAN_LINE_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+(.+)$")


def plan_lines_from_prose(prose: str) -> list[str]:
    """Extract the numbered/bulleted subtask prompts from already-stripped prose.

    A ``Plan:`` header, when present, bounds where the list starts; otherwise the whole
    text is scanned, because a policy that omits the header is still emitting a plan and
    the reward should see it.

    Args:
        prose: Assistant text with tool-call payloads already removed, line breaks intact.

    Returns:
        Plan item texts in order, each with at least :data:`MIN_PLAN_LINE_TOKENS`
        distinct word tokens. Empty when no plan is present.
    """
    header = _PLAN_HEADER_RE.search(prose or "")
    body = prose[header.end() :] if header else prose or ""
    return [
        line
        for match in _PLAN_LINE_RE.finditer(body)
        if len(set(_TOKEN_RE.findall((line := match.group(1).strip()).lower()))) >= MIN_PLAN_LINE_TOKENS
    ]


#: Words that open a "preserve the previous render, then edit it" clause.
_PRESERVE_START = r"(?:keep|maintain|preserve|retain|do not change|without changing)"
_EDIT_VERB = r"(?:edit|add|refine|make|apply|render|enhance|integrate|illuminate|ensure|adjust|update)"

#: ``… and edit with the following details. <content>`` — the dominant subtask-1 form,
#: plus the ``make the following edits:`` and ``add the following details:`` variants.
#: Bounded to one sentence (``[^.!?]``, no ``DOTALL``): an unbounded ``.*?`` would find a
#: later ``these details`` and swallow the real content between the two.
_DETAILS_CLAUSE_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?\b(?:following|these)\b[^.!?]*?\b(?:details?|edits?)\b\s*[:.;]?\s*",
    re.IGNORECASE,
)
#: ``Keep all previously rendered elements unchanged. <content>`` — the dominant
#: subtask-2 form. The preserve sentence ends before the content starts.
_PRESERVE_SENTENCE_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?\b(?:unchanged|the same|as (?:is|before)|intact)\b\s*[.!?]\s*",
    re.IGNORECASE,
)
#: ``Without changing any composed details, apply …`` — preserve clause and content share
#: one comma-joined sentence, so split at the comma before the edit verb.
_PRESERVE_COMMA_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?,\s*(?={_EDIT_VERB}\b)(?P<content>.+)$",
    re.IGNORECASE | re.DOTALL,
)
#: ``Maintain all previously rendered elements, and make sure …`` — preserve clause and
#: content share one sentence, so split at the conjunction before the edit verb.
_PRESERVE_CONJUNCTION_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?(?:,\s*|\s+)and\s+(?={_EDIT_VERB}\b)(?P<content>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_LEAD_IN_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (_DETAILS_CLAUSE_RE, False),
    (_PRESERVE_SENTENCE_RE, False),
    (_PRESERVE_COMMA_RE, True),
    (_PRESERVE_CONJUNCTION_RE, True),
)
#: A strip that removes most of the text is a mis-parse, not a lead-in, so the original is
#: kept. Deliberately low: a legitimate subtask can have a long preserve clause and one
#: short content sentence, and the patterns are already bounded to the opening sentence.
_MIN_RETAINED_FRACTION = 0.15


def strip_edit_lead_in(subtask: str) -> str:
    """Drop an image-edit lead-in from a source subtask, keeping the content.

    Args:
        subtask: Raw UniCoT-Breakdown subtask text.

    Returns:
        The subtask content with the leading "preserve the previous render and edit it"
        clause removed and a dangling conjunction cleaned up. Returned unchanged when the
        text carries no lead-in, when stripping would empty it, or when the strip would
        retain less than :data:`_MIN_RETAINED_FRACTION` of the text (a mis-parse).
    """
    text = (subtask or "").strip()
    if not text:
        return text
    for pattern, uses_content_group in _LEAD_IN_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        retained = (match.group("content") if uses_content_group else text[match.end() :]).strip()
        retained = re.sub(r"^(?:,|and|then)\s+", "", retained, flags=re.IGNORECASE).strip()
        if not retained or len(retained) < _MIN_RETAINED_FRACTION * len(text):
            continue
        return retained
    return text


def cumulative_subtasks(subtasks: Sequence[str]) -> tuple[str, ...]:
    """Flatten edit-style subtasks into standing self-contained prompts.

    The harness generates with a stateless text-to-image tool, so step *i* cannot be
    conditioned on the render from step *i-1*. Each returned prompt therefore restates
    every earlier subtask, which is the only way the accumulated plan survives to the
    tool.

    Args:
        subtasks: Source subtasks in order, without any sentinel.

    Returns:
        One prompt per source subtask, where element *i* is subtasks 0..*i* joined with
        spaces and stripped of edit-tool lead-ins. Empty input returns ``()``.
    """
    cleaned = [str(subtask).strip() for subtask in subtasks if str(subtask).strip()]
    if not cleaned:
        return ()
    for index in range(1, len(cleaned)):
        cleaned[index] = strip_edit_lead_in(cleaned[index])
    return tuple(" ".join(cleaned[: index + 1]) for index in range(len(cleaned)))
