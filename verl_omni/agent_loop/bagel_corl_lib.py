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

"""Hermes tool protocol, J×K IDs, serial episode, and flatten for Bagel Co-RL (Joint-Training)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from verl_omni.agent_loop.rpco_turn_protocol import (
    build_forced_reflection,
    derive_good_enough_from_scores,
    format_rm_scores_as_judge_text,
)

# Single trajectory-state layer (audit T1.6): the bagel lane shares the Mode-2a
# tool loop's registry/latch/context under tools.trajectory. The legacy
# agent_loop/image_gen_trajectory_context module remains only for its own
# deprecated consumers and must not be imported here.
from verl_omni.agent_loop.utils import derive_rollout_seed
from verl_omni.tools.trajectory.artifacts import build_generate_call_meta, register_tool_artifact
from verl_omni.tools.trajectory.context import get_active_user_prompt
from verl_omni.tools.trajectory.judge_latch import (
    clear_good_enough_yes_reached,
    get_good_enough_yes_reached,
    set_good_enough_yes_reached,
)

logger = logging.getLogger(__name__)

# ``BAGEL_CORL_DEBUG=1`` re-enables this package's INFO diagnostics. vLLM calls
# ``logging.basicConfig(level=ERROR)`` in every process that imports it, which raises the
# *root* level and silently swallows the ``bagel_corl_sync`` lifecycle traces, the
# per-turn timing below and ``_wake_und_rollout_replicas``' confirmation -- the driver
# log goes quiet at the exact moment the engines start, so a stalled rollout looks like
# an idle one. An explicit level on this package's logger wins over the root's, and
# leaves vLLM's own verbosity untouched.
def _force_info_logging() -> None:
    """Make this process actually *emit* INFO diagnostics.

    Setting a child logger's level is not enough. ``vllm`` (imported lazily, on the first
    decode, i.e. *after* the episode has already printed its first lines) calls
    ``logging.basicConfig(..., force=True)``, which REPLACES the root handlers with one
    carrying vLLM's own level. From that moment every record below that level is dropped
    even though ``verl_omni``/``verl`` loggers still say INFO -- which is exactly what a
    diagnostic run must not do.

    Measured 2026-09-20 20:23:07 on hk01dgx039 (devices 3,5,6,7): worker pids 1666904 and
    1666910 printed ``bagel_corl_turn turn=1`` + ``bagel_dual_role_generate`` for the
    step-0 rollout and then *nothing* for 25 minutes, while ``ray.util.state.list_tasks``
    showed their ``generate_sequences`` calls completing and the AR engine serving decodes
    -- the rollout was running fine, its log was swallowed. Fix: raise the root logger AND
    every root handler (including vLLM's replacement) to INFO, and re-install a handler if
    ``force=True`` ever left none behind.
    """
    level = logging.INFO
    for name in ("verl_omni", "verl"):
        logging.getLogger(name).setLevel(level)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)
    if not root.handlers:
        logging.basicConfig(level=level, force=True)


if os.getenv("BAGEL_CORL_DEBUG") == "1":
    _force_info_logging()
# Per-turn trace of the serial UND/GEN episode. Off by default: one line per decode is
# too noisy for a long run, but it is the only way to tell "the AR engine is slow" from
# "the client never got its response" when an episode stops making progress.
_BAGEL_CORL_TURN_DEBUG = os.getenv("BAGEL_CORL_DEBUG") == "1"

GENERATE_IMAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image from a text prompt using the Bagel GEN pathway.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt for image generation."},
            },
            "required": ["prompt"],
        },
    },
}

HERMES_SPECIAL_TOKENS: tuple[str, ...] = (
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Tagged-call dialect (``<tools>{...}</tools>``). NOT a synonym for ``<tool_call>``: ``<tools>`` is
# the tag the recipe's prompt uses to *describe* the available functions, so the checkpoint
# re-emits it as its own call tag. See ``_tools_tagged_call``.
_TAGGED_CALL_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
_DONE_RE = re.compile(r"\bDone\.\s*$", re.IGNORECASE)

# Role labels the checkpoint sometimes re-emits as *text* before a payload
# ("ASSISTANT\n {json}"), and control/tag markers it echoes back from the prompt itself
# ("<tools>\n {json}"). Both are stripped before a bare-payload call is recognized; prose is
# not, which is what keeps a JSON example inside the model's plan from matching.
_ROLE_ECHO_RE = re.compile(r"^\s*(?:assistant|system|user)\s*[:\-]?\s*", re.IGNORECASE)
_LEADING_MARKER_RE = re.compile(r"^\s*(?:<[^>{}\n]{0,48}>\s*)+")

# Fenced dialect (see ``parse_und_tool_call``). The optional info string (```json) is dropped.
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+.-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
# Longest label a fenced call may carry in front of its fence (measured: "Content Request: ",
# 17 chars). The bound is what separates "the turn *is* a labelled call" from "a prose plan that
# happens to quote one": a quoted example sits inside paragraphs, a call does not. It is applied
# per fenced block -- the text between one fence and the next -- so a turn that lists one fenced
# object per step (a plan, then the call) still counts, while a lone quoted example does not.
_FENCED_PREAMBLE_LIMIT = 120

_GENERATE_IMAGE_TOOL = "generate_image"
# The recipe's system prompt asks the UND lane to ``judge_image`` after the last image, but the
# loop's tool registry has only ``generate_image`` (judging is the RM's job, see
# ``bagel_corl_rm``). A *Hermes*-tagged unsupported tool stays fail-closed
# (``test_unsupported_tool_is_fail_closed``); every other dialect the checkpoint actually samples
# (bare, fenced, and ``<tools>``) treats ``judge_image`` as a non-action instead of killing the
# episode, because there the protocol's judge turn is routine.
_INERT_BARE_TOOLS = frozenset({"judge_image"})


class GenerateImageCapError(RuntimeError):
    """Raised when a second GEN call is attempted under ``max_generate_passes=1``."""


def parse_hermes_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a Hermes ``<tool_call>{...}</tool_call>`` span.

    Returns:
        Parsed JSON object, or ``None`` if no tool call is present.
    """
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    payload = json.loads(match.group(1))
    if "name" not in payload and "function" in payload:
        payload = payload["function"]
    return payload


def _leading_json_object(text: str) -> dict[str, Any] | None:
    """Parse the JSON object that starts ``text`` (ignoring anything that follows it).

    A brace counter that respects string state, so a ``prompt`` containing ``{``/``}`` or an
    escaped quote does not end the scan early.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def _named_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Accept ``{"function": {...}}`` (OpenAI shape) and require an action ``name``.

    A payload with no ``name`` is *not* a call: that is the guard that keeps a JSON example
    inside the model's prose from being executed.
    """
    if payload is None:
        return None
    if "name" not in payload and "function" in payload:
        function = payload["function"]
        payload = function if isinstance(function, dict) else payload
    return payload if "name" in payload else None


def _labelled_fenced_call(text: str) -> dict[str, Any] | None:
    """Parse the first *labelled* fenced block in ``text`` that names a tool.

    The label bound is measured **per block** -- the text between one fence and the next -- and
    not against the whole prefix. That distinction is load-bearing for a turn that lists one
    fenced object per step:

        Sure, here's your request processed:

        1. Plan:
        ```json
        {"subtasks": [{"prompt": "A chaotic anime-style cartoon ..."}]}
        ```
        2. Generate Image:
        ```json
        {"name": "generate_image", "arguments": {"prompt": "A chaotic anime-style cartoon ..."}}
        ```

    Measured 2026-09-21 07:50 on hk01dgx039 (devices 2-5, ``bagel_corl_pr1`` step 1). The call
    sits in the *second* fence behind the short label ``"\\n2. Generate Image:\\n"``, while the
    first fence is the plan and carries no ``name``. Reading only the first block missed the
    call, and measuring the label from the start of the turn would have too -- the plan's braces
    trip the bound. ``und_turn_kind`` then returned ``continue`` on every turn, ``K`` stayed 0
    and the GEN lane never ran (``gen/num_rows: 0``, ``gen/skipped_no_groups: 1``).

    A single *quoted* example still does not match: with one fence the label is the whole prose
    plan, which exceeds ``_FENCED_PREAMBLE_LIMIT``.
    """
    cursor = 0
    for match in _FENCED_BLOCK_RE.finditer(text):
        label = text[cursor:match.start()]
        cursor = match.end()
        if len(label) > _FENCED_PREAMBLE_LIMIT or "{" in label or "}" in label:
            continue
        payload = _named_payload(_leading_json_object(match.group(1)))
        if payload is not None:
            return payload
    return None


def _tools_tagged_call(text: str) -> dict[str, Any] | None:
    """Parse the first ``<tools>{...}</tools>`` span that carries a *complete call*.

    The checkpoint re-uses the prompt's own schema tag. The recipe's system prompt lists the tools
    as ``You are provided with function signatures within <tools></tools> XML tags`` and then
    ``For each function call, return a json object ... within <tool_call></tool_call> XML tags``.
    Both tags are in context, but only ``<tools>`` is ever *shown carrying a tool*, so the
    checkpoint emits ``<tools>`` for its calls. Measured 2026-09-21 15:36 on hk01dgx039 (devices
    2-5, ``bagel_corl_pr1``), a turn that makes a perfectly good call and is thrown away:

        Sure, I've got the plan ready!

        <plan>
          1. Create an image depicting a destroyed school in an anime style.
          ...
        </plan>

        <tools>
          {"name": "generate_image", "arguments": {"prompt": "A chaotic anime-style cartoon ..."}}
        </tools>

    Neither of the pre-existing untagged paths can see this. The bare path requires the payload to
    *start* the turn (after a role echo and echoed ``<...>`` markers), and here it sits behind a
    prose sentence plus a ``<plan>`` block; the fenced path needs a ``` fence, and there is none.
    So ``und_turn_kind`` read the turn as prose -- which is not cosmetic: with the call dropped the
    loop issued no GEN request, appended no image observation, and the model went on to *invent*
    the exchange, emitting ``<output>{...}</output>`` blocks as if the tools had answered and
    repeating ``judge_image`` with empty arguments until the token budget was gone. ``K`` stayed 0,
    ``gen_lane_skipped`` was true and the reward was a flat 0 on all 90 steps of that run.

    Only ``<tools>`` is a call tag. ``<output>`` is deliberately *not* accepted: it is the model
    role-playing the tool's side of the conversation, and it is where the degenerate repeats live
    (``judge_image`` with ``arguments: {}``, dozens of times). Executing that would turn a
    hallucinated transcript into real GEN requests.

    The ``arguments`` requirement is what separates a call from a *schema echo*: a call carries
    ``arguments``, whereas the prompt's own function signature carries ``parameters``/``description``
    and no ``arguments``. Without it, a turn that quotes the prompt's ``<tools>`` schema block
    parses as a call, and ``run_serial_episode`` would read ``arguments.get("prompt", "")`` off the
    schema's ``parameters`` -- a GEN request with an empty prompt.

    Every ``<tools>`` block is considered in turn, so a turn that plans first and calls later still
    matches; the first block carrying a real call wins.
    """
    for match in _TAGGED_CALL_RE.finditer(text):
        payload = _named_payload(_leading_json_object(match.group(1)))
        if payload is None or "arguments" not in payload:
            continue
        return payload
    return None


def parse_und_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a tool call emitted by the UND lane -- tagged, ``<tools>``, bare, or fenced.

    Four dialects have to be accepted, because the *same* published checkpoint produced all of
    them here:

    * Hermes, ``<tool_call>{"name": ...}</tool_call>`` -- what ``HERMES_SPECIAL_TOKENS``, the
      spike gate and the RFC call the contract, and what the chat template re-renders from
      ``message.tool_calls``.
    * the **bare payload**, ``{"name": "generate_image", "arguments": {...}}`` with no
      surrounding tags -- what the checkpoint actually samples once it is handed the ``<tools>``
      block. Measured 2026-09-21 21:45 on hk01dgx039 with the schema in the prompt:

          bagel_corl_turn ... turn=1 kind=continue out_tokens=79
            text='{"name": "generate_image", "arguments": {"prompt": "Full body portrait of ...'

      and, one turn later, ``'ASSISTANT\\n {"name": "judge_image", ...}<|im_end|>'`` -- the model
      writing role labels as *text*. ``text`` there is ``tokenizer.decode(token_ids,
      skip_special_tokens=False)`` on the engine's raw ids, and other turns of the same episode
      still contain ``<|im_end|>``, so nothing strips tags on the way out: the tags are simply
      never sampled. Without a tolerant parse, ``und_turn_kind`` sees ``continue`` on every
      turn, K stays 0 and the GEN half of the Co-RL recipe never runs
      (``gen/num_rows: 0`` / ``gen/skipped_no_groups: 1``).
    * the **fenced payload** behind a short label. Measured 2026-09-21 23:05 on hk01dgx039
      (``run_bagel_diag.sh``, devices 3/5/6/7), same recipe, same weights:

          bagel_corl_turn episode=e032501e turn=1 kind=continue out_tokens=186 text='Content
            Request: \n```\n{\n  "name": "generate_image",\n  "arguments": {\n    "prompt":
            "A full body portrait of a handsome bald male ...'

      The payload is the turn's whole content -- the model *makes* the call, it does not quote
      one -- so it must be executed. Unparsed, the loop read the call as prose and kept decoding,
      and the checkpoint went on to role-play the next speaker (turn 2: ``'\nuser\nThanks for
      the feedback! Modify the prompt and generate a new image! ...'``) with K stuck at 0.
    * the **``<tools>`` payload** behind a prose plan. Measured 2026-09-21 15:36 on hk01dgx039
      (devices 2-5, ``bagel_corl_pr1``), a turn that leads with prose and a ``<plan>`` block
      before the call:

          Sure, I've got the plan ready!

          <plan>
            1. Create an image depicting a destroyed school in an anime style.
            ...
          </plan>

          <tools>
            {"name": "generate_image", "arguments": {"prompt": "A chaotic anime-style cartoon
              depiction of a destroyed school"}}
          </tools>

      This is the same checkpoint's most common shape, and it is the one that cost the whole run:
      the payload neither starts the text nor sits in a fence, so every turn read as ``continue``
      -- 0 GEN calls and a flat 0 reward on all 90 steps. See :func:`_tools_tagged_call` for why
      ``<output>`` is not accepted alongside it.

    Only a payload that *starts* the text (bare, after an optional role echo and echoed ``<...>``
    markers), that sits inside a ``<tools>`` block, or that is a fenced block naming a tool behind
    a label of at most ``_FENCED_PREAMBLE_LIMIT`` characters counting no braces counts; a JSON
    example inside the model's prose plan satisfies none of those, so it is not mistaken for a
    call. Every fenced block is considered in turn, each against its **own** label, because the
    checkpoint also emits a numbered plan whose call is a later fence than the plan itself.
    """
    tagged = parse_hermes_tool_call(text)
    if tagged is not None:
        return tagged
    head = _LEADING_MARKER_RE.sub("", _ROLE_ECHO_RE.sub("", text, count=1).lstrip())
    if head.startswith("{"):
        return _named_payload(_leading_json_object(head))
    tools_tagged = _tools_tagged_call(text)
    if tools_tagged is not None:
        return tools_tagged
    if "```" not in text:
        return None
    return _labelled_fenced_call(text)


#: Stop strings for the UND lane. The checkpoint role-plays the transcript: once it has made a
#: call it carries on writing the *tool's* side as an ``<output>`` block, repeats
#: ``judge_image`` with empty arguments dozens of times, and only then stops -- at the token cap.
#: ``<output>`` is never part of the protocol (the chat template wraps tool results in
#: ``<tool_response>``), so ending the turn there is safe and cuts the spiral at the call
#: boundary. vLLM excludes the stop string from the output, so a ``<tools>...</tools>`` call
#: still arrives complete and parseable.
UND_STOP_SEQUENCES: tuple[str, ...] = ("<output>",)

#: Fraction of the *remaining* context a single UND turn may consume.
UND_TURN_FRACTION: float = 0.5


def und_turn_max_tokens(
    *,
    context_used: int,
    max_context_tokens: int,
    fraction: float = UND_TURN_FRACTION,
) -> int:
    """Per-turn ``max_tokens`` for the UND lane.

    Without an explicit cap, ``ARStrategy.preprocess_input`` falls back to
    ``min(response_length, prompt_length + response_length - len(prompt))``. For the recipe's
    1024+1024 budget that is the *whole* episode response budget, so one degenerate turn ends
    the episode: measured 2026-09-22 on hk01dgx039 (devices 2-5), turn 1 spent all 1024 tokens
    emitting ``assistant\\n`` and the episode closed at ``response_tokens=1665`` with
    ``gen_calls=0`` -- no image observation, no second turn, nothing to train on.

    Capping each turn at ``fraction`` of what is *left* guarantees the loop keeps several
    attempts and always leaves room for a tool observation, while still comfortably fitting a
    full ``<plan>`` + ``<tools>`` call (measured ~120 tokens).

    Args:
        context_used: ``len(prompt_ids) + len(response_ids)`` already in the episode.
        max_context_tokens: the episode budget (``prompt_length + response_length``).
        fraction: share of the remaining budget one turn may take; ``1.0`` disables the cap.

    Returns:
        The per-turn token cap, or 0 when the budget is already spent.
    """
    remaining = int(max_context_tokens) - int(context_used)
    if remaining <= 0:
        return 0
    if fraction >= 1.0:
        return remaining
    if fraction <= 0.0:
        raise ValueError(f"und turn fraction must be in (0, 1], got {fraction!r}")
    # ceil: never hand back 0 while there is room, or the turn would be a no-op that still
    # consumes an episode step.
    share = max(1, math.ceil(remaining * float(fraction)))
    return min(remaining, share)


#: Prompt for the AR-replica health canary. Deliberately trivial: a working language model
#: answers it in a handful of varied tokens, so anything else is a finding.
UND_HEALTH_CANARY: str = "Name three fruits."

#: A canary answer shorter than this cannot be judged (``""`` is degenerate on its own; a
#: 2-3 token answer is just a terse model).
UND_HEALTH_MIN_WORDS: int = 6

#: Share of the canary answer one repeated token may occupy before the replica is called
#: degenerate. Measured on the corrupted AR replica 2026-09-21: ``' . . . . . . . . .'``
#: (share 1.00) and ``''``, against ``'Apples, bananas, and cherries.'`` (share 0.14) once the
#: sleep fix was in.
UND_HEALTH_REPEAT_SHARE: float = 0.4


def und_health_verdict(
    text: str,
    *,
    min_words: int = UND_HEALTH_MIN_WORDS,
    repeat_share: float = UND_HEALTH_REPEAT_SHARE,
) -> tuple[bool, float]:
    """Judge an AR-replica canary answer as coherent or degenerate.

    A corrupted UND replica does not raise -- it stops being a language model and collapses onto
    one token, so the *only* way to notice is to look at the text. This is that judgement, split
    out from the trainer so it is pinned on CPU instead of being re-derived by eye in a log.

    Args:
        text: the decoded canary answer.
        min_words: answers shorter than this are unjudgeable and called degenerate.
        repeat_share: a single repeated token taking more than this share is a loop.

    Returns:
        ``(degenerate, top_token_share)``. An empty answer is ``(True, 1.0)``.
    """
    words = text.split()
    if not words:
        return True, 1.0
    counts: dict[str, int] = {}
    for word in words:
        key = word.lower().strip(",.")
        counts[key] = counts.get(key, 0) + 1
    top_share = max(counts.values()) / len(words)
    degenerate = len(words) < min_words or top_share > repeat_share
    return degenerate, top_share


#: A UND turn must carry at least this many tokens before a repeat can be called a loop. Shorter
#: ``continue`` turns are legitimate (a terse plan, a clarifying sentence); the measured failure
#: emitted ``assistant\n`` hundreds of times.
UND_LOOP_MIN_TOKENS: int = 6


def count_degenerate_und_turns(turn_trace: list[dict[str, Any]]) -> int:
    """Count UND turns that collapsed onto one repeated token.

    This is the artifact-level oracle for a corrupted AR replica. An HTTP canary cannot do the
    job: this checkpoint only behaves when the ``<tools>`` schema is rendered into the system
    turn, and the omni chat endpoint rejects ``tools``, so a bare chat probe answers
    ``' . . . . . .'`` on a *healthy* replica too (measured 2026-09-22, post-fix, while the same
    run's rollouts were emitting valid ``generate_image`` calls). The real turns are the only
    honest signal.

    A ``generate_image`` turn is never counted -- producing the call is the success condition.
    A short ``continue`` turn is not counted either: it is only a loop when it keeps going and
    says the same thing, which is exactly the ``assistant\n`` spiral measured 2026-09-21 on
    ``hk01dgx039`` (every UND turn of every step, ``gen_calls=0``).

    Args:
        turn_trace: an episode's ``turn_trace`` (records with ``record``/``kind``/``text``).

    Returns:
        The number of looping UND turns.
    """
    loops = 0
    for record in turn_trace:
        if record.get("record") != "und_turn":
            continue
        if str(record.get("kind")) == "generate_image":
            continue
        text = str(record.get("text") or "")
        if len(text.split()) < UND_LOOP_MIN_TOKENS:
            continue
        degenerate, top_share = und_health_verdict(text)
        if degenerate and top_share > UND_HEALTH_REPEAT_SHARE:
            loops += 1
    return loops


def und_turn_kind(text: str) -> str:
    """Classify an UND decode: ``generate_image``, ``done``, or ``continue``.

    Accepts every tool-call dialect of :func:`parse_und_tool_call` (Hermes-tagged, ``<tools>``,
    bare, fenced). A *Hermes-tagged* unsupported tool stays fail-closed (the Qwen-tool guard the
    RFC asks for); the same name in the dialects the checkpoint actually samples
    (``judge_image``) is inert -- the recipe's prompt asks the lane to judge after the last image
    and the RM does that judging, so killing the episode there would drop rows for following the
    prompt.
    """
    tagged = parse_hermes_tool_call(text)
    call = tagged if tagged is not None else parse_und_tool_call(text)
    if call is not None:
        name = str(call.get("name", ""))
        if name == "generate_image":
            return "generate_image"
        if tagged is None and name in _INERT_BARE_TOOLS:
            logger.debug("bagel_corl_und_bare_inert_tool name=%s (no loop tool; RM judges)", name)
            return "continue"
        raise ValueError(f"Bagel CoRL UND emitted unsupported tool {name!r}; Qwen/other tools are fail-closed")
    if _DONE_RE.search(text.strip()):
        return "done"
    return "continue"


def bind_episode_ids(
    *,
    dataset_task_uid: str,
    episode_uid: str | None = None,
    policy_version: int = 0,
    gen_call_id: str | None = None,
) -> dict[str, str | int]:
    """RFC component-1 IDs. FlowGRPO groups by ``gen_group_uid``, never by semantic slot."""
    episode_uid = episode_uid or str(uuid.uuid4())
    gen_call_id = gen_call_id or str(uuid.uuid4())
    return {
        "dataset_task_uid": dataset_task_uid,
        "und_group_uid": dataset_task_uid,
        "episode_uid": episode_uid,
        "policy_version": int(policy_version),
        "gen_call_id": gen_call_id,
        "gen_group_uid": gen_call_id,
    }


def gen_sample_uid(gen_call_id: str, seed_index: int) -> str:
    return f"{gen_call_id}:{seed_index}"


def conditioning_uid(token_ids: list[int]) -> str:
    """RFC §4.4.2a ``cond_uid``: content hash of the conditioning token slice.

    Derived, never hand-set. This is the identity every reuse mechanism keys on
    (R2 here; R3/R5 later), so it must be exact: hashing the **token ids the GEN
    stage actually encodes** rather than the decoded text keeps the key stable
    and unambiguous. Two conditionings that tokenize identically are the same
    conditioning for attention purposes; a text-level hash would instead depend
    on whitespace/normalization round-trips and could make equal KVs look
    distinct (a silent cache miss) or, worse, collide.

    Reusing ``encode_prompt`` on the same ids is exactly what the conditioning
    cache does, so this key and the cache's own argument-derived key agree on
    which requests are interchangeable.
    """
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def cond_reuse_metrics(
    *,
    calls: int,
    hits: int | None,
    misses: int | None,
    bypassed: int | None = 0,
) -> dict[str, float]:
    """RFC §4.4.4 R2 metrics, from the engine's own conditioning-cache counters.

    ``PromptEmbedCache.stats()`` is the only authority on whether a conditioning
    was re-encoded, so these are **measured**, not modelled: a client-side count
    of "we submitted S seeds with one conditioning" would read ``1/S`` even when
    the cache is off or the seed fan-out missed the worker that holds the entry,
    which is precisely the failure §8.7 exists to catch.

    Definitions (RFC §4.4.4):

    * ``calls`` is the number of GEN **generate requests**, i.e. ``S`` per
      ``generate_image`` turn — the same granularity as ``PromptEmbedCache``'s own
      hit/miss counters. Dividing by the turn count instead would report ``1``
      (no reuse) for a perfectly warm cache.
    * ``gen/cond_recompute_ratio`` = conditioning prefills / requests, target
      ``1/S``. Prefills are misses **plus** bypasses: a bypassed call never
      consults the cache, so it always encoded (the cache wrapper bypasses on
      precomputed embeds or arguments it cannot hash safely).
    * ``gen/prompt_embed_cache_hit_rate`` = hits / (hits + misses), target
      ``1 - 1/S``. Bypasses are excluded from the denominator — they are not
      cache decisions. Absent when nothing consulted the cache.
    * ``gen/cond_amortization`` = ``1 / cond_recompute_ratio`` — the headline
      "one encode serves this many calls" number.

    Returns ``{}`` when the engine exposed no counters or the episode made no
    GEN request, so callers never publish a fabricated zero.
    """
    if calls <= 0 or hits is None or misses is None:
        return {}
    hits = max(0, int(hits))
    misses = max(0, int(misses))
    bypassed = max(0, int(bypassed or 0))
    prefills = misses + bypassed
    metrics: dict[str, float] = {
        "gen/cond_cache_hits": float(hits),
        "gen/cond_cache_misses": float(misses),
        "gen/cond_cache_bypassed": float(bypassed),
        "gen/cond_recompute_ratio": float(prefills) / float(calls),
    }
    consulted = hits + misses
    if consulted > 0:
        metrics["gen/prompt_embed_cache_hit_rate"] = float(hits) / float(consulted)
    ratio = metrics["gen/cond_recompute_ratio"]
    if ratio > 0.0:
        metrics["gen/cond_amortization"] = 1.0 / ratio
    return metrics


@dataclass
class GenSample:
    """One of S FlowGRPO seeds for a single ``generate_image`` turn."""

    gen_sample_uid: str
    gen_group_uid: str
    seed_index: int
    valid: bool
    prompt_token_ids: list[int]
    all_latents: Any | None = None
    timesteps: Any | None = None
    rollout_log_probs: Any | None = None
    rm_score: float | None = None
    image_path: str | None = None
    call_role: str = "initial"
    good_enough: bool | None = None
    # RFC §4.4.2a conditioning identity. Derived by the GEN path from the token
    # slice it actually encodes; constant across the S seeds of one call (that is
    # the reuse group R2 exploits) and equal across calls whose conditioning
    # coincides. ``None`` when the serving path did not report it.
    cond_uid: str | None = None
    cond_len: int = 0


@dataclass
class EpisodeRollout:
    """One serial UND episode (zero or more GEN calls).

    Runtime counters (RFC): ``turns`` is in-episode ``J`` (UND policy turns);
    ``num_gen_calls`` is in-episode ``K`` (``generate_image`` invocations). Invariant ``J >= K``.
    """

    und_group_uid: str
    episode_uid: str
    policy_version: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    turns: int
    # π_rollout per response position (the AR replica's own sampling log-probs). Published to the
    # TransferQueue as ``rollout_log_probs`` so the trainer's π_rollout/π_θ correction has a
    # rollout distribution; empty on a replay that never sampled (tests, pattern-3 stubs).
    rollout_log_probs: list[float] = field(default_factory=list)
    gen_samples: list[GenSample] = field(default_factory=list)
    used_image_credit: bool = False
    forced_reflection: bool = False
    und_reward: float = 0.0
    judge_text: str | None = None
    stop_required: bool = False
    num_gen_calls: int = 0
    # RFC §4.4.4 R2 metrics measured from the engine's conditioning-cache
    # counters for this episode's GEN calls. Empty when the engine exposed none.
    r2_metrics: dict[str, float] = field(default_factory=dict)
    # Per-turn trace: one record per UND decode (``kind``, ``text``, token counts) and one
    # per GEN call. Recorded unconditionally -- it is a few KB per episode -- because the
    # failure mode worth diagnosing is an episode that *never* reaches a
    # ``generate_image`` verdict: there K stays 0, ``gen/skipped_no_groups`` is 1, and the
    # only remaining evidence of what the UND lane actually emitted is this text.
    # ``bagel_corl.dump_episode_trace`` renders it under ``rollout_trajectories/``.
    turn_trace: list[dict[str, Any]] = field(default_factory=list)
    # Where :func:`~verl_omni.agent_loop.bagel_corl.dump_episode_trace` wrote this episode,
    # or ``None`` when the dump is disabled/failed. Set by the agent loop, not the lib.
    trace_dump_path: str | None = None

    def __post_init__(self) -> None:
        """π_rollout must stay positionally aligned with the response it describes.

        ``rollout_log_probs`` is read by the trainer per response position, so a
        length mismatch would silently pair a token with another token's log-prob
        (or truncate the correction) instead of failing. Empty is allowed: a
        replayed/deserialized episode that never sampled carries none, and the
        trainer simply does not get a rollout distribution for it.
        """
        if self.rollout_log_probs and len(self.rollout_log_probs) != len(self.response_ids):
            raise ValueError(
                "Bagel Co-RL rollout_log_probs must align with response_ids "
                f"(got {len(self.rollout_log_probs)} for {len(self.response_ids)} tokens)"
            )


class BagelGenerateImageTool:
    """One GEN turn per call (RFC ``K`` += 1).

    ``gen_samples_per_call`` is **S** (FlowGRPO seeds under the same conditioning),
    not the episode GEN-turn count **K**. One UND ``generate_image`` tool verdict
    invokes this once → one GEN turn; the seed fan-out is group sampling for FlowGRPO.
    ``max_generate_passes`` bounds how many such GEN turns an episode may enqueue.
    """

    def __init__(
        self,
        *,
        gen_samples_per_call: int,
        max_generate_passes: int = 1,
        generate_fn: Callable[..., Any] | None = None,
        seed_base: int = 0,
    ):
        if gen_samples_per_call < 1:
            raise ValueError("gen_samples_per_call must be >= 1")
        if max_generate_passes < 1:
            raise ValueError("max_generate_passes must be >= 1")
        self.s = int(gen_samples_per_call)
        self.max_generate_passes = int(max_generate_passes)
        self._passes = 0
        self._generate_fn = generate_fn
        # Episode-scoped seed base + call counter for the default FlowGRPO seeds.
        self._seed_base = int(seed_base)
        self._call_index = 0

    def remaining_passes(self) -> int:
        return max(0, self.max_generate_passes - self._passes)

    def _default_seeds(self, call_index: int) -> list[int]:
        """Fresh FlowGRPO seeds for one ``generate_image`` turn.

        These seeds pin the diffusion noise: the bagel pipeline does
        ``torch.manual_seed(req.sampling_params.seed)`` (vllm_omni
        ``pipeline_bagel.py``), so the *same* seed yields a byte-identical PNG.

        Returning the constant ``range(self.s)`` for every episode therefore made the
        whole GEN group a function of the prompt alone: measured 2026-09-21 on
        hk01dgx039, ``/tmp/bagel_corl_gen/`` held 42 PNGs of which only 22 were
        distinct -- eight groups of three byte-identical files written 22:15, 23:27
        and 02:17, i.e. by three separate runs. For FlowGRPO that is fatal: a
        constant group has no variance, so its advantages collapse and the GEN lane
        receives no learning signal regardless of how many seeds S it samples.

        Deriving from ``seed_base`` (the episode's rollout seed) keeps a run
        reproducible while making each episode's group independent.
        """
        return [
            derive_rollout_seed(self._seed_base, call_index * self.s + seed_index)
            for seed_index in range(self.s)
        ]

    async def __call__(
        self,
        *,
        prompt: str,
        prompt_token_ids: list[int],
        gen_call_id: str,
        seeds: list[int] | None = None,
    ) -> list[GenSample]:
        if self._passes >= self.max_generate_passes:
            raise GenerateImageCapError(
                f"max_generate_passes={self.max_generate_passes} refuses a second generate_image call"
            )
        self._passes += 1
        call_index = self._call_index
        self._call_index += 1
        if seeds is None:
            seeds = self._default_seeds(call_index)
        if len(seeds) != self.s:
            raise ValueError(f"expected S={self.s} FlowGRPO seeds for one GEN turn, got {len(seeds)}")
        raw_rows: list[dict[str, Any]]
        if self._generate_fn is None:
            raw_rows = [{"valid": True} for _ in seeds]
        else:
            maybe = self._generate_fn(
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                seeds=seeds,
                gen_call_id=gen_call_id,
            )
            raw_rows = await maybe if asyncio.iscoroutine(maybe) else maybe
        samples: list[GenSample] = []
        for seed_index, (seed, row) in enumerate(zip(seeds, raw_rows, strict=True)):
            valid = bool(row.get("valid", True))
            cond_uid = row.get("cond_uid")
            samples.append(
                GenSample(
                    gen_sample_uid=gen_sample_uid(gen_call_id, seed_index),
                    gen_group_uid=gen_call_id,
                    seed_index=seed_index,
                    valid=valid,
                    prompt_token_ids=list(prompt_token_ids),
                    all_latents=row.get("all_latents"),
                    timesteps=row.get("timesteps"),
                    rollout_log_probs=row.get("rollout_log_probs"),
                    image_path=row.get("image_path"),
                    call_role="initial",
                    cond_uid=str(cond_uid) if cond_uid else None,
                    cond_len=int(row.get("cond_len") or 0),
                )
            )
            _ = seed
        return samples


def compact_image_observation(path: str) -> str:
    """UND-facing observation: path only; S seed trajectories stay on the GEN batch."""
    return f"path={path}"


def judge_text_from_gen_samples(
    samples: list[GenSample],
    *,
    reduction: str = "any",
    threshold: float | None = None,
) -> str | None:
    """Reduce RM scores into Mode-2a judge text for ``build_forced_reflection``.

    Args:
        samples: GEN samples of one ``generate_image`` call (up to ``S`` seeds).
        reduction: episode-level stop bit over per-seed explicit flags —
            ``"any"`` (default, best-of-S), ``"all"``, or ``"mean"``
            (fraction of YES flags >= ``threshold``).
        threshold: ``good_enough_threshold`` from the single SoT
            (``agentic_image_gen.good_enough_threshold``); ``None`` keeps the
            turn-protocol default for callers that did not wire the knob.
    """
    scores = [float(s.rm_score) for s in samples if s.valid and s.rm_score is not None]
    if not scores:
        return None
    mean = float(sum(scores) / len(scores))
    explicit = [bool(s.good_enough) for s in samples if s.good_enough is not None]
    if explicit:
        if reduction == "all":
            good_enough = all(explicit)
        elif reduction == "mean":
            effective_threshold = float(threshold) if threshold is not None else 0.7
            good_enough = (sum(1.0 for flag in explicit if flag) / len(explicit)) >= effective_threshold
        else:
            good_enough = any(explicit)
    else:
        good_enough = derive_good_enough_from_scores(
            correctness=mean,
            aesthetics=mean,
            similarity=None,
            **({"threshold": float(threshold)} if threshold is not None else {}),
        )
    for sample in samples:
        if sample.good_enough is None:
            sample.good_enough = good_enough
    return format_rm_scores_as_judge_text(
        correctness=mean,
        aesthetics=mean,
        good_enough=good_enough,
        similarity=None,
    )


def _encode_text(tokenizer: Any | None, text: str, *, fallback_ids: list[int] | None = None) -> list[int]:
    if tokenizer is not None:
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(fallback_ids or [])


def turn_histogram(turns: list[int]) -> dict[str, float]:
    if not turns:
        return {
            "turns_per_episode/mean": 0.0,
            "turns_per_episode/p50": 0.0,
            "turns_per_episode/p95": 0.0,
            "turns_per_episode/max": 0.0,
            "tail_frac": 0.0,
        }
    arr = np.asarray(turns, dtype=np.float64)
    p95 = float(np.percentile(arr, 95))
    max_t = float(arr.max())
    tail = float(np.mean(arr > p95)) if max_t > 0 else 0.0
    return {
        "turns_per_episode/mean": float(arr.mean()),
        "turns_per_episode/p50": float(np.percentile(arr, 50)),
        "turns_per_episode/p95": p95,
        "turns_per_episode/max": max_t,
        "tail_frac": tail,
    }


async def run_serial_episode(
    *,
    dataset_task_uid: str,
    policy_version: int,
    prompt_ids: list[int],
    und_decode: Callable[..., Any],
    generate_tool: BagelGenerateImageTool,
    score_fn: Callable[[list[GenSample]], Any] | None = None,
    build_judge_text_fn: Callable[[list[GenSample]], str | None] | None = None,
    tokenizer: Any | None = None,
    user_prompt: str = "",
    max_und_turns: int = 8,
    max_context_tokens: int | None = None,
    forced_reflection_text: str = "Done.",
    episode_uid: str | None = None,
    non_image_reward: float | None = None,
    good_enough_threshold: float | None = None,
    good_enough_reduction: str = "any",
) -> EpisodeRollout:
    """Serial UND turn(s) → optional one GEN turn (S FlowGRPO seeds) → RM → reflection / Done.

    Episode shape follows the RFC: ``UND (+ GEN) (+ UND…)* + Done``, where each
    ``generate_image`` verdict enqueues exactly one GEN turn (``K += 1``). ``S`` is
    seed fan-out inside that turn for FlowGRPO — not additional GEN turns.

    ``non_image_reward`` is the RFC "non-image UND scalar" used when the episode
    makes zero ``generate_image`` calls (pattern 3, ``K = 0``). When it is ``None``
    and no image was scored, the episode reward is 0.0 and the caller should let the
    reward model fill it post-hoc.

    ``max_context_tokens`` is the episode's whole context budget, i.e.
    ``rollout.prompt_length + rollout.response_length``. ``und_decode`` is handed
    ``prompt_ids + response_ids`` as the decode prompt, so the context grows by every
    turn's tokens and the loop has to stop once it is spent -- otherwise it asks the
    engine for a decode it cannot emit a single token for, and the AR strategy rejects
    it outright:

        ValueError: Prompt length (2048) meets or exceeds the model's maximum context
        length (2048), leaving no space for generation.

    Measured 2026-09-17 09:30 with ``max_und_turns=8`` over a 1024+1024 budget. The
    turns that fit are kept and the post-loop forced reflection still closes the
    episode. ``None`` disables the guard.
    """
    ids = bind_episode_ids(
        dataset_task_uid=dataset_task_uid,
        episode_uid=episode_uid,
        policy_version=policy_version,
    )
    response_ids: list[int] = []
    response_mask: list[int] = []
    # π_rollout per response position, from the AR replica's own sampling. Kept in lockstep
    # with ``response_ids``/``response_mask`` by ``_append_response`` below.
    response_log_probs: list[float] = []
    gen_samples: list[GenSample] = []
    used_image_credit = False
    forced = False
    turns = 0
    num_gen_calls = 0
    judge_text: str | None = None
    stop_required = False
    # Per-turn trajectory trace (see ``EpisodeRollout.turn_trace``). Collected even when
    # ``BAGEL_CORL_DEBUG`` is off: the live per-turn ``logger.info`` below is the only
    # other place the raw UND text surfaces, and it is exactly the text that explains a
    # K=0 episode -- so the dump must not depend on the debug flag being set.
    trace: list[dict[str, Any]] = []
    clear_good_enough_yes_reached()
    active_user_prompt = user_prompt or get_active_user_prompt() or ""
    if _BAGEL_CORL_TURN_DEBUG:
        # Re-assert INFO logging per episode: the first decode of the step lazily imports
        # ``vllm``, whose logger setup reconfigures the *root* logger (``force=True``) and
        # would otherwise drop every INFO/WARNING line emitted from here on -- the exact
        # failure that made a healthy rollout look wedged (see ``_force_info_logging``).
        _force_info_logging()

    def _context_budget_spent() -> bool:
        """True once ``prompt_ids + response_ids`` has used up ``max_context_tokens``.

        One predicate for both decode sites below: the engine rejects a prompt with no
        room left for generation, so neither may be attempted past the budget.
        """
        return max_context_tokens is not None and len(prompt_ids) + len(response_ids) >= max_context_tokens

    def _append_response(tokens: Any, mask_value: int, log_probs: Any | None = None) -> None:
        """Extend the episode with ``tokens``, keeping π_rollout aligned with ``response_ids``.

        The trainer pairs ``rollout_log_probs`` with our recomputed ``old_log_probs`` position by
        position, so the three lists must never drift. Tokens the policy did not sample --
        ``Done.`` templates and forced reflection -- carry 0.0: the former are environment
        fallbacks and the latter are masked out of the loss (``response_mask`` 0), so no
        fabricated value can ever be read as a policy ratio *and* passed off as measured.

        Args:
            tokens: token ids (or tokens already extracted from a decode step).
            mask_value: 1 for policy tokens, 0 for forced/environment tokens.
            log_probs: the AR engine's per-token log-probs, if this decode sampled them.
        """
        token_list = [int(t) for t in tokens]
        response_ids.extend(token_list)
        response_mask.extend([int(mask_value)] * len(token_list))
        score = None if log_probs is None else [float(v) for v in log_probs]
        if score is not None and len(score) != len(token_list):
            raise ValueError(
                "Bagel Co-RL UND pass: rollout_log_probs must align with response_ids "
                f"(got {len(score)} log-probs for {len(token_list)} tokens)"
            )
        response_log_probs.extend(score if score is not None else [0.0] * len(token_list))

    # Defined before the loop so the budget guard below can break on its very first
    # iteration (a prompt that already fills the context) and still leave the post-loop
    # fallbacks that read ``step`` well-defined. A valid launch cannot reach that: the
    # framework caps the prompt at ``rollout.prompt_length`` while the budget is
    # ``prompt_length + response_length``.
    step: dict[str, Any] = {}

    for _ in range(max_und_turns):
        if _context_budget_spent():
            # Episode context exhausted: the next decode would leave the AR engine no
            # room for a single token. Keep the turns that fit and fall through to the
            # forced reflection / Done handling rather than raising.
            break
        turns += 1
        if _BAGEL_CORL_TURN_DEBUG:
            # Logged *before* the await: when an episode stops making progress, the last
            # line here names the turn that never returned, which the after-the-fact log
            # below cannot do. ``ctx`` is what the engine is asked to prefill this turn.
            logger.info(
                "bagel_corl_turn episode=%s turn=%d req_ctx=%d resp_tokens=%d",
                ids["episode_uid"][:8],
                turns,
                len(prompt_ids) + len(response_ids),
                len(response_ids),
            )
        _turn_t0 = time.perf_counter()
        _req_ctx = len(prompt_ids) + len(response_ids)
        decode = und_decode(prompt_ids=prompt_ids, response_ids=response_ids)
        step = await decode if asyncio.iscoroutine(decode) else decode
        token_ids: list[int] = list(step["token_ids"])
        text: str = str(step["text"])
        kind = und_turn_kind(text)
        trace.append(
            {
                "record": "und_turn",
                "turn": turns,
                "kind": kind,
                "req_ctx": _req_ctx,
                "out_tokens": len(token_ids),
                "decode_s": round(time.perf_counter() - _turn_t0, 3),
                "text": text,
            }
        )
        if _BAGEL_CORL_TURN_DEBUG:
            logger.info(
                "bagel_corl_turn episode=%s turn=%d kind=%s out_tokens=%d decode_s=%.1f text=%r",
                ids["episode_uid"][:8],
                turns,
                kind,
                len(token_ids),
                time.perf_counter() - _turn_t0,
                # The raw UND text is the only way to tell a model that never emits the
                # Hermes grammar from one whose ``<tool_call>`` block is being dropped or
                # mangled on the way out of the engine (see the tools-schema note in
                # ``bagel_corl.BagelMultiturnAgentLoop.run``). Truncated: one line per turn.
                text[:240],
            )
        _append_response(token_ids, 1, step.get("log_probs"))
        if kind == "done":
            break
        if kind != "generate_image":
            continue

        if get_good_enough_yes_reached():
            # Env hard-stop: prior YES latch blocks further generate_image.
            break

        call = parse_und_tool_call(text) or {}
        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        prompt = str(arguments.get("prompt", ""))
        meta = build_generate_call_meta(prompt=prompt, user_prompt=active_user_prompt)
        # RFC component-1 IDs: GEN TQ keys must be deterministic per (episode, call)
        # so a replay reproduces the same keys. Call 0 keeps the episode's base
        # ``gen_call_id``; later calls derive from the episode id and their local
        # index instead of a fresh uuid4, which left every K > 1 key unreproducible
        # (``num_gen_calls`` is the 0-based local call index here; it increments below).
        gen_call_id = (
            str(ids["gen_call_id"])
            if generate_tool._passes == 0
            else f"{ids['episode_uid']}:call{num_gen_calls}"
        )
        num_gen_calls += 1
        call_samples = await generate_tool(
            prompt=prompt,
            prompt_token_ids=list(prompt_ids) + list(response_ids),
            gen_call_id=gen_call_id,
        )
        call_role = str(meta.get("call_role") or "initial")
        for sample in call_samples:
            sample.call_role = call_role

        valid_paths = [s.image_path for s in call_samples if s.valid and s.image_path]
        if valid_paths:
            register_tool_artifact(prompt=prompt, paths=[str(p) for p in valid_paths])
        trace.append(
            {
                "record": "gen_call",
                "turn": turns,
                "gen_call_id": gen_call_id,
                "call_role": call_role,
                "prompt": prompt,
                "num_samples": len(call_samples),
                "num_valid": sum(1 for s in call_samples if s.valid),
                "image_paths": [str(p) for p in valid_paths],
            }
        )

        if score_fn is not None:
            scored = score_fn(call_samples)
            if asyncio.iscoroutine(scored):
                scored = await scored
            call_samples = scored

        # Accumulate — every generate_image call contributes its S seeds to the
        # episode (RFC: all K calls survive to gen_batch). Replacing the list here
        # silently dropped all but the last call's trajectories for K > 1.
        gen_samples.extend(call_samples)

        if valid_paths:
            used_image_credit = True
            obs = compact_image_observation(valid_paths[0])
            obs_ids = _encode_text(
                tokenizer,
                obs,
                fallback_ids=list(step.get("obs_token_ids") or [0]),
            )
            if not obs_ids:
                obs_ids = [0]
            _append_response(obs_ids, 0)

        if build_judge_text_fn is not None:
            maybe_text = build_judge_text_fn(call_samples)
            if asyncio.iscoroutine(maybe_text):
                maybe_text = await maybe_text
            judge_text = maybe_text
        if not judge_text:
            judge_text = judge_text_from_gen_samples(
                call_samples,
                reduction=good_enough_reduction,
                threshold=good_enough_threshold,
            )

        if judge_text:
            remaining = generate_tool.remaining_passes()
            force_done = remaining == 0
            reflection = build_forced_reflection(
                judge_text,
                force_done=force_done,
                generate_pass=generate_tool.max_generate_passes - remaining,
                max_passes=generate_tool.max_generate_passes,
            )
            if reflection is not None:
                reflection_text, stop_required = reflection
                forced_ids = _encode_text(
                    tokenizer,
                    reflection_text,
                    fallback_ids=list(step.get("forced_token_ids") or []),
                )
                if forced_ids:
                    forced = True
                    _append_response(forced_ids, 0)
                if any(s.good_enough for s in call_samples if s.good_enough is not None):
                    set_good_enough_yes_reached(True)
                elif "good_enough=YES" in judge_text:
                    set_good_enough_yes_reached(True)

                if stop_required:
                    if _context_budget_spent():
                        # Same rule as the turn loop: with the context spent there is no
                        # room for the closing ``Done.`` decode, so take the tokenizer
                        # fallback below instead of asking the engine for a turn it
                        # cannot generate.
                        done_step = {}
                    else:
                        done_decode = und_decode(prompt_ids=prompt_ids, response_ids=response_ids)
                        done_step = await done_decode if asyncio.iscoroutine(done_decode) else done_decode
                    done_ids = list(done_step.get("token_ids") or [])
                    done_text = str(done_step.get("text") or "")
                    if done_ids and und_turn_kind(done_text) == "done":
                        _append_response(done_ids, 1, done_step.get("log_probs"))
                    else:
                        fallback = _encode_text(
                            tokenizer,
                            "Done.",
                            fallback_ids=list(done_step.get("done_token_ids") or step.get("done_token_ids") or []),
                        )
                        if fallback:
                            _append_response(fallback, 1)
                    break

                if remaining > 0:
                    continue
                # remaining == 0 should have set stop_required via force_done
                break

        # No judge text: keep legacy forced/done token fallbacks for unit tests.
        forced_ids = list(step.get("forced_token_ids") or [])
        if forced_ids:
            forced = True
            _append_response(forced_ids, 0)
        else:
            done_ids = list(step.get("done_token_ids") or [])
            if done_ids:
                _append_response(done_ids, 1)
        break
    else:
        forced = True
        _ = forced_reflection_text

    image_scores = [float(s.rm_score) for s in gen_samples if s.valid and s.rm_score is not None]
    if image_scores:
        und_reward = float(np.mean(image_scores))
    elif non_image_reward is not None:
        # Pattern 3 (K=0): use the non-image UND scalar (RFC und/no_image_credit),
        # never a hard-coded 0 that would zero out token GRPO signal.
        und_reward = float(non_image_reward)
    else:
        und_reward = 0.0

    if _BAGEL_CORL_TURN_DEBUG:
        logger.info(
            "bagel_corl_episode episode=%s turns=%d K=%d resp_tokens=%d forced=%s "
            "stop_required=%s reward=%.3f",
            str(ids["episode_uid"])[:8],
            turns,
            num_gen_calls,
            len(response_ids),
            forced,
            stop_required,
            und_reward,
        )

    return EpisodeRollout(
        und_group_uid=str(ids["und_group_uid"]),
        episode_uid=str(ids["episode_uid"]),
        policy_version=int(ids["policy_version"]),
        prompt_ids=list(prompt_ids),
        response_ids=response_ids,
        response_mask=response_mask,
        rollout_log_probs=response_log_probs,
        turns=turns,
        gen_samples=gen_samples,
        used_image_credit=used_image_credit,
        forced_reflection=forced,
        und_reward=und_reward,
        judge_text=judge_text,
        stop_required=stop_required,
        num_gen_calls=num_gen_calls,
        turn_trace=trace,
    )


@dataclass
class FlattenResult:
    und_batch: list[dict[str, Any]]
    gen_batch: list[dict[str, Any]]
    gen_episode_map: list[dict[str, Any]]
    metrics: dict[str, float]


def episode_und_reward(episode: EpisodeRollout) -> tuple[float, bool]:
    """RFC ``und/no_image_credit``: the episode reward carried on the UND row.

    Returns ``(reward, used_non_image_scalar)``. The reward is the mean of the *rated* GEN seeds
    (``valid`` and ``rm_score`` set) when the RM scored images, else the non-image UND scalar the
    loop recorded -- pattern 3 (K=0) episodes must not have their token-GRPO signal zeroed.

    This is the single source of truth for both reward consumers: the TQ row the v1 advantage
    reads (``bagel_corl_tq.und_row_reward``) and the flatten row
    (``flatten_multiturn_rollouts``' ``token_level_scores``). Keeping them on one derivation is
    what stops UND and GEN from training against different numbers for the same episode.
    """
    image_scores = [float(s.rm_score) for s in episode.gen_samples if s.valid and s.rm_score is not None]
    if image_scores:
        return float(np.mean(image_scores)), False
    return float(episode.und_reward), True


def und_episode_rm_scores(response_mask: Any, reward: float) -> list[float]:
    """Per-token UND episode reward, placed on the last *trainable* response position.

    The v1 UND advantage reads this column directly -- ``trainer_base.py:1595`` does
    ``data.batch["token_level_scores"] = data.batch["rm_scores"]`` -- so an episode whose row
    never carries it kills the step with

        KeyError: 'key "rm_scores" not found in TensorDict with keys
        ['old_log_probs', 'response_mask', 'rollout_log_probs', 'uid']'

    (measured 2026-09-18 on `hk01dgx012`, devices 4-7, first step after the agent loop finally
    ran). ``AgentLoopOutput.as_dict`` only emits the column when ``reward_score`` is set, and it
    writes it at ``rm_scores[-1]`` (``verl/experimental/agent_loop/agent_loop.py:143-147``).
    That index is wrong for this loop: an episode that exhausts its ``generate_image`` budget ends
    on a *forced* turn whose tokens are appended with ``mask_value=0``
    (``bagel_corl_lib.run_serial_episode``), and an episode can also end on an observation. A
    score parked on a masked position is dropped by ``compute_advantage_for_multi_trajectories``
    and the episode silently carries no signal at all.

    So the reward goes on the last unmasked token, mirroring the pinned reward manager
    (``reward_manager/base.py:78-80`` -> ``rm_scores[arange, valid_response_length - 1]``). An
    all-masked row is a bug, not a zero: raise instead of publishing an invisible reward.

    Returns plain floats so this module stays torch-free; ``bagel_corl_tq.und_row_reward`` wraps
    it into the ``torch.float32`` column the TransferQueue stores.
    """
    if hasattr(response_mask, "ndim") and response_mask.ndim != 1:
        raise ValueError(
            "Bagel Co-RL UND rm_scores expects a 1-D response mask for one episode, got shape "
            f"{tuple(response_mask.shape)}"
        )
    mask = list(response_mask)
    valid = [index for index, flag in enumerate(mask) if bool(flag)]
    if not valid:
        raise ValueError(
            "Bagel Co-RL UND rm_scores: response_mask has no trainable position, so the episode "
            f"reward {float(reward)} could never reach the advantage computation"
        )
    scores = [0.0] * len(mask)
    scores[valid[-1]] = float(reward)
    return scores


def flatten_multiturn_rollouts(
    episodes: list[EpisodeRollout],
    *,
    expected_s: int,
) -> FlattenResult:
    """Split token UND rows from latent GEN rows. Never concatenate the two.

    Incomplete S-seed groups are dropped. Reflection-only episodes contribute zero GEN rows.
    One finished ``generate_image`` tool call is one GEN turn (RFC K); ``expected_s`` is
    FlowGRPO seeds under that single turn.
    """
    if expected_s < 1:
        raise ValueError("expected_s must be >= 1")
    und_batch: list[dict[str, Any]] = []
    gen_batch: list[dict[str, Any]] = []
    gen_episode_map: list[dict[str, Any]] = []
    dropped_incomplete = 0
    no_image_credit = 0

    for und_index, episode in enumerate(episodes):
        # Single source of truth with the TQ ingest (``bagel_corl_tq.und_row_reward``); see
        # ``episode_und_reward`` for why the two must not derive this independently.
        und_reward, used_non_image_scalar = episode_und_reward(episode)
        if used_non_image_scalar:
            # Pattern 3 (K=0): propagate the non-image UND scalar instead of zeroing
            # token GRPO signal (RFC und/no_image_credit).
            no_image_credit += 1
        # RFC §4.4.2a: the conditioning identity of this episode's GEN calls.
        # Carried on the UND row so R2's per-call grouping survives the flatten
        # (a GEN row's own ``cond_uid`` alone cannot tell "S seeds, one
        # conditioning" from "S calls, S conditionings").
        cond_uids = sorted({str(s.cond_uid) for s in episode.gen_samples if s.valid and s.cond_uid})
        und_batch.append(
            {
                "und_group_uid": episode.und_group_uid,
                "episode_uid": episode.episode_uid,
                "policy_version": episode.policy_version,
                "prompt_ids": list(episode.prompt_ids),
                "response_ids": list(episode.response_ids),
                "response_mask": list(episode.response_mask),
                "rollout_log_probs": list(episode.rollout_log_probs),
                "token_level_scores": und_reward,
                "used_image_credit": episode.used_image_credit,
                "gen_cond_uids": cond_uids,
                "gen_cond_len": int(max((s.cond_len for s in episode.gen_samples if s.valid), default=0)),
            }
        )
        valid = [s for s in episode.gen_samples if s.valid]
        if not valid:
            continue
        # Group by gen_group_uid — every complete call (exactly S valid seeds)
        # contributes its rows; a K>1 episode keeps ALL of its calls. Demanding
        # len(valid) == expected_s for the whole episode was a K=1 assumption that
        # silently dropped every multi-call episode.
        groups: dict[str, list[GenSample]] = {}
        for sample in valid:
            groups.setdefault(str(sample.gen_group_uid), []).append(sample)
        for rows in groups.values():
            if len(rows) != expected_s:
                dropped_incomplete += 1
                continue
            for sample in sorted(rows, key=lambda s: int(s.seed_index)):
                gen_batch.append(
                    {
                        "gen_group_uid": sample.gen_group_uid,
                        "gen_sample_uid": sample.gen_sample_uid,
                        "seed_index": sample.seed_index,
                        "prompt_token_ids": list(sample.prompt_token_ids),
                        "all_latents": sample.all_latents,
                        "timesteps": sample.timesteps,
                        "rollout_log_probs": sample.rollout_log_probs,
                        "rm_score": sample.rm_score,
                        "call_role": sample.call_role,
                        "cond_uid": sample.cond_uid,
                        "cond_len": int(sample.cond_len or 0),
                    }
                )
                gen_episode_map.append(
                    {
                        "und_index": und_index,
                        "episode_uid": episode.episode_uid,
                        "gen_sample_uid": sample.gen_sample_uid,
                        "gen_group_uid": sample.gen_group_uid,
                    }
                )

    metrics = {
        "gen/dropped_incomplete_groups": float(dropped_incomplete),
        "und/no_image_credit": float(no_image_credit),
        "gen/num_rows": float(len(gen_batch)),
        "und/num_rows": float(len(und_batch)),
        "gen/skipped_no_groups": 1.0 if not gen_batch else 0.0,
    }
    return FlattenResult(
        und_batch=und_batch,
        gen_batch=gen_batch,
        gen_episode_map=gen_episode_map,
        metrics=metrics,
    )


def aggregate_episode_metrics(records: list[Any]) -> dict[str, float]:
    """Batch-aggregate per-episode metrics into step-level logging values.

    Per-episode ``J``/``K`` must be **averaged over the sibling batch**, and
    ``gen/dropped_incomplete_groups`` summed — first-row-wins collapse hides every
    episode but one (audit module A). Records are the UND TQ rows (``{"fields": …}``
    or plain field dicts) carrying ``episode_J`` / ``episode_K`` and the
    ``bagel_corl_metrics`` blob written by ``pack_dual_lane_episode``.
    """
    j_values: list[float] = []
    k_values: list[float] = []
    dropped = 0.0
    no_credit_flags: list[float] = []
    skipped_flags: list[float] = []
    # RFC §4.4.4 R2: counts are additive over the batch, ratios are batch means
    # (each episode's ratio was already measured against the engine counters).
    cond_sums: dict[str, float] = {
        "gen/cond_cache_hits": 0.0,
        "gen/cond_cache_misses": 0.0,
        "gen/cond_cache_bypassed": 0.0,
    }
    cond_means: dict[str, list[float]] = {
        "gen/cond_recompute_ratio": [],
        "gen/prompt_embed_cache_hit_rate": [],
        "gen/cond_amortization": [],
    }
    for rec in records:
        fields = rec.get("fields") if isinstance(rec, dict) else rec
        if not isinstance(fields, dict):
            fields = {}
        metrics = fields.get("bagel_corl_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        j = fields.get("episode_J", metrics.get("episode/J"))
        k = fields.get("episode_K", metrics.get("episode/K"))
        if j is not None:
            j_values.append(float(j))
        if k is not None:
            k_values.append(float(k))
        dropped += float(metrics.get("gen/dropped_incomplete_groups", 0.0) or 0.0)
        if "und/no_image_credit" in metrics:
            no_credit_flags.append(float(metrics["und/no_image_credit"] or 0.0))
        if "gen/skipped_no_groups" in metrics:
            skipped_flags.append(float(metrics["gen/skipped_no_groups"] or 0.0))
        for key in cond_sums:
            if key in metrics:
                cond_sums[key] += float(metrics[key] or 0.0)
        for key, values in cond_means.items():
            if key in metrics:
                values.append(float(metrics[key] or 0.0))
    aggregated: dict[str, float] = {}
    if j_values:
        aggregated["episode/J"] = float(sum(j_values) / len(j_values))
    if k_values:
        aggregated["episode/K"] = float(sum(k_values) / len(k_values))
    aggregated["gen/dropped_incomplete_groups"] = float(dropped)
    if no_credit_flags:
        aggregated["und/no_image_credit"] = float(sum(no_credit_flags) / len(no_credit_flags))
    if skipped_flags:
        aggregated["gen/skipped_no_groups"] = float(sum(skipped_flags) / len(skipped_flags))
    if any(value > 0.0 for value in cond_sums.values()):
        aggregated.update(cond_sums)
    for key, values in cond_means.items():
        if values:
            aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _as_gen_sample(item: Any) -> GenSample | None:
    if isinstance(item, GenSample):
        return item
    if not isinstance(item, dict):
        return None
    return GenSample(
        gen_sample_uid=str(item.get("gen_sample_uid", "")),
        gen_group_uid=str(item.get("gen_group_uid", "")),
        seed_index=int(item.get("seed_index", 0)),
        valid=bool(item.get("valid", False)),
        prompt_token_ids=list(item.get("prompt_token_ids") or []),
        all_latents=item.get("all_latents"),
        timesteps=item.get("timesteps"),
        rollout_log_probs=item.get("rollout_log_probs"),
        rm_score=item.get("rm_score"),
        image_path=item.get("image_path"),
        call_role=str(item.get("call_role", "initial")),
        good_enough=item.get("good_enough"),
        cond_uid=str(item["cond_uid"]) if item.get("cond_uid") else None,
        cond_len=int(item.get("cond_len") or 0),
    )


def _ntb_from_extra_fields(extras: Any) -> dict[str, list[Any]]:
    rows = list(extras or [])
    ntb: dict[str, list[Any]] = {
        "gen_samples": [],
        "und_group_uid": [],
        "episode_uid": [],
        "prompt_ids": [],
        "response_ids": [],
        "response_mask": [],
        "turns": [],
        "used_image_credit": [],
    }
    for extra in rows:
        mapping = extra if isinstance(extra, dict) else {}
        ntb["gen_samples"].append(mapping.get("gen_samples") or [])
        ntb["und_group_uid"].append(mapping.get("und_group_uid", "missing_task"))
        ntb["episode_uid"].append(mapping.get("episode_uid"))
        ntb["prompt_ids"].append(mapping.get("prompt_ids") or [])
        ntb["response_ids"].append(mapping.get("response_ids") or [])
        ntb["response_mask"].append(mapping.get("response_mask") or [])
        ntb["turns"].append(mapping.get("turns", 1))
        ntb["used_image_credit"].append(mapping.get("used_image_credit", False))
    return ntb


def flatten_from_agent_output(output: Any, *, expected_s: int) -> FlattenResult:
    """RFC Flatten after ``generate_sequences``: UND vs GEN rows from extra_fields."""
    ntb = dict(getattr(output, "non_tensor_batch", None) or {})
    if ntb.get("gen_samples") is None and ntb.get("extra_fields") is not None:
        ntb.update(_ntb_from_extra_fields(ntb["extra_fields"]))
    gen_col = ntb.get("gen_samples")
    if gen_col is None:
        return flatten_multiturn_rollouts([], expected_s=expected_s)

    length = len(gen_col)

    def _col(name: str, default: Any) -> list[Any]:
        arr = ntb.get(name)
        if arr is None:
            return [default] * length
        values = list(arr)
        if len(values) < length:
            values.extend([default] * (length - len(values)))
        return values[:length]

    episodes: list[EpisodeRollout] = []
    und_groups = _col("und_group_uid", "missing_task")
    episode_uids = _col("episode_uid", None)
    prompt_ids = _col("prompt_ids", [])
    response_ids = _col("response_ids", [])
    response_mask = _col("response_mask", [])
    rollout_log_probs = _col("rollout_log_probs", [])
    turns = _col("turns", 1)
    used = _col("used_image_credit", False)
    for index in range(length):
        raw_samples = gen_col[index] or []
        samples = []
        for item in raw_samples:
            sample = _as_gen_sample(item)
            if sample is not None:
                samples.append(sample)
        episodes.append(
            EpisodeRollout(
                und_group_uid=str(und_groups[index]),
                episode_uid=str(episode_uids[index] or f"ep{index}"),
                policy_version=0,
                prompt_ids=list(prompt_ids[index] or []),
                response_ids=list(response_ids[index] or []),
                response_mask=list(response_mask[index] or []),
                rollout_log_probs=list(rollout_log_probs[index] or []),
                turns=int(turns[index] or 1),
                gen_samples=samples,
                used_image_credit=bool(used[index]),
                num_gen_calls=len({str(s.gen_group_uid) for s in samples if s.valid}),
            )
        )
    return flatten_multiturn_rollouts(episodes, expected_s=expected_s)


def strip_pixels_for_actor(row: dict[str, Any]) -> dict[str, Any]:
    """Bagel GEN trains on ``prompt_token_ids``, not pixels or ``prompt_embeds``."""
    cleaned = dict(row)
    for key in ("images", "pixel_values", "prompt_embeds", "negative_prompt_embeds"):
        cleaned.pop(key, None)
    return cleaned

