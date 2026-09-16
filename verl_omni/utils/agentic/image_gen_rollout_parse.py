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

"""Rollout turn parsing helpers for agentic image-gen monitoring."""

from __future__ import annotations

import json
import re
from typing import Any

# Hermes JSON (``{"name": "generate_image"}``) or Qwen XML (``<function=...>``).
_TOOL_CALL_NAME_PAT = r"<function={name}\b|\"name\"\s*:\s*\"{name}\""
#: Hermes block: ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``.
#: Non-greedy ``\{.*?\}`` plus the ``</tool_call>`` anchor matches the *outermost*
#: object, so nested ``arguments`` braces stay inside the captured group.
_HERMES_TOOL_CALL_PAT = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
#: Qwen block: ``<tool_call><function=NAME><parameter=K>v</parameter></function></tool_call>``.
_QWEN_TOOL_CALL_PAT = r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>\s*</tool_call>"
_QWEN_PARAM_PAT = r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>"


def tool_call_order(decode: str) -> list[str]:
    """Return the tool names this turn calls, in the order they appear.

    One assistant turn may hold several ``<tool_call>`` blocks. Only the first
    ``multi_turn.max_parallel_calls`` of them are executed, so anything after the
    first is context the trainer never ran and must not decide the turn's label.

    Args:
        decode: Decoded assistant text for one turn.

    Returns:
        Tool names in appearance order, e.g. ``["generate_image", "judge_image"]``.
        Empty when the turn calls no tool.
    """
    found: list[tuple[int, str]] = []
    for name in ("generate_image", "judge_image"):
        match = re.search(_TOOL_CALL_NAME_PAT.format(name=name), decode or "", re.IGNORECASE)
        if match:
            found.append((match.start(), name))
    return [name for _, name in sorted(found)]


def turn_kind(decode: str, turn_prompt: str, response: str = "") -> str:
    """Label a turn so trajectory dumps make protocol stages grep-able.

    Args:
        decode: Decoded assistant text for this turn.
        turn_prompt: Prompt / observation text feeding the turn.
        response: Optional response text used for forced-reflection cues.

    Returns:
        Stage label string (for example ``call_generate_image``).
    """
    resp = response or ""
    forced_context = f"{turn_prompt or ''}\n{resp}"
    called = tool_call_order(decode)
    if called:
        # Label by the call the model emitted *first*: that is the only one that
        # executes (``multi_turn.max_parallel_calls``). Testing ``judge_image``
        # first mislabelled a "generate then judge" turn as ``call_judge_image``,
        # which read as a judge-first rollout in the dumps. Trailing calls are
        # appended as ``_then_call_<name>`` so the drop stays visible.
        first, trailing = called[0], called[1:]
        if first == "judge_image":
            label = "call_judge_image"
        elif re.search(r"\bagentic_forced_reflection=1\b", resp, re.IGNORECASE) or re.search(
            r"\bagentic_forced_reflection=1\b", turn_prompt or "", re.IGNORECASE
        ):
            label = "agent_rewrite_after_forced_reflection_then_call_generate_image"
        else:
            label = "call_generate_image"
        for name in trailing:
            label = f"{label}_then_call_{name}"
        return label
    if re.search(
        r"(?is)^\s*(?:Reflection\s*:.*?)?Done\.\s*(?:<\|im_end\|>)?\s*$",
        decode or "",
    ) and re.search(r"\bagentic_stop_decision_required=1\b", forced_context, re.IGNORECASE):
        if re.search(r"\bagentic_force_stop_max_passes=1\b", forced_context):
            return "agent_done_after_max_passes"
        return "agent_done_after_forced_reflection"
    if re.search(r"\bagentic_force_stop_max_passes=1\b", resp) or (
        not (decode or "").strip() and re.search(r"\bagentic_force_stop_max_passes=1\b", turn_prompt or "")
    ):
        return "forced_reflection_max_passes_stop_cue"
    if re.search(r"\bagentic_forced_reflection=1\b", resp, re.IGNORECASE):
        if re.search(r"\bagentic_stop_decision_required=1\b", resp, re.IGNORECASE):
            return "forced_reflection_stop_cue"
        return "forced_reflection_continue"
    if not (decode or "").strip() and re.search(r"\bagentic_forced_reflection=1\b", turn_prompt or "", re.IGNORECASE):
        if re.search(r"\bagentic_stop_decision_required=1\b", turn_prompt or "", re.IGNORECASE):
            return "forced_reflection_stop_cue"
        return "forced_reflection_continue"
    if re.search(r"\bReflection\s*:", decode or "", re.IGNORECASE):
        if re.search(
            r"<function=generate_image\b|\"name\"\s*:\s*\"generate_image\"",
            decode or "",
            re.IGNORECASE,
        ):
            return "agent_reflection_rewrite_then_call_generate_image"
        return "agent_reflection_done"
    if re.search(r"\b(?:VL judge|agentic_judge)\b", turn_prompt or "", re.IGNORECASE):
        return "after_judge_feedback"
    if "path=" in (turn_prompt or "") and "agentic_tool" in (turn_prompt or ""):
        return "after_generate_image"
    return "other"


def extract_tool_calls(decoded_response: str) -> list[dict[str, Any]]:
    """Parse the assistant tool calls in one decoded turn, in emission order.

    Args:
        decoded_response: Decoded assistant / trajectory text.

    Returns:
        ``[{"name": str, "arguments": dict}, ...]`` ordered by position. Calls whose
        payload is unparseable are skipped rather than guessed at.
    """
    found: list[tuple[int, dict[str, Any]]] = []
    for match in re.finditer(_HERMES_TOOL_CALL_PAT, decoded_response or "", re.IGNORECASE | re.DOTALL):
        call = _parse_hermes_call(match.group(1))
        if call is not None:
            found.append((match.start(), call))
    for match in re.finditer(_QWEN_TOOL_CALL_PAT, decoded_response or "", re.IGNORECASE | re.DOTALL):
        params = re.findall(_QWEN_PARAM_PAT, match.group(2), re.IGNORECASE | re.DOTALL)
        found.append(
            (
                match.start(),
                {
                    "name": match.group(1).strip(),
                    "arguments": {key.strip(): value.strip() for key, value in params},
                },
            )
        )
    return [call for _, call in sorted(found, key=lambda item: item[0])]


def _parse_hermes_call(raw: str) -> dict[str, Any] | None:
    """Parse one Hermes ``<tool_call>`` JSON body into ``{"name", "arguments"}``.

    Args:
        raw: JSON object text captured between the ``<tool_call>`` tags.

    Returns:
        Normalised call dict, or ``None`` when the payload is unusable.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name", "")).strip()
    if not name:
        return None
    args = payload.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def first_tool_call(decode: str) -> dict[str, Any] | None:
    """Return the call this turn actually runs, or ``None``.

    ``ToolAgentLoop`` executes only ``tool_calls[:multi_turn.max_parallel_calls]``
    (default 1), so the first call is the accepted one and any trailing call is
    dropped context.

    Args:
        decode: Decoded assistant text for one turn.

    Returns:
        First parsed call, or ``None`` when the turn emits no parseable tool call.
    """
    calls = extract_tool_calls(decode)
    return calls[0] if calls else None


def tool_prompt_of(call: dict[str, Any] | None) -> str:
    """Return the prompt string the accepted tool call carries.

    ``generate_image`` submits a fresh prompt to the frozen diffusion model;
    ``judge_image`` inspects an existing image and echoes the prompt it judged in
    ``image_prompt``. Both are captured, so read the value together with
    :func:`tool_name`: under ``generate_image`` this is a new submission, under
    ``judge_image`` it is an echo. The judge schema invites the literal shortcut
    ``image_prompt="last"`` instead of a real echo ("Do not re-paste long prompts"),
    so ``"last"`` there means the model took the shortcut rather than that no prompt
    was involved.

    Args:
        call: Parsed call dict from :func:`first_tool_call`.

    Returns:
        Stripped prompt string, or ``""`` when the call carries none.
    """
    if not call:
        return ""
    args = call.get("arguments") or {}
    if call.get("name") == "generate_image":
        value = args.get("prompt")
    elif call.get("name") == "judge_image":
        value = args.get("image_prompt")
    else:
        return ""
    return value.strip() if isinstance(value, str) else ""


def extract_generate_image_prompts(decoded_response: str) -> list[str]:
    """Extract ordered ``generate_image`` prompts from decoded trajectory text.

    Args:
        decoded_response: Decoded assistant / trajectory text.

    Returns:
        List of prompt strings in call order.
    """
    prompts: list[str] = []
    for call in extract_tool_calls(decoded_response):
        if call["name"] != "generate_image":
            continue
        prompt = call["arguments"].get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompts.append(prompt.strip())
    return prompts


def split_env_blob(blob: str) -> tuple[str, str]:
    """Split mask=0 env text into turn prompt and response.

    Args:
        blob: Environment / observation text blob.

    Returns:
        ``(turn_prompt, response)``.
    """
    text = (blob or "").strip()
    if not text:
        return "", ""
    force_idx = -1
    for match in re.finditer(r"(?:^|\n)\s*Reflection\s*:", text, re.IGNORECASE):
        # Prefer the forced marker when present.
        window = text[match.start() : match.start() + 400]
        if "agentic_forced_reflection=1" in window or force_idx < 0:
            force_idx = match.start()
            if "agentic_forced_reflection=1" in window:
                break
    if force_idx < 0 or not re.search(r"\bagentic_forced_reflection=1\b", text, re.IGNORECASE):
        # No injected assistant response — entire blob is the next-turn prompt.
        return text, ""
    # Include any chat-template role tags immediately before Reflection.
    cut = force_idx
    preamble = text[:force_idx]
    # If the decode left a trailing bare ``assistant`` / think block before Reflection,
    # keep tool_response in turn_prompt and put Reflection(+trailing) in response.
    tool_end = preamble.rfind("</tool_response>")
    if tool_end >= 0:
        turn_prompt = text[: tool_end + len("</tool_response>")].strip()
        response = text[tool_end + len("</tool_response>") :].strip()
        # Drop leading role/think scaffolding noise from response but keep Reflection.
        refl = re.search(r"Reflection\s*:", response, re.IGNORECASE)
        if refl:
            response = response[refl.start() :].strip()
        return turn_prompt, response
    return text[:cut].strip(), text[cut:].strip()


def unpad_left_ids(token_ids, pad_token_id: int | None) -> list[int]:
    """Strip left padding from prompt token ids.

    Args:
        token_ids: Prompt token ids (possibly left-padded).
        pad_token_id: Pad id, or ``None`` to return as-is.

    Returns:
        Unpadded token id list.
    """
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    pads = {0}
    if pad_token_id is not None:
        pads.add(int(pad_token_id))
    start = 0
    while start < len(ids) and int(ids[start]) in pads:
        start += 1
    return [int(x) for x in ids[start:]]


def turn_record(
    *,
    turn: int,
    turn_prompt: str,
    response: str,
    decode: str,
    turn_input: str = "",
) -> dict[str, Any]:
    """Build one trajectory-dump turn record.

    Args:
        turn: 1-based turn index.
        turn_prompt: Prompt / observation text for the turn.
        response: Assistant response text.
        decode: Decoded model tokens for the turn.
        turn_input: Optional raw turn input text.

    Returns:
        Dict with turn / prompt / tool / decode fields.
    """
    accepted = first_tool_call(decode or "")
    return {
        "turn": turn,
        "turn_prompt": turn_prompt or "",
        "turn_input": turn_input or "",
        "tool_name": accepted["name"] if accepted else "",
        # The prompt string the accepted call carries: a fresh diffusion prompt on a
        # ``generate_image`` turn, the judged-prompt echo on a ``judge_image`` turn
        # (often the literal shortcut ``last``). ``turn_prompt`` is the whole chat
        # template and ``turn_obs`` the raw env delta, so neither exposes either.
        "tool_prompt": tool_prompt_of(accepted),
        "decode": decode or "",
        "response": response or "",
        "decode_has_tool_call": "<tool_call>" in (decode or "").lower(),
    }


class _RolloutTurnSplitter:
    """Stateful splitter for ``response_mask`` model / env spans."""

    def __init__(self, token_ids, response_mask, tokenizer, prompt_ids=None):
        self.ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        self.mask = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
        self.tokenizer = tokenizer
        self.prompt_prefix = (
            unpad_left_ids(prompt_ids, getattr(tokenizer, "pad_token_id", None)) if prompt_ids is not None else None
        )
        self.turns: list[dict[str, Any]] = []
        self.current_model: list[int] = []
        self.current_tool: list[int] = []
        self.pending_prompt = ""
        self.pending_response = ""
        self.model_start = 0

    def _decode_input(self, response_prefix_len: int) -> str:
        if self.prompt_prefix is None:
            return ""
        prefix = self.prompt_prefix + [int(x) for x in self.ids[:response_prefix_len]]
        return self.tokenizer.decode(prefix, skip_special_tokens=False)

    def _flush_tool(self) -> None:
        if not self.current_tool:
            return
        blob = self.tokenizer.decode(self.current_tool, skip_special_tokens=True).strip()
        self.current_tool = []
        prompt, response = split_env_blob(blob)
        self.pending_prompt = prompt
        self.pending_response = response

    def _flush_model(self) -> None:
        if not self.current_model:
            return
        decode = self.tokenizer.decode(self.current_model, skip_special_tokens=False)
        self.turns.append(
            turn_record(
                turn=len(self.turns) + 1,
                turn_prompt=self.pending_prompt,
                response=self.pending_response,
                decode=decode,
                turn_input=self._decode_input(self.model_start),
            )
        )
        self.pending_prompt = ""
        self.pending_response = ""
        self.current_model = []

    def run(self) -> list[dict[str, Any]]:
        for idx, (token_id, is_model_token) in enumerate(zip(self.ids, self.mask, strict=True)):
            if int(is_model_token) == 1:
                if self.current_tool:
                    self._flush_tool()
                if not self.current_model:
                    self.model_start = idx
                self.current_model.append(int(token_id))
            else:
                if self.current_model:
                    self._flush_model()
                self.current_tool.append(int(token_id))
        if self.current_model:
            self._flush_model()
        # Trailing env (e.g. final judge + forced Done with no further decode).
        if self.current_tool:
            self._flush_tool()
            if self.pending_prompt or self.pending_response:
                self.turns.append(
                    turn_record(
                        turn=len(self.turns) + 1,
                        turn_prompt=self.pending_prompt,
                        response=self.pending_response,
                        decode="",
                        turn_input=self._decode_input(len(self.ids)),
                    )
                )
            self.pending_prompt = ""
            self.pending_response = ""
        return self.turns


def split_assistant_rollouts(token_ids, response_mask, tokenizer) -> list[str]:
    """Decode contiguous model-token spans from a response.

    Args:
        token_ids: Response token ids.
        response_mask: Mask where model tokens are 1 and tool obs are 0.
        tokenizer: Tokenizer used for decode.

    Returns:
        List of decoded assistant span strings.
    """
    return [turn["decode"] for turn in split_rollout_turns(token_ids, response_mask, tokenizer)]


def split_rollout_turns(
    token_ids,
    response_mask,
    tokenizer,
    prompt_ids=None,
) -> list[dict[str, Any]]:
    """Split a response into turns with prompt / response / decode fields.

    Args:
        token_ids: Response token ids.
        response_mask: Model-token mask.
        tokenizer: Tokenizer used for decode.
        raw_prompt: Optional raw prompt used to seed the first turn.

    Returns:
        List of turn record dicts.
    """
    return _RolloutTurnSplitter(token_ids, response_mask, tokenizer, prompt_ids=prompt_ids).run()


def last_user_prompt(raw_prompt: Any) -> str:
    """Return the last user prompt from a raw prompt payload.

    Args:
        raw_prompt: String, message list, or other prompt payload.

    Returns:
        Last user text, or ``""`` if none.
    """
    messages = list(raw_prompt) if raw_prompt is not None else []
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content)
    return ""
