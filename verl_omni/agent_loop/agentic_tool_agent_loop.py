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

"""ToolAgentLoop that forces a ``Reflection:`` assistant turn after successful judge_image.

When ``judge_image`` returns ``agentic_judge ok=1`` (VL sidecar HTTP 200 with
scores), this loop injects an assistant message:

  Reflection: <VL summary> Done.          # if good_enough=YES → then terminate
  Reflection: <VL summary>                # if good_enough=NO  → then generate again

so rollout dumps always contain ``Reflection:`` after a successful VL judge.
Toggle with ``AGENTIC_FORCE_REFLECTION_AFTER_JUDGE`` (default ``1``).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

logger = logging.getLogger(__name__)

_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
_CORRECTNESS_RE = re.compile(r"\bcorrectness\s*=\s*([0-9.]+)", re.IGNORECASE)
_AESTHETICS_RE = re.compile(r"\baesthetics\s*=\s*([0-9.]+)", re.IGNORECASE)
_GOOD_ENOUGH_RE = re.compile(r"\bgood_enough\s*=\s*(YES|NO)", re.IGNORECASE)
_FINDINGS_RE = re.compile(
    r"\bfindings:\s*(.+?)(?:\n\s*suggested_fixes:|\n\s*agentic_judge\b)", re.IGNORECASE | re.DOTALL
)
_FIXES_RE = re.compile(r"\bsuggested_fixes:\s*(.+?)(?:\n\s*agentic_judge\b|\n\n|\Z)", re.IGNORECASE | re.DOTALL)


def _force_enabled() -> bool:
    return os.getenv("AGENTIC_FORCE_REFLECTION_AFTER_JUDGE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _tool_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content or "")


def build_forced_reflection(tool_text: str) -> tuple[str, bool] | None:
    """Build ``(assistant_text, done)`` from a successful judge tool observation."""
    if not _JUDGE_OK_RE.search(tool_text or ""):
        return None
    c_m = _CORRECTNESS_RE.search(tool_text)
    a_m = _AESTHETICS_RE.search(tool_text)
    g_m = _GOOD_ENOUGH_RE.search(tool_text)
    f_m = _FINDINGS_RE.search(tool_text)
    x_m = _FIXES_RE.search(tool_text)
    correctness = c_m.group(1) if c_m else "?"
    aesthetics = a_m.group(1) if a_m else "?"
    good_enough = (g_m.group(1).upper() == "YES") if g_m else False
    findings = re.sub(r"\s+", " ", (f_m.group(1) if f_m else "").strip())[:220]
    fixes = re.sub(r"\s+", " ", (x_m.group(1) if x_m else "").strip())[:160]
    if not findings:
        findings = "see VL facet scores above"
    if good_enough:
        text = (
            f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
            f"good_enough=YES. {findings} Done."
        )
        return text, True
    fix_note = f" Suggested fixes: {fixes}." if fixes and fixes.lower() != "none" else ""
    text = (
        f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
        f"good_enough=NO. {findings}.{fix_note} Rewriting the diffusion prompt next."
    )
    return text, False


@register("agentic_tool_agent")
class AgenticToolAgentLoop(ToolAgentLoop):
    """Stock tool agent + forced Reflection after successful ``judge_image``."""

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if not _force_enabled():
            return state
        if state == AgentState.TERMINATED:
            return state

        # Inspect tool messages just appended for a successful VL judge.
        forced: tuple[str, bool] | None = None
        for message in reversed(agent_data.messages):
            if message.get("role") != "tool":
                break
            forced = build_forced_reflection(_tool_message_text(message))
            if forced is not None:
                break
        if forced is None:
            return state

        reflection_text, done = forced
        # Marker keeps dumps / metrics grep-able even if the model also continues.
        reflection_text = f"{reflection_text} agentic_forced_reflection=1"
        assistant_msg = {"role": "assistant", "content": reflection_text}
        agent_data.messages.append(assistant_msg)

        # Same encoding path as tool responses (strip system prompt; keep gen prompt
        # so the policy can continue with rewritten generate_image when not done).
        response_ids = await self.apply_chat_template(
            [assistant_msg],
            remove_system_prompt=True,
        )
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        # Mask 0: keep Reflection in context/dumps, but do not train on injected tokens
        # (logprobs were not sampled from the policy).
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.assistant_turns += 1
        agent_data.extra_fields["forced_reflection"] = True
        logger.info(
            "Forced Reflection after judge_image (done=%s, chars=%d)",
            done,
            len(reflection_text),
        )

        if done:
            return AgentState.TERMINATED
        # Not good enough: let the policy emit rewritten generate_image next.
        return AgentState.GENERATING
