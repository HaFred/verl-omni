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

"""Parser for structured agent LLM output with XML-like tags."""

from __future__ import annotations

import re
from typing import Any

AGENT_SYSTEM_PROMPT = """You are a visual creation agent. Analyze the user request, reason about what to generate, produce a prompt for the diffusion model, evaluate the result, and decide whether to continue or stop.

You MUST respond in the following format:
<reasoning>
[Your analysis of the request, what needs to be generated, and any constraints]
</reasoning>
<prompt>
[The exact prompt to send to the diffusion model]
</prompt>
<decision>
[continue or stop]
</decision>
"""


def parse_agent_output(text: str) -> dict[str, str | None]:
    """Parse structured agent output into reasoning, prompt, and decision.

    On parse failure, returns None for missing fields and defaults
    decision to "stop" (fail-safe).
    """
    result: dict[str, Any] = {
        "reasoning": None,
        "prompt": None,
        "decision": "stop",
        "raw_text": text,
    }

    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    prompt_match = re.search(r"<prompt>\s*(.*?)\s*</prompt>", text, re.DOTALL)
    if prompt_match:
        result["prompt"] = prompt_match.group(1).strip()

    decision_match = re.search(r"<decision>\s*(.*?)\s*</decision>", text, re.DOTALL)
    if decision_match:
        raw_decision = decision_match.group(1).strip().lower()
        result["decision"] = "continue" if "continue" in raw_decision else "stop"

    return result
