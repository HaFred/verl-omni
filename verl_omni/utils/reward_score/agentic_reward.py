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
"""Scalar agentic reward for Mode (2a) GRPO acceptance smoke."""

from __future__ import annotations

from typing import Any


def _trajectory_texts(extra_info: dict | None, kwargs: dict) -> list[str]:
    """Extract per-turn agent texts from extra_info / kwargs."""
    extra_info = dict(extra_info or {})
    traj = extra_info.get("agentic_trajectory")
    if traj is None:
        traj = kwargs.get("agentic_trajectory")

    texts: list[str] = []
    if isinstance(traj, dict):
        for turn in traj.get("turns") or []:
            if isinstance(turn, dict):
                text = turn.get("agent_text") or ""
                if text:
                    texts.append(str(text))
            elif isinstance(turn, str) and turn:
                texts.append(turn)
    elif isinstance(traj, list):
        for item in traj:
            if isinstance(item, str) and item:
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("agent_text") or item.get("text") or item.get("content") or ""
                if text:
                    texts.append(str(text))
    elif extra_info.get("trajectory_text"):
        raw = extra_info["trajectory_text"]
        if isinstance(raw, list):
            texts.extend(str(t) for t in raw if t)
        elif raw:
            texts.append(str(raw))
    return texts


def _tool_call_score(text: str) -> float:
    """Recognize the stock tool-agent's serialized function call."""
    lowered = text.lower()
    return 1.0 if "generate_image" in lowered else min(1.0, len(text.strip()) / 256.0)


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """DAPO-compatible scalar reward for stock ``ToolAgentLoop`` rollouts."""
    del data_source, ground_truth  # interface compatibility
    texts = _trajectory_texts(extra_info, kwargs)
    if texts:
        scores = [_tool_call_score(t) for t in texts]
        score = float(sum(scores) / len(scores))
        return {
            "score": score,
            "method": "agentic_tool_call_heuristic",
            "num_turns_scored": len(texts),
            "tool_stubbed": bool((extra_info or {}).get("tool_stubbed", False)),
        }

    text = (solution_str or "").strip()
    if "generate_image" in text.lower():
        return {
            "score": 1.0,
            "method": "agentic_tool_call_heuristic",
            "num_turns_scored": 1,
        }

    score = 0.0 if not text else min(1.0, len(text) / 256.0)
    return {"score": float(score), "method": "response_length_heuristic"}
