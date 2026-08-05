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
"""Multi-turn agentic reward for Mode (2a) GRPO acceptance smoke.

Produces meaningful score variance by rewarding: tool-call iteration depth,
prompt-refinement diversity, response engagement, and completion quality.
"""

from __future__ import annotations

from typing import Any


def _extract_prompts(traj: dict) -> list[str]:
    """Extract diffusion prompts from tool_call params in each trajectory turn."""
    prompts: list[str] = []
    for turn in traj.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        tc = turn.get("tool_call")
        if isinstance(tc, dict):
            p = tc.get("params", {}).get("prompt")
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
    return prompts


def _prompt_diversity(prompts: list[str]) -> float:
    """Word-level Jaccard dissimilarity between consecutive prompts.

    Returns 0.0 when prompts are identical, 1.0 when completely disjoint.
    For <2 prompts returns 0.0.
    """
    if len(prompts) < 2:
        return 0.0
    diffs: list[float] = []
    for i in range(1, len(prompts)):
        prev_words = set(prompts[i - 1].lower().split())
        curr_words = set(prompts[i].lower().split())
        union = prev_words | curr_words
        if not union:
            continue
        jaccard = len(prev_words & curr_words) / len(union)
        diffs.append(1.0 - jaccard)
    return sum(diffs) / len(diffs) if diffs else 0.0


def _compute_reward(
    texts: list[str],
    traj: dict | None,
    raw_prompt: str,
    extra_info: dict | None,
) -> dict[str, Any]:
    """Compute multi-dimensional reward from trajectory data.

    Reward composition (total in [0, 1]):

    * **tool_depth** (0.30): scaled by number of tool-call turns (cap at max_turns).
    * **diversity** (0.30): word-level dissimilarity between consecutive diffusion prompts.
    * **engagement** (0.20): total response word count relative to a 100-word target.
    * **completion** (0.20): presence of a non-trivial final answer after the last tool call.
    """
    extra_info = dict(extra_info or {})
    turns = (traj or {}).get("turns") or []
    max_turns = extra_info.get("max_assistant_turns", 5)
    total_words = sum(len(t.split()) for t in texts)

    # --- tool_depth (0.30): reward calling the tool across multiple turns ---
    n_tool_calls = sum(1 for t in texts if "generate_image" in t.lower())
    tool_depth = min(n_tool_calls / max(max_turns, 1), 1.0) * 0.30

    # --- diversity (0.30): reward varying prompts across turns ---
    prompts = _extract_prompts(traj or {})
    div = _prompt_diversity(prompts)
    diversity = div * 0.30

    # --- engagement (0.20): reward longer, more elaborate responses ---
    engagement = min(total_words / 100.0, 1.0) * 0.20

    # --- completion (0.20): reward ending with a non-tool-call answer ---
    completion = 0.0
    if turns:
        last_turn = turns[-1]
        if isinstance(last_turn, dict):
            last_decision = last_turn.get("decision", "")
            last_text = (last_turn.get("agent_text") or "").strip()
            # Reward when the agent stops and the final text is substantial
            # and does *not* just contain another bare tool call.
            if last_decision == "stop" and len(last_text) > 20:
                has_tool = "generate_image" in last_text.lower()
                completion = 0.10 if has_tool else 0.20
    if not turns and texts:
        last = texts[-1].strip()
        if len(last) > 20:
            has_tool = "generate_image" in last.lower()
            completion = 0.10 if has_tool else 0.20

    score = tool_depth + diversity + engagement + completion

    return {
        "score": float(score),
        "method": "agentic_multi_turn_heuristic",
        "num_turns": len(texts),
        "num_tool_calls": n_tool_calls,
        "prompt_diversity": round(div, 3),
        "tool_stubbed": bool(extra_info.get("tool_stubbed", False)),
        "decomp__tool_depth": round(tool_depth, 3),
        "decomp__diversity": round(diversity, 3),
        "decomp__engagement": round(engagement, 3),
        "decomp__completion": round(completion, 3),
    }


def _fallback_reward(text: str) -> dict[str, Any]:
    """Length-based fallback when no trajectory data is available."""
    if not text:
        return {
            "score": 0.0,
            "method": "response_length_heuristic",
            "num_turns": 0,
            "num_tool_calls": 0,
            "prompt_diversity": 0.0,
        }
    if "generate_image" in text.lower():
        return {
            "score": 0.30,
            "method": "agentic_multi_turn_heuristic",
            "num_turns": 1,
            "num_tool_calls": 1,
            "prompt_diversity": 0.0,
        }
    score = 0.0 if len(text) == 0 else min(1.0, len(text) / 256.0)
    return {
        "score": float(score),
        "method": "response_length_heuristic",
        "num_turns": 1,
        "num_tool_calls": 0,
        "prompt_diversity": 0.0,
    }


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


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """Multi-turn agentic reward with meaningful score variance for GRPO.

    4-component heuristic: tool iteration depth (0.30), prompt-refinement
    diversity (0.30), response engagement (0.20), and completion quality (0.20).
    """
    del data_source, ground_truth  # interface compatibility
    extra_info = dict(extra_info or {})
    texts = _trajectory_texts(extra_info, kwargs)
    traj = extra_info.get("agentic_trajectory") or kwargs.get("agentic_trajectory")
    raw_prompt = extra_info.get("raw_prompt", "")

    if texts:
        return _compute_reward(texts, traj if isinstance(traj, dict) else None, raw_prompt, extra_info)

    return _fallback_reward((solution_str or "").strip())
