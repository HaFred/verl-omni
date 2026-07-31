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
"""Scalar agentic reward for Mode (2a) GRPO (PR1).

PR2 replaces/extends this module with multi-dimensional UTPCR scorers
(``compute_score_format``, ``compute_score_plan``, …). For PR1 DAPO/GRPO
smoke we only need a finite scalar so group-relative advantages are defined.
"""

from __future__ import annotations


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """DAPO-compatible scalar reward for agentic GRPO.

    Prefer a mild length prior on the decoded response so und-only / toy
    rollouts still produce a non-degenerate reward signal. Signature matches
    verl's ``custom_reward_function`` / DAPO contract.
    """
    del data_source, ground_truth, extra_info, kwargs  # interface compatibility
    text = (solution_str or "").strip()
    score = 0.0 if not text else min(1.0, len(text) / 256.0)
    return {"score": float(score), "method": "response_length_heuristic"}
