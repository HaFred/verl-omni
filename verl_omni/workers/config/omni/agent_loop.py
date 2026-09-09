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
"""Omni / Bagel Co-RL rollout agent config extensions (do not patch upstream verl)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from verl.workers.config.rollout import AgentLoopConfig

__all__ = ["BagelCorlAgentLoopConfig"]


@dataclass
class BagelCorlAgentLoopConfig(AgentLoopConfig):
    """``AgentLoopConfig`` plus Bagel Co-RL knobs used by ``bagel_multiturn_agent``.

    Kept in verl-omni so upstream ``verl.workers.config.AgentLoopConfig`` stays untouched.
    """

    # S seeds per generate_image (FlowGRPO group). Independent of sibling N / in-episode J,K.
    gen_samples_per_call: Optional[int] = None
    # PR1 fail-closed: must be 1 (bounds in-episode K).
    max_generate_passes: Optional[int] = None
    # UND turn budget before force-stop (bounds in-episode J).
    max_und_turns: Optional[int] = None
