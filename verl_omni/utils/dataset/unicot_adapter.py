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
"""Adapter: UniCoT-Self-Reflection-6K -> AgenticTrajectory.

Mapping per RFC S7:
  current_image = input_image[i]
  reflection = eval_summary[i] (fallback: eval[i])
  action = "continue" iff output_image[i] is non-null
  edit = edit[i] for continue
  next_image = output_image[i] for continue

Fail-closed on: length mismatches, empty reflections, continue-with-empty-edit,
contradictory terminal rows, missing images, image hash mismatches.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from verl_omni.agent_loop.agentic_trajectory import (
    AgenticMetadata,
    AgenticTrajectory,
    AgenticTurn,
    ToolCall,
    ToolOutput,
)

logger = logging.getLogger(__file__)

_to_tensor = transforms.ToTensor()


def _img_to_tensor(path: str) -> torch.Tensor:
    return _to_tensor(Image.open(path).convert("RGB"))


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.md5(t.numpy().tobytes()).hexdigest()


def load_unicot_dataset(
    path: str, split: str = "train", eval_ratio: float = 0.1
) -> list[AgenticTrajectory]:
    """Load UniCoT-Self-Reflection-6K parquet, return AgenticTrajectory list."""
    import pandas as pd

    df = pd.read_parquet(path)
    trajectories = []

    for data_id, group in df.groupby("data_id"):
        try:
            traj = _adapt_group(data_id, group)
            if traj is not None:
                trajectories.append(traj)
        except Exception as e:
            logger.warning("Skipping UniCoT record %s: %s", data_id, e)

    eval_ids = set()
    for traj in trajectories:
        h = int(hashlib.md5(traj.prompt.encode()).hexdigest(), 16)
        if (h % 100) < (eval_ratio * 100):
            eval_ids.add(traj.prompt)

    if split == "train":
        return [t for t in trajectories if t.prompt not in eval_ids]
    return [t for t in trajectories if t.prompt in eval_ids]


def _adapt_group(data_id: int, group) -> Optional[AgenticTrajectory]:
    rows = group.sort_values("state_index")
    input_imgs = rows["input_image"].tolist()
    output_imgs = rows["output_image"].tolist()
    refs = rows["eval_summary"].tolist()
    edits = rows["edit"].tolist()
    n = len(rows)

    if not (len(input_imgs) == len(output_imgs) == len(refs) == len(edits)):
        return None

    turns = []
    for i in range(n):
        ref_text = str(refs[i] or "").strip()
        if not ref_text:
            return None

        has_next = (i + 1 < n) and (output_imgs[i] is not None)

        if has_next:
            edit_text = str(edits[i] or "").strip()
            if not edit_text:
                return None
            curr_out = _img_to_tensor(output_imgs[i])
            next_in = _img_to_tensor(input_imgs[i + 1])
            if _hash_tensor(curr_out) != _hash_tensor(next_in):
                return None
            tc = ToolCall("image_edit", {"prompt": edit_text})
            to = ToolOutput("image", curr_out)
            decision = "continue"
        else:
            tc = None
            to = None
            decision = "terminate"

        agent_text = (
            f"<reasoning>{ref_text}</reasoning>\n"
            f"<prompt>{edits[i] if i < n else ''}</prompt>\n"
            f"<decision>{decision}</decision>"
        )
        turns.append(AgenticTurn(i, [], [], agent_text, tc, to, decision))

    meta = AgenticMetadata(len(turns), turns[-1].decision == "terminate",
                           "agent_stop" if turns[-1].decision == "terminate" else "max_turns")
    return AgenticTrajectory(str(data_id), turns, metadata=meta)
