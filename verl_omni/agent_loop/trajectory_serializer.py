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
"""Serialize AgenticTrajectory objects into batched DataProto tensors."""

from __future__ import annotations

import torch

from verl_omni.agent_loop.agentic_trajectory import AgenticTrajectory

PAD_TOKEN_ID = 0
OBS_TOKEN_ID = -1  # placeholder for observation slots (never fed as model input)


def _pad_or_truncate_int(lst: list[int], max_len: int, pad_value: int) -> list[int]:
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))


def _pad_or_truncate_float(lst: list[float], max_len: int, pad_value: float) -> list[float]:
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))


def serialize_trajectories(
    trajectories: list[AgenticTrajectory],
    max_prompt_length: int,
    observation_token_length: int,
    max_total_tokens: int = 4096,
) -> dict[str, torch.Tensor]:
    """Serialize a batch of AgenticTrajectories into tensors for DataProto.

    Returns dict with keys:
        prompt_tokens: [bsz, max_prompt_length]
        agent_tokens: [bsz, max_total_tokens]
        agent_logprobs: [bsz, max_total_tokens]
        loss_mask: [bsz, max_total_tokens]
        responses: [bsz, C, H, W]  # final turn's image
    """
    prompt_tokens_list = []
    agent_tokens_list = []
    agent_logprobs_list = []
    loss_mask_list = []
    responses_list = []

    for traj in trajectories:
        prompt_tok = _pad_or_truncate_int([ord(c) for c in traj.prompt], max_prompt_length, PAD_TOKEN_ID)
        prompt_tokens_list.append(prompt_tok)

        all_tokens: list[int] = []
        all_logprobs: list[float] = []
        all_mask: list[int] = []

        for turn in traj.turns:
            # Agent text tokens (trainable)
            all_tokens.extend(turn.agent_tokens)
            all_logprobs.extend(turn.agent_logprobs)
            all_mask.extend([1] * len(turn.agent_tokens))

            # Observation placeholder (frozen, loss_mask=0)
            if turn.tool_output is not None:
                all_tokens.extend([OBS_TOKEN_ID] * observation_token_length)
                all_logprobs.extend([0.0] * observation_token_length)
                all_mask.extend([0] * observation_token_length)

        # Pad/truncate to fixed length
        all_tokens = _pad_or_truncate_int(all_tokens, max_total_tokens, PAD_TOKEN_ID)
        all_logprobs = _pad_or_truncate_float(all_logprobs, max_total_tokens, 0.0)
        all_mask = _pad_or_truncate_int(all_mask, max_total_tokens, 0)

        agent_tokens_list.append(all_tokens)
        agent_logprobs_list.append(all_logprobs)
        loss_mask_list.append(all_mask)

        # Final turn's image for reward computation
        final_image = torch.zeros(3, 512, 512)
        if traj.turns and traj.turns[-1].tool_output is not None:
            final_image = traj.turns[-1].tool_output.output_data
        responses_list.append(final_image)

    return {
        "prompt_tokens": torch.tensor(prompt_tokens_list, dtype=torch.long),
        "agent_tokens": torch.tensor(agent_tokens_list, dtype=torch.long),
        "agent_logprobs": torch.tensor(agent_logprobs_list, dtype=torch.float32),
        "loss_mask": torch.tensor(loss_mask_list, dtype=torch.float32),
        "responses": torch.stack(responses_list),
    }
