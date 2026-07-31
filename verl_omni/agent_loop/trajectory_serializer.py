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

from typing import Protocol

import torch
import torch.nn.functional as F

from verl_omni.agent_loop.agentic_trajectory import AgenticTrajectory

PAD_TOKEN_ID = 0
OBS_TOKEN_ID = -1  # placeholder for observation slots (never fed as model input)
DEFAULT_RESPONSE_CHW = (3, 512, 512)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def _pad_or_truncate_int(lst: list[int], max_len: int, pad_value: int) -> list[int]:
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))


def _pad_or_truncate_float(lst: list[float], max_len: int, pad_value: float) -> list[float]:
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))


def _normalize_response_image(image: torch.Tensor, target_chw: tuple[int, int, int]) -> torch.Tensor:
    """Resize/pad a CHW (or HWC) image tensor to a fixed ``(C, H, W)`` for batching."""
    c, h, w = target_chw
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    img = image.detach().float()
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims (CHW/HWC), got shape {tuple(img.shape)}")
    if img.shape[0] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)
    if img.shape[0] != c:
        if img.shape[0] == 1 and c == 3:
            img = img.repeat(3, 1, 1)
        else:
            # Channel truncate/pad to target C
            if img.shape[0] > c:
                img = img[:c]
            else:
                pad = torch.zeros(c - img.shape[0], img.shape[1], img.shape[2], dtype=img.dtype)
                img = torch.cat([img, pad], dim=0)
    if img.shape[-2:] != (h, w):
        img = F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    return img


def serialize_trajectories(
    trajectories: list[AgenticTrajectory],
    max_prompt_length: int,
    observation_token_length: int,
    max_total_tokens: int = 4096,
    *,
    tokenizer: TokenizerLike,
    response_image_size: tuple[int, int, int] = DEFAULT_RESPONSE_CHW,
) -> dict[str, torch.Tensor]:
    """Serialize a batch of AgenticTrajectories into tensors for DataProto.

    Args:
        trajectories: Batch of trajectories.
        max_prompt_length: Pad/truncate width for prompt token ids.
        observation_token_length: Placeholder slots per tool observation.
        max_total_tokens: Pad/truncate width for interleaved agent/obs tokens.
        tokenizer: Tokenizer used to encode ``traj.prompt`` into real token ids.
        response_image_size: Fixed ``(C, H, W)`` for stacked response images.

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
        prompt_ids = list(tokenizer.encode(traj.prompt, add_special_tokens=False))
        prompt_tok = _pad_or_truncate_int(prompt_ids, max_prompt_length, PAD_TOKEN_ID)
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

        # Final turn's image for reward computation (normalized for stack)
        final_image = torch.zeros(*response_image_size)
        if traj.turns and traj.turns[-1].tool_output is not None:
            final_image = _normalize_response_image(traj.turns[-1].tool_output.output_data, response_image_size)
        responses_list.append(final_image)

    return {
        "prompt_tokens": torch.tensor(prompt_tokens_list, dtype=torch.long),
        "agent_tokens": torch.tensor(agent_tokens_list, dtype=torch.long),
        "agent_logprobs": torch.tensor(agent_logprobs_list, dtype=torch.float32),
        "loss_mask": torch.tensor(loss_mask_list, dtype=torch.float32),
        "responses": torch.stack(responses_list),
    }
