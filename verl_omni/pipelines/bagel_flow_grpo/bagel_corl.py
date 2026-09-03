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

"""Bagel UND+GEN Co-RL actor surface: untied ``lm_head``, dual disjoint LoRA, UND log-probs.

PR1 keeps one FSDP owner (``BagelForTraining`` + ``lm_head``) with two forwards:
token-path UND causal log-probs and the existing GEN velocity / FlowGRPO path.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import (
    BagelForTraining,
    BagelTrainingConfig,
    _map_checkpoint_to_training,
)

# Text-path UND LoRA (never ``*_moe_gen``).
UND_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# Generation-path LoRA (OCR / FlowGRPO list).
GEN_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj_moe_gen",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
    "o_proj_moe_gen",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
)

DUAL_LORA_TARGET_MODULES: tuple[str, ...] = UND_LORA_TARGET_MODULES + GEN_LORA_TARGET_MODULES


def validate_disjoint_lora_targets(target_modules: Iterable[str]) -> tuple[set[str], set[str]]:
    """Fail if UND and GEN LoRA target sets overlap.

    Args:
        target_modules: PEFT ``target_modules`` list from config.

    Returns:
        ``(und_selected, gen_selected)`` subsets of the known target lists.

    Raises:
        ValueError: Overlap, or a name that is neither UND nor GEN.
    """
    selected = [str(name) for name in target_modules]
    und = set(UND_LORA_TARGET_MODULES)
    gen = set(GEN_LORA_TARGET_MODULES)
    selected_set = set(selected)
    unknown = selected_set - und - gen
    if unknown:
        raise ValueError(f"Bagel CoRL LoRA target_modules contain unknown names: {sorted(unknown)}")
    und_sel = selected_set & und
    gen_sel = selected_set & gen
    overlap = und_sel & gen_sel
    if overlap:
        raise ValueError(f"Bagel CoRL UND and GEN LoRA targets overlap: {sorted(overlap)}")
    return und_sel, gen_sel


def dual_lora_param_groups(module: nn.Module) -> list[dict]:
    """Split trainable parameters into UND (text-path) and GEN (``moe_gen``) groups."""
    und_params: list[nn.Parameter] = []
    gen_params: list[nn.Parameter] = []
    other: list[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if "moe_gen" in name:
            gen_params.append(param)
        elif "lora_" in name.lower() or "lora_A" in name or "lora_B" in name:
            und_params.append(param)
        else:
            other.append(param)
    groups: list[dict] = []
    if und_params:
        groups.append({"params": und_params, "name": "und_lora"})
    if gen_params:
        groups.append({"params": gen_params, "name": "gen_lora"})
    if other:
        groups.append({"params": other, "name": "other_trainable"})
    if not groups:
        raise ValueError("Bagel CoRL optimizer requires at least one trainable parameter group")
    return groups


def _assert_untied_lm_head_config(model_path: str) -> None:
    llm_path = os.path.join(model_path, "llm_config.json")
    cfg: dict = {}
    if os.path.isfile(llm_path):
        with open(llm_path) as f:
            cfg.update(json.load(f))
    root_path = os.path.join(model_path, "config.json")
    if os.path.isfile(root_path):
        with open(root_path) as f:
            root = json.load(f)
        cfg.update(root.get("llm_config", {}))
        if "tie_word_embeddings" in root:
            cfg.setdefault("tie_word_embeddings", root["tie_word_embeddings"])
    if bool(cfg.get("tie_word_embeddings", False)):
        raise ValueError("BAGEL CoRL requires an untied lm_head checkpoint (tie_word_embeddings=false)")


def map_checkpoint_to_corl(state_dict: dict[str, Tensor], config: BagelTrainingConfig) -> dict[str, Tensor]:
    """Map ``ema.safetensors`` onto ``BagelForCoRL``, including ``language_model.lm_head``."""
    mapped = _map_checkpoint_to_training(state_dict, config)
    lm_head = None
    for src_key, tensor in state_dict.items():
        if src_key == "language_model.lm_head.weight" or src_key.endswith("lm_head.weight"):
            if src_key.startswith("language_model.model."):
                continue
            lm_head = tensor
            break
    if lm_head is None:
        raise ValueError("BAGEL CoRL requires language_model.lm_head.weight; tied/missing heads are not supported")
    mapped["lm_head.weight"] = lm_head
    return mapped


def route_actor_weight_for_und_replica(name: str) -> str | None:
    """Map FSDP actor parameter names onto the vLLM-Omni Bagel language model.

    ``transformer.lm_head.*`` must reach the UND replica; GEN ``transformer.`` MoT
    weights keep the existing ``model.`` prefix used by ``language_model.load_weights``.
    """
    if name.startswith("transformer.lm_head."):
        return "lm_head." + name[len("transformer.lm_head.") :]
    if name.startswith("transformer."):
        return "model." + name[len("transformer.") :]
    return None


class BagelForCoRL(BagelForTraining):
    """``BagelForTraining`` plus an untied ``lm_head`` for UND causal log-probs."""

    def __init__(self, config: BagelTrainingConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16) -> BagelForCoRL:
        """Load pretrained weights including ``lm_head``; fail if the head is tied/missing."""
        _assert_untied_lm_head_config(model_path)
        config = BagelTrainingConfig.from_model_path(model_path)
        ckpt_path = os.path.join(model_path, "ema.safetensors")
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)

        if "latent_pos_embed.pos_embed" in state_dict:
            actual_len = state_dict["latent_pos_embed.pos_embed"].shape[0]
            grid = int(actual_len**0.5)
            if grid * grid == actual_len and grid != config.max_latent_size:
                config.max_latent_size = grid

        model = cls(config)
        mapped = map_checkpoint_to_corl(state_dict, config)
        missing, _unexpected = model.load_state_dict(mapped, strict=False)
        if any(key == "lm_head.weight" or key.startswith("lm_head.") for key in missing):
            raise ValueError("BAGEL CoRL failed to load lm_head from the published checkpoint")
        return model.to(torch_dtype)

    def compute_und_log_prob(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        """Causal next-token log-probs on the text MoT path (no GEN / latent tokens).

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` 1 = valid.
            response_mask: ``(B, L)`` 1 = policy tokens to score (forced reflection is 0).

        Returns:
            ``(B, L-1)`` log-probs aligned with ``input_ids[:, 1:]``, zeroed where
            ``response_mask[:, 1:]`` is 0.
        """
        if input_ids.ndim != 2:
            raise ValueError("compute_und_log_prob expects input_ids of shape (B, L)")
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        response_mask = response_mask.to(device=input_ids.device)
        batch_size, seq_len = input_ids.shape
        text_embeds = self.embed_tokens(input_ids)
        latent_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        l_ctx = seq_len
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        key_padding_mask = None if bool(attention_mask.all()) else attention_mask

        sequence = text_embeds
        for layer in self.layers:

            def _layer_fn(seq, pos_ids, text_mask_, latent_mask_, kpm, *, _layer=layer):
                return _layer(seq, pos_ids, text_mask_, latent_mask_, l_ctx, key_padding_mask=kpm)

            sequence = self._checkpointed_call(
                _layer_fn, sequence, position_ids, attention_mask, latent_mask, key_padding_mask
            )

        hidden = sequence.new_zeros(sequence.shape)
        text_idx = attention_mask.nonzero(as_tuple=True)
        hidden[text_idx] = self.norm(sequence[text_idx])
        logits = self.lm_head(hidden)
        log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        labels = input_ids[:, 1:].unsqueeze(-1)
        token_logp = log_probs.gather(-1, labels).squeeze(-1)
        score_mask = response_mask[:, 1:].to(dtype=token_logp.dtype)
        return token_logp * score_mask
