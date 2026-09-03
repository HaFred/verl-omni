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
"""CPU tests for BagelForCoRL lm_head, UND log-probs, and disjoint dual LoRA."""

from __future__ import annotations

import pytest
import torch

from verl_omni.pipelines.bagel_flow_grpo.bagel_corl import (
    DUAL_LORA_TARGET_MODULES,
    GEN_LORA_TARGET_MODULES,
    UND_LORA_TARGET_MODULES,
    BagelForCoRL,
    dual_lora_param_groups,
    route_actor_weight_for_und_replica,
    validate_disjoint_lora_targets,
)
from verl_omni.pipelines.bagel_flow_grpo.bagel_model import BagelTrainingConfig


def test_disjoint_lora_targets_ok():
    und, gen = validate_disjoint_lora_targets(DUAL_LORA_TARGET_MODULES)
    assert und == set(UND_LORA_TARGET_MODULES)
    assert gen == set(GEN_LORA_TARGET_MODULES)


def test_disjoint_lora_targets_fail_on_unknown():
    with pytest.raises(ValueError, match="unknown"):
        validate_disjoint_lora_targets(["q_proj", "not_a_bagel_proj"])


def test_route_lm_head_to_und_replica():
    assert route_actor_weight_for_und_replica("transformer.lm_head.weight") == "lm_head.weight"
    assert route_actor_weight_for_und_replica("transformer.layers.0.self_attn.q_proj.weight").startswith("model.")


def test_compute_und_log_prob_text_path_tiny():
    config = BagelTrainingConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_latent_size=4,
    )
    model = BagelForCoRL(config).eval()
    input_ids = torch.randint(0, 128, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    response_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    logp = model.compute_und_log_prob(input_ids, attention_mask, response_mask)
    assert logp.shape == (2, 5)
    assert torch.all(logp[:, :1] == 0)


def test_dual_lora_param_groups_split_moe_gen():
    config = BagelTrainingConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_latent_size=4,
    )
    model = BagelForCoRL(config)
    for name, param in model.named_parameters():
        param.requires_grad = "moe_gen" in name
    groups = dual_lora_param_groups(model)
    names = {g["name"] for g in groups}
    assert names == {"gen_lora"}
