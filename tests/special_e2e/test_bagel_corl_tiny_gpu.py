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
"""Tiny-Bagel GPU smoke for Co-RL UND log-probs (J=2,K=2 grouping is covered on CPU)."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tiny-Bagel Co-RL smoke needs GPU")
def test_tiny_bagel_corl_und_forward_gpu():
    pytest.importorskip("vllm_omni")
    pytest.importorskip("safetensors")
    from tests.special_e2e.build_bagel_tiny_random import ensure_tiny_bagel_checkpoint
    from verl_omni.pipelines.bagel_flow_grpo.bagel_corl import (
        DUAL_LORA_TARGET_MODULES,
        BagelForCoRL,
        validate_disjoint_lora_targets,
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = ensure_tiny_bagel_checkpoint(os.path.join(tmp, "BAGEL-MoT"), skip_if_exists=False)
        model = BagelForCoRL.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
        input_ids = torch.randint(0, 32, (2, 8), device="cuda")
        attention_mask = torch.ones(2, 8, dtype=torch.long, device="cuda")
        response_mask = torch.ones(2, 8, dtype=torch.long, device="cuda")
        logp = model.compute_und_log_prob(input_ids, attention_mask, response_mask)
        assert logp.shape == (2, 7)
        validate_disjoint_lora_targets(DUAL_LORA_TARGET_MODULES)
        assert logp.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tiny-Bagel Co-RL smoke needs GPU")
def test_tiny_bagel_corl_dual_lora_groups_and_und_backward_gpu():
    """Co-RL core on GPU: dual LoRA grouping (lr_gen override) + UND backward.

    Locks the composite training surface the RFC's (a) first claims: the text
    path trains through ``compute_und_log_prob`` while the optimizer carries
    separate UND/GEN groups with the per-group ``lr_gen`` override.
    """
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    from tests.special_e2e.build_bagel_tiny_random import ensure_tiny_bagel_checkpoint
    from verl_omni.pipelines.bagel_flow_grpo.bagel_corl import (
        DUAL_LORA_TARGET_MODULES,
        BagelForCoRL,
        dual_lora_param_groups,
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = ensure_tiny_bagel_checkpoint(os.path.join(tmp, "BAGEL-MoT"), skip_if_exists=False)
        model = BagelForCoRL.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda()
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=list(DUAL_LORA_TARGET_MODULES),
        )
        model = get_peft_model(model, lora)

        groups = dual_lora_param_groups(model, lr_gen=3e-5)
        by_name = {g["name"]: g for g in groups}
        assert "gen_lora" in by_name and by_name["gen_lora"]["lr"] == pytest.approx(3e-5)
        assert "und_lora" in by_name and "lr" not in by_name["und_lora"]
        assert sum(len(g["params"]) for g in groups) > 0

        input_ids = torch.randint(0, 32, (2, 8), device="cuda")
        attention_mask = torch.ones(2, 8, dtype=torch.long, device="cuda")
        response_mask = torch.ones(2, 8, dtype=torch.long, device="cuda")
        logp = model.compute_und_log_prob(input_ids, attention_mask, response_mask)
        # Token-GRPO surrogate: -(advantage * ratio) proxy — CE on the selected
        # tokens is enough to prove the text path carries gradient.
        loss = -logp.masked_select(response_mask[:, 1:].bool()).mean()
        loss.backward()
        und_grads = [
            p.grad
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and "moe_gen" not in n
        ]
        assert any(g is not None and g.abs().sum() > 0 for g in und_grads)
        model.zero_grad(set_to_none=True)
