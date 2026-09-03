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
