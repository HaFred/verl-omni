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
"""CPU tests for Bagel dual-role UND AR / GEN diffusion client routing."""

from __future__ import annotations

import pytest

from verl_omni.workers.rollout.bagel_dual_role_llm_server import (
    BagelDualRoleLLMServerClient,
    is_bagel_gen_sampling_params,
)


def test_is_gen_sampling_detects_flowgrpo_keys():
    assert is_bagel_gen_sampling_params({"num_inference_steps": 10, "seed": 1})
    assert is_bagel_gen_sampling_params({"noise_level": 0.7})
    assert is_bagel_gen_sampling_params({"bagel_role": "gen"})
    assert not is_bagel_gen_sampling_params({"temperature": 0.7, "top_p": 0.9})
    assert not is_bagel_gen_sampling_params({"bagel_role": "und", "temperature": 0.7})
    assert not is_bagel_gen_sampling_params({})


def test_dual_role_client_routes_und_vs_gen():
    import asyncio

    class _Fake:
        def __init__(self, name):
            self.name = name
            self.calls = []

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            self.calls.append(
                {"request_id": request_id, "prompt_ids": prompt_ids, "sampling_params": dict(sampling_params)}
            )
            return {"role": self.name, "token_ids": [1]}

    async def _run():
        und = _Fake("und")
        gen = _Fake("gen")
        client = BagelDualRoleLLMServerClient(config=object(), und_client=und, gen_client=gen)

        await client.generate("r1", prompt_ids=[1, 2], sampling_params={"temperature": 0.7, "bagel_role": "und"})
        await client.generate(
            "r2",
            prompt_ids=[3],
            sampling_params={
                "num_inference_steps": 8,
                "noise_level": 0.7,
                "bagel_role": "gen",
                "temperature": 0.9,
            },
        )
        return und, gen

    und, gen = asyncio.run(_run())
    assert len(und.calls) == 1
    assert "num_inference_steps" not in und.calls[0]["sampling_params"]
    assert und.calls[0]["sampling_params"].get("bagel_role") is None
    assert len(gen.calls) == 1
    assert gen.calls[0]["sampling_params"]["num_inference_steps"] == 8
    assert "temperature" not in gen.calls[0]["sampling_params"]
