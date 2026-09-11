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
"""Dual-role LLM client: UND AR + GEN diffusion for Bagel Co-RL.

One vLLM-Omni replica is AR xor Diffusion (strategy chosen at server init).
Bagel Co-RL therefore keeps two ``LLMServerManager`` pools and routes
``generate()`` by sampling-params shape.
"""

from __future__ import annotations

from typing import Any, Optional

from omegaconf import DictConfig
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from verl_omni.agent_loop.bagel_corl_gen_serve import _AR_ONLY_SAMPLING_KEYS

# Diffusion / FlowGRPO request markers (inverse of AR-only decode knobs).
_GEN_SAMPLING_KEYS = frozenset(
    {
        "num_inference_steps",
        "noise_level",
        "sde_window_size",
        "sde_window_range",
        "sde_type",
        "height",
        "width",
        "cfg_text_scale",
        "cfg_img_scale",
    }
)


def is_bagel_gen_sampling_params(sampling_params: dict[str, Any] | None) -> bool:
    """True when the request is a GEN / FlowGRPO denoise (not UND AR decode)."""
    if not sampling_params:
        return False
    keys = set(sampling_params.keys())
    if keys & _GEN_SAMPLING_KEYS:
        return True
    # Explicit role tag from BagelMultiturnAgentLoop / gen serve helpers.
    role = sampling_params.get("bagel_role") or sampling_params.get("role")
    return str(role).lower() in {"gen", "diffusion", "generate_image"}


class BagelDualRoleLLMServerClient(LLMServerClient):
    """Route ``generate`` to UND AR or GEN diffusion clients.

    Agent loops keep calling ``server_manager.generate``; this client is what
    ``get_llm_client()`` returns for ``bagel_corl_sync``.
    """

    def __init__(
        self,
        config: DictConfig,
        *,
        und_client: LLMServerClient,
        gen_client: LLMServerClient,
    ):
        # No load-balancer of our own; each child owns its pool.
        super().__init__(config=config, load_balancer_handle=None)
        self.und_client = und_client
        self.gen_client = gen_client

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        client = self.gen_client if is_bagel_gen_sampling_params(sampling_params) else self.und_client
        # Strip cross-role knobs so AR/Diffusion strategies do not reject the request.
        params = dict(sampling_params or {})
        if client is self.und_client:
            for key in list(params):
                if key in _GEN_SAMPLING_KEYS or key in {"bagel_role", "role"}:
                    params.pop(key, None)
        else:
            for key in list(params):
                if key in _AR_ONLY_SAMPLING_KEYS:
                    params.pop(key, None)
        return await client.generate(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            **kwargs,
        )
