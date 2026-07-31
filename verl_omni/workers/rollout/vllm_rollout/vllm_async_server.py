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
"""Stock vLLM server that accepts OmniModelConfig (strips Omni-only keys)."""

import ray
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer as _VerlVLLMHttpServer
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica as _VerlVLLMReplica

from verl_omni.workers.rollout.vllm_rollout.hf_model_config import as_hf_model_config


class vLLMHttpServer(_VerlVLLMHttpServer):
    """Stock vLLM HTTP server tolerant of ``OmniModelConfig`` extras."""

    def _init_model_config(self, model_config):
        return as_hf_model_config(model_config)


class vLLMReplica(_VerlVLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(vLLMHttpServer)
