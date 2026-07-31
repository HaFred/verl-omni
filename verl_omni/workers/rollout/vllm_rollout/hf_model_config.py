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
"""Helpers to adapt OmniModelConfig → HFModelConfig for stock vLLM servers."""

from dataclasses import fields, is_dataclass

from omegaconf import DictConfig, OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig


def as_hf_model_config(model_config) -> HFModelConfig:
    """Strip Omni-/diffusion-only keys before building ``HFModelConfig``.

    Agentic recipes pass ``OmniModelConfig`` (``architecture``, ``freeze``,
    ``agentic``, …). PEFT/FSDP need those; stock ``vLLMHttpServer`` only accepts
    the HFModelConfig subset and raises ``ConfigKeyError`` on extras.
    """
    if is_dataclass(model_config) and type(model_config) is HFModelConfig:
        return model_config

    if isinstance(model_config, DictConfig | dict):
        cfg = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
    elif is_dataclass(model_config):
        cfg = OmegaConf.structured(model_config)
    else:
        cfg = model_config

    allowed = {f.name for f in fields(HFModelConfig)}
    filtered = {k: cfg[k] for k in cfg.keys() if k in allowed}
    return omega_conf_to_dataclass(filtered, dataclass_type=HFModelConfig)
