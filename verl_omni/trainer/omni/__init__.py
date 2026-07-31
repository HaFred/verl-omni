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

# Omni sync trainer pulls verl's V1 stack (TransferQueue). Keep import soft so
# Mode (2a) agentic GRPO via ``verl.trainer.main_ppo`` still loads without it.
try:
    from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync  # noqa: F401

    __all__ = ["OmniPPOTrainerSync"]
except ModuleNotFoundError as exc:  # pragma: no cover - optional V1 dependency
    if getattr(exc, "name", None) != "transfer_queue" and "transfer_queue" not in str(exc):
        raise
    OmniPPOTrainerSync = None  # type: ignore[misc, assignment]
    __all__: list[str] = []
