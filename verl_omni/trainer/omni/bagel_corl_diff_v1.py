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
"""Bind diffusion V1 GEN hooks onto the Co-RL ``TaskRunnerV1`` owner.

RFC: one job. ``OmniBagelCoRLTrainerSync`` is the PPO V1 owner.
``PolicyGradientDiffusionTrainerV1`` methods run as a **lane** on the same
``actor_rollout_wg`` / config. Never ``fit()``, never a second ``optimizer.step``.

Import of ``trainer_base`` is deferred: some verl pins lack ``ReplayBufferAsync``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DiffusionV1GenLane", "diffusion_v1_gen_hooks"]


def diffusion_v1_gen_hooks():
    """Return the three GEN methods from ``PolicyGradientDiffusionTrainerV1``."""
    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

    return (
        PolicyGradientDiffusionTrainerV1._compute_old_log_prob,
        PolicyGradientDiffusionTrainerV1._compute_ref_log_prob,
        PolicyGradientDiffusionTrainerV1._compute_advantage,
    )


class DiffusionV1GenLane:
    """GEN-only view of ``PolicyGradientDiffusionTrainerV1`` (no second trainer process).

    Bound methods expect ``self.config`` and ``self.actor_rollout_wg`` exactly as
    the OCR V1 trainer does. Optional ref fields are copied when the owner has them.
    """

    def __init__(self, owner: Any):
        self.config = owner.config
        self.actor_rollout_wg = owner.actor_rollout_wg
        self.ref_in_actor = bool(getattr(owner, "ref_in_actor", True))
        self.ref_policy_wg = getattr(owner, "ref_policy_wg", None)
        old, ref, adv = diffusion_v1_gen_hooks()
        self._compute_old_log_prob = old.__get__(self, DiffusionV1GenLane)
        self._compute_ref_log_prob = ref.__get__(self, DiffusionV1GenLane)
        self._compute_advantage = adv.__get__(self, DiffusionV1GenLane)
