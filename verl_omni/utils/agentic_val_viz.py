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

"""Fixed holdout tasks for agentic validation visualization.

These are experiment-tracking fixtures, not part of the agent-loop protocol.
On each validate step the metrics manager runs this holdout *first* (samples
9001/9002 by default), commits ``val/generations(_plan)`` to W&B, then
evaluates the UniCoT val set. The agent loop only generates whatever batch a
provider builds and forwards prompt/image rows to
``AgenticValidationGenerationsLogger``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValVizCase:
    """One fixed validation visualization sample."""

    sample_index: int
    table_key: str
    viz_id: str
    task_type: str
    system_prompt: str
    user_request: str
    expected_num_images: int = 1
    reference_subtasks: tuple[str, ...] | None = None


class AgenticValVizProvider:
    """Build a fixed DataProto batch for holdout validation visualization."""

    def __init__(self, cases: Sequence[ValVizCase]):
        if not cases:
            raise ValueError("AgenticValVizProvider requires at least one ValVizCase")
        self.cases = list(cases)

    @property
    def sample_table_keys(self) -> dict[str, str]:
        """Map on-disk ``sample_<id>`` folders to W&B table keys."""
        return {f"sample_{case.sample_index}": case.table_key for case in self.cases}

    def build_batch(self, step, *, eos_token_id: int | None, pad_token_id: int | None) -> Any:
        """Construct a validation DataProto for the configured holdout cases."""
        from verl import DataProto

        non_tensor: dict[str, list[Any]] = {
            key: [] for key in ("raw_prompt", "index", "data_source", "reward_model", "extra_info")
        }
        for case in self.cases:
            ground_truth: dict[str, Any] = {
                "user_request": case.user_request,
                "task_type": case.task_type,
                "expected_num_images": case.expected_num_images,
            }
            if case.reference_subtasks:
                ground_truth["reference_subtasks"] = list(case.reference_subtasks)
            non_tensor["raw_prompt"].append(
                [
                    {"role": "system", "content": case.system_prompt},
                    {"role": "user", "content": case.user_request},
                ]
            )
            non_tensor["index"].append(case.sample_index)
            non_tensor["data_source"].append("agentic_val_viz")
            non_tensor["reward_model"].append({"style": "rule", "ground_truth": ground_truth})
            non_tensor["extra_info"].append(
                {
                    "viz_id": case.viz_id,
                    "task_type": case.task_type,
                    "expected_num_images": case.expected_num_images,
                    "raw_prompt": case.user_request,
                }
            )
        batch = DataProto.from_single_dict({key: np.array(value, dtype=object) for key, value in non_tensor.items()})
        batch.meta_info = {
            "global_steps": step,
            "validate": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
        }
        return batch


def _cafe_poster_cases() -> list[ValVizCase]:
    """UniCoT reflect/plan cafe-poster holdout used by the RPCO e2e recipe."""
    from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl

    task = (
        'A vertical artistic cafe poster. The headline at the top reads "ARTISAN ROAST". '
        "The center features a detailed, warm-toned illustration of a ceramic coffee cup sitting "
        "on a rustic wooden table with soft steam rising and gentle morning sunlight coming through "
        'a nearby window. Surrounding text at the bottom reads "Freshly Brewed Daily — Open at 7 AM". '
        "Cozy, warm amber and brown color grading, shallow depth of field, cozy aesthetic."
    )
    user_text = build_unicot_agentic_rl._with_brevity(task)
    return [
        ValVizCase(
            sample_index=9001,
            table_key="val/generations",
            viz_id="reflect_prompt",
            task_type="reflect",
            system_prompt=build_unicot_agentic_rl.REFLECT_SYSTEM_PROMPT,
            user_request=user_text,
            expected_num_images=1,
        ),
        ValVizCase(
            sample_index=9002,
            table_key="val/generations_plan",
            viz_id="plan_prompt",
            task_type="plan",
            system_prompt=build_unicot_agentic_rl.PLAN_SYSTEM_PROMPT,
            user_request=user_text,
            expected_num_images=1,
            reference_subtasks=(task,),
        ),
    ]


def resolve_agentic_val_viz_provider() -> AgenticValVizProvider | None:
    """Return the enabled holdout viz provider, or ``None`` when gated off.

    Gate: ``AGENTIC_VAL_VIZ=1``. Recipe selection defaults to the cafe-poster
    holdout; override with ``AGENTIC_VAL_VIZ_PROVIDER=cafe_poster``.
    """
    if os.getenv("AGENTIC_VAL_VIZ", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    provider_name = os.getenv("AGENTIC_VAL_VIZ_PROVIDER", "cafe_poster").strip().lower() or "cafe_poster"
    try:
        if provider_name in {"cafe_poster", "cafe", "default"}:
            return AgenticValVizProvider(_cafe_poster_cases())
    except Exception as exc:  # noqa: BLE001
        logger.warning("val viz disabled (provider %s failed): %s", provider_name, exc)
        return None
    logger.warning("Unknown AGENTIC_VAL_VIZ_PROVIDER=%s; val viz disabled", provider_name)
    return None
