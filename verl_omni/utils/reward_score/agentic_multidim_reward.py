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
"""RPCO multi-dimensional reward for Hermes agentic image-generation trajectories.

The Hermes scorer of RFC #302 stage 3, composed from the shared
``agentic_reward_core`` trajectory assessment. Other RPCO scorers reuse the same core with
their own dimension set and dialect.

Wire with ``reward.reward_manager.name=naive`` so verl passes ``solution_str``.
``image_gen_tool_agent`` publishes the runtime-owned ``trajectory_messages`` in
its ``extra_fields``, and ``NaiveRewardManager`` merges those into ``extra_info``
(a ``trajectory_messages`` keyword argument is accepted as well). Assistant
entries carry ``content`` and OpenAI-style ``tool_calls`` (``id`` and
``function`` with ``name`` / ``arguments``); tool entries carry
``role="tool"``, ``tool_call_id`` and text ``content``. Structured messages are
authoritative: tool results are matched once per call ID, and judge image paths
must match the latest successful image at both call and response time. Plain
``solution_str`` or decoded tokens alone cannot authenticate tool outcomes and
receive zero reward. ``VisualRewardManager``'s ``solution_image`` raises.

The active reward set is ``{reflect, plan, format, tool, result}``. ``plan`` is
active only for plan rows. ``done`` and ``tool_call`` reproduce the PR1
closed-loop indicators for metrics, but are not additional score dimensions.
Invalid rollouts (no parsed ``generate_image`` call or no successful PNG)
receive score zero and ``rollout_valid=0``. ``task_type`` is required.

Judge C/A is trusted only in a matched runtime ``judge_image`` observation with
the tool observation header and finite scores in [0, 1]. Coverage is token F1,
so dumping reference words into a long blob does not max ``R_reflect`` /
``R_plan``. A runtime block (``blocked_after_*=1``) or a rewrite after YES zeros
``R_result`` and the Done indicator. Reflect ``R_result`` requires a terminal
trusted YES. ``R_tool`` needs a successful PNG plus a trusted judge.
"""

from __future__ import annotations

from typing import Any

from verl_omni.utils.reward_score import agentic_reward_core as core

DIMS = ("reflect", "plan", "format", "tool", "result")
#: Names consumed by ``AgenticMetricsAgentLoopManager`` when PR1 and PR3 are composed.
REWARD_COMPONENTS = tuple(f"reward_{name}" for name in (*DIMS, "done", "tool_call"))


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str | int | None]:
    """Compute the RFC #302 stage-3 reward and its complete metric schema.

    Args:
        data_source: Unused; kept for the verl ``compute_score`` signature.
        solution_str: Decoded trajectory text (NaiveRewardManager).
        ground_truth: Must include ``task_type`` (``reflect`` / ``plan``) plus
            optional references and ``w_*`` weights.
        extra_info: Fallback for task type / weights, tokenizer, and runtime
            ``trajectory_messages`` with role and tool-call identity preserved.
        **kwargs: May include runtime ``trajectory_messages``, decoded
            ``responses`` + tokenizer, or ``solution_image`` (rejected).

    Returns:
        Dict with ``score``, per-dim ``reward_*``, and metric schema fields.
    """
    del data_source
    gt = core.as_dict(ground_truth)
    metadata = dict(extra_info or {})
    task_type = core.require_task_type(gt, metadata)
    if task_type is None:
        return core.zero_result(method="agentic_multidim_missing_task_type", components=REWARD_COMPONENTS)
    weights = core.active_weights(gt, metadata, task_type=task_type, dims=DIMS)
    if weights is None:
        return core.zero_result(method="agentic_multidim_bad_weights", components=REWARD_COMPONENTS)
    expected = core.expected_num_images(gt, metadata)

    text = core.resolve_solution_text(
        solution_str, kwargs=kwargs, extra_info=metadata, scorer="agentic_multidim_reward"
    )
    kwargs.pop("solution_image", None)
    if not text.strip() and "trajectory_messages" not in kwargs and "trajectory_messages" not in metadata:
        result = core.zero_result(method="agentic_multidim_empty", components=REWARD_COMPONENTS)
        result.update(task_type=task_type, expected_num_images=expected)
        return result

    assessment = core.TrajectoryAssessment.from_messages(
        text, kwargs.get("trajectory_messages", metadata.get("trajectory_messages"))
    )
    result = core.zero_result(method="agentic_multidim", components=REWARD_COMPONENTS)
    result.update(assessment.metrics(), expected_num_images=expected, task_type=task_type)
    if not assessment.valid:
        return result

    rewards = assessment.dimension_scores(task_type=task_type, expected=expected, ground_truth=gt)
    result.update(
        score=core.weighted_score(rewards, weights),
        **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
        reward_done=float(assessment.closed),
        protocol_ok=int(rewards["format"] == 1.0),
        rollout_valid=1,
    )
    return result
