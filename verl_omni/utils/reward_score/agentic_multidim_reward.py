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
``agentic_reward_core`` primitives. Other RPCO scorers reuse the same core with
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

from verl_omni.utils.reward_score.agentic_reward_core import (
    BLOCKED_RE,
    active_weights,
    as_dict,
    expected_num_images,
    extract_plan_lines,
    extract_tool_calls,
    format_score,
    generate_prompts,
    generates_after_first_yes,
    judge_feedback,
    judge_quality,
    plan_coverage,
    reference_reflection,
    reference_subtasks,
    reflection_score,
    reflection_text,
    require_task_type,
    resolve_solution_text,
    resolve_trajectory,
    result_score,
    terminal_decision,
    tool_name,
    trusted_judges,
    weighted_score,
    zero_result,
)

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
    gt = as_dict(ground_truth)
    metadata = dict(extra_info or {})
    task_type = require_task_type(gt, metadata)
    if task_type is None:
        return zero_result(method="agentic_multidim_missing_task_type", components=REWARD_COMPONENTS)
    weights = active_weights(gt, metadata, task_type=task_type, dims=DIMS)
    if weights is None:
        return zero_result(method="agentic_multidim_bad_weights", components=REWARD_COMPONENTS)
    expected = expected_num_images(gt, metadata)

    text = resolve_solution_text(solution_str, kwargs=kwargs, extra_info=metadata, scorer="agentic_multidim_reward")
    kwargs.pop("solution_image", None)
    if not text.strip() and "trajectory_messages" not in kwargs and "trajectory_messages" not in metadata:
        result = zero_result(method="agentic_multidim_empty", components=REWARD_COMPONENTS)
        result.update(task_type=task_type, expected_num_images=expected)
        return result

    trajectory = resolve_trajectory(text, kwargs.get("trajectory_messages", metadata.get("trajectory_messages")))
    text, observations = trajectory.text, trajectory.observations
    calls = extract_tool_calls(text)
    prompts = generate_prompts(calls)
    names = [tool_name(call) for _, _, call in calls]
    judges = trusted_judges(observations)
    judge_ok = len(judges)
    judge_failed = sum(observation.name == "judge_image" for observation in observations) - judge_ok
    judge_rate = judge_ok / (judge_ok + judge_failed) if judge_ok + judge_failed else 0.0
    successful_generates = sum(observation.name == "generate_image" for observation in observations)
    last_generate_end = max((obs.end for obs in observations if obs.name == "generate_image"), default=0)
    last_judge_end = max((obs.end for obs in observations if obs.name == "judge_image"), default=0)
    terminal_judges = judges if judges and judges[-1].end == last_judge_end > last_generate_end else []
    decision = terminal_decision(text, terminal_judges)
    blocked = trajectory.blocked or bool(BLOCKED_RE.search(text))
    rewrites_after_yes = generates_after_first_yes(calls, judges)
    tool_reward = float(successful_generates >= 1 and bool(judges))
    # Plan rows keep exact-count-or-zero; only reflect discounts the rewrite a NO verdict requests.
    submission_images = (
        sum(obs.name == "generate_image" and obs.end < judges[0].end for obs in observations)
        if task_type == "reflect" and judges
        else None
    )

    result = zero_result(method="agentic_multidim", components=REWARD_COMPONENTS)
    result.update(
        num_hermes_tool_calls=len(calls),
        num_generate_image_prompts=len(prompts),
        num_judge_image_calls=sum(name == "judge_image" for name in names),
        judge_parse_ok=judge_ok,
        judge_parse_fail=judge_failed,
        judge_parse_ok_rate=float(judge_rate),
        terminal_done=int(decision.terminal_done),
        terminal_policy_reflection=int(decision.policy_reflection),
        forced_reflection_context=int(decision.forced_context),
        n_successful_generates=successful_generates,
        expected_num_images=expected,
        task_type=task_type,
        rewrite_after_yes=rewrites_after_yes,
        reward_tool_call=float(bool(calls)),
        reward_tool=tool_reward,
    )
    if not prompts or successful_generates == 0:
        return result

    valid_terminal_context = judge_ok > 0 and not blocked and rewrites_after_yes == 0
    closed = (
        valid_terminal_context and decision.terminal_done and (decision.policy_reflection or decision.forced_context)
    )
    trusted_ends = {hit.end for hit in judges}
    reference = reference_reflection(gt) or judge_feedback(
        observation.content for observation in observations if observation.end in trusted_ends
    )
    generate_indices = [index for index, name in enumerate(names) if name == "generate_image"]
    judge_indices = [index for index, name in enumerate(names) if name == "judge_image"]
    rewards = {
        "reflect": reflection_score(judge_quality(judges), reflection_text(text), reference) if judges else 0.0,
        "plan": plan_coverage(extract_plan_lines(text), reference_subtasks(gt)),
        "format": format_score(
            text,
            task_type=task_type,
            successful_generates=successful_generates,
            judged_after_last_generate=bool(judge_indices)
            and (not generate_indices or max(judge_indices) > max(generate_indices)),
            terminal_done=decision.terminal_done,
            policy_reflection=decision.policy_reflection,
            forced_context=decision.forced_context,
            plan_lines=extract_plan_lines(text),
        ),
        "tool": tool_reward,
        "result": result_score(
            task_type=task_type,
            expected=expected,
            successful_generates=successful_generates,
            terminal_done=decision.terminal_done,
            blocked=blocked,
            rewrite_after_yes=rewrites_after_yes,
            judges=judges,
            submission_images=submission_images,
        ),
    }
    result.update(
        score=weighted_score(rewards, weights),
        **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
        reward_done=float(closed),
        reward_tool_call=float(bool(calls)),
        protocol_ok=int(rewards["format"] == 1.0),
        rollout_valid=1,
    )
    return result
