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
"""CPU tests for the RPCO multi-dimensional reward scorer."""

import json

import pytest

from verl_omni.utils.reward_score.agentic_multidim_reward import compute_score


def _gen_call(prompt: str) -> str:
    payload = json.dumps({"name": "generate_image", "arguments": {"prompt": prompt}})
    return f"<tool_call>\n{payload}\n</tool_call>"


def _gen_obs(path: str, prompt: str, *, ok: bool = True) -> str:
    return (
        f"vLLM-Omni generated the requested image. path={path} "
        f"agentic_tool ok={1 if ok else 0} stub=0 images={1 if ok else 0} backend=vllm_omni prompt={prompt[:60]!r}"
    )


def _judge_call() -> str:
    return (
        '<tool_call>\n{"name": "judge_image", "arguments": '
        '{"user_request": "same as user message", "image_prompt": "last"}}\n</tool_call>'
    )


def _judge_obs(path: str, *, correctness: float, aesthetics: float, good_enough: bool, findings: str) -> str:
    return (
        "VL judge on the last generated image:\n"
        f"  path={path}\n"
        f"  correctness={correctness:.2f}\n"
        f"  aesthetics ={aesthetics:.2f}\n"
        f"  good_enough ={'YES' if good_enough else 'NO'}\n"
        f"  findings: {findings}\n"
        "  suggested_fixes: none\n"
        "  agentic_judge ok=1 parse_ok=1 stub=0 backend=vllm parse_retries=0"
    )


def _closed_reflect_trajectory(
    *, correctness: float = 0.80, aesthetics: float = 0.76, good_enough: bool = True, extra_gens: int = 0
) -> str:
    prompt = "A vertical cafe poster with bold headline text."
    parts = [
        _gen_call(prompt),
        _gen_obs("/tmp/x/image_00_a.png", prompt),
    ]
    for i in range(extra_gens):
        parts.append(_gen_call(f"rewritten prompt {i}"))
        parts.append(_gen_obs(f"/tmp/x/image_0{i + 1}_b.png", f"rewritten prompt {i}"))
    parts.append(_judge_call())
    parts.append(
        _judge_obs(
            "/tmp/x/image_00_a.png",
            correctness=correctness,
            aesthetics=aesthetics,
            good_enough=good_enough,
            findings="headline legible high contrast footer present",
        )
    )
    parts.append("Reflection: The image renders the headline and footer correctly with high contrast. Done.")
    return "\n".join(parts)


def _gt(task_type: str = "reflect", expected: int = 1, **extra) -> dict:
    ground_truth = {
        "user_request": "A vertical cafe poster with bold headline text.",
        "task_type": task_type,
        "expected_num_images": expected,
    }
    ground_truth.update(extra)
    return ground_truth


def test_closed_reflect_trajectory_scores_near_full():
    blob = _closed_reflect_trajectory()
    gt = _gt(
        reference_steps=[
            {
                "reflection": "The image renders the headline and footer correctly with high contrast.",
                "action": "stop",
                "edit": "",
            },
        ]
    )
    out = compute_score(solution_str=blob, ground_truth=gt)

    assert out["rollout_valid"] == 1
    assert out["reward_format"] == 1.0
    assert out["reward_tool_call"] == 1.0
    assert "reward_tool" not in out
    assert out["reward_result"] == 1.0
    assert out["reward_done"] == 1.0
    assert 0.5 < out["reward_reflect"] <= 1.0
    assert out["reward_plan"] == 0.0
    assert out["score"] > 0.9
    assert out["n_successful_generates"] == 1


def test_plan_trajectory_exact_count_and_coverage():
    plan_lines = [
        "1. A snowy winter market with wooden stalls and string lights.",
        "2. The same market adding a decorated carousel in the center.",
        "3. The same scene adding a hot cocoa stand with steaming mugs.",
    ]
    blob = "\n".join(
        ["Plan:"]
        + plan_lines
        + [
            _gen_call(plan_lines[0]),
            _gen_obs("/tmp/x/image_00.png", plan_lines[0]),
            _gen_call(plan_lines[1]),
            _gen_obs("/tmp/x/image_01.png", plan_lines[1]),
            _gen_call(plan_lines[2]),
            _gen_obs("/tmp/x/image_02.png", plan_lines[2]),
            _judge_call(),
            _judge_obs(
                "/tmp/x/image_02.png",
                correctness=0.82,
                aesthetics=0.80,
                good_enough=True,
                findings="market stalls carousel and cocoa stand all present",
            ),
            "Reflection: All three subtask images compose the requested market scene. Done.",
        ]
    )
    gt = _gt(task_type="plan", expected=3, reference_subtasks=plan_lines)
    out = compute_score(solution_str=blob, ground_truth=gt)

    assert out["reward_plan"] > 0.5
    assert out["reward_result"] == 1.0
    assert out["reward_format"] == 1.0
    assert out["reward_tool_call"] == 1.0
    assert out["score"] > 0.8


def test_plan_result_exact_count_mismatch():
    plan_lines = [
        "1. A snowy winter market with wooden stalls and string lights.",
        "2. The same market adding a decorated carousel in the center.",
    ]
    blob = "\n".join(
        ["Plan:"]
        + plan_lines
        + [
            _gen_call(plan_lines[0]),
            _gen_obs("/tmp/x/image_00.png", plan_lines[0]),
            _judge_call(),
            _judge_obs(
                "/tmp/x/image_00.png",
                correctness=0.80,
                aesthetics=0.76,
                good_enough=True,
                findings="market stalls present",
            ),
            "Reflection: The market is rendered. Done.",
        ]
    )
    out = compute_score(
        solution_str=blob, ground_truth=_gt(task_type="plan", expected=2, reference_subtasks=plan_lines)
    )

    assert out["reward_result"] == 0.0  # 1 generated image vs expected 2


def test_reflect_result_is_lenient_stop_validity():
    # Early YES stop with one image on a 3-state reference row is valid.
    blob = _closed_reflect_trajectory()
    out = compute_score(solution_str=blob, ground_truth=_gt(expected=3))
    assert out["reward_result"] == 1.0

    # Count within reference is valid even when the judge said NO
    # (lenient stop-validity: R_result only guards over-generation).
    blob_no = _closed_reflect_trajectory(good_enough=False)
    out = compute_score(solution_str=blob_no, ground_truth=_gt(expected=3))
    assert out["reward_result"] == 1.0

    # Over-generating past the reference count is only rescued by a final YES.
    blob_over = _closed_reflect_trajectory(extra_gens=2, good_enough=False)
    out = compute_score(solution_str=blob_over, ground_truth=_gt(expected=1))
    assert out["reward_result"] == 0.0
    blob_over_yes = _closed_reflect_trajectory(extra_gens=2, good_enough=True)
    out = compute_score(solution_str=blob_over_yes, ground_truth=_gt(expected=1))
    assert out["reward_result"] == 1.0


def test_tool_reward_is_presence_based():
    prompt = "A poster."
    # No successful image → rollout invalid, score 0.
    blob = "\n".join(
        [
            _gen_call(prompt),
            _gen_obs("/tmp/x/none.png", prompt, ok=False),
            _judge_call(),
            _judge_obs("/tmp/x/none.png", correctness=0.1, aesthetics=0.1, good_enough=False, findings="no image"),
            "Reflection: Failed. Done.",
        ]
    )
    out = compute_score(solution_str=blob, ground_truth=_gt())
    assert out["rollout_valid"] == 0
    assert out["score"] == 0.0

    # Tool calls present but no terminal Done → f_tool_call=1, f_done=0.
    blob = "\n".join(
        [
            _gen_call(prompt),
            _gen_obs("/tmp/x/image_00.png", prompt),
            _judge_call(),
            _judge_obs("/tmp/x/image_00.png", correctness=0.8, aesthetics=0.76, good_enough=True, findings="ok"),
            "Reflection: The image looks good.",
        ]
    )
    out = compute_score(solution_str=blob, ground_truth=_gt())
    assert out["reward_tool_call"] == 1.0
    assert out["reward_done"] == 0.0


def test_rewrite_after_yes_downgrades_done():
    prompt = "A poster."
    blob = "\n".join(
        [
            _gen_call(prompt),
            _gen_obs("/tmp/x/image_00.png", prompt),
            _judge_call(),
            _judge_obs("/tmp/x/image_00.png", correctness=0.85, aesthetics=0.80, good_enough=True, findings="ok"),
            _gen_call("a rewrite after YES"),
            _gen_obs("/tmp/x/image_01.png", "a rewrite after YES"),
            _judge_call(),
            _judge_obs("/tmp/x/image_01.png", correctness=0.5, aesthetics=0.5, good_enough=False, findings="worse"),
            "Reflection: The rewrite made it worse. Done.",
        ]
    )
    out = compute_score(solution_str=blob, ground_truth=_gt())

    assert out["rewrite_after_yes"] == 1
    assert out["reward_tool_call"] == 1.0  # calls exist — presence, not the old ladder
    assert out["reward_done"] == 0.0  # rewrite-after-YES breaks the closed loop


def test_injected_forced_reflection_never_earns_credit():
    prompt = "A poster."
    blob = "\n".join(
        [
            _gen_call(prompt),
            _gen_obs("/tmp/x/image_00.png", prompt),
            _judge_call(),
            _judge_obs("/tmp/x/image_00.png", correctness=0.80, aesthetics=0.76, good_enough=True, findings="ok"),
            "Reflection: VL judge reports the image is good. agentic_forced_reflection=1",
            "Done.",
        ]
    )
    out = compute_score(solution_str=blob, ground_truth=_gt())

    assert out["terminal_done"] == 1
    assert out["forced_reflection_context"] == 1
    assert out["terminal_policy_reflection"] == 0
    assert out["reward_tool_call"] == 1.0
    assert out["reward_done"] == 1.0  # forced stop context still closes the loop


def test_weighted_total_respects_active_set():
    blob = _closed_reflect_trajectory()
    gt = _gt(
        reference_steps=[
            {
                "reflection": "The image renders the headline and footer correctly with high contrast.",
                "action": "stop",
                "edit": "",
            },
        ]
    )
    base = compute_score(solution_str=blob, ground_truth=gt)

    # w_plan is ignored on reflect rows (not in the active set W).
    gt_plan_heavy = {**gt, **{f"w_{dim}": 1.0 for dim in ("reflect", "plan", "format", "tool_call", "result")}}
    gt_plan_heavy["w_plan"] = 99.0
    out = compute_score(solution_str=blob, ground_truth=gt_plan_heavy)
    assert out["score"] == pytest.approx(base["score"])

    # All dimensions weighted zero → score 0 despite valid rollout.
    gt_zero = dict(gt, **{f"w_{dim}": 0.0 for dim in ("reflect", "format", "tool_call", "result")})
    out = compute_score(solution_str=blob, ground_truth=gt_zero)
    assert out["score"] == 0.0
    assert out["rollout_valid"] == 1

    # Legacy parquet ``w_tool`` still drives the mix (same as ``w_tool_call``).
    gt_legacy = dict(gt, **{f"w_{dim}": 0.0 for dim in ("reflect", "format", "result")})
    gt_legacy["w_tool"] = 1.0
    out = compute_score(solution_str=blob, ground_truth=gt_legacy)
    assert out["score"] == pytest.approx(out["reward_tool_call"])


def test_empty_and_invalid_rollouts_zero():
    out = compute_score(solution_str="", ground_truth=_gt())
    assert out["score"] == 0.0
    assert out["rollout_valid"] == 0
    assert out["method"] == "agentic_multidim_empty"

    out = compute_score(solution_str="Reflection: Done.", ground_truth=_gt())
    assert out["score"] == 0.0
    assert out["rollout_valid"] == 0


def _assert_full_schema(out: dict) -> None:
    for dim in ("reflect", "plan", "format", "tool_call", "result"):
        assert f"reward_{dim}" in out
    assert "reward_tool" not in out
    for key in (
        "reward_done",
        "num_hermes_tool_calls",
        "num_generate_image_prompts",
        "num_judge_image_calls",
        "judge_parse_ok",
        "protocol_ok",
        "rollout_valid",
        "terminal_done",
        "n_successful_generates",
        "task_type",
        "method",
    ):
        assert key in out


def test_full_schema_is_emitted():
    _assert_full_schema(compute_score(solution_str=_closed_reflect_trajectory(), ground_truth=_gt()))
    _assert_full_schema(compute_score(solution_str="", ground_truth=_gt()))
    _assert_full_schema(compute_score(solution_str="Reflection: Done.", ground_truth=_gt()))


def test_multidim_does_not_emit_pr1_correctness_aesthetics():
    out = compute_score(solution_str=_closed_reflect_trajectory(), ground_truth=_gt())
    assert "reward_correctness" not in out
    assert "reward_aesthetics" not in out
    empty = compute_score(solution_str="", ground_truth=_gt())
    assert "reward_correctness" not in empty
    assert "reward_aesthetics" not in empty
