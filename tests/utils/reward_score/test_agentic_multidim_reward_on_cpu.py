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
"""CPU tests for the self-contained RPCO multi-dimensional reward."""

import json

import pytest

from verl_omni.agent_loop.rpco_turn_protocol import (
    build_forced_reflection,
    format_rm_scores_as_judge_text,
)
from verl_omni.utils.reward_score.agentic_multidim_reward import (
    DIMS,
    REWARD_COMPONENTS,
    compute_score,
)


def _call(name: str, **arguments: str) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def _generate(prompt: str, path: str, *, ok: bool = True) -> str:
    return "\n".join(
        (
            _call("generate_image", prompt=prompt),
            f"agentic_tool ok={int(ok)} images={int(ok)} path={path}",
        )
    )


def _judge(
    path: str,
    *,
    correctness: float = 0.8,
    aesthetics: float = 0.8,
    accepted: bool = True,
    findings: str = "headline legible and composition balanced",
) -> str:
    return "\n".join(
        (
            _call("judge_image", user_request="same as user message", image_prompt="last"),
            "VL judge on the last generated image:",
            f"path={path}",
            f"correctness={correctness}",
            f"aesthetics={aesthetics}",
            f"good_enough={'YES' if accepted else 'NO'}",
            f"findings: {findings}",
            "suggested_fixes: none",
            "agentic_judge ok=1 parse_ok=1 stub=0",
        )
    )


def _reflect_trajectory(
    *,
    correctness: float = 0.8,
    aesthetics: float = 0.8,
    accepted: bool = True,
) -> str:
    return "\n".join(
        (
            _generate("A vertical cafe poster with a bold headline.", "/tmp/image_00.png"),
            _judge(
                "/tmp/image_00.png",
                correctness=correctness,
                aesthetics=aesthetics,
                accepted=accepted,
            ),
            "Reflection: The headline is legible and the composition is balanced. Done.",
        )
    )


def _ground_truth(task_type: str = "reflect", expected: int = 1, **extra) -> dict:
    result = {
        "user_request": "A vertical cafe poster with a bold headline.",
        "task_type": task_type,
        "expected_num_images": expected,
    }
    result.update(extra)
    return result


def _plan_trajectory(lines: list[str], generated: int | None = None) -> str:
    count = len(lines) if generated is None else generated
    parts = ["Plan:", *(f"{index}. {line}" for index, line in enumerate(lines, start=1))]
    for index, line in enumerate(lines[:count]):
        parts.append(_generate(line, f"/tmp/image_{index:02d}.png"))
    parts.extend(
        (
            _judge(f"/tmp/image_{max(0, count - 1):02d}.png"),
            "Reflection: The planned subtask images satisfy the request. Done.",
        )
    )
    return "\n".join(parts)


def test_reflect_reward_blends_judge_quality_and_reference_coverage():
    reference = "The headline is legible and the composition is balanced."
    output = compute_score(
        solution_str=_reflect_trajectory(correctness=0.8, aesthetics=0.6),
        ground_truth=_ground_truth(reference_steps=[{"reflection": reference, "action": "stop"}]),
    )

    assert output["rollout_valid"] == 1
    assert output["reward_reflect"] == pytest.approx(0.85)
    assert output["reward_plan"] == 0.0
    assert output["reward_done"] == 1.0


def test_reflect_reward_falls_back_to_live_judge_findings():
    output = compute_score(
        solution_str=_reflect_trajectory(correctness=0.8, aesthetics=0.6),
        ground_truth=_ground_truth(),
    )

    # Quality is 0.7; findings tokens are a subset of the longer policy reflection
    # so F1 coverage is 5/6, not 1.0 (recall-only used to report 0.85).
    assert output["reward_reflect"] == pytest.approx(0.5 * 0.7 + 0.5 * (10 / 12))


def test_plan_reward_covers_each_reference_subtask():
    subtasks = [
        "A snowy market with wooden stalls and warm string lights.",
        "A decorated carousel centered in the same winter market.",
        "A cocoa stand with steaming mugs beside the carousel.",
    ]
    output = compute_score(
        solution_str=_plan_trajectory(subtasks),
        ground_truth=_ground_truth(task_type="plan", expected=3, reference_subtasks=subtasks),
    )

    assert output["reward_plan"] == pytest.approx(1.0)
    assert output["reward_result"] == 1.0
    assert output["reward_format"] == 1.0


def test_plan_forced_reflection_counts_toward_format():
    subtasks = ["A snowy market with wooden stalls and warm string lights."]
    parts = ["Plan:", f"1. {subtasks[0]}", _generate(subtasks[0], "/tmp/image_00.png"), _judge("/tmp/image_00.png")]
    parts.extend(("Reflection: injected stop cue agentic_forced_reflection=1", "Done."))
    output = compute_score(
        solution_str="\n".join(parts),
        ground_truth=_ground_truth(task_type="plan", expected=1, reference_subtasks=subtasks),
    )
    assert output["forced_reflection_context"] == 1
    assert output["terminal_policy_reflection"] == 0
    assert output["reward_format"] == 1.0
    assert output["protocol_ok"] == 1


def test_format_reward_is_structural_check_ratio():
    complete = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())
    open_loop = compute_score(
        solution_str="\n".join(
            (
                _generate("A cafe poster.", "/tmp/image.png"),
                _judge("/tmp/image.png"),
            )
        ),
        ground_truth=_ground_truth(),
    )

    assert complete["reward_format"] == 1.0
    assert 0.0 < open_loop["reward_format"] < 1.0
    assert open_loop["protocol_ok"] == 0


def test_tool_reward_requires_successful_generate_and_trusted_judge():
    output = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())

    assert output["reward_tool"] == 1.0
    assert output["reward_tool_call"] == 1.0

    open_loop = compute_score(
        solution_str=_generate("A poster.", "/tmp/image.png"),
        ground_truth=_ground_truth(),
    )
    assert open_loop["rollout_valid"] == 1
    assert open_loop["reward_tool"] == 0.0
    assert open_loop["reward_tool_call"] == 1.0
    assert open_loop["judge_parse_ok_rate"] == 0.0

    malformed = compute_score(
        solution_str="<tool_call>{bad json}</tool_call>\nagentic_tool ok=1 path=/tmp/image.png",
        ground_truth=_ground_truth(),
    )
    assert malformed["reward_tool"] == 0.0
    assert malformed["reward_tool_call"] == 0.0
    assert malformed["rollout_valid"] == 0


def test_plan_result_requires_exact_successful_image_count():
    subtasks = [
        "A snowy market with wooden stalls and warm lights.",
        "A decorated carousel in the same winter market.",
    ]
    exact = compute_score(
        solution_str=_plan_trajectory(subtasks),
        ground_truth=_ground_truth(task_type="plan", expected=2, reference_subtasks=subtasks),
    )
    short = compute_score(
        solution_str=_plan_trajectory(subtasks, generated=1),
        ground_truth=_ground_truth(task_type="plan", expected=2, reference_subtasks=subtasks),
    )

    assert exact["reward_result"] == 1.0
    assert short["reward_result"] == 0.0


def test_reflect_result_requires_terminal_yes_and_rejects_over_generation():
    early_yes = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(expected=3))
    early_no = compute_score(
        solution_str=_reflect_trajectory(accepted=False),
        ground_truth=_ground_truth(expected=3),
    )
    over = "\n".join(
        (
            _generate("version one", "/tmp/one.png"),
            _generate("version two", "/tmp/two.png"),
            _judge("/tmp/two.png", accepted=False),
            "Reflection: The second version is still weak. Done.",
        )
    )
    over_output = compute_score(solution_str=over, ground_truth=_ground_truth(expected=1))

    assert early_yes["reward_result"] == 1.0
    assert early_no["reward_result"] == 0.0
    assert over_output["reward_result"] == 0.0


def test_weighted_total_uses_only_the_task_active_set():
    text = _reflect_trajectory(correctness=0.8, aesthetics=0.6)
    ground_truth = _ground_truth(
        reference_steps=[{"reflection": "unrelated reference tokens", "action": "stop"}],
        w_reflect=2.0,
        w_plan=99.0,
        w_format=1.0,
        w_tool=1.0,
        w_result=1.0,
    )
    output = compute_score(solution_str=text, ground_truth=ground_truth)
    expected = (
        2 * output["reward_reflect"] + output["reward_format"] + output["reward_tool"] + output["reward_result"]
    ) / 5

    assert output["score"] == pytest.approx(expected)
    without_plan_weight = compute_score(
        solution_str=text,
        ground_truth={**ground_truth, "w_plan": 0.0},
    )
    assert without_plan_weight["score"] == pytest.approx(output["score"])


def test_zero_weights_keep_valid_rollout_but_zero_score():
    ground_truth = _ground_truth(**{f"w_{dim}": 0.0 for dim in DIMS})
    output = compute_score(solution_str=_reflect_trajectory(), ground_truth=ground_truth)

    assert output["rollout_valid"] == 1
    assert output["score"] == 0.0


def test_garbage_weights_fail_closed():
    garbage = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(w_reflect="not-a-float"))
    assert garbage["method"] == "agentic_multidim_bad_weights"
    assert garbage["rollout_valid"] == 0
    assert garbage["score"] == 0.0

    negative = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(w_format=-1.0))
    assert negative["method"] == "agentic_multidim_bad_weights"


def test_done_indicator_requires_successful_judge_and_terminal_decision():
    open_output = compute_score(
        solution_str="\n".join((_generate("A poster.", "/tmp/image.png"), _judge("/tmp/image.png"))),
        ground_truth=_ground_truth(),
    )
    closed_output = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())

    assert open_output["reward_done"] == 0.0
    assert closed_output["reward_done"] == 1.0


def test_rewrite_after_first_yes_breaks_done_indicator():
    text = "\n".join(
        (
            _generate("version one", "/tmp/one.png"),
            _judge("/tmp/one.png", accepted=True),
            _generate("version two", "/tmp/two.png"),
            _judge("/tmp/two.png", accepted=False),
            "Reflection: The unnecessary rewrite is worse. Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["rewrite_after_yes"] == 1
    assert output["reward_done"] == 0.0
    assert output["reward_result"] == 0.0


def test_forced_reflection_text_does_not_count_as_policy_reflection():
    text = "\n".join(
        (
            _generate("A poster.", "/tmp/image.png"),
            _judge("/tmp/image.png"),
            "Reflection: injected stop cue agentic_forced_reflection=1",
            "Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["forced_reflection_context"] == 1
    assert output["terminal_policy_reflection"] == 0
    assert output["terminal_done"] == 1
    assert output["reward_done"] == 1.0
    assert output["reward_format"] == 1.0
    assert output["protocol_ok"] == 1


def test_forced_reflection_text_does_not_inflate_reference_coverage():
    injected = "The headline is legible and the composition is balanced."
    text = "\n".join(
        (
            _generate("A poster.", "/tmp/image.png"),
            _judge("/tmp/image.png", correctness=0.8, aesthetics=0.6),
            f"Reflection: {injected} agentic_forced_reflection=1",
            "Done.",
        )
    )
    output = compute_score(
        solution_str=text,
        ground_truth=_ground_truth(reference_steps=[{"reflection": injected, "action": "stop"}]),
    )

    # Injected text contributes no coverage: only half of the 0.7 judge quality.
    assert output["reward_reflect"] == pytest.approx(0.35)


def test_missing_or_invalid_task_type_fails_closed():
    missing = compute_score(solution_str=_reflect_trajectory(), ground_truth={"user_request": "x"})
    assert missing["method"] == "agentic_multidim_missing_task_type"
    assert missing["rollout_valid"] == 0
    assert missing["score"] == 0.0

    bad = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(task_type="other"))
    assert bad["method"] == "agentic_multidim_missing_task_type"
    assert bad["score"] == 0.0

    from_extra = compute_score(
        solution_str=_reflect_trajectory(),
        ground_truth={"user_request": "x", "expected_num_images": 1},
        extra_info={"task_type": "reflect"},
    )
    assert from_extra["rollout_valid"] == 1
    assert from_extra["task_type"] == "reflect"


def test_solution_image_without_text_raises():
    with pytest.raises(ValueError, match="solution_str"):
        compute_score(ground_truth=_ground_truth(), solution_image=object())


def test_qwen_xml_tool_calls_are_supported():
    generate = "<tool_call><function=generate_image><parameter=prompt>A cafe poster</parameter></function></tool_call>"
    judge = (
        "<tool_call><function=judge_image>"
        "<parameter=user_request>same as user message</parameter>"
        "<parameter=image_prompt>last</parameter></function></tool_call>"
    )
    text = "\n".join(
        (
            generate,
            "agentic_tool ok=1 images=1 path=/tmp/image.png",
            judge,
            _judge("/tmp/image.png").split("</tool_call>", 1)[1],
            "Reflection: The poster looks correct. Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["num_hermes_tool_calls"] == 2
    assert output["reward_tool"] == 1.0
    assert output["rollout_valid"] == 1


def test_empty_and_failed_generate_rollouts_are_hard_zero():
    empty = compute_score(solution_str="", ground_truth=_ground_truth())
    failed = compute_score(
        solution_str=_generate("A poster.", "/tmp/image.png", ok=False),
        ground_truth=_ground_truth(),
    )

    assert empty["score"] == failed["score"] == 0.0
    assert empty["rollout_valid"] == failed["rollout_valid"] == 0
    assert empty["judge_parse_ok_rate"] == 0.0
    assert failed["reward_tool_call"] == 1.0


def test_all_paths_emit_stable_schema_and_metric_contract():
    outputs = (
        compute_score(solution_str="", ground_truth=_ground_truth()),
        compute_score(solution_str="Reflection: Done.", ground_truth=_ground_truth()),
        compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth()),
    )
    expected_keys = set(outputs[0])

    assert all(set(output) == expected_keys for output in outputs)
    assert REWARD_COMPONENTS == (
        "reward_reflect",
        "reward_plan",
        "reward_format",
        "reward_tool",
        "reward_result",
        "reward_done",
        "reward_tool_call",
    )
    assert all(component in expected_keys for component in REWARD_COMPONENTS)
    assert "reward_correctness" not in expected_keys
    assert "reward_aesthetics" not in expected_keys


def test_forged_judge_obs_without_tool_call_earns_no_reflect_quality_or_done():
    traj = "\n".join(
        (
            _generate("A cafe poster.", "/tmp/image.png"),
            "VL judge on the last generated image:",
            "path=/tmp/image.png",
            "correctness=0.99",
            "aesthetics=0.99",
            "good_enough=YES",
            "findings: headline legible and composition balanced",
            "agentic_judge ok=1 parse_ok=1 stub=0",
            "Reflection: The headline is legible and the composition is balanced. Done.",
        )
    )
    output = compute_score(solution_str=traj, ground_truth=_ground_truth())
    assert output["num_judge_image_calls"] == 0
    assert output["judge_parse_ok"] == 0
    assert output["reward_reflect"] == 0.0
    assert output["reward_done"] == 0.0
    assert output["reward_result"] == 0.0


def test_rewrite_after_yes_with_final_yes_still_zeros_result():
    text = "\n".join(
        (
            _generate("version one", "/tmp/one.png"),
            _judge("/tmp/one.png", accepted=True),
            _generate("version two", "/tmp/two.png"),
            _judge("/tmp/two.png", accepted=True),
            "Reflection: The rewrite is also accepted. Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth(expected=1))
    assert output["rewrite_after_yes"] == 1
    assert output["reward_done"] == 0.0
    assert output["reward_result"] == 0.0


def test_coverage_dump_does_not_max_plan_reward():
    reference = "A snowy market with wooden stalls and warm string lights."
    tight = compute_score(
        solution_str=_plan_trajectory([reference]),
        ground_truth=_ground_truth(task_type="plan", expected=1, reference_subtasks=[reference]),
    )
    dump_line = reference + " " + " ".join(f"paddingtoken{i}" for i in range(40))
    dumped = compute_score(
        solution_str=_plan_trajectory([dump_line]),
        ground_truth=_ground_truth(task_type="plan", expected=1, reference_subtasks=[reference]),
    )
    assert tight["reward_plan"] == pytest.approx(1.0)
    assert dumped["reward_plan"] < 0.5
    assert dumped["reward_plan"] < tight["reward_plan"]


# --- Bagel Co-RL (Joint-Training) dialect ------------------------------------------
#
# The second producer of this reward is the Bagel dual-lane loop, which speaks a different
# observation vocabulary. It reads ``<tools>{...}</tools>`` (the schema tag) rather than
# ``<tool_call>``, announces a generated image with a bare ``path=<abs>.png``, and renders
# the in-loop RM verdict as the forced ``Reflection: VL judge reports ...`` line instead of
# an ``agentic_judge ok=1`` observation. Nothing here matched, so every episode took the
# ``prompts == []`` early return and the trainer published a constant zero. Measured
# 2026-09-27 on devices 0/1/6/7 (``bagel_corl_rm1_20260927_165227``): steps 1-8 all logged
# ``critic/score/mean: 0.0`` and ``actor/pg_loss: 0.0`` -- a zero-gradient composite step --
# while the same episodes' GEN lane trained normally.


def _bagel_call(name: str, **arguments) -> str:
    """A Bagel call. No trailing newline: the loop appends its observations to token ids."""
    return f"<tools>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tools>"


def _bagel_transcript(
    *,
    prompt: str = "a cafe poster with a bold headline",
    path: str = "/tmp/bagel_corl_gen/gen_ab.png",
    correctness: float = 0.31,
    aesthetics: float = 0.31,
    accepted: bool = False,
    force_done: bool = True,
    call_text: str | None = None,
    done: bool = True,
) -> str:
    """The exact decoded shape of ``bagel_corl_lib.run_serial_episode``.

    Model turn, then ``compact_image_observation``'s bare path, then
    ``build_forced_reflection``'s forced Reflection, then the policy's terminal ``Done.``.
    Nothing separates them -- the loop appends token ids, it never inserts a newline -- so
    the path is glued to whatever the model wrote last and the stop cue is glued to ``Done.``.
    """
    judge_text = format_rm_scores_as_judge_text(
        correctness=correctness, aesthetics=aesthetics, good_enough=accepted
    )
    built = build_forced_reflection(
        judge_text, force_done=force_done, generate_pass=1, max_passes=1 if force_done else 3
    )
    assert built is not None, "the forced-reflection producer rejected its own verdict format"
    parts = [call_text if call_text is not None else _bagel_call("generate_image", prompt=prompt)]
    parts.append(f"path={path}")
    parts.append(built[0])
    if done:
        parts.append("Done.")
    return "".join(parts)


def test_bagel_tagged_calls_and_bare_observation_are_not_a_structural_zero():
    """The regression: this transcript used to score 0.0 and pin ``critic/score`` at zero."""
    text = _bagel_transcript()
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["rollout_valid"] == 1
    assert output["method"] == "agentic_multidim"
    assert output["num_hermes_tool_calls"] == 1
    assert output["num_generate_image_prompts"] == 1
    assert output["n_successful_generates"] == 1
    assert output["judge_parse_ok"] == 1
    assert output["terminal_done"] == 1
    assert output["reward_tool"] == 1.0
    assert output["reward_reflect"] == pytest.approx(0.31)
    assert output["score"] > 0.0


def test_bagel_truncated_turn_glues_the_observation_but_still_counts_it():
    """A turn that runs out of context is cut mid-JSON, so ``path=`` has no line start.

    ``_count_successful_generates`` is line-oriented and needs ``\\bagentic_tool``, which
    cannot match across the ``</tools`` + ``agentic_tool`` glue, so the Bagel form is counted
    by occurrence instead. The call itself is unparseable, so the episode is still a hard
    zero -- an observation without a parsed request cannot buy ``R_tool``.
    """
    truncated = '<tools>\n{"name": "generate_image", "arguments": {"prompt": "a cafe poster"}'
    text = _bagel_transcript(call_text=truncated)

    # The observation is glued straight onto the model's last token -- no closing tag, no
    # newline -- so the line-oriented Hermes matcher can never see the ``path=``.
    assert "}path=/" in text
    output = compute_score(solution_str=text, ground_truth=_ground_truth())
    assert output["n_successful_generates"] == 1
    assert output["num_generate_image_prompts"] == 0
    assert output["score"] == 0.0
    assert output["rollout_valid"] == 0


def test_bagel_glued_stop_cue_still_marks_the_episode_terminal():
    """``agentic_stop_decision_required=1`` is glued to ``Done.``, so the cue cannot use ``\\b``."""
    text = _bagel_transcript()
    assert "agentic_force_stop_max_passes=1 agentic_stop_decision_required=1Done." in text

    output = compute_score(solution_str=text, ground_truth=_ground_truth())
    assert output["terminal_done"] == 1
    assert output["forced_reflection_context"] == 1
    assert output["reward_done"] == 1.0
    assert output["reward_format"] == pytest.approx(1.0)


def test_bagel_accepted_verdict_earns_the_result_dimension():
    output = compute_score(
        solution_str=_bagel_transcript(correctness=0.9, aesthetics=0.9, accepted=True),
        ground_truth=_ground_truth(expected=1),
    )

    assert output["judge_parse_ok"] == 1
    assert output["terminal_done"] == 1
    assert output["reward_result"] == 1.0
    assert output["reward_reflect"] == pytest.approx(0.9)


def test_bagel_verdict_is_only_credited_after_a_generate_call():
    """The verdict judges an image *request*; without one it is not the policy's to earn."""
    text = _bagel_transcript(call_text=_bagel_call("judge_image", image_prompt="last"))
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["judge_parse_ok"] == 0
    assert output["reward_reflect"] == 0.0
    assert output["reward_tool"] == 0.0
    assert output["score"] == 0.0
    assert output["rollout_valid"] == 0


def test_bagel_transcript_without_a_generate_stays_a_hard_zero():
    output = compute_score(
        solution_str=_bagel_call("judge_image", image_prompt="last") + "Done.",
        ground_truth=_ground_truth(),
    )

    assert output["n_successful_generates"] == 0
    assert output["score"] == 0.0
    assert output["rollout_valid"] == 0


def test_bagel_normalization_is_inert_on_a_hermes_transcript():
    """Normalization rewrites *calls only*; the Hermes lane must not move.

    The prompt itself shows a ``<tools>`` block, but ``_TAGGED_CALL_RE`` accepts a JSON
    *object* inside it and the schema scaffold is a JSON *array* of function definitions, so
    the rewrite can never fire on the policy's own prompt text. Prepending that scaffold to a
    Hermes transcript must therefore leave every dim bit-identical.
    """
    plain = _reflect_trajectory(correctness=0.31, aesthetics=0.31)
    schema = '<tools>\n[{"name": "generate_image", "description": "render", "parameters": {}}]\n</tools>\n'
    assert compute_score(solution_str=schema + plain, ground_truth=_ground_truth()) == compute_score(
        solution_str=plain, ground_truth=_ground_truth()
    )

    # Dialect parity where the protocol admits it: a structurally equivalent single-generate
    # episode earns the same tool/format/done credit in either lane.
    tagged = compute_score(solution_str=_bagel_transcript(), ground_truth=_ground_truth())
    hermes = compute_score(solution_str=plain, ground_truth=_ground_truth())
    assert tagged["reward_tool"] == hermes["reward_tool"] == 1.0
    assert tagged["reward_format"] == hermes["reward_format"] == 1.0
    assert tagged["reward_done"] == hermes["reward_done"] == 1.0
    assert tagged["reward_reflect"] > 0.0
    assert hermes["reward_reflect"] > 0.0


def test_bagel_bare_line_anchored_payloads_are_normalized():
    """Measured 2026-09-27 18:12, ``step_000041/sample_19b9f8e0-6656-43f6-a8e6-eafdf2c64abe.02``.

    The checkpoint also samples its calls with *no* tag at all: the turn leads with a bare
    ``{"name": "generate_image", ...}`` object and then replays a bare ``{"name":
    "judge_image", "arguments": {}}``. ``_TAGGED_CALL_RE`` can see neither, so ``raw_blocks``
    was 0, ``_extract_tool_calls`` was empty, and the episode scored a flat ``0.0`` -- in the
    same step whose sibling, written in the ``<tools>`` dialect, scored ``0.43``. A hard zero
    that tracks the call dialect rather than the episode's quality both poisons the policy
    gradient and collapses the 2-sibling GRPO group it sits in (no baseline -> no advantage).
    """
    bare_generate = '{\n  "name": "generate_image",\n  "arguments": {\n    "prompt": "a cafe poster"\n  }\n}'
    bare_judge = '{\n  "name": "judge_image",\n  "arguments": {}\n}'
    text = _bagel_transcript(call_text=f"{bare_generate}\n\n{bare_judge}\n")
    result = compute_score(solution_str=text, ground_truth=_ground_truth())
    assert result["num_hermes_tool_calls"] == 2
    assert result["n_successful_generates"] == 1
    assert result["judge_parse_ok"] == 1
    assert result["terminal_done"] == 1
    assert result["reward_format"] == 1.0
    assert result["score"] > 0.0

    # ...and the tag-free dialect must not swallow a JSON example that sits *inside* prose.
    # The line-start anchor is the whole guard: ``emit {"name": ...}`` mid-sentence is not a call.
    prose_example = '1. Plan: emit {"name": "generate_image", "arguments": {"prompt": "x"}} for each step.'
    with_prose = compute_score(
        solution_str=_bagel_transcript(call_text=f"{prose_example}\n{_bagel_call('generate_image', prompt='a cafe poster')}"),
        ground_truth=_ground_truth(),
    )
    assert with_prose["num_hermes_tool_calls"] == 1


def test_bagel_native_toolcall_fence_is_normalized():
    """Measured 2026-09-27 18:52+ in ``outputs/bagel_corl_rm1_20260927_183640``, ``sample_1723296b``.

    The checkpoint's third dialect: its own ``<|begin_of ToolCall|>[{...}]<|end_of ToolCall|>``
    fence, whose payload is a JSON **array** of calls. ``_TAGGED_CALL_RE`` sees only ``<tools>``
    objects and ``_LINE_START_OBJECT_RE`` only line-anchored ones, so the span was invisible:
    ``raw_blocks`` 0, no parsed call, and a flat ``0.0`` for the episode whose sibling -- the
    ``<tools>`` dialect, same step, same task family -- scored ``0.38``. The loop accepts this
    fence (``bagel_corl_lib._NATIVE_CALL_TAGS``); the scorer has to read the same payload, or a
    recovered episode is *executed* by the rollout and then punished by the reward.
    """
    native_call = '<|begin_of ToolCall|>[{"name":"generate_image","arguments":{"prompt":"a cafe poster"}}]<|end_of ToolCall|>'
    result = compute_score(
        solution_str=_bagel_transcript(call_text=native_call),
        ground_truth=_ground_truth(),
    )
    assert result["num_hermes_tool_calls"] == 1
    assert result["n_successful_generates"] == 1
    assert result["reward_format"] == 1.0
    assert result["reward_tool"] == 1.0
    assert result["score"] > 0.0

    # ...and the rewrite is equivalent to the tag the scorer was written for, so a recovered
    # episode earns exactly the credit its ``<tools>`` sibling does (no dialect bonus/penalty).
    tagged = compute_score(
        solution_str=_bagel_transcript(call_text=_bagel_call("generate_image", prompt="a cafe poster")),
        ground_truth=_ground_truth(),
    )
    assert result["score"] == pytest.approx(tagged["score"])


def test_bagel_native_fence_takes_the_first_actionable_call():
    """A multi-call array drives the loop on its *first* named element, and so must the scorer."""
    array_call = (
        '<|start of ToolCall|>[{"arguments": {}}, {"name": "generate_image", "arguments": '
        '{"prompt": "a cafe poster"}}, {"name": "judge_image", "arguments": {}}]<|end of ToolCall|>'
    )
    result = compute_score(
        solution_str=_bagel_transcript(call_text=array_call),
        ground_truth=_ground_truth(),
    )
    assert result["n_successful_generates"] == 1


def test_bagel_native_fence_without_a_named_call_is_left_alone():
    """A body naming no tool must not be rewritten into a call the model never emitted."""
    prose = '<|begin_of ToolCall|>[{"subtasks": [{"prompt": "a cafe poster"}]}]<|end_of ToolCall|>'
    result = compute_score(
        solution_str=_bagel_transcript(call_text=prose),
        ground_truth=_ground_truth(),
    )
    assert result["num_hermes_tool_calls"] == 0
    assert result["rollout_valid"] == 0
    assert result["score"] == 0.0


def test_parquet_round_trip_reference_arrays_are_not_a_crash():
    """``reference_steps`` / ``reference_subtasks`` survive parquet as ``ndarray``, not list.

    ``value or []`` raises ``ValueError: The truth value of an array with more than one
    element is ambiguous``. Measured on ``outputs/data/agentic_unicot/train.parquet``: 5278
    rows carry a 1-3 element ``reference_steps`` array and 1107 a 2-3 element
    ``reference_subtasks`` one, so ~2.5k of 8679 rows would have raised the moment the
    scorer stopped taking the ``<tools>`` early return. The one-element arrays happened to
    be truthy-safe, which is why a single-sample smoke test would not have caught this.

    The reference strings are kept realistic on purpose: ``_extract_plan_lines`` drops any plan
    line under 4 tokens as noise, and the measured parquet subtasks bottom out at exactly 4
    (min tokens = 4 over 2974 items), so a toy ``"a snowy market"`` would zero ``R_plan`` for a
    reason that has nothing to do with the array coercion under test.
    """
    import numpy as np

    reference = "The headline is legible and the composition is balanced."
    steps = np.array(
        [{"reflection": reference, "action": "stop"}, {"reflection": "also fine", "action": "stop"}],
        dtype=object,
    )
    reflect = compute_score(
        solution_str=_reflect_trajectory(correctness=0.8, aesthetics=0.6),
        ground_truth=_ground_truth(reference_steps=steps),
    )
    assert reflect["reward_reflect"] > 0.0

    planned = "An African American librarian floats while reading a book in an underwater library in a cave."
    subtasks = np.array(
        [
            planned,
            "Keep the outline of the image unchanged and edit with the following details about the gown.",
        ],
        dtype=object,
    )
    plan = compute_score(
        solution_str=_plan_trajectory([planned]),
        ground_truth=_ground_truth(task_type="plan", expected=1, reference_subtasks=subtasks),
    )
    assert plan["reward_plan"] > 0.0
    assert plan["rollout_valid"] == 1

    scalar = compute_score(
        solution_str=_reflect_trajectory(),
        ground_truth=_ground_truth(reference_steps="The headline is legible."),
    )
    assert scalar["rollout_valid"] == 1
