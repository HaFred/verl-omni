# Copyright 2026 Bytedance Ltd. and/or its affiliates
from verl_omni.utils.reward_score.agentic_reward import compute_r_reflect, compute_score


def test_tool_echo_without_hermes_is_penalized():
    out = compute_score(
        "x",
        solution_str=("Lance frozen MoT tool generated the requested image. Review the returned image and continue."),
        ground_truth={"user_request": "cat blue hat", "expected_num_images": 2},
        tool_extra_fields={
            "num_forced_tool_calls": 0,
            "num_successful_images": 0,
            "num_voluntary_hermes": 0,
            "num_voluntary_successful_images": 0,
            "diffusion_prompts": [],
        },
    )
    assert out["echo_penalty"] >= 0.3
    assert out["score"] == 0.0


def test_forced_only_gets_consolation_not_full_score():
    gt = {
        "user_request": "cat blue hat",
        "expected_num_images": 2,
        "w_format": 0.25,
        "w_reflect": 0.35,
        "w_tool": 0.2,
        "w_result": 0.2,
    }
    out = compute_score(
        "x",
        solution_str="agentic_tool ok=1 path=/tmp/a.png\nagentic_tool ok=1 path=/tmp/b.png",
        ground_truth=gt,
        tool_extra_fields={
            "num_forced_tool_calls": 2,
            "num_successful_images": 2,
            "num_voluntary_hermes": 0,
            "num_voluntary_successful_images": 0,
            "num_tool_calls_executed": 2,
            "diffusion_prompts": ["cat blue hat", "cat blue hat"],
        },
    )
    # No Hermes / Reflection in text → format stays 0; forced tools get consolation only.
    assert out["r_format"] == 0.0
    assert out["r_tool"] == 0.35
    assert out["score"] < 0.55


def test_teacher_forced_hermes_in_text_gets_partial_format():
    gt = {
        "user_request": "cat blue hat",
        "expected_num_images": 2,
        "w_format": 0.25,
        "w_reflect": 0.35,
        "w_tool": 0.2,
        "w_result": 0.2,
    }
    sol = (
        "Reflection: Initial generation for: cat blue hat\n"
        '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a cat with a blue hat"}}\n</tool_call>\n'
        "agentic_tool ok=1 path=/tmp/a.png\n"
        "Reflection: rewriting the diffusion prompt (pass 2).\n"
        '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": '
        '"a cat with a blue hat, highly detailed"}}\n</tool_call>\n'
        "agentic_tool ok=1 path=/tmp/b.png"
    )
    out = compute_score(
        "x",
        solution_str=sol,
        ground_truth=gt,
        tool_extra_fields={
            "num_forced_tool_calls": 2,
            "num_successful_images": 2,
            "num_voluntary_hermes": 0,
            "num_voluntary_successful_images": 0,
            "num_tool_calls_executed": 2,
            "diffusion_prompts": [
                "a cat with a blue hat",
                "a cat with a blue hat, highly detailed",
            ],
        },
    )
    assert out["r_format"] >= 0.7
    assert out["r_reflect"] >= 0.4
    assert out["score"] > 0.4


def test_rewrite_reflect_scores_high():
    r = compute_r_reflect(
        diffusion_prompts=[
            "a cat with a blue hat",
            "a cat with a blue hat, highly detailed, sharp focus, richer colors",
        ],
        user_request="cat wearing a blue hat",
        n_voluntary_hermes=2,
    )
    assert r >= 0.9


def test_voluntary_two_hermes_full_credit():
    gt = {
        "user_request": "cat blue hat",
        "expected_num_images": 2,
        "w_format": 0.25,
        "w_reflect": 0.35,
        "w_tool": 0.2,
        "w_result": 0.2,
    }
    sol = (
        "Reflection: need clearer hat color.\n"
        '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a cat with a blue hat"}}\n'
        "</tool_call>\n"
        "agentic_tool ok=1 path=/tmp/a.png\n"
        "Reflection: add detail and lighting.\n"
        '<tool_call>\n{"name": "generate_image", "arguments": '
        '{"prompt": "a cat with a blue hat, highly detailed"}}\n</tool_call>\n'
        "agentic_tool ok=1 path=/tmp/b.png"
    )
    out = compute_score(
        "x",
        solution_str=sol,
        ground_truth=gt,
        tool_extra_fields={
            "num_forced_tool_calls": 0,
            "num_successful_images": 2,
            "num_voluntary_hermes": 2,
            "num_voluntary_successful_images": 2,
            "num_tool_calls_executed": 2,
            "diffusion_prompts": [
                "a cat with a blue hat",
                "a cat with a blue hat, highly detailed",
            ],
        },
    )
    assert out["r_format"] >= 1.0
    assert out["r_reflect"] >= 0.7
    assert out["r_tool"] == 1.0
    assert out["r_result"] == 1.0
    assert out["score"] >= 0.9
