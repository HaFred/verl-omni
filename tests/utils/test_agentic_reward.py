# Copyright 2026 Bytedance Ltd. and/or its affiliates

from verl_omni.utils.reward_score.agentic_reward import compute_score


def test_empty_response_has_zero_reward():
    assert compute_score("smoke", solution_str="")["score"] == 0.0


def test_response_length_provides_bounded_reward_variance():
    short = compute_score("smoke", solution_str="short")
    long = compute_score("smoke", solution_str="x" * 512)

    assert 0.0 < short["score"] < long["score"]
    assert long == {"score": 1.0, "method": "response_length_heuristic"}
