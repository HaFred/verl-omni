# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json

import pytest

from examples.agenticrpco_trainer.agent_llm.qwen_vl_reflect_server import (
    AESTHETICS_QUESTIONS,
    CORRECTNESS_QUESTIONS,
    _parse_scores,
    _rubric_good_enough,
)


def test_parse_ten_question_rubric_and_aggregate_means():
    correctness = {
        "subject_entities": 0.9,
        "attributes": 0.7,
        "relations_layout": 0.5,
        "scene_context": 0.8,
        "completeness": 0.6,
    }
    aesthetics = {
        "composition": 0.8,
        "lighting": 0.6,
        "color": 0.7,
        "fidelity": 0.4,
        "appeal": 0.5,
    }
    raw = (
        "```json\n"
        + json.dumps(
            {
                "correctness_scores": correctness,
                "aesthetics_scores": aesthetics,
                "findings": "soft detail",
                "suggested_fixes": "increase sharpness",
            }
        )
        + "\n```"
    )

    parsed = _parse_scores(raw)

    assert parsed["correctness_scores"] == correctness
    assert parsed["aesthetics_scores"] == aesthetics
    assert parsed["correctness"] == pytest.approx(sum(correctness.values()) / 5)
    assert parsed["aesthetics"] == pytest.approx(sum(aesthetics.values()) / 5)


def test_old_two_scalar_response_remains_compatible():
    parsed = _parse_scores('{"correctness": 0.75, "aesthetics": 0.6}')

    assert parsed["correctness"] == pytest.approx(0.75)
    assert parsed["aesthetics"] == pytest.approx(0.6)
    assert set(parsed["correctness_scores"]) == set(CORRECTNESS_QUESTIONS)
    assert set(parsed["aesthetics_scores"]) == set(AESTHETICS_QUESTIONS)


def test_good_enough_uses_weakest_rubric_facet(monkeypatch):
    monkeypatch.setattr(
        "examples.agenticrpco_trainer.agent_llm.qwen_vl_reflect_server.GOOD_ENOUGH_THRESHOLD",
        0.72,
    )
    correctness = {key: 0.9 for key in CORRECTNESS_QUESTIONS}
    aesthetics = {key: 0.9 for key in AESTHETICS_QUESTIONS}
    assert _rubric_good_enough(correctness, aesthetics)

    # High means are insufficient when one visible defect remains.
    aesthetics["fidelity"] = 0.4
    assert not _rubric_good_enough(correctness, aesthetics)
