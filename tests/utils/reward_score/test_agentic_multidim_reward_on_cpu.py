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

"""CPU tests for the terminal / forced-context parsing of the RPCO multi-dim reward.

``_terminal_decision`` reports whether an episode closed and whether its closing
reflection was policy prose or harness-injected context. ``build_forced_reflection``
writes its stop cue on the *same* line as the verdict it paraphrases, so the cue scan has
to start at the verdict rather than at the next line break. Anchoring one line late made
``forced_context`` a constant ``False``, which pinned ``reward_done`` and
``forced_reflection_context`` to 0 for every Bagel Co-RL episode (both read 0 at all 150
validation steps of ``bagel_corl_rm1_0123_20261012_084724``) while letting ``R_format``'s
forced-context check be satisfied by the harness's own ``Reflection:`` line.
"""

import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "verl_omni" / "utils" / "reward_score" / "agentic_multidim_reward.py"
_SPEC = importlib.util.spec_from_file_location("agentic_multidim_reward_under_test", _SRC)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_score = _MODULE.compute_score
_normalize_trajectory_dialect = _MODULE._normalize_trajectory_dialect
_terminal_decision = _MODULE._terminal_decision

# The Bagel Co-RL loop renders calls in the <tools> schema dialect, announces the accepted
# image as a bare absolute path, and paraphrases the in-loop RM verdict into the forced
# Reflection line (see agentic_multidim_reward's module docstring).
_GENERATE_CALL = '<tools>{"name": "generate_image", "arguments": {"prompt": "a cat riding a bicycle"}}</tools>'
_PNG_OBSERVATION = "path=/tmp/bagel_corl/gen_1.png"
_STOP_CUE = "agentic_stop_decision_required=1"
_FORCED_REFLECTION = (
    "Reflection: VL judge reports correctness=0.9000, aesthetics=0.8500, good_enough=YES. "
    "rubber_stamp=False. findings: see VL facet scores above. Stop now; do not call another "
    "tool. Your next and only action must be exactly Done."
)


def _bagel_transcript(*, cue: str = _STOP_CUE, trailing: str = "Done.") -> str:
    """One forced-closure episode: generate -> image -> forced Reflection -> policy Done."""
    forced = f"{_FORCED_REFLECTION} {cue}".strip()
    return "\n".join([_GENERATE_CALL, _PNG_OBSERVATION, forced, trailing])


def _score(transcript: str, *, task_type: str = "reflect") -> dict:
    return compute_score(
        solution_str=transcript,
        ground_truth={"task_type": task_type, "expected_num_images": 1},
    )


def test_stop_cue_on_verdict_line_sets_forced_context():
    transcript = _normalize_trajectory_dialect(_bagel_transcript())
    terminal_done, policy_reflection, forced_context = _terminal_decision(transcript)
    assert forced_context is True
    # The harness's own line must not be read as the policy's reflection.
    assert policy_reflection is False
    assert terminal_done is True


def test_forced_closure_scores_reward_done():
    result = _score(_bagel_transcript())
    assert result["reward_done"] == 1.0
    assert result["forced_reflection_context"] == 1
    assert result["terminal_done"] == 1
    assert result["terminal_policy_reflection"] == 0


def test_closure_without_any_cue_does_not_count():
    """The fix scans from the verdict; it must not invent a cue that is not there."""
    transcript = _normalize_trajectory_dialect(_bagel_transcript(cue=""))
    terminal_done, policy_reflection, forced_context = _terminal_decision(transcript)
    assert forced_context is False
    assert policy_reflection is False
    assert terminal_done is True

    result = _score(_bagel_transcript(cue=""))
    assert result["reward_done"] == 0.0
    assert result["forced_reflection_context"] == 0


def test_cue_before_the_verdict_does_not_count():
    """A cue the policy never followed a verdict with is not forced context."""
    transcript = "\n".join(
        [
            _GENERATE_CALL,
            f"{_PNG_OBSERVATION} {_STOP_CUE}",
            _FORCED_REFLECTION,
            "Done.",
        ]
    )
    terminal_done, policy_reflection, forced_context = _terminal_decision(_normalize_trajectory_dialect(transcript))
    assert forced_context is False
    assert policy_reflection is False
    assert terminal_done is True


def test_policy_reflection_after_the_verdict_still_counts():
    """The legacy path is untouched: a policy-written reflection closes the episode."""
    transcript = "\n".join(
        [
            _GENERATE_CALL,
            _PNG_OBSERVATION,
            _FORCED_REFLECTION,
            "Reflection: the wheels were too small, so I enlarged them. Done.",
        ]
    )
    terminal_done, policy_reflection, forced_context = _terminal_decision(_normalize_trajectory_dialect(transcript))
    assert policy_reflection is True
    assert terminal_done is True
    assert _score(transcript)["reward_done"] == 1.0


def test_mode_2a_cue_on_a_later_line_is_detected():
    """The Mode-2a loops append their cue on its own line; the anchor still advances."""
    transcript = "\n".join(
        [
            _GENERATE_CALL,
            _PNG_OBSERVATION,
            _FORCED_REFLECTION,
            "agentic_forced_reflection=1",
            "Done.",
        ]
    )
    terminal_done, policy_reflection, forced_context = _terminal_decision(_normalize_trajectory_dialect(transcript))
    assert forced_context is True
    assert terminal_done is True


@pytest.mark.parametrize("task_type", ["reflect", "plan"])
def test_no_verdict_means_no_terminal_decision(task_type):
    transcript = "\n".join([_GENERATE_CALL, _PNG_OBSERVATION, "Done."])
    assert _terminal_decision(transcript) == (False, False, False)
    result = _score(transcript, task_type=task_type)
    assert result["reward_done"] == 0.0
    assert result["forced_reflection_context"] == 0
