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

import pytest

from verl_omni.utils.reward_score import agentic_reward
from verl_omni.utils.reward_score.agentic_reward import compute_score

# Class 0: single generate → actor reflection + Done
_SINGLE = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": \
"a bright red apple on a white table, soft studio lighting, sharp focus"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png image_vis=512x512 mean_luma=168 edges=sharp colors=red_rich
Reflection: bright red apple on white table, sharp edges, rich color. Done.
"""

# Class 1: gen → reflection+rewrite gen → reflection+Done
_TWO_PASS = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a red apple on a white table, soft lighting"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png image_vis=512x512 mean_luma=92 edges=soft colors=red_muted
Reflection: apple present but muted reds and soft edges; rewrite for brighter lighting.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": \
"a bright red apple on a white table, strong studio lighting, highly detailed, sharp focus, richer reds"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png image_vis=512x512 mean_luma=155 edges=medium colors=red_rich
Reflection: bright red apple now matches; richer color and sharper focus. Done.
"""

_ONE = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
Done.
"""

# Gen + Done without visual reflection prose — must NOT be protocol_ok
_DONE_NO_REFLECT = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
Done. Looks good.
"""

_SINGLE_JUDGED = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a bright red apple on a white table"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
<tool_call>
{"name": "judge_image", "arguments": {"user_request": "same as user message", "image_prompt": "last"}}
</tool_call>
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.91
  aesthetics =0.88
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: bright red apple on white table, sharp edges, rich color. Done.
"""

_TWO_PASS_JUDGED = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a red apple on a white table"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
<tool_call>
{"name": "judge_image", "arguments": {"user_request": "same as user message", "image_prompt": "last"}}
</tool_call>
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.70
  aesthetics =0.70
  good_enough =NO
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: muted color; rewrite for brighter lighting.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a bright red apple, sharp focus, richer reds"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
<tool_call>
{"name": "judge_image", "arguments": {"user_request": "same as user message", "image_prompt": "last"}}
</tool_call>
VL judge on the last generated image:
  path=/tmp/b.png
  correctness=0.88
  aesthetics =0.84
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: bright red apple now matches; richer color and sharper focus. Done.
"""


def _vl_scores_for_path(image_path: str | None) -> dict:
    """Deterministic mock VL judgments keyed by fixture image path."""
    path = image_path or ""
    if path.endswith("/b.png") or "b.png" in path:
        return {
            "ok": True,
            "correctness": 0.88,
            "aesthetics": 0.84,
            "match": 0.86,
            "good_enough": True,
            "correctness_scores": {},
            "aesthetics_scores": {},
            "findings": "",
            "suggested_fixes": "none",
            "backend": "qwen3_vl",
        }
    if "low" in path:
        return {
            "ok": True,
            "correctness": 0.72,
            "aesthetics": 0.72,
            "match": 0.72,
            "good_enough": True,
            "correctness_scores": {},
            "aesthetics_scores": {},
            "findings": "",
            "suggested_fixes": "none",
            "backend": "qwen3_vl",
        }
    if "dup" in path:
        return {
            "ok": True,
            "correctness": 0.80,
            "aesthetics": 0.78,
            "match": 0.79,
            "good_enough": True,
            "correctness_scores": {},
            "aesthetics_scores": {},
            "findings": "",
            "suggested_fixes": "none",
            "backend": "qwen3_vl",
        }
    # Default: high single-pass scores (/tmp/a.png)
    return {
        "ok": True,
        "correctness": 0.91,
        "aesthetics": 0.88,
        "match": 0.90,
        "good_enough": True,
        "correctness_scores": {},
        "aesthetics_scores": {},
        "findings": "",
        "suggested_fixes": "none",
        "backend": "qwen3_vl",
    }


@pytest.fixture(autouse=True)
def _mock_vl_reflect(monkeypatch):
    """CPU tests never hit the GPU reflect server."""

    def fake_call_reflect_vlm(*, user_request, image_prompt, notes="", image_path=None):
        del user_request, image_prompt, notes
        return _vl_scores_for_path(image_path)

    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", fake_call_reflect_vlm)


def test_qwen35_xml_tool_calls_score_like_hermes():
    """Qwen3.5 native XML <function=/<parameter=> must parse like Hermes JSON."""
    xml = """\
<tool_call>
<function=generate_image>
<parameter=prompt>
a bright red apple on a white table, soft studio lighting, sharp focus
</parameter>
</function>
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png image_vis=512x512 mean_luma=168 edges=sharp colors=red_rich
<tool_call>
<function=judge_image>
<parameter=user_request>same as user message</parameter>
<parameter=image_prompt>last</parameter>
</function>
</tool_call>
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.91
  aesthetics =0.88
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: bright red apple on white table, sharp edges, rich color. Done.
"""
    out = compute_score("smoke", solution_str=xml)
    hermes = compute_score("smoke", solution_str=_SINGLE_JUDGED)
    assert out["reward_tool_call"] == 1.0
    assert out["num_generate_image_prompts"] == 1
    assert out["num_judge_image_calls"] == 1
    assert out["protocol_ok"] == 1
    assert out["score"] >= 0.55
    assert abs(out["score"] - hermes["score"]) < 1e-6


def test_vl_rubric_subscores_are_exposed_as_reward_metrics(monkeypatch):
    def fake_call(*, user_request, image_prompt, notes="", image_path=None):
        del user_request, image_prompt, notes, image_path
        return {
            "ok": True,
            "correctness": 0.70,
            "aesthetics": 0.60,
            "match": 0.65,
            "good_enough": False,
            "correctness_scores": {
                "attributes": 0.5,
                "completeness": 0.6,
                "relations_layout": 0.7,
                "scene_context": 0.8,
                "subject_entities": 0.9,
            },
            "aesthetics_scores": {
                "appeal": 0.4,
                "color": 0.5,
                "composition": 0.6,
                "fidelity": 0.7,
                "lighting": 0.8,
            },
            "findings": "",
            "suggested_fixes": "none",
            "backend": "qwen3_vl",
        }

    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", fake_call)
    out = compute_score("smoke", solution_str=_SINGLE)

    assert out["reward_correctness_subject_entities"] == 0.9
    assert out["reward_correctness_attributes"] == 0.5
    assert out["reward_aesthetics_composition"] == 0.6
    assert out["reward_aesthetics_appeal"] == 0.4


def test_empty_and_lazy_spam_are_hard_zero():
    empty = compute_score("smoke", solution_str="")
    assert empty["score"] == 0.0
    assert empty["reward_tool_call"] == 0.0
    spam = "Adorable\n" * 80
    spam_out = compute_score("smoke", solution_str=spam)
    assert spam_out["score"] == 0.0
    assert spam_out["reward_tool_call"] == 0.0
    assert spam_out["protocol_ok"] == 0


def test_one_tool_call_gets_near_zero_without_reflection():
    out = compute_score("smoke", solution_str=_ONE)
    assert out["score"] < 0.10
    assert out["protocol_ok"] == 0
    assert out["num_generate_image_prompts"] == 1
    assert out["num_judge_image_calls"] == 0
    assert out["reward_tool_call"] == 1.0
    assert out["reward_reflection"] == 0.0
    assert out["reward_tool_usage"] <= 0.05
    assert out["reward_result"] <= 0.05


def test_reward_tool_call_is_binary_decode_has_tool_call():
    """Per-rollout 0/1; batch mean == fraction of rollouts with a Hermes tool call."""
    no = compute_score("smoke", solution_str="Just thinking, no tools.")
    yes = compute_score("smoke", solution_str=_ONE)
    assert no["reward_tool_call"] == 0.0
    assert yes["reward_tool_call"] == 1.0
    batch_mean = 0.5 * (no["reward_tool_call"] + yes["reward_tool_call"])
    assert batch_mean == 0.5


def test_bare_json_is_hard_zero_vs_full_protocol():
    bare = compute_score(
        "smoke",
        solution_str='{"name": "generate_image", "arguments": {"prompt": "a cat"}}',
    )
    single = compute_score("smoke", solution_str=_SINGLE_JUDGED)
    two = compute_score("smoke", solution_str=_TWO_PASS_JUDGED)
    assert bare["score"] == 0.0
    assert bare["reward_tool_call"] == 0.0
    assert single["score"] >= 0.55
    assert single["protocol_ok"] == 1
    assert two["score"] >= 0.55
    assert two["protocol_ok"] == 1
    assert two["num_generate_image_prompts"] == 2
    assert two["num_judge_image_calls"] == 2


def test_done_without_visual_reflection_is_not_protocol_ok():
    prose = compute_score("smoke", solution_str=_DONE_NO_REFLECT)
    two = compute_score("smoke", solution_str=_TWO_PASS_JUDGED)
    assert prose["protocol_ok"] == 0
    assert prose["reward_reflection"] == 0.0
    assert prose["score"] < two["score"]


def test_two_pass_outranks_gen_only():
    two = compute_score("smoke", solution_str=_TWO_PASS_JUDGED)
    one = compute_score("smoke", solution_str=_ONE)
    assert two["score"] > one["score"]


def test_single_pass_protocol_ok_and_ca_from_vl():
    out = compute_score(
        "smoke",
        solution_str=_SINGLE_JUDGED,
        ground_truth={"user_request": "Generate an image of a bright red apple on a white table"},
    )
    assert out["protocol_ok"] == 1
    assert out["num_generate_image_prompts"] == 1
    assert out["num_judge_image_calls"] == 1
    assert out["reward_reflection"] >= 0.7
    assert out["reward_correctness"] >= 0.9
    assert out["reward_aesthetics"] >= 0.85


def test_higher_final_ca_scores_higher():
    low = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a red apple"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/low.png
Reflection: red apple present, acceptable lighting. Done.
"""
    high = compute_score("smoke", solution_str=_SINGLE)
    low_out = compute_score("smoke", solution_str=low)
    assert high["reward_correctness"] > low_out["reward_correctness"]
    assert high["score"] > low_out["score"]


def test_same_prompt_twice_below_distinct_rewrite():
    same = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a bright red apple on a white table"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
Reflection: muted color; try again.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a bright red apple on a white table"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/dup.png
Reflection: bright red apple matches. Done.
"""
    two = compute_score("smoke", solution_str=_TWO_PASS)
    dup = compute_score("smoke", solution_str=same)
    assert dup["score"] < two["score"]
    assert dup["protocol_ok"] == 0


def test_gen_without_reflection_is_not_full_protocol():
    no_reflect = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a detailed cat wearing a blue hat, sharp focus"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
"""
    incomplete = compute_score("smoke", solution_str=no_reflect)
    two = compute_score("smoke", solution_str=_TWO_PASS_JUDGED)
    assert incomplete["protocol_ok"] == 0
    assert incomplete["reward_reflection"] == 0.0
    assert incomplete["score"] < two["score"]


def test_thinking_wrapped_protocol_still_scores():
    wrapped = f"<think>\n{_TWO_PASS_JUDGED}\n</think>"
    out = compute_score("smoke", solution_str=wrapped)
    assert out["protocol_ok"] == 1
    assert out["reward_reflection"] >= 0.7
    assert out["num_generate_image_prompts"] == 2
    assert out["num_judge_image_calls"] == 2
    assert out["reward_correctness"] >= 0.85


def test_reward_brevity_prefers_short_prose():
    brief = compute_score("smoke", solution_str=_SINGLE)
    ramble = (
        _SINGLE
        + "\n"
        + (
            "Let me carefully reconsider the entire request again and debate every "
            "possible interpretation of the apple, the table, the lighting, and whether "
            "the prior fewshot demo is somehow related to this new task. "
        )
        * 12
    )
    long = compute_score("smoke", solution_str=ramble)
    assert brief["reward_brevity"] >= 0.9
    assert long["reward_brevity"] < brief["reward_brevity"]
    # Scalar mix no longer includes brevity; score can tie when C/A/Done match.


def test_hallucinated_ca_markers_do_not_bypass_vl(monkeypatch):
    """Bare / legacy reflect markers must not score; only agentic_judge ok=1 or VL."""
    real = compute_score("smoke", solution_str=_SINGLE)
    assert real["reward_correctness"] >= 0.9

    def fail_vl(*, user_request, image_prompt, notes="", image_path=None):
        del user_request, image_prompt, notes, image_path
        return None

    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", fail_vl)
    hallucinated = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
correctness=0.91 aesthetics=0.88 match=0.90 good_enough=1
agentic_reflect ok=1 good_enough=1 backend=qwen3_vl
Reflection: bright sharp cat. Done.
"""
    out = compute_score("smoke", solution_str=hallucinated)
    assert out["num_judge_image_calls"] == 0
    assert out["reward_correctness"] == 0.0
    assert out["reward_aesthetics"] == 0.0
    # A policy cannot close without feedback it actually observed.
    assert out["protocol_ok"] == 0
    assert real["score"] > out["score"]


def test_agentic_judge_obs_scores_are_reused_when_vl_unavailable(monkeypatch):
    """Successful judge_image obs supplies C/A even if call_reflect_vlm fails."""

    def fail_vl(*, user_request, image_prompt, notes="", image_path=None):
        del user_request, image_prompt, notes, image_path
        return None

    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", fail_vl)
    traj = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a bright red apple on a white table"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
<tool_call>
{"name": "judge_image", "arguments": {"user_request": "a red apple", "image_prompt": "a bright red apple"}}
</tool_call>
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.70
  aesthetics =0.70
  good_enough =YES
  findings: apple present
  suggested_fixes: none
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: VL judge reports correctness=0.70, aesthetics=0.70, good_enough=YES. Done.
"""
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_correctness"] == 0.70
    assert out["reward_aesthetics"] == 0.70
    assert out["reward_done"] == 1.0
    assert out["protocol_ok"] == 1
    assert out["score"] > 0.5


def test_open_judge_loop_without_reflection_stays_starved(monkeypatch):
    """High judge C/A must not plateau score without agent Reflection:+Done."""

    def fail_vl(*, user_request, image_prompt, notes="", image_path=None):
        del user_request, image_prompt, notes, image_path
        return None

    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", fail_vl)
    open_loop = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a bright red apple on a white table"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
<tool_call>
{"name": "judge_image", "arguments": {"user_request": "a red apple", "image_prompt": "a bright red apple"}}
</tool_call>
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.95
  aesthetics =0.90
  good_enough =YES
  findings: apple present bright color sharp
  suggested_fixes: none
  agentic_judge ok=1 stub=0 backend=vllm
"""
    closed = open_loop + "\nReflection: bright red apple matches; sharp edges, rich color. Done.\n"
    open_out = compute_score("smoke", solution_str=open_loop)
    closed_out = compute_score("smoke", solution_str=closed)
    assert open_out["reward_correctness"] >= 0.9
    assert open_out["reward_done"] == 0.0
    assert open_out["protocol_ok"] == 0
    assert open_out["score"] < 0.15
    assert closed_out["reward_done"] == 1.0
    assert closed_out["protocol_ok"] == 1
    assert closed_out["score"] > 0.7
    assert closed_out["score"] - open_out["score"] > 0.5


def test_last_agentic_judge_obs_wins(monkeypatch):
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    traj = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "apple v1"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.00
  aesthetics =0.00
  good_enough =NO
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: rewrite next.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "bright red apple sharp"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
VL judge on the last generated image:
  path=/tmp/b.png
  correctness=0.80
  aesthetics =0.75
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: bright red apple now matches. Done.
"""
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_correctness"] == 0.80
    assert out["reward_aesthetics"] == 0.75


def test_first_yes_judge_beats_failed_later_rewrite(monkeypatch):
    """YES → Done protocol: do not let a later failed rewrite overwrite C/A."""
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    traj = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "apple v1"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.95
  aesthetics =0.90
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: looks good but rewrite anyway.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "broken apple mush"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
VL judge on the last generated image:
  path=/tmp/b.png
  correctness=0.00
  aesthetics =0.00
  good_enough =NO
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: failed. Done.
"""
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_correctness"] == 0.95
    assert out["reward_aesthetics"] == 0.90
    assert out["rewrite_after_yes"] == 1
    assert out["protocol_ok"] == 0
    assert out["reward_done"] <= 0.4
    assert out["reward_delta_c"] == 0.0


def test_delta_c_bonus_after_first_no(monkeypatch):
    """NO → rewrite → higher C earns reward_delta_c (multiturn headroom)."""
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    traj = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "apple v1"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.50
  aesthetics =0.50
  good_enough =NO
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: missing color; rewrite.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "bright red apple sharp"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
VL judge on the last generated image:
  path=/tmp/b.png
  correctness=0.90
  aesthetics =0.85
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
Reflection: now matches. Done.
"""
    out = compute_score("smoke", solution_str=traj)
    assert out["first_judge_no"] == 1
    assert out["first_correctness"] == 0.50
    assert abs(float(out["reward_delta_c"]) - 0.40) < 1e-6
    assert out["reward_correctness"] == 0.90
    assert out["protocol_ok"] == 1
    assert out["rewrite_after_yes"] == 0


def test_forced_max_pass_done_does_not_earn_done_credit(monkeypatch):
    """Env-injected Done at max passes must not inflate closed-loop reward."""
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    open_loop = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "soldier letter by lamp"}}
</tool_call>
vLLM-Omni generated the requested image. path=/tmp/a.png agentic_tool ok=1 images=1 backend=vllm_omni
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.95
  aesthetics =0.84
  good_enough =NO
  agentic_judge ok=1 stub=0 backend=vllm
"""
    forced = (
        open_loop + "\nReflection: VL judge reports correctness=0.95, aesthetics=0.84, "
        "good_enough=NO after generate_image pass 3/3. 3-pass max reached — stopping. "
        "Done. agentic_force_stop_max_passes=1 agentic_forced_reflection=1\n"
    )
    policy_done = open_loop + "\nReflection: aesthetics still short of bar; stopping. Done.\n"
    out_forced = compute_score("smoke", solution_str=forced)
    out_policy = compute_score("smoke", solution_str=policy_done)
    out_open = compute_score("smoke", solution_str=open_loop)
    assert float(out_forced["reward_done"]) <= float(out_open["reward_done"]) + 1e-6
    assert float(out_policy["reward_done"]) > float(out_forced["reward_done"])
    assert float(out_policy["score"]) > float(out_forced["score"])


def test_sampled_done_after_forced_reflection_stop_cue_earns_credit(monkeypatch):
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    judged = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "soldier letter by lamp"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png backend=vllm_omni
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.95
  aesthetics =0.86
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
"""
    cue = (
        "Reflection: VL judge reports good_enough=YES. Stop now; your next action "
        "must be exactly Done. agentic_stop_decision_required=1 "
        "agentic_forced_reflection=1\n"
    )
    without_policy_done = compute_score("smoke", solution_str=judged + cue)
    with_policy_done = compute_score("smoke", solution_str=judged + cue + "Done.\n")
    assert without_policy_done["reward_done"] == 0.0
    assert without_policy_done["protocol_ok"] == 0
    assert with_policy_done["reward_done"] == 1.0
    assert with_policy_done["protocol_ok"] == 1
    assert float(with_policy_done["score"]) > float(without_policy_done["score"])


def test_planning_phrase_stop_when_done_gets_no_done_credit(monkeypatch):
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    traj = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "soldier letter by lamp"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png backend=vllm_omni
VL judge on the last generated image:
  path=/tmp/a.png
  correctness=0.95
  aesthetics =0.86
  good_enough =YES
  agentic_judge ok=1 stub=0 backend=vllm
I'll stop when Done.
"""
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_done"] == 0.0
    assert out["protocol_ok"] == 0


def test_blocked_or_no_png_rollout_cannot_earn_done_credit(monkeypatch):
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    traj = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "soldier letter by lamp"}}
</tool_call>
generate_image blocked: stale YES latch agentic_block_generate_after_yes=1
Reflection: looks accepted. Done.
"""
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_done"] == 0.0
    assert out["protocol_ok"] == 0
    assert out["rollout_valid"] == 0
    assert out["score"] == 0.0


def test_vl_unset_zeros_correctness_and_aesthetics(monkeypatch):
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)
    out = compute_score("smoke", solution_str=_SINGLE)
    assert out["reward_correctness"] == 0.0
    assert out["reward_aesthetics"] == 0.0
    assert out["protocol_ok"] == 0  # no successful judge observation
