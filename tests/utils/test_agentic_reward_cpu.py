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

from verl_omni.utils.reward_score.agentic_reward import compute_score

_GOOD = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png image_vis=512x512 mean_luma=90 edges=soft colors=muted
Reflection: image_vis edges=soft colors=muted; rewrite for sharper detail and richer color.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat, sharp focus, highly detailed"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
Done. Refined after reflecting on the first image.
"""

_ONE = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
Done.
"""


def test_empty_and_lazy_spam_are_hard_zero():
    assert compute_score("smoke", solution_str="")["score"] == 0.0
    spam = "Adorable\n" * 80
    assert compute_score("smoke", solution_str=spam)["score"] == 0.0
    assert compute_score("smoke", solution_str=spam)["protocol_ok"] == 0


def test_one_tool_call_gets_partial_credit():
    out = compute_score("smoke", solution_str=_ONE)
    assert out["score"] >= 0.35
    assert out["protocol_ok"] == 0
    assert out["num_generate_image_prompts"] == 1


def test_bare_json_is_hard_zero_vs_full_protocol():
    bare = compute_score(
        "smoke",
        solution_str='{"name": "generate_image", "arguments": {"prompt": "a cat"}}',
    )
    good = compute_score("smoke", solution_str=_GOOD)
    assert bare["score"] == 0.0
    assert good["score"] >= 0.55
    assert good["protocol_ok"] == 1
    assert good["reward_tool_usage"] == 1.0
    assert good["num_generate_image_prompts"] == 2


def test_two_calls_outrank_one_call():
    good = compute_score("smoke", solution_str=_GOOD)
    one = compute_score("smoke", solution_str=_ONE)
    assert good["score"] > one["score"]


def test_same_prompt_twice_below_distinct_rewrite():
    same = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat"}}
</tool_call>
Reflection: looking at the generated image, need sharper detail.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat"}}
</tool_call>
Done.
"""
    good = compute_score("smoke", solution_str=_GOOD)
    dup = compute_score("smoke", solution_str=same)
    assert dup["score"] < good["score"]
    assert dup["protocol_ok"] == 0


def test_distinct_second_call_without_reflection_is_not_full_protocol():
    no_reflection = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a detailed cat wearing a blue hat, sharp focus"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
"""
    incomplete = compute_score("smoke", solution_str=no_reflection)
    good = compute_score("smoke", solution_str=_GOOD)
    assert incomplete["protocol_ok"] == 0
    assert incomplete["reward_reflection"] == 0.0
    assert incomplete["score"] < good["score"]


def test_thinking_wrapped_reflection_and_calls_still_score():
    wrapped = f"<think>\n{_GOOD}\n</think>"
    out = compute_score("smoke", solution_str=wrapped)
    assert out["protocol_ok"] == 1
    assert out["reward_reflection"] >= 0.7
    assert out["num_generate_image_prompts"] == 2
