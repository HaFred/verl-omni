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
agentic_tool ok=1 images=1 path=/tmp/a.png
Reflection: looking at the generated image, the hat is blurry; rewrite for sharper detail.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat wearing a blue hat, sharp focus, highly detailed"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/b.png
Done. Refined after reflecting on the first image.
"""


def test_empty_and_lazy_spam_are_hard_zero():
    assert compute_score("smoke", solution_str="")["score"] == 0.0
    spam = "Adorable\n" * 80
    assert compute_score("smoke", solution_str=spam)["score"] == 0.0
    assert compute_score("smoke", solution_str=spam)["protocol_ok"] == 0


def test_one_tool_call_is_hard_zero():
    one = """\
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat"}}
</tool_call>
agentic_tool ok=1 images=1 path=/tmp/a.png
Done.
"""
    out = compute_score("smoke", solution_str=one)
    assert out["score"] == 0.0
    assert out["protocol_ok"] == 0


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
    assert good["reward_reflection"] >= 0.7
    assert good["num_generate_image_prompts"] == 2


def test_reflection_must_sit_between_tool_calls():
    wrong_place = """\
Reflection: looking at the image, soft edges.
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat"}}
</tool_call>
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "a cat, sharp focus"}}
</tool_call>
"""
    good = compute_score("smoke", solution_str=_GOOD)
    bad = compute_score("smoke", solution_str=wrong_place)
    assert bad["score"] == 0.0
    assert good["score"] > bad["score"]


def test_same_prompt_twice_is_hard_zero():
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
    assert dup["score"] == 0.0
    assert good["score"] > dup["score"]
