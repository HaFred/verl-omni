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
"""CPU tests for shared RPCO turn protocol helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _ensure_pkg(name: str, path: Path) -> None:
    """Register a package stub so submodule imports skip heavy ``__init__`` side effects."""
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")
    sys.modules[name] = pkg


def _load_protocol():
    root = Path(__file__).resolve().parents[2]
    _ensure_pkg("verl_omni", root / "verl_omni")
    _ensure_pkg("verl_omni.agent_loop", root / "verl_omni" / "agent_loop")
    path = root / "verl_omni" / "agent_loop" / "rpco_turn_protocol.py"
    name = "verl_omni.agent_loop.rpco_turn_protocol"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


proto = _load_protocol()


def _judge(*, good_enough: bool, rubber_stamp: bool = False) -> str:
    return proto.format_rm_scores_as_judge_text(
        correctness=0.9 if good_enough else 0.4,
        aesthetics=0.8 if good_enough else 0.3,
        good_enough=good_enough,
        findings="subject matches",
        suggested_fixes="none" if good_enough else "add more detail",
        rubber_stamp=rubber_stamp,
    )


def test_good_enough_yes_requires_stop():
    result = proto.build_forced_reflection(_judge(good_enough=True))
    assert result is not None
    text, stop_required = result
    assert stop_required is True
    assert "good_enough=YES" in text
    assert "agentic_stop_decision_required=1" in text
    assert "Done." in text


def test_good_enough_no_continues():
    result = proto.build_forced_reflection(_judge(good_enough=False), force_done=False, generate_pass=1, max_passes=3)
    assert result is not None
    text, stop_required = result
    assert stop_required is False
    assert "good_enough=NO" in text
    assert "Rewriting" in text


def test_force_done_max_pass_requires_stop():
    result = proto.build_forced_reflection(
        _judge(good_enough=False),
        force_done=True,
        generate_pass=3,
        max_passes=3,
    )
    assert result is not None
    text, stop_required = result
    assert stop_required is True
    assert "agentic_force_stop_max_passes=1" in text
    assert "agentic_stop_decision_required=1" in text


def test_rubber_stamp_parsing():
    assert proto.parse_rubber_stamp("rubber_stamp=True") is True
    assert proto.parse_rubber_stamp("rubber_stamp=YES") is True
    assert proto.parse_rubber_stamp("rubber_stamp=1") is True
    assert proto.parse_rubber_stamp("rubber_stamp=False") is False
    assert proto.parse_rubber_stamp("rubber_stamp=NO") is False
    assert proto.parse_rubber_stamp("no stamp here") is False
    stamped = _judge(good_enough=True, rubber_stamp=True)
    assert proto.parse_rubber_stamp(stamped) is True


def test_format_rm_scores_round_trips_through_forced_reflection():
    judge = proto.format_rm_scores_as_judge_text(
        correctness=0.91,
        aesthetics=0.77,
        good_enough=True,
        findings="faces aligned",
        suggested_fixes="none",
        similarity=0.85,
    )
    assert "agentic_judge ok=1" in judge
    assert "unicot_similarity=0.8500" in judge
    result = proto.build_forced_reflection(judge)
    assert result is not None
    text, stop_required = result
    assert stop_required is True
    assert "0.91" in text
    assert "0.77" in text


def test_hermes_tool_call_wire_format():
    text = proto.hermes_tool_call("generate_image", prompt="a cat")
    assert text.startswith("<tool_call>")
    assert '"name": "generate_image"' in text
    assert '"prompt": "a cat"' in text
    assert text.endswith("</tool_call>")
