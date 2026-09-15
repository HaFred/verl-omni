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
    """Register a transparent package stub so submodule imports skip heavy ``__init__``.

    A bare stub leaks for the rest of the pytest session: any later test file asking
    for a name the stub lacks (``from verl_omni.tools.trajectory import ...``, or a
    ``monkeypatch.setattr`` on ``verl_omni.tools.trajectory.hydra_env``) then dies with
    ImportError or AttributeError, depending on collection order. ``_missing`` answers
    those from the real package without making the heavy import eager.
    """
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")

    def _missing(attr: str):
        # PEP 562 hook, invoked only for names this stub lacks, so anything a test file
        # registered here still wins. Resolution mirrors a real package in two steps,
        # only the second of which executes an ``__init__``:
        #   1. a submodule of that name — ``getattr(verl_omni, "tools")``, and the
        #      ``verl_omni.tools.trajectory.hydra_env`` that dotted-path patching needs;
        #   2. otherwise the real ``__init__.py``, lazily — a re-export such as
        #      ``from verl_omni.tools.trajectory import active_trajectory_relpath``.
        if attr.startswith("__"):
            raise AttributeError(attr)
        try:
            child = importlib.import_module(f"{pkg.__name__}.{attr}")
        except ImportError:
            pass
        else:
            pkg.__dict__[attr] = child  # real packages expose submodules as attributes
            return child
        if path.is_dir() and not pkg.__dict__.get("_real_init_loaded"):
            try:
                spec = importlib.util.spec_from_file_location(
                    pkg.__name__, path / "__init__.py", submodule_search_locations=[str(path)]
                )
                real = importlib.util.module_from_spec(spec)
                # Execute the real ``__init__`` under the package name — that is what
                # makes its relative ``from .x import y`` resolve — while this stub stays
                # the canonical ``sys.modules`` entry, so the hook above keeps working.
                real.__package__ = pkg.__name__
                spec.loader.exec_module(real)
            except Exception:  # optional/heavy deps absent: stay a stub
                pass
            else:
                pkg.__dict__["_real_init_loaded"] = True
                pkg.__dict__.update(
                    {k: v for k, v in vars(real).items() if k not in {"__getattr__", "__dict__"}}
                )
        try:
            return pkg.__dict__[attr]
        except KeyError:
            raise AttributeError(f"module {pkg.__name__!r} has no attribute {attr!r}") from None

    pkg.__getattr__ = _missing
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
