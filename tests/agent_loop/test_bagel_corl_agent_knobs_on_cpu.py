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
"""RFC §5 knob SoT: ``bagel_multiturn_agent`` must fail loud, never default.

Per the RFC, ``gen_samples_per_call`` (S) / ``max_generate_passes`` /
``max_und_turns`` live solely under ``actor_rollout_ref.rollout.agent.*`` and have
**no code fallback**: a silent ``4``/``1``/``8`` would let an episode run with a
group size or turn budget nobody configured and would bypass the launch-time
checks in ``verl_omni/utils/config.py``. ``good_enough_reduction`` is read from
the same struct and previously degraded silently to best-of-S on a typo.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl_omni.agent_loop.bagel_corl import _GOOD_ENOUGH_REDUCTIONS, resolve_bagel_agent_knobs

_VALID = {
    "gen_samples_per_call": 2,
    "max_generate_passes": 1,
    "max_und_turns": 8,
}


def _cfg(**overrides):
    merged = {**_VALID, **overrides}
    return OmegaConf.create({k: v for k, v in merged.items() if v is not None})


@pytest.mark.parametrize("knob", sorted(_VALID))
def test_missing_knob_fails_loud(knob):
    with pytest.raises(ValueError, match=knob):
        resolve_bagel_agent_knobs(_cfg(**{knob: None}))


def test_resolves_declared_values_without_inventing_any():
    assert resolve_bagel_agent_knobs(_cfg()) == {
        "gen_samples_per_call": 2,
        "max_generate_passes": 1,
        "max_und_turns": 8,
        "good_enough_reduction": "any",
    }


@pytest.mark.parametrize("reduction", sorted(_GOOD_ENOUGH_REDUCTIONS))
def test_known_reductions_pass_through(reduction):
    assert resolve_bagel_agent_knobs(_cfg(good_enough_reduction=reduction))["good_enough_reduction"] == reduction


def test_unknown_reduction_fails_loud_instead_of_best_of_s():
    """A typo used to fall through to ``any``, silently changing the stop cue."""
    with pytest.raises(ValueError, match="good_enough_reduction"):
        resolve_bagel_agent_knobs(_cfg(good_enough_reduction="bogus"))


def test_knob_is_declared_on_the_agent_loop_config():
    """An undeclared knob cannot be set from a recipe (Hydra rejects the kwarg)."""
    from dataclasses import fields

    from verl_omni.workers.config.omni import BagelCorlAgentLoopConfig

    declared = {f.name for f in fields(BagelCorlAgentLoopConfig)}
    assert {"good_enough_reduction", "gen_samples_per_call", "max_generate_passes", "max_und_turns"} <= declared
    assert BagelCorlAgentLoopConfig().good_enough_reduction == "any"
