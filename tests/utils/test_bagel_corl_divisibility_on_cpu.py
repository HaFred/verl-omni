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
"""CPU tests for RFC §4.8.4 launch divisibility, the §4.4.2b R2 knob pairing, and
the launch surface's Hydra schema conformance.

The arithmetic is asserted against ``_bagel_corl_divisibility_failures`` itself,
because that is the copy the trainer actually runs; ``validate_bagel_corl_config``
is then checked once to prove the failures are surfaced rather than returned.

The last group guards a failure mode that is invisible to every other test here:
the recipes compose against ``omni_trainer``, so an override of a field that only
exists on ``DiffusionRolloutConfig`` needs Hydra's ``+`` append prefix. Without it
the job dies at startup with ``Could not override ...`` before any code under test
runs, which is exactly how the R2 knobs shipped broken once.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from verl_omni.utils.config import _bagel_corl_divisibility_failures, validate_bagel_corl_config

_RECIPE_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "agenticllmgrpo_trainer" / "bagel"
_RECIPES = ("run_agentic_bagel_rpco_lora.sh", "run_agentic_bagel_rpco_lora_longrun.sh")


def _cfg(**over: object):
    """A bagel_corl_sync recipe that passes every §4.8.4 rule, then perturb it."""
    base = {
        "trainer": {"v1": {"trainer_mode": "bagel_corl_sync"}, "n_gpus_per_node": 2},
        "data": {"train_batch_size": 4},
        "actor_rollout_ref": {
            "model": {"path": "/models/ByteDance-Seed/BAGEL-7B-MoT", "lora_rank": 64},
            "actor": {"ppo_mini_batch_size": 4, "ppo_micro_batch_size_per_gpu": 1},
            "rollout": {
                "n": 2,
                "tensor_model_parallel_size": 2,
                "engine_kwargs": {"vllm_omni": {"output_mode": "diffusion"}},
                "agent": {
                    "gen_samples_per_call": 2,
                    "max_generate_passes": 1,
                    "und_ar_serving_ready": True,
                    "und_deploy_config": "examples/agenticllmgrpo_trainer/bagel/bagel_corl_deploy_ar.yaml",
                },
            },
        },
        "reward": {"reward_model": {"rollout": {"tensor_model_parallel_size": 2}}},
    }
    cfg = OmegaConf.create(base)
    for path, value in over.items():
        OmegaConf.update(cfg, path.replace("__", "."), value)
    return cfg


# --------------------------------------------------------------------------- #
# §4.8.4 — the five checks
# --------------------------------------------------------------------------- #
def test_valid_recipe_has_no_divisibility_failures():
    assert _bagel_corl_divisibility_failures(_cfg(), n=2) == []


def test_train_batch_size_must_be_divisible_by_mini_batch():
    cfg = _cfg(data__train_batch_size=3)
    failures = _bagel_corl_divisibility_failures(cfg, n=2)
    assert len(failures) == 1
    assert "data.train_batch_size=3" in failures[0]
    assert "ppo_mini_batch_size=4" in failures[0]


def test_episode_count_mini_times_n_must_split_across_the_pool():
    """The check is on ``mini x N``, not on ``mini``: N siblings turn 1 task into N episodes."""
    cfg = _cfg(**{"trainer__n_gpus_per_node": 4, "actor_rollout_ref__rollout__tensor_model_parallel_size": 4})
    cfg.reward.reward_model.rollout.tensor_model_parallel_size = 4
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 3
    cfg.data.train_batch_size = 6
    failures = _bagel_corl_divisibility_failures(cfg, n=2)
    assert len(failures) == 1
    assert "(ppo_mini_batch_size=3 x rollout.n=2) = 6" in failures[0]
    assert "n_gpus_per_node=4" in failures[0]


def test_per_rank_episode_count_must_be_divisible_by_micro_batch():
    cfg = _cfg()
    cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 3
    failures = _bagel_corl_divisibility_failures(cfg, n=2)
    # mini*N/pool = 4*2/2 = 4, which is not a multiple of 3.
    assert len(failures) == 1
    assert "= 4" in failures[0]
    assert "ppo_micro_batch_size_per_gpu=3" in failures[0]


@pytest.mark.parametrize(
    ("override_path", "needle"),
    [
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", "(ROLLOUT_TP)=3"),
        ("reward.reward_model.rollout.tensor_model_parallel_size", "(REWARD_TP)=3"),
    ],
)
def test_pool_must_be_divisible_by_each_tensor_parallel_width(override_path: str, needle: str):
    def _pool_of_four():
        cfg = _cfg(**{"trainer__n_gpus_per_node": 4, "actor_rollout_ref__rollout__tensor_model_parallel_size": 4})
        cfg.reward.reward_model.rollout.tensor_model_parallel_size = 4
        return cfg

    # Both TP widths divide the pool of 4 -> clean.
    assert _bagel_corl_divisibility_failures(_pool_of_four(), n=2) == []

    # Breaking exactly one of them must produce exactly one failure naming it.
    broken = _pool_of_four()
    OmegaConf.update(broken, override_path, 3)
    failures = _bagel_corl_divisibility_failures(broken, n=2)
    assert len(failures) == 1
    assert "n_gpus_per_node=4" in failures[0]
    assert needle in failures[0]


def test_n_zero_is_treated_as_one_episode():
    """``rollout.n`` is validated to be >= 1 upstream, so 0 here means "unset"."""
    cfg = _cfg()
    assert _bagel_corl_divisibility_failures(cfg, n=0) == []
    assert _bagel_corl_divisibility_failures(cfg, n=1) == []


def test_absent_batch_knobs_are_skipped_not_defaulted():
    """Inventing a mini/micro here would defeat the check, so missing values pass."""
    cfg = _cfg()
    del cfg.actor_rollout_ref.actor.ppo_mini_batch_size
    del cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
    del cfg.data.train_batch_size
    assert _bagel_corl_divisibility_failures(cfg, n=2) == []


def test_non_positive_pool_is_reported_rather_than_divided_by():
    cfg = _cfg()
    cfg.trainer.n_gpus_per_node = 0
    assert _bagel_corl_divisibility_failures(cfg, n=2) == [
        "trainer.n_gpus_per_node must be a positive integer to check divisibility."
    ]


def test_validate_bagel_corl_config_surfaces_the_divisibility_failures():
    cfg = _cfg(actor_rollout_ref__actor__ppo_mini_batch_size=3)
    with pytest.raises(ValueError, match=r"§4\.8\.4"):
        validate_bagel_corl_config(cfg)


# --------------------------------------------------------------------------- #
# §4.4.2b — affinity without a cache cannot buy anything
# --------------------------------------------------------------------------- #
def test_affinity_without_cache_is_rejected():
    cfg = _cfg(
        actor_rollout_ref__rollout__enable_prompt_embed_cache=False,
        actor_rollout_ref__rollout__enable_prompt_embed_cache_routing_affinity=True,
    )
    with pytest.raises(ValueError, match=r"§4\.4\.2b"):
        validate_bagel_corl_config(cfg)


def test_cache_with_affinity_passes():
    cfg = _cfg(
        actor_rollout_ref__rollout__enable_prompt_embed_cache=True,
        actor_rollout_ref__rollout__enable_prompt_embed_cache_routing_affinity=True,
        actor_rollout_ref__rollout__prompt_embed_cache_size=32,
    )
    validate_bagel_corl_config(cfg)


def test_cache_without_affinity_is_still_legal():
    """A local (single-worker) cache is coherent on its own; only the pin needs a cache."""
    cfg = _cfg(
        actor_rollout_ref__rollout__enable_prompt_embed_cache=True,
        actor_rollout_ref__rollout__enable_prompt_embed_cache_routing_affinity=False,
    )
    validate_bagel_corl_config(cfg)


# --------------------------------------------------------------------------- #
# The recipes are the launch surface; the guards have to be there too
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("recipe", _RECIPES)
def test_recipe_sets_the_r2_knobs_and_agrees_with_the_divisibility_rules(recipe: str):
    text = (_RECIPE_DIR / recipe).read_text()

    # §4.4.2b: the pair of knobs, sized from the live working set, plus the worker
    # count the sizing depends on. The three diffusion-only knobs must carry the ``+``
    # append prefix (see the last test in this module for why the schema demands it);
    # ``agent.num_workers`` must not, or Hydra rejects it as an existing key.
    assert "+actor_rollout_ref.rollout.enable_prompt_embed_cache=$PEC_FLAG" in text
    assert "+actor_rollout_ref.rollout.enable_prompt_embed_cache_routing_affinity=$PEC_AFFINITY" in text
    assert "+actor_rollout_ref.rollout.prompt_embed_cache_size=$PEC_SIZE" in text
    assert "actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS" in text
    assert "PEC_SIZE=${PEC_SIZE:-$(( S * N * AGENT_WORKERS ))}" in text
    assert "PEC_SIZE < 32" in text

    # §4.8.4: all five checks, three of them added here.
    assert "TRAIN_BSZ % PPO_MINI != 0" in text
    assert "EPISODES_PER_UPDATE=$(( PPO_MINI * N ))" in text
    assert "EPISODES_PER_UPDATE % NUM_GPUS_ACTOR_ROLLOUT_REWARD != 0" in text
    assert "PER_RANK_EPISODES % PPO_MICRO != 0" in text
    assert "NUM_GPUS_ACTOR_ROLLOUT_REWARD % ROLLOUT_TP != 0" in text
    assert "NUM_GPUS_ACTOR_ROLLOUT_REWARD % REWARD_TP != 0" in text

    # The recipe's own copy has to be threaded into the launch, or it checks
    # nothing the run actually uses.
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI" in text
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO" in text


# ---------------------------------------------------------------------------
# Launch-surface Hydra conformance.
#
# The recipes launch ``verl_omni.trainer.main_omni`` (``config_name="omni_trainer"``),
# whose ``actor_rollout_ref.rollout`` node is typed
# ``verl.workers.config.RolloutConfig``. Diffusion-only fields (``pipeline.*``,
# ``algo.*``, and the §4.4.2b conditioning-cache trio) are *not* in that schema --
# ``_rewrite_bagel_corl_configs`` re-targets the node at ``DiffusionRolloutConfig``
# only once the trainer is running -- so at compose time they are new keys and Hydra
# demands the ``+`` append prefix. A bare override aborts the job with
# "Could not override ... To append to your config use +..." before any of the code
# the rest of this module covers ever executes.
# ---------------------------------------------------------------------------

_OMNI_TRAINER_CFG = (
    pathlib.Path(__file__).resolve().parents[2] / "verl_omni" / "trainer" / "config" / "_generated_omni_trainer.yaml"
)
# ``key=value`` / ``+key=value``; values may hold ``[]``, quotes, ``$``, ``{}``.
_OVERRIDE = re.compile(r"^(\+?)([A-Za-z_][A-Za-z0-9_.]*)(\[[^\]]*\])?=(.+)$")
_CONFIG_GROUPS = ("actor_rollout_ref.", "data.", "reward.", "trainer.", "algorithm.")


def _recipe_overrides(recipe: str) -> list[tuple[str, str]]:
    """``(prefix, key)`` for every Hydra override the recipe's launch passes."""
    found: list[tuple[str, str]] = []
    for raw in (_RECIPE_DIR / recipe).read_text().split():
        token = raw.rstrip("\\").strip().strip("'\"")
        match = _OVERRIDE.match(token)
        if not match:
            continue
        prefix, key, _, _ = match.groups()
        if key.startswith(_CONFIG_GROUPS):
            found.append((prefix, key))
    return found


def _in_composed_config(cfg: object, path: str) -> bool:
    """Whether ``path`` is a key Hydra can override in place.

    Membership must be tested against ``keys()``, not ``key in node``: OmegaConf's
    ``__contains__`` reports a ``???`` (MISSING) value as absent, yet Hydra overrides
    such a key fine without ``+`` -- e.g. ``reward.reward_model.rollout.name``, whose
    default in the generated config is ``???``. Reading such a leaf raises
    ``MissingMandatoryValue``, so the walk stops one step short on purpose.
    """
    parts = [re.sub(r"\[.*\]$", "", part) for part in path.split(".")]
    node = cfg
    for index, base in enumerate(parts):
        if not OmegaConf.is_dict(node) or base not in node.keys():  # type: ignore[union-attr]
            return False
        if index == len(parts) - 1:
            return True
        try:
            node = node[base]  # type: ignore[index]
        except MissingMandatoryValue:
            # A ``???`` node cannot be descended into, so nothing below it is a key.
            return False
    return True


@pytest.mark.parametrize("recipe", _RECIPES)
def test_every_in_place_recipe_override_exists_in_the_launched_schema(recipe: str):
    """An override without ``+`` must name a key the composed ``omni_trainer`` knows.

    This is Hydra's own check, run offline: the broken launch died with
    ``Could not override 'actor_rollout_ref.rollout.enable_prompt_embed_cache'``.
    """
    cfg = OmegaConf.load(_OMNI_TRAINER_CFG)
    offenders = [
        key for prefix, key in _recipe_overrides(recipe) if prefix != "+" and not _in_composed_config(cfg, key)
    ]
    assert offenders == [], (
        f"{recipe} passes these in place, but they are not keys of "
        f"_generated_omni_trainer.yaml, so Hydra aborts the launch before training "
        f"starts. Prefix each with '+' to append: {sorted(set(offenders))}"
    )


@pytest.mark.parametrize("recipe", _RECIPES)
def test_the_r2_knobs_are_appended_and_the_agent_worker_count_is_not(recipe: str):
    """The diffusion-only trio needs ``+``; ``agent.num_workers`` needs its absence."""
    overrides = {key: prefix for prefix, key in _recipe_overrides(recipe)}
    for key in (
        "actor_rollout_ref.rollout.enable_prompt_embed_cache",
        "actor_rollout_ref.rollout.enable_prompt_embed_cache_routing_affinity",
        "actor_rollout_ref.rollout.prompt_embed_cache_size",
    ):
        assert overrides.get(key) == "+", f"{recipe}: {key} must be appended with a '+' prefix"
    assert overrides.get("actor_rollout_ref.rollout.agent.num_workers") == "", (
        f"{recipe}: agent.num_workers already exists on RolloutConfig, so '+' there "
        f"would be rejected as a duplicate key"
    )


def _recipe_override_values(recipe: str) -> dict[str, str]:
    """``key -> raw value`` for every Hydra override the recipe's launch passes."""
    values: dict[str, str] = {}
    for raw in (_RECIPE_DIR / recipe).read_text().split():
        token = raw.rstrip("\\").strip().strip("'\"")
        match = _OVERRIDE.match(token)
        if not match:
            continue
        _, key, _, value = match.groups()
        if key.startswith(_CONFIG_GROUPS):
            values[key] = value
    return values


@pytest.mark.parametrize("recipe", _RECIPES)
def test_recipe_pins_the_rollout_lengths_to_the_dataset_lengths(recipe: str):
    """``rollout.prompt_length``/``response_length`` must track ``data.max_*``.

    They are the shape every trajectory is padded to and the budget ``run_serial_episode``
    stops the UND loop at, while ``data.max_*`` sizes the dataset and the UND AR engine's
    ``max_model_len``. Left at the ``RolloutConfig`` defaults (512/512) the recipe ran a
    1024/1024 dataset against 512-wide padding, and ``_pad_token_ids`` does not truncate:
    an over-long prompt or response comes back as a *longer* tensor, so the batch loses
    its uniform shape. ``validate_bagel_corl_config`` rejects the mismatch, so both pairs
    have to come from the same ``$MAX_PROMPT_LEN``.
    """
    values = _recipe_override_values(recipe)
    for key in (
        "data.max_prompt_length",
        "data.max_response_length",
        "actor_rollout_ref.rollout.prompt_length",
        "actor_rollout_ref.rollout.response_length",
    ):
        assert values.get(key) == "$MAX_PROMPT_LEN", (
            f"{recipe}: {key} must be $MAX_PROMPT_LEN so the rollout padding pair matches "
            f"the data pair; got {values.get(key)!r}"
        )


def test_length_contract_requires_the_rollout_pair_to_match_the_data_pair():
    """The measured 512-vs-1024 drift is rejected rather than silently mis-shaped."""
    cfg = _cfg(
        data__max_prompt_length=1024,
        data__max_response_length=1024,
        actor_rollout_ref__rollout__prompt_length=512,
        actor_rollout_ref__rollout__response_length=512,
    )
    with pytest.raises(ValueError, match="length contract"):
        validate_bagel_corl_config(cfg)


def test_length_contract_is_silent_when_both_pairs_agree():
    cfg = _cfg(
        data__max_prompt_length=1024,
        data__max_response_length=1024,
        actor_rollout_ref__rollout__prompt_length=1024,
        actor_rollout_ref__rollout__response_length=1024,
    )
    validate_bagel_corl_config(cfg)


def test_length_contract_is_skipped_when_the_data_pair_is_absent():
    """Fixtures that never declare ``data.max_*`` are unaffected."""
    assert "max_prompt_length" not in _cfg().data
    validate_bagel_corl_config(_cfg())


# --- UniCoT parquet build: self-sufficient defaults + a fatal failure ------------
#
# The recipes invoke ``build_unicot_agentic_rl`` on every launch (``REBUILD_UNICOT``
# defaults to 1) and pass ``--mix_ratio "$UNICOT_MIX_RATIO" --val_ratio
# "$UNICOT_VAL_RATIO" --seed "$UNICOT_SPLIT_SEED"``. Those used to have no ``:-``
# default in the recipe -- only ``bagel_gpu_test_env.sh`` defined them, and the
# recipe neither sources it nor requires it -- so a clean shell produced
#
#   build_unicot_agentic_rl.py: error: argument --mix_ratio/--reflect_ratio:
#   invalid float value: ''
#
# The recipe has no ``set -e`` (only ``set -x``), so the build failure was swallowed
# and the launch continued against a stale parquet: a 6-day-old
# ``outputs/data/agentic_unicot/train.parquet`` was silently trained on, so any
# dataset change was ignored. Both halves are guarded below.

#: The knobs the build command reads unguarded, with values that must be non-empty.
_REQUIRED_UNICOT_KNOBS = (
    "UNICOT_REFLECTION_DIR",
    "UNICOT_BREAKDOWN_DIR",
    "UNICOT_MIX_RATIO",
    "UNICOT_VAL_RATIO",
    "UNICOT_SPLIT_SEED",
)
#: Declared so ``${VAR:+...}`` is well-defined, but legitimately allowed to be empty.
_OPTIONAL_UNICOT_KNOBS = ("UNICOT_TRAIN_SIZE", "UNICOT_VAL_SIZE")

_UNICOT_ASSIGNMENT = re.compile(r"^(UNICOT_[A-Z0-9_]+)=(.*)$", re.MULTILINE)


def _unicot_assignments(recipe: str) -> dict[str, str]:
    """``name -> raw right-hand side`` for every ``UNICOT_*`` assignment in the recipe."""
    return dict(_UNICOT_ASSIGNMENT.findall((_RECIPE_DIR / recipe).read_text()))


@pytest.mark.parametrize("recipe", _RECIPES)
def test_recipe_gives_the_unicot_knobs_self_sufficient_defaults(recipe: str):
    """Every knob the builder reads unguarded must carry a ``:-`` default in the recipe.

    Asserted by *running* the recipe's own assignment lines in a shell with all
    ``UNICOT_*`` stripped from the environment, rather than by grepping for ``:-``:
    that is what proves a clean shell can launch, which is the property that was
    missing. ``bagel_gpu_test_env.sh`` having the same defaults is not enough,
    because nothing in the launch path sources it.
    """
    assignments = _unicot_assignments(recipe)
    for name in _REQUIRED_UNICOT_KNOBS + _OPTIONAL_UNICOT_KNOBS:
        assert name in assignments, (
            f"{recipe}: {name} is read by the UniCoT build but never assigned. Without a "
            "':-' default a clean shell passes an empty argument and argparse aborts with "
            "'invalid float value' / 'invalid int value'."
        )

    script = "\n".join(f"{name}={value}" for name, value in assignments.items())
    script += "\n" + "\n".join(f'printf "%s\\n" "${name}"' for name in _REQUIRED_UNICOT_KNOBS)
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("UNICOT_")}
    result = subprocess.run(["bash", "-c", script], env=clean_env, capture_output=True, text=True, check=True)
    resolved = dict(zip(_REQUIRED_UNICOT_KNOBS, result.stdout.splitlines(), strict=True))
    for name in _REQUIRED_UNICOT_KNOBS:
        assert resolved[name], (
            f"{recipe}: {name} resolves to empty with all UNICOT_* unset, so the build would "
            "be invoked with an empty argument."
        )


@pytest.mark.parametrize("recipe", _RECIPES)
def test_a_failed_unicot_build_refuses_to_launch(recipe: str):
    """A failed parquet build must abort the launch, not fall through to the trainer.

    With no ``set -e`` the old recipe printed the argparse error and then launched
    against whatever parquet happened to be on disk. ``exit 1`` has to be tied to the
    build command itself, before the ``else`` branch that reports reuse.
    """
    text = (_RECIPE_DIR / recipe).read_text()
    # The handler body contains ``${TRAIN_FILE}``, so the closing brace has to be matched
    # at the start of its own line rather than as the first ``}`` -- otherwise the body
    # capture stops inside that expansion and never reaches the ``exit 1``.
    build = re.search(
        r"build_unicot_agentic_rl\b[\s\S]*?\|\|\s*\{(?P<body>[\s\S]*?)\n\s*\}",
        text,
    )
    assert build, (
        f"{recipe}: the UniCoT build is not followed by a '|| {{ ... }}' failure handler, so a "
        "failed build silently falls through to the trainer (the recipe has no 'set -e')."
    )
    assert re.search(r"\bexit\s+1\b", build.group("body")), (
        f"{recipe}: the UniCoT build's failure handler does not 'exit 1', so the launch continues."
    )
    # The handler has to sit on the build branch, i.e. before the reuse branch.
    assert text.index(build.group(0)) < text.index("reusing existing UniCoT parquet"), (
        f"{recipe}: the failure handler is not attached to the build branch."
    )
