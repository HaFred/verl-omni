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
"""Rank-0 colocation for the Bagel Co-RL recipe: placement, not budgets.

``_ensure_dual_role_rollout`` places the UND AR replica on the **first**
``und_n_gpus``-wide slice of the actor pool, so the cards it spans also carry a GEN
engine rank and an actor shard. ``free_cache_engine=True`` (forced by the rewrite)
means a GEN engine is asleep at that point and ``wake_up`` re-maps its whole pool via
``cumem create_and_map``. Measured on a 4-device run (``/tmp/gpu_trace.log``,
``/tmp/bagel_memprobe.log``)::

    GPU 0: 71.9GiB   GPUs 1-3: 2.8GiB          # both AR stages pinned to device 0
    (Worker 0) Process-scoped GPU memory after model loading: 28.21 GiB.
    Sleep Level 1: physically freed 28.56 GiB, 2.15 GiB is still in use.
    ERROR [diffusion_worker.py:765] Wake-up failed on Rank 0:
        CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:163

The cause is placement, not a budget knob:

* every ``bagel_think``/``bagel_single_stage`` engine stage loads its **own** copy of
  the full ~27.5GiB Bagel checkpoint (~28.21GiB process-scoped) -- the GEN rank (one
  copy) plus the AR Thinker (one copy) plus the AR DiT (one copy) is ~85GiB of weights
  on one 80GiB card, and
* ``gpu_memory_utilization`` does not bound that: a GEN rank declared 0.25
  (19.9GiB) and still held 28.21GiB, while the AR's stage-0 budget only sizes its KV
  cache (6.93GiB at 0.45, against ~31.1GiB of util-independent weights + peak).

So the fix under test is the split: Thinker on logical 0, DiT on logical 1, with an
``und_n_gpus`` pool wide enough for the replica's process to see both -- which also needs
the replica to *declare* that width, or ``gpus_per_replica_node`` collapses the server
back onto one card. These checks keep the recipe's pool width, the AR deploy yaml and the
trainer's width declaration in step, keep at most one AR stage per card, and keep the AR
budget above vLLM's own engine-init floor.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RECIPE_DIR = _REPO_ROOT / "examples" / "agenticllmgrpo_trainer" / "bagel"
_RECIPES = (
    _RECIPE_DIR / "run_agentic_bagel_rpco_lora.sh",
    _RECIPE_DIR / "run_agentic_bagel_rpco_lora_longrun.sh",
)
_AR_DEPLOY = _RECIPE_DIR / "bagel_corl_deploy_ar.yaml"

# H800 SXM, the box the recipe comments target. Budgets below are fractions of this.
_CARD_GIB = 81559 / 1024

# Floor imposed by vLLM's own init check, not by free memory: 0.30 and 0.35 aborted
# ``_init_und`` with "No available memory for the cache blocks" while 0.45 initialised
# fine. The non-KV part is util-independent (28.16GiB consumed + 2.96GiB peak over a
# 27.54GiB checkpoint = 31.12GiB), so the floor is 31.12 / 79.66 = 0.391 -> round up.
_UND_UTIL_FLOOR = 0.40


# Measured per-engine footprints on the target box (see the module docstring for the log
# lines). These are what the placement has to fit, and none of them is a util knob.
_AR_STAGE0_GIB = 28.16 + 2.96  # Thinker: consumed (weights + non-torch) + peak activation
_AR_STAGE1_GIB = 28.21  # DiT: "Process-scoped GPU memory after model loading"
_GEN_WAKE_REMAP_GIB = 28.56  # GEN level-1 sleep: "physically freed 28.56 GiB" to re-map
_ACTOR_RESIDENT_GIB = 9.84  # actor "device memory used/total (GB): 9.84/79.11"

# What GPU 0 actually measured on the 2026-09-17 01:43 run, from the allocator's own
# report: actor 12.17GiB, four vLLM-Omni front-ends 4 x 520MiB, GEN 28.99GiB, AR Thinker
# 35.73GiB, against a 79.11GiB card. The first ``update_weights`` then died in
# ``restore_base_model_weights`` wanting 1.016GiB with 0.943GiB free -- ~74MiB short.
_OOM_RUN_ACTOR_GIB = 12.17
_OOM_RUN_FRONTENDS_GIB = 4 * (520 / 1024)
_OOM_RUN_GEN_GIB = 28.99
_OOM_RUN_AR_STAGE0_GIB = 35.73
_OOM_RUN_ALLOC_GIB = 1090519040 / 1024**3
_OOM_RUN_FREE_GIB = 1012465664 / 1024**3


def _recipe_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _default_after_env(text: str, name: str) -> float:
    """Extract ``NAME=${NAME:-<default>}`` from a recipe."""
    match = re.search(rf"^{name}=\$\{{{name}:-([0-9.]+)\}}$", text, flags=re.MULTILINE)
    assert match is not None, f"{name} default not found in recipe"
    return float(match.group(1))


def _gen_tp_and_deploy(text: str) -> tuple[tuple[int, str], tuple[int, str]]:
    """Extract the recipe's (ROLLOUT_TP, GEN deploy yaml) for >=4 cards and for <4.

    The two are chosen together on purpose: a diffusion stage's ``devices`` list is a
    per-replica template whose length must equal ``tensor_parallel_size``, so the yaml
    cannot be paired with the wrong width.
    """
    match = re.search(
        r'if \[\[ "\$\{NUM_GPUS_ACTOR_ROLLOUT_REWARD\}" -ge 4 \]\]; then\s*'
        r"ROLLOUT_TP=\$\{ROLLOUT_TP:-(\d+)\}\s*"
        r'_GEN_DEPLOY_DEFAULT="\$\(dirname "\$0"\)/([^"]+)"\s*'
        r"else.*?"
        r"ROLLOUT_TP=\$\{ROLLOUT_TP:-(\d+)\}\s*"
        r'_GEN_DEPLOY_DEFAULT="\$\(dirname "\$0"\)/([^"]+)"',
        text,
        flags=re.DOTALL,
    )
    assert match is not None, "recipe no longer selects ROLLOUT_TP and the GEN deploy yaml together"
    big_tp, big_yaml, small_tp, small_yaml = match.groups()
    return (int(big_tp), big_yaml), (int(small_tp), small_yaml)


def _gen_stage_devices(yaml_name: str) -> list[list[str]]:
    """Per-stage device lists from a GEN deploy yaml (comma-separated templates)."""
    deploy = yaml.safe_load((_RECIPE_DIR / yaml_name).read_text(encoding="utf-8"))
    return [[d.strip() for d in str(stage.get("devices", "")).split(",") if d.strip()] for stage in deploy["stages"]]


def _ar_stage_devices() -> dict[int, str]:
    deploy = yaml.safe_load(_AR_DEPLOY.read_text(encoding="utf-8"))
    return {int(stage["stage_id"]): str(stage.get("devices")) for stage in deploy["stages"]}


@pytest.mark.parametrize("recipe", _RECIPES, ids=lambda p: p.name)
def test_recipe_wires_the_und_budget_into_the_agent_config(recipe):
    """Without this override the AR engine silently falls back to the 0.40 code default."""
    text = _recipe_text(recipe)
    assert "+actor_rollout_ref.rollout.agent.und_gpu_memory_utilization=$UND_GPU_MEM_UTIL" in text
    assert "+actor_rollout_ref.rollout.agent.und_n_gpus=$UND_N_GPUS" in text


@pytest.mark.parametrize("recipe", _RECIPES, ids=lambda p: p.name)
def test_recipe_und_budget_matches_the_ar_deploy_stage(recipe):
    """The RolloutConfig knob and the stage yaml must not drift apart.

    Either one can win in vLLM-Omni's stage resolution, so a mismatch means the
    engine budget is whatever the loser says -- exactly the state that OOM'd.
    """
    recipe_util = _default_after_env(_recipe_text(recipe), "UND_GPU_MEM_UTIL")
    deploy = yaml.safe_load(_AR_DEPLOY.read_text(encoding="utf-8"))
    stage_utils = {stage["stage_id"]: stage.get("gpu_memory_utilization") for stage in deploy["stages"]}
    assert stage_utils[0] == pytest.approx(recipe_util), (
        f"{recipe.name}: UND_GPU_MEM_UTIL={recipe_util} but {_AR_DEPLOY.name} stage 0 "
        f"gpu_memory_utilization={stage_utils[0]}"
    )


def test_ar_stages_do_not_share_one_card():
    """One card must never hold both AR stages: that is three checkpoint copies.

    Each stage loads its own ~28.21GiB copy of the checkpoint. Pinned together they
    left GPU 0 at 71.9GiB with the GEN rank on the same card, so the GEN level-1
    sleep's 28.56GiB re-map had ~7.8GiB to land in and the run died in
    ``cumem create_and_map`` ("Wake-up failed on Rank 0").
    """
    devices = _ar_stage_devices()
    assert len(set(devices.values())) == len(devices), (
        f"AR stages share a card: {devices}. Stage devices are logical indices into the "
        "replica's CUDA_VISIBLE_DEVICES, so 'devices: \"0\", \"1\"' spreads Thinker (GPU 0) "
        "and DiT (GPU 1) over the replica's pool -- the layout run_bagel_und_ar_serve.sh "
        "already applies for two visible cards"
    )


@pytest.mark.parametrize("recipe", _RECIPES, ids=lambda p: p.name)
def test_und_pool_is_wide_enough_for_every_stage_device(recipe):
    """``und_n_gpus`` sets the replica's slice, i.e. its ``CUDA_VISIBLE_DEVICES``.

    A stage asking for logical device 1 cannot be resolved out of a 1-GPU pool, which
    is how both stages silently ended up on GPU 0. Keep the pool at least as wide as
    the highest stage device index + 1.
    """
    und_n_gpus = int(_default_after_env(_recipe_text(recipe), "UND_N_GPUS"))
    needed = 1 + max(int(device) for device in _ar_stage_devices().values())
    assert und_n_gpus >= needed, (
        f"{recipe.name}: UND_N_GPUS={und_n_gpus} but {_AR_DEPLOY.name} spreads its stages over "
        f"{needed} logical devices; stage devices are indices into the replica's own "
        f"CUDA_VISIBLE_DEVICES, so device {needed - 1} would be invisible"
    )


def _trainer_text() -> str:
    return (_REPO_ROOT / "verl_omni" / "trainer" / "omni" / "bagel_corl_trainer.py").read_text(encoding="utf-8")


def test_und_ar_replica_declares_its_pool_width():
    """The AR clone must declare ``world_size`` = the actor-pool slice it is handed.

    ``RolloutReplica`` builds the server's ``CUDA_VISIBLE_DEVICES`` from
    ``gpus_per_replica_node = min(n_gpus_per_node, world_size)`` with
    ``world_size = TP * DP * PP``, and ``vLLMReplica.launch_servers`` asserts the
    worker count equals ``world_size``. Declaring 1 (the old value, with
    ``und_n_gpus`` workers in the pool) is what left the server on GPU 0 alone and
    pinned both AR stages there -- the OOM in the module docstring. The width has to
    be declared as TP: the DP path also injects ``data_parallel_size_local``, which is
    neither an ``OrchestratorArgs`` field nor a deploy field, so it is not filtered out
    as an orchestrator key and would reach every stage against a DP=1 pin
    ("data_parallel_size_local (2) must be <= data_parallel_size (1)").
    """
    text = _trainer_text()
    assert '"tensor_model_parallel_size": und_n_gpus' in text, (
        "the UND AR replica must declare its width through tensor_parallel_size=und_n_gpus; "
        "see this test's docstring for why DP cannot be used instead"
    )
    assert '"data_parallel_size": 1' in text and '"pipeline_model_parallel_size": 1' in text
    assert '"n_gpus_per_node": und_n_gpus' in text


def test_und_stage_overrides_pin_every_declared_stage_to_one_rank():
    """Every stage of the AR deploy yaml must be pinned back to a single rank.

    A plain top-level engine arg is copied into *every* stage by the deploy-config
    path (``build_stage_runtime_overrides`` keeps non-orchestrator keys, and for a
    diffusion stage ``_apply_diffusion_parallel_runtime_overrides`` folds
    ``tensor_parallel_size`` into its ``parallel_config``), so the replica-width TP
    above would otherwise ask each stage for two ranks while the yaml gives it one
    ``devices`` entry. Exercised against vLLM-Omni itself rather than by pattern
    matching, and the derived id list must cover the yaml's stages.
    """
    from vllm_omni.config.stage_config import build_stage_runtime_overrides

    from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync

    stage_ids = sorted(_ar_stage_devices())
    derived = sorted(OmniBagelCoRLTrainerSync._und_ar_stage_ids(str(_AR_DEPLOY)))
    assert derived == stage_ids, (
        f"trainer derives stage ids {derived} from {_AR_DEPLOY.name} but the yaml declares {stage_ids}"
    )

    # Reproduce what the AR server receives: a plain (replica-width) TP plus the
    # stage-scoped pins the trainer emits.
    cli_overrides = {"tensor_parallel_size": len(stage_ids)}
    for stage_id in stage_ids:
        cli_overrides.update({f"stage_{stage_id}_tensor_parallel_size": 1})
    for stage_id in stage_ids:
        resolved = build_stage_runtime_overrides(stage_id, dict(cli_overrides))
        assert resolved["tensor_parallel_size"] == 1, (
            f"stage {stage_id} would inherit tensor_parallel_size={resolved['tensor_parallel_size']} "
            f"from the replica-width declaration while holding one card"
        )

    # Guard the mechanism this pin exists for: if a plain key ever stops reaching a
    # stage, the per-stage pins are vestigial and this note (and the trainer comment)
    # should be re-derived from a fresh check.
    unpinned = build_stage_runtime_overrides(stage_ids[0], {"tensor_parallel_size": len(stage_ids)})
    assert unpinned["tensor_parallel_size"] == len(stage_ids), (
        "vLLM-Omni no longer propagates a plain tensor_parallel_size into stage args; "
        "re-derive the UND AR width declaration before trusting the per-stage pins"
    )


def test_und_pool_width_must_cover_the_ar_stage_devices():
    """``und_n_gpus`` has to reach the deploy config's widest logical stage device.

    Stage ``devices`` are indices into the replica's own ``CUDA_VISIBLE_DEVICES`` (the
    ``und_n_gpus``-wide slice), so a 1-wide pool cannot resolve ``devices: "1"``. The
    05:06 run failed as ``StageEngineCoreProc_stage0_replica0 ... ValueError: No
    available memory for the cache blocks`` -> ``Orchestrator initialization failed``
    with ``und_n_gpus=1`` against this very yaml -- the check exists so that mistake
    reports itself instead.
    """
    from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync

    indices = OmniBagelCoRLTrainerSync._und_stage_device_indices(str(_AR_DEPLOY))
    declared = {stage: int(device) for stage, device in _ar_stage_devices().items()}
    assert indices == declared, (
        f"trainer reads stage devices {indices} from {_AR_DEPLOY.name} but the yaml declares {declared}"
    )
    needed = 1 + max(indices.values())
    for recipe in _RECIPES:
        und_n_gpus = int(_default_after_env(_recipe_text(recipe), "UND_N_GPUS"))
        assert und_n_gpus >= needed, (
            f"{recipe.name}: UND_N_GPUS={und_n_gpus} < {needed}, the width {_AR_DEPLOY.name} needs; "
            "the trainer raises before launching, but fix the recipe default too"
        )


@pytest.mark.parametrize("recipe", _RECIPES, ids=lambda p: p.name)
def test_und_budget_stays_above_the_engine_init_floor(recipe):
    """Trimming the AR budget to buy rank-0 headroom cannot go below vLLM's own check.

    ``_init_und`` aborts with "No available memory for the cache blocks" once
    ``util * total`` drops under weights + profiling peak, and that peak is
    independent of the budget -- so this fails at startup on an empty card, before
    any colocation pressure exists. It also cannot buy GEN wake-up headroom.
    """
    und_util = _default_after_env(_recipe_text(recipe), "UND_GPU_MEM_UTIL")
    assert und_util >= _UND_UTIL_FLOOR, (
        f"{recipe.name}: UND_GPU_MEM_UTIL={und_util} is below the {_UND_UTIL_FLOOR} floor; "
        "0.30 and 0.35 aborted _init_und with 'No available memory for the cache blocks' "
        "(non-KV is ~31.1GiB of a 79.66GiB card)"
    )


@pytest.mark.parametrize("recipe", _RECIPES, ids=lambda p: p.name)
def test_split_layout_fits_the_engine_wake_up_on_every_card(recipe):
    """The card that carries GEN rank 0 must also fit the wake_up re-map plus one AR stage.

    Modelled from the measured pieces above: the sleeping GEN rank leaves ~2GiB behind
    and must re-map ``_GEN_WAKE_REMAP_GIB / ROLLOUT_TP`` during the first
    ``update_weights`` (TP shards the checkpoint, so the re-map shrinks with it), the
    actor keeps ~9.84GiB resident, and the AR stage on that card reserves
    ``util * total``. Both halves matter: the layout has to fit, and TP=1 must *not* --
    that second case is the OOM this guards.
    """
    text = _recipe_text(recipe)
    und_util = _default_after_env(text, "UND_GPU_MEM_UTIL")
    (big_tp, big_yaml), (small_tp, small_yaml) = _gen_tp_and_deploy(text)
    assert big_yaml == "bagel_corl_deploy_tp2.yaml" and small_yaml == "bagel_corl_deploy.yaml"
    # The Thinker's pool reserves ``util * total``, but its weights + peak are
    # util-independent, so the reserve is at least that measured floor.
    ar_stage0_reserve = max(und_util * _CARD_GIB, _AR_STAGE0_GIB)

    def layout(gen_tp: int) -> float:
        """GEN wake_up re-map (sharded by TP) + one AR stage + the actor shard."""
        return _GEN_WAKE_REMAP_GIB / gen_tp + ar_stage0_reserve + _ACTOR_RESIDENT_GIB

    # The layout that first OOM'd -- *both* AR stages on one card on top of this -- stays
    # impossible, so the split in bagel_corl_deploy_ar.yaml is still required.
    assert layout(small_tp) + _AR_STAGE1_GIB > _CARD_GIB, (
        f"{recipe.name}: the modelled shared-card layout ({layout(small_tp) + _AR_STAGE1_GIB:.2f}GiB) "
        "no longer exceeds the card, so this guard's numbers are stale -- re-derive the "
        "constants in this module's header from a fresh run before trusting the AR split"
    )
    # The recipe's own pairing must fit, and sharding GEN must actually buy something.
    assert layout(big_tp) <= _CARD_GIB, (
        f"{recipe.name}: GEN TP={big_tp} wake_up {_GEN_WAKE_REMAP_GIB / big_tp:.2f}GiB + AR stage 0 "
        f"{ar_stage0_reserve:.2f}GiB + actor {_ACTOR_RESIDENT_GIB:.2f}GiB = {layout(big_tp):.2f}GiB "
        f"on a {_CARD_GIB:.2f}GiB card. Trim UND_GPU_MEM_UTIL toward its floor, or spread the AR"
    )
    assert layout(big_tp) < layout(small_tp), (
        f"{recipe.name}: ROLLOUT_TP={big_tp} buys no wake-up headroom over {small_tp}, so the "
        "sharded deploy yaml is paying for itself in name only"
    )


@pytest.mark.parametrize("recipe", _RECIPES, ids=lambda p: p.name)
def test_gen_deploy_devices_match_the_rollout_tp(recipe):
    """The GEN stage's device template must name exactly ``ROLLOUT_TP`` devices.

    ``split_devices_for_replicas`` splits a diffusion stage's ``devices`` by
    ``parallel_config.world_size``, and a bare ``"0"`` against TP=2 leaves the second
    rank of each replica without a card -- silently wasting it at best, and at worst
    re-creating the mis-declaration that once collapsed the UND AR replica onto GPU 0.
    """
    text = _recipe_text(recipe)
    assert '"${_PY}" - "$BAGEL_DEPLOY_CONFIG" "$ROLLOUT_TP"' in text, (
        "the recipe must read the GEN deploy yaml back and refuse a TP/template mismatch "
        "before Ray is up, not just intend to pair them"
    )
    for tp, yaml_name in _gen_tp_and_deploy(text):
        for index, devices in enumerate(_gen_stage_devices(yaml_name)):
            assert len(devices) == tp, (
                f"{recipe.name}: {yaml_name} stage index {index} names {devices} "
                f"({len(devices)} device(s)) but pairs with ROLLOUT_TP={tp}"
            )


def test_the_two_gen_deploys_differ_only_in_devices():
    """Keep the TP=2 yaml a copy of the TP=1 one apart from its device template.

    They must agree on pipeline, max lengths and sampling, or the two pool sizes stop
    being the same experiment at a different width.
    """
    base = yaml.safe_load((_RECIPE_DIR / "bagel_corl_deploy.yaml").read_text(encoding="utf-8"))
    tp2 = yaml.safe_load((_RECIPE_DIR / "bagel_corl_deploy_tp2.yaml").read_text(encoding="utf-8"))
    assert base.keys() == tp2.keys()
    assert len(base["stages"]) == len(tp2["stages"]) == 1
    for stage in (base["stages"][0], tp2["stages"][0]):
        stage.pop("devices")
    assert base == tp2, "the TP=1 and TP=2 GEN deploy yamls drifted apart beyond ``devices``"


def test_the_oom_run_layout_is_what_the_sharded_recipe_fixes():
    """Pin the 2026-09-17 numbers that motivated sharding GEN.

    Re-derived from the allocator's own OOM report rather than from the modelled
    constants, so this fails if the record and the model diverge.
    """
    measured = _OOM_RUN_ACTOR_GIB + _OOM_RUN_FRONTENDS_GIB + _OOM_RUN_GEN_GIB + _OOM_RUN_AR_STAGE0_GIB
    assert measured == pytest.approx(78.97, abs=0.05), (
        f"the recorded OOM run adds up to {measured:.2f}GiB, not the ~78.97GiB of the 79.11GiB card"
    )
    # The failing allocation was a single base-weight staging copy, and it missed by only
    # ~74MiB -- the card was full, not fragmented.
    assert _OOM_RUN_ALLOC_GIB - _OOM_RUN_FREE_GIB == pytest.approx(0.073, abs=0.01)
    # Halving the GEN checkpoint is what makes room for it.
    assert measured - _OOM_RUN_GEN_GIB / 2 <= _CARD_GIB
