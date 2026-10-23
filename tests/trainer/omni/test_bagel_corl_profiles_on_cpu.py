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

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_explicit_validation_panel_reaches_recipe_environment(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "profiles_val_forward", ROOT / "examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env, _ = module.build_profile(
        "B0",
        11,
        {
            "START_CHECKPOINT": "/model",
            "UNICOT_TRAIN": "/fixed/train.parquet",
            "UNICOT_VAL": "/fixed/panel.parquet",
            "EXPERIMENT_PREFIX": "validation",
            "EXPERIMENT_ROOT": str(tmp_path),
        },
    )
    assert env["UNICOT_TEST"] == "/fixed/panel.parquet"
    assert env["REBUILD_UNICOT"] == "0"


def load():
    spec = importlib.util.spec_from_file_location(
        "profiles", ROOT / "examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ["reproduction", "B0", "B1", "B2-1e-6", "B2-1e-5", "B3", "B4"])
def test_profiles_require_explicit_provenance_and_isolation(profile):
    tool = load()
    with pytest.raises(ValueError):
        tool.build_profile(profile, 11, {})
    env, overrides = tool.build_profile(
        profile,
        11,
        {
            "START_CHECKPOINT": "/model/base",
            "UNICOT_TRAIN": "/data/train.parquet",
            "UNICOT_VAL": "/data/panel.parquet",
            "EXPERIMENT_PREFIX": "test",
        },
    )
    assert env["RESUME_MODE"] == "disable"
    assert env["NOISE_LEVEL"] == "0.8" and env["SDE_WINDOW_RANGE"] == "[0,2]"
    assert env["REBUILD_UNICOT"] == "0"
    assert env["DEFAULT_LOCAL_DIR"].endswith(f"test_{profile}_seed11")
    assert "data.seed=11" in overrides
    assert "actor_rollout_ref.actor.data_loader_seed=11" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.seed=11" in overrides
    assert "actor_rollout_ref.rollout.seed=11" in overrides
    assert env["CFG_TEXT_SCALE"] == "4.0" and env["CFG_IMG_SCALE"] == "1.5"
    if profile == "B0":
        assert "+actor_rollout_ref.model.corl_no_optimizer_step=True" in overrides
    if profile == "B1":
        assert "+actor_rollout_ref.model.corl_freeze_und=True" in overrides
        assert "+actor_rollout_ref.rollout.agent.corl_fixed_gen_prompt=__user_request__" in overrides
        assert env["S"] == "2" and env["MAX_GENERATE_PASSES"] == "1"
    if profile.startswith("B2"):
        assert "+actor_rollout_ref.model.corl_freeze_gen=True" in overrides
    if profile == "B4":
        assert "++agentic_image_gen.corl_reward_version=v2" in overrides
        assert env["MAX_GENERATE_PASSES"] == "3"


def test_dry_run_cli_does_not_launch_or_write(tmp_path):
    env = dict(
        os.environ,
        START_CHECKPOINT="/model/base",
        UNICOT_TRAIN="/data/train.parquet",
        UNICOT_VAL="/data/panel.parquet",
        EXPERIMENT_PREFIX="dry",
        EXPERIMENT_ROOT=str(tmp_path),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py"),
            "B0",
            "--seed",
            "11",
            "--dry-run",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    assert "corl_no_optimizer_step=True" in value["command"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("profile", ["reproduction", "B0", "B1", "B2-1e-6", "B2-1e-5", "B3", "B4"])
def test_profiles_choose_explicit_logprob_convention(profile):
    env, _ = load().build_profile(
        profile,
        11,
        {
            "START_CHECKPOINT": "/model",
            "UNICOT_TRAIN": "/train",
            "UNICOT_VAL": "/val",
            "EXPERIMENT_PREFIX": "logp",
            "LOGPROBS_MODE": "inherited_wrong_mode",
        },
    )
    assert env["LOGPROBS_MODE"] == ("processed_logprobs" if profile == "reproduction" else "raw_logprobs")


@pytest.mark.parametrize("mode", [None, "raw_logprobs", "processed_logprobs"])
def test_recipe_emits_actual_logprob_and_unchanged_sampling_arguments(mode):
    import shutil

    bash = shutil.which("bash")
    if not bash:
        git = shutil.which("git")
        candidate = Path(git).parent.parent / "bin/bash.exe" if git else Path("/nonexistent")
        if candidate.exists():
            bash = str(candidate)
    if not bash:
        pytest.skip("Bash is required to execute the recipe command fixture")
    source = (ROOT / "examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh").read_text(encoding="utf-8")
    assignments = "\n".join(
        line
        for line in source.splitlines()
        if line.startswith(
            (
                "LOGPROBS_MODE=",
                "ROLLOUT_TEMPERATURE=",
                "ROLLOUT_TOP_P=",
                "ROLLOUT_TOP_K=",
                "ROLLOUT_REPETITION_PENALTY=",
            )
        )
    )
    command = source[source.index('"${_PY}" -m verl_omni.trainer.main_omni') :]
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"LOGPROBS_MODE", "ROLLOUT_TEMPERATURE", "ROLLOUT_TOP_P", "ROLLOUT_TOP_K", "ROLLOUT_REPETITION_PENALTY"}
    }
    if mode:
        env["LOGPROBS_MODE"] = mode
    script = 'capture() { printf "%s\\n" "$@"; }; _PY=capture\n' + assignments + "\n" + command
    result = subprocess.run([bash, "-c", script], env=env, capture_output=True, text=True, check=True)
    argv = result.stdout.splitlines()
    assert f"actor_rollout_ref.rollout.logprobs_mode={mode or 'raw_logprobs'}" in argv
    assert "actor_rollout_ref.rollout.temperature=0.7" in argv
    assert "actor_rollout_ref.rollout.top_p=0.9" in argv
    assert "actor_rollout_ref.rollout.top_k=50" in argv
    assert "+actor_rollout_ref.rollout.repetition_penalty=1.05" in argv


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_und_builder_inherits_requested_logprob_mode(mode):
    import ast
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    path = ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniBagelCoRLTrainerSync")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_und_ar_entrypoint_config")
    node.decorator_list = []
    namespace = {"_actor_merges_lora": lambda model: True}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "data": {"max_prompt_length": 32, "max_response_length": 32},
                "actor_rollout_ref": {
                    "model": {"path": "/model"},
                    "rollout": {"logprobs_mode": mode, "agent": {"und_n_gpus": 1}},
                },
            }
        ),
        _und_deploy_config_path=lambda: "/fixture/deploy.yaml",
        _resolve_und_hf_config_path=lambda model, agent: None,
        _und_stage_max_model_len=lambda deploy: 128,
        _und_ar_stage_ids=lambda deploy: [0, 1],
        UND_AR_LOAD_FORMAT="auto",
        UND_AR_POLICY_STAGE_ID=0,
    )
    result = namespace["_build_und_ar_entrypoint_config"](trainer)
    assert result.actor_rollout_ref.rollout.get("logprobs_mode") == mode


@pytest.mark.parametrize("profile,mode", [("B0", None), ("B3", "processed_logprobs"), ("reproduction", "raw_logprobs")])
def test_profile_manifest_rejects_missing_or_conflicting_resolved_mode(profile, mode):
    config = {"actor_rollout_ref": {"rollout": {}}}
    if mode:
        config["actor_rollout_ref"]["rollout"]["logprobs_mode"] = mode
    tool = load()
    assert hasattr(tool, "validate_profile_logprobs"), "Missing resolved profile logprob convention validation"
    with pytest.raises(ValueError, match="logprobs_mode"):
        tool.validate_profile_logprobs(profile, config)


@pytest.mark.parametrize("enabled", [False, True])
def test_capture_settings_are_explicitly_forwarded_to_ray_only_when_set(enabled):
    inherited = dict(START_CHECKPOINT="/model", UNICOT_TRAIN="/train", UNICOT_VAL="/val", EXPERIMENT_PREFIX="capture")
    capture_env = dict(
        BAGEL_CORL_PARITY_CAPTURE_DIR="/runs/capture rows",
        BAGEL_CORL_PARITY_MANIFEST="/runs/verified manifest.json",
        BAGEL_CORL_PARITY_REPLICA_LABEL="isolated-server-pair",
    )
    if enabled:
        inherited.update(capture_env)
    env, overrides = load().build_profile("B0", 11, inherited)
    prefix = "++ray_kwargs.ray_init.runtime_env.env_vars."
    for key, value in capture_env.items():
        matches = [item for item in overrides if item.startswith(prefix + key + "=")]
        assert len(matches) == int(enabled)
        if enabled:
            assert env[key] == value
            assert json.loads(matches[0].split("=", 1)[1]) == value
