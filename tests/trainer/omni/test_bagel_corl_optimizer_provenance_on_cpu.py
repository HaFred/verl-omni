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

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_runtime():
    path = ROOT / "verl_omni/workers/bagel_corl_provenance.py"
    assert path.exists(), "Runtime optimizer snapshot helper is missing"
    spec = importlib.util.spec_from_file_location("runtime_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine():
    und, gen = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [und], "name": "und_lora", "lr": 3e-5},
            {"params": [gen], "name": "gen_lora", "lr": 1e-5},
        ]
    )
    return SimpleNamespace(optimizer=optimizer)


def test_snapshot_reads_restored_actual_lr_and_excludes_optimizer_secrets():
    tool = load_runtime()
    actor = engine()
    actor.optimizer.param_groups[0]["password"] = "DO_NOT_EXPORT"
    initial = tool.optimizer_snapshot(actor, phase="initialized")
    state = actor.optimizer.state_dict()
    state["param_groups"][0]["lr"] = 7e-7
    state["param_groups"][1]["lr"] = 2e-6
    actor.optimizer.load_state_dict(state)
    resumed = tool.optimizer_snapshot(actor, phase="restored", checkpoint_path="/private/global_step_25/actor")
    assert initial["optimizer_group_lrs"] == {"und": 3e-5, "gen": 1e-5}
    assert resumed["optimizer_group_lrs"] == {"und": 7e-7, "gen": 2e-6}
    assert resumed["checkpoint_step"] == 25
    assert "DO_NOT_EXPORT" not in json.dumps(resumed)
    assert "/private" not in json.dumps(resumed)


def test_frozen_lane_absent_and_invalid_actual_groups_fail():
    tool = load_runtime()
    actor = engine()
    actor.optimizer.param_groups.pop()
    assert tool.optimizer_snapshot(actor, phase="initialized")["optimizer_group_lrs"] == {"und": 3e-5, "gen": None}
    actor.optimizer.param_groups[0]["lr"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        tool.optimizer_snapshot(actor, phase="initialized")
    actor.optimizer.param_groups[0]["lr"] = 1e-5
    actor.optimizer.param_groups[0]["name"] = "unclassified_secret"
    with pytest.raises(ValueError, match="group"):
        tool.optimizer_snapshot(actor, phase="initialized")


def test_runtime_manifest_records_all_ranks_and_rejects_disagreement(tmp_path):
    tool = load_runtime()
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"optimizer_lrs_source": "configured_peak", "source": {"git_sha": "abc"}}))
    snapshot = tool.optimizer_snapshot(engine(), phase="initialized")
    tool.write_runtime_provenance(tmp_path, [[snapshot], [snapshot]], trainer_step=0)
    result = json.loads(manifest.read_text())
    assert result["optimizer_lrs_source"] == "runtime"
    assert result["optimizer_group_lrs"] == {"und": 3e-5, "gen": 1e-5}
    assert result["source"] == {"git_sha": "abc"}
    assert len(result["runtime_optimizer_snapshots"][-1]["workers"]) == 2
    different = tool.optimizer_snapshot(engine(), phase="initialized")
    different["optimizer_group_lrs"]["gen"] = 9e-5
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="disagree"):
        tool.write_runtime_provenance(tmp_path, [[snapshot], [different]], trainer_step=1)
    assert manifest.read_bytes() == before


def method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name),
        None,
    )
    assert node is not None, f"Missing runtime hook: {method_name}"
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


def test_checkpoint_hook_snapshots_only_after_successful_restore(monkeypatch):
    tool = load_runtime()
    monkeypatch.setenv("BAGEL_CORL_PROVENANCE_DIR", "/fixture")
    actor_engine = engine()

    class Actor:
        engine = actor_engine

        def load_checkpoint(self, *args):
            self.engine.optimizer.param_groups[0]["lr"] = 8e-7

    worker = SimpleNamespace(
        role="actor",
        actor=Actor(),
        config=SimpleNamespace(model={"composite_mode": "bagel_corl"}),
    )
    capture = method(
        "verl_omni/workers/engine_workers.py",
        "ActorRolloutRefWorker",
        "_capture_bagel_optimizer_snapshot",
        {"os": __import__("os"), "optimizer_snapshot": tool.optimizer_snapshot},
    )
    worker._capture_bagel_optimizer_snapshot = lambda **kwargs: capture(worker, **kwargs)
    restore = method(
        "verl_omni/workers/engine_workers.py",
        "ActorRolloutRefWorker",
        "load_checkpoint",
        {},
    )
    restore(worker, "/private/global_step_41/actor")
    assert worker._bagel_optimizer_snapshots[-1]["optimizer_group_lrs"]["und"] == 8e-7
    assert worker._bagel_optimizer_snapshots[-1]["checkpoint_step"] == 41
    worker.actor.load_checkpoint = lambda *args: (_ for _ in ()).throw(RuntimeError("failed load"))
    with pytest.raises(RuntimeError):
        restore(worker, "/private/global_step_42/actor")
    assert len(worker._bagel_optimizer_snapshots) == 1


def test_trainer_records_blocking_rpc_results_after_resume(tmp_path, monkeypatch):
    tool = load_runtime()
    monkeypatch.setenv("BAGEL_CORL_PROVENANCE_DIR", str(tmp_path))
    (tmp_path / "run_manifest.json").write_text("{}")
    actor = engine()
    initialized = tool.optimizer_snapshot(actor, phase="initialized")
    actor.optimizer.param_groups[0]["lr"] = 6e-7
    restored = tool.optimizer_snapshot(actor, phase="restored", checkpoint_path="/private/global_step_25/actor")
    # ONE_TO_ALL defaults to blocking=True: the worker group returns values,
    # not ObjectRefs, so wrapping this list in ray.get is a runtime error.
    wg = SimpleNamespace(
        get_bagel_optimizer_snapshots=lambda: [
            [initialized, restored],
            [initialized, restored],
        ]
    )
    trainer = SimpleNamespace(actor_rollout_wg=wg, global_steps=26)
    record = method(
        "verl_omni/trainer/omni/bagel_corl_trainer.py",
        "OmniBagelCoRLTrainerSync",
        "_record_runtime_optimizer_provenance",
        {
            "os": __import__("os"),
            "write_runtime_provenance": tool.write_runtime_provenance,
        },
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "ray",
        SimpleNamespace(get=lambda values: (_ for _ in ()).throw(ValueError("blocking RPC already returned values"))),
    )
    record(trainer)
    result = json.loads((tmp_path / "run_manifest.json").read_text())
    assert result["optimizer_group_lrs"]["und"] == 6e-7
    assert result["runtime_optimizer_snapshots"][0]["trainer_step"] == 26


def test_worker_snapshot_opt_in_does_not_touch_other_pipelines(monkeypatch):
    tool = load_runtime()
    capture = method(
        "verl_omni/workers/engine_workers.py",
        "ActorRolloutRefWorker",
        "_capture_bagel_optimizer_snapshot",
        {"os": __import__("os"), "optimizer_snapshot": tool.optimizer_snapshot},
    )
    worker = SimpleNamespace(config=SimpleNamespace(model={"composite_mode": "bagel_corl"}))
    monkeypatch.delenv("BAGEL_CORL_PROVENANCE_DIR", raising=False)
    capture(worker, phase="initialized")
    monkeypatch.setenv("BAGEL_CORL_PROVENANCE_DIR", "/fixture")
    worker.config.model["composite_mode"] = None
    capture(worker, phase="initialized")
    assert not hasattr(worker, "_bagel_optimizer_snapshots")


def test_trainer_init_collects_before_rollout_publication():
    events = []

    class Parent:
        def on_init_end(self):
            events.append("parent")

    tree = ast.parse((ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniBagelCoRLTrainerSync")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "on_init_end")
    cls.body = [node]
    cls.decorator_list = []
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    namespace = {"Parent": Parent}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "<trainer-init>",
            "exec",
        ),
        namespace,
    )
    trainer = namespace["OmniBagelCoRLTrainerSync"]()
    trainer.global_steps = 0
    trainer._record_runtime_optimizer_provenance = lambda: events.append("provenance")
    trainer._ensure_dual_role_rollout = lambda: events.append("rollout")
    trainer._bagel_rm_enabled = lambda: False
    trainer._publish_und_ar_weights = lambda: events.append("publish")
    trainer._wake_rollout_replicas = lambda: events.append("wake")
    trainer.on_init_end()
    assert events == ["provenance", "rollout", "publish", "parent", "wake"]


def test_profiles_propagate_provenance_to_ray_workers(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "profiles",
        ROOT / "examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py",
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    env, overrides = tool.build_profile(
        "B0",
        11,
        {
            "START_CHECKPOINT": "/model",
            "UNICOT_TRAIN": "/train",
            "UNICOT_VAL": "/val",
            "EXPERIMENT_PREFIX": "runtime",
            "EXPERIMENT_ROOT": str(tmp_path),
        },
    )
    assert Path(env["BAGEL_CORL_PROVENANCE_DIR"]) == tmp_path / "runtime_B0_seed11" / "provenance"
    override = next(item for item in overrides if "env_vars.BAGEL_CORL_PROVENANCE_DIR=" in item)
    assert json.loads(override.split("=", 1)[1]) == env["BAGEL_CORL_PROVENANCE_DIR"]
