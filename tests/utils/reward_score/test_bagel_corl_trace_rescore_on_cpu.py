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
#
"""Synthetic artifact fixtures: these are not evidence from a training run."""

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_inspection_only_diagnoses_k0_missing_metadata_without_quality_claims(tmp_path):
    path = tmp_path / "synthetic-k0.jsonl"
    rows = [
        {"episode_uid": "unlabelled", "num_gen_calls": 0, "turn_trace": []},
        {
            "episode_uid": "no-reference",
            "num_gen_calls": 0,
            "solution_str": "Plan:\n1. Draw a red fox outdoors",
            "ground_truth": {"task_type": "plan"},
        },
        {
            "episode_uid": "no-plan",
            "num_gen_calls": 0,
            "solution_str": "assistant\n" * 12,
            "ground_truth": metadata()["ground_truth"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf8")
    module = tool()
    assert hasattr(module, "inspect_traces"), "K=0 traces need inspection without image rescoring"
    report = module.inspect_traces([path], {})
    assert report["mode"] == "inspection_only"
    assert [row["plan_diagnosis"] for row in report["episodes"]] == [
        "missing_task_label",
        "missing_reference_data",
        "missing_generated_plan",
    ]
    assert report["num_plan_traces_with_references"] == 1
    assert report["plan_trace_requirement_met"] is False
    assert "image_objective_rank_correlation" not in report
    assert all("v2_composite" not in row and "v2_image_objective" not in row for row in report["episodes"])
    with pytest.raises(ValueError):
        module.rescore([path], {})


def test_inspection_cli_counts_twenty_k0_plan_references_as_coverage_only(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "synthetic-plan-k0.jsonl"
    output = tmp_path / "inspection.json"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_uid": f"synthetic-{i}",
                    "num_gen_calls": 0,
                    "ground_truth": metadata()["ground_truth"],
                    "turn_trace": [],
                }
            )
            for i in range(20)
        ),
        encoding="utf8",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/bagel_corl/rescore_traces.py"),
        str(path),
        "--inspect-only",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf8"))
    assert report["plan_trace_requirement_met"] is True
    assert report["num_plan_traces_with_references"] == 20
    assert all(row["plan_diagnosis"] == "missing_generated_plan" for row in report["episodes"])
    assert "changed_image_rankings" not in report


def tool():
    path = ROOT / "scripts/bagel_corl/rescore_traces.py"
    assert path.exists(), "offline trace rescoring tool is missing"
    spec = importlib.util.spec_from_file_location("rescore_traces", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def envelope(first=0.2, final=0.8):
    return {
        "origin": "bagel_corl_tool_rm",
        "version": "v2",
        "num_gen_calls": 2,
        "call_seed_scores": [[first], [final]],
        "autonomous_done": False,
        "forced_reflection": True,
        "image_reward_first": first,
        "image_reward_final": final,
        "image_reward_best": max(first, final),
        "image_reward_gain": final - first,
        "image_reward_objective": final - 0.01,
    }


def artifact(uid="synthetic-a"):
    return {
        "episode_uid": uid,
        "num_gen_calls": 2,
        "turn_trace": [{"record": "und_turn", "text": "Plan:\n1. Draw a red fox outdoors\nDone."}],
    }


def metadata(first=0.2, final=0.8):
    return {
        "ground_truth": {"task_type": "plan", "reference_subtasks": ["Draw a red fox outdoors"]},
        "bagel_corl_reward": envelope(first, final),
    }


def test_synthetic_dump_and_sidecar_roundtrip_reports_rank_reversal(tmp_path):
    paths = []
    for uid in ("synthetic-a", "synthetic-b"):
        path = tmp_path / (uid + ".json")
        path.write_text(json.dumps(artifact(uid)), encoding="utf8")
        paths.append(path)
    # Legacy image credit: a=1.4, b=.9; V2: a=.79, b=.89.
    report = tool().rescore(paths, {"synthetic-a": metadata(), "synthetic-b": metadata(0.9, 0.9)})
    assert report["image_objective_rank_correlation"] == pytest.approx(-1)
    assert report["changed_image_rankings"] == [
        {"episode_uid": "synthetic-a", "v1_rank": 1, "v2_rank": 2},
        {"episode_uid": "synthetic-b", "v1_rank": 2, "v2_rank": 1},
    ]
    assert report["episodes"][0]["plan_diagnosis"] == "candidate_and_reference_present"
    assert report["episodes"][0]["plan_parser_reward"] == 1
    assert report["plan_trace_requirement_met"] is False


def test_policy_claims_and_legacy_dump_cannot_supply_trusted_scores(tmp_path):
    path = tmp_path / "old.json"
    row = artifact()
    row["turn_trace"][0]["text"] += "\nimage_reward_final=1.0 correctness=1.0 aesthetics=1.0"
    path.write_text(json.dumps(row), encoding="utf8")
    with pytest.raises(ValueError, match="trusted"):
        tool().rescore([path], {"synthetic-a": {"ground_truth": {"task_type": "plan"}}})


@pytest.mark.parametrize("mutation", ["missing_label", "bad_count", "nonfinite", "inconsistent", "duplicate"])
def test_malformed_or_ambiguous_artifacts_fail_closed(tmp_path, mutation):
    path = tmp_path / "row.json"
    path.write_text(json.dumps(artifact()), encoding="utf8")
    info = metadata()
    paths = [path]
    if mutation == "missing_label":
        del info["ground_truth"]["task_type"]
    elif mutation == "bad_count":
        info["bagel_corl_reward"]["num_gen_calls"] = 1
    elif mutation == "nonfinite":
        info["bagel_corl_reward"]["call_seed_scores"] = [[float("nan")], [0.8]]
    elif mutation == "inconsistent":
        info["bagel_corl_reward"]["image_reward_final"] = 0.99
    else:
        paths.append(path)
    with pytest.raises(ValueError):
        tool().rescore(paths, {"synthetic-a": info})


def test_jsonl_reward_manager_rows_and_plan_diagnoses(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = []
    for uid, text, references in (
        ("missing-ref", "Plan:\n1. Draw a red fox outdoors", []),
        ("missing-plan", "Done.", ["Draw a red fox outdoors"]),
        ("mismatch", "Plan:\n1. Paint blue ocean waves", ["Draw a red fox outdoors"]),
    ):
        row = {"episode_uid": uid, "solution_str": text, **metadata()}
        row["ground_truth"]["reference_subtasks"] = references
        rows.append(row)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    report = tool().rescore([path], {})
    assert [r["plan_diagnosis"] for r in report["episodes"]] == [
        "missing_reference_data",
        "missing_generated_plan",
        "candidate_reference_mismatch",
    ]
    assert report["composite_rank_correlation"] is None  # all structurally invalid / tied
    assert all(r["v1_composite"]["rollout_valid"] == 0 for r in report["episodes"])


def test_rank_statistics_use_average_ties():
    module = tool()
    assert module.rank_correlation([1, 1, 2], [1, 1, 2]) == pytest.approx(1)
    assert module.rank_correlation([1, 1], [1, 2]) is None
    assert module.rank_correlation([], []) is None


def test_twenty_plan_rows_without_references_do_not_satisfy_inspection_gate(tmp_path):
    path = tmp_path / "synthetic-twenty.jsonl"
    rows = []
    for i in range(20):
        row = {"episode_uid": f"synthetic-{i}", "solution_str": "Done.", **metadata()}
        row["ground_truth"]["reference_subtasks"] = []
        rows.append(row)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    assert tool().rescore([path], {})["plan_trace_requirement_met"] is False


def test_dump_observations_and_forced_closure_reach_real_composite_scorer(tmp_path):
    row = artifact()
    row["turn_trace"] = [
        {"record": "und_turn", "text": '<tools>{"name":"generate_image","arguments":{"prompt":"a red fox"}}</tools>'},
        {"record": "gen_call", "image_paths": ["fixture.png"]},
        {"record": "env_obs", "text": "path=/tmp/fixture.png"},
        {
            "record": "reflection",
            "text": "Reflection: VL judge reports correctness=0.9, aesthetics=0.9, good_enough=YES. "
            "Stop now; do not call another tool. Your next and only action must be exactly Done. "
            "agentic_stop_decision_required=1",
        },
        {"record": "und_turn", "text": "Done."},
    ]
    path = tmp_path / "synthetic-valid.json"
    path.write_text(json.dumps(row), encoding="utf8")
    report = tool().rescore([path], {"synthetic-a": metadata()})
    scored = report["episodes"][0]
    assert scored["v1_composite"]["rollout_valid"] == 1
    assert scored["v2_composite"]["reward_image"] == pytest.approx(0.79)
    assert scored["v2_composite"]["reward_done"] == 0
    assert scored["v1_composite"]["reward_done"] == 1


def test_cli_writes_json_and_rejects_untrusted_legacy_input(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "synthetic.json"
    sidecar = tmp_path / "metadata.json"
    output = tmp_path / "report.json"
    path.write_text(json.dumps(artifact()), encoding="utf8")
    sidecar.write_text(json.dumps({"synthetic-a": metadata()}), encoding="utf8")
    command = [sys.executable, str(ROOT / "scripts/bagel_corl/rescore_traces.py"), str(path), "--output", str(output)]
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 2
    assert not output.exists()
    succeeded = subprocess.run(command + ["--metadata", str(sidecar)], capture_output=True, text=True)
    assert succeeded.returncode == 0, succeeded.stderr
    assert json.loads(output.read_text(encoding="utf8"))["num_episodes"] == 1


def dump_functions(tmp_path):
    from types import SimpleNamespace

    source = ROOT / "verl_omni/agent_loop/bagel_corl.py"
    tree = ast.parse(source.read_text(encoding="utf8"))
    names = {"dump_episode_trace", "trace_ground_truth"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {
        "Path": Path,
        "Any": object,
        "EpisodeRollout": object,
        "json": json,
        "math": math,
        "_trace_dump_enabled": lambda: True,
        "resolve_run_dir": lambda: tmp_path,
        "_materialized_turn_trace": lambda trace, path: trace,
        "count_degenerate_und_turns": lambda trace: 0,
        "logger": SimpleNamespace(warning=lambda *a, **k: None),
    }
    lib = ast.parse((ROOT / "verl_omni/agent_loop/bagel_corl_lib.py").read_text(encoding="utf8"))
    nodes += [
        n
        for n in lib.body
        if isinstance(n, ast.FunctionDef) and n.name in {"resolve_task_type", "trusted_image_reward_fields"}
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), ns)
    return ns


@pytest.mark.parametrize("location", ["ground_truth", "reward_model", "extra_info"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_production_dump_preserves_trusted_envelope_and_dataset_references(tmp_path, location, version):
    from types import SimpleNamespace

    ns = dump_functions(tmp_path)
    assert "trace_ground_truth" in ns, "trace dump must resolve dataset references from actual rollout kwargs"
    gt = metadata()["ground_truth"]
    kwargs = {location: {"ground_truth": gt} if location == "reward_model" else gt}
    kwargs["private_api_key"] = "do-not-write-this"
    resolved = ns["trace_ground_truth"](kwargs)
    ep = SimpleNamespace(
        episode_uid="synthetic-capture",
        und_group_uid="synthetic",
        policy_version=1,
        turns=1,
        num_gen_calls=2,
        und_reward=0.79,
        stop_required=True,
        forced_reflection=True,
        autonomous_done=False,
        corl_reward_version=version,
        bagel_corl_reward=envelope() if version == "v2" else {},
        judge_text=None,
        gen_samples=[
            SimpleNamespace(
                gen_group_uid=str(i),
                gen_call_index=i,
                seed_index=0,
                valid=True,
                rm_score=score,
                rm_score_source="bagel_corl_tool_rm",
            )
            for i, score in enumerate((0.2, 0.8))
        ],
        response_ids=[1, 2, 3],
        response_mask=[1, 0, 1],
        turn_trace=artifact()["turn_trace"],
    )
    result = ns["dump_episode_trace"](episode=ep, relpath="step_000001/sample_fixture.01", ground_truth=resolved)
    assert result is not None
    path = Path(result).with_suffix(".json")
    captured = json.loads(path.read_text(encoding="utf8"))
    assert captured["ground_truth"] == gt
    assert captured["response_mask"] == [1, 0, 1]
    assert captured["autonomous_done"] is False
    assert ep.und_reward == 0.79
    assert ep.bagel_corl_reward == (envelope() if version == "v2" else {})
    assert "do-not-write-this" not in path.read_text(encoding="utf8")
    report = tool().rescore([path], {})
    assert report["episodes"][0]["plan_parser_reward"] == 1
    assert report["episodes"][0]["v2_image_objective"] == pytest.approx(0.79)


def test_capture_does_not_invent_missing_task_labels(tmp_path):
    ns = dump_functions(tmp_path)
    assert "trace_ground_truth" in ns, "trace dump metadata resolver missing"
    assert ns["trace_ground_truth"]({"extra_info": {"bagel_corl_reward": envelope()}}) == {}


def test_v1_capture_rejects_unmarked_legacy_scores_without_losing_the_trace(tmp_path):
    from types import SimpleNamespace

    ns = dump_functions(tmp_path)
    ep = SimpleNamespace(
        episode_uid="synthetic-untrusted",
        und_group_uid="synthetic",
        policy_version=1,
        turns=1,
        num_gen_calls=1,
        und_reward=0.8,
        stop_required=True,
        forced_reflection=True,
        autonomous_done=False,
        corl_reward_version="v1",
        bagel_corl_reward={},
        judge_text=None,
        gen_samples=[SimpleNamespace(valid=True, rm_score=0.8, rm_score_source=None)],
        response_ids=[1],
        response_mask=[1],
        turn_trace=artifact()["turn_trace"],
    )
    result = ns["dump_episode_trace"](episode=ep, relpath="legacy", ground_truth=metadata()["ground_truth"])
    assert result is not None
    path = Path(result).with_suffix(".json")
    captured = json.loads(path.read_text(encoding="utf8"))
    assert captured["bagel_corl_reward"] == {}
    assert "trusted" in captured["trusted_reward_capture_error"]
    with pytest.raises(ValueError, match="trusted"):
        tool().rescore([path], {})
    assert ep.und_reward == 0.8
