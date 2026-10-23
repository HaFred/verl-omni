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
"""Offline V1/V2 comparison; never obtain trusted image scores from policy text.

Accepts dump_episode_trace JSON (turn_trace), or JSON/JSONL reward-manager
exports (solution_str, ground_truth, extra_info). Legacy dumps omit ground truth
and trusted scores: supply --metadata JSON keyed by episode_uid, each value
containing ground_truth and bagel_corl_reward from the harness. The envelope is
the exact trusted_image_reward_fields output; its provenance must be verified by
the operator. An origin string is a schema marker, not a cryptographic signature.

V1 composite reproduces the legacy transcript parser for comparison ONLY. Its
parsed numeric claims never become V2 trusted data. Image-objective rankings use
the same independently stored per-call/per-seed scores for both versions. This
does not recover old UnifiedReward/rubric disagreement or rejudge images.

Example:
  python scripts/bagel_corl/rescore_traces.py run/rollout_trajectories \
      --metadata harness_metadata.json --output rescore.json

Use --inspect-only to diagnose planning metadata/text even for K=0 episodes
and legacy dumps lacking trusted scores or task labels. Inspection coverage is
not image-quality evidence and does not imply that a parser fix is warranted.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _scorer():
    # Self-contained reward module; avoid importing GPU/serving package parents.
    path = ROOT / "verl_omni/utils/reward_score/agentic_multidim_reward.py"
    spec = importlib.util.spec_from_file_location("offline_agentic_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ranks(values):
    """Descending average ranks, including ties."""
    return [1 + sum(v > value for v in values) + (sum(v == value for v in values) - 1) / 2 for value in values]


def rank_correlation(left, right):
    """Spearman correlation with average ties; null for undefined variance."""
    if len(left) != len(right):
        raise ValueError("rank inputs must have equal length")
    if len(left) < 2:
        return None
    a, b = ranks(left), ranks(right)
    a = [v - sum(a) / len(a) for v in a]
    b = [v - sum(b) / len(b) for v in b]
    denominator = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / denominator if denominator else None


def _rows(path):
    if path.suffix == ".jsonl":
        return [
            (f"{path}:{i}", json.loads(line))
            for i, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1)
            if line.strip()
        ]
    data = json.loads(path.read_text(encoding="utf8"))
    return [(str(path), data)] if isinstance(data, dict) else [(f"{path}[{i}]", row) for i, row in enumerate(data)]


def _text(row):
    if isinstance(row.get("solution_str"), str):
        return row["solution_str"]
    trace = row.get("turn_trace")
    if not isinstance(trace, list):
        raise ValueError("expected solution_str or dump_episode_trace turn_trace")
    return "\n".join(str(r.get("text", "")) for r in trace if r.get("record") in {"und_turn", "env_obs", "reflection"})


def inspect_traces(paths, metadata):
    """Inspect stored plans without requiring images or claiming reward results."""
    scorer = _scorer()
    episodes, seen = [], set()
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for source, row in _rows(path):
            uid = row.get("episode_uid")
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError(f"missing/duplicate episode_uid: {source}")
            seen.add(uid)
            sidecar = metadata.get(uid, {})
            extra = row.get("extra_info") or {}
            raw_gt = sidecar.get(
                "ground_truth", row.get("ground_truth", (row.get("reward_model") or {}).get("ground_truth"))
            )
            gt = scorer._as_dict(raw_gt)
            task_type = gt.get("task_type", extra.get("task_type"))
            references = [
                str(item).strip() for item in scorer._as_sequence(gt.get("reference_subtasks")) if str(item).strip()
            ]
            text = scorer._normalize_trajectory_dialect(_text(row))
            candidates = scorer._extract_initial_plan_lines(text)
            coverage = scorer._plan_reward(text, gt, initial_only=True)
            legacy_candidates = scorer._extract_plan_lines(text)
            legacy_coverage = scorer._plan_reward(text, gt)
            diagnosis = (
                "missing_task_label"
                if not task_type
                else "invalid_task_label"
                if task_type not in {"plan", "reflect"}
                else "not_plan_task"
                if task_type != "plan"
                else "missing_reference_data"
                if not references
                else "missing_generated_plan"
                if not candidates
                else "candidate_reference_mismatch"
                if not coverage
                else "candidate_and_reference_present"
            )
            episodes.append(
                {
                    "episode_uid": uid,
                    "source": source,
                    "source_sha256": digest,
                    "task_type": task_type,
                    "num_gen_calls": row.get("num_gen_calls"),
                    "reference_subtasks": references,
                    "generated_plan_lines": candidates,
                    "plan_diagnosis": diagnosis,
                    "plan_parser_coverage": coverage,
                    "legacy_anywhere_plan_lines": legacy_candidates,
                    "legacy_anywhere_plan_parser_coverage": legacy_coverage,
                }
            )
    if not episodes:
        raise ValueError("no episode artifacts supplied")
    count = sum(row["task_type"] == "plan" and bool(row["reference_subtasks"]) for row in episodes)
    return {
        "mode": "inspection_only",
        "episodes": episodes,
        "num_episodes": len(episodes),
        "num_plan_traces_with_references": count,
        "plan_trace_requirement_met": count >= 20,
        "coverage_scope": "stored metadata/text inspection only; no image or composite reward comparison",
        "scorer_sha256": hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }


def rescore(paths, metadata):
    scorer = _scorer()
    episodes, seen = [], set()
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for source, row in _rows(path):
            uid = row.get("episode_uid")
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError(f"missing/duplicate episode_uid: {source}")
            seen.add(uid)
            sidecar = metadata.get(uid, {})
            extra = dict(row.get("extra_info") or {})
            gt = sidecar.get(
                "ground_truth", row.get("ground_truth", (row.get("reward_model") or {}).get("ground_truth"))
            )
            if not isinstance(gt, dict) or gt.get("task_type") not in {"plan", "reflect"}:
                raise ValueError(f"missing ground_truth task label for {uid}")
            trusted = sidecar.get("bagel_corl_reward", row.get("bagel_corl_reward", extra.get("bagel_corl_reward")))
            scorer._trusted_v2_reward({"bagel_corl_reward": trusted})
            if row.get("num_gen_calls", trusted["num_gen_calls"]) != trusted["num_gen_calls"]:
                raise ValueError(f"trace/trusted call count differs: {uid}")
            calls = trusted["call_seed_scores"]
            if not calls:
                raise ValueError(f"no trusted image scores for image ranking: {uid}")
            text = _text(row)
            extra["bagel_corl_reward"] = trusted
            results = {}
            for version in ("v1", "v2"):
                extra["corl_reward_version"] = version
                results[version] = scorer.compute_score(solution_str=text, ground_truth=gt, extra_info=extra)
            means = [sum(scores) / len(scores) for scores in calls]
            first, final = means[0], means[-1]
            references = scorer._as_sequence(gt.get("reference_subtasks"))
            candidates = scorer._extract_initial_plan_lines(text)
            plan_reward = scorer._plan_reward(text, gt, initial_only=True)
            diagnosis = "not_plan_task"
            if gt["task_type"] == "plan":
                diagnosis = (
                    "missing_reference_data"
                    if not references
                    else "missing_generated_plan"
                    if not candidates
                    else "candidate_reference_mismatch"
                    if not plan_reward
                    else "candidate_and_reference_present"
                )
            episodes.append(
                {
                    "episode_uid": uid,
                    "source": source,
                    "source_sha256": digest,
                    "task_type": gt["task_type"],
                    "reference_subtasks": references,
                    "generated_plan_lines": candidates,
                    "plan_diagnosis": diagnosis,
                    "plan_parser_reward": plan_reward,
                    "legacy_anywhere_plan_lines": scorer._extract_plan_lines(
                        scorer._normalize_trajectory_dialect(text)
                    ),
                    "legacy_anywhere_plan_parser_reward": scorer._plan_reward(
                        scorer._normalize_trajectory_dialect(text), gt
                    ),
                    "v1_image_credit": final + min(1.0, max(0.0, final - first)),
                    "v2_image_objective": trusted["image_reward_objective"],
                    "v1_composite": results["v1"],
                    "v2_composite": results["v2"],
                }
            )
    if not episodes:
        raise ValueError("no episode artifacts supplied")
    report = {
        "episodes": episodes,
        "num_episodes": len(episodes),
        "plan_trace_requirement_met": sum(r["task_type"] == "plan" and bool(r["reference_subtasks"]) for r in episodes)
        >= 20,
        "v1_composite_provenance": "legacy transcript parser; not trusted image data",
        "scorer_sha256": hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }
    for name, left, right in (
        ("image_objective", [r["v1_image_credit"] for r in episodes], [r["v2_image_objective"] for r in episodes]),
        ("composite", [r["v1_composite"]["score"] for r in episodes], [r["v2_composite"]["score"] for r in episodes]),
    ):
        report[name + "_rank_correlation"] = rank_correlation(left, right)
        changes = [
            {"episode_uid": row["episode_uid"], "v1_rank": a, "v2_rank": b}
            for row, a, b in zip(episodes, ranks(left), ranks(right), strict=True)
            if a != b
        ]
        report["changed_image_rankings" if name == "image_objective" else "changed_composite_rankings"] = changes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--metadata", type=Path, help="trusted harness sidecar keyed by episode_uid")
    parser.add_argument(
        "--inspect-only", action="store_true", help="inspect plans including K=0/missing metadata; no reward comparison"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = []
    for path in args.artifacts:
        paths.extend(sorted(p for p in path.rglob("*") if p.suffix in {".json", ".jsonl"}) if path.is_dir() else [path])
    metadata = json.loads(args.metadata.read_text(encoding="utf8")) if args.metadata else {}
    try:
        result = inspect_traces(paths, metadata) if args.inspect_only else rescore(paths, metadata)
    except (ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
