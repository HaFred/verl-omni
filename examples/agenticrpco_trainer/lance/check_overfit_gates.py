#!/usr/bin/env python3
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
"""Overfit gate sidecar for Lance agentic GRPO e2e.

Expected 10-step overfit behavior (force + stable teacher + hard-gated reward):

  Steps 1–3
    - Almost every rollout executes ≥2 generate_image calls (force)
    - Hermes actions dominated by overfit_teacher / wrap_* / final_confirm
    - Artifacts under rollout_trajectories/step_XXXXXX/sample_*.**
    - Reward contrast: teacher-replaced trajs high; on-policy garbage ≈0

  Steps 4–7
    - Escape rate (tools=0 AND forced=0) stays low — no spam-to-max-length majority
    - Still mostly multi-turn (≥3 assistant/tool turns on successful trajs)

  Steps 8–10
    - Same force/protocol health as above
    - Bonus (soft): num_voluntary_hermes rising is nice-to-have, not required
      for a 10-step smoke (cold und usually needs teacher longer)

Hard gates (fail the sidecar if broken):
  G1  After ≥1 finished step: ≥1 traj with num_tool_calls_executed ≥ 2
  G2  Escape rate across all scored trajs < --max-escape-rate (default 0.25)
  G3  Trajectories live under step_[0-9]+/ (not only step_unknown/)
  G4  Among last N finished steps (default 3): mean tools_executed ≥ 1.0
  G5  At least one traj uses a teacher/force Hermes mode (not all empty)

Usage:
  python3 check_overfit_gates.py --run-dir outputs/e2e/<exp> --final
  python3 check_overfit_gates.py --run-dir outputs/e2e/<exp> --watch --total-steps 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

_STEP_DIR_RE = re.compile(r"^step_(\d+)$")
_TEACHER_MODES = {
    "overfit_teacher",
    "teacher",
    "wrap_caption",
    "wrap_json",
    "final_confirm",
}


def _iter_traj_json(run_dir: Path) -> list[tuple[int, Path, dict]]:
    root = run_dir / "rollout_trajectories"
    if not root.is_dir():
        return []
    out: list[tuple[int, Path, dict]] = []
    for step_dir in sorted(root.iterdir()):
        if not step_dir.is_dir():
            continue
        m = _STEP_DIR_RE.match(step_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        for path in sorted(step_dir.glob("sample_*.json")):
            if path.name.endswith(".hermes_actions.json"):
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append((step, path, data))
    return out


def _summarize(trajs: list[tuple[int, Path, dict]]) -> dict:
    if not trajs:
        return {
            "n_traj": 0,
            "steps": [],
            "escape_rate": 1.0,
            "mean_tools": 0.0,
            "n_protocol_like": 0,
            "n_escape": 0,
            "modes": {},
            "unknown_step_dirs": 0,
        }
    steps = sorted({s for s, _, _ in trajs})
    escapes = 0
    tools = []
    protocol_like = 0
    modes: Counter[str] = Counter()
    for _, _, data in trajs:
        executed = int(data.get("num_tool_calls_executed") or 0)
        forced = int(data.get("num_forced_tool_calls") or 0)
        tools.append(executed)
        if executed == 0 and forced == 0:
            escapes += 1
        if executed >= 2:
            protocol_like += 1
        for action in data.get("hermes_actions") or []:
            if isinstance(action, dict) and action.get("mode"):
                modes[str(action["mode"])] += 1
        # also count impose mode string if present
        mode = data.get("hermes_impose_mode")
        if isinstance(mode, str) and mode not in {"none", ""}:
            modes[mode] += 0  # keep key visible without double-count spam
    n = len(trajs)
    unknown = 0
    traj_root = trajs[0][1].parents[1] if trajs else None
    if traj_root is not None:
        unknown = 1 if (traj_root / "step_unknown").is_dir() and any((traj_root / "step_unknown").iterdir()) else 0
    return {
        "n_traj": n,
        "steps": steps,
        "escape_rate": escapes / max(1, n),
        "mean_tools": sum(tools) / max(1, n),
        "n_protocol_like": protocol_like,
        "n_escape": escapes,
        "modes": dict(modes),
        "unknown_step_dirs": unknown,
        "mean_voluntary": sum(int(d.get("num_voluntary_hermes") or 0) for _, _, d in trajs) / max(1, n),
    }


def evaluate(
    run_dir: Path,
    *,
    max_escape_rate: float,
    last_k_steps: int,
    min_last_mean_tools: float,
) -> tuple[list[tuple[str, bool, str]], dict]:
    trajs = _iter_traj_json(run_dir)
    summary = _summarize(trajs)
    gates: list[tuple[str, bool, str]] = []

    # G1
    ok1 = summary["n_protocol_like"] >= 1
    gates.append(
        (
            "G1_two_tool_traj_exists",
            ok1,
            f"protocol_like(tools≥2)={summary['n_protocol_like']} / {summary['n_traj']}",
        )
    )

    # G2
    ok2 = summary["n_traj"] == 0 or summary["escape_rate"] <= max_escape_rate
    gates.append(
        (
            "G2_low_escape_rate",
            ok2,
            f"escape_rate={summary['escape_rate']:.2f} (max {max_escape_rate:.2f}); escapes={summary['n_escape']}",
        )
    )

    # G3
    numbered = [s for s in summary["steps"] if s >= 0]
    ok3 = len(numbered) >= 1
    gates.append(
        (
            "G3_step_numbered_artifacts",
            ok3,
            f"steps={numbered[:12]}{'...' if len(numbered) > 12 else ''}; "
            f"step_unknown_populated={summary['unknown_step_dirs']}",
        )
    )

    # G4 last-k mean tools
    if numbered:
        last_steps = set(numbered[-last_k_steps:])
        last_trajs = [t for t in trajs if t[0] in last_steps]
        last_mean = sum(int(d.get("num_tool_calls_executed") or 0) for _, _, d in last_trajs) / max(1, len(last_trajs))
    else:
        last_mean = 0.0
    ok4 = summary["n_traj"] == 0 or last_mean >= min_last_mean_tools
    gates.append(
        (
            "G4_recent_mean_tools",
            ok4,
            f"last_{last_k_steps}_steps mean_tools={last_mean:.2f} (min {min_last_mean_tools:.2f})",
        )
    )

    # G5 teacher/force mode present
    mode_hits = sum(v for k, v in summary["modes"].items() if k in _TEACHER_MODES)
    ok5 = summary["n_traj"] == 0 or mode_hits >= 1 or summary["n_protocol_like"] >= 1
    gates.append(
        (
            "G5_teacher_or_protocol_signal",
            ok5,
            f"teacherish_mode_events={mode_hits}; modes={summary['modes']}",
        )
    )

    summary["last_mean_tools"] = last_mean
    return gates, summary


def _print_report(gates: list[tuple[str, bool, str]], summary: dict, *, header: str) -> int:
    print(header)
    print(
        f"  trajs={summary.get('n_traj', 0)} steps={summary.get('steps', [])} "
        f"escape_rate={summary.get('escape_rate', 0):.2f} "
        f"mean_tools={summary.get('mean_tools', 0):.2f} "
        f"mean_voluntary={summary.get('mean_voluntary', 0):.2f}"
    )
    failed = 0
    for name, ok, detail in gates:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}: {detail}")
    return failed


def watch_loop(
    run_dir: Path,
    *,
    total_steps: int,
    interval_s: float,
    max_escape_rate: float,
    last_k_steps: int,
    min_last_mean_tools: float,
) -> int:
    print(
        f"[GATE] watching {run_dir} every {interval_s:.0f}s "
        f"(expect ~{total_steps} steps; hard gates apply once trajs appear)"
    )
    print_expected_behavior(total_steps)
    seen_steps: set[int] = set()
    last_failed = 0
    while True:
        gates, summary = evaluate(
            run_dir,
            max_escape_rate=max_escape_rate,
            last_k_steps=last_k_steps,
            min_last_mean_tools=min_last_mean_tools,
        )
        steps = set(summary.get("steps") or [])
        new = sorted(steps - seen_steps)
        if new or summary.get("n_traj", 0) == 0:
            if new:
                seen_steps |= set(new)
            last_failed = _print_report(
                gates,
                summary,
                header=f"[GATE] snapshot steps={sorted(steps)} (+{new})",
            )
        # Stop watching once we have all expected numbered steps (best-effort).
        if total_steps > 0 and len([s for s in steps if 1 <= s <= total_steps]) >= total_steps:
            print("[GATE] watch: reached expected step count; exiting watch loop")
            return last_failed
        # Parent may kill us; otherwise idle until final check.
        time.sleep(interval_s)


def print_expected_behavior(total_steps: int) -> None:
    print(
        f"""
[GATE] Expected behavior for this {total_steps}-step overfit smoke
  • Force stays on (MIN_TOOL_CALLS=2, STABLE_TEACHER=1): most rollouts do 2× generate_image
  • Hard-gated reward: incomplete protocol → score 0; full protocol → ≥0.55
  • ~{int(100 * 0.3)}% forced turns keep raw decode (ON_POLICY_FRAC) for GRPO contrast
  • Artifacts: rollout_trajectories/step_XXXXXX/sample_i.nn.{{json,txt}}
  • Soft (not gated in 10 steps): voluntary Hermes may still be rare on cold und
""".rstrip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true", help="Poll until killed or step count reached")
    parser.add_argument("--final", action="store_true", help="One-shot final gate evaluation")
    parser.add_argument("--expect-only", action="store_true", help="Print expected behavior and exit 0")
    parser.add_argument("--total-steps", type=int, default=10)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--max-escape-rate", type=float, default=0.25)
    parser.add_argument("--last-k-steps", type=int, default=3)
    parser.add_argument("--min-last-mean-tools", type=float, default=1.0)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no trajs yet, treat gates as pass (used mid-watch before step 1)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    if args.expect_only:
        print_expected_behavior(args.total_steps)
        return 0

    if args.watch:
        return watch_loop(
            run_dir,
            total_steps=args.total_steps,
            interval_s=args.interval_s,
            max_escape_rate=args.max_escape_rate,
            last_k_steps=args.last_k_steps,
            min_last_mean_tools=args.min_last_mean_tools,
        )

    # default / --final
    print_expected_behavior(args.total_steps)
    gates, summary = evaluate(
        run_dir,
        max_escape_rate=args.max_escape_rate,
        last_k_steps=args.last_k_steps,
        min_last_mean_tools=args.min_last_mean_tools,
    )
    if summary.get("n_traj", 0) == 0 and args.allow_empty:
        print(f"[GATE] no trajectories yet under {run_dir}; --allow-empty → pass")
        return 0
    failed = _print_report(gates, summary, header=f"[GATE] final report for {run_dir}")
    report_path = run_dir / "overfit_gates.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "failed": failed,
                "gates": [{"name": n, "ok": ok, "detail": d} for n, ok, d in gates],
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[GATE] wrote {report_path}")
    if failed:
        print(f"[GATE] FAILED {failed} gate(s)", file=sys.stderr)
        return 2
    print("[GATE] all hard gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
