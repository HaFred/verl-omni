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
"""Overfit gate sidecar for Qwen3-VL visual-agent GRPO (force OFF).

Expected 10-step voluntary overfit (no teacher/force):

  Steps 1–10
    - Measure real Hermes: Decode should grow ``<tool_call>`` / voluntary modes
    - Artifacts under rollout_trajectories/step_XXXXXX/sample_*.**
    - Track native Hermes calls, two-call rewrites, and generated artifacts

Hard gates (no-force):
  G3  Trajectories live under step_[0-9]+/ (not only step_unknown/)
  G0  At least one trajectory JSON was written

Soft (reported, do not fail the run in --no-force):
  G1  ≥1 traj with tools≥2
  G2  escape rate
  G4  recent mean tools
  G5  teacher modes (should be absent when force is off)

Usage:
  python3 check_overfit_gates.py --run-dir outputs/e2e/<exp> --final --no-force
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
            "n_two_tool": 0,
            "n_escape": 0,
            "modes": {},
            "unknown_step_dirs": 0,
            "mean_voluntary": 0.0,
            "n_voluntary": 0,
            "judge_parse_ok": 0,
            "judge_parse_fail": 0,
            "judge_parse_ok_rate": 1.0,
        }
    steps = sorted({s for s, _, _ in trajs})
    escapes = 0
    tools = []
    protocol_like = 0
    two_tool = 0
    voluntary = 0
    modes: Counter[str] = Counter()
    judge_ok = 0
    judge_fail = 0
    for _, _, data in trajs:
        executed = int(data.get("num_tool_calls_executed") or 0)
        forced = int(data.get("num_forced_tool_calls") or 0)
        vol = int(data.get("num_voluntary_hermes") or 0)
        tools.append(executed)
        voluntary += vol
        if executed == 0 and forced == 0:
            escapes += 1
        if executed >= 2:
            two_tool += 1
            decodes = "\n".join(
                str(turn.get("decode") or "") for turn in (data.get("rollout_turns") or []) if isinstance(turn, dict)
            )
            if re.search(r"(?im)^\s*Reflection\s*:", decodes):
                protocol_like += 1
        blob = json.dumps(data)
        judge_ok += len(re.findall(r"agentic_judge\s+ok=1", blob, flags=re.IGNORECASE))
        judge_fail += len(re.findall(r"agentic_judge\s+ok=0", blob, flags=re.IGNORECASE))
        if judge_fail == 0 and re.search(r"unparseable response", blob, flags=re.IGNORECASE):
            judge_fail += len(re.findall(r"unparseable response", blob, flags=re.IGNORECASE))
        for action in data.get("hermes_actions") or []:
            if isinstance(action, dict) and action.get("mode"):
                modes[str(action["mode"])] += 1
        mode = data.get("hermes_impose_mode")
        if isinstance(mode, str) and mode not in {"none", ""}:
            modes[mode] += 0
    n = len(trajs)
    unknown = 0
    traj_root = trajs[0][1].parents[1] if trajs else None
    if traj_root is not None:
        unknown = 1 if (traj_root / "step_unknown").is_dir() and any((traj_root / "step_unknown").iterdir()) else 0
    judge_attempts = judge_ok + judge_fail
    return {
        "n_traj": n,
        "steps": steps,
        "escape_rate": escapes / max(1, n),
        "mean_tools": sum(tools) / max(1, n),
        "n_protocol_like": protocol_like,
        "n_two_tool": two_tool,
        "n_escape": escapes,
        "modes": dict(modes),
        "unknown_step_dirs": unknown,
        "mean_voluntary": voluntary / max(1, n),
        "n_voluntary": voluntary,
        "judge_parse_ok": judge_ok,
        "judge_parse_fail": judge_fail,
        "judge_parse_ok_rate": (judge_ok / judge_attempts) if judge_attempts else 1.0,
    }


def evaluate(
    run_dir: Path,
    *,
    max_escape_rate: float,
    last_k_steps: int,
    min_last_mean_tools: float,
    no_force: bool,
    min_judge_parse_ok_rate: float = 0.99,
) -> tuple[list[tuple[str, bool, str, bool]], dict]:
    """Return gates as (name, ok, detail, hard)."""
    trajs = _iter_traj_json(run_dir)
    summary = _summarize(trajs)
    gates: list[tuple[str, bool, str, bool]] = []

    ok0 = summary["n_traj"] >= 1
    gates.append(("G0_has_trajectories", ok0, f"n_traj={summary['n_traj']}", True))

    ok1 = summary["n_protocol_like"] >= 1
    gates.append(
        (
            "G1_reflected_rewrite_traj_exists",
            ok1,
            f"reflection+tools≥2={summary['n_protocol_like']}; tools≥2={summary['n_two_tool']} / {summary['n_traj']}",
            not no_force,
        )
    )

    ok2 = summary["n_traj"] == 0 or summary["escape_rate"] <= max_escape_rate
    gates.append(
        (
            "G2_low_escape_rate",
            ok2,
            f"escape_rate={summary['escape_rate']:.2f} (max {max_escape_rate:.2f}); escapes={summary['n_escape']}",
            not no_force,
        )
    )

    numbered = [s for s in summary["steps"] if s >= 0]
    ok3 = len(numbered) >= 1
    gates.append(
        (
            "G3_step_numbered_dirs",
            ok3,
            f"steps={numbered[:12]}{'...' if len(numbered) > 12 else ''}; "
            f"step_unknown_populated={summary['unknown_step_dirs']}",
            True,
        )
    )

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
            f"last_{last_k_steps}_steps mean_tools={last_mean:.2f} (min {min_last_mean_tools:.2f}); "
            f"mean_voluntary={summary.get('mean_voluntary', 0):.2f}",
            not no_force,
        )
    )

    mode_hits = sum(v for k, v in summary["modes"].items() if k in _TEACHER_MODES)
    if no_force:
        # Prefer voluntary / empty teacher modes when force is off.
        ok5 = mode_hits == 0 or summary.get("n_voluntary", 0) >= 1 or summary["n_traj"] == 0
        detail = (
            f"teacherish_mode_events={mode_hits} voluntary={summary.get('n_voluntary', 0)}; modes={summary['modes']}"
        )
        gates.append(("G5_voluntary_not_teacher", ok5, detail, False))
    else:
        ok5 = summary["n_traj"] == 0 or mode_hits >= 1 or summary["n_protocol_like"] >= 1
        gates.append(
            (
                "G5_teacher_or_protocol_signal",
                ok5,
                f"teacherish_mode_events={mode_hits}; modes={summary['modes']}",
                True,
            )
        )

    parse_rate = float(summary.get("judge_parse_ok_rate", 1.0))
    n_ok = int(summary.get("judge_parse_ok", 0))
    n_fail = int(summary.get("judge_parse_fail", 0))
    # Only enforce once we have observed judge attempts.
    ok6 = summary["n_traj"] == 0 or (n_ok + n_fail) == 0 or parse_rate >= min_judge_parse_ok_rate
    gates.append(
        (
            "G6_judge_parse_ok_rate",
            ok6,
            f"parse_ok_rate={parse_rate:.3f} (min {min_judge_parse_ok_rate:.2f}); ok={n_ok} fail={n_fail}",
            True,
        )
    )

    summary["last_mean_tools"] = last_mean
    return gates, summary


def _print_report(gates: list[tuple[str, bool, str, bool]], summary: dict, *, header: str, no_force: bool) -> int:
    print(header)
    print(
        f"  trajs={summary.get('n_traj', 0)} steps={summary.get('steps', [])} "
        f"escape_rate={summary.get('escape_rate', 0):.2f} "
        f"mean_tools={summary.get('mean_tools', 0):.2f} "
        f"mean_voluntary={summary.get('mean_voluntary', 0):.2f}"
    )
    failed = 0
    for name, ok, detail, hard in gates:
        if ok:
            mark = "PASS"
        elif hard:
            mark = "FAIL"
            failed += 1
        else:
            mark = "SOFT"
        print(f"  [{mark}] {name}: {detail}")
    if no_force:
        print("  (no-force mode: only hard gates fail the sidecar)")
    return failed


def watch_loop(
    run_dir: Path,
    *,
    total_steps: int,
    interval_s: float,
    max_escape_rate: float,
    last_k_steps: int,
    min_last_mean_tools: float,
    no_force: bool,
    min_judge_parse_ok_rate: float = 0.99,
) -> int:
    print(f"[GATE] watching {run_dir} every {interval_s:.0f}s (expect ~{total_steps} steps; no_force={no_force})")
    print_expected_behavior(total_steps, no_force=no_force)
    seen_steps: set[int] = set()
    last_failed = 0
    while True:
        gates, summary = evaluate(
            run_dir,
            max_escape_rate=max_escape_rate,
            last_k_steps=last_k_steps,
            min_last_mean_tools=min_last_mean_tools,
            no_force=no_force,
            min_judge_parse_ok_rate=min_judge_parse_ok_rate,
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
                no_force=no_force,
            )
        if total_steps > 0 and len([s for s in steps if 1 <= s <= total_steps]) >= total_steps:
            print("[GATE] watch: reached expected step count; exiting watch loop")
            return last_failed
        time.sleep(interval_s)


def print_expected_behavior(total_steps: int, *, no_force: bool = True) -> None:
    if no_force:
        print(
            f"""
[GATE] Expected behavior for this {total_steps}-step VOLUNTARY overfit (force OFF)
  • No teacher/force replace — Decode == Used for tool calls
  • Success signal: Decode contains Hermes <tool_call> generate_image (voluntary)
  • Tool obs includes image_vis=... so Reflection can cite real PNG stats
  • Reward: ≥1 Hermes call scores; ≥2 distinct prompts scores higher
  • Soft in {total_steps} steps: cold und may still emit mostly prose
""".rstrip()
        )
    else:
        print(
            f"""
[GATE] Expected behavior for this {total_steps}-step forced overfit
  • Force/teacher on: most rollouts do 2× generate_image via Used templates
  • Artifacts: rollout_trajectories/step_XXXXXX/sample_i.nn.{{json,txt}}
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
        "--no-force",
        action="store_true",
        help="Voluntary overfit: G0/G3/G6 hard; tool/teacher gates soft",
    )
    parser.add_argument(
        "--min-judge-parse-ok-rate",
        type=float,
        default=0.99,
        help="Hard gate G6: min agentic_judge parse_ok rate (default 0.99)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no trajs yet, treat gates as pass (used mid-watch before step 1)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    no_force = bool(args.no_force)
    if no_force:
        # Do not fail short voluntary smokes on high escape rate.
        args.max_escape_rate = max(args.max_escape_rate, 1.0)
        args.min_last_mean_tools = min(args.min_last_mean_tools, 0.0)

    if args.expect_only:
        print_expected_behavior(args.total_steps, no_force=no_force)
        return 0

    if args.watch:
        return watch_loop(
            run_dir,
            total_steps=args.total_steps,
            interval_s=args.interval_s,
            max_escape_rate=args.max_escape_rate,
            last_k_steps=args.last_k_steps,
            min_last_mean_tools=args.min_last_mean_tools,
            no_force=no_force,
            min_judge_parse_ok_rate=args.min_judge_parse_ok_rate,
        )

    print_expected_behavior(args.total_steps, no_force=no_force)
    gates, summary = evaluate(
        run_dir,
        max_escape_rate=args.max_escape_rate,
        last_k_steps=args.last_k_steps,
        min_last_mean_tools=args.min_last_mean_tools,
        no_force=no_force,
        min_judge_parse_ok_rate=args.min_judge_parse_ok_rate,
    )
    if summary.get("n_traj", 0) == 0 and args.allow_empty:
        print(f"[GATE] no trajectories yet under {run_dir}; --allow-empty → pass")
        return 0
    failed = _print_report(gates, summary, header=f"[GATE] final report for {run_dir}", no_force=no_force)
    report_path = run_dir / "overfit_gates.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "failed": failed,
                "no_force": no_force,
                "gates": [{"name": n, "ok": ok, "detail": d, "hard": hard} for n, ok, d, hard in gates],
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[GATE] wrote {report_path}")
    if failed:
        print(f"[GATE] FAILED {failed} hard gate(s)", file=sys.stderr)
        return 2
    print("[GATE] all hard gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
