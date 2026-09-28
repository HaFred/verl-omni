"""Reconstruct a real Bagel Co-RL UND transcript and score it (diagnostic, run by hand).

Reads an episode's artifacts from a completed run, rebuilds the exact text the colocated
reward manager decodes (``prompt`` is excluded; ``responses`` is the concatenation of the
per-turn model text, the image observation, the forced reflection, and the closing
``Done.``), and reports the episode scorer's dims. Before the dialect fix this returned
``score=0.0`` for every episode, which is what pinned ``critic/score/mean`` at 0.0.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_omni.agent_loop.rpco_turn_protocol import build_forced_reflection  # noqa: E402
from verl_omni.utils.reward_score.agentic_multidim_reward import compute_score  # noqa: E402

RUN = os.environ.get("VERIFY_RUN", "outputs/bagel_corl_rm1_20260927_165227")
STEP = os.environ.get("VERIFY_STEP", "step_000003")
LIMIT = int(os.environ.get("VERIFY_LIMIT", "3"))


def _model_text(traj_txt: str) -> str:
    """The model's own decoded turns, un-indented out of the human-readable dump."""
    lines = traj_txt.splitlines()
    out, capturing = [], False
    for line in lines:
        if re.match(r"^  turn=\d+ ", line):
            capturing = True
            continue
        if re.match(r"^  gen_call ", line):
            capturing = False
            continue
        if capturing and line.startswith("      "):
            out.append(line[6:])
        elif capturing and line == "":
            out.append("")
    return "\n".join(out)


def build_transcript(record: dict, traj_txt: str, *, png: str, force_done: bool = True) -> str:
    judge_text = record.get("judge_text") or ""
    turns = _model_text(traj_txt)
    parts = [turns, f"path={png}"]
    if judge_text:
        built = build_forced_reflection(
            judge_text,
            force_done=force_done,
            generate_pass=1,
            max_passes=1 if force_done else 3,
        )
        if built is not None:
            parts.append(built[0])
    parts.append("Done.")
    return "".join(parts)


def main() -> int:
    df = pd.read_parquet("outputs/data/agentic_unicot/train.parquet")
    gt = df.iloc[0]["reward_model"]["ground_truth"]
    print(f"ground_truth task_type={gt['task_type']!r} expected={gt['expected_num_images']}\n")

    rows = []
    for line in open(f"{RUN}/hermes_actions/{STEP}.jsonl"):
        rows.append(json.loads(line))

    for record in rows[:LIMIT]:
        relpath = record["trajectory_relpath"]
        traj = open(f"{RUN}/rollout_trajectories/{relpath}.txt", errors="ignore").read()
        png = (record.get("image_paths") or ["/tmp/bagel_corl_gen/gen_missing.png"])[0]
        text = build_transcript(record, traj, png=png)
        print("=" * 88)
        print(f"episode {record['episode_uid'][:8]}  und_reward={record.get('und_reward')}")
        print(f"  judge_text: {re.sub(r'\\s+', ' ', record.get('judge_text') or '')[:110]}")
        print(f"  transcript tail: ...{text[-160:]!r}")
        result = compute_score(ground_truth=gt, solution_str=text, extra_info={})
        print(
            "  -> score=%.4f method=%s calls=%s generates=%s judge_ok=%s "
            "done=%s reflect=%.4f plan=%.4f format=%.4f tool=%.4f result=%.4f"
            % (
                result["score"],
                result["method"],
                result["num_hermes_tool_calls"],
                result["n_successful_generates"],
                result["judge_parse_ok"],
                result["terminal_done"],
                result["reward_reflect"],
                result["reward_plan"],
                result["reward_format"],
                result["reward_tool"],
                result["reward_result"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
