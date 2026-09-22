#!/usr/bin/env python3
"""Summarize a Bagel Co-RL run's rollout evidence: UND turns, GEN calls, images.

Prints one line per dumped sample plus a roll-up, so each training step can be judged
against the acceptance criteria (meaningful UND turns; GEN images at the configured
denoising steps) without opening the trajectory JSONs by hand.

Usage:
    monitor_bagel_rollout.py <run_dir>            # e.g. outputs/bagel_corl_20260922_xxxxxx
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys
import time


def _png_size(path: str) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", head[16:24])
    except OSError:
        return None


def _turn_head(text: str | None, n: int = 90) -> str:
    s = " ".join(str(text or "").split())
    return s[:n]


def main() -> int:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    traj = os.path.join(run_dir, "e2e")
    if not os.path.isdir(traj):
        print(f"[monitor] no e2e dir under {run_dir}")
        return 1

    steps = sorted(glob.glob(os.path.join(traj, "*", "rollout_trajectories", "step_*")))
    total = with_gen = 0
    for step in steps:
        step_name = os.path.basename(step)
        rows = []
        for f in sorted(glob.glob(os.path.join(step, "*.json"))):
            try:
                d = json.load(open(f))
            except (OSError, json.JSONDecodeError):
                continue
            total += 1
            gen = int(d.get("num_gen_calls") or 0)
            with_gen += 1 if gen else 0
            rows.append(
                (
                    os.path.basename(f).replace(".json", ""),
                    gen,
                    d.get("response_tokens"),
                    d.get("turn_kind_counts"),
                    d.get("und_reward"),
                    [t for t in (d.get("turn_trace") or []) if t.get("record") == "und_turn"],
                )
            )
        print(f"\n===== {step_name} ({len(rows)} sample(s)) =====")
        for name, gen, rew, kinds, und_rew, turns in rows:
            print(f"  {name}: gen_calls={gen} rew={rew} kinds={kinds} und_reward={und_rew}")
            for t in turns:
                kind = t.get("kind")
                flag = "OK " if kind == "generate_image" else "   "
                print(
                    f"    {flag}turn{t.get('turn')} kind={kind} out_tokens={t.get('out_tokens')} "
                    f"req_ctx={t.get('req_ctx')} :: {_turn_head(t.get('text'))}"
                )

    imgs = sorted(glob.glob(os.path.join(traj, "*", "rollout_images", "**", "*.png"), recursive=True))
    print(f"\n===== images ({len(imgs)}) =====")
    for img in imgs[-12:]:
        size = _png_size(img)
        rel = os.path.relpath(img, traj)
        print(f"  {rel} {size[0]}x{size[1]}" if size else f"  {rel} <unreadable>")

    print(f"\n[monitor] samples={total} with_gen_call={with_gen} images={len(imgs)} at {time.strftime('%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
