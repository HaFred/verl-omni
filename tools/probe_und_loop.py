#!/usr/bin/env python3
"""Reproduce and cure the UND "assistant\\n" repetition loop against the live AR replica.

The trainer's UND prompt is literally ``apply_chat_template(messages, tools=[schema],
add_generation_prompt=True)``, which is exactly what the OpenAI chat endpoint renders
server-side -- so this probe exercises the same prompt the rollout sees, in seconds instead of
a ~17 minute trainer boot.

Usage:
    probe_und_loop.py <base_url> [trajectory.json]

Steps:
  1. replay the prompt from a rollout dump and show the raw output
  2. sweep decoding parameters to find which ones break the repetition
"""
from __future__ import annotations

import json
import re
import statistics
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.248.12.145:42905"
DUMP = sys.argv[2] if len(sys.argv) > 2 else (
    "outputs/bagel_corl_20260921_234105/e2e/bagel_corl_pr1/rollout_trajectories/step_000002/sample_0.01.json"
)
sys.path.insert(0, ".")
from verl_omni.agent_loop.bagel_corl_lib import GENERATE_IMAGE_TOOL_SCHEMA  # noqa: E402

MODEL = "/scratch/fq9hpsac/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b"


def split_prompt(prompt_text: str) -> tuple[str, str]:
    """Recover the (system, user) messages from a rendered training prompt."""
    body = prompt_text.split("<|im_start|>system\n", 1)[1]
    system, rest = body.split("\n\n# Tools\n\n", 1)
    user = rest.split("<|im_start|>user\n", 1)[1]
    user = user.split("<|im_end|>", 1)[0]
    return system, user


def ask(system: str, user: str, *, timeout: int = 180, **sampling) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "tools": [GENERATE_IMAGE_TOOL_SCHEMA],
        "max_tokens": sampling.pop("max_tokens", 250),
        **sampling,
    }
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=timeout)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    ch = r.json()["choices"][0]
    msg = ch.get("message") or {}
    text = (msg.get("content") or "") + "".join(
        f"<tool_call>{{'name': '{tc.get('function', {}).get('name')}'}}</tool_call>"
        for tc in (msg.get("tool_calls") or [])
    )
    return {"text": text, "finish": ch.get("finish_reason"), "usage": r.json().get("usage")}


def loop_score(text: str) -> tuple[float, str]:
    """(repeat ratio, verdict). A healthy turn is a plan + a generate_image call."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 1.0, "EMPTY"
    top = max(statistics.Counter(lines).values()) / len(lines)
    has_call = bool(re.search(r'"name"\s*:\s*"generate_image"', text)) or "generate_image" in text
    verdict = "CALL" if has_call else ("LOOP" if top > 0.5 else "prose")
    return top, verdict


def main() -> int:
    dump = json.load(open(DUMP))
    system, user = split_prompt(dump["prompt_text"])
    print("=== PROMPT ===")
    print("system:", repr(system[:160]))
    print("user  :", repr(user[:200]))

    print("\n=== 0. as-trained (temperature 1.0, no penalty) ===")
    r = ask(system, user, temperature=1.0, top_p=1.0, top_k=-1)
    if "error" in r:
        print(r["error"])
        return 1
    print(f"  finish={r['finish']} verdict={loop_score(r['text'])}")
    print("  raw:", repr(r["text"][:260]))

    SWEEPS = {
        "recipe (0.7/0.9/50/1.05)": dict(temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05),
        "+freq 0.3": dict(temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05, frequency_penalty=0.3),
        "+freq 0.7": dict(temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05, frequency_penalty=0.7),
        "+pres 0.3": dict(temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05, presence_penalty=0.3),
        "rep 1.2": dict(temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2),
        "temp 0.3": dict(temperature=0.3, top_p=0.9, top_k=50, repetition_penalty=1.05),
        "greedy": dict(temperature=0.0, top_p=1.0, top_k=-1),
    }
    print("\n=== 1. parameter sweep (3 trials each) ===")
    for label, params in SWEEPS.items():
        verdicts, heads = [], []
        for _ in range(3):
            out = ask(system, user, **params)
            if "error" in out:
                verdicts.append("ERR")
                heads.append(out["error"][:60])
                continue
            _ratio, verdict = loop_score(out["text"])
            verdicts.append(verdict)
            heads.append(out["text"][:110].replace("\n", " "))
        calls = verdicts.count("CALL")
        print(f"  {label:26s} calls={calls}/3 verdicts={verdicts}")
        print(f"      e.g. {heads[0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
