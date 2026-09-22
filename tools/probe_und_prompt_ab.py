#!/usr/bin/env python3
"""Probe the live UND AR replica with the *exact* training prompt, and one variant.

Decisive A/B for the checkpoint's ``assistant\\n`` repetition loop. Both arms send the same
``prompt_token_ids`` except for the trailing ``<|im_start|>assistant\\n`` generation prompt:

  full         the prompt as the trainer renders it (``add_generation_prompt=True``)
  no_header    the same text with that header stripped, so the model must emit the role
               header itself. If the loop is "the model continues the transcript and restates
               the role", this is the arm that should recover.

Same sampling the agent loop sends (temperature 0.7 / top_p 0.9 / top_k 50 /
repetition_penalty 1.05), and the same ``stop=["<output>"]`` the new UND decode adds.
"""
from __future__ import annotations

import json
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.248.12.145:42905"
MODEL = "/scratch/fq9hpsac/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b"

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

TRAJ = "outputs/bagel_corl_20260921_234105/e2e/bagel_corl_pr1/rollout_trajectories"
CASES = [
    ("LOOPS  step2/s0", f"{TRAJ}/step_000002/sample_0.01.json"),
    ("CALLS  step1/s1", f"{TRAJ}/step_000001/sample_1.01.json"),
]
HEADER = "<|im_start|>assistant\n"

SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_tokens": 220,
    "stop": ["<output>"],
    "seed": 12345,
}


def decode(ids: list[int], label: str) -> str:
    body = {"model": MODEL, "prompt": ids, **SAMPLING}
    try:
        r = requests.post(f"{BASE}/v1/completions", json=body, timeout=180)
    except Exception as exc:  # noqa: BLE001
        return f"<request failed: {exc}>"
    if r.status_code != 200:
        return f"<HTTP {r.status_code}: {r.text[:300]}>"
    out = r.json()["choices"][0]
    txt = out.get("text", "")
    ntok = out.get("token_ids") or []
    print(f"  [{label}] tokens={len(ntok) if ntok else '?'} finish={out.get('finish_reason')}")
    return txt


def head_repeat_ratio(text: str) -> float:
    """Fraction of lines that are the same role label -- the loop signature."""
    words = [w for w in text.split("\n") if w.strip()]
    if not words:
        return 0.0
    from collections import Counter

    top = Counter(words).most_common(1)[0]
    return top[1] / len(words)


for label, path in CASES:
    d = json.load(open(path))
    prompt_text = d["prompt_text"]
    print(f"\n===== {label}  (dump gen_calls={d.get('num_gen_calls')}) =====")
    assert prompt_text.endswith(HEADER), repr(prompt_text[-40:])

    for arm, text in (("full", prompt_text), ("no_header", prompt_text[: -len(HEADER)])):
        ids = tok.encode(text, add_special_tokens=False)
        print(f"\n--- arm={arm}  prompt_tokens={len(ids)} ---")
        out = decode(ids, arm)
        print(f"  repeat_ratio={head_repeat_ratio(out):.2f}")
        print("  raw:", repr(out[:400]))
