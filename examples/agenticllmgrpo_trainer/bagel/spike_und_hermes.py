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
"""UND Hermes serving spike for published Bagel (fail-closed; no Qwen fallback).

Exit 0 only when a Bagel replica emits ``<tool_call>`` / ``generate_image``.
``bagel_single_stage`` GEN deploy yaml is not UND proof.

Environment:
  BAGEL_MODEL_PATH   local Bagel checkpoint (tokenizer + specials)
  BAGEL_UND_URL      optional OpenAI-compatible chat URL for a live replica
  BAGEL_SPIKE_OFFLINE_SCHEMA=1  tokenizer/schema only (CI); does not prove serving
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from verl_omni.agent_loop.bagel_corl_lib import (
    GENERATE_IMAGE_TOOL_SCHEMA,
    HERMES_SPECIAL_TOKENS,
    parse_hermes_tool_call,
    und_turn_kind,
)

TOOLS = [GENERATE_IMAGE_TOOL_SCHEMA]


def inspect_tokenizer(model_path: str) -> dict[str, bool]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab_specials = set(tokenizer.all_special_tokens)
    added = set(getattr(tokenizer, "additional_special_tokens", None) or [])
    present = {}
    for token in HERMES_SPECIAL_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        present[token] = token in vocab_specials or token in added or (tid is not None and tid != unk)
    return present


def query_und_replica(url: str, prompt: str, timeout_s: float = 120.0) -> str:
    payload = {
        "model": "bagel",
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "temperature": 0.7,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"] or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bagel UND Hermes generate_image spike (no Qwen fallback)")
    parser.add_argument("--model-path", default=os.environ.get("BAGEL_MODEL_PATH"))
    parser.add_argument("--und-url", default=os.environ.get("BAGEL_UND_URL"))
    parser.add_argument(
        "--prompt",
        default="Draw a red circle on a white background. Use generate_image if you can render it.",
    )
    parser.add_argument("--offline-schema", action="store_true", default=os.environ.get("BAGEL_SPIKE_OFFLINE_SCHEMA") == "1")
    args = parser.parse_args(argv)

    print("tool_schema:", json.dumps(GENERATE_IMAGE_TOOL_SCHEMA))
    print("prompt_token_ids: Bagel UND should prefer skipping decode→re-tokenize when the replica returns ids")

    if args.model_path:
        specials = inspect_tokenizer(args.model_path)
        print("tokenizer_hermes_specials:", json.dumps(specials))

    if args.offline_schema and not args.und_url:
        print("offline schema check only; this is NOT UND serving proof")
        return 0

    if not args.und_url:
        print(
            "FAIL-CLOSED: set BAGEL_UND_URL to a Bagel AR replica. "
            "Do not substitute Qwen3-VL. bagel_single_stage is GEN-only.",
            file=sys.stderr,
        )
        return 2

    try:
        text = query_und_replica(args.und_url, args.prompt)
    except urllib.error.URLError as exc:
        print(f"FAIL-CLOSED: UND replica request failed: {exc}", file=sys.stderr)
        return 2

    print("und_text:", text)
    kind = und_turn_kind(text)
    call = parse_hermes_tool_call(text)
    if kind != "generate_image" or call is None or call.get("name") != "generate_image":
        print("FAIL-CLOSED: Bagel UND did not emit Hermes <tool_call> generate_image", file=sys.stderr)
        return 1
    print("spike_ok: generate_image from Bagel UND (not Qwen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
