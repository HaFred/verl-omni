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
"""Env-config utilities for agentic GRPO launch (compute and print to stdout)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def model_slug(model_path: str) -> str:
    """Derive a short filesystem-safe slug from a HuggingFace model id or path."""
    raw = (model_path or "").strip().rstrip("/")
    p = Path(raw)
    text = raw
    for part in reversed(p.parts):
        if part.startswith("models--") and "--" in part:
            text = part[len("models--") :].replace("--", "/")
            break
        if part not in {"snapshots", "refs", "blobs"} and not re.fullmatch(r"[0-9a-f]{8,}", part):
            if "/" in raw and not raw.startswith("/"):
                text = raw
            elif part:
                text = part
            break
    name = text.split("/")[-1] if text else "model"
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "model"


def detect_tool_format(model_path: str) -> str:
    """Detect tool-call wire format from model config (``hermes`` or ``qwen3_coder``)."""
    from transformers import AutoConfig, AutoProcessor

    model_type = str(getattr(AutoConfig.from_pretrained(model_path, trust_remote_code=True), "model_type", "") or "")
    if model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_coder"}:
        return "qwen3_coder"
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tmpl = (
        getattr(proc, "chat_template", None) or getattr(getattr(proc, "tokenizer", None), "chat_template", None) or ""
    )
    return "qwen3_coder" if "<function=" in tmpl else "hermes"


def compute_gpu_mem_util() -> float:
    """Auto-compute vLLM GPU memory utilization from currently free VRAM."""
    raw = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
    free_fracs = []
    for i in idxs:
        if i >= len(raw):
            continue
        total_s, free_s = [x.strip() for x in raw[i].split(",")]
        total, free = float(total_s), float(free_s)
        if total > 0:
            # Leave headroom for FSDP residual + LoRA sync / wake_up fragmentation.
            free_fracs.append(0.25 * free / total)
    if not free_fracs:
        return 0.12
    return max(0.08, min(0.15, min(free_fracs)))


def build_ray_env_json() -> str:
    """Build RAY_RUNTIME_ENV_JSON forwarding agentic env vars to Ray workers."""
    keys = [
        "LD_LIBRARY_PATH",
        "PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "HF_HOME",
        "VERL_USE_EXTERNAL_MODULES",
        "WANDB_API_KEY",
        "WANDB_MODE",
        "WANDB_SERVICE_TRANSPORT",
        "WANDB_SILENT",
        "WANDB_DIR",
        "WANDB_PROJECT",
        "AGENTIC_VLLM_OMNI_URL",
        "AGENTIC_VLLM_URL",
        "AGENTIC_QWEN_IMAGE_URL",
        "AGENTIC_REFLECT_VLM_URL",
        "AGENTIC_REFLECT_VLM_PATH",
        "AGENTIC_DIFFUSION_TOOL_URL",
        "AGENTIC_DIFFUSION_TOOL_TOKEN",
        "AGENTIC_DIFFUSION_TOOL_TIMEOUT",
        "AGENTIC_DIFFUSION_IMAGE_DIR",
        "AGENTIC_E2E_ROOT",
        "AGENTIC_E2E_RUN_NAME",
        "AGENTIC_FORCE_REFLECTION_AFTER_JUDGE",
        "AGENTIC_MAX_GENERATE_IMAGE_PASSES",
        "AGENTIC_FORCE_FIRST_GENERATE",
        "AGENTIC_FORCE_FIRST_WARMUP_STEPS",
        "AGENTIC_FORCE_FIRST_END_STEP",
        "AGENTIC_REWRITE_JUDGE_BEFORE_GENERATE",
        "AGENTIC_REFLECT_VLM_TIMEOUT",
        "AGENTIC_REFLECT_MAX_NEW_TOKENS",
        "AGENTIC_JUDGE_PARSE_RETRIES",
        "AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD",
        "AGENTIC_REFLECT_GOOD_ENOUGH",
        "AGENTIC_BLOCK_GENERATE_AFTER_YES",
        "AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES",
        "QWEN_IMAGE_SEED",
        "QWEN_IMAGE_DIVERSIFY_SEED",
    ]
    base: dict = {}
    try:
        base = json.loads(os.environ.get("RAY_RUNTIME_ENV_JSON") or "{}")
    except json.JSONDecodeError:
        base = {}
    env_vars = dict(base.get("env_vars") or {})
    for k in keys:
        if os.environ.get(k):
            env_vars[k] = os.environ[k]
    base["env_vars"] = env_vars
    return json.dumps(base)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <command> [args...]", file=sys.stderr)
        print("  commands: slug, tool-format, gpu-mem-util, ray-env", file=sys.stderr)
        sys.exit(2)

    cmd = sys.argv[1]

    if cmd == "slug":
        print(model_slug(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL_PATH", "")))
    elif cmd == "tool-format":
        print(detect_tool_format(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL_PATH", "")))
    elif cmd == "gpu-mem-util":
        print(f"{compute_gpu_mem_util():.2f}")
    elif cmd == "ray-env":
        print(build_ray_env_json())
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
