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
"""Preflight checks for agentic GRPO launch (exit non-zero on failure)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from urllib.request import Request, urlopen

# ── service checks ────────────────────────────────────────────────────────────


def check_vllm_omni(url: str, *, required: bool = True) -> None:
    """Health-check a vLLM-Omni image-generation server."""
    base = url.rstrip("/")

    def _get(path: str) -> tuple[int, bytes]:
        with urlopen(Request(f"{base}{path}", method="GET"), timeout=5) as resp:  # noqa: S310
            return resp.status, resp.read()

    try:
        status, body = _get("/health")
        if status != 200:
            raise RuntimeError(f"HTTP {status}")
        if body.strip():
            print(f"[INFO] vLLM-omni health OK: {json.loads(body)}")
        else:
            status, body = _get("/v1/models")
            if status != 200:
                raise RuntimeError(f"/health empty and /v1/models HTTP {status}")
            models = json.loads(body) if body.strip() else {}
            n = len(models.get("data") or [])
            print(f"[INFO] vLLM-omni health OK (empty /health); /v1/models count={n}")
    except Exception as exc:
        print(f"[ERROR] vLLM-omni health check failed at {base}/health: {exc}", file=sys.stderr)
        if required:
            sys.exit(2)


def check_qwen_image(url: str, *, required: bool = True) -> None:
    """Health-check a legacy Qwen-Image HTTP service."""
    health = url.rstrip("/").rsplit("/", 1)[0] + "/health"
    try:
        with urlopen(health, timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read())
        if not payload.get("ok"):
            raise RuntimeError(f"unhealthy response: {payload}")
        print(f"[INFO] Qwen-Image health OK: {payload}")
    except Exception as exc:
        print(f"[ERROR] Qwen-Image health check failed at {health}: {exc}", file=sys.stderr)
        if required:
            sys.exit(2)


def check_vllm_judge(url: str, *, required: bool = True) -> None:
    """Health-check a vLLM VL judge server (OpenAI /v1/chat/completions)."""
    base = url.rstrip("/")

    def _get(path: str) -> tuple[int, bytes]:
        with urlopen(Request(f"{base}{path}", method="GET"), timeout=5) as resp:  # noqa: S310
            return resp.status, resp.read()

    try:
        status, body = _get("/health")
        if status != 200:
            raise RuntimeError(f"HTTP {status}")
        if body.strip():
            print(f"[INFO] vLLM judge health OK: {json.loads(body)}")
        else:
            status, body = _get("/v1/models")
            if status != 200:
                raise RuntimeError(f"/health empty and /v1/models HTTP {status}")
            models = json.loads(body) if body.strip() else {}
            n = len(models.get("data") or [])
            print(f"[INFO] vLLM judge health OK (empty /health); /v1/models count={n}")
    except Exception as exc:
        print(f"[ERROR] vLLM judge health check failed at {base}/health: {exc}", file=sys.stderr)
        if required:
            sys.exit(2)


def check_legacy_reflect(url: str, *, required: bool = True) -> None:
    """Check a legacy /reflect endpoint (custom FastAPI)."""
    health = url.rstrip("/").rsplit("/", 1)[0] + "/health"
    try:
        with urlopen(health, timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read())
        print(f"[INFO] Reflect VLM health OK: {payload}")
    except Exception:
        try:
            req = Request(
                url,
                data=json.dumps({"user_request": "health check", "image_prompt": "a red apple", "notes": ""}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=30) as response:  # noqa: S310
                payload = json.loads(response.read())
            print(f"[INFO] Reflect VLM endpoint reachable: keys={sorted(payload)[:8]}")
        except Exception as exc:
            print(f"[ERROR] Reflect VLM check failed at {url}: {exc}", file=sys.stderr)
            if required:
                sys.exit(2)


# ── model checks ──────────────────────────────────────────────────────────────


def check_model_tool_template(model_path: str) -> None:
    """Verify the model exposes a tool-aware chat template and image processor."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    template = getattr(processor, "chat_template", "") or getattr(processor.tokenizer, "chat_template", "")
    if "<tool_call>" not in template or "tools" not in template:
        raise SystemExit(f"{model_path} does not expose the required tool-aware chat template")
    if not getattr(processor, "image_processor", None):
        raise SystemExit(f"{model_path} does not expose an image_processor")
    print(f"[INFO] Verified native tool template + image processor: {model_path}")


# ── GPU checks ────────────────────────────────────────────────────────────────


def check_gpu_free_memory(min_free_gb: float = 24) -> None:
    """Refuse to start if training GPUs lack free VRAM (prior crashed EngineCore)."""
    raw = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
    bad = []
    for i in idxs:
        if i >= len(raw):
            continue
        parts = [x.strip() for x in raw[i].split(",")]
        idx, total, free, used = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        print(f"[INFO] GPU {idx}: used={used / 1024:.1f}GiB free={free / 1024:.1f}GiB / {total / 1024:.1f}GiB")
        if free / 1024.0 < min_free_gb:
            bad.append((idx, free / 1024.0, used / 1024.0))
    if bad:
        print("[ERROR] Training GPUs do not have enough free VRAM before launch.", file=sys.stderr)
        print("[ERROR] A prior crashed VLLM::EngineCore often leaves ~60GiB occupied.", file=sys.stderr)
        for idx, free, used in bad:
            print(
                f"[ERROR]   GPU {idx}: free={free:.1f}GiB used={used:.1f}GiB (need >= {min_free_gb}GiB free)",
                file=sys.stderr,
            )
        print("[ERROR] Inspect: nvidia-smi", file=sys.stderr)
        print("[ERROR] Free zombies (keep the separate Qwen-Image server GPU if running):", file=sys.stderr)
        print(
            "[ERROR]   nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv", file=sys.stderr
        )
        print("[ERROR]   kill <EngineCore/Worker pids on CUDA_VISIBLE_DEVICES>   # or pick empty GPUs", file=sys.stderr)
        print("[ERROR] Override gate with MIN_FREE_GB=0 if you insist.", file=sys.stderr)
        sys.exit(2)


# ── main dispatcher ───────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <check> [args...]", file=sys.stderr)
        print("  checks: vllm-omni, qwen-image, vllm-judge, legacy-reflect, model, gpu", file=sys.stderr)
        sys.exit(2)

    check = sys.argv[1]

    if check == "vllm-omni":
        url, required = sys.argv[2], sys.argv[3] == "1" if len(sys.argv) > 3 else True
        check_vllm_omni(url, required=required)
    elif check == "qwen-image":
        url, required = sys.argv[2], sys.argv[3] == "1" if len(sys.argv) > 3 else True
        check_qwen_image(url, required=required)
    elif check == "vllm-judge":
        url, required = sys.argv[2], sys.argv[3] == "1" if len(sys.argv) > 3 else True
        check_vllm_judge(url, required=required)
    elif check == "legacy-reflect":
        url, required = sys.argv[2], sys.argv[3] == "1" if len(sys.argv) > 3 else True
        check_legacy_reflect(url, required=required)
    elif check == "model":
        check_model_tool_template(sys.argv[2])
    elif check == "gpu":
        min_free_gb = float(os.environ.get("MIN_FREE_GB", "24"))
        check_gpu_free_memory(min_free_gb)
    else:
        print(f"unknown check: {check}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
