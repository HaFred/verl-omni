#!/usr/bin/env bash
# Qwen3.5 launch helpers for agentic GRPO (source, do not execute).
#
# After MODEL_PATH is set:
#   source "${REPO_ROOT}/data/qwen35_env.sh"
#
# Sets:
#   GDN_PREFILL_BACKEND   — Triton by default (FlashInfer GDN JIT breaks on mismatched CTK)
#   TOOL_PARSER_FORMAT    — qwen3_coder for Qwen3.5 / Qwen3-Coder; hermes otherwise
#
# Safe when the actor is Qwen3-VL (Hermes): GDN preflight no-ops and the parser
# auto-detects hermes. Wrong parser → "Failed to decode tool call..." / tool_calls=0.
#
# shellcheck shell=bash

: "${MODEL_PATH:?MODEL_PATH must be set before sourcing data/qwen35_env.sh}"

# Qwen3.5 GDN: pip nvidia-cu13 is often headers 13.3 + nvcc 13.2, so FlashInfer
# GDN JIT dies with CCCL "CUDA compiler and CUDA toolkit headers are incompatible".
# Force Triton/FLA prefill (vLLM --gdn-prefill-backend triton) and skip that JIT.
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
export GDN_PREFILL_BACKEND

# Tool-call wire format must match the actor chat template.
# Qwen3.5 / Qwen3-Coder emit XML (<function=...><parameter=...>); Hermes is JSON.
if [[ -z "${TOOL_PARSER_FORMAT:-}" ]]; then
  TOOL_PARSER_FORMAT="$(python3 - "$MODEL_PATH" <<'PY'
import sys
from transformers import AutoConfig, AutoProcessor

path = sys.argv[1]
model_type = str(getattr(AutoConfig.from_pretrained(path, trust_remote_code=True), "model_type", "") or "")
if model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_coder"}:
    print("qwen3_coder")
    raise SystemExit(0)
proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
tmpl = (getattr(proc, "chat_template", None) or getattr(getattr(proc, "tokenizer", None), "chat_template", None) or "")
print("qwen3_coder" if "<function=" in tmpl else "hermes")
PY
)"
fi
export TOOL_PARSER_FORMAT
echo "[INFO] multi_turn.format=${TOOL_PARSER_FORMAT}"

# Qwen3.5 GDN backend preflight (no-op for non-qwen3_5 model_type).
python3 - "$MODEL_PATH" "${GDN_PREFILL_BACKEND}" <<'PY'
import os
import sys
from pathlib import Path

from transformers import AutoConfig

model_path, backend = sys.argv[1], sys.argv[2].strip().lower()
model_type = str(getattr(AutoConfig.from_pretrained(model_path, trust_remote_code=True), "model_type", "") or "")
if model_type != "qwen3_5":
    print(f"[INFO] model_type={model_type}; skipping Qwen3.5 GDN preflight")
    raise SystemExit(0)
print(f"[INFO] Qwen3.5 GDN prefill backend={backend}")
if backend == "flashinfer":
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    nvcc = Path(cuda_home) / "bin" / "nvcc" if cuda_home else Path()
    if not cuda_home or not nvcc.is_file():
        raise SystemExit(
            "GDN_PREFILL_BACKEND=flashinfer needs CUDA_HOME with bin/nvcc "
            f"(got CUDA_HOME={cuda_home!r}). Prefer GDN_PREFILL_BACKEND=triton on this box."
        )
    print(f"[WARN] flashinfer GDN JIT needs matching nvcc+headers; this box often mismatches.")
    print(f"[INFO] CUDA_HOME={cuda_home} nvcc={nvcc}")
PY
