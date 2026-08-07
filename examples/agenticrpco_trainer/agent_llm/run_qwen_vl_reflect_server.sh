#!/usr/bin/env bash
# Frozen Qwen3-VL reflect sidecar for agentic GRPO (correctness + aesthetics).
#
# Dedicated GPU (separate from train & Qwen-Image):
#   CUDA_VISIBLE_DEVICES=2 bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
#
# Co-locate with Qwen-Image on a single H800 (80 GB) — Qwen3-VL-2B is only ~4 GB:
#   CUDA_VISIBLE_DEVICES=0 bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
#   CUDA_VISIBLE_DEVICES=0 QWEN_IMAGE_MEMORY_MODE=model_offload \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8093}"

export AGENTIC_REFLECT_VLM_PATH="${AGENTIC_REFLECT_VLM_PATH:-${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}}"
export AGENTIC_REFLECT_GOOD_ENOUGH="${AGENTIC_REFLECT_GOOD_ENOUGH:-0.72}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] Set CUDA_VISIBLE_DEVICES to a free GPU not used by GRPO." >&2
  echo "[ERROR] Example: CUDA_VISIBLE_DEVICES=2 $0" >&2
  echo "[ERROR] Co-location with Qwen-Image on one H800 is fine; just set the same device for both." >&2
  exit 2
fi

echo "[INFO] Frozen Qwen3-VL reflect sidecar"
echo "[INFO]   model       : ${AGENTIC_REFLECT_VLM_PATH}"
echo "[INFO]   cuda devices: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO]   listen      : http://${HOST}:${PORT}"
echo "[INFO] Trainer should export AGENTIC_REFLECT_VLM_URL=http://${HOST}:${PORT}/reflect"

exec python3 -m uvicorn qwen_vl_reflect_server:app \
  --app-dir "${SCRIPT_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
