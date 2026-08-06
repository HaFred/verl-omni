#!/usr/bin/env bash
# Frozen Qwen-Image service for Qwen3-VL agentic GRPO.
#
# Recommended when Qwen-Image OOMs on one GPU (split across 2 free H800s):
#   CUDA_VISIBLE_DEVICES=0,1 QWEN_IMAGE_MEMORY_MODE=balanced \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#
# Single large free GPU:
#   CUDA_VISIBLE_DEVICES=0 QWEN_IMAGE_MEMORY_MODE=full \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#
# Single GPU, lower peak VRAM (slower):
#   CUDA_VISIBLE_DEVICES=0 QWEN_IMAGE_MEMORY_MODE=sequential_offload \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#
# Quantized MMDiT (requires bitsandbytes):
#   QWEN_IMAGE_MEMORY_MODE=mmdit_nf4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8092}"

export QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:-Qwen/Qwen-Image}"
# Prefer balanced when the operator already exported multi-GPU visibility.
if [[ -z "${QWEN_IMAGE_MEMORY_MODE:-}" ]]; then
  _n_vis=0
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _ids <<< "${CUDA_VISIBLE_DEVICES}"
    _n_vis="${#_ids[@]}"
  fi
  if [[ "${_n_vis}" -ge 2 ]]; then
    export QWEN_IMAGE_MEMORY_MODE=balanced
  else
    export QWEN_IMAGE_MEMORY_MODE=model_offload
  fi
  unset _n_vis _ids
fi
export QWEN_IMAGE_WIDTH="${QWEN_IMAGE_WIDTH:-512}"
export QWEN_IMAGE_HEIGHT="${QWEN_IMAGE_HEIGHT:-512}"
export QWEN_IMAGE_STEPS="${QWEN_IMAGE_STEPS:-20}"
export QWEN_IMAGE_TRUE_CFG_SCALE="${QWEN_IMAGE_TRUE_CFG_SCALE:-4.0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] Set CUDA_VISIBLE_DEVICES to GPU(s) not used by GRPO." >&2
  echo "[ERROR] Example: CUDA_VISIBLE_DEVICES=0,1 QWEN_IMAGE_MEMORY_MODE=balanced $0" >&2
  exit 2
fi

if [[ "${QWEN_IMAGE_MEMORY_MODE}" == "balanced" ]]; then
  IFS=',' read -r -a _ids <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ "${#_ids[@]}" -lt 2 ]]; then
    echo "[ERROR] balanced mode needs >=2 CUDA_VISIBLE_DEVICES (got: ${CUDA_VISIBLE_DEVICES})" >&2
    exit 2
  fi
  unset _ids
fi

if [[ "${QWEN_IMAGE_MEMORY_MODE}" == "mmdit_nf4" ]]; then
  python3 -c "import bitsandbytes" 2>/dev/null || {
    echo "[ERROR] mmdit_nf4 requires bitsandbytes. Use balanced/sequential_offload or install bitsandbytes." >&2
    exit 2
  }
fi

echo "[INFO] Frozen Qwen-Image tool"
echo "[INFO]   model       : ${QWEN_IMAGE_MODEL}"
echo "[INFO]   memory mode : ${QWEN_IMAGE_MEMORY_MODE}"
echo "[INFO]   cuda devices: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO]   resolution  : ${QWEN_IMAGE_WIDTH}x${QWEN_IMAGE_HEIGHT}"
echo "[INFO]   steps / CFG : ${QWEN_IMAGE_STEPS} / ${QWEN_IMAGE_TRUE_CFG_SCALE}"
echo "[INFO]   listen      : http://${HOST}:${PORT}"
echo "[INFO] Trainer should export AGENTIC_QWEN_IMAGE_URL=http://${HOST}:${PORT}/generate"
echo "[INFO] Kill any stale :${PORT} server before relaunch (old OOM processes stay healthy on /health)."

exec python3 -m uvicorn qwen_image_tool_server:app \
  --app-dir "${SCRIPT_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
