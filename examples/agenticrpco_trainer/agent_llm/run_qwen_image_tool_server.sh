#!/usr/bin/env bash
# Frozen Qwen-Image service for Qwen3-VL agentic GRPO (vLLM-Omni).
#
# Qwen-Image is a diffusion model. The required ``--omni`` flag makes the CLI
# detect model_index.json and serve POST /v1/images/generations. Without it,
# vLLM takes the LLM ModelConfig path and incorrectly demands config.json.
#
# Recommended (tensor-parallel across 2 free H800s):
#   CUDA_VISIBLE_DEVICES=0,1 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#
# Single large free GPU:
#   CUDA_VISIBLE_DEVICES=0 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#
# Lower peak VRAM (slower):
#   CUDA_VISIBLE_DEVICES=0 QWEN_IMAGE_ENABLE_CPU_OFFLOAD=1 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8092}"

export QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:-Qwen/Qwen-Image}"
export QWEN_IMAGE_WIDTH="${QWEN_IMAGE_WIDTH:-512}"
export QWEN_IMAGE_HEIGHT="${QWEN_IMAGE_HEIGHT:-512}"
export QWEN_IMAGE_STEPS="${QWEN_IMAGE_STEPS:-20}"
export QWEN_IMAGE_TRUE_CFG_SCALE="${QWEN_IMAGE_TRUE_CFG_SCALE:-4.0}"
export QWEN_IMAGE_ENABLE_CPU_OFFLOAD="${QWEN_IMAGE_ENABLE_CPU_OFFLOAD:-0}"

VLLM_OMNI_BIN="${VLLM_OMNI_BIN:-$(command -v vllm-omni || true)}"
if [[ -z "${VLLM_OMNI_BIN}" ]]; then
  echo "[ERROR] vllm-omni is not on PATH; activate the vLLM-Omni virtualenv first." >&2
  exit 2
fi
PYTHON_BIN="$(dirname "${VLLM_OMNI_BIN}")/python"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] Set CUDA_VISIBLE_DEVICES to GPU(s) not used by GRPO." >&2
  echo "[ERROR] Example: CUDA_VISIBLE_DEVICES=0,1 $0" >&2
  exit 2
fi

# vLLM 0.24 in this environment needs the pip CUDA 13 runtime + nvcc.
_SP="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
_CUDA_HOME_CANDIDATE="${_SP}/nvidia/cu13"
if [[ -x "${_CUDA_HOME_CANDIDATE}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-${_CUDA_HOME_CANDIDATE}}"
  export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
_CUDA_COMPAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.cache/cuda-compat-13"
export LD_LIBRARY_PATH="${_CUDA_COMPAT_DIR}:${_SP}/nvidia/cu13/lib:${_SP}/nvidia/cuda_runtime/lib:${_SP}/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${QWEN_IMAGE_NUM_GPUS:-${#_GPU_IDS[@]}}"
unset _GPU_IDS

_EXTRA_ARGS=()
if [[ "${QWEN_IMAGE_ENABLE_CPU_OFFLOAD}" == "1" ]]; then
  _EXTRA_ARGS+=(--enable-cpu-offload)
fi

echo "[INFO] Frozen Qwen-Image tool (vLLM-Omni /v1/images/generations)"
echo "[INFO]   model       : ${QWEN_IMAGE_MODEL}"
echo "[INFO]   cuda devices: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO]   num GPUs    : ${NUM_GPUS}"
echo "[INFO]   resolution  : ${QWEN_IMAGE_WIDTH}x${QWEN_IMAGE_HEIGHT}"
echo "[INFO]   steps / CFG : ${QWEN_IMAGE_STEPS} / ${QWEN_IMAGE_TRUE_CFG_SCALE}"
echo "[INFO]   listen      : http://${HOST}:${PORT}"
echo "[INFO] Trainer should export AGENTIC_VLLM_OMNI_URL=http://${HOST}:${PORT}"

exec "${VLLM_OMNI_BIN}" serve "${QWEN_IMAGE_MODEL}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --tensor-parallel-size "${NUM_GPUS}" \
  "${_EXTRA_ARGS[@]}" \
  "$@"
