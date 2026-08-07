#!/usr/bin/env bash
# Frozen Qwen3-VL judge sidecar for agentic GRPO (continuous batching via vLLM).
#
# Dedicated GPU (separate from train & Qwen-Image):
#   CUDA_VISIBLE_DEVICES=2 bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
#
# Co-locate with Qwen-Image on one H800 (e.g. device 7) — start THIS first, then image:
#   CUDA_VISIBLE_DEVICES=7 bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
#   CUDA_VISIBLE_DEVICES=7 QWEN_IMAGE_ENABLE_CPU_OFFLOAD=1 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#
# Colocation default: gpu_memory_utilization=0.12 (~10 GiB on H800) so Qwen-Image
# keeps the rest. Override with AGENTIC_REFLECT_GPU_MEM_UTIL if needed.
#
# Or use the legacy FastAPI server for the custom /reflect endpoint:
#   AGENTIC_REFLECT_VLM_URL=http://127.0.0.1:8093/reflect
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8093}"

export AGENTIC_REFLECT_VLM_PATH="${AGENTIC_REFLECT_VLM_PATH:-${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}}"
export AGENTIC_REFLECT_GOOD_ENOUGH="${AGENTIC_REFLECT_GOOD_ENOUGH:-0.72}"
# Low util by default so this can share an H800 with Qwen-Image (CPU-offloaded).
REFLECT_GPU_MEM_UTIL="${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.12}"
REFLECT_MAX_NUM_SEQS="${AGENTIC_REFLECT_MAX_NUM_SEQS:-4}"
REFLECT_MAX_MODEL_LEN="${AGENTIC_REFLECT_MAX_MODEL_LEN:-4096}"

VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
if [[ -z "${VLLM_BIN}" ]]; then
  echo "[ERROR] vllm is not on PATH; activate the vLLM-Omni virtualenv first." >&2
  exit 2
fi
PYTHON_BIN="$(dirname "${VLLM_BIN}")/python"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] Set CUDA_VISIBLE_DEVICES to a free GPU not used by GRPO." >&2
  echo "[ERROR] Example: CUDA_VISIBLE_DEVICES=7 $0" >&2
  echo "[ERROR] Co-location with Qwen-Image on one H800: start reflect first, then image with CPU offload." >&2
  exit 2
fi

# Same CUDA 13 / nvcc bootstrap as train + Qwen-Image (FlashInfer JIT needs this).
_SP="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
_CUDA_HOME_CANDIDATE="${_SP}/nvidia/cu13"
if [[ -x "${_CUDA_HOME_CANDIDATE}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-${_CUDA_HOME_CANDIDATE}}"
  export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
_CUDA_COMPAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.cache/cuda-compat-13"
export LD_LIBRARY_PATH="${_CUDA_COMPAT_DIR}:${_SP}/nvidia/cu13/lib:${_SP}/nvidia/cuda_runtime/lib:${_SP}/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Prefer Triton attention when FlashInfer JIT would otherwise recompile.
export GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
# vLLM 0.24 defaults to FlashInfer sampler; JIT fails here (nvcc 13.2 vs cu13
# headers 13.3 + missing curand.h). Disable so EngineCore can start.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "[INFO] Frozen Qwen3-VL judge sidecar (vLLM, continuous batching)"
echo "[INFO]   model       : ${AGENTIC_REFLECT_VLM_PATH}"
echo "[INFO]   cuda devices: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO]   gpu mem util: ${REFLECT_GPU_MEM_UTIL} (max_num_seqs=${REFLECT_MAX_NUM_SEQS})"
echo "[INFO]   CUDA_HOME   : ${CUDA_HOME:-<unset>}"
echo "[INFO]   flashinfer sampler: ${VLLM_USE_FLASHINFER_SAMPLER}"
echo "[INFO]   listen      : http://${HOST}:${PORT}"
echo "[INFO] Trainer should export AGENTIC_VLLM_URL=http://${HOST}:${PORT}"
echo "[INFO] (Legacy custom endpoint: AGENTIC_REFLECT_VLM_URL=http://${HOST}:${PORT}/reflect)"
echo "[INFO] Colocate tip: start this first on GPU 7, then Qwen-Image with QWEN_IMAGE_ENABLE_CPU_OFFLOAD=1."

exec "${VLLM_BIN}" serve "${AGENTIC_REFLECT_VLM_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-num-seqs "${REFLECT_MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${REFLECT_GPU_MEM_UTIL}" \
  --max-model-len "${REFLECT_MAX_MODEL_LEN}" \
  --trust-remote-code \
  "$@"
