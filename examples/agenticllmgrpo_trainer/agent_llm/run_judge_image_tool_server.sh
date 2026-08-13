#!/usr/bin/env bash
# Frozen image-judge sidecar for agentic GRPO (vLLM continuous batching).
# Requires sourced JUDGE_IMAGE_MODEL (see fred_verlomni_agentic_multiturn_pr1.sh).
# Per-request C/A score lines go to this process stdout via
# judge_image_log_middleware.py.
#
# Qwen3.5 GDN: on H800, vLLM ``auto`` selects FlashInfer JIT which often fails
# here. Pass ``--gdn-prefill-backend triton`` (env GDN_PREFILL_BACKEND alone is
# NOT read by vLLM serve).
set -x

MODEL="${JUDGE_IMAGE_MODEL:?JUDGE_IMAGE_MODEL is unset; source the operator env first}"
HOST=127.0.0.1
PORT=8093
MAX_NUM_SEQS="${AGENTIC_REFLECT_MAX_NUM_SEQS:-2}"
GPU_MEM_UTIL="${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.32}"
MAX_MODEL_LEN="${AGENTIC_REFLECT_MAX_MODEL_LEN:-4096}"
GDN_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "[INFO] image judge MODEL=${MODEL}"
echo "[INFO] gdn-prefill-backend=${GDN_BACKEND} GPU_MEM_UTIL=${GPU_MEM_UTIL}"

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gdn-prefill-backend "${GDN_BACKEND}" \
  --trust-remote-code \
  --middleware judge_image_log_middleware.judge_score_log_middleware \
  "$@"
