#!/usr/bin/env bash
# Frozen image-gen service for agentic GRPO (vLLM-Omni).
# Requires sourced IMAGE_GEN_MODEL (see fred_verlomni_agentic_multiturn_pr1.sh).
# ``--omni`` makes the CLI detect model_index.json and serve
# POST /v1/images/generations.
set -x

MODEL="${IMAGE_GEN_MODEL:-Qwen/Qwen-Image}"
HOST=127.0.0.1
PORT=8092
NUM_GPUS="${QWEN_IMAGE_NUM_GPUS:-2}"

echo "[INFO] image gen MODEL=${IMAGE_GEN_MODEL}"

exec vllm-omni serve "${MODEL}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --tensor-parallel-size "${NUM_GPUS}" \
  --enable-cpu-offload \
  "$@"
