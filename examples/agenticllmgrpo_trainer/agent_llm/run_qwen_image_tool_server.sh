#!/usr/bin/env bash
# Frozen Qwen-Image service for agentic GRPO (vLLM-Omni).
# Qwen-Image is a diffusion model. ``--omni`` makes the CLI detect model_index.json
# and serve POST /v1/images/generations.
set -x

MODEL=Qwen/Qwen-Image
HOST=127.0.0.1
PORT=8092
NUM_GPUS=1

exec vllm-omni serve "$MODEL" \
  --omni \
  --host "$HOST" \
  --port "$PORT" \
  --num-gpus "$NUM_GPUS" \
  --tensor-parallel-size "$NUM_GPUS" \
  --enable-cpu-offload \
  "$@"
