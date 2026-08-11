#!/usr/bin/env bash
# Frozen Qwen-Image service for agentic GRPO (vLLM-Omni).
#
#   source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh
#   CUDA_VISIBLE_DEVICES=7 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#
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
