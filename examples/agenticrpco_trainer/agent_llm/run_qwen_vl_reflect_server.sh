#!/usr/bin/env bash
# Frozen Qwen3-VL judge sidecar for agentic GRPO (vLLM continuous batching).
#
#   source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh
#   CUDA_VISIBLE_DEVICES=7 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
#
# Per-request C/A score lines go to this process stdout via
# qwen_vl_judge_log_middleware.py.
set -x

MODEL=Qwen/Qwen3-VL-2B-Instruct
HOST=127.0.0.1
PORT=8093
MAX_NUM_SEQS=4
GPU_MEM_UTIL=0.12
MAX_MODEL_LEN=4096

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --middleware qwen_vl_judge_log_middleware.judge_score_log_middleware \
  "$@"
