#!/usr/bin/env bash
# Lance-3B Mode (2a) agentic GRPO — FlowGRPO-style launch (stock ppo_trainer + CLI).
#
# Diffusion remains frozen: ToolAgentLoop dispatches generate_image outside the
# actor optimizer. Launch from the verl-omni repo root.
#
# Machine-local env (CUDA compat LD_LIBRARY_PATH, Ray env propagation, GPU ids,
# WANDB, MODEL_PATH, NCCL_IB_DISABLE, VERL_USE_EXTERNAL_MODULES) should be set by
# the operator before launch — e.g. source a personal env file.
#
#   source ~/path/to/local_env.sh
#   MODEL_PATH=/path/to/Lance_3B_hf_und \
#     bash examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agentic_grpo_overrides.sh
source "${SCRIPT_DIR}/agentic_grpo_overrides.sh"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a prepared HF understanding export (see README)}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/agentic/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/agentic/val.parquet}"
N_GPUS="${N_GPUS:-2}"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    "${AGENTIC_GRPO_OVERRIDES[@]}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    "$@"
