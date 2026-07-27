#!/usr/bin/env bash
# Lance-3B Agentic GRPO training (FSDP + vLLM-Omni multi-turn rollout).
# Mode (2a): trains the agent LLM (Lance understanding path) to reason,
# rewrite prompts, and reflect, with the diffusion generation path frozen.
# Hardware: 4x H100 80GB.
#
# Recipe config lives in config/lance_agentic_grpo.yaml (inherits verl's
# ppo_trainer). Only volatile values (paths, GPU/node counts) are set here.
set -x

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Load verl_omni on the driver (rollout adapter + agent loop registration).
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"bytedance-research/Lance-3B"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/agentic/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/agentic/val.parquet"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=lance_agentic_grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.architecture=LanceForConditionalGeneration \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    "$@"
