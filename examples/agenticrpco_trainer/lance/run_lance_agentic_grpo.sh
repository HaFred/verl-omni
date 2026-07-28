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

MODEL_PATH="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/models--bytedance-research--Lance/snapshots/7395315758865e6f56ab87ad06a88c7ac172f056/Lance_3B"

# Load verl_omni on the driver (rollout adapter + agent loop registration).
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"bytedance-research/Lance-3B"}
# Toy seeds for PR1 acceptance (create with tests/special_e2e/create_dummy_agentic_data.py).
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/agentic/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/agentic/val.parquet"}

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  echo "Missing toy agentic parquet files."
  echo "Generate them with:"
  echo "  python3 tests/special_e2e/create_dummy_agentic_data.py --local_save_dir \$(dirname \"${TRAIN_FILE}\")"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=lance_agentic_grpo \
    'hydra.run.dir=${hydra:runtime.cwd}/outputs/latest' \
    'hydra.sweep.dir=${hydra:runtime.cwd}/outputs/latest' \
    hydra.output_subdir=null \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    ++actor_rollout_ref.model.architecture=LanceForConditionalGeneration \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    "$@"
