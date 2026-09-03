# Bagel UND+GEN Co-RL (PR1 correctness). Entry: python3 -m verl_omni.trainer.main_omni
#
# Fail-closed: UND must be the published Bagel checkpoint (Hermes tool-call).
# Do not swap Qwen3-VL for UND. Do not edit Mode (2a) run_agenticrpco_grpo_lora.sh.
#
# UniCoT parquet must carry extra_info.reference_image_path. Rebuild with:
#   REBUILD_UNICOT=1 python3 examples/agenticllmgrpo_trainer/bagel/stamp_unicot_reference_paths.py \
#       --input $UNICOT_PARQUET --output $UNICOT_PARQUET
#
# GPUs: trainer.n_gpus_per_node follows CUDA_VISIBLE_DEVICES (e.g. 2,3,4,5 → 4).
# Default REWARD_TP=N so one Qwen RM is TP-sharded, not copied per GPU.
set -x
# Prefer local verl checkout (tokenizer package layout) over site-packages flat tokenizer.py.
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
_LOCAL_VERL="${_LOCAL_VERL:-${_REPO_ROOT}/../verl}"
if [[ -d "${_LOCAL_VERL}/verl" ]]; then
  export PYTHONPATH="${_LOCAL_VERL}:${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

export BAGEL_MODEL_PATH=/scratch/fq9hpsac/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b
WORKSPACE=${WORKSPACE:-$HOME}
BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-"$(dirname "$0")/bagel_corl_deploy.yaml"}

model_name=${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}
unicot_train_path=${UNICOT_TRAIN:-"${_REPO_ROOT}/outputs/data/agentic_unicot/train.parquet"}
unicot_test_path=${UNICOT_TEST:-"${_REPO_ROOT}/outputs/data/agentic_unicot/val.parquet"}
reward_model_name=${REWARD_MODEL:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc}

# Count cards Ray will actually see (CUDA_VISIBLE_DEVICES=2,3,4,5 → 4).
_count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local n=0 id
    IFS=',' read -ra _ids <<< "${CUDA_VISIBLE_DEVICES}"
    for id in "${_ids[@]}"; do
      id="${id#"${id%%[![:space:]]*}"}"
      id="${id%"${id##*[![:space:]]}"}"
      [[ -n "$id" ]] && n=$((n + 1))
    done
    echo "$n"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
    return
  fi
  echo 1
}
N_VISIBLE="$(_count_visible_gpus)"
N_VISIBLE="${N_VISIBLE//[[:space:]]/}"
if [[ -z "$N_VISIBLE" || "$N_VISIBLE" -lt 1 ]]; then
  echo "Need at least 1 visible GPU (got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})." >&2
  exit 1
fi
NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-$N_VISIBLE}
ROLLOUT_TP=${ROLLOUT_TP:-1}
# One colocated RM tensor-parallel across the pool (not N copies of Qwen). Override REWARD_TP=1 for per-GPU workers.
REWARD_TP=${REWARD_TP:-$NUM_GPUS_ACTOR_ROLLOUT_REWARD}
REWARD_ENGINE=${REWARD_ENGINE:-vllm}
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % ROLLOUT_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}." >&2
  exit 1
fi
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % REWARD_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by REWARD_TP=${REWARD_TP}." >&2
  exit 1
fi
# Tight util when few cards (2-GPU OOM: actor FSDP + vLLM-Omni + RM). More GPUs → slightly more KV.
if [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -le 2 ]]; then
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.12}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
else
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.15}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.40}
fi

# Tiny smoke defaults (override for production: J=8 K=4, larger LoRA). J must stay 2K.
J=${J:-2}
K=${K:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
TRAIN_BSZ=${TRAIN_BSZ:-2}

python3 -m verl_omni.trainer.main_omni \
    trainer.v1.trainer_mode=bagel_corl_sync \
    data.train_files=$unicot_train_path \
    data.val_files=$unicot_test_path \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=1024 \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name \
    +actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.model_type=diffusion_model \
    +actor_rollout_ref.model.composite_mode=bagel_corl \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora_dtype=bfloat16 \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj','q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=$J \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=bagel_multiturn_agent \
    +actor_rollout_ref.rollout.agent.gen_samples_per_call=$K \
    +actor_rollout_ref.rollout.agent.max_generate_passes=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.gpu_memory_utilization=$REWARD_GPU_MEM_UTIL \
    reward.reward_model.rollout.enforce_eager=True \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.logger="['console']" \
    trainer.project_name=bagel_corl \
    trainer.experiment_name=bagel_corl_pr1 \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1
