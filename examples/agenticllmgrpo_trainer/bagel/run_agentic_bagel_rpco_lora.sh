# Bagel UND+GEN Co-RL (PR1 correctness). Entry: python3 -m verl_omni.trainer.main_omni
#
# Fail-closed: UND must be the published Bagel checkpoint (Hermes tool-call).
# Do not swap Qwen3-VL for UND. Do not edit Mode (2a) run_agenticrpco_grpo_lora.sh.
#
# UniCoT parquet must carry extra_info.reference_image_path. Rebuild with:
#   REBUILD_UNICOT=1 python3 examples/agenticllmgrpo_trainer/bagel/stamp_unicot_reference_paths.py \
#       --input $UNICOT_PARQUET --output $UNICOT_PARQUET
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

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-2}
ROLLOUT_TP=${ROLLOUT_TP:-1}
# Must be ≤ visible GPUs (CUDA_VISIBLE_DEVICES count). Colocated RM shares the pool.
REWARD_TP=${REWARD_TP:-1}
REWARD_ENGINE=${REWARD_ENGINE:-vllm}

# Tiny 2-GPU smoke defaults (override for production: J=8 K=4, larger LoRA).
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
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.logger="['console']" \
    trainer.project_name=bagel_corl \
    trainer.experiment_name=bagel_corl_pr1 \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1
