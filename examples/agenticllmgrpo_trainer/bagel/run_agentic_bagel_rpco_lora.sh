# Bagel UND+GEN Co-RL (PR1 correctness). Entry: python3 -m verl_omni.trainer.main_omni
#
# Fail-closed: UND must be the published Bagel checkpoint (Hermes tool-call).
# Do not swap Qwen3-VL for UND. Do not edit Mode (2a) run_agenticrpco_grpo_lora.sh.
#
# Required on vllm-omni >= 0.24: actor_rollout_ref.model.lora.merge=True.
# Fused MoT gen experts cannot bind *_moe_gen adapters (verl-omni#552 / vllm-omni#7190);
# merged full-weight sync is the same fix as run_bagel_ocr_lora.sh.
#
# UniCoT parquet must carry extra_info.reference_image_path. Rebuild with:
#   REBUILD_UNICOT=1 python3 examples/agenticllmgrpo_trainer/bagel/stamp_unicot_reference_paths.py \
#       --input $UNICOT_PARQUET --output $UNICOT_PARQUET
#
# GPUs: trainer.n_gpus_per_node = len(CUDA_VISIBLE_DEVICES) (e.g. 2,3,4,5 → 4).
# UND AR is colocated on that same actor placement group (not a second Ray pool).
# Default REWARD_TP=N so one Qwen RM is TP-sharded, not copied per GPU.
# 1-GPU smoke (CUDA_VISIBLE_DEVICES=3): ENABLE_RM=0 by default — actor+Omni+RM cannot fit.
# Re-enable with ENABLE_RM=1 once you have headroom.
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
# Prefer Slurm job GPUs 2-5 when the caller did not set CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

model_name=${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}
reward_model_name=${REWARD_MODEL:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc}

TRAIN_FILE=${UNICOT_TRAIN:-"${_REPO_ROOT}/outputs/data/agentic_unicot/train.parquet"}
VAL_FILE=${UNICOT_TEST:-"${_REPO_ROOT}/outputs/data/agentic_unicot/val.parquet"}
# Build the mixed UniCoT train/val parquet (system + user only; UniCoT fields
# are reward ground truth, never fewshot). Skip when both files already exist
# unless REBUILD_UNICOT=1 (avoids import-heavy rebuild on resume).
if [[ "${REBUILD_UNICOT:-1}" == "1" || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  python3 -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl \
      --breakdown_dir "$UNICOT_BREAKDOWN_DIR" \
      --reflection_dir "$UNICOT_REFLECTION_DIR" \
      --local_save_dir "$(dirname "$TRAIN_FILE")" \
      --mix_ratio "$UNICOT_MIX_RATIO" \
      --val_ratio "$UNICOT_VAL_RATIO" \
      --seed "$UNICOT_SPLIT_SEED" \
      ${UNICOT_TRAIN_SIZE:+--train_size "$UNICOT_TRAIN_SIZE"} \
      ${UNICOT_VAL_SIZE:+--val_size "$UNICOT_VAL_SIZE"}
else
  echo "[INFO] reusing existing UniCoT parquet: $TRAIN_FILE / $VAL_FILE (set REBUILD_UNICOT=1 to rebuild)"
fi

# Count cards Ray will actually see (CUDA_VISIBLE_DEVICES=2,3,4,5 → 4).
# All visible cards go to the actor+GEN hybrid pool. UND AR is colocated on that
# same placement group (init_colocated) — it does NOT need spare free Ray GPUs.
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
# Colocated UND AR footprint inside the actor PG (TP=1 smoke → 1 card share).
UND_N_GPUS=${UND_N_GPUS:-1}
BAGEL_UND_DEPLOY_CONFIG=${BAGEL_UND_DEPLOY_CONFIG:-"$(dirname "$0")/bagel_corl_deploy_ar.yaml"}
# Live Hermes proof (optional before long runs):
#   bash examples/agenticllmgrpo_trainer/bagel/run_bagel_und_ar_serve.sh
#   BAGEL_UND_URL=http://127.0.0.1:8094 python3 .../spike_und_hermes.py --model-path "$BAGEL_MODEL_PATH"
BAGEL_UND_AR_SERVING_READY=${BAGEL_UND_AR_SERVING_READY:-1}
echo "bagel_corl 4-device e2e: actor+GEN=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} UND_AR_colocated=${UND_N_GPUS} visible=${N_VISIBLE} und_ready=${BAGEL_UND_AR_SERVING_READY}"
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
# 1-GPU cannot colocate Bagel FSDP + vLLM-Omni + Qwen RM. Default: drop RM, tiny util.
# 2-GPU: actor + Omni + colocated UND AR is already tight — keep RM off.
# 4-GPU e2e: still default RM off until Hermes+composite step are green; ENABLE_RM=1 to re-enable.
if [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -eq 1 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.08}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.18}
  TRAIN_BSZ=${TRAIN_BSZ:-1}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
  GEN_STEPS=${GEN_STEPS:-4}
  GEN_HW=${GEN_HW:-256}
  LORA_RANK=${LORA_RANK:-4}
  LORA_ALPHA=${LORA_ALPHA:-8}
elif [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -le 2 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.12}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
  TRAIN_BSZ=${TRAIN_BSZ:-2}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
  GEN_STEPS=${GEN_STEPS:-10}
  GEN_HW=${GEN_HW:-512}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
else
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.15}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
  TRAIN_BSZ=${TRAIN_BSZ:-2}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
  GEN_STEPS=${GEN_STEPS:-10}
  GEN_HW=${GEN_HW:-512}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
fi
if [[ "$ENABLE_RM" != "1" ]]; then
  ENABLE_RM=0
fi

# Smoke defaults. N = sibling episodes (Token GRPO); S = seeds per generate_image (FlowGRPO).
# In-episode J (UND turns) and K (GEN calls) are runtime with J>=K — not these env vars.
# Production example: N=8 S=4. S must be >= 2.
# Do not inherit stale K=1 from an old shell (`S=${K:-2}` would still pick K=1).
if [[ -z "${N:-}" ]]; then
  if [[ -n "${J:-}" ]]; then N=$J; else N=2; fi
fi
if [[ -z "${S:-}" ]]; then
  if [[ -n "${K:-}" && "$K" -ge 2 ]]; then S=$K; else S=2; fi
fi
if [[ "$S" -lt 2 ]]; then
  echo "gen_samples_per_call S must be >= 2 (FlowGRPO seeds; got S=$S). Use S=2 or unset stale K." >&2
  exit 1
fi
if [[ "$N" -lt 1 ]]; then
  echo "rollout.n N must be >= 1 (got N=$N)." >&2
  exit 1
fi
echo "bagel_corl smoke: N=$N siblings, S=$S seeds/call (in-episode J/K are runtime)"

python3 -m verl_omni.trainer.main_omni \
    trainer.v1.trainer_mode=bagel_corl_sync \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_PROMPT_LEN \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name \
    +actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.model_type=diffusion_model \
    +actor_rollout_ref.model.composite_mode=bagel_corl \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora_dtype=bfloat16 \
    actor_rollout_ref.model.lora.merge=True \
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
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=$MAX_PROMPT_LEN \
    +actor_rollout_ref.rollout.pipeline.height=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.width=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.num_inference_steps=$GEN_STEPS \
    +actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.algo.noise_level=${NOISE_LEVEL:-0.7} \
    +actor_rollout_ref.rollout.algo.sde_window_size=${SDE_WINDOW_SIZE:-2} \
    +actor_rollout_ref.rollout.algo.sde_window_range=${SDE_WINDOW_RANGE:-[0,7]} \
    actor_rollout_ref.rollout.agent.default_agent_loop=bagel_multiturn_agent \
    +actor_rollout_ref.rollout.agent.gen_samples_per_call=$S \
    +actor_rollout_ref.rollout.agent.max_generate_passes=1 \
    +actor_rollout_ref.rollout.agent.und_ar_serving_ready=$BAGEL_UND_AR_SERVING_READY \
    +actor_rollout_ref.rollout.agent.und_deploy_config=$BAGEL_UND_DEPLOY_CONFIG \
    +actor_rollout_ref.rollout.agent.und_n_gpus=$UND_N_GPUS \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG \
    reward.num_workers=$(( ENABLE_RM == 1 ? NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP : 0 )) \
    reward.reward_model.enable=$ENABLE_RM \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.gpu_memory_utilization=$REWARD_GPU_MEM_UTIL \
    reward.reward_model.rollout.enforce_eager=True \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.logger="['console']" \
    trainer.project_name=bagel_corl \
    trainer.experiment_name=bagel_corl_pr1 \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1
