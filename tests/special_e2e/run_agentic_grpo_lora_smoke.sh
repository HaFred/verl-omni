#!/usr/bin/env bash
# System smoke for agentic GRPO LoRA.
#
# Mirrors examples/agenticllmgrpo_trainer/agent_llm/run_agenticrpco_grpo_lora.sh with
# smoke-scale knobs: mock overfit parquet, 1 train step, small rollout.n.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-2}"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/qwen/qwen-3.5-9b}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_DIR="${SMOKE_DIR:-${REPO_ROOT}/outputs/agentic_smoke/${RUN_TS}}"
DATA_DIR="${DATA_DIR:-${SMOKE_DIR}/data}"
CKPT_DIR="${CKPT_DIR:-${SMOKE_DIR}/ckpt}"
EXPERIMENT_NAME="agentic_grpo_smoke_${RUN_TS}"
AGENTIC_E2E_ROOT="${SMOKE_DIR}/e2e"
AGENTIC_E2E_RUN_NAME="${EXPERIMENT_NAME}"

# Short context for smoke; fewshot off to keep prompt/KV cheap.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESP_LEN="${MAX_RESP_LEN:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-8}"
MAX_USER_TURNS="${MAX_USER_TURNS:-8}"

mkdir -p "${SMOKE_DIR}" "${DATA_DIR}" "${CKPT_DIR}" "${AGENTIC_E2E_ROOT}"


if [[ -z "${AGENTIC_VLLM_OMNI_URL:-}" && -z "${AGENTIC_QWEN_IMAGE_URL:-}" && -z "${AGENTIC_DIFFUSION_TOOL_URL:-}" ]]; then
  echo "[ERROR] No frozen image service URL; start image-gen sidecar or set AGENTIC_VLLM_OMNI_URL." >&2
  exit 2
fi
if [[ -z "${AGENTIC_VLLM_URL:-}" ]]; then
  echo "[ERROR] AGENTIC_VLLM_URL unset; start judge sidecar." >&2
  exit 2
fi

# Mock overfit parquet in an isolated dir (does not touch data/agentic used by long e2e).
python3 "${REPO_ROOT}/examples/agenticllmgrpo_trainer/data_process/create_dummy_agentic_data.py" \
  --local_save_dir "${DATA_DIR}" \
  --overfit --train_size 2 --val_size 1 \
  --tool_call_format hermes \
  --model_path "${MODEL_PATH}"

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"

set -x
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=2 \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  data.filter_overlong_prompts=true \
  data.truncation=left \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding=true \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.layered_summon=false \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.agent.default_agent_loop=image_gen_tool_agent \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.omni_agent_loop.OmniAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_reward \
  reward.custom_reward_function.name=compute_score \
  trainer.val_before_train=false \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.total_epochs="${TOTAL_STEPS}" \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.logger='["console"]' \
  trainer.project_name=verl_omni_agentic_smoke \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "$@"

echo "[PASS] agentic GRPO LoRA smoke completed (TOTAL_STEPS=${TOTAL_STEPS})."
echo "[PASS] artifacts under ${AGENTIC_E2E_ROOT}/${EXPERIMENT_NAME}/"
