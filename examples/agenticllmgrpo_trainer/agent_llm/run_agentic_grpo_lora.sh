#!/usr/bin/env bash
# Agentic GRPO overfit.
#
# The actor is a pretrained tool-calling VLM (default: Qwen3.5-2B via MODEL_PATH).
# Frozen tools (vLLM-omni Qwen-Image + vLLM Qwen3-VL) serve generate_image and
# judge_image with continuous batching. The agent follows the gen→judge→reflect
# protocol: generate_image → judge_image(VL feedback) → Done / rewrite.
#
#   # pane A — vLLM-omni image gen server (GPUs 0,1):
#   CUDA_VISIBLE_DEVICES=0,1 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_qwen_image_tool_server.sh
#   # pane B — vLLM VL judge server (GPU 0):
#   CUDA_VISIBLE_DEVICES=0 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_qwen_vl_reflect_server.sh
#   # pane C — training (GPUs 2-3):
#   CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 TOTAL_STEPS=100 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_agentic_grpo_lora.sh
#
# File call list:
#   agent_llm/run_qwen_image_tool_server.sh   — frozen Qwen-Image via vLLM-Omni
#   agent_llm/run_qwen_vl_reflect_server.sh   — frozen Qwen3-VL judge via vLLM
#   data_process/create_dummy_agentic_data.py — overfit train/val parquet
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/agentic/train.parquet}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/data/agentic/val.parquet}"
N_GPUS="${N_GPUS:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
ROLLOUT_N="${ROLLOUT_N:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="agentic_grpo_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/verl_omni_agentic/${EXPERIMENT_NAME}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
# 3-pass protocol needs ~7 assistant msgs (gen/judge/rewrite…); leave margin.
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-12}"
MAX_USER_TURNS="${MAX_USER_TURNS:-12}"

echo "[INFO] wandb online experiment_name=${EXPERIMENT_NAME} (WANDB_SERVICE_TRANSPORT=${WANDB_SERVICE_TRANSPORT})"
echo "[INFO] ckpt dir=${CKPT_DIR}"
echo "[INFO] agent loop=agentic_tool_agent (AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=${AGENTIC_FORCE_REFLECTION_AFTER_JUDGE:-<unset>}; max generate_image passes=${AGENTIC_MAX_GENERATE_IMAGE_PASSES:-<unset>})"
echo "[INFO] force-first generate=${AGENTIC_FORCE_FIRST_GENERATE:-<unset>} warmup=${AGENTIC_FORCE_FIRST_WARMUP_STEPS:-<unset>} end=${AGENTIC_FORCE_FIRST_END_STEP:-<unset>}"
echo "[INFO] rewrite_judge_before_generate=${AGENTIC_REWRITE_JUDGE_BEFORE_GENERATE:-<unset>}"
echo "[INFO] good_enough threshold=${AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD:-<unset>} block_generate_after_yes=${AGENTIC_BLOCK_GENERATE_AFTER_YES:-<unset>} block_after_max_passes=${AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES:-<unset>} rollout_n=${ROLLOUT_N}"
echo "[INFO] agent MODEL_PATH=${MODEL_PATH}"
echo "[INFO] reflect judge vLLM URL=${AGENTIC_VLLM_URL:-<unset>} legacy=${AGENTIC_REFLECT_VLM_URL:-<unset>} path=${AGENTIC_REFLECT_VLM_PATH:-<unset>}"
if [[ -z "${AGENTIC_VLLM_OMNI_URL:-}" && -z "${AGENTIC_QWEN_IMAGE_URL:-}" && -z "${AGENTIC_DIFFUSION_TOOL_URL:-}" ]]; then
  echo "[ERROR] No frozen image service is configured; visual reflection cannot be trained on stubs." >&2
  echo "[ERROR] Start: CUDA_VISIBLE_DEVICES=<free_gpu> bash examples/agenticllmgrpo_trainer/agent_llm/run_qwen_image_tool_server.sh" >&2
  exit 2
fi
if [[ -z "${AGENTIC_VLLM_URL:-}" && -z "${AGENTIC_REFLECT_VLM_URL:-}" ]]; then
  echo "[ERROR] AGENTIC_VLLM_URL and AGENTIC_REFLECT_VLM_URL are both unset; judge_image has no backend." >&2
  echo "[ERROR] Start: CUDA_VISIBLE_DEVICES=<free_gpu> bash examples/agenticllmgrpo_trainer/agent_llm/run_qwen_vl_reflect_server.sh" >&2
  exit 2
fi

# Create tiny overfit dataset: Hermes-format with same-task fewshot.
python3 "${REPO_ROOT}/examples/agenticllmgrpo_trainer/data_process/create_dummy_agentic_data.py" \
    --local_save_dir "${REPO_ROOT}/data/agentic" \
    --overfit --train_size 8 --val_size 2 \
    --tool_call_format hermes \
    --model_path "$MODEL_PATH" \
    --with_fewshot

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=2 \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESP_LEN \
    data.filter_overlong_prompts=true \
    data.truncation=left \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=32 \
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
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.layered_summon=false \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.agent.default_agent_loop=agentic_tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_reward \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=$TOTAL_STEPS \
    trainer.test_freq=-1 \
    trainer.save_freq=5 \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_omni_agentic \
    trainer.experiment_name=$EXPERIMENT_NAME \
    "$@"
