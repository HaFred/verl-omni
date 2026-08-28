#!/usr/bin/env bash
# RPCO stage 3 — multi-task RL co-optimization on UniCoT (PR 2 of RFC #302).
#
# Mixed single-image (reflect) + multi-image (plan) GRPO with the
# multi-dimensional reward set {reflection, plan, format, tool, result}
# (VisionCreator-R1, arXiv:2603.08812). UniCoT fields are reward ground truth
# only; the frozen gen/judge sidecars and the agentic loop are PR 1's.
#
#   # pane A — image gen server (GPUs 0,1):
#   CUDA_VISIBLE_DEVICES=0,1 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_image_gen_tool_server.sh
#   # pane B — image judge server (GPU 0):
#   CUDA_VISIBLE_DEVICES=0 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_judge_image_tool_server.sh
#   # pane C — training on GPUs that are NOT serving gen/judge (this box: 4,5):
#   CUDA_VISIBLE_DEVICES=4,5 N_GPUS=2 TOTAL_STEPS=200 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_agenticrpco_grpo_lora.sh
#   CUDA_VISIBLE_DEVICES=2,3,4,5 N_GPUS=4 TOTAL_STEPS=200 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_agenticrpco_grpo_lora.sh
#
# Optional stage-1 init (strong-reflection checkpoint from a prior run):
#   RPCO_INIT_CKPT=/path/to/stage1_ckpt \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_agentic_rpco.sh
#
# File call list:
#   agent_llm/run_image_gen_tool_server.sh   — frozen image gen (default Qwen-Image / vLLM-Omni)
#   agent_llm/run_judge_image_tool_server.sh      — frozen image judge (default Qwen3.5 / vLLM)
#   python -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl — UniCoT → agentic RL parquet (RPCO stage 3)
set -x
# A failed data build must abort the run instead of training on a stale/empty
# parquet ("Train dataloader is empty!").
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Pin verl_omni resolution to this checkout: a sourced env script may point
# PYTHONPATH / VERLOMNI_ROOT at a sibling repo without the PR 2 modules.
# Prefix CUDA_VISIBLE_DEVICES on the launch line: sourcing the operator env
# hard-assigns CUDA_VISIBLE_DEVICES=4,5 for the 2-GPU recipe.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERLOMNI_ROOT="${REPO_ROOT}"
# Qwen3.5-9B's detailed findings are useful, but it frequently emits identical
# high facet scores. In this RPCO run the old flat-high guard turned many
# "fully satisfies / no fixes" cases into NO and corrupted the terminal signal.
# Set to 1 to restore score capping + forced-NO rubber-stamp protection.
export AGENTIC_JUDGE_RUBBER_STAMP_GUARD="${AGENTIC_JUDGE_RUBBER_STAMP_GUARD:-0}"
HF_HOME_DIR="${HF_HOME_DIR:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub}"
UNICOT_BREAKDOWN_DIR="${UNICOT_BREAKDOWN_DIR:-${HF_HOME_DIR}/datasets--Fr0zencr4nE--UniCoT-Breakdown-3K}"
UNICOT_REFLECTION_DIR="${UNICOT_REFLECTION_DIR:-${HF_HOME_DIR}/datasets--Fr0zencr4nE--UniCoT-Self-Reflection-6K}"
UNICOT_MIX_RATIO="${UNICOT_MIX_RATIO:-0.5}"
UNICOT_VAL_RATIO="${UNICOT_VAL_RATIO:-0.05}"
UNICOT_SPLIT_SEED="${UNICOT_SPLIT_SEED:-42}"
# A stage-1 strong-reflection checkpoint (when set) initializes stage 3.
RPCO_INIT_CKPT="${RPCO_INIT_CKPT:-}"
MODEL_PATH="${RPCO_INIT_CKPT:-${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}}"
# UniCoT agentic RL parquet only (built below from HF UniCoT snapshots).
TRAIN_FILE="${REPO_ROOT}/outputs/data/agentic_unicot/train.parquet"
VAL_FILE="${REPO_ROOT}/outputs/data/agentic_unicot/val.parquet"
N_GPUS="${N_GPUS:-2}"
# Actor vLLM TP; default 1 (one replica per trainer GPU). Agent-loop workers
# follow N_GPUS / ROLLOUT_TP unless AGENT_NUM_WORKERS is set.
ROLLOUT_TP="${ROLLOUT_TP:-1}"
if (( N_GPUS % ROLLOUT_TP != 0 )); then
  echo "[ERROR] N_GPUS=${N_GPUS} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}" >&2
  exit 2
fi
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-$((N_GPUS / ROLLOUT_TP))}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cuda_devs <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#_cuda_devs[@]} != N_GPUS )); then
    echo "[ERROR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has ${#_cuda_devs[@]} GPU(s) but N_GPUS=${N_GPUS}." >&2
    echo "[ERROR] Prefix the launch line (sourcing the operator env sets CUDA_VISIBLE_DEVICES=4,5)." >&2
    exit 2
  fi
  unset _cuda_devs
fi
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 || PPO_MINI_BATCH_SIZE % N_GPUS != 0 )); then
  echo "[ERROR] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} and PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} must be divisible by N_GPUS=${N_GPUS}" >&2
  exit 2
fi
TOTAL_STEPS="${TOTAL_STEPS:-200}"
ROLLOUT_N="${ROLLOUT_N:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="agentic_rpco_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/verl_omni_agentic/${EXPERIMENT_NAME}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
# 3-subtask plan + final judge + stop needs ~10 assistant msgs; leave margin.
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-16}"
MAX_USER_TURNS="${MAX_USER_TURNS:-16}"

# Co-locate images with traj/hermes under outputs/e2e/<experiment>/ (not /tmp).
export AGENTIC_E2E_ROOT="${AGENTIC_E2E_ROOT:-${REPO_ROOT}/outputs/e2e}"
export AGENTIC_E2E_RUN_NAME="${EXPERIMENT_NAME}"
# Val viz order (per validate step): cafe 9001/9002 + CN poster 9003/9004 first →
# commit W&B ``val/generations(_plan)`` and ``val/generations(_plan)_cn``
# (FlowGRPO-style commit=True at exact global_steps; Ray worker must flush) +
# dual-write ``run.summary`` (Media prev/next) → UniCoT val set (~250) →
# trainer Tracking.log ``val-core`` at the same step.
# Provider: ``verl_omni.utils.agentic_val_viz``; tables:
# ``AgenticValidationGenerationsLogger``. Do not bump past tip mid-val or
# ``val-core`` at N is dropped when tip is already N+1.
export AGENTIC_VAL_VIZ="${AGENTIC_VAL_VIZ:-1}"
export UNICOT_BREAKDOWN_DIR UNICOT_REFLECTION_DIR UNICOT_MIX_RATIO UNICOT_VAL_RATIO UNICOT_SPLIT_SEED
# Per-dimension weights. Default ``w_reflect=1.5`` so last-image C/A outranks
# format/tool presence. RPCO_W_* env vars override parquet-baked ``w_*``.
if [[ -z "${RPCO_W_TOOL_CALL:-}" && -n "${RPCO_W_TOOL:-}" ]]; then
  export RPCO_W_TOOL_CALL="${RPCO_W_TOOL}"
fi
for _dim in REFLECT PLAN FORMAT TOOL_CALL RESULT; do
  _key="RPCO_W_${_dim}"
  if [[ -z "${!_key:-}" ]]; then
    if [[ "${_dim}" == "REFLECT" ]]; then
      export "${_key}=1.5"
    else
      export "${_key}=1.0"
    fi
  fi
done

echo "[INFO] wandb online experiment_name=${EXPERIMENT_NAME} (WANDB_SERVICE_TRANSPORT=${WANDB_SERVICE_TRANSPORT})"
echo "[INFO] ckpt dir=${CKPT_DIR}"
echo "[INFO] rpco stage-3: full dataset (val_ratio=${UNICOT_VAL_RATIO}; sizes only for smoke: train=${UNICOT_TRAIN_SIZE:-<unset>} val=${UNICOT_VAL_SIZE:-<unset>} mix=${UNICOT_MIX_RATIO}) seed=${UNICOT_SPLIT_SEED} weights=${RPCO_W_REFLECT}/${RPCO_W_PLAN}/${RPCO_W_FORMAT}/${RPCO_W_TOOL_CALL}/${RPCO_W_RESULT}"
echo "[INFO] init ckpt=${RPCO_INIT_CKPT:-<cold start>} resume_mode=${RESUME_MODE:-disable}"
echo "[INFO] agent loop=agentic_tool_agent (AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=${AGENTIC_FORCE_REFLECTION_AFTER_JUDGE:-<unset>}; max generate_image passes=${AGENTIC_MAX_GENERATE_IMAGE_PASSES:-<unset>})"
echo "[INFO] force-first generate=${AGENTIC_FORCE_FIRST_GENERATE:-<unset>} warmup=${AGENTIC_FORCE_FIRST_WARMUP_STEPS:-<unset>} end=${AGENTIC_FORCE_FIRST_END_STEP:-<unset>}"
echo "[INFO] n_gpus=${N_GPUS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>} rollout_tp=${ROLLOUT_TP} agent_num_workers=${AGENT_NUM_WORKERS} train_batch=${TRAIN_BATCH_SIZE} ppo_mini_batch=${PPO_MINI_BATCH_SIZE} gpu_mem_util=${GPU_MEM_UTIL}"
echo "[INFO] agent MODEL_PATH=${MODEL_PATH}"
echo "[INFO] image judge vLLM URL=${AGENTIC_VLLM_URL:-<unset>} model=${JUDGE_IMAGE_MODEL:-<unset>}"
if [[ -z "${AGENTIC_VLLM_OMNI_URL:-}" && -z "${AGENTIC_QWEN_IMAGE_URL:-}" && -z "${AGENTIC_DIFFUSION_TOOL_URL:-}" ]]; then
  echo "[ERROR] No frozen image service is configured; visual reflection cannot be trained on stubs." >&2
  echo "[ERROR] Start: CUDA_VISIBLE_DEVICES=<free_gpu> bash examples/agenticllmgrpo_trainer/agent_llm/run_image_gen_tool_server.sh" >&2
  exit 2
fi
if [[ -z "${AGENTIC_VLLM_URL:-}" ]]; then
  echo "[ERROR] AGENTIC_VLLM_URL is unset; judge_image requires the vLLM OpenAI sidecar." >&2
  echo "[ERROR] Start: CUDA_VISIBLE_DEVICES=<free_gpu> bash examples/agenticllmgrpo_trainer/agent_llm/run_judge_image_tool_server.sh" >&2
  exit 2
fi

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

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESP_LEN \
    data.filter_overlong_prompts=true \
    data.truncation=left \
    data.return_raw_chat=true \
    data.seed=$UNICOT_SPLIT_SEED \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.layered_summon=false \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N:-1} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.agent.default_agent_loop=agentic_tool_agent \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager.AgenticMetricsAgentLoopManager \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_multidim_reward \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-true} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=$TOTAL_STEPS \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.save_freq=5 \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_omni_agentic \
    trainer.experiment_name=$EXPERIMENT_NAME \
    "$@"
