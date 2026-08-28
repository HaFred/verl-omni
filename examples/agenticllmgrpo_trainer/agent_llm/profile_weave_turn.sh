#!/usr/bin/env bash
# One-step agentic RPCO GRPO with W&B Weave rollout traces.
#
# Enables verl's ``actor_rollout_ref.rollout.trace.backend=weave`` so each
# multi-turn Hermes trajectory (ToolAgentLoop.run + per-turn generate /
# extract_tool_calls / tool execute) shows up under the W&B project → Weave →
# Traces. This is observability only; it does not change actor tensors.
#
# Prerequisites (same sidecars as the long RPCO run):
#   pane A: CUDA_VISIBLE_DEVICES=0,1 bash .../run_image_gen_tool_server.sh
#   pane B: CUDA_VISIBLE_DEVICES=<free GPU> bash .../run_judge_image_tool_server.sh
#   pane C: export WANDB_API_KEY=...   # required for Weave
#           CUDA_VISIBLE_DEVICES=6,7 N_GPUS=2 \
#             bash .../profile_weave_turn.sh
#
# Token2text for per-turn ``LLMServerClient.generate`` needs the #7204 decode
# path (input.prompt_ids + TokenOutput.token_ids + llm_client.tokenizer). That
# is patched into this env's site-packages verl — do **not** put a full newer
# ``fred/verl`` checkout on PYTHONPATH (PPO v1 TensorDict breaks verl_omni's
# AgenticMetricsAgentLoopManager, which expects DataProto.meta_info).
#
# Env knobs (optional):
#   TOTAL_STEPS                 training steps (default 1)
#   TRACE_TOKEN2TEXT            1|0 decode prompt/response text in traces (default 1)
#   TRACE_MAX_SAMPLES_PER_WORKER  unique prompts traced per agent worker per step
#                                 (default 2; null/empty = trace all — heavy)
#   TRAIN_BATCH_SIZE / ROLLOUT_N / N_GPUS / MODEL_PATH / … same as RPCO recipe
#
# View: W&B project ``verl_omni_agentic`` → Weave sidebar → Traces.
# Filter by step / sample_index / rollout_n / experiment_name.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
# Keep venv verl (DataProto trainer) + verl_omni. Strip any accidental fred/verl
# prefix from the shell so PPO v1 does not shadow the wheel.
_strip_fred_verl_from_pythonpath() {
  local out="" part
  IFS=':' read -r -a _pp_parts <<< "${PYTHONPATH:-}"
  for part in "${_pp_parts[@]}"; do
    [[ -z "${part}" ]] && continue
    if [[ "${part}" == */fred/verl || "${part}" == */fred/verl/ ]]; then
      echo "[WARN] removing ${part} from PYTHONPATH (incompatible with verl_omni DataProto path)" >&2
      continue
    fi
    out="${out:+${out}:}${part}"
  done
  export PYTHONPATH="${out}"
}
_strip_fred_verl_from_pythonpath
unset -f _strip_fred_verl_from_pythonpath
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERLOMNI_ROOT="${REPO_ROOT}"
export AGENTIC_JUDGE_RUBBER_STAMP_GUARD="${AGENTIC_JUDGE_RUBBER_STAMP_GUARD:-0}"

HF_HOME_DIR="${HF_HOME_DIR:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub}"
UNICOT_BREAKDOWN_DIR="${UNICOT_BREAKDOWN_DIR:-${HF_HOME_DIR}/datasets--Fr0zencr4nE--UniCoT-Breakdown-3K}"
UNICOT_REFLECTION_DIR="${UNICOT_REFLECTION_DIR:-${HF_HOME_DIR}/datasets--Fr0zencr4nE--UniCoT-Self-Reflection-6K}"
UNICOT_MIX_RATIO="${UNICOT_MIX_RATIO:-0.5}"
UNICOT_VAL_RATIO="${UNICOT_VAL_RATIO:-0.05}"
UNICOT_SPLIT_SEED="${UNICOT_SPLIT_SEED:-42}"
RPCO_INIT_CKPT="${RPCO_INIT_CKPT:-}"
MODEL_PATH="${RPCO_INIT_CKPT:-${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}}"
TRAIN_FILE="${REPO_ROOT}/outputs/data/agentic_unicot/train.parquet"
VAL_FILE="${REPO_ROOT}/outputs/data/agentic_unicot/val.parquet"

N_GPUS="${N_GPUS:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-$((N_GPUS / ROLLOUT_TP))}"
if (( N_GPUS % ROLLOUT_TP != 0 )); then
  echo "[ERROR] N_GPUS=${N_GPUS} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}" >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cuda_devs <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#_cuda_devs[@]} != N_GPUS )); then
    echo "[ERROR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has ${#_cuda_devs[@]} GPU(s) but N_GPUS=${N_GPUS}." >&2
    exit 2
  fi
  unset _cuda_devs
fi

# One-step Weave profile defaults (override freely).
TOTAL_STEPS="${TOTAL_STEPS:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRACE_TOKEN2TEXT="${TRACE_TOKEN2TEXT:-1}"
# Cap free-plan Weave volume: 2 unique prompts × workers × n rollouts.
TRACE_MAX_SAMPLES_PER_WORKER="${TRACE_MAX_SAMPLES_PER_WORKER:-2}"

PROJECT_NAME="${PROJECT_NAME:-verl_omni_agentic}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-weave_profiling_${RUN_TS}}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/verl_omni_agentic/${EXPERIMENT_NAME}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-16}"
MAX_USER_TURNS="${MAX_USER_TURNS:-16}"

export AGENTIC_E2E_ROOT="${AGENTIC_E2E_ROOT:-${REPO_ROOT}/outputs/weave_e2e}"
export AGENTIC_E2E_RUN_NAME="${EXPERIMENT_NAME}"
export AGENTIC_VAL_VIZ="${AGENTIC_VAL_VIZ:-0}"
export UNICOT_BREAKDOWN_DIR UNICOT_REFLECTION_DIR UNICOT_MIX_RATIO UNICOT_VAL_RATIO UNICOT_SPLIT_SEED

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

# ── Preflight ───────────────────────────────────────────────────────────────
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] WANDB_API_KEY is unset. Weave uses the W&B API key." >&2
  echo "        export WANDB_API_KEY=... before launching." >&2
  exit 2
fi

if ! python3 -c "import weave" >/dev/null 2>&1; then
  echo "[ERROR] Python package 'weave' is not importable in this env." >&2
  echo "        Install into the training env: pip install weave" >&2
  exit 2
fi
# Fail fast if the env verl still lacks #7204-style per-turn token2text.
_verl_trace_file="$(
  python3 - <<'PY'
import verl.utils.rollout_trace as rt
print(rt.__file__)
PY
)"
echo "[INFO] verl.utils.rollout_trace -> ${_verl_trace_file}"
if ! python3 -c "from verl.utils.rollout_trace import _add_token2text" >/dev/null 2>&1; then
  echo "[ERROR] Loaded verl lacks _add_token2text (need #7204-style decode)." >&2
  echo "        Expected a patched site-packages rollout_trace.py in this venv." >&2
  exit 2
fi
if [[ "${_verl_trace_file}" == */fred/verl/* ]]; then
  echo "[ERROR] Still loading fred/verl rollout_trace (${_verl_trace_file})." >&2
  echo "        Unset PYTHONPATH entries that point at fred/verl and retry." >&2
  exit 2
fi

if [[ -z "${AGENTIC_VLLM_OMNI_URL:-}" && -z "${AGENTIC_QWEN_IMAGE_URL:-}" && -z "${AGENTIC_DIFFUSION_TOOL_URL:-}" ]]; then
  echo "[ERROR] No frozen image service is configured (AGENTIC_VLLM_OMNI_URL / …)." >&2
  exit 2
fi
if [[ -z "${AGENTIC_VLLM_URL:-}" ]]; then
  echo "[ERROR] AGENTIC_VLLM_URL is unset; start the judge sidecar." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "[ERROR] UniCoT parquet missing ($TRAIN_FILE / $VAL_FILE)." >&2
  echo "        Build via run_agenticrpco_grpo_lora.sh (or REBUILD_UNICOT=1) first." >&2
  exit 2
fi

if [[ "${TRACE_TOKEN2TEXT}" == "1" || "${TRACE_TOKEN2TEXT}" == "true" || "${TRACE_TOKEN2TEXT}" == "True" ]]; then
  _TOKEN2TEXT=True
else
  _TOKEN2TEXT=False
fi

TRACE_OVERRIDES=(
  "actor_rollout_ref.rollout.trace.backend=weave"
  "actor_rollout_ref.rollout.trace.token2text=${_TOKEN2TEXT}"
)
if [[ -n "${TRACE_MAX_SAMPLES_PER_WORKER}" && "${TRACE_MAX_SAMPLES_PER_WORKER}" != "null" ]]; then
  TRACE_OVERRIDES+=(
    "actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=${TRACE_MAX_SAMPLES_PER_WORKER}"
  )
  _trace_cap_msg="max_samples_per_worker=${TRACE_MAX_SAMPLES_PER_WORKER}"
else
  _trace_cap_msg="max_samples_per_worker=all"
fi

echo "[INFO] weave profile: steps=${TOTAL_STEPS} token2text=${_TOKEN2TEXT} ${_trace_cap_msg}"
echo "[INFO] project=${PROJECT_NAME} experiment=${EXPERIMENT_NAME}"
echo "[INFO] After the step finishes: W&B → ${PROJECT_NAME} → Weave → Traces"
echo "[INFO]   filter tags: step / sample_index / rollout_n / experiment_name=${EXPERIMENT_NAME}"

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
    trainer.val_before_train=false \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=$TOTAL_STEPS \
    trainer.test_freq=0 \
    trainer.save_freq=-1 \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    "${TRACE_OVERRIDES[@]}" \
    "$@"
