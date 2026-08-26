#!/usr/bin/env bash
# Profile the agentic RPCO pipeline with two complementary layers.
#
# Layer 1 — offline GPU/kernel profiling (verl `global_profiler`):
#   torch.profiler / nsys / torch_memory capture the FSDP actor / ref / vLLM
#   rollout boundaries into a Chrome trace or nsys report. This sees the
#   CUDA-dense parts (update_actor / gen / weight transfer) but is blind to
#   the I/O-bound multi-turn sidecar loop.
#
# Layer 2 — online rl-insight observability (Prometheus + Tempo + Grafana):
#   `trace_state` swim lanes paint each rollout's `decode` -> `generate_image`
#   -> `judge_image` timeline, and `metric_histogram` publishes
#   `agentic_tool_{generate_image,judge_image}_latency_seconds` — surfacing the
#   sidecar bubbles that layer 1 cannot see. Instrumentation lives in
#   verl_omni/agent_loop/rl_insight_profiler.py (facade), agentic_tool_agent_loop.py
#   (decode lanes), tools.py (tool lanes + latency histograms), and
#   agentic_metrics_manager.py (init in driver + workers). All of it is a
#   no-op unless `rl-insight` is installed and RL_INSIGHT_SERVER_URL is set.
#
# Usage (four panes — gen sidecar, judge sidecar, rl-insight server, then this):
#   pane A: CUDA_VISIBLE_DEVICES=0,1 bash .../run_image_gen_tool_server.sh
#   pane B: CUDA_VISIBLE_DEVICES=7   bash .../run_judge_image_tool_server.sh
#   pane C: RL_INSIGHT_SERVER_ONLY=1 bash .../profile_agenticrpco_grpo.sh
#   pane D: CUDA_VISIBLE_DEVICES=4,5 N_GPUS=2 \
#             bash .../profile_agenticrpco_grpo.sh
# Defaults to 2 train steps, no val-before-train / test_freq (minimal profile).
#
# Env knobs:
#   PROFILER_TOOL          torch | nsys | torch_memory   (default torch)
#   PROFILER_STEPS         comma-separated step list     (default 1)
#   RL_INSIGHT             1|0  enable rl-insight layer  (default 1)
#   RL_INSIGHT_SERVER_URL  rl-insight server URL         (default http://127.0.0.1:18080)
#   RL_INSIGHT_SERVER_ONLY 1|0  only start/stop the stack, skip training
#   RL_INSIGHT_SERVER_STOP 1|0  stop the stack and exit
#
# See docs/perf/profiler.md for the verl profiler field reference and
# https://github.com/verl-project/rl-insight for the observability stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILER_TOOL="${PROFILER_TOOL:-torch}"
PROFILER_STEPS="${PROFILER_STEPS:-1}"
RL_INSIGHT="${RL_INSIGHT:-1}"
RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"

# Shared overrides. All keys already exist in the composed verl config, so use
# plain `key=value` (a `+key=value` append fails with "An item is already at ...").
COMMON=(
  "global_profiler.tool=${PROFILER_TOOL}"
  "global_profiler.steps=[${PROFILER_STEPS}]"
  "global_profiler.profile_continuous_steps=false"
  "global_profiler.save_path=./outputs/profile"
)

case "$PROFILER_TOOL" in
  torch)
    OVERRIDES=(
      "${COMMON[@]}"
      "actor_rollout_ref.actor.profiler.enable=True"
      "actor_rollout_ref.actor.profiler.all_ranks=True"
      "actor_rollout_ref.actor.profiler.tool=torch"
      "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda]"
      "actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False"
    )
    ;;
  nsys)
    OVERRIDES=(
      "${COMMON[@]}"
      "actor_rollout_ref.actor.profiler.enable=True"
      "actor_rollout_ref.actor.profiler.all_ranks=True"
      "actor_rollout_ref.actor.profiler.tool=nsys"
    )
    ;;
  torch_memory)
    OVERRIDES=(
      "${COMMON[@]}"
      "actor_rollout_ref.actor.profiler.enable=True"
      "actor_rollout_ref.actor.profiler.all_ranks=True"
      "actor_rollout_ref.actor.profiler.tool=torch_memory"
    )
    ;;
  *)
    echo "[ERROR] unknown PROFILER_TOOL=${PROFILER_TOOL} (want torch|nsys|torch_memory)" >&2
    exit 2
    ;;
esac

# ── Layer 2: rl-insight online observability ────────────────────────────────
# Start/stop the Prometheus + Tempo + Grafana stack and export the server URL so
# the training driver + agent-loop workers emit metrics/traces through the Ray
# monitor hub. The Python instrumentation (verl_omni/agent_loop/rl_insight_profiler.py)
# is a silent no-op when `rl-insight` is not importable or the URL is unset, so
# layer 2 can be disabled at any time without touching the training code.
if [[ "${RL_INSIGHT}" == "1" ]]; then
  if ! command -v rl-insight >/dev/null 2>&1; then
    echo "[ERROR] rl-insight CLI not found. Install it into this environment first:" >&2
    echo "          pip install rl-insight" >&2
    echo "        (set RL_INSIGHT=0 to skip layer 2 and run layer 1 only)" >&2
    exit 2
  fi

  # Make the server URL reachable from both the driver and Ray workers.
  export RL_INSIGHT_SERVER_URL

  if [[ "${RL_INSIGHT_SERVER_STOP:-0}" == "1" ]]; then
    echo "[rl-insight] stopping server stack..."
    rl-insight server stop
    exit 0
  fi

  # `install` is idempotent (skips when binaries exist); `start --detach` runs
  # Prometheus/Tempo/Grafana in the background and returns immediately.
  echo "[rl-insight] ensuring server dependencies (one-time download if missing)..."
  rl-insight server install
  echo "[rl-insight] starting server stack (detached) at ${RL_INSIGHT_SERVER_URL}..."
  rl-insight server start --detach

  if [[ "${RL_INSIGHT_SERVER_ONLY:-0}" == "1" ]]; then
    echo "[rl-insight] server-only mode: stack is up. Grafana: http://<host>:3000"
    exit 0
  fi

  # Guard: the training env must import rl_insight for the facade to activate.
  # (verl-omni runs in its own venv; the package is optional there.)
  if ! python3 -c "import rl_insight" >/dev/null 2>&1; then
    echo "[WARN] rl-insight is not importable in the training python." >&2
    echo "       Layer 2 will be a silent no-op; layer 1 (${PROFILER_TOOL}) still runs." >&2
    echo "       To enable layer 2: pip install rl-insight into the training env." >&2
  else
    echo "[rl-insight] layer 2 enabled: metrics -> Prometheus:9090, traces -> Tempo:3200, dashboards -> Grafana:3000"
  fi
else
  echo "[rl-insight] layer 2 disabled (RL_INSIGHT=0); running layer 1 only."
fi

# ── Minimal 2-step trainer (same hydra as run_agenticrpco_grpo_lora.sh) ─────
# Same GRPO LoRA recipe as the long run; only the *duration* is cut: 2 steps,
# skip val-before-train and intra-run test so profiling is actor/rollout/tools.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
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
TOTAL_STEPS="${TOTAL_STEPS:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="agentic_rpco_profile_${RUN_TS}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/verl_omni_agentic/${EXPERIMENT_NAME}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-16}"
MAX_USER_TURNS="${MAX_USER_TURNS:-16}"
export AGENTIC_E2E_ROOT="${AGENTIC_E2E_ROOT:-${REPO_ROOT}/outputs/profile_e2e}"
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

if [[ -z "${AGENTIC_VLLM_OMNI_URL:-}" && -z "${AGENTIC_QWEN_IMAGE_URL:-}" && -z "${AGENTIC_DIFFUSION_TOOL_URL:-}" ]]; then
  echo "[ERROR] No frozen image service is configured." >&2
  exit 2
fi
if [[ -z "${AGENTIC_VLLM_URL:-}" ]]; then
  echo "[ERROR] AGENTIC_VLLM_URL is unset; start the judge sidecar." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "[ERROR] UniCoT parquet missing ($TRAIN_FILE / $VAL_FILE). Build via run_agenticrpco_grpo_lora.sh first." >&2
  exit 2
fi
echo "[INFO] profile trainer: steps=${TOTAL_STEPS} val_before_train=false test_freq=0 experiment=${EXPERIMENT_NAME}"

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
    trainer.save_freq=5 \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_omni_agentic \
    trainer.experiment_name=$EXPERIMENT_NAME \
    "${OVERRIDES[@]}" \
    "$@"
