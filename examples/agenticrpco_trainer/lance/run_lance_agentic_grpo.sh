#!/usr/bin/env bash
# Lance-3B Mode (2a) agentic GRPO — FlowGRPO-style launch (stock ppo_trainer + CLI).
#
# Diffusion remains frozen: ToolAgentLoop dispatches generate_image outside the
# actor optimizer. For real Lance MoT images, start the frozen tool server on a
# *free* GPU (not the GRPO GPUs), then set AGENTIC_LANCE_SERVER_URL (see README).
# Without it the tool returns a text stub — still valid for multi-turn if the
# agent emits Hermes <tool_call>s.
#
#   source ~/path/to/local_env.sh
#   # pane A: CUDA_VISIBLE_DEVICES=0 bash examples/.../run_lance_frozen_diffusion_tool_server.sh
#   # pane B:
#   MODEL_PATH=/path/to/Lance_3B_hf_und \
#     AGENTIC_LANCE_SERVER_URL=http://127.0.0.1:8091 \
#     TOTAL_STEPS=10 \
#     bash examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh
#
# Short smokes (TOTAL_STEPS≤20) auto-enable GATE_SIDECAR=1: prints expected
# behavior, watches rollout_trajectories during train, then writes
# overfit_gates.json (exit 2 on gate fail). Disable with GATE_SIDECAR=0.
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a prepared HF understanding export (see README)}"
TRAIN_FILE="${TRAIN_FILE:-data/agentic/train.parquet}"
VAL_FILE="${VAL_FILE:-data/agentic/val.parquet}"
N_GPUS="${N_GPUS:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
TOOL_CHAT_TEMPLATE="${TOOL_CHAT_TEMPLATE:-$SCRIPT_DIR/qwen2_tool_chat_template.jinja2}"
if [[ ! -f "$TOOL_CHAT_TEMPLATE" && -f "$SCRIPT_DIR/qwen2_tool_chat_template.yaml" ]]; then
  TOOL_CHAT_TEMPLATE="$SCRIPT_DIR/qwen2_tool_chat_template.yaml"
fi
# Unique WandB run + ckpt dir every launch (override with EXPERIMENT_NAME / RUN_TS).
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lance_agentic_grpo_${RUN_TS}}"
CKPT_DIR="${CKPT_DIR:-$PWD/checkpoints/verl_omni_agentic/${EXPERIMENT_NAME}}"
# Per-run rollout artifacts: PNGs (or stub manifests) land here when generate_image fires.
E2E_ROOT="${E2E_ROOT:-$PWD/outputs/e2e}"
E2E_RUN_DIR="${E2E_RUN_DIR:-$E2E_ROOT/${EXPERIMENT_NAME}}"
export AGENTIC_E2E_ROOT="${AGENTIC_E2E_ROOT:-$E2E_ROOT}"
export AGENTIC_E2E_RUN_NAME="${AGENTIC_E2E_RUN_NAME:-$EXPERIMENT_NAME}"
export AGENTIC_DIFFUSION_IMAGE_DIR="${AGENTIC_DIFFUSION_IMAGE_DIR:-$E2E_RUN_DIR/rollout_images}"
# Force / teacher / reflection curriculum (defaults for cold Lance und overfit).
# Override in the operator env. Keep force on so GRPO sees real 2-call trajectories;
# agentic_reward then ranks Reflection: + rewritten 2nd generate_image.
export AGENTIC_FORCE_GENERATE_IMAGE="${AGENTIC_FORCE_GENERATE_IMAGE:-1}"
export AGENTIC_FORCE_MIN_TOOL_CALLS="${AGENTIC_FORCE_MIN_TOOL_CALLS:-2}"
export AGENTIC_FORCE_PROB="${AGENTIC_FORCE_PROB:-1.0}"
export AGENTIC_TEACHER_FORCE_HERMES="${AGENTIC_TEACHER_FORCE_HERMES:-1}"
# Stable overfit teacher: fixed gen→reflect→rewrite targets (not wrap_caption of garbage).
# Keep ~30% forced turns on-policy (no token replace) so GRPO groups keep reward contrast.
export AGENTIC_OVERFIT_STABLE_TEACHER="${AGENTIC_OVERFIT_STABLE_TEACHER:-1}"
export AGENTIC_OVERFIT_ON_POLICY_FRAC="${AGENTIC_OVERFIT_ON_POLICY_FRAC:-0.30}"
export AGENTIC_PREFER_LLM_REFLECTION="${AGENTIC_PREFER_LLM_REFLECTION:-1}"
FORCE_AGENT_LOOP_CFG="${FORCE_AGENT_LOOP_CFG:-$SCRIPT_DIR/agentic_force_tool_agent_loop.yaml}"
GATE_SCRIPT="${GATE_SCRIPT:-$SCRIPT_DIR/check_overfit_gates.py}"
# Sidecar gates for short overfit smokes (TOTAL_STEPS≤20 default on). Override GATE_SIDECAR=0 to skip.
if [[ -z "${GATE_SIDECAR:-}" ]]; then
  if [[ "${TOTAL_STEPS}" -le 20 ]]; then
    GATE_SIDECAR=1
  else
    GATE_SIDECAR=0
  fi
fi
GATE_PID=""

# Online WandB + local rollout_images. TCP transport avoids Ray/uvloop Unix socket
# crashes (UnixTransport closed). WANDB_DIR keeps run files under this e2e folder.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SERVICE_TRANSPORT="${WANDB_SERVICE_TRANSPORT:-tcp}"
export WANDB_DIR="${WANDB_DIR:-$E2E_RUN_DIR/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_omni_agentic}"
mkdir -p "$AGENTIC_DIFFUSION_IMAGE_DIR" "$E2E_RUN_DIR" "$WANDB_DIR"
cat >"$E2E_RUN_DIR/README.txt" <<EOF
e2e run: ${EXPERIMENT_NAME}
wandb: mode=${WANDB_MODE} project=${WANDB_PROJECT} experiment=${EXPERIMENT_NAME}
wandb dir: ${WANDB_DIR}
ckpt: ${CKPT_DIR}
rollout images: ${AGENTIC_DIFFUSION_IMAGE_DIR}
rollout trajectories: ${E2E_RUN_DIR}/rollout_trajectories
hermes actions (per step): ${E2E_RUN_DIR}/hermes_actions/step_XXXXXX.txt
Lance tool URL: ${AGENTIC_LANCE_SERVER_URL:-<unset — stub only, no real PNGs>}
overfit gates: GATE_SIDECAR=${GATE_SIDECAR} (see overfit_gates.json)
EOF
echo "[INFO] wandb online experiment_name=${EXPERIMENT_NAME} (WANDB_SERVICE_TRANSPORT=${WANDB_SERVICE_TRANSPORT})"
echo "[INFO] ckpt dir=${CKPT_DIR}"
echo "[INFO] e2e rollout images -> ${AGENTIC_DIFFUSION_IMAGE_DIR}"
echo "[INFO] e2e full trajectories -> ${E2E_RUN_DIR}/rollout_trajectories"
echo "[INFO] e2e hermes actions -> ${E2E_RUN_DIR}/hermes_actions/"
echo "[INFO] force/teacher env: FORCE_GENERATE_IMAGE=${AGENTIC_FORCE_GENERATE_IMAGE:-<unset>} MIN_TOOL_CALLS=${AGENTIC_FORCE_MIN_TOOL_CALLS:-<unset>} TEACHER_FORCE_HERMES=${AGENTIC_TEACHER_FORCE_HERMES:-<unset>} OVERFIT_STABLE_TEACHER=${AGENTIC_OVERFIT_STABLE_TEACHER:-<unset>} ON_POLICY_FRAC=${AGENTIC_OVERFIT_ON_POLICY_FRAC:-<unset>} PREFER_LLM_REFLECTION=${AGENTIC_PREFER_LLM_REFLECTION:-<unset>} (agent_loop=agentic_force_tool_agent)"
python3 "$GATE_SCRIPT" --run-dir "$E2E_RUN_DIR" --expect-only --total-steps "${TOTAL_STEPS}"
if [[ -z "${AGENTIC_LANCE_SERVER_URL:-}" && -z "${AGENTIC_DIFFUSION_TOOL_URL:-}" ]]; then
  echo "[WARN] No AGENTIC_LANCE_SERVER_URL set — generate_image will write STUB_NO_IMAGE.txt only."
  echo "[WARN] Start: CUDA_VISIBLE_DEVICES=0 bash examples/agenticrpco_trainer/lance/run_lance_frozen_diffusion_tool_server.sh"
fi

_run_final_gates() {
  local train_rc="${1:-0}"
  if [[ -n "${GATE_PID}" ]] && kill -0 "${GATE_PID}" 2>/dev/null; then
    kill "${GATE_PID}" 2>/dev/null || true
    wait "${GATE_PID}" 2>/dev/null || true
  fi
  if [[ "${GATE_SIDECAR}" != "1" ]]; then
    return 0
  fi
  echo "[GATE] running final overfit gates on ${E2E_RUN_DIR}"
  if python3 "$GATE_SCRIPT" --run-dir "$E2E_RUN_DIR" --final --total-steps "${TOTAL_STEPS}"; then
    echo "[GATE] final gates PASS (train_rc=${train_rc})"
    return 0
  else
    echo "[GATE] final gates FAIL (train_rc=${train_rc})" >&2
    return 2
  fi
}

# Refresh Ray worker env so tool artifacts / Lance URL / WandB reach TaskRunner.
export RAY_RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json, os
keys = [
    "LD_LIBRARY_PATH",
    "VERL_USE_EXTERNAL_MODULES",
    "WANDB_API_KEY",
    "WANDB_MODE",
    "WANDB_SERVICE_TRANSPORT",
    "WANDB_SILENT",
    "WANDB_DIR",
    "WANDB_PROJECT",
    "AGENTIC_LANCE_SERVER_URL",
    "AGENTIC_DIFFUSION_TOOL_URL",
    "AGENTIC_DIFFUSION_TOOL_TOKEN",
    "AGENTIC_DIFFUSION_TOOL_TIMEOUT",
    "AGENTIC_DIFFUSION_IMAGE_DIR",
    "AGENTIC_DIFFUSION_ATTACH_IMAGE",
    "AGENTIC_FORCE_GENERATE_IMAGE",
    "AGENTIC_FORCE_MIN_TOOL_CALLS",
    "AGENTIC_FORCE_PROB",
    "AGENTIC_TEACHER_FORCE_HERMES",
    "AGENTIC_PREFER_LLM_REFLECTION",
    "AGENTIC_OVERFIT_STABLE_TEACHER",
    "AGENTIC_OVERFIT_ON_POLICY_FRAC",
    "AGENTIC_E2E_ROOT",
    "AGENTIC_E2E_RUN_NAME",
    "AGENTIC_LANCE_HEIGHT",
    "AGENTIC_LANCE_WIDTH",
    "AGENTIC_LANCE_STEPS",
    "AGENTIC_LANCE_SEED",
    "AGENTIC_LANCE_CFG_TEXT_SCALE",
]
base = {}
try:
    base = json.loads(os.environ.get("RAY_RUNTIME_ENV_JSON") or "{}")
except Exception:
    base = {}
env_vars = dict(base.get("env_vars") or {})
for k in keys:
    if os.environ.get(k):
        env_vars[k] = os.environ[k]
base["env_vars"] = env_vars
print(json.dumps(base))
PY
)"

# ToolAgentLoop passes tools=schemas into apply_chat_template; without a tools-
# aware Jinja, schemas are silently dropped and the base model never sees the
# Hermes format → single-turn only. Patch MODEL_PATH once if needed.
python3 - "$MODEL_PATH" "$TOOL_CHAT_TEMPLATE" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
tmpl_path = Path(sys.argv[2])
raw = tmpl_path.read_text()
if tmpl_path.suffix in {".yaml", ".yml"}:
    import yaml

    payload = yaml.safe_load(raw)
    tmpl = payload["chat_template"] if isinstance(payload, dict) else str(payload)
else:
    tmpl = raw
tok_cfg_path = model_path / "tokenizer_config.json"
tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}
current = tok_cfg.get("chat_template") or ""
if "tool_call" in current and "tools" in current:
    print(f"[INFO] MODEL_PATH already has tool-aware chat template: {model_path}")
else:
    tok_cfg["chat_template"] = tmpl
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")
    (model_path / "chat_template.jinja").write_text(tmpl)
    print(f"[INFO] Installed tool-aware chat template into {model_path}")
PY

# Expect parquet from examples/agenticrpco_trainer/data_process/create_data.sh
# (default: data/agentic/{train,val}.parquet under the repo root).

# Colocated FSDP actor + vLLM: after actor init, free VRAM << gpu_memory_utilization
# * total. Cap util from currently free memory. Also refuse to start if a prior
# crashed VLLM::EngineCore still owns the GPU (seen as "Free memory ... less than
# desired GPU memory utilization").
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"
if command -v nvidia-smi >/dev/null 2>&1; then
  python3 - <<PY
import os, subprocess, sys
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
min_free_gb = float(os.environ.get("MIN_FREE_GB", "35"))
bad = []
for i in idxs:
    if i >= len(raw):
        continue
    parts = [x.strip() for x in raw[i].split(",")]
    idx, total, free, used = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    print(f"[INFO] GPU {idx}: used={used/1024:.1f}GiB free={free/1024:.1f}GiB / {total/1024:.1f}GiB")
    if free / 1024.0 < min_free_gb:
        bad.append((idx, free / 1024.0, used / 1024.0))
if bad:
    print("[ERROR] Training GPUs do not have enough free VRAM before launch.", file=sys.stderr)
    print("[ERROR] A prior crashed VLLM::EngineCore often leaves ~60GiB occupied.", file=sys.stderr)
    for idx, free, used in bad:
        print(f"[ERROR]   GPU {idx}: free={free:.1f}GiB used={used:.1f}GiB (need >= {min_free_gb}GiB free)", file=sys.stderr)
    print("[ERROR] Inspect: nvidia-smi", file=sys.stderr)
    print("[ERROR] Free zombies (keep GPU0 Lance diffusion server if running):", file=sys.stderr)
    print("[ERROR]   nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv", file=sys.stderr)
    print("[ERROR]   kill <EngineCore/Worker pids on CUDA_VISIBLE_DEVICES>   # or pick empty GPUs", file=sys.stderr)
    print("[ERROR] Override gate with MIN_FREE_GB=0 if you insist.", file=sys.stderr)
    sys.exit(2)
PY
fi
if [[ -z "$GPU_MEM_UTIL" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_UTIL="$(python3 - <<'PY'
import os, subprocess
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
free_fracs = []
for i in idxs:
    if i >= len(raw):
        continue
    total_s, free_s = [x.strip() for x in raw[i].split(",")]
    total, free = float(total_s), float(free_s)
    if total > 0:
        # Leave headroom for FSDP residual + LoRA sync / wake_up fragmentation.
        free_fracs.append(0.25 * free / total)
if not free_fracs:
    print("0.12")
else:
    util = max(0.08, min(0.15, min(free_fracs)))
    print(f"{util:.2f}")
PY
)"
fi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.12}"
echo "[INFO] rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"

# Sequence budget for colocated KV: Hermes tool calls are small, but two tool
# observations + refine turns need more room than single-turn free text.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESP_LEN="${MAX_RESP_LEN:-512}"
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESP_LEN))

if [[ "${GATE_SIDECAR}" == "1" ]]; then
  echo "[GATE] starting watch sidecar -> ${E2E_RUN_DIR}/gate_watch.log"
  python3 "$GATE_SCRIPT" \
    --run-dir "$E2E_RUN_DIR" \
    --watch \
    --total-steps "${TOTAL_STEPS}" \
    --interval-s "${GATE_INTERVAL_S:-20}" \
    >"$E2E_RUN_DIR/gate_watch.log" 2>&1 &
  GATE_PID=$!
  echo "[GATE] watch pid=${GATE_PID}"
fi

TRAIN_RC=0
# Mode (2a) essentials + memory-safe Lance sizing for ~100 steps on 2 GPUs.
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=2 \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  data.filter_overlong_prompts=true \
  data.truncation=left \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=16 \
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  +actor_rollout_ref.model.override_config.tie_word_embeddings=false \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.function_tool_path=verl_omni/agent_loop/diffusion_tool.py \
  actor_rollout_ref.rollout.agent.default_agent_loop=agentic_force_tool_agent \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${FORCE_AGENT_LOOP_CFG}" \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager.AgenticMetricsAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers=2 \
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
  trainer.save_freq=50 \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  "trainer.default_local_dir=${CKPT_DIR}" \
  'trainer.logger=["console","wandb"]' \
  trainer.project_name=verl_omni_agentic \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "$@" || TRAIN_RC=$?

GATE_RC=0
_run_final_gates "${TRAIN_RC}" || GATE_RC=$?
if [[ "${TRAIN_RC}" -ne 0 ]]; then
  exit "${TRAIN_RC}"
fi
exit "${GATE_RC}"
