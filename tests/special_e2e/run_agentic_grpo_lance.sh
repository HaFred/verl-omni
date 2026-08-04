#!/usr/bin/env bash
# ============================================================================
# PR1 Merge Gate: Lance-3B Agentic GRPO GPU Smoke (ST-1)
# ============================================================================
#
# ST-1 (AC1): 1-step toy training completes — no OOM, finite non-zero loss,
#             stock HF + vLLM path.
#
# Usage (from verl-omni repo root). Set MODEL_PATH to a prepared Lance_3B_hf_und
# export (see tests/special_e2e/prepare_lance_hf_und.py). Machine-local env
# (CUDA/Ray LD_LIBRARY_PATH, GPU ids, WANDB, NCCL, VERL_USE_EXTERNAL_MODULES)
# belongs in the operator shell — not this script — then:
#   bash tests/special_e2e/run_agentic_grpo_lance.sh
# Do NOT point MODEL_PATH at raw Lance_3B (no chat_template → empty dataset).
#
# Output:
#   outputs/pr1_smoke/st1_agentic_onestep.log
# ============================================================================
set -euo pipefail

# ---- Config -----------------------------------------------------------------
# Lance MoT HF layout (Lance_3B) is incomplete: no config.json / chat_template.
# Smoke uses the prepared understanding-only export (prepare_lance_hf_und.py).
# Do NOT point MODEL_PATH at raw Lance_3B — tokenizer has no chat_template and
# every prompt is skipped → filter dataset len: 0.
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a prepared HF understanding export (see prepare_lance_hf_und.py)}"

DATA_DIR="${DATA_DIR:-$HOME/data/agentic}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/pr1_smoke}"
# Merge gate always uses its own toy parquet unless ST1_USE_ENV_DATA=1
# (operator env often exports overfit TRAIN_FILE/VAL_FILE).
if [[ "${ST1_USE_ENV_DATA:-0}" == "1" ]]; then
  TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
  VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
else
  TRAIN_FILE="$DATA_DIR/train.parquet"
  VAL_FILE="$DATA_DIR/val.parquet"
fi
# Prefer CUDA_VISIBLE_DEVICES count when the operator set it; else portable default.
if [[ -z "${N_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _cuda_devs <<< "${CUDA_VISIBLE_DEVICES// /}"
    N_GPUS="${#_cuda_devs[@]}"
  else
    N_GPUS=2
  fi
fi

# Always use the active venv interpreter when present (bare `python3` may be
# miniconda/base and would install TransferQueue into the wrong env).
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Optional diagnostics only (not passed into HFModelConfig — it has no
# architecture/freeze fields; those belonged to the removed Omni agentic path).
if [[ -z "${MODEL_ARCHITECTURE:-}" && -f "$MODEL_PATH/config.json" ]]; then
  MODEL_ARCHITECTURE="$("$PYTHON_BIN" -c "import json; print(json.load(open('$MODEL_PATH/config.json'))['architectures'][0])")"
fi
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-Qwen2ForCausalLM}"

mkdir -p "$OUTPUT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
_pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
_fail() { echo -e "${RED}[FAIL]${NC} $*"; }
_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOL_CHAT_TEMPLATE="${TOOL_CHAT_TEMPLATE:-$REPO_ROOT/tests/special_e2e/qwen2_tool_chat_template.yaml}"

# Install Hermes/Qwen2.5 tool-aware chat template if MODEL_PATH still has a
# tools-blind stub (otherwise ToolAgentLoop silently drops tool schemas).
"$PYTHON_BIN" - "$MODEL_PATH" "$TOOL_CHAT_TEMPLATE" <<'PY'
import json, sys
from pathlib import Path

import yaml

model_path, tmpl_path = Path(sys.argv[1]), Path(sys.argv[2])
payload = yaml.safe_load(tmpl_path.read_text())
tmpl = payload["chat_template"] if isinstance(payload, dict) else str(payload)
tok_cfg_path = model_path / "tokenizer_config.json"
tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}
current = tok_cfg.get("chat_template") or ""
if "tool_call" in current and "tools" in current:
    print("[INFO] MODEL_PATH already has tool-aware chat template")
else:
    tok_cfg["chat_template"] = tmpl
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")
    (model_path / "chat_template.jinja").write_text(tmpl)
    print(f"[INFO] Installed tool-aware chat template into {model_path}")
PY

# Colocated hybrid: vLLM checks free >= util * total *after* FSDP actor init.
# Reserve an actor footprint, then pick util from the remaining free memory.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
if [[ -z "$GPU_MEM_UTIL" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_UTIL="$("$PYTHON_BIN" - <<'PY'
import os, subprocess, sys
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
# FSDP+LoRA residual observed on this box is ~12–20 GiB/GPU with offload.
ACTOR_RESERVE_GIB = 20.0
utils = []
for i in idxs:
    if i >= len(raw):
        continue
    total_s, free_s = [x.strip() for x in raw[i].split(",")]
    total, free = float(total_s), float(free_s)
    if total <= 0:
        continue
    remain = max(0.0, free - ACTOR_RESERVE_GIB * 1024)
    utils.append(remain / total)
if not utils:
    print("0.12")
else:
    util = min(utils)
    # Toy KV only needs a small util; clamp high enough for vLLM, low enough
    # for busy/shared boxes.
    util = max(0.08, min(0.15, util))
    print(f"{util:.2f}")
PY
)"
fi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.12}"

# Fail fast if any visible GPU is already too full for colocated FSDP+vLLM.
if command -v nvidia-smi >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY' || exit 2
import os, subprocess, sys
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
min_free_gib = 24.0
bad = []
for i in idxs:
    if i >= len(raw):
        continue
    total_s, free_s = [x.strip() for x in raw[i].split(",")]
    free_gib = float(free_s) / 1024.0
    if free_gib < min_free_gib:
        bad.append(f"GPU{i} free={free_gib:.1f}GiB")
if bad:
    print(
        "[FAIL] ST-1 needs >=24GiB free on every CUDA_VISIBLE_DEVICES GPU "
        f"for colocated FSDP+vLLM; busy: {', '.join(bad)}. "
        "Pick free GPUs (e.g. CUDA_VISIBLE_DEVICES=1,4) or stop the occupant.",
        file=sys.stderr,
    )
    raise SystemExit(2)
print(f"[INFO] visible GPUs free-memory check passed: {idxs}")
PY
fi

# Mode (2a) + toy sizing inlined for the PR1 1-step merge gate.
# Multi-step Lance e2e / overfit recipes live on the PR2 working branch.
SMOKE_OVERRIDES=(
  algorithm.adv_estimator=grpo

  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.load_format=safetensors

  +actor_rollout_ref.model.override_config.tie_word_embeddings=false
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa

  actor_rollout_ref.rollout.multi_turn.enable=true
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
  actor_rollout_ref.rollout.multi_turn.max_user_turns=5
  actor_rollout_ref.rollout.multi_turn.function_tool_path=verl_omni/agent_loop/diffusion_tool.py

  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path=null
  # train_batch_size must divide agent.num_workers (DataProto.chunk).
  actor_rollout_ref.rollout.agent.num_workers=2

  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_reward
  # Length heuristic: Mode (2a) compute_score needs Hermes/tools; cold und →
  # zero advantages → actor/loss=0. Smoke only needs reward variance for GRPO.
  reward.custom_reward_function.name=compute_score_smoke

  data.train_batch_size=4
  data.max_prompt_length=1024
  data.max_response_length=1024
  data.filter_overlong_prompts=true
  data.truncation=left
  # Cap below HF config max (128k) so colocated util~0.15 has enough KV.
  actor_rollout_ref.rollout.max_model_len=4096

  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=32
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
  actor_rollout_ref.model.enable_gradient_checkpointing=true

  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true

  actor_rollout_ref.rollout.n=2
  actor_rollout_ref.rollout.temperature=0.8
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2

  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true

  trainer.val_before_train=false
  trainer.nnodes=1
)
if [[ "$N_GPUS" -eq 1 ]]; then
  SMOKE_OVERRIDES+=(
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.ref.fsdp_config.param_offload=false
    actor_rollout_ref.rollout.layered_summon=false
  )
fi

# ---- Pre-flight -------------------------------------------------------------
# pr-fredfork omni V1 path imports TransferQueue at package load (verl ppo.v1).
if ! "$PYTHON_BIN" -c "import transfer_queue" >/dev/null 2>&1; then
  _info "Installing TransferQueue into $($PYTHON_BIN -c 'import sys; print(sys.executable)') ..."
  "$PYTHON_BIN" -m pip install 'TransferQueue==0.1.8'
fi

_info "=== PR1 Agentic GRPO GPU Smoke (ST-1) ==="
_info "Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
_info "Model:  $MODEL_PATH"
_info "Data:   $DATA_DIR"
_info "GPUs:   $N_GPUS  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
_info "vLLM gpu_memory_utilization=${GPU_MEM_UTIL}"
_info "Output: $OUTPUT_DIR"

# Generate / refresh toy agentic data if missing or still on the pre-Hermes schema.
NEED_DATA=0
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  NEED_DATA=1
elif ! "$PYTHON_BIN" - "$TRAIN_FILE" <<'PY'
import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
prompt = df.iloc[0]["prompt"]
# Hermes few-shot schema: system + fewshot user/assistant + real user
ok = isinstance(prompt, list) and len(prompt) >= 4 and any(
    isinstance(m, dict) and m.get("role") == "assistant" and "<tool_call>" in str(m.get("content", ""))
    for m in prompt
)
raise SystemExit(0 if ok else 1)
PY
then
  NEED_DATA=1
fi
if [[ "$NEED_DATA" -eq 1 ]]; then
  _info "Generating Hermes-format toy agentic parquet data ..."
  "$PYTHON_BIN" tests/special_e2e/create_dummy_agentic_data.py \
    --local_save_dir "$DATA_DIR" --train_size 8 --val_size 4
fi

ST1_FAIL=0
ST1_LOG="$OUTPUT_DIR/st1_agentic_onestep.log"
ST1_CKPT="$OUTPUT_DIR/st1_ckpt"

# Record a failed assertion without putting FAIL= on the same line as _fail
# (avoids brittle `; VAR=1` parsing after echo -e / ANSI).
_record_fail() {
  _fail "$1"
  ST1_FAIL=1
}

# ============================================================================
# ST-1: 1-step toy training completes (AC1)
# ============================================================================
_info ""
_info "=== ST-1: 1-Step Agentic GRPO Training ==="
_info "Log:   $ST1_LOG"
_info "Arch:  $MODEL_ARCHITECTURE  rollout=vllm"

# Fresh smoke: never resume a prior ST-1 ckpt (LoRA/FSDP keys can mismatch across runs).
rm -rf "$ST1_CKPT"
mkdir -p "$ST1_CKPT"

set +e  # do not exit on failure so we can report
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    'hydra.run.dir='"$OUTPUT_DIR" \
    'hydra.sweep.dir='"$OUTPUT_DIR" \
    hydra.output_subdir=null \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    "${SMOKE_OVERRIDES[@]}" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$ST1_CKPT" \
    trainer.logger=console \
    "$@" 2>&1 | tee "$ST1_LOG"
ST1_EXIT=$?
set -e

# --- ST-1 assertions ---
if [[ "$ST1_EXIT" -ne 0 ]]; then
  _record_fail "ST-1: Training exited with code $ST1_EXIT"
fi
# Real CUDA/torch OOM only — do not match FSDP warnings like "risks CPU OOM".
if grep -Eiq 'cuda\s*out\s*of\s*memory|torch\.OutOfMemoryError|OutOfMemoryError:\s*CUDA' "$ST1_LOG"; then
  _record_fail "ST-1: OOM detected"
fi
if ! grep -q "actor/loss" "$ST1_LOG"; then
  _record_fail "ST-1: No 'actor/loss' metric in log"
fi
# Guard: recipe must stay on stock HF + vLLM (no removed custom agentic/Omni path).
if grep -Eq "AgenticLLMFSDPEngine|model_type.?.?agentic_llm|vllm_omni_model" "$ST1_LOG"; then
  _record_fail "ST-1: Unexpected custom worker/model path detected"
else
  _pass "ST-1: Stock language-model worker/rollout path used"
fi
# Non-zero finite loss ⇒ actor gradients flowed (AC: agent LLM weights update).
LOSS_CHECK_EXIT=0
"$PYTHON_BIN" - "$ST1_LOG" <<'PY' || LOSS_CHECK_EXIT=$?
import math
import re
import sys

loss_re = re.compile(
    r"actor/loss(?:[:\s=]+|:)(?:np\.(?:float64|float32)\()?([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?|nan|[-]?inf)\)?",
    re.IGNORECASE,
)
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for line in f:
        m = loss_re.search(line)
        if not m:
            continue
        raw = m.group(1)
        try:
            val = float(raw)
        except ValueError:
            continue
        if math.isfinite(val) and val != 0.0:
            sys.exit(0)
        if not math.isfinite(val):
            sys.exit(2)
sys.exit(1)
PY
if [[ "$LOSS_CHECK_EXIT" -eq 2 ]]; then
  _record_fail "ST-1: NaN/Inf in actor/loss"
elif [[ "$LOSS_CHECK_EXIT" -ne 0 ]]; then
  _record_fail "ST-1: Could not extract non-zero actor/loss"
else
  _pass "ST-1: Non-zero actor/loss confirms gradients flowed"
fi
# Multi-turn evidence on the toy agentic dataset (AC: rewrites across iterations).
MULTITURN_EXIT=0
python3 - "$ST1_LOG" <<'PY' || MULTITURN_EXIT=$?
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
# Match num_turns/... metrics with value >= 2 (plain float or numpy wrapper).
for m in re.finditer(
    r"num_turns[^0-9\n]{0,48}?(?:np\.\w+\()?([2-9]\d*(?:\.\d+)?)",
    text,
):
    sys.exit(0)
sys.exit(1)
PY
if [[ "$MULTITURN_EXIT" -ne 0 ]]; then
  _record_fail "ST-1: No multi-turn evidence (num_turns >= 2) in log"
else
  _pass "ST-1: Multi-turn num_turns >= 2 observed"
fi

if [[ "$ST1_FAIL" -eq 0 ]]; then
  _pass "ST-1: PASSED"
else
  _fail "ST-1: FAILED - ${ST1_FAIL} assertion group(s) failed"
fi

echo ""
echo "================================================================================"
echo "  PR1 Merge Gate: GPU Smoke Test Results"
echo "================================================================================"
printf "  ST-1 (AC1: 1-step training):     %s\n" "$([ "$ST1_FAIL" -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "--------------------------------------------------------------------------------"
echo "  Log: $ST1_LOG"
echo "================================================================================"

if [ "$ST1_FAIL" -eq 0 ]; then
  echo ""
  echo "  ✅ PR1 GPU SMOKE (ST-1): PASSED"
  echo ""
  exit 0
else
  echo ""
  echo "  ❌ PR1 GPU SMOKE (ST-1): FAILED"
  echo ""
  exit 1
fi
