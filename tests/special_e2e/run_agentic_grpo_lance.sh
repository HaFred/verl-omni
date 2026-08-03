#!/usr/bin/env bash
# ============================================================================
# PR1 Merge Gate: Lance-3B Agentic GRPO GPU Smoke (ST-1)
# ============================================================================
#
# ST-1 (AC1): 1-step toy training completes — no OOM, finite non-zero loss,
#             stock HF + vLLM path.
#
# Usage (from verl-omni repo root). Set MODEL_PATH to a prepared Lance_3B_hf_und
# export (see examples/agenticrpco_trainer/README.md). Machine-local env
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
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a prepared HF understanding export (see examples/agenticrpco_trainer/README.md)}"

DATA_DIR="${DATA_DIR:-$HOME/data/agentic}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/pr1_smoke}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
# Prefer CUDA_VISIBLE_DEVICES count when the operator set it; else portable default.
if [[ -z "${N_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _cuda_devs <<< "${CUDA_VISIBLE_DEVICES// /}"
    N_GPUS="${#_cuda_devs[@]}"
  else
    N_GPUS=2
  fi
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# FlowGRPO-style: stock ppo_trainer + shared CLI overrides (no custom recipe YAML).
# shellcheck source=../../examples/agenticrpco_trainer/lance/agentic_grpo_overrides.sh
source "$REPO_ROOT/examples/agenticrpco_trainer/lance/agentic_grpo_overrides.sh"

# Optional diagnostics only (not passed into HFModelConfig — it has no
# architecture/freeze fields; those belonged to the removed Omni agentic path).
if [[ -z "${MODEL_ARCHITECTURE:-}" && -f "$MODEL_PATH/config.json" ]]; then
  MODEL_ARCHITECTURE="$(python3 -c "import json; print(json.load(open('$MODEL_PATH/config.json'))['architectures'][0])")"
fi
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-Qwen2ForCausalLM}"

mkdir -p "$OUTPUT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
_pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
_fail() { echo -e "${RED}[FAIL]${NC} $*"; }
_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

# Path + 1-GPU offload tweaks only; shared toy/not-OOM sizing is in agentic_grpo_overrides.sh.
SMOKE_MODEL_OVERRIDES=(
  "actor_rollout_ref.model.path=$MODEL_PATH"
)
if [[ "$N_GPUS" -eq 1 ]]; then
  SMOKE_MODEL_OVERRIDES+=(
    "actor_rollout_ref.actor.fsdp_config.param_offload=false"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false"
    "actor_rollout_ref.ref.fsdp_config.param_offload=false"
    "actor_rollout_ref.rollout.layered_summon=false"
  )
fi

# ---- Pre-flight -------------------------------------------------------------
# pr-fredfork omni V1 path imports TransferQueue at package load (verl ppo.v1).
if ! python3 -c "import transfer_queue" >/dev/null 2>&1; then
  _info "Installing TransferQueue (required by verl V1 / omni_sync import path) ..."
  python3 -m pip install 'TransferQueue==0.1.8'
fi

_info "=== PR1 Agentic GRPO GPU Smoke (ST-1) ==="
_info "Model:  $MODEL_PATH"
_info "Data:   $DATA_DIR"
_info "GPUs:   $N_GPUS"
_info "Output: $OUTPUT_DIR"

# Generate toy agentic data if missing
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  _info "Generating toy agentic parquet data ..."
  python3 tests/special_e2e/create_dummy_agentic_data.py \
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
python3 -m verl.trainer.main_ppo \
    'hydra.run.dir='"$OUTPUT_DIR" \
    'hydra.sweep.dir='"$OUTPUT_DIR" \
    hydra.output_subdir=null \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    "${AGENTIC_GRPO_OVERRIDES[@]}" \
    "${SMOKE_MODEL_OVERRIDES[@]}" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=1 \
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
python3 - "$ST1_LOG" <<'PY' || LOSS_CHECK_EXIT=$?
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
