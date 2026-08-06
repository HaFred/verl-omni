#!/usr/bin/env bash
# Build overfit agentic parquet for Mode (2a) GRPO.
#
# Run from the repo root (verlomni-pr-fredfork):
#   bash examples/agenticrpco_trainer/data_process/create_data.sh
#
# Writes:
#   data/agentic/train.parquet
#   data/agentic/val.parquet
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/data/agentic}"
TRAIN_SIZE="${OVERFIT_TRAIN_SIZE:-8}"
VAL_SIZE="${OVERFIT_VAL_SIZE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$SAVE_DIR"
TOOL_CALL_FORMAT="${TOOL_CALL_FORMAT:-auto}"
echo "[INFO] Writing overfit agentic parquet under $SAVE_DIR (tool_call_format=${TOOL_CALL_FORMAT})"
"$PYTHON_BIN" "$SCRIPT_DIR/create_dummy_agentic_data.py" \
  --local_save_dir "$SAVE_DIR" \
  --overfit \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --tool_call_format "${TOOL_CALL_FORMAT}" \
  ${MODEL_PATH:+--model_path "$MODEL_PATH"}

echo "[INFO] Done: $SAVE_DIR/train.parquet , $SAVE_DIR/val.parquet"
