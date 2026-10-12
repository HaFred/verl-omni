#!/usr/bin/env bash
# Launch the Bagel Co-RL RM-on e2e run on devices 0/1/2/3.
#
# Recipe defaults this launch relies on (examples/agenticllmgrpo_trainer/bagel/
# run_agentic_bagel_rpco_lora.sh):
#   GEN  stop cue  unified_good_enough_threshold 0.62 (p90 of the 2026-10-11
#        bagel_corl_reflect_k3 run; 0.41 fired YES on 99.3% of first images)
#   GEN  S=1 image per call, K <= MAX_GENERATE_PASSES=3; single-image calls are
#        FlowGRPO-grouped by task across the N sibling episodes
#   GEN  GEN_DECODE=sde training kernel (bounded SDE window), val stays ODE
#   GEN  ENABLE_RM=1 (without it skip_gen=True on every step and the GEN lane is frozen)
#   UND  strip_reflection_tail() -- the checkpoint appended its in-turn
#        "Short reflection: ..." text to the diffusion prompt
#
# Logging: console + remote wandb. The WANDB_* credentials are read from WANDB_ENV_FILE
# (only the WANDB_* exports -- sourcing the whole operator env would clobber the
# recipe's state). The full console log is saved as logs/<run_name>.console.log, with a
# symlink in $WD for the watch scripts; wandb's local files land in the run dir.
#
# RAY_TMPDIR is deliberately distinct from the 4/5/6/7 job's /tmp/ray_fred_r9_4gpu so
# the two Ray clusters cannot collide.
#
# ENABLE_PEC=0: the prompt-embed cache (RFC §4.4.0(b) / R2) is unserviceable in the
# pinned vllm_omni -- ``clear_prompt_embed_cache`` / ``get_prompt_embed_cache_stats``
# are defined on ``DiffusionModelRunner`` but ``collective_rpc`` resolves against the
# diffusion *worker*, which does not bridge to ``model_runner``. The server latches both
# RPCs off after the first missing-method report (see the DESIGN.md caveat), and the net
# effect is that **the conditioning cache is never flushed**. The cache holds
# text-encoder outputs -- a function of (conditioning tokens, *weights*) -- and the LoRA
# targets exactly those modules, so stale entries survive every weight sync. That is a
# confound for an image-quality run, so it is off here. Set ENABLE_PEC=1 to restore R2
# once the worker->model_runner bridge lands upstream.
set -u

REPO=/home/fq9hpsac/fq9hpsacuser11/fred/verlomni-fredfork
RECIPE="$REPO/examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh"
WD=/home/fq9hpsac/fq9hpsacuser11/fred/.gpu_watch
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/home/fq9hpsac/fq9hpsacuser11/fred/fred_verlomni_agentic_multiturn_pr1.sh}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="bagel_corl_rm1_0123_${TS}"
RUN_DIR="$REPO/outputs/$RUN_NAME"
# Not under $RUN_DIR: the recipe keeps that dir for Hydra to create.
LOG="$REPO/logs/${RUN_NAME}.console.log"
mkdir -p "$WD" "$REPO/logs"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if [[ ! -f "$WANDB_ENV_FILE" ]]; then
    echo "WANDB_API_KEY unset and WANDB_ENV_FILE=$WANDB_ENV_FILE missing; refusing a wandb-less launch" >&2
    exit 1
  fi
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_(API_KEY|MODE|SERVICE_TRANSPORT|SILENT|ENTITY)=' "$WANDB_ENV_FILE")"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "no WANDB_API_KEY found in $WANDB_ENV_FILE; refusing a wandb-less launch" >&2
  exit 1
fi
export WANDB_API_KEY WANDB_MODE="${WANDB_MODE:-online}" WANDB_DIR="$RUN_DIR"

ln -sfn "$LOG" "$WD/bagel_rm1_0123_${TS}.log"
echo "$LOG" > "$WD/last_bagel_rm1.log"
echo "[$(date '+%F %T')] launching RM-on ${TOTAL_STEPS:-200}-step run on ${BAGEL_DEVICES:-0,1,2,3} -> $LOG"
cd "$REPO" || exit 1

setsid nohup env \
  CUDA_VISIBLE_DEVICES="${BAGEL_DEVICES:-0,1,2,3}" \
  ENABLE_RM="${ENABLE_RM:-1}" \
  ENABLE_PEC="${ENABLE_PEC:-0}" \
  TOTAL_STEPS="${TOTAL_STEPS:-200}" \
  RUN_TS="$TS" \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_NAME}" \
  DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$RUN_DIR/checkpoints}" \
  RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_fred_0123}" \
  RAY_DEDUP_LOGS=0 \
  PYTHONUNBUFFERED=1 \
  bash "$RECIPE" >"$LOG" 2>&1 </dev/null &
PID=$!
echo "$PID" > "$WD/bagel_rm1_0123.pid"
echo "[$(date '+%F %T')] LAUNCHED pid=$PID log=$LOG"
