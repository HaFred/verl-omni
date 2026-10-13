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
# N=4 siblings per task (recipe default is 2). The 2026-10-12 `_084724` run at N=2 could not
# converge: with two episodes in the token-GRPO group the normalized advantage is a constant
# +-1/sqrt(2) at every step (`critic/advantages/{min,max}` = -/+0.7071), so the *magnitude* of
# a better episode carried no gradient -- only its sign. Four siblings make the advantages
# graded. `episode_und_reward` was also rewritten (2026-10-13) from the mean of every rated
# image to the final call plus the rewrite gain, so a successful prompt rewrite is no longer
# diluted by the bad image it replaced.
#
# Three more fixes (2026-10-13) close the drift that killed `_084724` after step ~47, where
# `actor/grad_norm` ran 0.32 -> 310 and `rollout_corr/kl` 0.03 -> 175 with `actor/entropy`
# collapsing to 0.02:
#   * NORM_ADV_BY_STD=False -- stock GRPO's std division is *gap-independent* (any N=2 gap
#     gives +/-0.7071), i.e. a full-amplitude push on a coin flip every step. Raw centred
#     advantage (Dr.GRPO) scales with the real reward gap instead. Recipe §6.
#   * LR_WARMUP_RATIO / LR_SCHEDULER / MIN_LR_RATIO -- the actor ran `lr=1e-4` constant for
#     all 200 steps with no warmup; now 5% warmup + cosine decay to 10% of peak.
#   * Divergence tripwire (`OmniBagelCoRLTrainerSync._divergence_tripwire`) -- fails loud/
#     warns the step `rollout_corr/kl` > 20 or `actor/grad_norm` > 100 instead of silently
#     burning the remaining budget. Set `BAGEL_CORL_DIVERGENCE_ABORT=1` to make it fatal.
#
# Reward-scorer fix (2026-10-13), from `_024557` step 30 -- `reward_done` was pinned to 0.0 on
# all 450 val rows while `terminal_done` read 0.75. Two defects in `_terminal_decision`:
#   * the harness appends its stop cue to the *verdict's own line*, and the cue loop floored on
#     `line_end + 1`, so the cue was always discarded and `forced_context` never became True;
#     `closed` needs `(policy_reflection or forced_context)`, so the dim was unreachable.
#   * `build_forced_reflection` says the next action "must be exactly Done.", but the
#     checkpoint answers in its tool grammar with `{"name": "Done"}` glued to the cue; only
#     prose `Done.` was accepted, so those 48/260 terminations scored as never terminating,
#     also zeroing `reward_result` and docking `reward_format` (0.4 vs 0.8).
# Rescoring one such episode: composite 0.550 -> 0.706. Tests:
# tests/utils/reward_score/test_agentic_multidim_reward_on_cpu.py::test_bagel_*.
#
# VAL_MAX_SAMPLES=32 here (recipe default 8 is sized for plumbing runs). `_051218` was stopped at
# step 60 because 8 val prompts could not resolve the effect being measured: one episode moves a
# binary dim (`reward_done`, `terminal_done`) by ~0.17, while `val-core` only moved
# 0.835 -> 0.814 -> 0.837 over steps 0/30/60. 32 prompts cut that granularity to ~0.03 at ~4x the
# per-pass val cost (every TEST_FREQ=30 steps).
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
# Sibling episodes per task. 4 (>2) so the token-GRPO group has a graded advantage; see the
# header. `N=2` left `critic/advantages` pinned at +-1/sqrt(2) on the 2026-10-12 `_084724` run.
N="${N:-4}"
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
  N="$N" \
  ENABLE_RM="${ENABLE_RM:-1}" \
  ENABLE_PEC="${ENABLE_PEC:-0}" \
  TOTAL_STEPS="${TOTAL_STEPS:-200}" \
  VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}" \
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
