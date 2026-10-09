#!/usr/bin/env bash
# Launch the Bagel Co-RL RM-on e2e run on devices 0/1/2/3.
#
# Same proven fix set as tools/launch_bagel_rm1_0167.sh, plus the four follow-up
# fixes for the UND prompt-contract gap and the GEN image-quality gap:
#   GEN  stop cue  unified_good_enough_threshold 0.6 -> 0.41
#        (0.6 sat above the judge's measured 0.490 ceiling: YES fired 0/663 episodes,
#         so the branch the loop keys on was a constant)
#   GEN  S=4 FlowGRPO group (was 2; seeds now run sequentially per call, so this is
#        wall-clock, not peak-VRAM, cost)
#   GEN  ENABLE_RM=1 by default on >=4 GPUs (without it skip_gen=True on every step
#        and the GEN lane is frozen)
#   UND  strip_reflection_tail() -- the checkpoint appended its in-turn
#        "Short reflection: ..." text to the diffusion prompt, conditioning the model
#        on a critique of the previous sample (~11% of measured gen prompts)
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
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$WD/bagel_rm1_0123_${TS}.log"
mkdir -p "$WD"

echo "$LOG" > "$WD/last_bagel_rm1.log"
echo "[$(date '+%F %T')] launching RM-on ${TOTAL_STEPS:-200}-step run on 0/1/2/3 -> $LOG"
cd "$REPO" || exit 1

setsid nohup env \
  CUDA_VISIBLE_DEVICES="${BAGEL_DEVICES:-0,1,2,3}" \
  ENABLE_RM="${ENABLE_RM:-1}" \
  ENABLE_PEC="${ENABLE_PEC:-0}" \
  TOTAL_STEPS="${TOTAL_STEPS:-200}" \
  RUN_NAME="bagel_corl_rm1_0123_${TS}" \
  RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_fred_0123}" \
  RAY_DEDUP_LOGS=0 \
  PYTHONUNBUFFERED=1 \
  bash "$RECIPE" >"$LOG" 2>&1 </dev/null &
PID=$!
echo "$PID" > "$WD/bagel_rm1_0123.pid"
echo "[$(date '+%F %T')] LAUNCHED pid=$PID log=$LOG"
