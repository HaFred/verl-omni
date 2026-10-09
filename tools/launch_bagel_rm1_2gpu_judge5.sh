#!/usr/bin/env bash
# Launch the Bagel Co-RL RM-on e2e run on the free cards of hk01dgx012 as
# **2 training GPUs on 2,3 + the RM judge off the training pool on 5** (layout A).
#
#   CUDA_VISIBLE_DEVICES=2,3   -> actor pool width 2 (trainer.n_gpus_per_node=2)
#   JUDGE_GPU=5                -> the 9B RM judge, on a card the trainer never claims
#
# Why this shape rather than all three cards for training:
#
# * The dominant run-killer is the level-1 sleep's wake re-map. vLLM's sleep-mode allocator
#   needs a large *physically contiguous* region (``cuMemCreate``/``cuMemMap``), so it can fail
#   even with ~41GiB free when a co-tenant holds a slice of the card (measured 2026-09-25
#   00:21). Removing the failure *class* means keeping every tenant this run controls off the
#   cards it trains on -- which is what putting the judge on its own card does. ``JUDGE_GPU``
#   defaults to the LAST visible device (``${CUDA_VISIBLE_DEVICES##*,}`` == 3 here), so it MUST
#   be pinned explicitly or the judge lands on a training card.
# * 3 is a structurally awkward actor-pool width. ``split_resource_pool(actor_pool,
#   split_size=und_n_gpus)`` builds ``world_size / split_size`` EQUAL sub-pools, so
#   ``UND_N_GPUS`` has to divide 3 -> {1, 3}. The AR deploy yaml puts the Thinker on logical 0
#   and the DiT on logical 1, so 1 is impossible (a 1-card pool cannot resolve logical device
#   1), and 3 makes the AR replica declare TP=3 against a 2-stage yaml -- a shape nothing has
#   exercised. On 2 cards ``UND_N_GPUS=2`` passes every guard untouched.
# * The launch divisibility gate also has to hold: ``TRAIN_BSZ * N = 2 * 2 = 4`` must divide the
#   pool. On 3 cards that fails (4 % 3 != 0) and needs a TRAIN_BSZ bump; on 2 it already holds.
#
# Every guard in the recipe is therefore satisfied with **no overrides at all**:
#   UND_N_GPUS=2 % 2 == 0                     (the AR slice divides the pool)
#   ROLLOUT_TP=1 (pool < 4)                   -> bagel_corl_deploy.yaml, devices "0"
#   REWARD_TP=2 % 2 == 0                      -> reward.num_workers = 2/2 = 1
#   EPISODES_PER_UPDATE = TRAIN_BSZ(2) * N(2) = 4, 4 % 2 == 0
#   PER_RANK_EPISODES = 2, 2 % PPO_MICRO(1) == 0
#
# Per-card peak this relies on (79.66GiB cards): GEN rank 28.21 + one 27.5GiB AR stage + the
# actor's FSDP param shard (~28.4GiB/2 ranks ~ 14.2) + half the RM (~4.8) ~ 74.7GiB, so the
# 28.56GiB wake re-map still has room *because the sleeping GEN engine has already handed its
# 28.2GiB back to the driver* by then. That is the same arithmetic the 4-card path uses; what
# is different here is that no co-tenant is holding the contiguous region.
#
# NOTE ON CARD 5: the judge shares it with whatever else is resident. Sharing the *judge* card
# does not touch the training cards' wake path, but the judge does need its own
# ``AGENTIC_REFLECT_GPU_MEM_UTIL`` (0.32 ~ 25.5GiB) to fit beside an existing tenant.
#
# Carries the K=0 fix this run exists to measure: ``und_turn_kind`` now reads a bare
# ``<|im_end|>`` / the template's own ``End of conversation.`` as ``done``, so an episode that
# never emits a ``generate_image`` call stops at turn 2 instead of spinning the full
# ``max_und_turns=8`` cap (measured 2026-09-27: 14 of 173 episodes on ``...165227``). Watch
# ``K0=`` in the step log for the incidence and ``J=`` for the spin.
set -u

REPO=/home/fq9hpsac/fq9hpsacuser11/fred/verlomni-fredfork
RECIPE="$REPO/examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh"
WD=/home/fq9hpsac/fq9hpsacuser11/fred/.gpu_watch
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$WD/bagel_rm1_${TS}.log"
mkdir -p "$WD"

# The monitor derives the run dir from a YYYYmmdd_HHMMSS in the log basename and expects
# ``outputs/bagel_corl_rm1_<ts>``, so keep this name pattern (tools/monitor_bagel_rm1.sh:38).
echo "$LOG" > "$WD/last_bagel_rm1.log"
echo "[$(date '+%F %T')] launching RM-on ${TOTAL_STEPS:-200}-step run: train=2,3 judge=5 -> $LOG"
cd "$REPO" || exit 1

setsid nohup env \
  CUDA_VISIBLE_DEVICES=2,3 \
  JUDGE_GPU=5 \
  ENABLE_RM=1 \
  TOTAL_STEPS="${TOTAL_STEPS:-200}" \
  RUN_NAME="bagel_corl_rm1_${TS}" \
  RAY_TMPDIR=/tmp/ray_fred_2gpu_j5 \
  RAY_DEDUP_LOGS=0 \
  PYTHONUNBUFFERED=1 \
  bash "$RECIPE" >"$LOG" 2>&1 </dev/null &
PID=$!
echo "$PID" > "$WD/bagel_rm1.pid"
echo "[$(date '+%F %T')] LAUNCHED pid=$PID log=$LOG"
