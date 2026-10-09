#!/usr/bin/env bash
# Wait for card 5 to be released, then launch the layout-A Bagel run on 2,3 (judge on 5).
#
# Why: the operator chose to WAIT for the judge card rather than share it with the Boogu
# diffusion run that currently holds ~21.8GiB there. Cards 2/3 -- the ones the trainer actually
# uses -- are free now, but the run cannot start without the judge, and the judge card is the
# last place a co-tenant can still break the reward path.
#
# All four launch conditions must hold, checked every POLL_SECONDS:
#   * card 5 used memory below FREE_MIB for REQUIRED_CONSECUTIVE consecutive polls
#     (one dip is not a release -- a sleeping engine hands memory back and takes it again)
#   * cards 2 and 3 are still free (someone else may have taken them while we waited)
#   * no bagel run is already live (bagel_rm1.pid pointing at a running process)
# If a *training* card was taken we do NOT launch blind: we log it and exit non-zero so the
# operator can re-pick, rather than burning hours on a contended pool.
set -u

REPO=/home/fq9hpsac/fq9hpsacuser11/fred/verlomni-fredfork
WD=/home/fq9hpsac/fq9hpsacuser11/fred/.gpu_watch
LAUNCH="$REPO/tools/launch_bagel_rm1_2gpu_judge5.sh"
LOG="$WD/wait_card5.$(date +%Y%m%d_%H%M%S).log"

POLL_SECONDS="${POLL_SECONDS:-60}"
FREE_MIB="${FREE_MIB:-2048}"
# The training-card gate gets its OWN threshold. Reusing FREE_MIB here inverted the check the
# moment FREE_MIB was raised (a card at 73GiB "passed" a 999999MiB bar), so the watcher launched
# onto a contended pool -- measured 2026-09-28 08:21. Keep the two independent.
TRAIN_FREE_MIB="${TRAIN_FREE_MIB:-2048}"
REQUIRED_CONSECUTIVE="${REQUIRED_CONSECUTIVE:-2}"
JUDGE_CARD="${JUDGE_CARD:-5}"
TRAIN_CARDS_STR="${TRAIN_CARDS_STR:-2 3}"

mkdir -p "$WD"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

bagel_live() {
  local pidfile="$WD/bagel_rm1.pid" pid
  [ -f "$pidfile" ] || return 1
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

log "watcher start: waiting for card $JUDGE_CARD below ${FREE_MIB}MiB for ${REQUIRED_CONSECUTIVE} consecutive polls"
log "train cards: $TRAIN_CARDS_STR ; poll ${POLL_SECONDS}s ; launcher $LAUNCH"

consecutive=0
while true; do
  if bagel_live; then
    log "a bagel run is already live (pid $(cat "$WD/bagel_rm1.pid")) -- watcher exiting"
    exit 0
  fi

  j5="$(used_mib "$JUDGE_CARD")"
  if [ -z "$j5" ]; then
    log "WARN could not read card $JUDGE_CARD usage; retrying"
    consecutive=0
    sleep "$POLL_SECONDS"
    continue
  fi

  if [ "$j5" -lt "$FREE_MIB" ]; then
    consecutive=$((consecutive + 1))
    log "card $JUDGE_CARD = ${j5}MiB (free) [$consecutive/$REQUIRED_CONSECUTIVE]"
  else
    if [ "$consecutive" -ne 0 ]; then
      log "card $JUDGE_CARD went back to ${j5}MiB -- resetting the counter"
    fi
    consecutive=0
  fi

  if [ "$consecutive" -ge "$REQUIRED_CONSECUTIVE" ]; then
    blocked=0
    for card in $TRAIN_CARDS_STR; do
      u="$(used_mib "$card")"
      if [ -z "$u" ] || [ "$u" -ge "$TRAIN_FREE_MIB" ]; then
        log "ABORT: training card $card is not free (${u:-unreadable}MiB) -- re-pick the pool"
        blocked=1
      fi
    done
    if [ "$blocked" -ne 0 ]; then
      exit 1
    fi
    log "card $JUDGE_CARD released and training cards free -> launching"
    if bash "$LAUNCH" >>"$LOG" 2>&1; then
      log "launcher returned; log=$(cat "$WD/last_bagel_rm1.log" 2>/dev/null)"
      exit 0
    fi
    log "ABORT: launcher failed (see above)"
    exit 1
  fi

  sleep "$POLL_SECONDS"
done
