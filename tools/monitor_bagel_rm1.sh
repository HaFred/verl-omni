#!/usr/bin/env bash
# Watch the live Bagel Co-RL RM-on run until it dies or finishes.
#
# Emits these lines so a `notify_on_output` watcher can catch them without tailing the whole
# (very chatty) log:
#   BAGEL_OK       every STEP_INTERVAL steps: progress, dump/reward/image counts
#   BAGEL_STALL    no new step for STALL_MINUTES: a wedged run, with the diagnostics that say why
#   BAGEL_FATAL    a fatal exception (known-noisy PEC RPC errors filtered out)
#   BAGEL_DONE     launcher exited / run reached its step budget
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
LOG="${1:-$(cat .gpu_watch/last_bagel_rm1.log 2>/dev/null)}"
PID_FILE=.gpu_watch/bagel_rm1.pid
STEP_INTERVAL="${STEP_INTERVAL:-10}"
# A step takes 95-140 s. A wedged run produces *no* new `update_actor` line ever again, so a
# threshold well above the slowest observed step is unambiguous. Measured 2026-09-23: the run on
# 0/1/6/7 stalled after step 30 at 22:36 and sat at 0% GPU utilisation for ~1 h, because the
# engine's subprocesses segfaulted and the trainer blocked forever on a rollout that could never
# complete. Nothing in the log said "stalled" -- only the *absence* of progress did.
STALL_MINUTES="${STALL_MINUTES:-20}"
WATCHDOG_KILL="${WATCHDOG_KILL:-0}"
RUN_DIR="${RUN_DIR:-}"
# Derive the run dir from the *log's* timestamp rather than newest-mtime. The launcher names both
# from one TS (log `.gpu_watch/bagel_rm1_<TS>.log`, run `outputs/bagel_corl_rm1_<TS>`), so this is
# exact. Newest-mtime is not: a relaunch leaves the just-killed run with the newest mtime until
# hydra creates the new dir, and the monitor then reports the dead run's frozen dumps as if live
# (measured 2026-09-23 23:40: the monitor locked onto bagel_corl_rm1_20260923_233506, the run that
# had just been stopped).
# If the log name carries a TS, trust it *unconditionally* -- even before the run dir exists. The
# dir is created by hydra ~1 min after the launcher starts, so a monitor that starts with the
# launcher (the normal case) would find it missing and fall through to newest-mtime, i.e. lock onto
# the *previous* run for the whole of the new run's life. Measured 2026-09-25 00:06: the monitor
# started 11 s after the launcher and reported run=bagel_corl_rm1_20260924_034359 (the run that had
# been stopped), while the live run was ..._20260925_000612. The wait loop below already handles a
# dir that does not exist yet, so falling back here only ever does harm.
if [ -z "$RUN_DIR" ]; then
  ts="$(basename "$LOG" | rg -o '[0-9]{8}_[0-9]{6}' | tail -1 || true)"
  if [ -n "$ts" ]; then
    RUN_DIR="verlomni-fredfork/outputs/bagel_corl_rm1_$ts"
  else
    RUN_DIR="$(ls -dt verlomni-fredfork/outputs/bagel_corl_rm1_* 2>/dev/null | head -1)"
  fi
fi
# Validate, but allow for the boot window: hydra creates the run dir a few minutes after the
# launcher starts, and `rollout_trajectories` only appears once step 0 has *completed* (the boot
# itself is ~13 min -- measured 21:32 launch -> 21:45:48 step-0 wake on the 2026-09-23 run), so
# requiring the dumps up front fails a run that is perfectly healthy. Wait for the run dir only.
if [ -n "$RUN_DIR" ]; then
  for _ in $(seq 1 60); do
    [ -d "$RUN_DIR" ] && break
    echo "BAGEL_OK monitor: waiting for $RUN_DIR to be created (run still booting)"
    sleep 15
  done
  if [ ! -d "$RUN_DIR" ]; then
    echo "BAGEL_FATAL monitor: $RUN_DIR was never created after 15min (wrong run dir?)" >&2
    exit 1
  fi
  if [ ! -d "$RUN_DIR/rollout_trajectories" ]; then
    echo "BAGEL_OK monitor: $RUN_DIR exists but has no rollout_trajectories yet (step 0 still running)"
  fi
fi

echo "BAGEL_OK monitor starting: log=$LOG run=$RUN_DIR interval=${STEP_INTERVAL}steps stall=${STALL_MINUTES}min kill=${WATCHDOG_KILL}"
last_reported=0
last_step_seen=-1
last_step_change_epoch=$(date +%s)

# One `update_actor` line per completed step; see the note at the health summary below for why
# this is used instead of tqdm's progress bar.
completed_steps() {
  rg -c 'bagel_corl_sync update_actor' "$LOG" 2>/dev/null || echo 0
}

while true; do
  sleep 60

  # 1. fatal exceptions, minus the PEC RPC noise that is a known benign mismatch
  fatal="$(rg -n '^(RuntimeError|KeyError|ValueError|AssertionError|TypeError|AttributeError):' "$LOG" 2>/dev/null \
           | rg -v 'get_prompt_embed_cache_stats' | tail -2)"
  if [ -n "$fatal" ]; then
    echo "BAGEL_FATAL $(date '+%F %T')"
    echo "$fatal"
    exit 1
  fi

  # 2. launcher gone?
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    echo "BAGEL_DONE launcher pid=$pid exited at $(date '+%F %T') steps_done=$(completed_steps)"
    exit 0
  fi

  # 3. periodic health summary
  #
  # Count *completed steps*, not tqdm's percentage. tqdm renders its bar with carriage returns,
  # which the log capture only materializes as a full line when the terminal happens to flush --
  # so `Training Progress:` stays frozen on the last flushed value (measured 2026-09-23 21:53:
  # the line still read 3/200 while `update_actor` was already on policy_version=4, and it would
  # have reported 3% for the rest of the run). `bagel_corl_sync update_actor` is written once per
  # step as an ordinary log line, so it is the reliable progress counter.
  cur="$(completed_steps)"
  # 3. stall watchdog. Checked on every tick, independently of STEP_INTERVAL: the whole point is
  # to notice the run has produced nothing new, which no other signal reports. Armed only once the
  # run has completed a step: the boot is ~13 min of legitimate silence with steps=0.
  if [ "$cur" -eq 0 ]; then
    last_step_seen=0
    last_step_change_epoch=$(date +%s)
  elif [ "$cur" -ne "$last_step_seen" ]; then
    last_step_seen="$cur"
    last_step_change_epoch=$(date +%s)
  else
    idle=$(( ($(date +%s) - last_step_change_epoch) / 60 ))
    if [ "$idle" -ge "$STALL_MINUTES" ]; then
      echo "BAGEL_STALL $(date '+%F %T') no new step for ${idle}min (stuck at step=$cur) -- the run is wedged, not slow: a step takes ~2min"
      rg -o 'Training Progress: *[0-9]+%[^]]*\]' "$LOG" 2>/dev/null | tail -1 | sed 's/^/  last progress line: /'
      rg -n 'permanent failure|Segfault|tearing down' "$LOG" 2>/dev/null | tail -3 | cut -c1-200 | sed 's/^/  /'
      # Do not treat "no fatal exception" as healthy: the wedging failure mode is exactly an
      # engine subprocess dying while the parent actor stays alive, so the trainer blocks instead
      # of raising. Report the subprocess liveness that actually distinguishes the two.
      echo "  gpu util: $(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
      if [ "$WATCHDOG_KILL" = "1" ]; then
        echo "BAGEL_STALL WATCHDOG_KILL=1 -> terminating launcher $(cat "$PID_FILE" 2>/dev/null)"
        kill -TERM "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null
      fi
      last_step_change_epoch=$(date +%s)   # report once per STALL_MINUTES, not every tick
    fi
  fi

  if [ "$cur" -ge $((last_reported + STEP_INTERVAL)) ]; then
    last_reported="$cur"
    ndump=$(ls -d "$RUN_DIR"/rollout_trajectories/step_* 2>/dev/null | wc -l)
    npng=$(find "$RUN_DIR"/rollout_images -name '*.png' 2>/dev/null | wc -l)
    deg=$(rg -o '"degenerate_und_turns": *[0-9]+' "$RUN_DIR"/rollout_trajectories/step_*/*.json 2>/dev/null \
          | rg -o '[0-9]+$' | awk '{s+=$1} END {print s+0}')
    echo "BAGEL_OK $(date '+%F %T') step=${cur} steps_with_dumps=$ndump pngs=$npng degenerate_und_turns_total=$deg"
    # skip_gen / gen_rows for the most recent step: the two numbers that say the GEN lane is live.
    rg -o 'update_actor skip_gen=[A-Za-z]+ policy_version=[0-9]+ J=[0-9.]+ K=[0-9.]+' "$LOG" 2>/dev/null | tail -1 | sed 's/^/  last: /'
    rg -o 'old_log_prob batch_rows=[0-9]+ roles=\{[^}]*\} gen_rows=[0-9]+' "$LOG" 2>/dev/null | tail -1 | sed 's/^/  last: /'
    python3 - "$RUN_DIR" <<'PY' 2>/dev/null | tail -3
import glob, json, os, sys
run = sys.argv[1]
for d in sorted(glob.glob(f"{run}/rollout_trajectories/step_*"))[-3:]:
    fs = sorted(glob.glob(d + "/sample_*.json"))
    ok = skip = deg = calls = 0; rw = []
    for f in fs:
        try:
            e = json.load(open(f))
        except Exception:
            continue
        skip += int(bool(e.get("gen_lane_skipped"))); ok += int(not e.get("gen_lane_skipped"))
        deg += int(e.get("degenerate_und_turns") or 0); calls += int(e.get("num_gen_calls") or 0)
        if e.get("und_reward") is not None: rw.append(e["und_reward"])
    r = f"{min(rw):.3f}/{sum(rw)/len(rw):.3f}/{max(rw):.3f}" if rw else "n/a"
    print(f"  {os.path.basename(d)} n={len(fs)} gen_ok={ok} gen_skip={skip} calls={calls} deg={deg} reward={r}")
PY
  fi
done
