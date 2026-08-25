#!/usr/bin/env bash
# Profile the agentic RPCO pipeline with two complementary layers.
#
# Layer 1 — offline GPU/kernel profiling (verl `global_profiler`):
#   torch.profiler / nsys / torch_memory capture the FSDP actor / ref / vLLM
#   rollout boundaries into a Chrome trace or nsys report. This sees the
#   CUDA-dense parts (update_actor / gen / weight transfer) but is blind to
#   the I/O-bound multi-turn sidecar loop.
#
# Layer 2 — online rl-insight observability (Prometheus + Tempo + Grafana):
#   `trace_state` swim lanes paint each rollout's `decode` -> `generate_image`
#   -> `judge_image` timeline, and `metric_histogram` publishes
#   `agentic_tool_{generate_image,judge_image}_latency_seconds` — surfacing the
#   sidecar bubbles that layer 1 cannot see. Instrumentation lives in
#   verl_omni/agent_loop/rl_insight_profiler.py (facade), agentic_tool_agent_loop.py
#   (decode lanes), tools.py (tool lanes + latency histograms), and
#   agentic_metrics_manager.py (init in driver + workers). All of it is a
#   no-op unless `rl-insight` is installed and RL_INSIGHT_SERVER_URL is set.
#
# Usage (four panes — gen sidecar, judge sidecar, rl-insight server, then this):
#   pane A: CUDA_VISIBLE_DEVICES=0,1 bash .../run_image_gen_tool_server.sh
#   pane B: CUDA_VISIBLE_DEVICES=0   bash .../run_judge_image_tool_server.sh
#   pane C: RL_INSIGHT_SERVER_ONLY=1 bash .../profile_agentic_rpco.sh   (start stack)
#   pane D: CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 TOTAL_STEPS=4 \
#             bash .../profile_agentic_rpco.sh
#
# Env knobs:
#   PROFILER_TOOL          torch | nsys | torch_memory   (default torch)
#   PROFILER_STEPS         comma-separated step list     (default 1)
#   RL_INSIGHT             1|0  enable rl-insight layer  (default 1)
#   RL_INSIGHT_SERVER_URL  rl-insight server URL         (default http://127.0.0.1:18080)
#   RL_INSIGHT_SERVER_ONLY 1|0  only start/stop the stack, skip training
#   RL_INSIGHT_SERVER_STOP 1|0  stop the stack and exit
#
# See docs/perf/profiler.md for the verl profiler field reference and
# https://github.com/verl-project/rl-insight for the observability stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILER_TOOL="${PROFILER_TOOL:-torch}"
PROFILER_STEPS="${PROFILER_STEPS:-1}"
RL_INSIGHT="${RL_INSIGHT:-1}"
RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"

# Shared overrides. All keys already exist in the composed verl config, so use
# plain `key=value` (a `+key=value` append fails with "An item is already at ...").
COMMON=(
  "global_profiler.tool=${PROFILER_TOOL}"
  "global_profiler.steps=[${PROFILER_STEPS}]"
  "global_profiler.profile_continuous_steps=false"
  "global_profiler.save_path=./outputs/profile"
)

case "$PROFILER_TOOL" in
  torch)
    OVERRIDES=(
      "${COMMON[@]}"
      "actor_rollout_ref.actor.profiler.enable=True"
      "actor_rollout_ref.actor.profiler.all_ranks=True"
      "actor_rollout_ref.actor.profiler.tool=torch"
      "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda]"
      "actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False"
    )
    ;;
  nsys)
    OVERRIDES=(
      "${COMMON[@]}"
      "actor_rollout_ref.actor.profiler.enable=True"
      "actor_rollout_ref.actor.profiler.all_ranks=True"
      "actor_rollout_ref.actor.profiler.tool=nsys"
    )
    ;;
  torch_memory)
    OVERRIDES=(
      "${COMMON[@]}"
      "actor_rollout_ref.actor.profiler.enable=True"
      "actor_rollout_ref.actor.profiler.all_ranks=True"
      "actor_rollout_ref.actor.profiler.tool=torch_memory"
    )
    ;;
  *)
    echo "[ERROR] unknown PROFILER_TOOL=${PROFILER_TOOL} (want torch|nsys|torch_memory)" >&2
    exit 2
    ;;
esac

# ── Layer 2: rl-insight online observability ────────────────────────────────
# Start/stop the Prometheus + Tempo + Grafana stack and export the server URL so
# the training driver + agent-loop workers emit metrics/traces through the Ray
# monitor hub. The Python instrumentation (verl_omni/agent_loop/rl_insight_profiler.py)
# is a silent no-op when `rl-insight` is not importable or the URL is unset, so
# layer 2 can be disabled at any time without touching the training code.
if [[ "${RL_INSIGHT}" == "1" ]]; then
  if ! command -v rl-insight >/dev/null 2>&1; then
    echo "[ERROR] rl-insight CLI not found. Install it into this environment first:" >&2
    echo "          pip install rl-insight" >&2
    echo "        (set RL_INSIGHT=0 to skip layer 2 and run layer 1 only)" >&2
    exit 2
  fi

  # Make the server URL reachable from both the driver and Ray workers.
  export RL_INSIGHT_SERVER_URL

  if [[ "${RL_INSIGHT_SERVER_STOP:-0}" == "1" ]]; then
    echo "[rl-insight] stopping server stack..."
    rl-insight server stop
    exit 0
  fi

  # `install` is idempotent (skips when binaries exist); `start --detach` runs
  # Prometheus/Tempo/Grafana in the background and returns immediately.
  echo "[rl-insight] ensuring server dependencies (one-time download if missing)..."
  rl-insight server install
  echo "[rl-insight] starting server stack (detached) at ${RL_INSIGHT_SERVER_URL}..."
  rl-insight server start --detach

  if [[ "${RL_INSIGHT_SERVER_ONLY:-0}" == "1" ]]; then
    echo "[rl-insight] server-only mode: stack is up. Grafana: http://<host>:3000"
    exit 0
  fi

  # Guard: the training env must import rl_insight for the facade to activate.
  # (verl-omni runs in its own venv; the package is optional there.)
  if ! python3 -c "import rl_insight" >/dev/null 2>&1; then
    echo "[WARN] rl-insight is not importable in the training python." >&2
    echo "       Layer 2 will be a silent no-op; layer 1 (${PROFILER_TOOL}) still runs." >&2
    echo "       To enable layer 2: pip install rl-insight into the training env." >&2
  else
    echo "[rl-insight] layer 2 enabled: metrics -> Prometheus:9090, traces -> Tempo:3200, dashboards -> Grafana:3000"
  fi
else
  echo "[rl-insight] layer 2 disabled (RL_INSIGHT=0); running layer 1 only."
fi

exec bash "$SCRIPT_DIR/run_agentic_rpco.sh" "${OVERRIDES[@]}" "$@"
