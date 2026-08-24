#!/usr/bin/env bash
# Baseline profiling run for the agentic RPCO pipeline.
#
# Thin wrapper over run_agentic_rpco.sh that turns on the standard verl
# `global_profiler` + per-role `*.profiler.*` config so the FSDP actor / ref /
# vLLM rollout boundaries are captured into a Chrome trace / nsys report.
# Tool-level sidecar latency (generate_image / judge_image) is instrumented
# separately in function_tools/tools.py + agentic_metrics_manager.py and logged
# to wandb automatically — no extra flags needed here.
#
# Usage (three panes — gen sidecar, judge sidecar, then this):
#   pane A: CUDA_VISIBLE_DEVICES=0,1 bash .../run_image_gen_tool_server.sh
#   pane B: CUDA_VISIBLE_DEVICES=0   bash .../run_judge_image_tool_server.sh
#   pane C: CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 TOTAL_STEPS=4 \
#             bash .../run_agentic_rpco_profile.sh
#
# Env knobs:
#   PROFILER_TOOL    torch | nsys | torch_memory   (default torch)
#   PROFILER_STEPS   comma-separated step list     (default 1)
#
# See docs/perf/profiler.md for the full field reference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILER_TOOL="${PROFILER_TOOL:-torch}"
PROFILER_STEPS="${PROFILER_STEPS:-1}"

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

exec bash "$SCRIPT_DIR/run_agentic_rpco.sh" "${OVERRIDES[@]}" "$@"
