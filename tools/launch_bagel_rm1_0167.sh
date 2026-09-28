#!/usr/bin/env bash
# Launch the Bagel Co-RL RM-on e2e run on devices 0/1/6/7.
#
# Carries the full fix set that walked the GEN diffusion loss from "never runs"
# (skip_gen=True) to a healthy step, plus the agentic_rewards telemetry read:
#   diffusers_impl._run_forward_backward_batch micro_batch_size_per_gpu fallback
#   _reslice_per_row_non_tensors (per-row non-tensors into micro-batches)
#   trainer_base._compute_reward_colocate data_source/reward_model selection + unwrap
#   fold_gen_prompt_token_ids on the live actor-pool advantage path
#   losses: _loss_cfg_of / _align_view_to_device / _inherit_micro_batch_scalars
#   bagel_corl_composite: _flatten_losses + Metric-aware append_to_dict
#   trainer_base._extra_fields_rows (the 'LinkedList has no tolist' telemetry bug)
#
# Also regenerated _generated_omni_trainer.yaml against the runtime verl checkout
# (../verl, which the recipe prepends to PYTHONPATH), so the recipe-override guard
# test agrees with what Hydra actually composes.
set -u

REPO=/home/fq9hpsac/fq9hpsacuser11/fred/verlomni-fredfork
RECIPE="$REPO/examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh"
WD=/home/fq9hpsac/fq9hpsacuser11/fred/.gpu_watch
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$WD/bagel_rm1_${TS}.log"
mkdir -p "$WD"

echo "$LOG" > "$WD/last_bagel_rm1.log"
echo "[$(date '+%F %T')] launching RM-on ${TOTAL_STEPS:-200}-step run on 0/1/6/7 -> $LOG"
cd "$REPO" || exit 1

setsid nohup env \
  CUDA_VISIBLE_DEVICES=0,1,6,7 \
  ENABLE_RM=1 \
  TOTAL_STEPS="${TOTAL_STEPS:-200}" \
  RUN_NAME="bagel_corl_rm1_${TS}" \
  RAY_TMPDIR=/tmp/ray_fred_4gpu \
  RAY_DEDUP_LOGS=0 \
  PYTHONUNBUFFERED=1 \
  bash "$RECIPE" >"$LOG" 2>&1 </dev/null &
PID=$!
echo "$PID" > "$WD/bagel_rm1.pid"
echo "[$(date '+%F %T')] LAUNCHED pid=$PID log=$LOG"
