#!/usr/bin/env bash
# Frozen Lance-3B MoT image tool for Mode (2a) ToolAgentLoop.
#
# GRPO trains Lance_3B_hf_und (understanding) on GPUs in CUDA_VISIBLE_DEVICES
# from the operator env (e.g. 6,7). This script serves the **full** Lance MoT
# tree (moe_gen + Wan2.2 VAE) on a *different* GPU via vLLM-Omni so
# generate_image stays outside the actor optimizer.
#
# Usage (separate shell / tmux pane from training):
#   source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh   # for LD_LIBRARY_PATH
#   CUDA_VISIBLE_DEVICES=6 \
#     bash examples/agenticrpco_trainer/agent_llm/run_lance_frozen_diffusion_tool_server.sh
#
# Then point the trainer at this server:
#   export AGENTIC_LANCE_SERVER_URL=http://127.0.0.1:8091
#   bash examples/agenticrpco_trainer/agent_llm/run_agentic_grpo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Hub snapshot root (contains Lance_3B/, Wan2.2_VAE.pth, Qwen2.5-VL-ViT/).
# Do NOT point at Lance_3B_hf_und — that export dropped moe_gen.
LANCE_HUB_ROOT="${LANCE_HUB_ROOT:-${MODEL_PATH_LANCE_MOT:-}}"
if [[ -z "$LANCE_HUB_ROOT" ]]; then
  # Default: parent of the und export used by the trainer.
  if [[ -n "${MODEL_PATH:-}" && -d "$(dirname "$MODEL_PATH")/Lance_3B" ]]; then
    LANCE_HUB_ROOT="$(cd "$(dirname "$MODEL_PATH")" && pwd)"
  else
    LANCE_HUB_ROOT="/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub/models--bytedance-research--Lance/snapshots/7395315758865e6f56ab87ad06a88c7ac172f056"
  fi
fi

PORT="${PORT:-8091}"
# Prefer packaged deploy YAML from the active venv's vllm_omni.
DEPLOY_CONFIG="${DEPLOY_CONFIG:-}"
if [[ -z "$DEPLOY_CONFIG" ]]; then
  DEPLOY_CONFIG="$(python3 - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("vllm_omni")
if spec and spec.submodule_search_locations:
    p = Path(spec.submodule_search_locations[0]) / "deploy" / "lance.yaml"
    print(p if p.exists() else "")
PY
)"
fi
if [[ -z "$DEPLOY_CONFIG" || ! -f "$DEPLOY_CONFIG" ]]; then
  echo "[ERROR] Could not find vllm_omni/deploy/lance.yaml; set DEPLOY_CONFIG=" >&2
  exit 1
fi

if [[ ! -d "$LANCE_HUB_ROOT/Lance_3B" ]]; then
  echo "[ERROR] Expected full MoT tree at $LANCE_HUB_ROOT/Lance_3B" >&2
  exit 1
fi

echo "[INFO] Serving frozen Lance MoT tool"
echo "[INFO]   model root : $LANCE_HUB_ROOT"
echo "[INFO]   deploy cfg : $DEPLOY_CONFIG"
echo "[INFO]   port       : $PORT"
echo "[INFO]   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset — set to a GPU not used by GRPO>}"
echo "[INFO] Trainer should: export AGENTIC_LANCE_SERVER_URL=http://127.0.0.1:${PORT}"

# HF hub root (not Lance_3B_hf_und). vLLM-Omni Lance resolves Lance_3B/ + VAE.
exec vllm serve "$LANCE_HUB_ROOT" --omni \
  --deploy-config "$DEPLOY_CONFIG" \
  --port "$PORT" \
  "$@"
