#!/usr/bin/env bash
# Standalone Bagel UND AR serve for spike_und_hermes.py (fail-closed gate).
#
# Default: physical GPUs 2,3 (job 133253 IDX:2-5) → logical 0=Thinker, 1=DiT.
# Then prove Hermes:
#   BAGEL_UND_URL=http://127.0.0.1:8094 \
#     python3 examples/agenticllmgrpo_trainer/bagel/spike_und_hermes.py \
#       --model-path "$BAGEL_MODEL_PATH"
#
# Do not point the spike at bagel_single_stage / GEN. Do not use Qwen for UND.
set -euo pipefail
set -x

_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
_PY="${_REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${_PY}" ]]; then
  _PY="$(command -v python3)"
fi

export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-/scratch/fq9hpsac/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b}"
# Prefer job GPUs 2,3 for the spike; leave 4,5 free for later GEN/actor smoke.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8094}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
KV_ARENA_RATIO="${KV_ARENA_RATIO:-0.35}"
DEPLOY_CONFIG="${BAGEL_UND_DEPLOY_CONFIG:-${_SCRIPT_DIR}/bagel_corl_deploy_ar.yaml}"

_FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
_GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${_FIRST_GPU}" | head -1 | tr -d '[:space:]')"
if [[ -z "${_GPU_TOTAL_MIB}" || ! "${_GPU_TOTAL_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU ${_FIRST_GPU} memory.total via nvidia-smi." >&2
  exit 1
fi
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-$("${_PY}" -c "print(int(${_GPU_TOTAL_MIB} * 1024 * 1024 * float('${KV_ARENA_RATIO}')))")}"

# Two visible cards → split Thinker / DiT. One card → both on logical 0.
_N_VISIBLE=1
IFS=',' read -ra _ids <<< "${CUDA_VISIBLE_DEVICES}"
_N_VISIBLE="${#_ids[@]}"
if [[ -z "${STAGE_OVERRIDES:-}" ]]; then
  if (( _N_VISIBLE >= 2 )); then
    STAGE_OVERRIDES="$("${_PY}" -c "
import json
print(json.dumps({
  '0': {
    'devices': '0',
    'gpu_memory_utilization': float('${GPU_MEM_UTIL}'),
    'kv_cache_memory_bytes': int('${KV_CACHE_MEMORY_BYTES}'),
  },
  '1': {'devices': '1'},
}))
")"
  else
    STAGE_OVERRIDES="$("${_PY}" -c "
import json
print(json.dumps({
  '0': {
    'devices': '0',
    'gpu_memory_utilization': float('${GPU_MEM_UTIL}'),
    'kv_cache_memory_bytes': int('${KV_CACHE_MEMORY_BYTES}'),
  },
  '1': {'devices': '0'},
}))
")"
  fi
fi

echo "[INFO] UND AR serve model=${BAGEL_MODEL_PATH}"
echo "[INFO] deploy=${DEPLOY_CONFIG} host=${HOST} port=${PORT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} STAGE_OVERRIDES=${STAGE_OVERRIDES}"
echo "[INFO] spike: BAGEL_UND_URL=http://${HOST}:${PORT} python3 ${_SCRIPT_DIR}/spike_und_hermes.py --model-path \"\$BAGEL_MODEL_PATH\""

exec vllm-omni serve "${BAGEL_MODEL_PATH}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --deploy-config "${DEPLOY_CONFIG}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --stage-overrides "${STAGE_OVERRIDES}" \
  "$@"
