#!/usr/bin/env bash
# Standalone Bagel UND AR serve for spike_und_hermes.py (fail-closed gate).
#
# Default: physical GPUs 0-3 (4 free cards on this box) → logical 0=Thinker, 1=DiT.
# Same default as run_agentic_bagel_rpco_lora.sh so both see identical cards.
#
# This script preflights GPU 0 and refuses to start if the card is still held by
# a previous engine (>= GPU_MIN_FREE_PCT used); that collision otherwise dies as
# "torch.OutOfMemoryError: CUDA out of memory" during vllm-omni engine init.
# Stop this server before training — the trainer's colocated UND AR shares GPU 0.
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
# Use this repo's vllm-omni so the spike cannot silently serve from another
# checkout's venv that happens to come first on PATH (easy to hit when several
# verlomni-* trees each own a venv).
_VLLM_OMNI="${_REPO_ROOT}/.venv/bin/vllm-omni"
if [[ ! -x "${_VLLM_OMNI}" ]]; then
  _VLLM_OMNI="$(command -v vllm-omni)"
fi

# flashinfer JIT-compiles MM/attention kernels and needs nvcc. The CUDA
# toolkits on this box live under /cm/shared, not the /usr/local/cuda default,
# so without this the engine dies with:
#   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _cand in /cm/shared/apps/cuda-latest/toolkit/current /usr/local/cuda; do
    if [[ -x "${_cand}/bin/nvcc" ]]; then
      export CUDA_HOME="${_cand}"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" && ":${PATH}:" != *":${CUDA_HOME}/bin:"* ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-/scratch/fq9hpsac/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b}"
# Same default as run_agentic_bagel_rpco_lora.sh (0,1,2,3) so this standalone
# Hermes proof exercises the same cards the trainer uses. Honour the caller;
# never hard-assign (an unconditional export makes the default dead code).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8094}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
KV_ARENA_RATIO="${KV_ARENA_RATIO:-0.35}"
# Refuse to start unless the target card is at least this % free. Set to 0 to
# override (e.g. to deliberately co-tenant the AR engine with something else).
GPU_MIN_FREE_PCT="${GPU_MIN_FREE_PCT:-80}"
DEPLOY_CONFIG="${BAGEL_UND_DEPLOY_CONFIG:-${_SCRIPT_DIR}/bagel_corl_deploy_ar.yaml}"

_FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
_GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${_FIRST_GPU}" | head -1 | tr -d '[:space:]')"
_GPU_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${_FIRST_GPU}" | head -1 | tr -d '[:space:]')"
if [[ -z "${_GPU_TOTAL_MIB}" || ! "${_GPU_TOTAL_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU ${_FIRST_GPU} memory.total via nvidia-smi." >&2
  exit 1
fi
if [[ -z "${_GPU_FREE_MIB}" || ! "${_GPU_FREE_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU ${_FIRST_GPU} memory.free via nvidia-smi." >&2
  exit 1
fi

# Preflight: the KV arena below is a *fixed byte reservation*, so a card a
# previous engine still owns makes this run die deep inside vLLM with a wall of
# tracebacks whose only useful line is the last one:
#   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1020.00 MiB.
#     GPU 0 has a total capacity of 79.11 GiB of which 381.38 MiB is free.
#     Process <pid> has 31.43 GiB memory in use.
# Catch it here instead, and name the squatter so it can be reclaimed.
_GPU_USED_MIB=$(( _GPU_TOTAL_MIB - _GPU_FREE_MIB ))
_GPU_FREE_PCT=$(( _GPU_FREE_MIB * 100 / _GPU_TOTAL_MIB ))
if (( _GPU_FREE_PCT < GPU_MIN_FREE_PCT )); then
  echo "[ERROR] GPU ${_FIRST_GPU} is already occupied: ${_GPU_FREE_MIB} MiB free of ${_GPU_TOTAL_MIB} MiB (${_GPU_FREE_PCT}%, need >= ${GPU_MIN_FREE_PCT}%)." >&2
  echo "        ${_GPU_USED_MIB} MiB is held by:" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${_FIRST_GPU}" 2>/dev/null | sed 's/^/          /' >&2 || true
  echo "        A leftover 'vllm-omni serve' from an earlier run is the usual cause. Free it and retry:" >&2
  echo "          pkill -f 'vllm-omni serve'" >&2
  echo "        Or override the guard with GPU_MIN_FREE_PCT=0 if you really intend to co-tenant." >&2
  exit 1
fi

# Size the KV arena from *free* memory rather than total. Identical on an empty
# card (free ~= total) but it cannot overcommit a card that is already shared;
# a total-based figure is exactly what turns a busy card into the OOM above.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-$("${_PY}" -c "print(int(${_GPU_FREE_MIB} * 1024 * 1024 * float('${KV_ARENA_RATIO}')))")}"

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

exec "${_VLLM_OMNI}" serve "${BAGEL_MODEL_PATH}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --deploy-config "${DEPLOY_CONFIG}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --stage-overrides "${STAGE_OVERRIDES}" \
  "$@"
