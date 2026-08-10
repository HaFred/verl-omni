#!/usr/bin/env bash
# Qwen3.5 / Qwen3-VL Mode (2a) agentic GRPO overfit.
#
# The actor is a pretrained tool-calling VLM (default: Qwen3.5-2B via MODEL_PATH).
# Frozen tools (vLLM-omni Qwen-Image + vLLM Qwen3-VL) serve generate_image and
# judge_image with continuous batching. The agent follows the gen→judge→reflect
# protocol: generate_image → judge_image(VL feedback) → Done / rewrite.
#
#   source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh
#   cd ~/fred/verlomni-pr-fredfork
#   # pane A — vLLM-omni image gen server (GPUs 0,1):
#   CUDA_VISIBLE_DEVICES=0,1 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
#   # pane B — vLLM VL judge server (GPU 2):
#   CUDA_VISIBLE_DEVICES=2 \
#     bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
#   # pane C — training (GPUs 4-7):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS=4 TOTAL_STEPS=100 \
#     bash examples/agenticrpco_trainer/agent_llm/run_agentic_grpo_lora.sh
#
# Short smokes (TOTAL_STEPS≤20) auto-enable GATE_SIDECAR=1: prints expected
# behavior, watches rollout_trajectories during train, then writes
# overfit_gates.json. Force/teacher default OFF so voluntary Hermes is measurable.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/agentic/train.parquet}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/data/agentic/val.parquet}"
N_GPUS="${N_GPUS:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
ROLLOUT_N="${ROLLOUT_N:-8}"
OVERFIT_DATA="${OVERFIT_DATA:-1}"
# Unique WandB run + ckpt dir every launch (override with EXPERIMENT_NAME / RUN_TS).
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
# Slug from MODEL_PATH: Hub id (Qwen/Qwen3-VL-2B-Instruct) or HF cache
# (.../models--Qwen--Qwen3-VL-2B-Instruct/snapshots/...).
if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
  MODEL_SLUG="$(
    python3 - "$MODEL_PATH" <<'PY'
import re, sys
from pathlib import Path
raw = (sys.argv[1] or "").strip().rstrip("/")
p = Path(raw)
text = raw
for part in reversed(p.parts):
    if part.startswith("models--") and "--" in part:
        text = part[len("models--"):].replace("--", "/")
        break
    if part not in {"snapshots", "refs", "blobs"} and not re.fullmatch(r"[0-9a-f]{8,}", part):
        # Prefer a leaf that looks like a model id/name.
        if "/" in raw and not raw.startswith("/"):
            text = raw
        elif part:
            text = part
        break
# Keep org/name if present; else bare name.
name = text.split("/")[-1] if text else "model"
slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
print(slug or "model")
PY
  )"
  EXPERIMENT_NAME="${MODEL_SLUG}_agentic_grpo_${RUN_TS}"
fi
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/verl_omni_agentic/${EXPERIMENT_NAME}}"
# Per-run rollout artifacts: PNGs (or stub manifests) land here when generate_image fires.
E2E_ROOT="${E2E_ROOT:-${REPO_ROOT}/outputs/e2e_qwen3_vl_2b_instruct_agentic_grpo}"
E2E_RUN_DIR="${E2E_RUN_DIR:-${E2E_ROOT}/${EXPERIMENT_NAME}}"
export AGENTIC_E2E_ROOT="${AGENTIC_E2E_ROOT:-$E2E_ROOT}"
export AGENTIC_E2E_RUN_NAME="${AGENTIC_E2E_RUN_NAME:-$EXPERIMENT_NAME}"
export AGENTIC_DIFFUSION_IMAGE_DIR="${AGENTIC_DIFFUSION_IMAGE_DIR:-$E2E_RUN_DIR/rollout_images}"
export AGENTIC_REFLECT_VLM_TIMEOUT=300
REQUIRE_REAL_IMAGE_TOOL="${REQUIRE_REAL_IMAGE_TOOL:-1}"
REQUIRE_REFLECT_VLM="${REQUIRE_REFLECT_VLM:-1}"
GATE_SCRIPT="${GATE_SCRIPT:-$SCRIPT_DIR/check_overfit_gates.py}"
DATA_SCRIPT="${DATA_SCRIPT:-$SCRIPT_DIR/../data_process/create_dummy_agentic_data.py}"
# Sidecar gates for overfit / smoke runs (TOTAL_STEPS≤50 default on). Override GATE_SIDECAR=0 to skip.
if [[ -z "${GATE_SIDECAR:-}" ]]; then
  if [[ "${TOTAL_STEPS}" -le 50 ]]; then
    GATE_SIDECAR=1
  else
    GATE_SIDECAR=0
  fi
fi
GATE_PID=""

# Online WandB + local rollout_images. TCP transport avoids Ray/uvloop Unix socket
# crashes (UnixTransport closed). WANDB_DIR keeps run files under this e2e folder.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SERVICE_TRANSPORT="${WANDB_SERVICE_TRANSPORT:-tcp}"
export WANDB_DIR="${WANDB_DIR:-$E2E_RUN_DIR/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_omni_agentic}"
mkdir -p "$AGENTIC_DIFFUSION_IMAGE_DIR" "$E2E_RUN_DIR" "$WANDB_DIR"

# Fixed diffusion seed for overfit visual stability (override to empty to randomize).
export QWEN_IMAGE_SEED="${QWEN_IMAGE_SEED:-42}"
export AGENTIC_LANCE_SEED="${AGENTIC_LANCE_SEED:-${QWEN_IMAGE_SEED}}"
# Judge parse reliability (retry once by default; higher max tokens).
export AGENTIC_JUDGE_PARSE_RETRIES="${AGENTIC_JUDGE_PARSE_RETRIES:-1}"
export AGENTIC_REFLECT_MAX_NEW_TOKENS="${AGENTIC_REFLECT_MAX_NEW_TOKENS:-1024}"
# YES bar: 0.9, if lower than this for the overfitting, a single turn mostly get the reflection returing Done. We want more turns incited by the RL.
export AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD="${AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD:-0.9}"
export AGENTIC_REFLECT_GOOD_ENOUGH="${AGENTIC_REFLECT_GOOD_ENOUGH:-${AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD}}"
# YES bar: A≈0.84–0.86 under STEPS=8. thr 0.80 → first-judge YES → one-shot Done
# and inflated early critic/score. thr 0.90 forces NO → rewrite multiturn.
# Env hard-stop: refuse generate_image after good_enough=YES (default on).
export AGENTIC_BLOCK_GENERATE_AFTER_YES="${AGENTIC_BLOCK_GENERATE_AFTER_YES:-1}"
# Cap rewrite roulette at AGENTIC_MAX_GENERATE_IMAGE_PASSES (default on).
export AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES="${AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES:-1}"
SAVE_FREQ="${SAVE_FREQ:-5}"
MIN_JUDGE_PARSE_OK_RATE="${MIN_JUDGE_PARSE_OK_RATE:-0.99}"

cat >"$E2E_RUN_DIR/README.txt" <<EOF
e2e run: ${EXPERIMENT_NAME}
wandb: mode=${WANDB_MODE} project=${WANDB_PROJECT} experiment=${EXPERIMENT_NAME}
wandb dir: ${WANDB_DIR}
ckpt: ${CKPT_DIR}
rollout images: ${AGENTIC_DIFFUSION_IMAGE_DIR}
rollout trajectories: ${E2E_RUN_DIR}/rollout_trajectories
raw assistant rollouts (per step): ${E2E_RUN_DIR}/hermes_actions/step_XXXXXX.txt
agent model: ${MODEL_PATH}
vLLM-omni image URL: ${AGENTIC_VLLM_OMNI_URL:-<unset>}  legacy: ${AGENTIC_QWEN_IMAGE_URL:-<unset>}
vLLM judge URL:     ${AGENTIC_VLLM_URL:-<unset>}  legacy: ${AGENTIC_REFLECT_VLM_URL:-<unset>}
judge model path:   ${AGENTIC_REFLECT_VLM_PATH:-<defaults to MODEL_PATH / Qwen3-VL>}
overfit gates: GATE_SIDECAR=${GATE_SIDECAR} (see overfit_gates.json)
agent loop: agentic_tool_agent (force Reflection after judge ON; max generate_image passes=${AGENTIC_MAX_GENERATE_IMAGE_PASSES:-3})
good_enough threshold: ${AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD:-0.85} (headroom for multiturn ΔC)
block generate after YES: ${AGENTIC_BLOCK_GENERATE_AFTER_YES:-1}
block generate after max passes: ${AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES:-1}
rollout_n: ${ROLLOUT_N}
EOF
echo "[INFO] repo root=${REPO_ROOT}"
echo "[INFO] wandb online experiment_name=${EXPERIMENT_NAME} (WANDB_SERVICE_TRANSPORT=${WANDB_SERVICE_TRANSPORT})"
echo "[INFO] ckpt dir=${CKPT_DIR}"
echo "[INFO] e2e rollout images -> ${AGENTIC_DIFFUSION_IMAGE_DIR}"
echo "[INFO] e2e full trajectories -> ${E2E_RUN_DIR}/rollout_trajectories"
echo "[INFO] e2e raw assistant rollouts -> ${E2E_RUN_DIR}/hermes_actions/"
# After every successful judge_image, inject Reflection that carries the
# stop/continue verdict (Done. on YES / max-pass; else rewrite next).
# Tokens are response_mask=0; reward strips agentic_forced_reflection markers.
export AGENTIC_FORCE_REFLECTION_AFTER_JUDGE="${AGENTIC_FORCE_REFLECTION_AFTER_JUDGE:-1}"
export AGENTIC_MAX_GENERATE_IMAGE_PASSES="${AGENTIC_MAX_GENERATE_IMAGE_PASSES:-5}"
export AGENTIC_FORCE_FIRST_GENERATE="${AGENTIC_FORCE_FIRST_GENERATE:-1}"
export AGENTIC_FORCE_FIRST_WARMUP_STEPS="${AGENTIC_FORCE_FIRST_WARMUP_STEPS:-10}"
export AGENTIC_FORCE_FIRST_END_STEP="${AGENTIC_FORCE_FIRST_END_STEP:-20}"
echo "[INFO] agent loop=agentic_tool_agent (AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=${AGENTIC_FORCE_REFLECTION_AFTER_JUDGE}; max generate_image passes=${AGENTIC_MAX_GENERATE_IMAGE_PASSES})"
echo "[INFO] force-first generate=${AGENTIC_FORCE_FIRST_GENERATE} p=1 through step ${AGENTIC_FORCE_FIRST_WARMUP_STEPS}, linear -> 0 at step ${AGENTIC_FORCE_FIRST_END_STEP} (teacher-force Hermes gen+judge; response_mask=1)"
echo "[INFO] good_enough threshold=${AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD} block_generate_after_yes=${AGENTIC_BLOCK_GENERATE_AFTER_YES} block_after_max_passes=${AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES} rollout_n=${ROLLOUT_N}"
if [[ "${AGENTIC_FORCE_REFLECTION_AFTER_JUDGE}" == "0" ]]; then
  echo "[WARN] AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=0: Reflection after judge is policy-only; max-pass soft-stop still applies." >&2
fi
echo "[INFO] agent MODEL_PATH=${MODEL_PATH}"
echo "[INFO] reflect judge vLLM URL=${AGENTIC_VLLM_URL:-<unset>} legacy=${AGENTIC_REFLECT_VLM_URL:-<unset>} path=${AGENTIC_REFLECT_VLM_PATH:-<unset>}"
python3 "$GATE_SCRIPT" --run-dir "$E2E_RUN_DIR" --expect-only --total-steps "${TOTAL_STEPS}" --no-force
if [[ -z "${AGENTIC_VLLM_OMNI_URL:-}" && -z "${AGENTIC_QWEN_IMAGE_URL:-}" && -z "${AGENTIC_DIFFUSION_TOOL_URL:-}" && -z "${AGENTIC_LANCE_SERVER_URL:-}" ]]; then
  echo "[ERROR] No frozen image service is configured; visual reflection cannot be trained on stubs." >&2
  echo "[ERROR] Start: CUDA_VISIBLE_DEVICES=<free_gpu> bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh" >&2
  if [[ "${REQUIRE_REAL_IMAGE_TOOL}" == "1" ]]; then
    exit 2
  fi
fi
if [[ -z "${AGENTIC_VLLM_URL:-}" && -z "${AGENTIC_REFLECT_VLM_URL:-}" ]]; then
  echo "[ERROR] AGENTIC_VLLM_URL and AGENTIC_REFLECT_VLM_URL are both unset; judge_image has no backend." >&2
  echo "[ERROR] Start: CUDA_VISIBLE_DEVICES=<free_gpu> bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh" >&2
  if [[ "${REQUIRE_REFLECT_VLM}" == "1" ]]; then
    exit 2
  fi
fi
if [[ -n "${AGENTIC_VLLM_OMNI_URL:-}" ]]; then
  python3 - "${AGENTIC_VLLM_OMNI_URL}" "${REQUIRE_REAL_IMAGE_TOOL}" <<'PY'
import json, sys
from urllib.request import urlopen, Request

base = sys.argv[1].rstrip("/")
required = sys.argv[2] == "1"

def _get(path: str):
    req = Request(f"{base}{path}", method="GET")
    with urlopen(req, timeout=5) as resp:  # noqa: S310
        return resp.status, resp.read()

try:
    # vLLM-Omni /health is often HTTP 200 with an empty body.
    status, body = _get("/health")
    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    if body.strip():
        print(f"[INFO] vLLM-omni health OK: {json.loads(body)}")
    else:
        status, body = _get("/v1/models")
        if status != 200:
            raise RuntimeError(f"/health empty and /v1/models HTTP {status}")
        models = json.loads(body) if body.strip() else {}
        n = len(models.get("data") or [])
        print(f"[INFO] vLLM-omni health OK (empty /health); /v1/models count={n}")
except Exception as exc:
    print(f"[ERROR] vLLM-omni health check failed at {base}/health: {exc}", file=sys.stderr)
    if required:
        raise SystemExit(2)
PY
elif [[ -n "${AGENTIC_QWEN_IMAGE_URL:-}" ]]; then
  python3 - "${AGENTIC_QWEN_IMAGE_URL}" "${REQUIRE_REAL_IMAGE_TOOL}" <<'PY'
import json, sys
from urllib.request import urlopen

endpoint, required = sys.argv[1], sys.argv[2] == "1"
health = endpoint.rsplit("/", 1)[0] + "/health"
try:
    with urlopen(health, timeout=5) as response:  # noqa: S310
        payload = json.loads(response.read())
    if not payload.get("ok"):
        raise RuntimeError(f"unhealthy response: {payload}")
    print(f"[INFO] Qwen-Image health OK: {payload}")
except Exception as exc:
    print(f"[ERROR] Qwen-Image health check failed at {health}: {exc}", file=sys.stderr)
    if required:
        raise SystemExit(2)
PY
fi
if [[ -n "${AGENTIC_VLLM_URL:-}" ]]; then
  python3 - "${AGENTIC_VLLM_URL}" "${REQUIRE_REFLECT_VLM}" <<'PY'
import json, sys
from urllib.request import urlopen, Request

base = sys.argv[1].rstrip("/")
required = sys.argv[2] == "1"

def _get(path: str):
    req = Request(f"{base}{path}", method="GET")
    with urlopen(req, timeout=5) as resp:  # noqa: S310
        return resp.status, resp.read()

try:
    status, body = _get("/health")
    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    if body.strip():
        print(f"[INFO] vLLM judge health OK: {json.loads(body)}")
    else:
        status, body = _get("/v1/models")
        if status != 200:
            raise RuntimeError(f"/health empty and /v1/models HTTP {status}")
        models = json.loads(body) if body.strip() else {}
        n = len(models.get("data") or [])
        print(f"[INFO] vLLM judge health OK (empty /health); /v1/models count={n}")
except Exception as exc:
    print(f"[ERROR] vLLM judge health check failed at {base}/health: {exc}", file=sys.stderr)
    if required:
        raise SystemExit(2)
PY
elif [[ -n "${AGENTIC_REFLECT_VLM_URL:-}" ]]; then
  python3 - "${AGENTIC_REFLECT_VLM_URL}" "${REQUIRE_REFLECT_VLM}" <<'PY'
import json, sys
from urllib.request import Request, urlopen

endpoint, required = sys.argv[1], sys.argv[2] == "1"
health = endpoint.rsplit("/", 1)[0] + "/health"
try:
    with urlopen(health, timeout=5) as response:  # noqa: S310
        payload = json.loads(response.read())
    print(f"[INFO] Reflect VLM health OK: {payload}")
except Exception:
    try:
        req = Request(
            endpoint,
            data=json.dumps(
                {
                    "user_request": "health check",
                    "image_prompt": "a red apple",
                    "notes": "",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read())
        print(f"[INFO] Reflect VLM endpoint reachable: keys={sorted(payload)[:8]}")
    except Exception as exc:
        print(f"[ERROR] Reflect VLM check failed at {endpoint}: {exc}", file=sys.stderr)
        if required:
            raise SystemExit(2)
PY
fi

_run_final_gates() {
  local train_rc="${1:-0}"
  if [[ -n "${GATE_PID}" ]] && kill -0 "${GATE_PID}" 2>/dev/null; then
    kill "${GATE_PID}" 2>/dev/null || true
    wait "${GATE_PID}" 2>/dev/null || true
  fi
  if [[ "${GATE_SIDECAR}" != "1" ]]; then
    return 0
  fi
  echo "[GATE] running final overfit gates on ${E2E_RUN_DIR}"
  if python3 "$GATE_SCRIPT" --run-dir "$E2E_RUN_DIR" --final --total-steps "${TOTAL_STEPS}" --no-force \
      --min-judge-parse-ok-rate "${MIN_JUDGE_PARSE_OK_RATE}"; then
    echo "[GATE] final gates PASS (train_rc=${train_rc})"
    return 0
  else
    echo "[GATE] final gates FAIL (train_rc=${train_rc})" >&2
    return 2
  fi
}

# Qwen3.5 helpers (GDN_PREFILL_BACKEND + TOOL_PARSER_FORMAT + GDN preflight).
# Safe no-op/preflight-skip when the actor is Qwen3-VL Hermes.
# shellcheck source=../../../data/qwen35_env.sh
source "${REPO_ROOT}/data/qwen35_env.sh"

# Refresh Ray worker env so tool artifacts / Qwen-Image URL / WandB reach TaskRunner.
export RAY_RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json, os
keys = [
    "LD_LIBRARY_PATH",
    "PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "HF_HOME",
    "VERL_USE_EXTERNAL_MODULES",
    "WANDB_API_KEY",
    "WANDB_MODE",
    "WANDB_SERVICE_TRANSPORT",
    "WANDB_SILENT",
    "WANDB_DIR",
    "WANDB_PROJECT",
    "AGENTIC_VLLM_OMNI_URL",
    "AGENTIC_VLLM_URL",
    "AGENTIC_QWEN_IMAGE_URL",
    "AGENTIC_REFLECT_VLM_URL",
    "AGENTIC_REFLECT_VLM_PATH",
    "AGENTIC_LANCE_SERVER_URL",
    "AGENTIC_DIFFUSION_TOOL_URL",
    "AGENTIC_DIFFUSION_TOOL_TOKEN",
    "AGENTIC_DIFFUSION_TOOL_TIMEOUT",
    "AGENTIC_DIFFUSION_IMAGE_DIR",
    "AGENTIC_E2E_ROOT",
    "AGENTIC_E2E_RUN_NAME",
    "AGENTIC_LANCE_HEIGHT",
    "AGENTIC_LANCE_WIDTH",
    "AGENTIC_LANCE_STEPS",
    "AGENTIC_LANCE_SEED",
    "AGENTIC_LANCE_CFG_TEXT_SCALE",
    "AGENTIC_FORCE_REFLECTION_AFTER_JUDGE",
    "AGENTIC_MAX_GENERATE_IMAGE_PASSES",
    "AGENTIC_FORCE_FIRST_GENERATE",
    "AGENTIC_FORCE_FIRST_WARMUP_STEPS",
    "AGENTIC_FORCE_FIRST_END_STEP",
    "AGENTIC_REFLECT_VLM_TIMEOUT",
    "AGENTIC_REFLECT_MAX_NEW_TOKENS",
    "AGENTIC_JUDGE_PARSE_RETRIES",
    "AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD",
    "AGENTIC_REFLECT_GOOD_ENOUGH",
    "AGENTIC_BLOCK_GENERATE_AFTER_YES",
    "AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES",
    "QWEN_IMAGE_SEED",
]
base = {}
try:
    base = json.loads(os.environ.get("RAY_RUNTIME_ENV_JSON") or "{}")
except Exception:
    base = {}
env_vars = dict(base.get("env_vars") or {})
for k in keys:
    if os.environ.get(k):
        env_vars[k] = os.environ[k]
base["env_vars"] = env_vars
print(json.dumps(base))
PY
)"

# Never replace the model's native template. Actor must expose Hermes tools +
# an image processor so generate_image pixels can be attached for reflection.
python3 - "$MODEL_PATH" <<'PY'
import sys
from transformers import AutoProcessor

model_path = sys.argv[1]
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
template = getattr(processor, "chat_template", "") or getattr(processor.tokenizer, "chat_template", "")
if "<tool_call>" not in template or "tools" not in template:
    raise SystemExit(f"{model_path} does not expose the required tool-aware chat template")
if not getattr(processor, "image_processor", None):
    raise SystemExit(f"{model_path} does not expose an image_processor")
print(f"[INFO] Verified native tool template + image processor: {model_path}")
PY

# Keep the tiny overfit parquet synchronized with the actor's native tool format.
# Default: GSM8K-style system+user only (no baked fewshot). Set OVERFIT_FEWSHOT=1
# for Class-1 same-task fewshot on soldier rows only (epic rows stay system+user).
if [[ "${OVERFIT_DATA}" == "1" ]]; then
  FEWSHOT_ARGS=()
  if [[ "${OVERFIT_FEWSHOT:-0}" == "1" ]]; then
    FEWSHOT_ARGS+=(--with_fewshot)
  fi
  python3 "${DATA_SCRIPT}" \
    --local_save_dir "$(dirname "$TRAIN_FILE")" \
    --overfit --train_size "${OVERFIT_TRAIN_SIZE:-8}" --val_size "${OVERFIT_VAL_SIZE:-2}" \
    --tool_call_format "${TOOL_PARSER_FORMAT}" \
    --model_path "${MODEL_PATH}" \
    "${FEWSHOT_ARGS[@]}"
fi

# Colocated FSDP actor + vLLM: after actor init, free VRAM << gpu_memory_utilization
# * total. Cap util from currently free memory. Also refuse to start if a prior
# crashed VLLM::EngineCore still owns the GPU (seen as "Free memory ... less than
# desired GPU memory utilization").
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
export MIN_FREE_GB="${MIN_FREE_GB:-24}"
if command -v nvidia-smi >/dev/null 2>&1; then
  python3 - <<PY
import os, subprocess, sys
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
min_free_gb = float(os.environ.get("MIN_FREE_GB", "24"))
bad = []
for i in idxs:
    if i >= len(raw):
        continue
    parts = [x.strip() for x in raw[i].split(",")]
    idx, total, free, used = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    print(f"[INFO] GPU {idx}: used={used/1024:.1f}GiB free={free/1024:.1f}GiB / {total/1024:.1f}GiB")
    if free / 1024.0 < min_free_gb:
        bad.append((idx, free / 1024.0, used / 1024.0))
if bad:
    print("[ERROR] Training GPUs do not have enough free VRAM before launch.", file=sys.stderr)
    print("[ERROR] A prior crashed VLLM::EngineCore often leaves ~60GiB occupied.", file=sys.stderr)
    for idx, free, used in bad:
        print(f"[ERROR]   GPU {idx}: free={free:.1f}GiB used={used:.1f}GiB (need >= {min_free_gb}GiB free)", file=sys.stderr)
    print("[ERROR] Inspect: nvidia-smi", file=sys.stderr)
    print("[ERROR] Free zombies (keep the separate Qwen-Image server GPU if running):", file=sys.stderr)
    print("[ERROR]   nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv", file=sys.stderr)
    print("[ERROR]   kill <EngineCore/Worker pids on CUDA_VISIBLE_DEVICES>   # or pick empty GPUs", file=sys.stderr)
    print("[ERROR] Override gate with MIN_FREE_GB=0 if you insist.", file=sys.stderr)
    sys.exit(2)
PY
fi
if [[ -z "$GPU_MEM_UTIL" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_UTIL="$(python3 - <<'PY'
import os, subprocess
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
idxs = [int(x) for x in vis.split(",") if x.strip() != ""] if vis.strip() else list(range(len(raw)))
free_fracs = []
for i in idxs:
    if i >= len(raw):
        continue
    total_s, free_s = [x.strip() for x in raw[i].split(",")]
    total, free = float(total_s), float(free_s)
    if total > 0:
        # Leave headroom for FSDP residual + LoRA sync / wake_up fragmentation.
        free_fracs.append(0.25 * free / total)
if not free_fracs:
    print("0.12")
else:
    util = max(0.08, min(0.15, min(free_fracs)))
    print(f"{util:.2f}")
PY
)"
fi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.12}"
echo "[INFO] rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"

# Context budget for all-3-class overfit fewshot (gen→judge→decide ≈6–7k toks)
# plus live user turn / chat template / tool schemas. Override via env if needed.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
# 3-pass protocol needs ~7 assistant msgs (gen/judge/rewrite…); leave margin.
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-12}"
MAX_USER_TURNS="${MAX_USER_TURNS:-12}"

if [[ "${GATE_SIDECAR}" == "1" ]]; then
  echo "[GATE] starting watch sidecar -> ${E2E_RUN_DIR}/gate_watch.log"
  python3 "$GATE_SCRIPT" \
    --run-dir "$E2E_RUN_DIR" \
    --watch \
    --total-steps "${TOTAL_STEPS}" \
    --interval-s "${GATE_INTERVAL_S:-20}" \
    --no-force \
    --min-judge-parse-ok-rate "${MIN_JUDGE_PARSE_OK_RATE}" \
    >"$E2E_RUN_DIR/gate_watch.log" 2>&1 &
  GATE_PID=$!
  echo "[GATE] watch pid=${GATE_PID}"
fi

# Overfit LoRA LR: verl default is 1e-6 (full-FT scale) which leaves the mean
# reward flat for 100 steps. LoRA overfit needs ~1e-4 (override via ACTOR_LR).
ACTOR_LR="${ACTOR_LR:-1e-4}"
ACTOR_WD="${ACTOR_WD:-0.01}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
echo "[INFO] actor.optim.lr=${ACTOR_LR} weight_decay=${ACTOR_WD} ppo_epochs=${PPO_EPOCHS} rollout.temperature=${ROLLOUT_TEMPERATURE}"

TRAIN_RC=0
# Mode (2a): optimize only Qwen3-VL language projections; ViT stays pretrained.
# layered_summon=false: FSDP LoRA leaf wraps + Qwen3-VL nesting produce names like
# ...lora_A.default._fsdp_wrapped_module.weight that vLLM rejects. Full summon is fine at 2B.
set +e
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=2 \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  data.filter_overlong_prompts=true \
  data.truncation=left \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=16 \
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding=true \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}" \
  "actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WD}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.layered_summon=false \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=${GDN_PREFILL_BACKEND}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}" \
  "actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  "actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER_FORMAT}" \
  actor_rollout_ref.rollout.multi_turn.function_tool_path=verl_omni/agent_loop/diffusion_tool.py \
  actor_rollout_ref.rollout.agent.default_agent_loop=agentic_tool_agent \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager.AgenticMetricsAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_reward \
  reward.custom_reward_function.name=compute_score \
  trainer.val_before_train=false \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.total_epochs="${TOTAL_STEPS}" \
  trainer.test_freq=-1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  "trainer.default_local_dir=${CKPT_DIR}" \
  'trainer.logger=["console","wandb"]' \
  trainer.project_name=verl_omni_agentic \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "$@"
TRAIN_RC=$?
set -e

GATE_RC=0
_run_final_gates "${TRAIN_RC}" || GATE_RC=$?
if [[ "${TRAIN_RC}" -ne 0 ]]; then
  exit "${TRAIN_RC}"
fi
exit "${GATE_RC}"
