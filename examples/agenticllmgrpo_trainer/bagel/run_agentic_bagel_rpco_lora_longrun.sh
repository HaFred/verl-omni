# Bagel UND+GEN Co-RL (Joint-Training) (PR1 correctness). Entry: ${_PY} -m verl_omni.trainer.main_omni
#
# The trainer MUST go through ``${_PY}``, never a bare ``python3``: this box's PATH
# resolves ``python3`` to miniconda, which has no ``orjson``, so the run dies at import
# with ``ModuleNotFoundError: No module named 'orjson'`` (via the local ``../verl``
# checkout's ``utils/tracking.py``). It only ever worked when the caller's shell happened
# to have the venv active, which is exactly the class of "works on my shell" the ``_PY``
# block at the top exists to remove.
#
# Fail-closed: UND must be the published Bagel checkpoint (Hermes tool-call).
# Do not swap Qwen3-VL for UND. Do not edit Mode (2a) run_agenticrpco_grpo_lora.sh.
#
# Required on vllm-omni >= 0.24: actor_rollout_ref.model.lora.merge=True.
# Fused MoT gen experts cannot bind *_moe_gen adapters (verl-omni#552 / vllm-omni#7190);
# merged full-weight sync is the same fix as run_bagel_ocr_lora.sh.
#
# UniCoT parquet must carry extra_info.reference_image_path. Rebuild with:
#   REBUILD_UNICOT=1 python3 examples/agenticllmgrpo_trainer/bagel/stamp_unicot_reference_paths.py \
#       --input $UNICOT_PARQUET --output $UNICOT_PARQUET
#
# GPUs: trainer.n_gpus_per_node = len(CUDA_VISIBLE_DEVICES) (e.g. 0,1,2,3 → 4).
# UND AR is colocated on that same actor placement group (not a second Ray pool).
# Default REWARD_TP=N so one Qwen RM is TP-sharded, not copied per GPU.
# 1-GPU smoke (CUDA_VISIBLE_DEVICES=3): ENABLE_RM=0 by default — actor+Omni+RM cannot fit.
# Re-enable with ENABLE_RM=1 once you have headroom.
#
# ENABLE_RM=1 also needs a *scoring* judge: the RM pool only carries the in-loop handle,
# and ``bagel_rm_image_scorer`` gets its C/A numbers from the frozen VL sidecar over HTTP
# (``agentic_image_gen.vllm_url``). So ENABLE_RM=1 defaults ``JUDGE_SERVER=1``, which starts
# ``agent_llm/run_judge_image_tool_server.sh`` on ``JUDGE_GPU`` (default: the last visible
# device) and proves it reachable before Ray comes up. Set ``JUDGE_SERVER=0`` plus
# ``JUDGE_URL`` to reuse a sidecar you started yourself; ``JUDGE_ENV_FILE`` points at an
# operator env (e.g. ``~/fred/fred_verlomni_agentic_multiturn_pr1.sh``) to pick up
# ``JUDGE_IMAGE_MODEL`` without sourcing it.
set -x
# Prefer local verl checkout (tokenizer package layout) over site-packages flat tokenizer.py.
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
_LOCAL_VERL="${_LOCAL_VERL:-${_REPO_ROOT}/../verl}"
if [[ -d "${_LOCAL_VERL}/verl" ]]; then
  export PYTHONPATH="${_LOCAL_VERL}:${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
# Same interpreter convention as run_bagel_und_ar_serve.sh: prefer this repo's
# venv (its editable install points here) and fall back to PATH. Without this,
# bare `python3` picks up whichever verlomni-* venv is active and the trainer can
# load another checkout's vllm_omni while PYTHONPATH points at this tree.
_PY="${_REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${_PY}" ]]; then
  _PY="$(command -v python3)"
fi

# One self-describing output folder per run: ``outputs/bagel_corl_<YYYYMMDD>_<hhmmss>``.
#
# Hydra's default run dir is the two-level ``outputs/<date>/<time>``, so the box fills up
# with one directory per calendar day holding a bag of bare ``HH-MM-SS`` names -- hard to
# name in a bug report and impossible to sort across day boundaries. Flatten it to a
# single folder that carries both the recipe and the stamp, matching the sibling e2e
# recipe's ``outputs/e2e/agentic_rpco_<ts>`` convention.
#
# ``RUN_TS`` is computed exactly once (here) for the whole run, and ``RUN_DIR`` follows
# from it, so the folder a run logs into is derivable from the launch time alone.
# Overridable for a re-launch that must land in an *existing* folder -- pass the same
# ``RUN_TS`` (e.g. ``RUN_TS=20260921_215500 bash run_agentic_bagel_rpco_lora.sh``), since
# Hydra refuses to write into a run dir that already exists.
#
# This is the hydra run dir: it holds ``.hydra/`` (the composed config + overrides) and
# ``main_omni.log``, and -- via ``agentic_image_gen.run_dir=$RUN_DIR`` below -- the three
# rollout artifact trees (``rollout_trajectories/``, ``rollout_images/``,
# ``hermes_actions/``), so a run's dumps are self-contained and cannot collide with
# another run that shares ``EXPERIMENT_NAME``. The checkpoints still go to
# ``checkpoints/<project_name>/<experiment_name>``, keyed on ``trainer.experiment_name``
# and therefore *shared* by every run that leaves ``EXPERIMENT_NAME`` alone. Set
# ``DEFAULT_LOCAL_DIR`` to stamp the checkpoints too (e.g.
# ``DEFAULT_LOCAL_DIR=$RUN_DIR/checkpoints``), which is what makes a run fully self-contained.
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-bagel_corl_${RUN_TS}}"
RUN_DIR="${RUN_DIR:-${_REPO_ROOT}/outputs/${RUN_NAME}}"
# Empty by default: the config's ``checkpoints/${project_name}/${experiment_name}`` stands.
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-}"
echo "[INFO] bagel_corl run dir: ${RUN_DIR} (RUN_TS=${RUN_TS})"

# UND AR sampling knobs. The Co-RL rollout is a Hermes tool-call decode: the AR lane has to
# emit ``<tool_call>{"name": "generate_image"...}`` for the GEN half to run at all. The
# vLLM/rollout defaults (temperature 1.0, top_p 1, top_k -1, repetition_penalty 1.0) are
# untruncated sampling with no anti-repetition pressure, and the published Bagel checkpoint
# degenerates under them. Measured 2026-09-20 on hk01dgx039: 940 of 943 UND turns came back
# as one role label repeated until the 1024-token budget was spent
# (``text='assistant\nassistant\nassistant\n...'``); ``und_turn_kind`` classified every one
# ``continue``, so K stayed 0, ``gen/num_rows`` was 0 and the GEN lane was never asked --
# every step reported ``skip_gen=True`` and ``critic/score/mean: 0.0`` for 60 steps.
# The standalone Hermes proof (``spike_und_hermes.py``) succeeds at temperature 0.7 with a
# 256-token cap, which is the shape reproduced here.
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-50}"
# ``rollout.repetition_penalty`` is only effective because the agent loops now read it from
# config; they hardcoded 1.0 before, which silently discarded this knob.
ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-1.05}"

export BAGEL_MODEL_PATH=/scratch/fq9hpsac/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b
WORKSPACE=${WORKSPACE:-$HOME}
# ``BAGEL_DEPLOY_CONFIG`` is selected further down, together with ``ROLLOUT_TP``: the
# GEN deploy yaml's per-replica device template has to match the GEN engine's width.
# 4 cards are free on this box (0-3); 4-7 are occupied by other jobs.
# Honour the caller and default to 0,1,2,3 — the same default as
# run_bagel_und_ar_serve.sh, so the optional Hermes proof and the trainer see
# the same cards. Never hard-assign here: an unconditional export clobbers a
# caller/job allocation and makes the default dead code.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# flashinfer JIT-compiles MM/attention kernels and needs nvcc. The CUDA
# toolkits on this box live under /cm/shared, not the /usr/local/cuda default,
# so without this the vllm_omni engine dies with:
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

# Driver 535.161.08 on this box tops out at CUDA 12.2, but the venv's torch is built
# for CUDA 13.0 (`torch 2.13.0+cu130`). Without a forward-compat `libcuda` every CUDA
# call dies at init with:
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12020)
# and it surfaces misleadingly early -- the *dataset* build imports
# `vllm_omni.diffusion.envs`, whose import-time PackageEnvChecker probes
# `torch.cuda.get_device_name()`, so the recipe aborts on the UniCoT parquet with no
# hint that CUDA is the cause. `.venv/cuda-compat` carries `libcuda.so.580.95.05` (the
# same shim the sibling boogu venv uses); prepend it plus the venv's own cu13 runtime.
# `PYTORCH_NVML_BASED_CUDA_CHECK=1` lets torch answer `is_available()` from NVML.
# Only applied when the shim is present, so a host with a native CUDA 13 driver keeps
# its own library and this stays a no-op there.
_cuda_compat="${_REPO_ROOT}/.venv/cuda-compat"
_cu13_lib="${_REPO_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "${_cuda_compat}" ]]; then
  export LD_LIBRARY_PATH="${_cuda_compat}:${_cu13_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
fi

model_name=${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}
reward_model_name=${REWARD_MODEL:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc}

# UniCoT sources for the parquet build below. These MUST have defaults here: the
# launch used to pass them through with no ``:-`` fallback, so an unset shell got
#   build_unicot_agentic_rl.py: error: argument --mix_ratio/--reflect_ratio:
#   invalid float value: ''
# and ``--seed ''`` besides. ``bagel_gpu_test_env.sh`` carries the same defaults,
# but the recipe neither sources it nor requires it, so a clean shell could not
# launch. Values mirror that script.
UNICOT_REFLECTION_DIR=${UNICOT_REFLECTION_DIR:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub/datasets--Fr0zencr4nE--UniCoT-Self-Reflection-6K}
UNICOT_BREAKDOWN_DIR=${UNICOT_BREAKDOWN_DIR:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub/datasets--Fr0zencr4nE--UniCoT-Breakdown-3K}
UNICOT_MIX_RATIO=${UNICOT_MIX_RATIO:-0.5}
UNICOT_VAL_RATIO=${UNICOT_VAL_RATIO:-0.05}
UNICOT_SPLIT_SEED=${UNICOT_SPLIT_SEED:-42}
# Optional size caps; declared empty so ``${VAR:+...}`` is well-defined.
UNICOT_TRAIN_SIZE=${UNICOT_TRAIN_SIZE:-}
UNICOT_VAL_SIZE=${UNICOT_VAL_SIZE:-}

TRAIN_FILE=${UNICOT_TRAIN:-"${_REPO_ROOT}/outputs/data/agentic_unicot/train.parquet"}
VAL_FILE=${UNICOT_TEST:-"${_REPO_ROOT}/outputs/data/agentic_unicot/val.parquet"}
# Build the mixed UniCoT train/val parquet (system + user only; UniCoT fields
# are reward ground truth, never fewshot). Skip when both files already exist
# unless REBUILD_UNICOT=1 (avoids import-heavy rebuild on resume).
if [[ "${REBUILD_UNICOT:-1}" == "1" || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  "${_PY}" -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl \
      --breakdown_dir "$UNICOT_BREAKDOWN_DIR" \
      --reflection_dir "$UNICOT_REFLECTION_DIR" \
      --local_save_dir "$(dirname "$TRAIN_FILE")" \
      --mix_ratio "$UNICOT_MIX_RATIO" \
      --val_ratio "$UNICOT_VAL_RATIO" \
      --seed "$UNICOT_SPLIT_SEED" \
      ${UNICOT_TRAIN_SIZE:+--train_size "$UNICOT_TRAIN_SIZE"} \
      ${UNICOT_VAL_SIZE:+--val_size "$UNICOT_VAL_SIZE"} || {
    echo "[ERROR] UniCoT parquet build failed (traceback above). Refusing to launch:" >&2
    echo "[ERROR] without 'set -e' the trainer would otherwise keep going and train on a" >&2
    echo "[ERROR] stale ${TRAIN_FILE} (or none at all) while silently ignoring any dataset change." >&2
    exit 1
  }
else
  echo "[INFO] reusing existing UniCoT parquet: $TRAIN_FILE / $VAL_FILE (set REBUILD_UNICOT=1 to rebuild)"
fi

# Count cards Ray will actually see (CUDA_VISIBLE_DEVICES=0,1,2,3 → 4).
# All visible cards go to the actor+GEN hybrid pool. UND AR is colocated on that
# same placement group (init_colocated) — it does NOT need spare free Ray GPUs.
_count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local n=0 id
    IFS=',' read -ra _ids <<< "${CUDA_VISIBLE_DEVICES}"
    for id in "${_ids[@]}"; do
      id="${id#"${id%%[![:space:]]*}"}"
      id="${id%"${id##*[![:space:]]}"}"
      [[ -n "$id" ]] && n=$((n + 1))
    done
    echo "$n"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
    return
  fi
  echo 1
}
N_VISIBLE="$(_count_visible_gpus)"
N_VISIBLE="${N_VISIBLE//[[:space:]]/}"
if [[ -z "$N_VISIBLE" || "$N_VISIBLE" -lt 1 ]]; then
  echo "Need at least 1 visible GPU (got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})." >&2
  exit 1
fi
NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-$N_VISIBLE}
# Colocated UND AR footprint inside the actor PG. The replica takes the FIRST
# ``UND_N_GPUS`` slices of the actor pool, so it is those cards -- not the whole node --
# that pay for it. It must span TWO cards: ``bagel_corl_deploy_ar.yaml`` puts the Thinker
# (stage 0) on logical device 0 and the DiT (stage 1) on logical device 1, and stage
# devices are logical indices into the replica's own ``CUDA_VISIBLE_DEVICES``, which is
# exactly the slice handed to it here. A 1-GPU pool cannot resolve logical device 1.
#
# It must also *divide* the actor pool: the slice comes from ``split_resource_pool``,
# which asserts ``split_size`` divides ``world_size``. On a 4-card pool that leaves
# {1, 2, 4}, so a 3-wide AR window is not expressible there. The recipe checks this
# before launching; the trainer re-checks it with a named error.
#
# Why the split is not optional: every ``bagel_think`` stage loads its own copy of the
# full 27.5GiB Bagel checkpoint, and the recipe always boots a GEN engine on the first
# actor GPU. With both stages on device 0 a 4-device run measured GPU 0 at 71.9GiB while
# GPUs 1-3 sat at 2.8GiB, and the first ``update_weights`` died with "Wake-up failed on
# Rank 0 ... CUDA Error: out of memory at cumem_allocator.cpp:163": the GEN engine's
# level-1 sleep backs 28.56GiB up in CPU and must re-map all of it on wake, against the
# ~7.8GiB that were still free. Splitting the stages leaves each first-two-GPU at two
# checkpoint copies (GEN ~28.21GiB + one AR stage), which fits with the wake-up headroom.
#
# The width also has to be *declared* to vLLM-Omni or the slice collapses back to one
# card: ``RolloutReplica`` derives ``gpus_per_replica_node = min(n_gpus_per_node,
# world_size)`` with ``world_size = TP * DP * PP``, and ``vLLMReplica.launch_servers``
# builds the server's ``CUDA_VISIBLE_DEVICES`` from that count (asserting the worker
# count equals ``world_size``). ``_build_und_ar_entrypoint_config`` therefore declares
# ``tensor_model_parallel_size = und_n_gpus`` and pins every stage back to TP=1 with
# ``--stage-overrides``, because a plain top-level engine arg reaches each stage and a
# stage holding a single ``devices`` entry cannot be asked for two ranks. Raise
# ``UND_N_GPUS`` only together with the stage ``devices`` in bagel_corl_deploy_ar.yaml;
# the mismatch is caught in ``_ensure_dual_role_rollout`` and by
# tests/agent_loop/test_bagel_corl_gpu_budget_on_cpu.py.
UND_N_GPUS=${UND_N_GPUS:-2}
# UND AR engine budget (``agent.und_gpu_memory_utilization``), mirroring
# bagel_corl_deploy_ar.yaml stage 0. This sizes the Thinker's KV cache: weights + peak
# activation are util-independent (~31.1GiB, measured as "28.16 GiB for consumed memory
# (weights + non-torch), 2.96 GiB for peak activation" over a 27.54GiB load), so vLLM
# aborts init under a budget of that size with "No available memory for the cache
# blocks" -- 0.30 *and* 0.35 both killed the run inside ``_init_und`` on an idle card.
# Floor ~0.393; 0.40 (~4.3GiB of KV) is the cheapest init-viable pool. This knob cannot
# buy GEN wake-up headroom, because the GEN engine's footprint is its weights, not its
# budget (it held 28.21GiB while told 0.25). Move the AR stages instead.
UND_GPU_MEM_UTIL=${UND_GPU_MEM_UTIL:-0.40}
BAGEL_UND_DEPLOY_CONFIG=${BAGEL_UND_DEPLOY_CONFIG:-"$(dirname "$0")/bagel_corl_deploy_ar.yaml"}
# These two are `${VAR:-default}`, so a value exported in an earlier shell/tmux
# session silently wins over the defaults above -- that is how a 4-device run went out
# with UND_N_GPUS=1 / UND_GPU_MEM_UTIL=0.35 and died as
#   StageEngineCoreProc_stage0_replica0 ... ValueError: No available memory for the
#   cache blocks -> Engine core initialization failed -> Orchestrator initialization failed
# (below ~0.393 the AR Thinker has no room for KV blocks at all, and width 1 cannot
# resolve the DiT stage's logical device 1). Refuse that combination instead of
# spending a run on it. Set UND_AR_ALLOW_TIGHT=1 to bypass deliberately.
if [[ "${UND_AR_ALLOW_TIGHT:-0}" != "1" && "${NUM_GPUS_ACTOR_ROLLOUT_REWARD}" -ge 2 ]]; then
  if (( UND_N_GPUS < 2 )); then
    echo "UND_N_GPUS=${UND_N_GPUS} cannot cover bagel_corl_deploy_ar.yaml (Thinker on logical \"0\", DiT on \"1\")." >&2
    echo "Use UND_N_GPUS=2, or put both stages on \"0\" in the yaml for a 1-GPU smoke test." >&2
    exit 1
  fi
  if awk "BEGIN {exit !(${UND_GPU_MEM_UTIL} < 0.40)}"; then
    echo "UND_GPU_MEM_UTIL=${UND_GPU_MEM_UTIL} is below the ~0.393 AR engine-init floor" >&2
    echo "(28.16GiB consumed + 2.96GiB peak over a 27.54GiB checkpoint on a 79.66GiB card)." >&2
    echo "0.30 and 0.35 abort _init_und with 'No available memory for the cache blocks'." >&2
    exit 1
  fi
fi
# Live Hermes proof (optional before long runs). STOP IT BEFORE THIS SCRIPT:
# its AR engine pins GPU 0, and the colocated UND AR below needs that card too,
# which shows up as the trainer OOMing during vllm-omni engine init.
#   bash examples/agenticllmgrpo_trainer/bagel/run_bagel_und_ar_serve.sh
#   BAGEL_UND_URL=http://127.0.0.1:8094 python3 .../spike_und_hermes.py --model-path "$BAGEL_MODEL_PATH"
#   pkill -f 'vllm-omni serve'   # free the cards again before training
BAGEL_UND_AR_SERVING_READY=${BAGEL_UND_AR_SERVING_READY:-1}
echo "bagel_corl 4-device e2e: actor+GEN=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} UND_AR_colocated=${UND_N_GPUS} visible=${N_VISIBLE} und_ready=${BAGEL_UND_AR_SERVING_READY} und_util=${UND_GPU_MEM_UTIL}"
# GEN engine width, and the GEN deploy yaml's per-replica device template that has to
# match it. With TP=1 every actor rank starts its own GEN server holding the whole
# ~28.2GiB Bagel checkpoint; on a 4-card pool those servers share cards with the UND AR
# stages, and GPU 0 measured on 2026-09-17 01:43 was
#   actor shard 12.17 + 4 vLLM-Omni front-ends 2.08 + GEN 28.99 + AR Thinker 35.73
#   = 78.97GiB of 79.11GiB
# so the first weight publish died in the LoRA-merge restore path:
#   restore_base_model_weights -> param.data.copy_(backup[name].to(param.device))
#   OOM on device 0 while trying to allocate 1090519040 bytes (free: 1012465664)
# (a 1.016GiB staging copy against 0.943GiB free). That is not a budget knob: a
# gpu_memory_utilization cannot shrink a full checkpoint, which is why the AR side had to
# move stages instead. Bagel's DiT *is* tensor-parallel -- MoTQKVParallelLinear /
# MoTRowParallelLinear shard heads by get_tensor_model_parallel_world_size, and
# 32 heads % 2 == 0 -- so TP=2 splits the checkpoint over the two cards of each replica
# (~14.1GiB each) and frees ~14GiB on exactly the two tight cards.
#
# The stage's ``devices`` list is a per-replica *template* whose length must equal the
# stage's ``parallel_config.world_size`` (see split_devices_for_replicas template mode),
# so it has to move with ROLLOUT_TP: ``"0,1"`` gives replica 0 -> 0,1 and replica
# 1 -> 2,3. Both are therefore chosen together here, and a guard below reads the yaml
# back so they cannot drift apart.
if [[ "${NUM_GPUS_ACTOR_ROLLOUT_REWARD}" -ge 4 ]]; then
  ROLLOUT_TP=${ROLLOUT_TP:-2}
  _GEN_DEPLOY_DEFAULT="$(dirname "$0")/bagel_corl_deploy_tp2.yaml"
else
  # 1- and 2-card runs keep one single-device GEN replica per card.
  ROLLOUT_TP=${ROLLOUT_TP:-1}
  _GEN_DEPLOY_DEFAULT="$(dirname "$0")/bagel_corl_deploy.yaml"
fi
BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-$_GEN_DEPLOY_DEFAULT}
# One colocated RM tensor-parallel across the pool (not N copies of Qwen). Override REWARD_TP=1 for per-GPU workers.
REWARD_TP=${REWARD_TP:-$NUM_GPUS_ACTOR_ROLLOUT_REWARD}
REWARD_ENGINE=${REWARD_ENGINE:-vllm}
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % ROLLOUT_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}." >&2
  exit 1
fi
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % REWARD_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by REWARD_TP=${REWARD_TP}." >&2
  exit 1
fi
# The UND AR replica's slice is taken with ``split_resource_pool``, which asserts the
# slice divides the actor PG -- an undivisible ``UND_N_GPUS`` dies ~8 minutes in as
# ``AssertionError: split_size must be a divisor of world_size`` (measured 2026-09-18
# 10:36 on hk01dgx012, devices 4-7, ``UND_N_GPUS=3`` on a 4-card pool). On the usual
# 4-card pool the valid widths are therefore {1, 2, 4}.
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % UND_N_GPUS != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by UND_N_GPUS=${UND_N_GPUS}" >&2
  echo "(the UND AR slice comes from split_resource_pool, which requires an exact split)." >&2
  exit 1
fi
# The GEN deploy yaml's stage must name exactly ``ROLLOUT_TP`` devices, because a
# diffusion stage's ``parallel_config.world_size`` is what splits its device template
# (``split_devices_for_replicas``). Too few and the extra rank has no card -- the same
# mis-declaration that once collapsed the UND AR replica onto GPU 0; too many and
# vLLM-Omni raises mid-launch. Read the file back instead of trusting the pairing above.
"${_PY}" - "$BAGEL_DEPLOY_CONFIG" "$ROLLOUT_TP" <<'PY' || exit 1
import sys

import yaml

path, tp = sys.argv[1], int(sys.argv[2])
try:
    stages = yaml.safe_load(open(path)).get("stages") or []
except OSError as exc:
    sys.exit(f"GEN deploy config {path!r} is unreadable: {exc}")
if not stages:
    sys.exit(f"GEN deploy config {path!r} declares no stages")
for stage in stages:
    devices = [d for d in str(stage.get("devices", "")).split(",") if d.strip()]
    if len(devices) != tp:
        sys.exit(
            f"GEN deploy config {path!r}: stage {stage.get('stage_id')} names {len(devices)} "
            f"device(s) {devices} but ROLLOUT_TP={tp}. A diffusion stage's device list is a "
            f"per-replica template whose length must equal tensor_parallel_size; pass a "
            f"matching yaml (bagel_corl_deploy.yaml for TP=1, bagel_corl_deploy_tp2.yaml for "
            f"TP=2) or clear ROLLOUT_TP."
        )
PY
# 1-GPU cannot colocate Bagel FSDP + vLLM-Omni + Qwen RM. Default: drop RM, tiny util.
# 2-GPU: actor + Omni + colocated UND AR is already tight — keep RM off.
# 4-GPU e2e: still default RM off until Hermes+composite step are green; ENABLE_RM=1 to re-enable.
if [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -eq 1 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.08}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.18}
  TRAIN_BSZ=${TRAIN_BSZ:-1}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
  GEN_STEPS=${GEN_STEPS:-4}
  GEN_HW=${GEN_HW:-256}
  LORA_RANK=${LORA_RANK:-4}
  LORA_ALPHA=${LORA_ALPHA:-8}
elif [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -le 2 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.12}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
  TRAIN_BSZ=${TRAIN_BSZ:-2}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
  # 15 mirrors the canonical BAGEL FlowGRPO recipes
  # (flowgrpo_trainer/bagel/run_bagel_{ocr,pickscore}_lora.sh), which train their rollout
  # at 15 and validate at 50. The 1-GPU branch above uses 4 only because it cannot fit the
  # actor + Omni + RM at all -- that is a memory concession, not a quality target.
  GEN_STEPS=${GEN_STEPS:-15}
  GEN_HW=${GEN_HW:-512}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
else
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.15}
  # GEN engine budget on a card that also carries the actor shard and one UND AR stage.
  # Measured: this knob does NOT bound the engine's footprint -- each GEN rank held
  # 28.21GiB "after model loading" (the whole 27.5GiB Bagel checkpoint) and backed 28.56GiB
  # up in CPU on level-1 sleep, while the budget said 0.25 * 79.66 = 19.9GiB. So it cannot
  # make room for the wake_up re-map; only moving the AR off rank 0 can (see UND_N_GPUS).
  # Keep it at 0.25 and let it size the GEN lane's own KV / workspace.
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
  TRAIN_BSZ=${TRAIN_BSZ:-2}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
  # See the 2-GPU branch: 15 matches the canonical BAGEL FlowGRPO training rollout.
  GEN_STEPS=${GEN_STEPS:-15}
  GEN_HW=${GEN_HW:-512}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
fi
if [[ "$ENABLE_RM" != "1" ]]; then
  ENABLE_RM=0
fi

# ---------------------------------------------------------------------------
# Frozen image-judge sidecar: the GEN reward source when ENABLE_RM=1.
#
# ``bagel_rm_image_scorer`` does not score anything itself -- its mid-loop branch
# forwards each PNG by HTTP to the frozen VL judge
# (``agentic_image_judge_client.call_reflect_vlm`` -> ``post_vllm_chat`` ->
# ``{vllm_url}/v1/chat/completions``). With ``agentic_image_gen.vllm_url`` at its
# schema default of "" the scorer fails loud, so ENABLE_RM=1 without a reachable judge
# only converts the current silent all-zero reward into a mid-episode crash. Hence the
# ordering: the judge is resolved and *proven reachable* before Ray comes up.
#
# The agent-facing ``judge_image`` tool is a red herring in this lane --
# ``bagel_corl_lib`` lists it in ``_INERT_BARE_TOOLS``, so the reward rides the in-loop
# RM handle and this sidecar is the only thing that produces a C/A score.
JUDGE_SERVER=${JUDGE_SERVER:-$ENABLE_RM}  # start it whenever the RM lane is on
JUDGE_URL=${JUDGE_URL:-http://127.0.0.1:8093}
# Mirrors the schema default of ``agentic_image_gen.good_enough_threshold``; pinning it
# here just makes the value the run actually used visible in the recipe.
JUDGE_GOOD_ENOUGH_THRESHOLD=${JUDGE_GOOD_ENOUGH_THRESHOLD:-0.80}
# Share a card the actor already owns. The 9B judge needs ~26GiB at mem-util 0.32 and the
# run's own cards hold ~29GiB, so the LAST visible device is the emptiest -- the GEN rank
# sits there while the actor's FSDP shards and the UND AR stages take the first ones.
# Override JUDGE_GPU to move it. Lower AGENTIC_REFLECT_GPU_MEM_UTIL if it still fights
# the trainer for room (the actor's update_actor re-map is the peak).
if [[ -z "${JUDGE_GPU:-}" ]]; then
  JUDGE_GPU="${CUDA_VISIBLE_DEVICES##*,}"
fi
# ``JUDGE_IMAGE_MODEL`` is normally exported by the operator env
# (``fred_verlomni_agentic_multiturn_pr1.sh``). That file also ``unset``s ~20 unrelated
# knobs and exports MODEL_PATH / IMAGE_GEN_MODEL / WANDB, so sourcing it here would
# clobber this recipe's own state; read the single variable in a subshell instead. Point
# ``JUDGE_ENV_FILE`` at it once and the sidecar is fully self-configuring.
if [[ -z "${JUDGE_IMAGE_MODEL:-}" && -n "${JUDGE_ENV_FILE:-}" ]]; then
  JUDGE_IMAGE_MODEL="$(bash -c "source '${JUDGE_ENV_FILE}' >/dev/null 2>&1; printf '%s' \"\${JUDGE_IMAGE_MODEL:-}\"")"
fi
# Fallback is a path verified to exist (the operator env's current judge), not a guess: a
# missing model must fail here with a clear message rather than deep inside ``vllm serve``.
JUDGE_IMAGE_MODEL="${JUDGE_IMAGE_MODEL:-/home/fq9hpsac/fq9hpsacuser11/fred/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"

_judge_ready() { curl -fsS -m 2 "${JUDGE_URL%/}/v1/models" >/dev/null 2>&1; }
JUDGE_SERVER_PID=""
if [[ "$JUDGE_SERVER" == "1" ]]; then
  if _judge_ready; then
    echo "[INFO] image judge already serving at ${JUDGE_URL}; reusing it"
  else
    if [[ ! -e "${JUDGE_IMAGE_MODEL}" ]]; then
      echo "[ERROR] JUDGE_IMAGE_MODEL=${JUDGE_IMAGE_MODEL} does not exist." >&2
      exit 1
    fi
    # Deliberately NOT under $RUN_DIR: Hydra refuses to write into a run dir that already
    # exists, so creating it here would break the launch.
    JUDGE_LOG="/tmp/bagel_judge_${RUN_TS}.log"
    echo "[INFO] starting image judge on GPU ${JUDGE_GPU} (mem-util ${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.32}); log ${JUDGE_LOG}"
    # Env prefix for the sidecar. Keep ONLY assignments in this list: a ``#`` comment on
    # any of the continuation lines terminates the command and silently drops every
    # assignment after it (measured: the judge then died on ``JUDGE_IMAGE_MODEL is unset``).
    #
    # PYTHONPATH — the middleware (``judge_image_log_middleware``) lives beside the sidecar
    #   and is not importable from the repo root; vLLM loads it by module path.
    # PATH — the sidecar ``exec``s a bare ``vllm``, and this recipe never activates the venv
    #   (it calls ``${_PY}`` by absolute path), so the binary has to be put on PATH here.
    # LD_LIBRARY_PATH / PYTORCH_NVML_BASED_CUDA_CHECK — the judge is a *second* torch process
    #   on a card whose driver is CUDA 12.2 while this venv's torch is cu13, so it needs the
    #   same cuda-compat shim the trainer block above exports. Inherited, restated so the
    #   dependency is not a silent accident of ordering.
    CUDA_VISIBLE_DEVICES="$JUDGE_GPU" \
      JUDGE_IMAGE_MODEL="$JUDGE_IMAGE_MODEL" \
      AGENTIC_REFLECT_GPU_MEM_UTIL="${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.32}" \
      AGENTIC_REFLECT_MAX_NUM_SEQS="${AGENTIC_REFLECT_MAX_NUM_SEQS:-2}" \
      AGENTIC_REFLECT_MAX_MODEL_LEN="${AGENTIC_REFLECT_MAX_MODEL_LEN:-4096}" \
      PYTHONPATH="${_SCRIPT_DIR}/../agent_llm${PYTHONPATH:+:${PYTHONPATH}}" \
      PATH="${_REPO_ROOT}/.venv/bin:${PATH}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}" \
      bash "${_SCRIPT_DIR}/../agent_llm/run_judge_image_tool_server.sh" >"$JUDGE_LOG" 2>&1 &
    JUDGE_SERVER_PID=$!
    for _ in $(seq 1 240); do
      _judge_ready && break
      if ! kill -0 "$JUDGE_SERVER_PID" 2>/dev/null; then
        echo "[ERROR] image judge exited during startup; last lines of ${JUDGE_LOG}:" >&2
        tail -n 40 "$JUDGE_LOG" >&2 || true
        exit 1
      fi
      sleep 2
    done
    if ! _judge_ready; then
      echo "[ERROR] image judge not ready at ${JUDGE_URL} after 480s; see ${JUDGE_LOG}" >&2
      exit 1
    fi
    echo "[INFO] image judge ready at ${JUDGE_URL} (pid ${JUDGE_SERVER_PID})"
    trap 'kill "${JUDGE_SERVER_PID}" 2>/dev/null || true' EXIT INT TERM
  fi
elif [[ "$ENABLE_RM" == "1" ]]; then
  # JUDGE_SERVER=0 with the RM lane on means the operator already runs the sidecar,
  # possibly on another host. Prove it answers rather than discovering it mid-step.
  if ! _judge_ready; then
    echo "[ERROR] ENABLE_RM=1 but no image judge at ${JUDGE_URL} (JUDGE_SERVER=0)." >&2
    echo "[ERROR] Start: CUDA_VISIBLE_DEVICES=<gpu> bash examples/agenticllmgrpo_trainer/agent_llm/run_judge_image_tool_server.sh" >&2
    echo "[ERROR] ...or launch with JUDGE_SERVER=1 to let this recipe start one." >&2
    exit 1
  fi
  echo "[INFO] image judge verified at ${JUDGE_URL} (JUDGE_SERVER=0)"
fi

# Validation rollout budget. ``data.val_batch_size`` only sets the *batch* size; the V1 val
# path still iterates the whole val set, because its cap is ``data.val_max_samples``
# (``trainer_base.py``: ``max_samples=... get("val_max_samples", -1)``), whose default of -1
# means "all of it". Measured on the 4-GPU run: the val set is 450 samples
# (``UNICOT_VAL_RATIO=0.05``) and validation rolls N=2 siblings each, so a single pass
# produced ~335 episode folders within ~5 minutes and was still growing — on track for
# ~900 episodes and ~1800 PNGs, at the pinned 50 denoise steps per image, per pass, and it
# re-runs every ``TEST_FREQ=30``. That dominated wall-clock for what is a plumbing run.
# 8 keeps a per-prompt signal (16 episodes / 32 PNGs) at a fraction of the cost; raise it,
# or set -1, when the validation number itself becomes the thing under test.
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}

# Smoke defaults. N = sibling episodes (Token GRPO); S = seeds per generate_image (FlowGRPO).
# In-episode J (UND turns) and K (GEN calls) are runtime with J>=K — not these env vars.
# Production example: N=8 S=4. S must be >= 2.
# Do not inherit stale K=1 from an old shell (`S=${K:-2}` would still pick K=1).
if [[ -z "${N:-}" ]]; then
  if [[ -n "${J:-}" ]]; then N=$J; else N=2; fi
fi
if [[ -z "${S:-}" ]]; then
  if [[ -n "${K:-}" && "$K" -ge 2 ]]; then S=$K; else S=2; fi
fi
if [[ "$S" -lt 2 ]]; then
  echo "gen_samples_per_call S must be >= 2 (FlowGRPO seeds; got S=$S). Use S=2 or unset stale K." >&2
  exit 1
fi
if [[ "$N" -lt 1 ]]; then
  echo "rollout.n N must be >= 1 (got N=$N)." >&2
  exit 1
fi
echo "bagel_corl smoke: N=$N siblings, S=$S seeds/call (in-episode J/K are runtime)"

# RFC §4.8.4 launch divisibility (the two `pool % *_TP` checks above are the other
# half). These are checked on the EPISODE count `ppo_mini_batch_size x N`, not on the
# prompt count the knobs are written in: `N` siblings turn one task into `N` episodes,
# and it is the episode count that has to split evenly across the actor pool and the
# per-GPU micro batch. A ragged split drops or duplicates episodes instead of failing.
# The authoritative copy lives in verl_omni/utils/config.py
# (``validate_bagel_corl_config`` -> ``_bagel_corl_divisibility_failures``, covered by
# tests/utils); this one only fails a couple of minutes earlier, before Ray is up.
PPO_MINI=${PPO_MINI:-$TRAIN_BSZ}
PPO_MICRO=${PPO_MICRO:-1}
if (( TRAIN_BSZ % PPO_MINI != 0 )); then
  echo "data.train_batch_size=${TRAIN_BSZ} must be divisible by ppo_mini_batch_size=${PPO_MINI}." >&2
  exit 1
fi
EPISODES_PER_UPDATE=$(( PPO_MINI * N ))
if (( EPISODES_PER_UPDATE % NUM_GPUS_ACTOR_ROLLOUT_REWARD != 0 )); then
  echo "(ppo_mini_batch_size=${PPO_MINI} x N=${N})=${EPISODES_PER_UPDATE} must be divisible by" >&2
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD}." >&2
  exit 1
fi
PER_RANK_EPISODES=$(( EPISODES_PER_UPDATE / NUM_GPUS_ACTOR_ROLLOUT_REWARD ))
if (( PPO_MICRO > 0 && PER_RANK_EPISODES % PPO_MICRO != 0 )); then
  echo "(ppo_mini_batch_size x N / n_gpus_per_node)=${PER_RANK_EPISODES} must be divisible by" >&2
  echo "ppo_micro_batch_size_per_gpu=${PPO_MICRO}." >&2
  exit 1
fi

# RFC §4.4.2b (R2): the GEN conditioning cache. The single-lane baseline also leaves it
# off (its single-round rollout has nothing to reuse), so this is a *shared*
# sub-optimality that Co-RL amplifies: S seeds per call x K calls per episode re-encode
# the very same conditioning. Two knobs, and they are a pair --
#   * enable_prompt_embed_cache: turn the cache on at all.
#   * enable_prompt_embed_cache_routing_affinity: pin one whole S-group to one replica.
# Each replica owns its own cache, and LLMServerClient acquires its replica from the
# load balancer using the request id, so without the pin seeds 2..S are scattered across
# replicas that hold no entry and a cold cache is guaranteed -- the cache would be on
# and buy nothing.
# Size the LRU against the live working set (S x N x agent-loop workers). The 32 default
# would evict a group's entry before the group finished for any real production shape
# (N=8, S=4, 8 workers -> 256 live conditionings).
#
# All three knobs are passed as ``+key=value`` (appended), not plain overrides. This
# entry composes against ``omni_trainer``, whose ``actor_rollout_ref.rollout`` node is
# typed ``verl.workers.config.RolloutConfig`` -- the *AR* config. The three fields are
# declared on ``DiffusionRolloutConfig`` and only take effect after
# ``_rewrite_bagel_corl_configs`` re-targets the node at runtime, so at compose time
# they are new keys and Hydra refuses a bare override with
# "Could not override ... To append to your config use +...". Same reason the sibling
# ``pipeline.*`` / ``algo.*`` knobs below carry the prefix. ``agent.num_workers`` does
# NOT: ``RolloutConfig.agent.num_workers`` already exists in the composed config.
AGENT_WORKERS=${AGENT_WORKERS:-8}
ENABLE_PEC=${ENABLE_PEC:-1}
PEC_SIZE=${PEC_SIZE:-$(( S * N * AGENT_WORKERS ))}
if (( PEC_SIZE < 32 )); then PEC_SIZE=32; fi
if [[ "${ENABLE_PEC}" == "1" ]]; then
  PEC_FLAG=True
  PEC_AFFINITY=True
else
  # Affinity without a cache is incoherent and is rejected in validate_bagel_corl_config.
  PEC_FLAG=False
  PEC_AFFINITY=False
fi
echo "bagel_corl R2: prompt_embed_cache=${PEC_FLAG} affinity=${PEC_AFFINITY} size=${PEC_SIZE} (S*N*workers=${S}x${N}x${AGENT_WORKERS})"

# UniGRPO-aligned knobs (RFC 453 §2/§4.4): per-expert LRs (UND base vs GEN
# lr_gen=3e-5), lane weights, and GRPO-Guard ratio normalization via
# +actor_rollout_ref.actor.diffusion_loss.loss_mode=grpo_guard as an ablation.
# NOTE: keep comment lines OUTSIDE the backslash-continued launch command below.
#
# Episode context budget. ``_und_decode`` hands the AR engine ``prompt_ids +
# response_ids`` -- the whole episode so far -- so the decode prompt grows with every
# turn, while the framework pads each trajectory to ``rollout.prompt_length`` /
# ``rollout.response_length`` and the AR engine's ``max_model_len`` is derived from
# ``data.max_*``. RolloutConfig defaults those two to 512/512 and this recipe used to
# leave them there, so they disagreed 2x with the ``data.*`` pair above: the dataset and
# engine were built for 1024-token prompts/responses and every episode would be
# decoded and padded against 512 (``_pad_token_ids`` does not truncate -- it returns a
# *longer* tensor, which breaks the uniform batch shapes). Pin both from the same
# ``MAX_PROMPT_LEN``. ``validate_bagel_corl_config`` fails loud if they drift, and
# ``run_serial_episode`` stops the UND loop at their sum.
# Two GEN-step choices below, both taken from the canonical BAGEL FlowGRPO recipes
# (flowgrpo_trainer/bagel/run_bagel_{ocr,pickscore}_lora.sh), which train their rollout at
# num_inference_steps=15 and validate at 50 with noise_level=0.0:
#
#   * pipeline.num_inference_steps (GEN_STEPS) drives the RL rollout. 15 is the canonical
#     training value; the 1-GPU branch sets 4 purely because it cannot fit actor + Omni +
#     RM at all, so that 4 is a memory concession and not a quality target.
#   * val_kwargs is set explicitly because validation must NOT inherit the training
#     rollout. Inheriting means validating at GEN_STEPS with noise_level=0.7 -- both
#     degraded and *stochastic*, so val images and val metrics would not be reproducible
#     between runs and could not be compared. The '+' prefix is required here (it is not
#     in the canonical recipes) because the omni schema's val_kwargs holds only
#     _target_/top_k/top_p/temperature/n, with no pipeline/algo subtree to override.
#
# The dump root is keyed on the run dir, not just on ``trainer.experiment_name``.
# ``agentic_image_gen.e2e_root`` namespaces as ``<e2e_root>/<experiment_name>/``
# (see ``paths.resolve_run_dir``), so two runs that share an ``EXPERIMENT_NAME`` would
# share ``outputs/e2e/<name>/rollout_trajectories/step_*`` and *overwrite each other's
# dumps in place* — a later run restarts at step_000001 and clobbers the earlier run's
# step_000001. On 2026-09-21 that made "step 1" and "step 2" of ``bagel_corl_pr1`` come
# from two different runs, so a within-run regression appeared where none existed.
#
# ``agentic_image_gen.run_dir`` is therefore pinned to ``$RUN_DIR``. It is used
# *verbatim* — no ``experiment_name`` is appended — so every run gets its own trace
# corpus and ``step_NNNNNN`` means one thing for the life of the folder. The three
# trees land directly in the run dir, beside ``.hydra/`` and ``main_omni.log``:
#   $RUN_DIR/rollout_trajectories/step_*/sample_*.{json,txt}
#   $RUN_DIR/rollout_images/step_*/sample_*/
#   $RUN_DIR/hermes_actions/step_*.jsonl
# The node exists in the frozen schema (``agentic/image_gen_tools@agentic_image_gen``
# declares ``run_dir: null``), so no '+' prefix is needed.
"${_PY}" -m verl_omni.trainer.main_omni \
    hydra.run.dir=$RUN_DIR \
    agentic_image_gen.run_dir=$RUN_DIR \
    agentic_image_gen.vllm_url=$JUDGE_URL \
    agentic_image_gen.good_enough_threshold=$JUDGE_GOOD_ENOUGH_THRESHOLD \
    trainer.v1.trainer_mode=bagel_corl_sync \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_PROMPT_LEN \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name \
    +actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.model_type=diffusion_model \
    +actor_rollout_ref.model.composite_mode=bagel_corl \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora_dtype=bfloat16 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj','q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    +actor_rollout_ref.model.lr_gen=3e-5 \
    +actor_rollout_ref.actor.diffusion_loss.loss_weight_und=1.0 \
    +actor_rollout_ref.actor.diffusion_loss.loss_weight_gen=1.0 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.response_length=$MAX_PROMPT_LEN \
    +actor_rollout_ref.rollout.pipeline.height=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.width=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.num_inference_steps=$GEN_STEPS \
    +actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    actor_rollout_ref.rollout.repetition_penalty=$ROLLOUT_REPETITION_PENALTY \
    +actor_rollout_ref.rollout.enable_prompt_embed_cache=$PEC_FLAG \
    +actor_rollout_ref.rollout.enable_prompt_embed_cache_routing_affinity=$PEC_AFFINITY \
    +actor_rollout_ref.rollout.prompt_embed_cache_size=$PEC_SIZE \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
    +actor_rollout_ref.rollout.algo.noise_level=${NOISE_LEVEL:-0.7} \
    +actor_rollout_ref.rollout.algo.sde_window_size=${SDE_WINDOW_SIZE:-2} \
    +actor_rollout_ref.rollout.algo.sde_window_range=${SDE_WINDOW_RANGE:-[0,7]} \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=${VAL_GEN_STEPS:-50} \
    +actor_rollout_ref.rollout.val_kwargs.algo.noise_level=${VAL_NOISE_LEVEL:-0.0} \
    actor_rollout_ref.rollout.agent.default_agent_loop=bagel_multiturn_agent \
    +actor_rollout_ref.rollout.agent.gen_samples_per_call=$S \
    +actor_rollout_ref.rollout.agent.max_generate_passes=1 \
    +actor_rollout_ref.rollout.agent.max_und_turns=${MAX_UND_TURNS:-8} \
    +actor_rollout_ref.rollout.agent.und_ar_serving_ready=$BAGEL_UND_AR_SERVING_READY \
    +actor_rollout_ref.rollout.agent.und_deploy_config=$BAGEL_UND_DEPLOY_CONFIG \
    +actor_rollout_ref.rollout.agent.und_n_gpus=$UND_N_GPUS \
    +actor_rollout_ref.rollout.agent.und_gpu_memory_utilization=$UND_GPU_MEM_UTIL \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG \
    reward.num_workers=$(( ENABLE_RM == 1 ? NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP : 0 )) \
    reward.reward_model.enable=$ENABLE_RM \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.bagel_rm_image_scorer \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.gpu_memory_utilization=$REWARD_GPU_MEM_UTIL \
    reward.reward_model.rollout.enforce_eager=True \
    trainer.val_before_train=False \
    trainer.test_freq=${TEST_FREQ:-30} \
    data.val_batch_size=${VAL_BSZ:-$TRAIN_BSZ} \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TRAIN_STEPS:-3} \
    trainer.logger="['console']" \
    trainer.project_name=bagel_corl \
    trainer.experiment_name=${EXPERIMENT_NAME:-bagel_corl_pr1_longrun} \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    ${DEFAULT_LOCAL_DIR:+trainer.default_local_dir="$DEFAULT_LOCAL_DIR"} \
    "$@"
