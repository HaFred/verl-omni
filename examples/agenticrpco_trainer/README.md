# Agentic Mode (2a) / RPCO Trainer

Last updated: 08/05/2026

Training recipes for **Mode (2a) Agentic LLM RL** ([#302](https://github.com/verl-project/verl-omni/issues/302)).

- **Lance Mode (2a) GRPO** (`lance/`): train HF und (`Lance_3B_hf_und`); frozen diffusion is an external `generate_image` tool.
- Reward: `pkg://verl_omni.utils.reward_score.agentic_reward` — Hermes format + **Reflection: between tools** + **exactly two** `generate_image` calls with a rewritten 2nd prompt.

Cold Lance und does **not** emit Hermes on its own. The 100-step e2e therefore combines:

1. Full few-shot parquet (call → image obs → reflect-on-image → 2nd call → done)
2. Force/teacher agent loop (`agentic_force_tool_agent`) so rollouts actually execute 2 tool turns
3. `agentic_reward` so GRPO ranks reflection + prompt rewrite over garbage / bare JSON

## Prerequisites

- Launch from the **verl-omni repo root** (repo-relative `function_tool_path`).
- Set **`MODEL_PATH`** to a local `Lance_3B_hf_und` export (not hub root / raw `Lance_3B`).
- Prefer **2 free GPUs** for GRPO (`CUDA_VISIBLE_DEVICES=1,4`) and a **third** GPU for the frozen Lance tool server when using real images.

### Make `Lance_3B_hf_und`

```bash
LANCE_ROOT=/path/to/bytedance-research/Lance   # contains Lance_3B/
python3 tests/special_e2e/prepare_lance_hf_und.py \
  --src "${LANCE_ROOT}/Lance_3B" \
  --dst "${LANCE_ROOT}/Lance_3B_hf_und"
export MODEL_PATH="${LANCE_ROOT}/Lance_3B_hf_und"
```

## 100-step overfit e2e (recommended)

Goal: overfit a tiny prompt pool so the policy learns **reflection on the 1st image → rewritten 2nd `generate_image`**.

### 1) Operator env (example)

```bash
source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh   # or your local env
export CUDA_VISIBLE_DEVICES=1,4
export MODEL_PATH=/path/to/Lance_3B_hf_und
export TRAIN_FILE=$HOME/data/agentic_overfit/train.parquet
export VAL_FILE=$HOME/data/agentic_overfit/val.parquet
# Force stays on so every rollout gets 2 tool turns; reward ranks quality.
export AGENTIC_FORCE_GENERATE_IMAGE=1
export AGENTIC_FORCE_MIN_TOOL_CALLS=2
export AGENTIC_FORCE_PROB=1.0
export AGENTIC_TEACHER_FORCE_HERMES=1
```

### 2) (Optional) real frozen Lance images — pane A

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash examples/agenticrpco_trainer/lance/run_lance_frozen_diffusion_tool_server.sh
```

Then in the trainer pane:

```bash
export AGENTIC_LANCE_SERVER_URL=http://127.0.0.1:8091
# LANCE_HUB_ROOT=/path/to/full/Lance/snapshot   # if the server script needs it
```

Without `AGENTIC_LANCE_SERVER_URL`, `generate_image` returns a **text stub** (still multi-turn; no PNGs).

### 3) Run 100-step GRPO — pane B

From repo root:

```bash
MODEL_PATH=$MODEL_PATH \
  TRAIN_FILE=$TRAIN_FILE \
  VAL_FILE=$VAL_FILE \
  TOTAL_STEPS=100 \
  N_GPUS=2 \
  OVERFIT_DATA=1 \
  bash examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh
```

What the script does:

| Step | Behavior |
| --- | --- |
| Template | Installs Hermes tool Jinja into `MODEL_PATH` if missing |
| Data | With `OVERFIT_DATA=1` (default), regenerates overfit parquet via `create_dummy_agentic_data.py --overfit` (2 prompts, reflection-heavy GT weights) |
| Agent loop | `agentic_force_tool_agent` — ensures ≥2 `generate_image` turns when und omits Hermes |
| Reward | `agentic_reward.compute_score` — high score only for Reflection between calls + rewritten 2nd prompt |
| Artifacts | `outputs/e2e/<experiment>/rollout_trajectories/`, `rollout_images/`, WandB |

Inspect learning:

```bash
# Prefer trajectories with Reflection: between two <tool_call> blocks and
# num_voluntary_hermes rising over steps (force alone is not enough).
ls outputs/e2e/lance_agentic_grpo_*/rollout_trajectories/step_000100/
```

### Data-only refresh (no train)

```bash
python3 tests/special_e2e/create_dummy_agentic_data.py \
  --local_save_dir ~/data/agentic_overfit \
  --overfit --train_size 8 --val_size 2
```

Each row’s few-shot is the full protocol: **LLM tool call → image obs → Reflection on image → rewritten tool call → obs → Done**, then the live user request.

## File map

```
examples/agenticrpco_trainer/lance/
├── agentic_force_tool_agent_loop.yaml
├── qwen2_tool_chat_template.jinja2
├── run_lance_agentic_grpo.sh
└── run_lance_frozen_diffusion_tool_server.sh

verl_omni/agent_loop/
├── diffusion_tool.py                    # generate_image tool (shared)
├── agentic_metrics_manager.py           # Stock manager + reward-component W&B logs
├── agentic_force_tool_agent_loop.py     # force/teacher multi-turn loop
├── agentic_image_reflection.py          # Reflection-on-image helper
└── agentic_trajectory_context.py        # Loop↔tool artifact context

verl_omni/utils/reward_score/agentic_reward.py   # format / reflect / 2x tool / result
tests/special_e2e/create_dummy_agentic_data.py   # --overfit few-shot + GT weights
```

Tool backends in `diffusion_tool.py` (first match wins):

1. `AGENTIC_LANCE_SERVER_URL` — vLLM-Omni Lance OpenAI serve  
2. `AGENTIC_DIFFUSION_TOOL_URL` — generic POST `{"prompt"}` → image/text  
3. unset — text stub  
