# Agentic LLM Trainer LoRA Overfitting

Last updated: 08/06/2026

Training recipes for **Mode (2a) Agentic LLM RL** ([#302](https://github.com/verl-project/verl-omni/issues/302)).

- **Qwen3-VL Mode (2a) GRPO** (`agent_llm/`): train `Qwen/Qwen3-VL-2B-Thinking`;
  frozen Qwen-Image is an external `generate_image` tool.
- Reward: `pkg://verl_omni.utils.reward_score.agentic_reward` — tool-call-first reward; one valid Hermes call earns partial credit and distinct later calls score higher.

The voluntary overfit combines:

1. A small two-call visual-refinement demonstration
2. Stock verl `tool_agent` for unmodified model → tool → model interaction
3. The model's native Hermes JSON tool-call template (no template patching)
4. `agentic_reward` so GRPO ranks reflection + rewritten calls over one call

## Prerequisites

- Launch from the **verl-omni repo root** (repo-relative `function_tool_path`).
  `run_agentic_grpo.sh` auto-`cd`s to the repo root.
- Set **`MODEL_PATH`** to `Qwen/Qwen3-VL-2B-Thinking` or a local snapshot.
- Prefer **2 free GPUs** for GRPO and a separate GPU for frozen Qwen-Image.
- The checked environment needs Qwen3-VL support in Transformers and vLLM.
  The launcher verifies the native tool template and image processor before
  allocating training workers.

## 100-step overfit e2e (recommended)

Goal: overfit a tiny prompt pool so the policy learns **reflection on the 1st image → rewritten 2nd `generate_image`**.

### 1) Operator env (example)

```bash
source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh   # or your local env
export CUDA_VISIBLE_DEVICES=3,4   # must be free; keep a third GPU for the tool
export MODEL_PATH=Qwen/Qwen3-VL-2B-Thinking
export TRAIN_FILE=$PWD/data/agentic/train.parquet
export VAL_FILE=$PWD/data/agentic/val.parquet
```

### 2) Start the frozen Qwen-Image tool — pane A

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  QWEN_IMAGE_MEMORY_MODE=balanced \
  bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
```

Then in the trainer pane:

```bash
export AGENTIC_QWEN_IMAGE_URL=http://127.0.0.1:8092/generate
```

Memory modes:

- `balanced`: split BF16 modules across **2+** visible GPUs (`device_map=balanced`).
  Prefer this when single-GPU generate OOMs.
- `full`: BF16 Qwen-Image resident on a large GPU.
- `model_offload`: component CPU offload; default single-GPU lower-memory mode.
- `sequential_offload`: lowest VRAM, substantially slower.
- `mmdit_nf4`: NF4 quantization of Qwen-Image's MMDiT transformer plus
  component offload; requires `bitsandbytes`.

Qwen-Image's MMDiT is not independently executable: generation still requires
the frozen text encoder, VAE, and scheduler. `mmdit_nf4` reduces the dominant
MMDiT footprint while preserving those required components.

Without an image service, `generate_image` returns a text stub. This can test
dispatch, but it cannot train visual reflection. The launcher therefore
requires a healthy image service by default; set `REQUIRE_REAL_IMAGE_TOOL=0`
only for a non-learning plumbing diagnostic.

### 3) Run 100-step GRPO — pane B

From repo root:

```bash
TOTAL_STEPS=100 \
  N_GPUS=2 \
  OVERFIT_DATA=1 \
  bash examples/agenticrpco_trainer/agent_llm/run_agentic_grpo.sh
```

What the script does:

| Step | Behavior |
| --- | --- |
| Template | Uses and verifies Qwen3-VL's native Hermes JSON template |
| Data | Reads `TRAIN_FILE` / `VAL_FILE` (defaults: `data/agentic/{train,val}.parquet`) |
| Agent loop | Stock verl `tool_agent` — no force or teacher replacement |
| Observation | Generated PIL pixels plus measurable `image_vis` facts |
| Reward | Tool bootstrap credit; full protocol requires reflection + distinct rewrite |
| LoRA sync | `layered_summon=false` (required for Qwen3-VL + FSDP LoRA → vLLM) |
| Artifacts | `outputs/e2e/<experiment>/rollout_trajectories/`, `rollout_images/`, WandB |

Inspect learning:

```bash
# Look for raw assistant decodes containing valid <tool_call> blocks.
ls outputs/e2e/qwen3_vl_agentic_grpo_*/rollout_trajectories/step_000100/
```

### Data-only refresh (no train)

```bash
python3 examples/agenticrpco_trainer/data_process/create_dummy_agentic_data.py \
  --local_save_dir data/agentic \
  --overfit --train_size 8 --val_size 2
```

Each row demonstrates the target protocol: **LLM tool call → image observation
→ Reflection on visible shortcomings → rewritten tool call → image observation
→ Done**, then presents the live user request. Runtime calls remain on-policy.

## File map

```
examples/agenticrpco_trainer/agent_llm/
├── qwen_image_tool_server.py
├── run_qwen_image_tool_server.sh
├── run_agentic_grpo.sh
├── check_overfit_gates.py
└── run_lance_frozen_diffusion_tool_server.sh   # legacy Lance backend (optional)

verl_omni/agent_loop/
├── diffusion_tool.py                    # generate_image tool (shared)
├── agentic_metrics_manager.py           # Stock manager + raw rollout/reward logs
├── agentic_image_reflection.py          # Supplemental image measurements
└── agentic_trajectory_context.py        # Artifact path helpers

verl_omni/utils/reward_score/agentic_reward.py   # format / reflect / 2x tool / result
examples/agenticrpco_trainer/data_process/create_dummy_agentic_data.py
```

Frozen tool backends in `diffusion_tool.py` (first match wins):

1. `AGENTIC_QWEN_IMAGE_URL` — bundled Qwen-Image service
2. `AGENTIC_DIFFUSION_TOOL_URL` — generic POST `{"prompt"}` → image/text
3. `AGENTIC_LANCE_SERVER_URL` — legacy Lance backend
4. unset — text stub
