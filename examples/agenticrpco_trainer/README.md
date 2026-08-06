# Agentic LLM Trainer LoRA Overfitting

Last updated: 08/06/2026

Training recipes for **Mode (2a) Agentic LLM RL** ([#302](https://github.com/verl-project/verl-omni/issues/302)).

- **Qwen3-VL Mode (2a) GRPO** (`agent_llm/`): train `Qwen/Qwen3-VL-2B-Instruct`
  (Thinking is optional; Instruct is preferred for short tool-first decodes).
- Frozen **Qwen-Image** is the external `generate_image` tool (`:8092`).
- Frozen **Qwen3-VL** sidecar (`:8093`) is the **reward judge only** for
  `reward_correctness` / `reward_aesthetics` (not an agent tool).
- Reward: `pkg://verl_omni.utils.reward_score.agentic_reward` — parseable
  `<tool_call>` protocol (Hermes JSON **or** Qwen3.5 XML) with actor
  self-reflection + `Done.` (or rewrite + `generate_image`) gating.
- Actor wire format is auto-selected from `MODEL_PATH`:
  - **Qwen3-VL** → `multi_turn.format=hermes` + Hermes JSON fewshots
  - **Qwen3.5** → `multi_turn.format=qwen3_coder` + XML fewshots
  Override with `TOOL_PARSER_FORMAT` / `--tool_call_format`.

Target protocol (fewshot + on-policy):

```
generate_image → (image obs attached)
  actor writes brief reflection, then either:
    Done.                                    OR
    rewrite + generate_image  (same assistant turn)
```

Fewshot demos in `create_dummy_agentic_data.py` follow that order for all three
classes (1-pass / 2-pass / 3-pass). They are **GRPO exploration examples**, not
supervised targets. Regenerate parquet when switching actor family so fewshot
`<tool_call>` syntax matches the chat template.

## Reward components

`compute_score` returns a scalar `score` plus per-component fields logged to
WandB as `agentic_reward/<name>/mean` (via `agentic_metrics_manager.py`).

### Primary fields (enter the weighted mix)

| Field | Default weight | Physical meaning | Zero when |
| --- | ---: | --- | --- |
| `reward_tool_call` | 0.05 | Binary: trajectory contains ≥1 parseable JSON/XML `<tool_call>`. | No parseable tool call |
| `reward_brevity` | 0.05 | Short assistant prose only (tool calls + tool obs stripped). | Long rambling / debate CoT |
| `reward_format` | 0.05 | Fraction of calls with valid required arguments. | No / malformed calls |
| `reward_reflection` | 0.10 | Quality of actor self-reflection prose (visual attrs / rewrite / Done). | No reflection prose after generate |
| `reward_tool_usage` | 0.10 | Protocol shape and distinct rewritten prompts. | No `generate_image` |
| `reward_result` | 0.05 | Closed-loop outcome quality. | No `generate_image` |
| `reward_correctness` | 0.30 | Frozen-VL correctness via `AGENTIC_REFLECT_VLM_URL` on last image. | URL unset / VL call fails |
| `reward_aesthetics` | 0.30 | Frozen-VL aesthetics via `AGENTIC_REFLECT_VLM_URL` on last image. | URL unset / VL call fails |

Weighted mix (then scaled by a protocol tier `base + scale * mix`):

```text
mix = Σ w_i * reward_i  /  Σ w_i
score = base + scale * mix
```

| Protocol tier | Condition | `base` | `scale` | `protocol_ok` |
| --- | --- | ---: | ---: | ---: |
| Closed + high C/A | reflection prose + `Done.`, single-pass or distinct rewrite, C/A ≥ 0.70 | 0.10 | 0.90 | 1 |
| Closed, VL down / weak C/A | closed loop; VL missing or C/A &lt; 0.70 | 0.05 | 0.65 | 1 / 0 |
| Reflection present | ≥1 `generate_image` + reflection prose, not fully closed | 0.05 | 0.45 | 0 |
| Gen-only (starved) | ≥1 `generate_image`, **no** reflection prose | 0.02 | 0.03 | 0 |

Weights are stored per row in parquet `ground_truth` / `extra_info` (`w_tool_call`,
`w_brevity`, `w_format`, `w_reflect`, `w_tool`, `w_result`, `w_correctness`,
`w_aesthetics`).

### Visual rubric diagnostics

| Field | Physical meaning |
| --- | --- |
| `reward_correctness_{subject_entities,attributes,relations_layout,scene_context,completeness}` | Five frozen-VL correctness answers |
| `reward_aesthetics_{composition,lighting,color,fidelity,appeal}` | Five frozen-VL aesthetics answers |
| `protocol_ok` | 1 iff closed loop (reflection + Done, single or distinct rewrite) |
| `num_hermes_tool_calls` | Legacy field name: count of parseable JSON/XML `<tool_call>` blocks |
| `num_generate_image_prompts` | Count of `generate_image` prompts |
| `num_reflect_image_calls` | Legacy (always 0; `reflect_image` is no longer an agent tool) |

## Prerequisites

- Launch from the **verl-omni repo root** (repo-relative `function_tool_path`).
  `run_agentic_grpo.sh` auto-`cd`s to the repo root.
- Set **`MODEL_PATH`** to a Qwen3-VL Instruct (or Thinking) snapshot.
- Prefer **2 free GPUs** for GRPO, **2 GPUs** for Qwen-Image, **1 GPU** for the
  reflect VLM sidecar.
- The launcher verifies the native Hermes tool template and image processor
  before allocating training workers.

## 100-step overfit e2e (recommended)

Goal: overfit a tiny prompt pool so the policy learns
**`generate_image` → (inspect image) → Done. or rewrite + generate_image**.

### 1) Operator env (example)

```bash
source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh   # or your local env
export CUDA_VISIBLE_DEVICES=3,4   # train GPUs; keep 0,1 for image and another for reflect
export MODEL_PATH=.../Qwen3-VL-2B-Instruct/...
export TRAIN_FILE=$PWD/data/agentic/train.parquet
export VAL_FILE=$PWD/data/agentic/val.parquet
```

### 2) Start frozen tools

Pane A — Qwen-Image (`generate_image`):

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  QWEN_IMAGE_MEMORY_MODE=balanced \
  bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
# trainer: export AGENTIC_QWEN_IMAGE_URL=http://127.0.0.1:8092/generate
```

Pane B — reflect VLM (reward judge only; not an agent tool):

```bash
CUDA_VISIBLE_DEVICES=2 \
  bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
# trainer: export AGENTIC_REFLECT_VLM_URL=http://127.0.0.1:8093/reflect
```

Restart the reflect server after rubric upgrades (e.g. 10-question rubric). If
scores stay stuck at ~0.95/0.92 without `correctness_scores` fields, the old
single-score process is still running.

If C/A rewards stay at 0, check that `AGENTIC_REFLECT_VLM_URL` is set and the
sidecar is healthy — the actor no longer calls a reflect tool during rollout.

Memory modes for Qwen-Image:

- `balanced`: split BF16 modules across **2+** visible GPUs.
- `full` / `model_offload` / `sequential_offload` / `mmdit_nf4`: see script help.

Without an image service, `generate_image` returns a text stub. Set
`REQUIRE_REAL_IMAGE_TOOL=0` only for plumbing diagnostics.

### 3) Run GRPO — pane C

```bash
TOTAL_STEPS=100 \
  N_GPUS=2 \
  OVERFIT_DATA=1 \
  bash examples/agenticrpco_trainer/agent_llm/run_agentic_grpo.sh
```

| Step | Behavior |
| --- | --- |
| Template | Qwen3-VL native Hermes JSON tool template |
| Data | `TRAIN_FILE` / `VAL_FILE` (defaults: `data/agentic/{train,val}.parquet`) |
| Agent loop | Stock verl `tool_agent` — no force / teacher token replacement |
| Tools | `diffusion_tool.py`: `generate_image` only |
| Observation | PIL pixels + `image_vis` facts; actor self-reflects |
| Reward | See tables above; frozen VL judges C/A offline via HTTP |
| Artifacts | `outputs/e2e/<experiment>/{rollout_trajectories,rollout_images,hermes_actions}/` |

Inspect learning:

```bash
# Valid <tool_call> blocks; agent reflection + Done. (or rewrite) after generate_image.
ls outputs/e2e/qwen3_vl_agentic_grpo_*/hermes_actions/
ls outputs/e2e/qwen3_vl_agentic_grpo_*/rollout_trajectories/step_*/
```

### Data-only refresh (no train)

```bash
python3 examples/agenticrpco_trainer/data_process/create_dummy_agentic_data.py \
  --local_save_dir data/agentic \
  --overfit --train_size 8 --val_size 2
```

Each row embeds one class demo:

| Class | Demo trajectory |
| ---: | --- |
| 0 | `generate` → Reflection + Done |
| 1 | `generate` → Reflection + rewrite `generate` → Reflection + Done |
| 2 | three `generate` / reflection-rewrite passes until Reflection + Done |

Then the live user turn (with brevity reminder). Runtime calls remain on-policy.

## File map

```
examples/agenticrpco_trainer/agent_llm/
├── qwen_image_tool_server.py
├── run_qwen_image_tool_server.sh
├── qwen_vl_reflect_server.py
├── run_qwen_vl_reflect_server.sh
├── run_agentic_grpo.sh
├── check_overfit_gates.py
└── run_lance_frozen_diffusion_tool_server.sh   # legacy Lance backend (optional)

verl_omni/agent_loop/
├── diffusion_tool.py                    # generate_image (attach image for self-reflect)
├── agentic_metrics_manager.py           # Stock manager + raw rollout/reward logs
├── agentic_image_reflection.py          # Supplemental image measurements
└── agentic_trajectory_context.py        # Artifact path helpers

verl_omni/utils/reward_score/agentic_reward.py
verl_omni/utils/reward_score/vl_reflect_client.py   # Frozen VL HTTP for C/A reward
examples/agenticrpco_trainer/data_process/create_dummy_agentic_data.py
```

Frozen image backends in `diffusion_tool.py` (first match wins):

1. `AGENTIC_QWEN_IMAGE_URL` — bundled Qwen-Image service
2. `AGENTIC_DIFFUSION_TOOL_URL` — generic POST `{"prompt"}` → image/text
3. `AGENTIC_LANCE_SERVER_URL` — legacy Lance backend
4. unset — text stub

Reflect backend (reward judge only; via `vl_reflect_client.call_reflect_vlm`):

1. `AGENTIC_REFLECT_VLM_URL` — frozen Qwen3-VL sidecar (`POST /reflect`)
2. unset / failure — C/A rewards are 0.0 (no heuristic fallback for reward)
