# Agentic LLM Reflection-Plan Co-Optimization (RPCO)

Last updated: 08/11/2026

Training recipes for **Agentic LLM RL** ([#302](https://github.com/verl-project/verl-omni/issues/302)).
In this example, we conduct LoRA overfitting on this, where the Agent LLM, diffusion tool, and VL judge 
can be **changed** as you need; this example uses:

- **Agent LLM** (`agent_llm/`): train `Qwen3-VL-2B-Instruct` (or `Qwen3.5` via `MODEL_PATH`).
- Frozen **Qwen-Image** via vLLM-Omni is the external `generate_image` tool (`:8092`).
- Frozen **Qwen3-VL** via vLLM (`:8093`) serves dual role: (1) in-turn
  `judge_image` after every generation, and (2) `reward_correctness` /
  `reward_aesthetics` fallback at reward time (prefer last live judge obs).
- Reward: `pkg://verl_omni.utils.reward_score.agentic_reward` — parseable
  `<tool_call>` protocol (Hermes JSON for Qwen3 **or** Qwen3.5 XML) with
  policy-sampled terminal `Done.` (or rewrite) gating.
- Actor wire format is auto-selected from `MODEL_PATH` in `data/qwen35_env.sh`
  (sourced by `run_agentic_grpo_lora.sh`):
  - **Qwen3-VL** → `multi_turn.format=hermes` + Hermes JSON fewshots
  - **Qwen3.5** → `multi_turn.format=qwen3_coder` + XML fewshots + GDN preflight
  Override with `TOOL_PARSER_FORMAT` / `--tool_call_format`.

Target protocol (fewshot + on-policy) — see **Multi-turn Behaviors** diagrams:

```
Turn k:
  1. generate_image(prompt_k) → image_k
  2. judge_image("same as user message", "last") → VL feedback
  3. forced Reflection (mask=0) then:
    YES / max-pass → stop cue → policy samples Done. (mask=1)   OR
    NO → policy rewrite + generate_image(prompt_{k+1}) → Turn k+1
```

Fewshot demos in `create_dummy_agentic_data.py` follow that order. They are
**GRPO exploration examples**, not supervised targets. Overfit fewshot omits the
terminal `Done.` so the live turn cannot copy Done-without-tools. Regenerate
parquet when switching actor family so fewshot `<tool_call>` syntax matches the
chat template.

## Rollout Trajectories

Each training step dumps per-rollout JSON under
`outputs/e2e_…/<run>/rollout_trajectories/step_XXXXXX/sample_{dataset}.{rollout_n}.json`
and matching PNGs under `rollout_images/…/sample_Y.ZZ/image_NN_<artifact_id>.png`.

Typical force-on trajectory (`AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=1`):

| `turn_kind` | Meaning |
| --- | --- |
| `call_generate_image` | First (or non-forced) `generate_image` in `decode` |
| `call_judge_image` | Compact `judge_image` tool call |
| `agent_rewrite_after_forced_reflection` | Forced Reflection in `response` + rewrite `generate_image` in `decode` (**counts as a live gen**) |
| `forced_reflection_stop_cue` | YES → masked Reflection asks the policy to emit terminal `Done.` |
| `forced_reflection_max_passes_stop_cue` | Cap reached → masked stop cue (`agentic_force_stop_max_passes=1`) |
| `agent_done_after_forced_reflection` / `agent_done_after_max_passes` | Policy-sampled terminal `Done.` (**earns Done credit**) |

Do not grep only `call_generate_image` when counting images — rewrites after forced
Reflection use `agent_rewrite_after_forced_reflection` but still write PNGs.

## Reward components

`compute_score` returns a scalar `score` plus per-component fields. WandB
`agentic_reward/*` logs **only the scalar mix terms** (via
`agentic_metrics_manager.REWARD_COMPONENTS`):

- `agentic_reward/tool_call/{mean,min,max}`
- `agentic_reward/correctness/{mean,min,max}`
- `agentic_reward/aesthetics/{mean,min,max}`
- `agentic_reward/done/{mean,min,max}`

`reward_correctness` / `reward_aesthetics` prefer the last successful
`agentic_judge ok=1` observation already in the trajectory (same C/A the actor
saw). If absent, reward falls back to `AGENTIC_VLLM_URL` / legacy
`AGENTIC_REFLECT_VLM_URL`. Per-dimension facet fields may still appear in
`hermes_actions` JSONL but are **not** logged under `agentic_reward/*`.

**Overfit learning signal:** open `generate→judge` loops no longer keep a mid
plateau from high frozen-judge C/A. Scalar mix credits C/A fully only after a
successful PNG + successful judge + policy-sampled terminal `Done.`. A masked
forced Reflection may supply the reflection context, but never the terminal
action. Incidental prose such as “Stop when Done.” earns zero Done credit, and
blocked/no-PNG trajectories cannot close the protocol. The launch script sets LoRA `actor.optim.lr=1e-4`
(verl default `1e-6` was too small to move reward in 100 steps).


## Multi-turn Behaviors (three turns max)

**Target contract**:

1. Every logical turn starts with `generate_image(prompt_k)`.
2. In the **same** logical turn, after the image returns:
   - agent calls compact `judge_image` → frozen Qwen3-VL (`:8093`) returns structured feedback;
   - env injects masked Reflection (stop cue on YES/max-pass; continue on NO).
3. On a stop cue, the **next** physical turn is a policy-sampled `Done.` (`mask=1`).
   On NO, the policy rewrites `generate_image` in the same decision turn.
4. Branching stops at 1 / 2 / … / N gens (`AGENTIC_MAX_GENERATE_IMAGE_PASSES`,
   typically 3; tunable).

### Physical turns (trajectory JSON) vs one logical gen→judge→reflect

A **physical turn** is one entry in `rollout_turns[]` (`turn`, `turn_kind`,
`turn_prompt`, `turn_obs`, `decode`, `response`) — e.g. turn 3 in
`sample_1.02.json` with `turn_kind=agent_rewrite_after_forced_reflection`:
full chat prefix in `turn_prompt`, latest judge obs in `turn_obs`, forced
`Reflection:` in `response`, rewrite `generate_image` in `decode`.

One **logical** pass (generate → judge → decide) spans **2–3 physical turns**:

| Physical `turn` | Typical `turn_kind` | Policy `decode` | Env `response` |
| ---: | --- | --- | --- |
| *t* | `call_generate_image` (first) **or** `agent_rewrite_after_forced_reflection` | `generate_image(…)` | empty on first gen; forced `Reflection:` on rewrite turns |
| *t+1* | `call_judge_image` | `judge_image(same as user message, last)` | empty |
| *t+2* if YES / max-pass | `forced_reflection_stop_cue` / `forced_reflection_max_passes_stop_cue` then `agent_done_after_*` | policy `Done.` (or `Reflection: … Done.`) | masked stop cue in `response` |
| *t+2* if NO | `agent_rewrite_after_forced_reflection` | rewrite `generate_image(…)` | `Reflection: …` (then this turn is also gen *t* of the next logical pass) |

Example: physical 1=`call_generate_image`, 2=`call_judge_image`,
3=`agent_rewrite_after_forced_reflection`, …,
7=`forced_reflection_max_passes_stop_cue`, 8=`agent_done_after_max_passes`.

```mermaid
sequenceDiagram
  autonumber
  participant In as Context
  participant A as Agent LLM
  participant G as Qwen-Image
  participant V as Qwen3-VL

  Note over In,A: Physical turn t - call_generate_image or rewrite
  In->>A: full turn_prompt
  A->>G: generate_image prompt_k
  G-->>A: tool obs with path

  Note over In,A: Physical turn t+1 - call_judge_image
  A->>V: judge_image compact args
  V-->>A: scores findings good_enough

  Note over A: Forced Reflection after successful judge
  alt stop YES or max passes
    Note over A: masked Reflection supplies a stop cue
    A-->>In: policy samples terminal Done
  else continue good_enough NO
    Note over A: agent_rewrite_after_forced_reflection
    A-->>In: Reflection plus rewritten generate_image
  end
```

### What “turn input” must carry

| Logical turn | Input to agent decode that emits `generate_image` | Same-turn after image |
| --- | --- | --- |
| 1 | User task (+ system / fewshot) | VL judge(`image_1`) → stop cue → policy `Done.` **or** rewrite `prompt_2` |
| 2 | **Turn-1 reflection results** (VL summary + rewrite), not only the raw image obs | VL judge(`image_2`) → stop cue → policy `Done.` **or** rewrite `prompt_3` |
| 3 | **Turn-2 reflection results** | VL judge(`image_3`) → max-pass stop cue → policy `Done.` |

### Current rollout behavior

Each logical turn follows the contract above. Default for this overfit path is
force-reflection **on** (`AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=1`):

- Turn k: `generate_image(prompt_k)` → compact `judge_image(...)` → **injected**
  `Reflection:` carrying the stop/continue verdict from `good_enough`
  (stop cue on YES / max-pass; else continue so the policy can rewrite).
  Injected tokens use `response_mask=0`. On a stop cue, the next turn is a
  policy-sampled `Done.` (`response_mask=1`) so GRPO receives real terminal credit.
- Cap successful generates with `AGENTIC_MAX_GENERATE_IMAGE_PASSES` (operator
  default **3**; launch fallback **5** — tune freely).
  `AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES=1` also refuses further
  `generate_image` calls past the cap.
- **Env hard-stop (default on):** after `good_enough=YES`, further
  `generate_image` calls are refused (`agentic_block_generate_after_yes=1`).
- **Compact `judge_image` args:** prefer `user_request="same as user message"` and
  `image_prompt="last"`; the tool expands from the bound user task + latest
  artifact so long pasted args do not truncate Hermes tool calls.
- **YES bar (default 0.86):** `AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD` /
  `AGENTIC_REFLECT_GOOD_ENOUGH` — YES iff `C≥thr` **and** `A≥thr` (model
  `good_enough` flag ignored). This yielded about 51% early first-pass YES at
  Qwen-Image STEPS=16, retaining both stop and rewrite exploration.
- **Image seeds:** `QWEN_IMAGE_SEED=42` with `QWEN_IMAGE_DIVERSIFY_SEED=1`
  (default) derives a stable per-rollout/per-pass seed so GRPO groups keep
  reward variance without fully randomizing images.
- **GRPO group size:** `ROLLOUT_N` defaults to **8**.
- Turn k+1 input is the prior tool obs + the forced Reflection (and any policy rewrite).

Set `AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=0` only if you want the **policy** to
emit Reflection itself (max-pass stop cue + policy Done still apply).

### Rollout trajectory JSON protocol

Each ``rollout_trajectories/step_XXXXXX/sample_Y.ZZ.json`` lists ``rollout_turns``
with this key order:

| Key | Meaning |
| --- | --- |
| `turn` | 1-based index in the response tensor |
| `turn_kind` | Grepable stage label (see **Rollout Trajectories** table) |
| `turn_prompt` | Chat context / prior obs the policy conditions on for this decode |
| `decode` | Policy-sampled assistant tokens (`<tool_call>…` or prose) |
| `response` | Injected assistant text after the tool obs (forced Reflection when force=1) |
| `decode_has_tool_call` | `true` iff **`decode`** contains `<tool_call>` (ignores `response`) |

Important: **`decode` of turn T need not equal `turn_prompt` of turn T+1`.**
Fewshot demos in `create_dummy_agentic_data.py` match live tool obs (`path=` +
`agentic_tool` / `agentic_judge` markers; no `image_vis` or instructional
`REQUIRED NEXT ACTION` lines).

Each live `generate_image` writes directly to the matching
`rollout_images/step_XXXXXX/sample_Y.ZZ/image_NN_<artifact_id>.png` directory
while that rollout is running. ``artifact_id`` is a sha12 of
`(trajectory_relpath, index, prompt)` so concurrent same-prompt overfit
rollouts never collide in judge lookup. Trajectory dumps use the **same**
`sample_Y.ZZ` id:

| Dump | Images |
| --- | --- |
| `rollout_trajectories/step_S/sample_Y.ZZ.json` | `rollout_images/step_S/sample_Y.ZZ/` |

The JSON field `image_dir` is the absolute path to that folder; `image_paths`
lists PNGs found there. Both are empty when that rollout generated no image,
and post-processing does not create a meta-only image folder. Prefer
`image_dir` over guessing from `sample_index` alone — GRPO uses
`sample_{dataset_index}.{rollout_n:02d}` (e.g. `sample_6.00` … `sample_6.07`
are eight rollouts of the same prompt). The dumper stamps `trajectory_relpath`
from the live agent loop so dump names match `path=` markers in tool obs.

### Primary fields (enter the weighted mix)

| Field | Default weight | Physical meaning | Zero when |
| --- | ---: | --- | --- |
| `reward_tool_call` | 0.10 | Binary: trajectory contains ≥1 parseable `<tool_call>`. **In scalar.** | No parseable tool call |
| `reward_done` | 0.20 | Closed loop: successful PNG + successful judge + **policy-sampled** terminal `Done.` (forced Reflection may supply context; never the terminal). **In scalar.** | No policy terminal / blocked / no judge |
| `reward_brevity` | — | Short assistant prose (≤4 sentences / ≤280 chars). **Metric only.** | Long rambling / debate CoT |
| `reward_format` | — | Fraction of `generate_image` calls with valid arguments. **Metric only.** | No / malformed calls |
| `reward_reflection` | — | Quality of agent self-reflection prose. **Metric only.** | No reflection prose |
| `reward_tool_usage` | — | Protocol shape and distinct rewritten prompts. **Metric only.** | No `generate_image` |
| `reward_result` | — | Closed-loop outcome quality. **Metric only.** | No `generate_image` |
| `reward_correctness` | 0.35 | Frozen-VL correctness (logged raw; **mix-gated** until closed). | URL unset / VL call fails |
| `reward_aesthetics` | 0.35 | Frozen-VL aesthetics (logged raw; **mix-gated** until closed). | URL unset / VL call fails |

Weighted mix (then scaled by a protocol tier `base + scale * mix`). C/A enter the
mix at full weight only after a closed policy terminal; open loops get 5% C/A:

```text
mix = Σ w_i * reward_i  /  Σ w_i
score = base + scale * mix
```

| Protocol tier | Condition | `base` | `scale` | `protocol_ok` |
| --- | --- | ---: | ---: | ---: |
| Closed + high C/A | policy terminal `Done.` (+ policy or forced Reflection context), single-pass or distinct rewrite, C/A ≥ 0.70 | 0.10 | 0.90 | 1 |
| Closed, VL down / weak C/A | closed loop; VL missing or C/A &lt; 0.70 | 0.05 | 0.65 | 1 / 0 |
| Reflection present | ≥1 `generate_image` + `Reflection:`, not fully closed | 0.04 | 0.30 | 0 |
| Gen-only (starved) | ≥1 `generate_image`, **no** closed terminal | 0.02 | 0.05 | 0 |

Weights are stored per row in parquet `ground_truth` / `extra_info` (`w_tool_call`,
`w_correctness`, `w_aesthetics`, `w_done`). Brevity, format, reflection, tool_usage, and
result are logged as metrics but excluded from the scalar reward (saturated from
step 1).

### Visual rubric diagnostics

| Field | Physical meaning |
| --- | --- |
| `reward_correctness` / `reward_aesthetics` | Last `agentic_judge ok=1` C/A (or VL fallback) — **logged to WandB** |
| `reward_correctness_*` / `reward_aesthetics_*` | Optional per-dimension diagnostics in JSONL only (not WandB mix) |
| `protocol_ok` | 1 iff closed loop (policy terminal Done after gen+judge, single or distinct rewrite) |
| `num_hermes_tool_calls` | Legacy field name: count of parseable JSON/XML `<tool_call>` blocks |
| `num_generate_image_prompts` | Count of `generate_image` prompts |
| `num_judge_image_calls` | Count of `judge_image` tool calls in the trajectory |

## Prerequisites

- Launch from the **verl-omni repo root** (repo-relative `function_tool_path`).
  `run_agentic_grpo_lora.sh` auto-`cd`s to the repo root.
- Set **`MODEL_PATH`** to a Qwen3-VL Instruct (or Qwen3.5) snapshot.
- Prefer **2–4 free GPUs** for GRPO, **2 GPUs** for Qwen-Image/Omni, **1 GPU** for the
  VL judge sidecar.
- The launcher verifies the native tool template and image processor, then
  sources `data/qwen35_env.sh` for parser/GDN helpers.

## 100-step overfit e2e (recommended)

Goal: overfit a tiny prompt pool so the policy learns
**`generate_image` → judge_image → (forced Reflection) → policy Done. or rewrite**.

### 1) Operator env (example)

```bash
source ~/fred/fred_verlomni_agentic_multiturn_pr1.sh   # or your local env
export CUDA_VISIBLE_DEVICES=4,5,6,7   # train GPUs; keep others for image + reflect
export MODEL_PATH=.../Qwen3-VL-2B-Instruct/...
export TRAIN_FILE=$PWD/data/agentic/train.parquet
export VAL_FILE=$PWD/data/agentic/val.parquet
```

### 2) Start frozen tools

Pane A — Qwen-Image / vLLM-Omni (`generate_image`):

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh
# trainer: export AGENTIC_VLLM_OMNI_URL=http://127.0.0.1:8092
```

Pane B — judge VLM sidecar (agent `judge_image` tool + reward fallback):

```bash
CUDA_VISIBLE_DEVICES=2 \
  bash examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh
# trainer: export AGENTIC_VLLM_URL=http://127.0.0.1:8093
```

Restart the image sidecar after changing `QWEN_IMAGE_STEPS` / CFG. Judge thr is
client-side (`AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD`) and does not require a judge
restart.

Without an image service, `generate_image` returns a text stub. Set
`REQUIRE_REAL_IMAGE_TOOL=0` only for plumbing diagnostics.

### 3) Run GRPO — pane C

```bash
TOTAL_STEPS=100 \
  N_GPUS=4 \
  OVERFIT_DATA=1 \
  bash examples/agenticrpco_trainer/agent_llm/run_agentic_grpo_lora.sh
```

| Step | Behavior |
| --- | --- |
| Template | Native tool template (Hermes for Qwen3-VL; XML for Qwen3.5) |
| Data | `TRAIN_FILE` / `VAL_FILE` (defaults: `data/agentic/{train,val}.parquet`) |
| Agent loop | `agentic_tool_agent` — forced Reflection stop cue + policy Done (default on); force-first gen curriculum |
| Tools | `diffusion_tool.py`: `generate_image` + `judge_image` |
| Observation | text tool obs (`path=`, judge C/A); sidecar VL scores PNGs |
| Reward | See tables above; prefer live judge C/A in traj |
| Artifacts | `outputs/e2e_…/<experiment>/{rollout_trajectories,rollout_images,hermes_actions}/` |

Inspect learning:

```bash
ls outputs/e2e_qwen3_vl_2b_instruct_agentic_grpo/*/hermes_actions/
ls outputs/e2e_qwen3_vl_2b_instruct_agentic_grpo/*/rollout_trajectories/step_*/
```

### Data-only refresh (no train)

```bash
python3 examples/agenticrpco_trainer/data_process/create_dummy_agentic_data.py \
  --local_save_dir data/agentic \
  --overfit --train_size 8 --val_size 2 \
  --with_fewshot --tool_call_format hermes
```

Each overfit row uses one of two prompts. With `OVERFIT_FEWSHOT=1`:

| Prompt | Fewshot |
| --- | --- |
| soldier (idx=0) | Class-1 two-pass same-task demo (ends on YES judge, no terminal `Done.`) |
| epic fantasy (idx=1) | system + user only (no baked demo) |

Non-overfit rows still cycle demo classes 0/1/2. Compact fewshot
`judge_image` args match live (`same as user message` / `last`).

Then the live user turn (with brevity reminder). Runtime calls remain on-policy.

## Convergence Examples: Before vs. After

Healthy signal after the Done-credit fix (5-step smoke
`…_stop_credit_seed_smoke_20260811_021810`): `agentic_reward/done/mean`
0.75→1.0 and `critic/score/mean` 0.71→0.94 with `response_length/clip_ratio=0`.
Older runs that injected env `Done.` and stripped it from reward stayed flat
at `reward_done≈0`.

<img width="600" alt="Image" src="https://github.com/user-attachments/assets/f69bc365-b57e-4d15-b9c3-8643cf8336a3" />

## File map

```
examples/agenticrpco_trainer/
├── README.md
├── hyperparam_tune_list.md
├── migrate.md                            # port tip → clean checklist
├── agent_llm/
│   ├── run_agentic_grpo_lora.sh          # main GRPO launcher (sources data/qwen35_env.sh)
│   ├── run_qwen_image_tool_server.sh
│   ├── qwen_image_tool_server.py
│   ├── run_qwen_vl_reflect_server.sh
│   ├── qwen_vl_reflect_server.py
│   ├── qwen_vl_judge_log_middleware.py   # optional judge score logging
└── data_process/create_dummy_agentic_data.py

data/qwen35_env.sh                        # TOOL_PARSER_FORMAT + GDN preflight

verl_omni/agent_loop/
├── agentic_tool_agent_loop.py            # forced Reflection stop cue + policy Done / force-first
├── diffusion_tool.py                     # generate_image + judge_image + seed diversify
├── agentic_metrics_manager.py            # traj dumps + WandB reward components
└── agentic_trajectory_context.py         # rollout-scoped artifact / YES latch

verl_omni/utils/judge_parse.py
verl_omni/utils/reward_score/agentic_reward.py   # policy-terminal Done credit
verl_omni/utils/reward_score/vl_reflect_client.py
```

Frozen image backends in `diffusion_tool.py` (first match wins):

1. `AGENTIC_VLLM_OMNI_URL` — vLLM-Omni image generations
2. `AGENTIC_QWEN_IMAGE_URL` — bundled Qwen-Image HTTP service
3. `AGENTIC_DIFFUSION_TOOL_URL` — generic POST `{"prompt"}` → image/text
4. unset — text stub

Judge backends:

1. `AGENTIC_VLLM_URL` — OpenAI `/v1/chat/completions` (preferred)
2. `AGENTIC_REFLECT_VLM_URL` — legacy FastAPI `/reflect`
3. unset / failure — judge obs error / C/A rewards 0.0 on fallback path
