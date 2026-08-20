# PR 2 Implementation Plan — Multi-Dimensional Reward System & RPCO Stage-3 Training

Last updated: 08/17/2026


Note that in PR1's schema, one noteworthy thing is that for `outputs/rollout_trajectories/step_0000xx/sample_xx.json`, the
fields `"turn_prompt"`, `"turn_obs"`, `"decode"`, and `"response"` are as below:
* agent llm input full
* agent llm input prompt extracted
* tool call input literally
* agent llm forced output, if it's forced reflection

The reason that we log all the inputs here only (no output) except for the forced agent llm reflection, is that in multi-turn scenarios, the inputs is basically coming from the body of the last turn output, thus to avoid duplication, we don't need to log every turn output.

---

Plan for PR 2 of [RFC #302](https://github.com/verl-project/verl-omni/issues/302) (7.2).
Targets the `feat/rpco_and_rewards` branch, whose runner stub is
[`examples/agenticllmgrpo_trainer/agent_llm/run_agentic_rpco.sh`](../../examples/agenticllmgrpo_trainer/agent_llm/run_agentic_rpco.sh).

## 1. Real targets (override the RFC where they differ)

PR 2 implements an **e2e full-dataset GRPO training on the agentic LLM** so the agent learns:

1. **which tools to call** to carry out an image generation/editing task and meet the user request;
2. **when to continue a new turn vs. stop** the multi-turn loop.

Real targets (prevail over the RFC text):

| # | Target |
| --- | --- |
| T1 | Train on **UniCoT-Breakdown-3K** and/or **UniCoT-Self-Reflection-6K** (local copies under `~/hf_home/hub/`) |
| T2 | Include RFC §6.5 **Stage 3** (multi-task RL co-optimization). Stages 1–2 are optional hooks, not deliverables. Other RFC scope that conflicts with the above (8-dim table, trajectory resampling §6.4, HTTP multi-dim protocol) is ignored |
| T3 | All changes are launched from `agent_llm/run_agentic_rpco.sh` |

What is **in scope**:

- UniCoT source adapters → agentic RL parquet (both datasets).
- A named multi-dimensional reward set **{reflection, plan, format, tool, result}** (`R_reflect`, `R_plan`, `R_format`, `R_tool`, `R_result`) with a configurable weighted total.
- Stage-3 multi-task co-optimization: one mixed run over single-image and multi-image tasks, per-task reward weight sets, checkpoint init from a prior strong-reflection run.

What is **out of scope** (per the real targets): RFC §6.4 trajectory resampling, the full
8-dimension table (`R_pointwise`, `R_pairwise`, `R_diffusion`), the multi-dim HTTP scorer
protocol (RFC D5), RPCO Stages 1–2 as shipped artifacts, trajectory analysis tools (RFC D8).

## 2. Current state on the branch

PR 1 ([#329](https://github.com/verl-project/verl-omni/pull/329)) already provides:

| Piece | File | Reuse for PR 2 |
| --- | --- | --- |
| Live multi-turn agent loop (forced Reflection, YES/max-pass → policy `Done.`) | [agentic_tool_agent_loop.py](../../verl_omni/agent_loop/agentic_tool_agent_loop.py) | unchanged core; plan tasks ride the same loop (see §5) |
| Frozen `generate_image` / `judge_image` tools + rollout-scoped artifact registry | [function_tools/tools.py](../../examples/agenticllmgrpo_trainer/function_tools/tools.py), [agentic_trajectory_context.py](../../verl_omni/agent_loop/agentic_trajectory_context.py) | unchanged; obs markers (`agentic_tool ok=…`, `agentic_judge ok=…`) are the reward inputs |
| Scalar gated reward with trajectory parsing helpers | [agentic_reward.py](../../verl_omni/utils/reward_score/agentic_reward.py) | import its parsers (`_extract_tool_calls`, `_gen_image_prompts`, judge-window iterators); PR 2 scorer subsumes the mix |
| Traj/hermes dumps + WandB metrics | [agentic_metrics_manager.py](../../verl_omni/agent_loop/agentic_metrics_manager.py) | extend `REWARD_COMPONENTS`; `rollout_valid` discard path reused as-is |
| Dummy parquet builder (row schema) | [create_dummy_agentic_data.py](../../examples/agenticllmgrpo_trainer/data_process/create_dummy_agentic_data.py) | row schema + `_tc()` wire-format helpers reused by the UniCoT builder |
| Generic visual-reflection data contracts + **UniCoT Self-Reflection adapter** | [visual_reflection/](../../verl_omni/utils/dataset/visual_reflection/) (`#313`) | `unicot.py` fail-closed parser + `partition.py` splits + `LocalImageResolver` reused; a **Breakdown adapter is new** |

`run_agentic_rpco.sh` is currently byte-identical to `run_agentic_grpo_lora.sh` — PR 2 fills it in.

## 3. Dataset pipeline

### 3.1 Task taxonomy

Measured on the local snapshots (08/17/2026):

| Dataset | Rows | Single-image | Multi-image (plan) | Fields |
| --- | ---: | ---: | ---: | --- |
| UniCoT-Self-Reflection-6K | 5970 | all (1–3 reflection states; 1686 rows contain ≥1 `continue`) | — | `prompt`, `input_image`, `eval`/`eval_summary`, `edit`, `output_image` |
| UniCoT-Breakdown-3K | 3569 | 2398 (`"No breakdown needed."`) | 7 × 1-subtask, 367 × 2-subtask, 797 × 3-subtask | `prompt`, `subtasks`, `subtask_images` |

Each row maps to one **task type**:

- **`reflect`** — Self-Reflection rows + Breakdown `"No breakdown needed."` rows: single-image
  task, the agent runs the PR 1 protocol (generate → judge → reflect → stop or rewrite).
- **`plan`** — Breakdown rows with ≥1 subtask: the agent first writes a plan (subtask
  prompts), generates one image per subtask, judges the **final** image, then reflects and
  stops. Reference subtasks are the ground truth for `R_plan`; the subtask count is the
  ground truth for `R_result` (expected image count).

Both task types share the same loop, tools, and forced-Reflection machinery; the behavioral
difference is carried by the **per-task system prompt** baked into the parquet (§3.3), not by
loop code.

### 3.2 Reference as reward ground truth (not supervised text)

UniCoT fields are used **only** as reward references, never as fewshot/supervised targets:

| UniCoT field | Used by |
| --- | --- |
| `subtasks` / `"No breakdown needed."` | `R_plan` reference; `expected_num_images` |
| `eval_summary` (fallback cleaned `eval`) | `R_reflect` reference |
| transition structure (`output_image[i]` null/non-null) | `expected_num_images` (number of states); optional strict continue/stop reference |
| `input_image` / `output_image` / `subtask_images` | **optional** image-backed judging (§3.5); not required for e2e v1 |

### 3.3 New builder: `visual_reflection/build_unicot_agentic_rl.py`

CLI:

```
python3 -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl \
    --breakdown_dir ~/hf_home/hub/datasets--Fr0zencr4nE--UniCoT-Breakdown-3K \
    --reflection_dir ~/hf_home/hub/datasets--Fr0zencr4nE--UniCoT-Self-Reflection-6K \
    --local_save_dir data/agentic_unicot \
    [--mix_ratio 0.5] [--train_size N] [--val_size N] [--seed ...]
```

Behavior:

1. Load both `metadata.json` files directly (no HF `datasets` dependency).
2. **Self-Reflection**: parse with the existing fail-closed
   [`parse_unicot_record`](../../verl_omni/utils/dataset/visual_reflection/unicot.py)
   (`manifest_id="agentic_rl_<stage>"`, `LocalImageResolver` only when images are needed).
   Rows that fail validation are logged and dropped (reuse
   [`RejectionLedger`](../../verl_omni/utils/dataset/visual_reflection/provenance.py)
   pattern from #313).
3. **Breakdown**: new fail-closed adapter `verl_omni/utils/dataset/visual_reflection/unicot_breakdown.py`
   (mirror `unicot.py` structure):
   - `prompt` non-empty; `subtasks` list of length 3, entries either `None` or non-empty text;
   - a row with `subtasks[0] == "No breakdown needed."` must have `subtasks[1:] == [None, None]`
     → `plan_required=False`, `expected_num_images=1`;
   - otherwise `expected_num_images = count of non-None subtasks` and all non-None subtasks
     precede all `None`s (no gaps); contradictory rows are rejected.
   - Canonical output mirrors `VisualReflectionTrajectory`-style dicts but is planner-agnostic:
     `{"data_id", "prompt", "task_type": "plan", "subtasks": [...], "expected_num_images": n}`.
4. Partition into train/val with the existing
   [`assign_source_splits` / `make_split_provenance`](../../verl_omni/utils/dataset/visual_reflection/partition.py)
   (hash-based split by `data_id`, so both task types land in both splits deterministically).
5. Emit parquet rows in the exact PR 1 schema (`data_source`, `prompt` = messages,
   `ability`, `reward_model.ground_truth`, `extra_info`):

```
data_source:   "unicot_reflection" | "unicot_breakdown"   # reward scorer branches on this
prompt:        [system (per task type), user task (+ brevity tail)]
ability:       "agentic_generate_self_reflect" | "agentic_plan_generate"
reward_model:  {"style": "rule", "ground_truth": gt}
gt:            {user_request, task_type, expected_num_images,
                reference_subtasks?            # plan rows
                reference_steps?               # reflect rows: [{reflection, action, edit}]
                w_reflect, w_plan, w_format, w_tool, w_result}   # per-stage weight set, see §4.3
extra_info:    {split, index, data_id, task_type, expected_num_images, raw_prompt, <weights>}
```

- **No fewshot demos** in v1 — PR 1 added fewshot only to bootstrap the overfit run; on
  3K/6K rows the system prompt carries the protocol, and `AGENTIC_FORCE_FIRST_GENERATE`
  warmup still applies. (A per-task fewshot can be added later if cold-start stalls.)
- The reflect system prompt is PR 1's `SYSTEM_PROMPT`; the plan system prompt is new and
  tells the agent: write a short plan (one subtask prompt per required image), call
  `generate_image` once per subtask, then `judge_image` on the final image, then
  `Reflection: … Done.` — same HARD RULES and brevity tail.

### 3.4 Images

`images.zip` exists in both snapshots but is **not required for e2e v1**: all five rewards
score trajectory text against text references. Image materialization (unzip →
`LocalImageResolver`) is deferred to the optional image-backed `R_reflect` checkpoint mode
(§4.2) and to evaluation.

## 4. Multi-dimensional reward system

### 4.1 Wiring decision

One scorer, not five. The token path (`verl.trainer.main_ppo`) invokes a single
`compute_score(data_source, solution_str, ground_truth, extra_info)` per rollout — the same
contract PR 1 uses. PR 2 therefore ships **one** scorer that computes the five dimensions
internally and returns `score` plus one `reward_<dim>` field each:

```
verl_omni/utils/reward_score/agentic_multidim_reward.py
    compute_score(...) -> {"score", "reward_reflect", "reward_plan", "reward_format",
                           "reward_tool", "reward_result", <PR1-compat counters>, ...}
```

This satisfies the RFC's "configurable active set W per training run" via the per-row weight
set in `gt`/`extra_info` (§4.3) — exactly the mechanism `agentic_reward.compute_score`
already uses for `w_tool_call` etc. Trajectory parsing helpers
(`_extract_tool_calls`, `_gen_image_prompts`, `_iter_successful_judge_scores`,
`_policy_terminal_decision`, `_assistant_prose`, `_zero_result` skeleton) are **imported**
from [agentic_reward.py](../../verl_omni/utils/reward_score/agentic_reward.py) rather than
copied (repo rule: reuse over duplication); `_zero_result` gains the five new keys so Ray
reward workers never KeyError.

### 4.2 The five dimensions

All values are per-trajectory, computed from `solution_str` (full multi-turn decode,
including tool obs) plus `gt` references:

| Dim | Range | Computation | Notes |
| --- | --- | --- | --- |
| **R_plan** | [0,1] | Requirement coverage of the agent's plan lines against the reference subtasks (per-subtask best token overlap, then mean); zero when no plan text | Only in W on `plan` rows. Paper (Eq. 7) uses an external LLM evaluator for completeness/coherence/tool–goal matching — the rule-based coverage is the deterministic v1 fallback; the LLM evaluator via `AGENTIC_VLLM_URL` remains the documented upgrade path |
| **R_reflect** | [0,1] | Satisfied-checkpoint ratio (paper Eq. 5): mean of the last successful judge's C/A facets (= checkpoints) on the final image, blended 0.5/0.5 with lexical coverage of the agent's `Reflection:` against the UniCoT reference `eval_summary` (or the live judge findings when no reference exists) | Policy-sampled Reflection only; injected cues are stripped |
| **R_format** | [0,1] | Rule-based check ratio: well-formed tool calls, judge after the last generate, terminal policy-sampled `Done.`, plus the task-type tags (`Plan:` + `Reflection:` on plan rows; `Reflection:` on reflect rows) | trajectory-level ratio across checks |
| **R_tool** | {0,1} | Tool-call presence, mapped to PR 1's `f_tool_call` (`agentic_reward.py`): 1.0 iff any tool call was parsed; the discrete self-correction ladder is dropped | emits `reward_tool_call` alias so the PR 1 WandB series revives |
| **R_result** | {0,1} | Output count/type match vs. `expected_num_images`. **Plan rows**: exact count match. **Reflect rows (lenient stop-validity)**: terminal `Done.` + ≥1 image + (count ≤ expected OR final judge YES) | count is the dataset's subtask/transition count — the primary **when-to-stop** signal. `f_done` (PR 1's closed-loop indicator) is logged separately as `reward_done`, not scored |

Gating kept from PR 1: no `generate_image` / no successful PNG ⇒ `score=0`,
`rollout_valid=0` (rollout is dropped from the GRPO update by
[`_discard_invalid_rollouts`](../../verl_omni/agent_loop/agentic_metrics_manager.py)).
Env-injected `Reflection` (mask 0, `agentic_forced_reflection=1` markers) is stripped before
`R_reflect` scoring so only policy-sampled text earns credit (same regex PR 1 uses).

### 4.3 Weighted total and Stage-3 active sets

```
score = (1 / |W|) * Σ w_i * R_i          # W = dims with w_i > 0 that apply to this row
```

Per VisionCreator-R1, **all weights default to 1.0** (overridable via `RPCO_W_*` env vars
at data-build time), and the active set is task-shaped: `plan` rows use the full
{reflection, plan, format, tool, result} set (paper §4.3); `reflect` rows use
{reflection, format, tool, result} — the paper's §4.1 set, since `R_plan` only applies
"at the plan stage". Plan/reflect are the dominant learning signals; format/tool/result
are the structural regularizers.

### 4.4 Metrics

Extend [`REWARD_COMPONENTS`](../../verl_omni/agent_loop/agentic_metrics_manager.py) to
`("reward_reflect", "reward_plan", "reward_format", "reward_tool", "reward_result")` so
WandB logs `agentic_reward/<dim>/{mean,min,max}`; the counters already emitted
(`num_generate_image_prompts`, `protocol_ok`, …) stay for dashboards.

## 5. Agent loop / protocol adaptations for plan tasks

The loop itself needs **no structural change** for `plan` rows:

- Multiple `generate_image` calls per rollout are already supported; the env hard-stop
  `AGENTIC_MAX_GENERATE_IMAGE_PASSES` (default 3) equals the dataset's max subtask count
  (3-subtask rows: 367+797). 3-subtask rows therefore never trip the blocker while
  single-image rows keep the PR 1 cap.
- `judge_image("…", "last")` judges the latest PNG — for plan rows that is the final
  subtask image, which is the semantically correct target.
- Forced Reflection after a successful judge
  ([`agentic_tool_agent_loop.py`](../../verl_omni/agent_loop/agentic_tool_agent_loop.py),
  `_handle_processing_tools_state`) applies identically: plan rows judge only at the end,
  so the stop/continue cue arrives after the full plan execution.

Changes actually needed:

1. **Force-first warmup stays on** (`AGENTIC_FORCE_FIRST_GENERATE`) — it teacher-forces the
   first `generate_image` but not plan text; the plan prompt comes from the data-driven
   system prompt.
2. **Plan parsing helper** shared by reward and metrics dumps: a small
   `_extract_plan_lines(text)` (regex for `Plan:`/numbered blocks) lives next to the
   multidim scorer so `R_plan` and any future analysis tool agree.
3. **(Optional, later)** an `edit_image` tool for `reflect` rows whose reference action is
   `continue` — needs a frozen edit sidecar (`AGENTIC_EDIT_IMAGE_URL`, same contract as
   `generate_image`). v1 treats "edit" as prompt rewriting + `generate_image` (the PR 1
   protocol) and scores it via `R_reflect`/`R_tool`; `edit_image` is explicitly deferred.

## 6. RPCO Stage 3 — multi-task RL co-optimization

What the runner must express (all in `run_agentic_rpco.sh`, env-overridable):

| Setting | Value |
| --- | --- |
| Datasets | mixed parquet: reflect pool (Self-Reflection + Breakdown no-breakdown rows) and plan pool (Breakdown subtask rows); `--mix_ratio` controls single:multi sampling (default 1:1, cap multi-image rows at ~3 gens to bound rollout latency) |
| Reward | `agentic_multidim_reward.compute_score` with §4.3 per-row weight sets |
| Init | `RPCO_INIT_CKPT` → `actor_rollout_ref.model.path` (LoRA weights from a prior Stage-1 reflection run) and/or `trainer.resume_mode=auto`; when unset, cold-start from the base VLM |
| Protocol env | `AGENTIC_MAX_GENERATE_IMAGE_PASSES=3`, `AGENTIC_FORCE_REFLECTION_AFTER_JUDGE=1`, `AGENTIC_BLOCK_GENERATE_AFTER_YES=1`, `AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES=1` (all existing flags) |

Stages 1–2 (reflection-only RL, advantage-complementary SFT) are **not implemented** as
shipped artifacts; the runner's `RPCO_INIT_CKPT` hook is their integration point if a
strong-reflection checkpoint exists. This matches real target T2.

## 7. `run_agentic_rpco.sh` changes (vs. `run_agentic_grpo_lora.sh`)

1. Replace the dummy-data call with
   `build_unicot_agentic_rl` (§3.3), default `TRAIN_FILE`/`VAL_FILE` →
   `data/agentic_unicot/{train,val}.parquet`.
2. `reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_multidim_reward`.
3. New env knobs: `UNICOT_BREAKDOWN_DIR`, `UNICOT_REFLECTION_DIR`, `UNICOT_MIX_RATIO`,
   `RPCO_INIT_CKPT`, `RPCO_W_*` weight overrides, `UNICOT_TRAIN_SIZE`/`UNICOT_VAL_SIZE`.
4. Bump batch geometry for full-dataset throughput (`data.train_batch_size` ≥ 16, higher
   `rollout.max_num_seqs`), keep LoRA/FSDP/offload settings from PR 1.
5. Keep the frozen-sidecar guard (both `AGENTIC_*_URL` checks) unchanged; the same
   gen/judge servers serve both task types.

## 8. Testing & validation

CPU tests (repo convention: `test_*_on_cpu.py`):

| Test | Covers |
| --- | --- |
| `tests/utils/dataset/test_unicot_breakdown_on_cpu.py` | Breakdown adapter fail-closed rules (gap in subtasks, "No breakdown needed." with trailing subtasks, missing fields) + `expected_num_images` derivation |
| `tests/utils/reward_score/test_agentic_multidim_reward_on_cpu.py` | Deterministic rule-based dims on synthetic trajectories: R_format tag order, R_tool {0, 0.1, 0.8, 1.0} ladder, R_result count match, R_plan coverage, R_reflect vs. injected-marker stripping, weight sets, zero-result schema completeness |
| `tests/utils/dataset/test_build_unicot_agentic_rl_on_cpu.py` | Parquet builder schema + split determinism on small synthetic metadata (or first-N real rows when `HF_HOME` is present, mirroring `test_visual_reflection_unicot_on_cpu.py`) |

E2E acceptance (pane A/B sidecars + pane C):

1. Mixed UniCoT parquet run completes N≥50 steps; traj dumps show `plan` rows with N
   successful gens == subtask count and `reflect` rows with stop/continue decisions.
2. WandB `agentic_reward/{reflection,plan,format,tool,result}` all move; `rollout_valid`
   stays high.
3. `RPCO_INIT_CKPT` resume from a Stage-1-style checkpoint starts and trains (structure
   check; full Stage-1 run is not required).
4. Agent LLM weights update while gen/judge sidecars stay frozen (Mode (2a) preserved).

## 9. Files touched

| File | Change |
| --- | --- |
| `verl_omni/utils/dataset/visual_reflection/build_unicot_agentic_rl.py` | UniCoT → agentic RL parquet builder |
| `verl_omni/utils/dataset/visual_reflection/unicot_breakdown.py` | new: fail-closed Breakdown adapter |
| `verl_omni/utils/reward_score/agentic_multidim_reward.py` | new: 5-dim scorer, imports PR 1 parsers |
| `verl_omni/agent_loop/agentic_metrics_manager.py` | `REWARD_COMPONENTS` → 5 dims (+ `agentic_multidim_reward` re-export if the manager references scorer keys) |
| `examples/agenticllmgrpo_trainer/agent_llm/run_agentic_rpco.sh` | §7 |
| `examples/agenticllmgrpo_trainer/README.md` | RPCO section + runner usage; `Last updated` bumped |
| `tests/utils/dataset/test_unicot_breakdown_on_cpu.py`, `tests/utils/reward_score/test_agentic_multidim_reward_on_cpu.py`, `tests/utils/dataset/test_build_unicot_agentic_rl_on_cpu.py` | new CPU tests |

## 10. Order of work

1. Breakdown adapter + CPU tests (pure data, no GPU).
2. `build_unicot_agentic_rl.py` builder + split/parquet tests.
3. `agentic_multidim_reward.py` + CPU reward tests (rule-based paths first, LLM-judge
   paths behind env flags).
4. Metrics manager extension.
5. `run_agentic_rpco.sh` wiring + overfit-scale smoke (a few hundred rows per task type).
6. Full-dataset e2e run + RPCO init-checkpoint check.

## 11. Open questions / risks

- **LLM-evaluator dims**: rule-based `R_plan`/`R_reflect` coverage is weak for paraphrased
  plans/reflections. Plan: ship rule-based v1 (done), add `AGENTIC_VLLM_URL`-based LLM
  judging as a second pass (deterministic fallback keeps CPU tests green).
- **Rollout latency**: 3-subtask rows × gen sidecar are the slow tail of the mixed batch;
  `--mix_ratio` and `AGENTIC_MAX_GENERATE_IMAGE_PASSES` bound it, may need per-task caps
  later.
- **Diffusion stochasticity on reflect rows** (the §6.5 asymmetry): mitigated only by
  Stage-3 weight structure + `RPCO_INIT_CKPT`; if `R_reflect` variance dominates in the
  first e2e, add the RFC's Stage-1 reflection-only warmup run before the mixed run.
- **Resolved**: `data_source` = `unicot_reflection` | `unicot_breakdown` (dataset origin);
  the scorer branches on `gt.task_type`. Reflect-row `R_result` uses lenient
  stop-validity (user decision). Weights default to 1.0 per the paper.
