# PR2 Gating Plan & Testing Plan

**Branch:** TBD (depends on `feat/multiturn-traj-dual-policy` — PR1)
**RFC:** `docs/agentic/verl-omni-rfc-agentic-rl_v1.md` §7.2
**Assignee:** Frederick Hong (HaFred)
**Depends on:** PR 1 (trajectory format + agentic rollout)

---

## 1. PR2 Deliverables (RFC §7.2 lines 350-358)

### D1: Seven Multi-Dimensional Reward Scorers

| Scorer | Type | What it measures |
|--------|------|-----------------|
| `R_plan` | LLM evaluator | Plan quality vs. expected reasoning |
| `R_reflect` | VLM judge | Reflection quality vs. visual checkpoints |
| `R_format` | Rule-based | XML tag validation in agent output |
| `R_tool` | Rule-based | Discrete success/correction rates |
| `R_result` | Rule-based | Binary count match |
| `R_pointwise` | MLLM judge | Final-output quality assessment |
| `R_pairwise` | Rule-based | Shuffled monotonic-improvement check |
| `R_diffusion` | Existing | FlowGRPO terminal reward (carried over from PR1) |

### D2: Per-Turn Reward Computation

- Each turn receives its own reward signal (not just final outcome)
- Reward aggregation across turns

### D3: Reward Weight Configuration

- `R_total = (1/|W|) * sum(w_i * R_i)` with configurable active set W per training run

### D4: Reward Loop Manager Extension

- Dispatch to multiple concurrent scorers (LLM judge + VLM judge + rule-based)
- Async overlap between scorers

### D5: HTTP Scorer Protocol Extension

- Accept and return multi-dimensional reward responses
- JSON array of `{dimension, score, metadata}`

### D6: Trajectory Resampling

- Oversample G' > G trajectories per prompt
- Uniformly resample G by turn count

### D7: RPCO Staged Training

- Configurable reward sets per stage
- Checkpoint loading between stages
- Decouple-then-fuse pipeline:
  - Stage 1: Isolate reflection on single-image tasks
  - Stage 2: Advantage-complementary SFT init
  - Stage 3: Multi-task RL co-optimization

### D8: Trajectory Analysis Tools

- Reflection quality classification: under / good / over-reflection
- Edit semantic similarity metrics

---

## 2. Tests Carried Over from PR1 (Post-Merge Scope)

These tests were scoped out of PR1 per RFC lines 326-329 (*"production-scale run is not required"*) and lines 331-339 (*"Post-merge evaluation plan, non-blocking for PR 1"*).

### 2.1 GPU Tests (from PR1 ST-4 through ST-7)

| # | Test | Device | File:Line | What it verifies |
|---|------|--------|-----------|-----------------|
| ST-4 | Multi-turn trajectory end-to-end | 4× H800 80GB | [tests/special_e2e/test_agentic_multiturn_e2e.py](../../tests/special_e2e/test_agentic_multiturn_e2e.py) *(new)* | Verify `AgenticTrajectory` in `extra_fields` after rollout; `num_turns` > 1; prompt rewriting between turns |
| ST-5 | Loss mask gradient isolation | 4× H800 80GB | [tests/special_e2e/test_agentic_loss_mask_grad.py](../../tests/special_e2e/test_agentic_loss_mask_grad.py) *(new)* | Verify gradients are zero on observation positions; only agent token positions receive non-zero gradients |
| ST-6 | Rollout-train logprob Pearson consistency | 4× H800 80GB | [tests/special_e2e/test_agentic_logprob_consistency.py](../../tests/special_e2e/test_agentic_logprob_consistency.py) *(new)* | `training/rollout_actor_probs_pearson_corr` > 0.95 after weight sync |
| ST-7 | Lance-3B multi-step run | 4× H800 80GB | [tests/special_e2e/test_agentic_multistep_run.py](../../tests/special_e2e/test_agentic_multistep_run.py) *(new)* | 10-step training: reward increases, Pearson > 0.95, no memory leak |

### 2.2 CPU Tests: UniCoT Adapter (from PR1 old UT-11/12/13)

| # | Test | Device | File:Line | What it verifies |
|---|------|--------|-----------|-----------------|
| UT-U1 | UniCoT adapter — train/val split | CPU | [tests/utils/dataset/test_unicot_adapter_on_cpu.py](../../tests/utils/dataset/test_unicot_adapter_on_cpu.py) *(new)* | Correct hold-out ratio by hash (default 90/10) |
| UT-U2 | UniCoT adapter — fail-closed validation | CPU | same file *(new)* | Corrupted rows, empty reflections, continue-with-empty-edit → None (not crash) |
| UT-U3 | UniCoT adapter — SFT→RL conversion | CPU | same file *(new)* | `visual_reflection_to_agentic` preserves all fields from `VisualReflectionTrajectory` |
| UT-U4 | UniCoT adapter — image hash matching | CPU | same file *(new)* | Hash-mismatch detection when images are available locally |

---

## 3. PR2-Specific Tests

### 3.1 CPU Unit Tests

| # | Test | Device | What it verifies |
|---|------|--------|-----------------|
| UT-R1 | `R_format` scorer in isolation | CPU | XML tag structure validation, malformed → zero reward, well-formed → positive |
| UT-R2 | `R_tool` scorer in isolation | CPU | Discrete success/correction rate computation |
| UT-R3 | `R_result` scorer in isolation | CPU | Binary count match logic |
| UT-R4 | `R_pairwise` scorer — shuffle correctness | CPU | Monotonic improvement verified across shuffled turn pairs |
| UT-R5 | Reward weight configuration | CPU | `R_total = (1/|W|) * sum(w_i * R_i)` with various W sets |
| UT-R6 | HTTP scorer protocol — multi-dim serialization | CPU | JSON `[{dimension, score, metadata}]` round-trip |
| UT-R7 | Trajectory resampling — turn count distribution | CPU | Uniform entropy after resampling G by turn count |
| UT-R8 | RPCO stage transition — config validation | CPU | Stage N checkpoint path → Stage N+1 loads correctly |
| UT-R9 | Reflection quality classifier | CPU | Under/good/over-reflection classification from trajectory text |

### 3.2 GPU Smoke Tests

| # | Test | Device | What it verifies |
|---|------|--------|-----------------|
| ST-R1 | Full RPCO training run (Stage 1 → 3) | 4× H800 80GB | Stage 1 (single-image reflection RL) → checkpoint → Stage 3 (multi-task co-optimization) |
| ST-R2 | All 8 reward dimensions compute | 4× H800 80GB | Each scorer produces valid scores on multi-turn trajectory |
| ST-R3 | Multi-scorer concurrency | 4× H800 80GB | Async overlap: LLM judge + VLM judge + rule-based scorers run concurrently |
| ST-R4 | HTTP scorer returns multi-dim rewards | 4× H800 80GB | External endpoint returns `[{dimension, score, metadata}]` array |
| ST-R5 | RPCO stage weight continuity | 4× H800 80GB | Stage N checkpoint loads into Stage N+1 without shape/key mismatches |
| ST-R6 | Resampled trajectory distribution | 4× H800 80GB | G' > G oversampling + uniform turn-count distribution after resampling |

---

## 4. Post-Implementation Evaluation (Non-Blocking)

Per RFC lines 370-371 (PR 2 evaluation plan):

- **UniCoT held-out evaluation:** All 8 reward dimensions against UniCoT references
  - `R_reflect` vs. `eval_summary`
  - `R_pairwise` using UniCoT's `output_image[i]` as improved reference
  - `R_diffusion` vs. final image quality
- **RPCO staging check:** Reflection-quality classifier shifts toward "good" after Stage 1 without regressing `R_plan`
- **Resampling entropy:** Uniform turn-count distribution verified
- **GRPO-trained agent vs. baselines:** Action accuracy, edit similarity, final-image quality vs. UniCoT-7B-MoT and SFT cold-start

---

## 5. Non-Blocking Carry-Over from PR1

| Item | Reason |
|------|--------|
| Co-located vs decoupled GPU pools | Not required for PR1 merge bar; configuration surface expands in PR2 with multi-scorer architecture |
| Production-scale Lance/BAGEL training run | RFC line 329 explicitly non-blocking |
