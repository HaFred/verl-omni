# Agentic GRPO overfit — hyperparameter tune list

Source of truth for knobs: `~/fred/fred_verlomni_agentic_multiturn_pr1.sh` (operator)
and defaults in `agent_llm/run_agentic_grpo_lora.sh` / `run_qwen_image_tool_server.sh`.

Evidence baseline (Qwen3-VL-2B Instruct, run `…_20260810_084206`): rollout protocol is
healthy — gen→judge→forced Reflection→rewrite (or max-pass Done), compact
`judge_image` args, rollout-scoped PNGs. Tune image quality / YES bar / pass cap
next; do **not** re-break force-reflection or fewshot Done endings.

### Current operator defaults (2026-08-10)

| Knob | Current | Notes |
| --- | --- | --- |
| `AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD` | **0.90** | YES iff `C≥thr` **and** `A≥thr` (ignore model flag) |
| `AGENTIC_REFLECT_GOOD_ENOUGH` | **0.90** | Keep locked to judge thr |
| `QWEN_IMAGE_STEPS` | **16** | Restart image sidecar after change |
| `QWEN_IMAGE_TRUE_CFG_SCALE` | **4.0** | |
| `AGENTIC_FORCE_REFLECTION_AFTER_JUDGE` | **1** | Inject Reflection after every successful judge (`mask=0`) |
| `AGENTIC_MAX_GENERATE_IMAGE_PASSES` | **3** (operator; launch fallback **5**) | Soft-stop Done + block further gen |
| `AGENTIC_BLOCK_GENERATE_AFTER_YES` | **1** | |
| `AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES` | **1** | |
| `OVERFIT_FEWSHOT` | **1** | Soldier Class-1 demo; **omit terminal Done** |

Requires: re-`source` operator env → **restart Qwen-Image sidecar** when changing
`QWEN_IMAGE_*` → restart train. Judge thr is read per request on the train/tool path.

Tune **one axis at a time** after the protocol is stable.

## Priority A — closed-loop reward / stop–continue pressure

| Knob | Where | Current | Suggested sweep | Effect | Watch |
| --- | --- | --- | --- | --- | --- |
| `AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD` | operator + Ray env | **0.90** | **0.90 → 0.85 → 0.80** | YES iff `C≥thr` **and** `A≥thr` (`judge_parse.py`). Lower → more YES → earlier natural `Done.` → full C/A mix | YES rate; `reward_done`; `protocol_ok`; `critic/score/mean`; keep some NO for multiturn ΔC |
| `AGENTIC_REFLECT_GOOD_ENOUGH` | operator (keep = judge thr) | **0.90** | **same as judge thr** | Legacy/FastAPI path; avoid split verdicts | same as above |
| `AGENTIC_MAX_GENERATE_IMAGE_PASSES` | operator (override launch) | **3** | **2 / 3 / 4 / 5** | Caps successful gens; then forced max-pass `Done.` + block further `generate_image` | `nturns`; rewrite rate; wall-clock / step; traj `forced_reflection_max_passes_done` |
| `AGENTIC_BLOCK_GENERATE_AFTER_YES` | operator | 1 | keep **1** | Hard-stops rewrite roulette after YES | `rewrite_after_yes` ≈ 0 |
| `AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES` | operator | 1 | keep **1** | Hard-stops gen past the pass cap (env dynamics) | no 4th+ live PNG after cap |

Do **not** jump thr to ≤0.50 — first-pass YES collapses multiturn / ΔC signal.

## Priority B — frozen image quality vs throughput

Sidecar must be restarted after changing these (`run_qwen_image_tool_server.sh` reads at launch).

| Knob | Where | Current | Suggested sweep | Effect | Watch |
| --- | --- | --- | --- | --- | --- |
| `QWEN_IMAGE_STEPS` | operator → Omni sidecar | **16** | **8 → 12 → 16 → 20** | Denoise steps. Low = fast queue, often softer / lower aesthetics → harder to hit YES | step time; `reward_aesthetics`; YES rate; rollout PNGs |
| `QWEN_IMAGE_TRUE_CFG_SCALE` | operator → Omni sidecar | **4.0** | **3.0 → 4.0 → 5.0** | Prompt adherence vs over-saturation / artifacts | C vs A balance; visual fidelity |
| `QWEN_IMAGE_WIDTH` / `HEIGHT` | operator | 512² | keep 512 for overfit | Resolution vs VRAM / latency | OOM; queue depth |
| `QWEN_IMAGE_SEED` | run script | 42 | fixed **42** for overfit; unset to randomize | Visual stability across GRPO group | image diversity within group |

**Coupling note:** thr 0.90 + STEPS 16 often leaves A≈0.84–0.88 → NO → rewrite until max passes.
If score stays open-loop gated, lower thr or raise STEPS; if too many first-pass YES, raise thr.

## Priority C — exploration / curriculum (already on)

| Knob | Where | Current | Suggested sweep | Effect | Watch |
| --- | --- | --- | --- | --- | --- |
| `OVERFIT_FEWSHOT` | operator | 1 (Class-1 same-task, **no terminal Done**) | 0 / 1 | Bake tool protocol on soldier rows only; epic = system+user | first-turn tool rate; never Done-without-tools |
| `AGENTIC_FORCE_FIRST_GENERATE` | operator | 1 | 0 / 1 | Teacher-force Hermes gen (+ compact judge if missing) with `mask=1` | early `tool_call`/`valid`; voluntary hermes after anneal |
| `AGENTIC_FORCE_FIRST_WARMUP_STEPS` | operator | 10 | 5 / 10 / 15 | `p=1` through this global step | force rate early |
| `AGENTIC_FORCE_FIRST_END_STEP` | operator | 20 | 15 / 20 / 30 | Linear decay to `p=0` | voluntary tool rate after end |
| `AGENTIC_FORCE_REFLECTION_AFTER_JUDGE` | operator | **1** | keep **1** for overfit; 0 = policy emits Reflection | Injects Reflection after every successful judge (`mask=0`); reward strips force markers | traj `agentic_forced_reflection=1`; `agent_rewrite_after_forced_reflection` |

## Priority D — GRPO / LoRA optim (secondary once score moves)

| Knob | Where | Current | Suggested sweep | Effect | Watch |
| --- | --- | --- | --- | --- | --- |
| `ACTOR_LR` | run script | 1e-4 | 5e-5 / 1e-4 / 2e-4 | LoRA step size (verl default 1e-6 is too small) | score slope; entropy collapse |
| `PPO_EPOCHS` | run script | 2 | 1 / 2 / 4 | Reuses each batch | KL / entropy; overfitting speed |
| `ROLLOUT_TEMPERATURE` | run script | 0.8 | 0.7 / 0.8 / 1.0 | Group diversity for GRPO advantages | within-group score std; tool-format junk |
| `actor_rollout_ref.actor.entropy_coeff` | run script (fixed) | 0.001 | 0 / 0.001 / 0.01 | Exploration regularizer | entropy; format stability |
| `ROLLOUT_N` | run script / operator | 8 | 4 / 8 | GRPO group size | advantage SNR vs wall-clock |

## Suggested experiment order

1. Keep protocol fixed: force-reflection **on**, compact judge args, omit fewshot Done, block after YES/max-pass.
2. Tune **`AGENTIC_MAX_GENERATE_IMAGE_PASSES`** (2–5) for wall-clock vs rewrite depth.
3. If YES rare at thr 0.90: **STEPS=20** or thr **0.85 / 0.80**.
4. If too many first-pass YES / flat multiturn: raise thr toward **0.90**.
5. If score rises but groups have no variance: raise **temperature** or **ROLLOUT_N**.
6. If tools regress after force-first anneal: extend `FORCE_FIRST_END_STEP` (e.g. 30).

## Quick reference — edit sites

| Knob | File |
| --- | --- |
| good_enough thr / force / pass cap | `~/fred/fred_verlomni_agentic_multiturn_pr1.sh` |
| image steps / CFG | same operator → restart Omni sidecar |
| force-first curriculum | same operator |
| launcher defaults / LoRA / Ray env | `examples/agenticrpco_trainer/agent_llm/run_agentic_grpo_lora.sh` |
| Qwen3.5 parser + GDN | `data/qwen35_env.sh` (sourced by launcher) |
| YES rule | `verl_omni/utils/judge_parse.py` (`correctness >= thr and aesthetics >= thr`) |
| image server defaults | `examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh` |
