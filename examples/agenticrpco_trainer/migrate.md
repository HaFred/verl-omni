# Source
* `verlomni-pr-fredfork`

# Target
* `verlomni-fredfork-clean`

# Workloads
`verlomni-fredfork-clean` is already at **`7ced948`** (same base). Port the
**entire ordered commit range after that base through the current source
HEAD**:

```text
7ced948..<latest-source-commit>
```

This means every source commit after `7ced948`, not only `95c7c3c`.
Commit `95c7c3c` is the current minimum runtime checkpoint containing the
policy-sampled Done fix, but it depends on earlier commits in the range.
Any newer commits must be included as well.

For a cherry-pick migration, enumerate the range oldest-first:

```bash
git rev-list --reverse 7ced948..<source-branch> | xargs git cherry-pick
```

Operator-only (do **not** port): `~/fred/fred_verlomni_agentic_multiturn_pr1.sh`.

### Protocol that must land on clean

| Behavior | Why |
| --- | --- |
| Rollout-scoped YES latch (`_rollout_scope_key`) | Cross-rollout YES leak blocked later `generate_image` and inflated old-run scores |
| Forced Reflection after judge is **mask=0 context only** | Injected `Done.` was stripped from reward → `reward_done=0` forever |
| YES / max-pass → stop cue + one policy-sampled `Done.` (`mask=1`) | GRPO needs a real terminal action to reinforce |
| Done credit = successful PNG + successful judge + policy terminal | Blocks planning prose (“Stop when Done.”), blocked gens, no-PNG stubs |
| YES bar **0.86** + `QWEN_IMAGE_DIVERSIFY_SEED=1` | ~half first-pass YES at STEPS=16; within-group reward variance for GRPO |

### Key runtime files to verify after porting the full range

This inventory is a verification checklist; it is **not** a substitute for
porting every commit in `7ced948..<source-branch>`.

| File | Action on clean | Role |
| --- | --- | --- |
| `examples/agenticrpco_trainer/agent_llm/run_agentic_grpo_lora.sh` | **add** | Train launcher (GRPO + LoRA `lr=1e-4` + agent loop + reward + force-first / thr / seed diversify env) |
| `examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh` | **add** | Frozen Qwen-Image via vLLM-Omni (`AGENTIC_VLLM_OMNI_URL`) |
| `examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh` | **add** | Frozen Qwen3-VL judge via vLLM (`AGENTIC_VLLM_URL`); optional judge-log middleware |
| `examples/agenticrpco_trainer/agent_llm/qwen_vl_judge_log_middleware.py` | **add** | Optional vLLM middleware for judge score logging (used by reflect launcher) |
| `examples/agenticrpco_trainer/data_process/create_dummy_agentic_data.py` | **add** | Overfit parquet; Class-1 two-pass same-task fewshot when `OVERFIT_FEWSHOT=1` (omit terminal `Done.`) |
| `examples/agenticrpco_trainer/agent_llm/check_overfit_gates.py` | **add** | Gate sidecar (auto-on if `TOTAL_STEPS≤50`) |
| `verl_omni/agent_loop/agentic_tool_agent_loop.py` | **add** | `agentic_tool_agent`: force-first Hermes; forced Reflection stop cue → policy Done |
| `verl_omni/agent_loop/agentic_metrics_manager.py` | **add** | Traj dumps + WandB `agentic_reward/*`; turn_kinds for stop cue / policy Done |
| `verl_omni/agent_loop/agentic_trajectory_context.py` | **add** | Rollout-scoped image paths + YES latch (`_rollout_scope_key`) |
| `verl_omni/agent_loop/agentic_image_reflection.py` | **add** | `image_vis=` stats inside `generate_image` obs |
| `verl_omni/agent_loop/diffusion_tool.py` | **replace/extend** | Omni + `judge_image` + YES/max-pass block + per-rollout seed diversify |
| `verl_omni/utils/judge_parse.py` | **add** | Shared VL JSON parse; YES iff `C≥thr` and `A≥thr` |
| `verl_omni/utils/reward_score/agentic_reward.py` | **replace** | Gated `compute_score`; policy-terminal Done credit; no credit for forced/blocked |
| `verl_omni/utils/reward_score/vl_reflect_client.py` | **add** | Reward C/A fallback (`AGENTIC_VLLM_URL` / legacy reflect) |
| `verl_omni/agent_loop/__init__.py` | **edit** | Import/register `agentic_tool_agent_loop` |
| `verl_omni/__init__.py` | **edit** | `import verl_omni.agent_loop` so registration runs |

### Files not required at launch time

These files are not required to start the default Omni+vLLM workload, but they
still travel with the full commit-range migration when changed by that range.

| File | Why skip initially |
| --- | --- |
| `…/qwen_image_tool_server.py`, `…/qwen_vl_reflect_server.py` | Legacy FastAPI; launchers use `vllm-omni` / `vllm serve` |
| `…/run_lance_frozen_diffusion_tool_server.sh` | Lance path; train prefers Omni URL |
| `…/data_process/create_data.sh` | Thin wrapper; run script invokes the Python directly |
| `examples/agenticrpco_trainer/README.md`, `hyperparam_tune_list.md`, `migrate.md` | Docs only; still port their commits |
| `tests/…` | Validation, not launch |

### Clean baseline note

| Already on clean at `7ced948` | Implication |
| --- | --- |
| Thin `agentic_reward.py`, Lance-era `agentic_trajectory.py`, basic `diffusion_tool.py` | Must be superseded by source-HEAD versions; do not merge piecemeal into the stub reward |

**Bottom line:** migrate the complete ordered history from immediately after
`7ced948` through the latest source HEAD. Do not cherry-pick `95c7c3c` alone;
the file list above only verifies that the critical runtime pieces arrived.
