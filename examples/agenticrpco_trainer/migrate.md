# Source
* `verlomni-pr-fredfork`

# Target
* `verlomni-fredfork-clean`

# Workloads
`verlomni-fredfork-clean` is already at **`7ced948`** (same base). Porting **`7ced948..da00642` plus the working-tree overfit fixes** is what makes `run_agentic_grpo_lora.sh` work for Qwen3-VL multiturn GRPO.

Tip commit at last note: `da00642`. Extra local (uncommitted) deltas that must travel with the port:

| Local delta (on tip) | Why it matters |
| --- | --- |
| Compact Class-1-only overfit fewshot (`OVERFIT_FEWSHOT`) | All-3-class ~25k-char demos → VL mimics `Done.` prose, zero tools → empty `response_mask` crash |
| Dump hermes **before** discard + all-invalid mask restore | Debug dumps stay truthful; verl `rollout_corr` no longer dies when every rollout lacks `generate_image` |
| Force-first `generate_image` curriculum (steps 10→20 anneal) | Bootstraps exploration on weak VL tool-calling without synthetic policy-token credit |
| Task-scoped `good_enough=YES` latch clear | Prevents cross-rollout YES leak that blocked later `generate_image` |

Operator-only (do **not** port): `~/fred/fred_verlomni_agentic_multiturn_pr1.sh`.

### Must port (runtime path)

| File | Action on clean | Role |
| --- | --- | --- |
| `examples/agenticrpco_trainer/agent_llm/run_agentic_grpo_lora.sh` | **add** | Train launcher (GRPO + LoRA `lr=1e-4` + agent loop + reward wiring + `OVERFIT_FEWSHOT` / force-first env) |
| `examples/agenticrpco_trainer/agent_llm/run_qwen_image_tool_server.sh` | **add** | Frozen Qwen-Image via vLLM-Omni (`AGENTIC_VLLM_OMNI_URL`) |
| `examples/agenticrpco_trainer/agent_llm/run_qwen_vl_reflect_server.sh` | **add** | Frozen Qwen3-VL judge via vLLM (`AGENTIC_VLLM_URL`) |
| `examples/agenticrpco_trainer/data_process/create_dummy_agentic_data.py` | **add** | Overfit parquet; Class-1 two-pass same-task fewshot when `OVERFIT_FEWSHOT=1`; system+user only otherwise |
| `examples/agenticrpco_trainer/agent_llm/check_overfit_gates.py` | **add** | Gate sidecar (auto-on if `TOTAL_STEPS≤50`; else optional via `GATE_SIDECAR=0`) |
| `verl_omni/agent_loop/agentic_tool_agent_loop.py` | **add** | Registered loop `agentic_tool_agent` (force-first generate curriculum; optional forced Reflection=debug only) |
| `verl_omni/agent_loop/agentic_metrics_manager.py` | **add** | Rollout dumps + WandB `agentic_reward/*`; dump-before-discard; all-invalid mask harden |
| `verl_omni/agent_loop/agentic_trajectory_context.py` | **add** | Per-rollout image paths / active traj context; task-scoped good_enough YES latch |
| `verl_omni/agent_loop/agentic_image_reflection.py` | **add** | `image_vis=` stats inside `generate_image` obs |
| `verl_omni/agent_loop/diffusion_tool.py` | **replace/extend** | Clean only has `generate_image`; tip adds Omni + `judge_image` + YES block |
| `verl_omni/utils/judge_parse.py` | **add** | Shared VL JSON parse used by judge tool |
| `verl_omni/utils/reward_score/agentic_reward.py` | **replace** | Clean stub (~35 LOC) → full gated `compute_score` (C/A mix-gated until Reflection+Done) |
| `verl_omni/utils/reward_score/vl_reflect_client.py` | **add** | Reward C/A fallback (`AGENTIC_VLLM_URL` / legacy reflect) |
| `verl_omni/agent_loop/__init__.py` | **edit** | Import/register `agentic_tool_agent_loop` |
| `verl_omni/__init__.py` | **edit** | `import verl_omni.agent_loop` so registration runs |

### Optional (not needed for default Omni+vLLM overfit)

| File | Why skip initially |
| --- | --- |
| `…/qwen_image_tool_server.py`, `…/qwen_vl_reflect_server.py` | Legacy FastAPI; launchers use `vllm-omni` / `vllm serve` |
| `…/run_lance_frozen_diffusion_tool_server.sh` | Lance path; train prefers Omni URL |
| `…/data_process/create_data.sh` | Thin wrapper; run script invokes the Python directly |
| `examples/agenticrpco_trainer/README.md` | Docs only |
| `tests/…` | Validation, not launch |
| Deletes under `tests/special_e2e/*`, `docs/algo/agentic_rpco.md`, `agentic_trajectory.py` | Cleanup of old Lance/toy path; run does not need them |

### Clean baseline note

| Already on clean at `7ced948` | Implication |
| --- | --- |
| Thin `agentic_reward.py`, Lance-era `agentic_trajectory.py`, basic `diffusion_tool.py` | Must be overwritten / superseded by tip versions; do not try to merge piecemeal into the stub reward |

**Bottom line:** ~15 files (13 adds + 2 small `__init__` edits + replace `diffusion_tool` / `agentic_reward`) are enough for a functioning `run_agentic_grpo_lora.sh` on clean; FastAPI/Lance/tests/docs can wait. Port the tip **and** the uncommitted overfit/force-first/mask-harden deltas above, or VL overfit will still crash / never call tools.
