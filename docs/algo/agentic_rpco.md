(agentic_rpco)=
# Multi-Turn Agentic Reflection–Plan Co-Optimization (RPCO)

Last updated: 08/05/2026

This note records what landed and was verified in **PR1** for Mode (2a) agentic
GRPO on Lance-3B understanding (`Lance_3B_hf_und`), with frozen diffusion as an
external tool. Full RPCO / multi-step Lance e2e / overfit diagnostics live on
the **PR2 working branch** (`verlomni-fredfork`), not in this merge gate.

## Goal

Prove the Mode (2a) infra boundary:

1. Train **only** the agent LLM with stock verl GRPO (`main_ppo` + vLLM).
2. Call frozen diffusion through a **function tool** outside the actor optimizer.
3. Keep existing single-turn FlowGRPO paths untouched.

PR1 (#329) does **not** claim Strong-Reflection convergence, voluntary Hermes mastery,
or real-MoT image quality learning.

## Current Design

| Area | Location | Role |
| --- | --- | --- |
| GPU merge gate (ST-1) | `tests/special_e2e/run_agentic_grpo_lance.sh` | 1-step Lance agentic GRPO smoke |
| Toy Hermes data | `tests/special_e2e/create_dummy_agentic_data.py` | Few-shot `<tool_call>` + reflection seed |
| HF und export | `tests/special_e2e/prepare_lance_hf_und.py` | MoT → HF CausalLM und checkpoint |
| Tool chat template | `tests/special_e2e/qwen2_tool_chat_template.yaml` | YAML-packaged Jinja for Hermes tools |
| Frozen tool stub/HTTP | `verl_omni/agent_loop/diffusion_tool.py` | `generate_image` function tool |
| Scalar reward | `verl_omni/utils/reward_score/agentic_reward.py` | Mode (2a) `compute_score`; ST-1 uses `compute_score_smoke` |

| Worker patches | `verl_omni/agent_loop/agentic_worker_patch.py` | Traj kwargs + reward metric logging |
| Trajectory helpers | `agentic_trajectory.py`, `trajectory_context.py` | Metadata / artifact context |
| CPU AC2 / AC3 | `tests/agent_loop/test_agentic_compat.py` | Tool boundary + FlowGRPO compat |

## How to run

```bash
# Operator env (CUDA / Ray / MODEL_PATH), then from repo root:
MODEL_PATH=/path/to/Lance_3B_hf_und \
  bash tests/special_e2e/run_agentic_grpo_lance.sh

# CPU ACs
pytest tests/agent_loop/test_agentic_compat.py \
       tests/special_e2e/test_create_dummy_agentic_data.py
```

Prepare und export if needed:

```bash
python3 tests/special_e2e/prepare_lance_hf_und.py \
  --src /path/to/Lance_3B \
  --dst /path/to/Lance_3B_hf_und
```
