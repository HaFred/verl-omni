# Agentic Mode (2a) trainer recipes

Last updated: 08/03/2026

Training recipes for **Mode (2a) agentic RL** in [#302](https://github.com/verl-project/verl-omni/issues/302). The actor is a standard HF language model; the diffusion generator is an external tool and is never part of the actor optimizer. **Diffusion remains frozen** in the Mode (2a) sense: GRPO updates only the agent LLM; `ToolAgentLoop` dispatches `generate_image` outside the actor FSDP graph.

## Lance-3B Agentic GRPO

Config: [`lance/config/lance_agentic_grpo.yaml`](lance/config/lance_agentic_grpo.yaml)

### Scope (PR1)

- **ST-1** is an **infra smoke**: 1-step GRPO completes with finite `actor/loss`. Without `AGENTIC_DIFFUSION_TOOL_URL`, the function tool returns a text-only stub; that is not a claim that real diffusion tooling works.
- Multi-turn orchestration uses verl's stock `ToolAgentLoop`: all assistant turns receive `response_mask=1`, while tool observations receive `0`.
- **ST-2** verifies the actor checkpoint/loss and that diffusion stays outside the actor optimizer (**Diffusion remains frozen**, Mode (2a) / ToolAgentLoop external-tool boundary). It does not assert selective MoT freeze inside a shared actor checkpoint.
- The recipe uses upstream verl's `HFModelConfig`, language-model FSDP engine, vLLM rollout, `ToolAgentLoop`, and function-tool configuration. Existing V1 Omni model, trainer, engine, and rollout code is unchanged.

### Prerequisites

- Launch from the **verl-omni repo root** (or otherwise keep CWD such that the recipe's repo-relative `function_tool_path` resolves). Keep that path relative in config — absolute paths are brittle across machines and users.
- Set **`MODEL_PATH`** to a prepared HF understanding export (e.g. from `tests/special_e2e/prepare_lance_hf_und.py`). The GPU smoke script does **not** ship a machine-local default snapshot path; unset `MODEL_PATH` will not find a usable checkpoint.
- Do **not** point `MODEL_PATH` at raw `Lance_3B` (no `chat_template` → empty filtered dataset).

### Toy data (acceptance smoke)

```bash
python3 tests/special_e2e/create_dummy_agentic_data.py \
    --local_save_dir ~/data/agentic \
    --train_size 8 \
    --val_size 4
```

### GPU merge-gate smoke (ST-1 / ST-2 / ST-3)

```bash
MODEL_PATH=/path/to/Lance_3B_hf_und \
  bash tests/special_e2e/run_agentic_grpo_lance.sh
```

Or launch training directly (from repo root):

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=examples/agenticrpco_trainer/lance/config \
  --config-name=lance_agentic_grpo \
  data.train_files=~/data/agentic/train.parquet \
  data.val_files=~/data/agentic/val.parquet \
  actor_rollout_ref.model.path=/path/to/Lance_3B_hf_und
```

## File map

```
examples/agenticrpco_trainer/
├── lance/
│   └── config/
│       └── lance_agentic_grpo.yaml
└── README.md

verl_omni/agent_loop/
├── agentic_trajectory.py
└── diffusion_tool.py
```

Optional operator hook (implemented in `verl_omni/agent_loop/diffusion_tool.py`, **not** exercised by UT/ST): set `AGENTIC_DIFFUSION_TOOL_URL` to a POST endpoint that accepts `{"prompt": "..."}` and may return JSON with `image_base64` / `images_base64`, `text`, and `reward`. UT (`test_diffusion_tool_stub_without_endpoint`) and ST-1 only cover the unset-URL text stub.
