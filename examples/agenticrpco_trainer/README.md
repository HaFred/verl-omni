# Agentic Mode (2a) Trainer

Last updated: 08/03/2026

Training recipes for **Mode (2a) agentic RL** in [#302](https://github.com/verl-project/verl-omni/issues/302). The actor is a standard HF language model; the diffusion generator is an external tool and is never part of the actor optimizer. **Diffusion remains frozen** in the Mode (2a) sense: GRPO updates only the agent LLM; `ToolAgentLoop` dispatches `generate_image` outside the actor FSDP graph.

## Lance-3B Agentic GRPO

Launch script (FlowGRPO-style: stock `ppo_trainer` + CLI overrides, no custom recipe YAML): [`lance/run_lance_agentic_grpo.sh`](lance/run_lance_agentic_grpo.sh). Shared overrides: [`lance/agentic_grpo_overrides.sh`](lance/agentic_grpo_overrides.sh).

### Scope (PR1)

- **ST-1** is an **infra smoke**: 1-step GRPO completes with finite non-zero `actor/loss` and a saved checkpoint. Without `AGENTIC_DIFFUSION_TOOL_URL`, the function tool returns a text-only stub; that is not a claim that real diffusion tooling works.
- Multi-turn orchestration uses verl's stock `ToolAgentLoop`: all assistant turns receive `response_mask=1`, while tool observations receive `0`.
- **AC2** (Mode 2a tool boundary) and **AC3** (FlowGRPO compat) are covered by `tests/agent_loop/test_agentic_compat.py`, not the GPU shell.
- The recipe uses upstream verl's `HFModelConfig`, language-model FSDP engine, vLLM rollout, `ToolAgentLoop`, and function-tool configuration. Existing V1 Omni model, trainer, engine, and rollout code is unchanged.

### Prerequisites

- Launch from the **verl-omni repo root** (or otherwise keep CWD such that the recipe's repo-relative `function_tool_path` resolves). Keep that path relative — absolute paths are brittle across machines and users.
- Set **`MODEL_PATH`** to a local `Lance_3B_hf_und` export (see below). That directory is **not** published on Hugging Face; it is curated from the hub `Lance_3B` MoT tree. Do **not** point `MODEL_PATH` at the hub snapshot root or at raw `Lance_3B` (no HF `config.json` / usable `chat_template` for stock `main_ppo`).

### Make `Lance_3B_hf_und`

[bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) ships MoT layout under `Lance_3B/` (`llm_config.json` + `language_model.*` / `*_moe_gen` weights). Stock verl FSDP + vLLM need a standard HF CausalLM directory. Curate one with [`tests/special_e2e/prepare_lance_hf_und.py`](../../tests/special_e2e/prepare_lance_hf_und.py):

1. Remap understanding-path tensors (`language_model.*`), drop `*_moe_gen` / connectors / VAE adapters.
2. Write Qwen2 `config.json` + remapped `model.safetensors`.
3. Copy tokenizer files and inject a Qwen2-style `chat_template` (required by RLHFDataset).

```bash
# After downloading the hub repo (or using a local HF cache snapshot):
#   https://huggingface.co/bytedance-research/Lance
LANCE_ROOT=/path/to/bytedance-research/Lance   # contains Lance_3B/
python3 tests/special_e2e/prepare_lance_hf_und.py \
  --src "${LANCE_ROOT}/Lance_3B" \
  --dst "${LANCE_ROOT}/Lance_3B_hf_und"
export MODEL_PATH="${LANCE_ROOT}/Lance_3B_hf_und"
```

`MODEL_PATH` must be the `Lance_3B_hf_und` directory itself (it must contain `config.json` and `tokenizer.json`), not the parent hub snapshot.

### Toy data (acceptance smoke)

```bash
python3 tests/special_e2e/create_dummy_agentic_data.py \
    --local_save_dir ~/data/agentic \
    --train_size 8 \
    --val_size 4
```

### GPU merge-gate smoke (ST-1)

```bash
MODEL_PATH=/path/to/Lance_3B_hf_und \
  bash tests/special_e2e/run_agentic_grpo_lance.sh
```

AC2 / AC3 (Mode 2a tool boundary + FlowGRPO backward compat) are CPU unit tests:

```bash
pytest tests/agent_loop/test_agentic_compat.py
```

Or launch training directly (from repo root):

```bash
MODEL_PATH=/path/to/Lance_3B_hf_und \
  bash examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh
```

## File map

```
examples/agenticrpco_trainer/
├── lance/
│   ├── agentic_grpo_overrides.sh
│   └── run_lance_agentic_grpo.sh
└── README.md

verl_omni/agent_loop/
├── agentic_trajectory.py
└── diffusion_tool.py
```

Optional operator hook (implemented in `verl_omni/agent_loop/diffusion_tool.py`, **not** exercised by UT/ST): set `AGENTIC_DIFFUSION_TOOL_URL` to a POST endpoint that accepts `{"prompt": "..."}` and may return JSON with `image_base64` / `images_base64`, `text`, and `reward`. UT (`test_diffusion_tool_stub_without_endpoint`) and ST-1 only cover the unset-URL text stub.
