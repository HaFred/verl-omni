# Agentic RPCO Trainer — Multi-Turn Agentic RL for Omni-Modal Generation

This directory contains training recipes for **Mode (2a) agentic RL** as proposed
in the [Agentic RL RFC](../../docs/agent/verl-omni-rfc-agentic-rl_v1.md).

## Architecture

```
Agent LLM (trainable)          Frozen Diffusion Tool
─────────────────────          ─────────────────────
Lance-3B understanding path    Lance-3B generation path
(LLM_UND: mlp, self_attn,     (LLM_GEN: mlp_moe_gen,
 lm_head, ViT)                 self_attn.*_moe_gen)

     │                              │
     │  Turn 0: reasoning + prompt  │
     ├──────────────────────────────►  generate image
     │                              │
     │  Turn 1: reflection +        │
     │          REWRITTEN prompt    │
     ├──────────────────────────────►  generate image
     │                              │
     │  Turn N: decision = "stop"   │
     ◄──────────────────────────────  trajectory complete
```

The agent learns to **rewrite prompts** between turns based on visual feedback.
The diffusion model is a frozen tool — only the understanding path receives
gradients via token-level GRPO.

## Recipes

| Recipe | Model | Algorithm | What it trains |
|--------|-------|-----------|----------------|
| [Lance-3B Agentic GRPO](lance/) | `bytedance-research/Lance` | GRPO (token-level) | Understanding path (LoRA r=64) |

## Prerequisites

Follow the [installation guide](../../docs/start/install.md) for the base
environment. Minimum stack:

| Component | Version |
|-----------|---------|
| vLLM | >= 0.22.0 |
| vLLM-Omni | >= 0.22.0 (Lance support since PR #3710) |
| transformers | >= 5.x |
| torch | >= 2.11.0 |
| verl | >= 0.9.0 |

```bash
# vLLM + vLLM-Omni
pip install vllm>=0.22.0
pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.22.0"

# verl-omni (this repo)
pip install -e ".[train,dev]"
```

## Prepare data

### UniCoT-Self-Reflection-6K (recommended)

Download and preprocess the UniCoT dataset from HuggingFace.  The script applies
fail-closed validation per RFC S7 and writes verl-compatible train/val parquet splits.

```bash
pip install datasets pandas pyarrow

python examples/agenticrpco_trainer/data_process/unicot.py \
    --local_save_dir ~/data/agentic \
    --eval_ratio 0.1
```

Output: `~/data/agentic/{train,val}.parquet` with ~5,400 train / ~600 val
prompts (90/10 split by `data_id` hash).  Each row contains `raw_prompt`,
`data_source`, `reward_model` (with ground-truth reflection steps), and
provenance `extra_info`.

### Toy smoke test (no download)

For a one-step acceptance check that doesn't require the 6K dataset:

```bash
python3 tests/special_e2e/create_dummy_agentic_data.py \
    --local_save_dir ~/data/agentic \
    --train_size 8 \
    --val_size 4
```

## Quick start — Lance-3B

```bash
# Launch training (defaults to ~/data/agentic/{train,val}.parquet)
bash examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh
```

For a one-step acceptance check, override runtime size (example):

```bash
TRAIN_FILE=~/data/agentic/train.parquet \
VAL_FILE=~/data/agentic/val.parquet \
bash examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=2 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.logger=console
```

CLI overrides:
```bash
MODEL_PATH=/local/Lance-3B \
bash examples/agenticrpco_trainer/lance/run_lance_agentic_grpo.sh \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.lr=2e-7
```

## Key signals

- `actor/loss` stable, no OOM, grad_norm finite
- `training/rollout_actor_probs_pearson_corr` > 0.95 after weight sync
- `actor/perf/max_memory_allocated_gb` < 65
- Validation reward increasing over steps
- Prompt rewriting captured in trajectory: `turn[i].tool_call.params["prompt"]` differs from `turn[i+1].tool_call.params["prompt"]`

## SFT → RL pipeline

Agentic RL training works best with **SFT cold-start** — pre-train the agent on
multi-turn reflection trajectories before RL. Use the UniCoT-Self-Reflection-6K
dataset for SFT (#295), then use this recipe for RL fine-tuning.

## File map

```
examples/agenticrpco_trainer/
├── lance/
│   ├── run_lance_agentic_grpo.sh     ← launch script (volatile overrides only)
│   └── config/
│       └── lance_agentic_grpo.yaml   ← recipe config (inherits verl ppo_trainer)
├── data_process/
│   └── unicot.py                     ← UniCoT-Self-Reflection-6K preprocessor
└── README.md                         ← (this file)
```

## What's next (PR 2)

PR 2 adds multi-dimensional rewards (plan, reflection, format, tool, result,
pointwise, pairwise, diffusion) and RPCO staged training (decouple-then-fuse
strategy for reflection-plan asymmetry).
