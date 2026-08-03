# Agentic Mode (2a) trainer recipes

Training recipes for **Mode (2a) agentic RL**. The actor is a standard HF
language model; the diffusion generator is an external tool and is never part
of the actor optimizer.

## Lance-3B Agentic GRPO

Config: [`lance/config/lance_agentic_grpo.yaml`](lance/config/lance_agentic_grpo.yaml)

### Scope (PR1)

- **ST-1** is an **infra smoke**: 1-step GRPO completes with finite `actor/loss`.
  Und-only Lance export may use a **stub tool image** (marked `tool_stubbed`);
  that is not a claim that real diffusion tooling works.
- **Online train path** masks turn ≥1 agent tokens (`response_mask=0`) until
  full chat-template retokenize (train↔rollout parity) lands in PR2.
- **ST-2** verifies the actor checkpoint/loss and guards against routing through
  the removed custom agentic/Omni worker stack.
- The recipe uses upstream verl's `HFModelConfig`, language-model FSDP engine,
  vLLM rollout, and agent-loop configuration mechanism. Existing V1 Omni model,
  trainer, and rollout code is unchanged.

### Toy data (acceptance smoke)

```bash
python3 tests/special_e2e/create_dummy_agentic_data.py \
    --local_save_dir ~/data/agentic \
    --train_size 8 \
    --val_size 4
```

### GPU merge-gate smoke (ST-1 / ST-2 / ST-3)

```bash
bash tests/special_e2e/run_agentic_grpo_lance.sh
```

Or launch training directly:

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
└── agent_loop.yaml
```
