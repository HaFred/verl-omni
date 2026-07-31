# Agentic Mode (2a) trainer recipes

Training recipes for **Mode (2a) agentic RL** (train agent LLM, freeze diffusion
tool) as proposed in the Agentic RL RFC.

## Lance-3B Agentic GRPO

Config: [`lance/config/lance_agentic_grpo.yaml`](lance/config/lance_agentic_grpo.yaml)

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
  actor_rollout_ref.model.path=/path/to/Lance_3B_hf_und \
  ++actor_rollout_ref.model.architecture=Qwen2ForCausalLM
```

## File map

```
examples/agenticrpco_trainer/
├── lance/
│   └── config/
│       └── lance_agentic_grpo.yaml
└── README.md
```
