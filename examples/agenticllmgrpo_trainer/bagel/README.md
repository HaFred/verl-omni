# Bagel Co-RL (RFC 453) — GPU run guide for agents

Multi-turn agentic Co-RL on one Bagel-7B-MoT actor: the UND text path is the
Hermes tool-calling agent, the GEN path (`*_moe_gen`) is its own `generate_image`
tool, both trained in one composite optimizer step (UND token GRPO + GEN
FlowGRPO). Design: RFC [#453](https://github.com/verl-project/verl-omni/issues/453)
(v3 refactor + implementation plan on the fork under `outputs/`).

**Fail-closed rules — do not violate:**

- UND must be the published Bagel checkpoint emitting Hermes `<tool_call>`. If
  dual-role serving cannot be proven, STOP — never fall back to a Qwen actor or a
  Qwen image sidecar.
- Never edit Mode (2a) `examples/agenticllmgrpo_trainer/agent_llm/run_agenticrpco_grpo_lora.sh`.
- No mid-episode weight sync, no update before the N-sibling gather completes.

## Glossary (do not conflate)

| Symbol | Meaning |
| --- | --- |
| `J` | UND policy turns inside one episode (`J >= K`) |
| `K` | `generate_image` calls inside the same episode (spike default caps `K` at 1) |
| `S` | FlowGRPO seeds per `generate_image` call (`agent.gen_samples_per_call`, must be >= 2) |
| `N` | Sibling episodes per dataset task for token GRPO (`rollout.n`) |

Episode patterns the stack must accept: paired (`J == K`), mixed (`J > K >= 1`,
reflection-only turns), gen-off (`K = 0`, UND-only). Metric glossary:
`und/no_image_credit` counts `K = 0` episodes only; `gen/dropped_incomplete_groups`
counts calls whose valid seeds != `S`.

## 0. Environment

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
pre-commit install
```

Data: the recipe builds UniCoT train/val parquet automatically
(`REBUILD_UNICOT=1` forces a rebuild). The parquet must carry
`extra_info.reference_image_path` (UniCoT refs for the RM); re-stamp with
`stamp_unicot_reference_paths.py` when the reference set changes. Weight knobs are
build-time CLI flags now: `--w_reflect/--w_plan/--w_format/--w_tool/--w_result`
(override the legacy `RPCO_W_*` env vars).

## 1. L1 test suite (run first — cheap, catches most breakage)

```bash
python -m pytest tests/agent_loop/test_bagel_corl_on_cpu.py \
  tests/agent_loop/test_bagel_corl_dual_lane_tq_on_cpu.py \
  tests/agent_loop/test_bagel_corl_rm_on_cpu.py \
  tests/agent_loop/test_tool_agent_loop_on_cpu.py \
  tests/tools/test_agentic_tools_on_cpu.py \
  tests/tools/test_agentic_image_gen_hydra_env_on_cpu.py \
  tests/tools/test_bind_order_on_cpu.py \
  tests/utils/reward_score/test_bagel_rm_image_scorer_on_cpu.py \
  tests/utils/reward_score/test_agentic_reward_on_cpu.py \
  tests/utils/reward_score/test_agentic_multidim_reward_on_cpu.py \
  tests/utils/dataset/test_build_unicot_agentic_rl_on_cpu.py \
  tests/workers/rollout/test_bagel_dual_role_llm_server_on_cpu.py \
  tests/trainer/omni/test_bagel_corl_trainer_on_cpu.py \
  tests/pipelines/test_bagel_corl_model_on_cpu.py \
  tests/utils/test_metrics_utils_on_cpu.py -q
```

On a GPU box every file above must import and pass (the dev laptop can only run
the torch-free subset — 37 passed there; torch-bound files must be green here).
Record the exact command + `N passed` for the PR body. If
`test_normalize_tq_kv_get_result_with_non_tensor_stack` fails, the tensordict pin
moved its API — report the traceback, do not "fix" the normalizer to match a
guess.

## 2. GPU unit smoke (2 GPUs)

```bash
bash tests/gpu_smoke/run_gpu_smoke_core.sh
```

Entry 9 ("bagel corl tiny composite") is the Co-RL gate: builds a tiny random
Bagel checkpoint, checks UND log-probs, dual-LoRA param groups (UND group without
`lr`, GEN group with `lr=3e-5`), and the UND backward path on CUDA. Entry 7
covers the Mode-2a tool loop and must keep passing. All 9 entries green → proceed.

## 3. M0 spike — prove dual-role serving (the go/no-go gate)

The trainer refuses to start Co-RL training without
`agent.und_ar_serving_ready=True` (fail-closed; no Qwen fallback).

```bash
# Terminal A: standalone UND AR server (bagel_think deploy, Hermes-capable)
bash examples/agenticllmgrpo_trainer/bagel/run_bagel_und_ar_serve.sh
# Terminal B: Hermes tool-call probe against the UND replica
BAGEL_UND_URL=http://127.0.0.1:8094 \
  python3 examples/agenticllmgrpo_trainer/bagel/spike_und_hermes.py \
    --model-path "$BAGEL_MODEL_PATH"
# No server? Schema-only sanity: add --offline-schema.
```

Pass criteria: UND decodes a Hermes `generate_image` tool call (and the GEN
replica denoises with FlowGRPO traj stash — latents/timesteps/logprobs all
present). Record the decoded tool call. If the base model's tool-calling is too
weak, report it — the fix is an SFT pass first (upstream #333 primitives), not a
loosened parser.

## 4. e2e training — `run_agentic_bagel_rpco_lora.sh`

```bash
# 4-GPU default (CUDA_VISIBLE_DEVICES defaults to 2,3,4,5 on the author's boxes):
bash examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh

# 1-GPU sanity (RM off by default — actor+Omni+RM cannot fit one card):
CUDA_VISIBLE_DEVICES=3 bash examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh
```

Ladder: run 1-GPU first (loop + composite step, RM off), then 2-GPU, then the
4-GPU acceptance run with `ENABLE_RM=1` (colocated Qwen RM, TP-sharded across the
pool). `BAGEL_UND_AR_SERVING_READY=1` is the recipe default — flip to `0` to
re-hear the fail-closed gate after any serving change.

**Acceptance checklist (record all of it for the PR body):**

- ≥3 composite optimizer steps complete; `und_batch` nonempty every step;
  `gen/skipped_no_groups < 1` on steps with complete groups.
- Validation batch contains pattern-1, pattern-2 and pattern-3 episodes; at least
  one pattern-2 episode shows the RM-cued arc rewrite → `good_enough=YES` → `Done.`
  (requires `ENABLE_RM=1`).
- `und/no_image_credit` == count of `K=0` episodes; `gen/dropped_incomplete_groups`
  == 0 when all `S` seeds succeed; a `K=2` episode yields `2*S` GEN rows.
- One `policy_version` per step on BOTH replicas (UND AR + GEN diffusion); no
  mid-episode sync events in the logs.
- `bagel_role_timing` (`und_decode_s` / `gen_s` / `rm_s`) present per episode —
  this is the serial-baseline KPI anchor; do not claim throughput gains without it.
- Reward-hacking watch (latent-KL is the interim regularizer): validation reward
  inverting after initial rise, texture artifacts → report immediately, do not
  tune through it.

## 5. Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| `UND decode hit the GEN diffusion replica ... refuse soft-empty TQ` | Rollout served by the diffusion strategy, not the AR replica | Run the M0 spike; only then `BAGEL_UND_AR_SERVING_READY=1` |
| `Bagel Co-RL GEN traj stash incomplete` | `calculate_log_probs` / `algo.noise_level` / SDE window missing | Keep the rewrite defaults; do not disable FlowGRPO SDE to "make it run" |
| `TypeError ... reward_loop_worker_handles` in worker `__init__` | Pinned verl base class signature drift | Report the traceback — this is a pin-compat break, not something to silently swallow |
| `bagel RM result missing 'reward_score'` / `failed to score N/M image(s)` | Judge URL unreachable or image missing; scorer refuses zero-fill by design | Check `agentic_image_gen.vllm_url` (colocated RM) and that payload images exist on the RM worker's FS |
| `good_enough_threshold missing from payload` | Scorer knobs not stamped | Loop stamps via `agentic_scorer_knobs_from_config`; verify `agentic_image_gen` node exists in the composed config |
| OOM at first weight publish | FSDP summon + live replicas | `free_cache_engine=True` is forced by the rewrite; reduce `ROLLOUT_GPU_MEM_UTIL` / `und_gpu_memory_utilization` before touching code |
| `unknown parameter lr_gen` / dataclass error | Tree/config mismatch | The serving tree and the config must come from the same commit |
| Two different `image_gen_tool_agent` behaviors between jobs | Legacy env-var loop imported directly | Both registrations must not fire on package import — report the import chain |

## 6. Knob reference (env → effect)

| Env | Default | Effect |
| --- | --- | --- |
| `BAGEL_MODEL_PATH` | author's snapshot | Published Bagel-7B-MoT checkpoint (must be untied `lm_head`) |
| `CUDA_VISIBLE_DEVICES` | `2,3,4,5` | All visible cards go to the actor+GEN hybrid pool; UND AR colocates on the same PG |
| `ENABLE_RM` | `0` (1/2-GPU), `0` until e2e green | Colocated Qwen RM; `1` enables mid-loop scoring + image-grounded episode rewards |
| `REWARD_MODEL` | author's snapshot | Qwen RM for `reward.reward_model.model_path` |
| `REWARD_TP` / `ROLLOUT_TP` | pool size / 1 | Tensor-parallel for RM / GEN rollout (pool size must divide both) |
| `BAGEL_UND_AR_SERVING_READY` | `1` | Fail-closed gate; set `0` to re-prove the spike |
| `S` / `N` / `LORA_RANK` / `LORA_ALPHA` / `GEN_STEPS` / `GEN_HW` / `TRAIN_BSZ` | see script | FlowGRPO seeds per call / siblings / LoRA shape / denoise steps / generation size / batch |
| `REBUILD_UNICOT` | `1` | Rebuild train/val parquet before launch |

Training-side knobs (in the launch command, RFC §4.4/§5): `actor.optim.lr` (UND),
`+actor_rollout_ref.model.lr_gen=3e-5` (GEN group), `loss_weight_und/gen`,
`loss_mode` (`flow_grpo`; `grpo_guard` = GRPO-Guard ratio normalization,
UniGRPO-aligned ablation), `gen_regularizer` (`latent_kl` interim;
`velocity_mse` intentionally unimplemented until Phase 2). CFG is off in
training (`cfg_text_scale=1.0`, UniGRPO); re-enable for eval only via val kwargs.
