# PR1 Architectural Design: Agentic Trajectory + Decoupled Agent–Tool Policy

**Branch:** `feat/multiturn-traj-dual-policy`
**RFC:** `outputs/verl-omni-rfc-agentic-rl_v1.md` §7.1
**Base:** `1c858e2` → 4 commits, 15 files, +1063/-39 lines


---

## 0. Ongoing Updating List Monitoring Current Implementation Changes

### Round 1 (done checking)
#### CC Returned Issues
What PR1 should do
Our current AgenticLLMFSDPEngine is a working prototype but should be refactored to inherit from FSDPEngineWithLMHead before merging. The current implementation:

Missing micro-batch support — will OOM on realistic batch sizes
Missing global DP normalization — loss scale is wrong in multi-GPU
Missing FSDP scaler — mixed-precision training may diverge
Hardcoded loss — can't share with verl's PolicyLoss infrastructure

#### Dev Suggestions
As PR #258 and #269 were proposed to support training the understanding ability of omni models (text only gspo and multimodal dpo), I suggest PR1 can be built upon these two PRs to use main_omni entrypoint, instead of using main_diffusion.


### Round 2
There is an issue for the current implementation: The #295 haven't been implemented and merge to the remote yet. We need a minimal Lance-3B loading, and GRPO token-wise on the reflection traces. Is this issue solved now for the implementation?

---

## 1. High-Level Architecture

PR1 introduces **Mode (2a) agentic RL** — the agent LLM learns to reason, rewrite prompts, and reflect across multiple turns, while the diffusion model is a frozen tool. The architecture has three layers:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CONFIG & ROUTING LAYER                            │
│                                                                          │
│  DiffusionAlgoConfig.adv_estimator                                       │
│    ├── "flow_grpo" (existing) ──► diffusion timestep path (untouched)    │
│  model_type: "agentic_llm" (NEW) ──► token-level agentic path (via omni)   │
│                                                                          │
│  DiffusionModelConfig.agentic (AgenticConfig)                            │
│    ├── max_turns: 5                                                      │
│    ├── early_termination: true                                           │
│    └── observation_token_length: 128                                     │
│                                                                          │
│  DiffusionModelConfig.freeze: ["moe_gen"]                                │
│    └── param-name prefix matching for selective freezing                 │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        ROLLOUT LAYER                                       │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │            DiffusionARMultiTurnAgentLoop                             │  │
│  │  @register("diffusion_ar_multi_turn_agent")                          │  │
│  │                                                                      │  │
│  │                                                                      │  │
│  │  ┌─ Turn loop (turn_idx = 0 .. max_turns-1) ────────────────────┐    │  │
│  │  │                                                              │    │  │
│  │  │  [turn_idx=0] Agent LLM ─► reasoning + initial prompt        │    │  │
│  │  │  [turn_idx=1+] Agent LLM ─► reflection + REWRITTEN prompt    │    │  │
│  │  │     │  (UND path, logprobs captured, loss_mask=1)            │    │  │
│  │  │     ▼                                                        │    │  │
│  │  │  Parse output ─► extract <reasoning>, <prompt>, <decision>   │    │  │
│  │  │     │                                                        │    │  │
│  │  │     ├─ decision="terminate" ─► break loop                    │    │  │
│  │  │     │  ─► AgenticTurn(turn_idx, ..., decision="terminate")   │    │  │
│  │  │     │                                                        │    │  │
│  │  │     └─ decision="continue" ─► Tool call                      │    │  │
│  │  │           │  (GEN path, FROZEN, no logprobs, loss_mask=0)    │    │  │
│  │  │           ▼                                                  │    │  │
│  │  │        image observation                                     │    │  │
│  │  │           │                                                  │    │  │
│  │  │           ▼                                                  │    │  │
│  │  │        AgenticTurn(turn_idx, tool_call, tool_output,         │    │  │
│  │  │                     decision="continue")                     │    │  │
│  │  │           │                                                  │    │  │
│  │  │           ▼                                                  │    │  │
│  │  │        Append to chat: [assistant text, user(image + "ok")]  │    │  │
│  │  │           │                                                  │    │  │
│  │  │           └──► next iteration (turn_idx += 1)                │    │  │
│  │  │                                                              │    │  │
│  │  └──────────────────────────────────────────────────────────────┘    │  │
│  │                                                                      │  │
│  │  → DiffusionAgentLoopOutput with AgenticTrajectory in extra_fields   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Single vLLM-Omni instance serves both:                                    │
│    • Understanding path: auto-regressive text gen (reasoning/prompts)      │
│    • Generation path: flow-matching image gen (tool calls)                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                       │
│                                                                         │
│  AgenticTrajectory (structured, inspectable)                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  prompt: "Generate a cat wearing a blue hat"                       │ │
│  │                                                                    │ │
│  │  turns[0]:                                                         │ │
│  │    agent_text = "<reasoning>...</reasoning><prompt>A cat with      │ │
│  │                  blue hat</prompt><decision>continue</decision>"   │ │
│  │    agent_tokens = [101, 2054, ...]    ←── loss_mask = 1            │ │
│  │    agent_logprobs = [-0.23, -1.45, ...]                            │ │
│  │    tool_call.params = {"prompt": "A cat with blue hat"}            │ │
│  │    tool_output.output_data = <image tensor>  ←── loss_mask = 0     │ │
│  │    decision = "continue"                                           │ │
│  │                                                                    │ │
│  │  turns[1]:                                                         │ │
│  │    agent_text = "<reasoning>Hat is red, not blue. Fixing...</reasoning>" │
│  │                  <prompt>A cat wearing a BLUE top hat,             │ │
│  │                          deep blue color</prompt>                  │ │
│  │                  <decision>terminate</decision>"                   │ │
│  │    tool_call.params = {"prompt": "A cat wearing a BLUE top hat,    │ │
│  │                                  deep blue color"}                 │ │
│  │              ↑ turn[1].tool_call.params["prompt"]                  │ │
│  │              differs from turn[0]: "A cat with blue hat"           │ │
│  │              → old prompt was "blue hat" (ambiguous)               │ │
│  │              → new prompt adds "BLUE top hat, deep blue" (specific)│ │
│  │    decision = "terminate"                                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  DataProto serialization (flattened, tensorized):                       │
│    prompt_tokens:  [bsz, max_prompt_len]                                │
│    agent_tokens:   [bsz, max_total_tokens]   ← all turns concatenated   │
│    agent_logprobs: [bsz, max_total_tokens]                              │
│    loss_mask:      [bsz, max_total_tokens]   ← 1=agent, 0=obs-ph        │
│    responses:      [bsz, C, H, W]            ← final image for reward   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING LAYER                                    │
│                                                                          │
│  adv_estimator: "grpo" (verl standard, group-relative on scalar rewards)   │
│    • Scalar reward per trajectory → group-relative normalization         │
│    • Scalar advantage broadcast to all agent token positions             │
│                                                                          │
│  AgenticLLMFSDPEngine (NEW, inherits verl's FSDPEngineWithLMHead)        │
│    • Registered: model_type="agentic_llm", backend=["fsdp","fsdp2"]      │
│    • Forward: model(input_ids) → logits (single pass, no timestep loop)  │
│    • Loss: token-level PPO via loss_function callable, not hardcoded     │
│    • Selective freezing: moe_gen params → requires_grad=False            │
│    • Free inherited capabilities:                                        │
│        micro-batch splitting (prepare_micro_batches)                     │
│        global DP loss normalization (all_reduce batch_num_tokens)        │
│        FSDP grad scaler + autocast                                       │
│        FSDPCheckpointManager (save/load)                                 │
│    • Gradient: only flows through LLM_UND + ViT + lm_head                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.0.1 AgenticLLMFSDPEngine vs. existing verl-omni FSDP engines

verl-omni has 4 FSDP engine classes. The first 3 share a common `DiffusersFSDPEngine` base; `AgenticLLMFSDPEngine` is the first to extend `BaseEngine` directly:

| Dimension | PPODiffusersFSDPEngine | DPODiffusersFSDPEngine | NFTDiffusersFSDPEngine | **AgenticLLMFSDPEngine** |
|---|---|---|---|---|
| model_type | `diffusion_model` | `diffusion_dpo_model` | `diffusion_nft_model` | **`agentic_llm`** |
| Base class | `DiffusersFSDPEngine` | `DiffusersFSDPEngine` | `DiffusersFSDPEngine` | **`BaseEngine`** (direct) |
| LoRA support | Yes (`LoRAAdapterMixin`) | Yes | Yes | No (could add later) |
| Data shape | `[bsz, steps, C, H, W]` latents | Same | Same | **`[bsz, max_total_tokens]` token IDs** |
| Forward pattern | Loop over timesteps: `prepare_model_inputs → forward_and_sample → loss_fn(step)` | Same | Same | **Single pass: `model(input_ids) → logits`** |
| Loss abstraction | `DiffusionLossFn` (registered in `DIFFUSION_LOSS_REGISTRY`) | Same | Same | **Inline PPO clipped loss — no `DiffusionLossFn`** |
| Scheduler | `self.scheduler` (SDE/discrete) | Same | Same | **None — no diffusion scheduler** |
| Gradients | Flows through diffusion path | Same | Same | **Flows through LLM_UND only (`moe_gen` frozen)** |
| Checkpoint manager | `FSDPCheckpointManager` | Same | Same | **Stub from `BaseEngine` (no custom save/load yet)** |

The fundamental split: the 3 diffusion engines operate on **continuous latent trajectories** (per-timestep denoising), while `AgenticLLMFSDPEngine` operates on **discrete token sequences**. This is why it bypasses `DiffusersFSDPEngine` entirely — none of the diffusion infrastructure (scheduler, `prepare_model_inputs`, timestep loops, latent padding) applies to token-level data.

## 1.1 Distributed/Parallel

### Intra-Trajectory: Why turns CANNOT be parallelized

A single trajectory's turns form a **strict causal chain** — each turn's input depends on the previous turn's output:

```
Turn 0: agent(prompt)            → initial_prompt_0
        tool(prompt_0)           → image_0        ─┐
Turn 1: agent(prompt, image_0)  → rewritten_prompt_1  ← MUST wait for image_0
        tool(prompt_1)           → image_1        ─┐
Turn 2: agent(prompt, image_1)  → rewritten_prompt_2  ← MUST wait for image_1
        ...
```

The agent at turn i+1 **must see image_i to reflect and rewrite**. This is not an implementation constraint — it is a logical requirement of the reflection mechanism. The agent's value comes from analyzing what went wrong in the previous generation and adjusting the prompt accordingly. Without image_i, turn i+1 has nothing to reflect on, making it identical to turn 0.

**Verdict:** Intra-trajectory turns are inherently sequential. No design can parallelize them without removing the reflection mechanism that is the whole point of multi-turn agentic RL.

### Inter-Trajectory: Parallelism already exists

Multiple trajectories (different prompts) run concurrently via two mechanisms:

```
Prompt A ─► [Turn 0 ─► Turn 1 ─► Turn 2]  ┐
Prompt B ─► [Turn 0 ─► Turn 1]            ├── asyncio.gather(*tasks)
Prompt C ─► [Turn 0 ─► Turn 1 ─► Turn 2]  ┘   in DiffusionAgentLoopWorker
         ...
Prompt N ─► [Turn 0]
```

| Mechanism | What it parallelizes | Where |
|-----------|---------------------|-------|
| `asyncio.gather(*tasks)` | All N prompts in a batch run their agent loops concurrently | `DiffusionAgentLoopWorker.generate_sequences()` |
| vLLM-Omni request batching | Text gen and image gen requests across different trajectories are batched at the server level | vLLM-Omni server |
| Ray actor pool | Rollout replicas on separate GPUs handle disjoint subsets of prompts | `AgentLoopManager` / `RayWorkerGroup` |

This is the same parallelism that `DiffusionSingleTurnAgentLoop` already uses — PR1 inherits it without changes.

### Turn-to-Turn Latency Budget (per trajectory)

For a 3-turn trajectory on Lance-3B:

```
Turn 0:  text_gen(~2s) + image_gen(~5s) = ~7s
Turn 1:  text_gen(~3s) + image_gen(~5s) = ~8s  (text is longer: includes reflection + image context)
Turn 2:  text_gen(~3s) + image_gen(~5s) = ~8s  (termination decision, no image gen on final turn)
                                       Total: ~23s per trajectory

With batch parallelism (N concurrent): still ~23s wall-clock for the full batch
(since the slowest trajectory dictates batch completion time)
```

### What COULD be parallelized (future work)

| Idea | Feasibility | Why not in PR1 |
|------|-------------|----------------|
| **Speculative turn execution** — predict what the agent will say at turn i+1 and pre-generate image_i+1 before turn i completes | Low | Requires a predictor model; if prediction is wrong, wasted compute; adds significant complexity |
| **Parallel tool calls within a turn** — if the agent requests multiple generations (e.g., 4 variants), run them concurrently | Medium | Requires extending the agent output format to support multi-call; not in PR1 scope |
| **Async reward during rollout** — compute reward on Turn 0's image while Turn 1's agent text is generating | High (already in PR1) | `_compute_score()` runs async via Ray actors during the rollout; not turn-level but trajectory-level |
| **Pipeline parallelism across turns** — have separate GPU pools for text gen vs image gen so they can overlap across trajectories | Medium | Requires co-located vs decoupled GPU pool config; noted as config option but not initially implemented |

---

## 2. Major Design Choices

### Choice 1: Lance-3B (not BAGEL) as the unified model

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Model | Lance-3B (ByteDance, 3B-active MoT) | BAGEL-7B (as originally in RFC); larger but requires #295 SFT infra | Smaller → faster dev iteration; already has vLLM-Omni support; Apache 2.0 license |
| Agent path | LLM_UND (mlp, self_attn.\*, lm_head) → trainable | Separate standalone LLM (e.g., Qwen3-VL) as agent; more modular but no shared backbone | Understanding path generates reasoning + rewritten prompts auto-regressively |
| Tool path | LLM_GEN (mlp_moe_gen, self_attn.\*_moe_gen) → frozen | External diffusion API (FLUX, SD3.5 via HTTP); model-agnostic but adds latency | Generation path produces images via flow matching; weights fixed |
| Selective freezing | Param-name prefix matching (`"moe_gen"`) | Per-layer explicit config list; more granular but complex config surface | Simple, model-agnostic; no need for model-specific freeze logic |

### Choice 2: Single vLLM-Omni instance for both text and image generation

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Text gen | vLLM-Omni understanding head, auto-regressive | Separate vLLM text server; more infrastructure but decouples text/image scaling | Same server, same model — no separate text LLM server needed |
| Image gen | vLLM-Omni generation head, flow matching | Direct HuggingFace diffusers pipeline; simpler but slower, no continuous batching | Frozen tool call within the multi-turn loop |
| Logprobs | Captured during text gen only | Recompute via actor forward pass (like `_compute_old_log_prob`); more consistent with existing pattern but 2x compute | Image gen returns no logprobs (frozen, not trained) |
| Overall | Single Lance instance serving both roles | Separate text LLM + separate image model (two server pools); modular but 2x GPU memory | Lance MoT already has both paths; single server minimizes infra complexity |

### Choice 3: Token-level GRPO loss — the first non-diffusion-timestep loss in verl-omni

**Context:** All 7 existing loss functions in verl-omni's `DIFFUSION_LOSS_REGISTRY` operate on diffusion-specific model outputs — per-timestep `log_probs`, `prev_sample_mean`, `std_dev_t`, `noise_pred`, etc. None handle token sequences:

| Existing loss | Required model_output keys | Paradigm |
|---|---|---|
| `flow_grpo` / `dance_grpo` | `log_probs` (per-timestep) | Diffusion SDE denoising |
| `flow_dppo` | `log_probs`, `prev_sample_mean`, `std_dev_t`, `sqrt_dt` | Diffusion DPPO |
| `grpo_guard` | `log_probs`, `prev_sample_mean`, `std_dev_t`, `sqrt_dt` | Diffusion GRPO-Guard |
| `dpo` | `noise`, `latent`, `noise_pred` | Diffusion DPO |
| `diffusion_nft` | `noise_pred` + batch-level `reward_prob` | Diffusion NFT |
| `kl` | `prev_sample_mean`, `std_dev_t` | KL penalty (no reward) |

PR1 introduces a fundamentally new data path — **token sequences** instead of diffusion latents. The `AgenticLLMFSDPEngine` skips the `DiffusionLossFn` abstraction entirely because no existing loss class expects `loss_mask` or per-token logprobs. This is a new loss paradigm, not an extension of an existing one.

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Loss type | Token-level PPO clipped loss (new to verl-omni) | Wrap tokens as pseudo-diffusion "latents"; hacky, mismatched abstraction | Agent LLM generates text tokens, not diffusion latents |
| Engine base | Inherit from verl's `FSDPEngineWithLMHead` (token-level LLM engine) | (a) Extend `BaseEngine` directly: simpler but loses micro-batching, DP norm, scaler, checkpointing. (b) Extend `PPODiffusersFSDPEngine` with token-mode branch: conflicts with diffusion timestep infrastructure | `FSDPEngineWithLMHead` is verl's battle-tested engine for token-level PPO/GRPO — inheriting it gives micro-batch splitting, global DP normalization, FSDP grad scaler, autocast, and checkpoint management for free (see Choice 8) |
| Advantage | One scalar per trajectory, broadcast to all agent tokens | Per-token advantage computation with a separate critic model; more granular but overkill for PR1 | Standard GRPO: group-relative reward → advantage |
| Loss mask | 1 for agent text tokens, 0 for observation placeholders | Separate sequence packing (pad each turn to fixed length); more memory but simpler engine | Tool outputs (images) are frozen — no gradient through them |
| Old logprobs | Rollout logprobs used directly (no recompute) | Recompute via actor forward pass (like current `_compute_old_log_prob`); more accurate but 2x compute | `agent_logprobs` from rollout = `old_log_probs`; avoids separate actor forward pass |

### Choice 4: Explicit prompt rewriting in trajectory structure

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Capture | `turn[i].tool_call.params["prompt"]` (typed field) | Parse from free-form text with regex; fragile, model-dependent | Structurally encoded, machine-comparable without parsing |
| Comparison | Direct struct diff: `turn[i].tool_call.params["prompt"]` vs `turn[i+1].tool_call.params["prompt"]` | Embedding similarity (cosine between old/new prompt embeddings); more semantic but less interpretable | The diff IS the rewriting action — exact, inspectable, no embedding model needed |
| Agent text | Also preserved in `agent_text` (free-form) | Drop free-form, keep only structured fields; clean but loses human readability | Human-readable reasoning + rewritten prompt annotation for debugging |
| Loss signal | All agent tokens share same trajectory-level advantage | Per-turn reward decomposition; more granular but requires PR2 multi-dim rewards | The quality of rewriting is optimized implicitly through the trajectory-level reward |

### Choice 5: XML-tag agent output format

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Format | `<reasoning>...</reasoning><prompt>...</prompt><decision>continue\|terminate</decision>` | JSON function-calling / tool-use API; requires model-specific tool-calling support (e.g., OpenAI-style) | Simple, regex-parseable, no tokenizer changes, works with any LLM |
| System prompt | Constrains output to this format | Few-shot examples in prompt; increases prompt length and token costs | `AGENT_SYSTEM_PROMPT` constant — zero-shot, deterministic |
| Fail-safe | Missing tags → `decision="terminate"` | Hard error (crash trajectory); simpler but wastes rollout compute | Don't loop forever on malformed output; trajectory still usable for training |
| Decision parsing | Substring match: `"continue" in raw_decision` | Exact match (`== "continue"`); stricter but brittle to whitespace/casing | Lenient parsing handles model output variability |

### Choice 6: Build on the omni entrypoint (`main_ppo` / `main_omni`), not `main_diffusion`

PRs #258 and #269 introduced infrastructure for training omni models' understanding path (text-only GSPO and multimodal DPO) via `verl.trainer.main_ppo` → `omni_trainer.yaml` → `FSDPEngineWithLMHead`. This path already handles token-level PPO/GRPO natively — no diffusion timestep loop, no scheduler, no latent padding. Building agentic RL on this path means we inherit all of verl's text LLM training infrastructure for free.

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Entrypoint | `verl.trainer.main_ppo` (today) / `verl_omni.trainer.main_omni` (after #258 merges) | `main_diffusion` + `ray_diffusion_trainer.py` with `is_agentic` branches (original implementation); would require adding token-level paths into a diffusion-specific trainer | Omni path already handles token-level data, GRPO advantage, PPO loss, micro-batching — no diffusion infrastructure to work around |
| Config | `agentic_trainer.yaml` inheriting `omni_trainer.yaml` → `ppo_trainer` | Custom `agentic` section in `DiffusionAlgoConfig`; would tie agentic RL to diffusion-specific config fields | Omni config already has `OmniModelConfig`, `PolicyLossConfig`, `AlgoConfig` — agentic just adds `AgenticConfig` + `freeze` list |
| Engine | `AgenticLLMFSDPEngine(FSDPEngineWithLMHead)` — 55 lines, only overrides `build_module` for selective freezing | `AgenticLLMFSDPEngine(BaseEngine)` — 170 lines, reimplemented micro-batching, DP norm, loss inline (original prototype) | Inheriting `FSDPEngineWithLMHead` gives micro-batch splitting, global DP normalization, FSDP scaler, autocast, checkpointing for free |
| Advantage | verl's standard `grpo` estimator (group-relative on scalar rewards) | Custom `agentic_grpo` estimator registered in `DIFFUSION_ADV_ESTIMATOR_REGISTRY`; identical math, wrong registry | Omni path uses verl's estimator registry — no need for a diffusion-specific copy |
| Diffusion trainer | Completely untouched — `ray_diffusion_trainer.py` reverted to original | Conditional `is_agentic` branches in `fit()` and `_update_actor()`; would add token-level paths alongside diffusion timestep paths | Zero diff to existing FlowGRPO — no risk of regression |

### Choice 7: AgenticTrajectory as non_tensor_batch (not tensors)

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Storage | Full `AgenticTrajectory` in `DataProto.non_tensor_batch["agentic_trajectory"]` | New dedicated container class (`AgenticDataProto`); more type-safe but creates parallel hierarchy to `DataProto` | Reward function + eval need access to all turns, rewritten prompts, images — fits existing `non_tensor_batch` pattern |
| Serialization | `agentic_trajectory_to_dict()` → JSON-serializable dict | Pickle; simpler but unsafe (arbitrary code execution on deserialization) | Safe for `non_tensor_batch` numpy object arrays; no pickle security risk |
| Tensors | Flattened `agent_tokens`, `agent_logprobs`, `loss_mask` in `DataProto.batch` | Keep per-turn tensors as separate named fields (`turn_0_tokens`, `turn_1_tokens`, ...); would need N sets of tensors for varying turn counts, makes micro-batching impossible across trajectories with different N | Single padded `[bsz, max_total_tokens]` tensor — same shape for all trajectories regardless of turn count, compatible with existing FSDP2 micro-batch infrastructure |
| Image tensors | `responses` = final turn's image only | Store all intermediate images as tensors; enables turn-level reward but O(N×images) memory | Final image is what the reward function scores; intermediate images kept in `AgenticTrajectory` object |
| Tensor reconstruction | `from_dict` creates `torch.zeros(shape)` placeholders | Store image file paths (string refs); lighter but requires filesystem access at eval time | Tensors can't be JSON-serialized; actual tensor data lives in rollout buffers; shape is sufficient for structural checks |

### Choice 8: Inherit verl's `FSDPEngineWithLMHead` (not `BaseEngine`) for the agentic training engine

verl upstream ships `FSDPEngineWithLMHead` (registered as `model_type="language_model"`) — a mature token-level PPO/GRPO engine used for all text LLM RL training in the verl ecosystem. It provides infrastructure that our `AgenticLLMFSDPEngine` would otherwise need to reimplement:

| Capability | `BaseEngine` (current PR1 prototype) | verl `FSDPEngineWithLMHead` (target) | Why it matters |
|---|---|---|---|
| Micro-batch splitting | None — single full-batch forward | `prepare_micro_batches()` with DP-aware sharding | Without it, realistic batch sizes OOM |
| Global loss normalization | Local `loss_mask.sum()` only | `all_reduce(batch_num_tokens)` across DP ranks | Loss scale is wrong in multi-GPU without it |
| FSDP grad scaler | None | `self.scaler` (FSDP2 mixed-precision) | Mixed-precision training may diverge without it |
| Autocast | None | `_autocast_dtype` (bf16/fp32) | No automatic dtype management |
| Remove-padding | Full padded tensors `[bsz, max_len]` | Fused kernels — only compute on non-pad tokens | Wasted compute on padding positions |
| Sequence parallelism | None | Ulysses SP (`ulysses_pad_and_slice_inputs`) | Can't scale sequence length across GPUs |
| Checkpoint management | Stub from `BaseEngine` | `FSDPCheckpointManager` (save/load/resume) | Can't resume training from checkpoint |
| Loss ownership | Hardcoded inline (`clip_ratio=0.2`) | `loss_function` callable passed by trainer | Can't share with verl's `PolicyLoss` infrastructure |
| Lines to implement | ~170 (from scratch) | ~50 (override `prepare_model_inputs` + provide `loss_function`) | Less code, more capability |

**The only things we need to provide:**

1. **Override `prepare_model_inputs`**: feed `agent_tokens` as `input_ids` with observation placeholders `loss_mask=0`, construct position_ids from turn boundaries

2. **A `loss_function` callable** matching verl's `PolicyLoss` signature:
   ```python
   def agentic_ppo_loss(model_output, data, dp_group):
       log_probs = model_output["log_probs"]      # from prepare_model_outputs
       old_log_probs = data["old_log_probs"]
       advantages = data["advantages"]
       loss_mask = data["loss_mask"]
       # standard PPO clipped loss × loss_mask
       ratio = torch.exp(log_probs - old_log_probs)
       pg_loss = -torch.min(ratio * advantages,
                            torch.clamp(ratio, 1-ε, 1+ε) * advantages)
       loss = (pg_loss * loss_mask).sum() / data["batch_num_tokens"]
       return loss, {"approx_kl": ...}
   ```

3. **Selective freezing** in `build_module` — already implemented, just moved to the inherited method

| Axis | Decision | Alternative Choices | Rationale |
|------|----------|--------------------|-----------|
| Base class | `FSDPEngineWithLMHead` from verl upstream | `BaseEngine` directly (current prototype): simpler but missing micro-batching, DP norm, scaler, checkpointing — all must be reimplemented | Free capabilities: micro-batch splitting, global DP normalization, FSDP grad scaler, autocast, remove-padding, sequence parallelism, checkpoint manager. Only ~50 lines of override code needed |
| Loss function | Separate `agentic_ppo_loss` callable, passed to `forward_backward_batch` | Hardcoded inline loss (current prototype): works but can't share infrastructure with verl's `PolicyLoss` | Matches verl's pattern — engine does forward + backward, trainer provides loss function |
| Registration | `model_type="agentic_llm"` — distinct from verl's `"language_model"` | Reuse `model_type="language_model"`: would conflict with verl's text LLM config expectations | Separate model_type allows agentic-specific config while reusing the same engine class |

---

## 3. File Map

```
verl_omni/
├── agent_loop/
│   ├── agentic_trajectory.py          ╶ NEW   AgenticTrajectory dataclass hierarchy
│   ├── agent_output_parser.py         ╶ NEW   XML-tag parser + AGENT_SYSTEM_PROMPT
│   ├── trajectory_serializer.py       ╶ NEW   AgenticTrajectory → DataProto tensors
│   ├── diffusion_ar_multi_turn_agent_loop.py  ╶ NEW  Main rollout loop (text + image)
│   ├── diffusion_agent_loop.py        ─ EDIT  Serialize AgenticTrajectory in _postprocess
│   └── __init__.py                    ─ EDIT  Export new types + register agent loop
│
âââ trainer/
â   âââ config/
â   â   âââ agentic_trainer.yaml       â¶ NEW   Agentic config (inherits omni_trainer â ppo_trainer)
â   â   â   âââ omni/
â   â   â       âââ model/
â   â   â           âââ omni_model.yaml    â EDIT  +agentic + freeze fields
â   â   âââ algorithm.py               (no change â omni uses verl grpo estimator)
â   âââ diffusion/
â       âââ diffusion_algos.py         (no change â reverted to original)
â       âââ ray_diffusion_trainer.py   (no change â reverted to original)

âââ workers/
â   âââ config/
â   â   âââ diffusion/
â   â   â   âââ model.py               (no change â reverted to original)
â   â   âââ omni/
â   â       âââ model.py               â EDIT  +AgenticConfig + freeze list on OmniModelConfig
â   âââ engine/
â       âââ fsdp/
â           âââ agentic_impl.py        â¶ NEW   AgenticLLMFSDPEngine(FSDPEngineWithLMHead)
â           âââ __init__.py            â EDIT  Export new engine
├── utils/
│   └── dataset/
│       └── unicot_adapter.py          ╶ NEW   UniCoT-Self-Reflection-6K → AgenticTrajectory
│
tests/
├── data/
│   └── multi_turn_toy.jsonl           ╶ NEW   3-example toy multi-turn dataset
└── test_agentic_trajectory.py         ╶ NEW   12 unit tests (8 pass, 4 env-limited)
```

**7 new files, 6 modified files.**

---

## 4. Key Innovelties

### 4.1. Zero diff to existing FlowGRPO — agentic RL uses the omni path

The agentic path uses `main_ppo` / `main_omni` with `model_type="agentic_llm"` — completely separate from `main_diffusion`. The diffusion trainer (`ray_diffusion_trainer.py`, `diffusion_algos.py`, `model.py`) has **zero changes** from the base commit. No conditional branches, no new enum values, no routing logic. Agentic RL is a parallel training pipeline, not a modification of the existing one.

### 4.2. Prompt rewriting is a first-class citizen

Existing approaches (GenAgent, VisionCreator, GEMS) structure tool calls in JSON/action blobs, so the rewritten prompt *is* present — but you must parse it from generic action fields (e.g., `action["arguments"]["prompt"]`). PR1 elevates prompt rewriting to a named, typed field: `turn[i].tool_call.params["prompt"]` vs `turn[i+1].tool_call.params["prompt"]`. The diff between consecutive prompts is a direct struct comparison rather than a parse-then-compare operation, making the rewriting action inspectable by eval tools and reward functions without manual parsing.

### 4.3. Same model, two roles

Lance-3B's Mixture-of-Transformers architecture naturally supports the decoupled agent–tool policy: the understanding path learns to write better prompts, while the generation path executes them. No separate models, no cross-model communication overhead — one model, two roles, one vLLM-Omni server.

### 4.4. Loss mask as the training boundary

The `loss_mask` tensor is the single mechanism that separates trainable agent tokens from frozen tool observations. The engine doesn't need to know about "turns" or "tools" — it just sees a sequence of tokens with a binary mask. This keeps the training engine simple and general.

---

## 5. Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| `test_single_turn_agent` can't run without `uvicorn`/`verl` installed | Low | Environment issue, not code |
| `AgenticLLMFSDPEngine` uses hardcoded `clip_ratio=0.2` | Low | Now inherited from `FSDPEngineWithLMHead` — clip_ratio comes from verl's `PolicyLossConfig` |


### 5.1 Solved Issues

#### 5.1.1 Engine base class: `BaseEngine` → `FSDPEngineWithLMHead`
**Date:** 2026-07-27
**Trigger:** Code review identified that extending `BaseEngine` directly was missing micro-batch support, global DP normalization, FSDP scaler, and checkpoint management.
**Resolution:** Refactored `AgenticLLMFSDPEngine` to inherit from verl's `FSDPEngineWithLMHead` (170 → 55 lines). Also reverted `ray_diffusion_trainer.py` to original (removed `is_agentic` branches), reverted `diffusion_algos.py` to original (removed `grpo (via omni)` enum + `token_level_grpo` function), and moved `AgenticConfig` from `DiffusionModelConfig` to `OmniModelConfig`. Agentic training now uses the `main_ppo` / `main_omni` entrypoint path instead of `main_diffusion`.

#### 5.1.2 Aligned with #295 (Multi-Turn Visual Reflection SFT)
**Date:** 2026-07-27
**Trigger:** Design review identified overlapping concerns with #295's SFT trajectory format and UniCoT adapter.
**Changes:**
- Added `VisualReflectionTrajectory`, `ReflectionStep`, `ImageRef` to `agentic_trajectory.py` — aligned with #295's canonical SFT format
- Added `visual_reflection_to_agentic()` converter — lifts SFT trajectories to RL `AgenticTrajectory` with sensible defaults for RL fields
- Refactored `unicot_adapter.py` to produce `VisualReflectionTrajectory` first (via `load_unicot_visual_reflection()`), then convert via `visual_reflection_to_agentic()` — avoids duplicating the UniCoT parsing logic when #295's data PR merges
- Changed decision vocabulary from `"terminate"` to `"stop"` across all files — aligns with #295's `action: "continue" | "stop"` vocabulary
- Added provenance fields (`trajectory_id`, `source_dataset`) to `AgenticTrajectory` — forward-compatible with #295's `VisualReflectionTrajectory`

#### 5.1.3 `.gitignore` stray `.*` line removed
**Date:** 2026-07-27
**Resolution:** Removed `.*` from `.gitignore` (was ignoring all dotfiles). Kept `superpowers` entry (our scratch directory).
