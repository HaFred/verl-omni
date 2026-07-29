# [RFC] Agentic RL for Omni-Modal Generation in verl-omni

## 1. Summary

This RFC proposes extending verl-omni from single-turn RL for diffusion models to **multi-turn agentic RL for omni-modal generation**. The current framework supports one-shot generation (prompt → image → reward → update). We propose adding: (1) multi-turn trajectory support with agent reasoning, iterative prompt rewriting, and tool invocation, (2) a Decoupled Agent–Tool Policy (train agent LLM, freeze diffusion), where an agent LLM and/or the diffusion model can be trained across Modes (2a)/(2b), and (3) a multi-dimensional reward system with per-turn granularity and VisionCreator-R1's RPCO staged training. These extensions are motivated by recent research demonstrating that agentic approaches — iterative prompt rewriting, explicit reflection, multi-step planning — achieve substantial gains over single-turn generation (e.g., +23.6% on GenEval++, +14.22 average across 9 benchmarks without any training).

## 2. Terminology

This RFC adopts the training taxonomy used by verl-omni maintainers, organized along two dimensions: **turn structure** (single-turn vs multi-turn) and **what is trained** (diffusion, agent LLM, or both):

| # | Name | Turn structure | Policy (trained) | Frozen | Description |
|---|------|---------------|-----------------|--------|-------------|
| **Mode (0)** | Single-stage RL | Single-turn | Diffusion (FlowGRPO) or AR / omni-modality (GSPO) | N/A | Current verl-omni capability: single-stage RL for generation-model output quality, not modified by this RFC. |
| **Mode (1)** | Multi-stage RL | Single-turn | Both agent LLM and diffusion (alternating across stages) | None | Co-optimization of agent reasoning and generation quality across staged training. Requires coordinated cross-model gradient computation. Deferred to future work. |
| **Mode (2a)** | Agentic LLM RL | Multi-turn | Agent LLM (e.g., Qwen3-VL) | Diffusion model(s) | Train the agent's reasoning, prompt rewriting, reflection, and tool-invocation behavior. The diffusion model is a frozen tool called by the agent. **This is the primary new capability proposed by this RFC.** |
| **Mode (2b)** | Agentic Diffusion RL | Multi-turn | Diffusion model (e.g., Qwen-Image) | Agent LLM | Train the diffusion model's generation quality using agentic trajectories — the frozen agent LLM provides prompt rewriting and per-turn judgments as intermediate reward signals. Not in initial PRs but acknowledged as a future direction. |
| **Mode (3)** | Agentic Co-RL | Multi-turn | Both agent LLM and diffusion (joint, in-loop) | None | Train both the agent's reasoning/prompt rewriting and the diffusion tool's generation quality end-to-end within the multi-turn agentic loop. Combines Mode (2a) and Mode (2b) into a single joint optimization. Requires differentiable tool integration or multi-turn credit assignment across both policies. Most ambitious; deferred to future work, not in initial PRs. |

**Unless otherwise specified, "agentic RL" in this RFC refers to Mode (2a).**

## 3. Motivation

### 3.1. The single-turn ceiling

Current verl-omni trains diffusion models via FlowGRPO: a prompt produces n stochastic SDE trajectories, each scored by a reward function, and group-relative advantages drive policy updates. This is effective for simple alignment tasks (e.g., OCR text rendering: 59% → 92%) but fundamentally limited for complex generation:

- **No reasoning**: the model cannot decompose a complex prompt, plan a multi-step creation workflow, or reflect on its output.
- **No iteration**: if the first generation fails, there is no mechanism to analyze the failure, rewrite the prompt, and retry with a refined approach.
- **No tool orchestration**: real-world visual creation involves multiple tools (text-to-image, image editing, text-to-video) used in sequence — current FlowGRPO cannot express this.

### 3.2. Evidence from research

Recent work demonstrates that agentic approaches to multimodal generation achieve gains far beyond what single-turn RL can reach. A key mechanism across these works is **prompt rewriting** — the agent modifies the prompt to the diffusion model between iterations based on visual feedback:

| Approach | Method | Gain | Key mechanism |
|----------|--------|------|---------------|
| GenAgent | SFT + agentic GRPO on agent LLM | +23.6% GenEval++ | Multi-turn: reason → **rewrite prompt** → call FLUX → judge → reflect → retry |
| VisionCreator | PST + VRL on agent LLM (8B/32B) | Outperforms larger closed-source | UTPC: understand → think → plan → create (15+ steps, 36 tools) |
| VisionCreator-R1 | RPCO on agent LLM (32B) | Outperforms Gemini 2.5 Pro | Explicit reflection → corrective planning → **rewrite tool call** → re-generate |
| GEMS | No training (inference-time) | +14.22 avg, 6B > SOTA | Agent loop: planner → decomposer → generator → verifier → **refiner (prompt rewriting)** |
A critical finding across these works: **RL trains the agent LLM (reasoning, planning, reflection, tool invocation), not the diffusion model itself**. The diffusion model is always a frozen external tool. This means the RL trajectory is a sequence of LLM token generations interleaved with tool-call observations — fundamentally different from the current single-turn SDE denoising trajectory in verl-omni.

### 3.3. Relationship to #295 — Multi-Turn Visual Reflection SFT for BAGEL

Issue [#295](https://github.com/verl-project/verl-omni/issues/295) (opened Jul 22, 2026) proposes multi-turn visual reflection **SFT** for BAGEL. It defines three SFT primitives (`t2i`, `reflect`, `edit`) with a reflect-edit loop and a 4-PR implementation plan. Our RFC and #295 are **complementary, not overlapping**:

| Dimension | #295 (SFT) | This RFC (RL) |
|-----------|------------|---------------|
| **Training paradigm** | Supervised fine-tuning on curated trajectories | Online RL via GRPO with multi-dimensional rewards |
| **What is learned** | Imitation of reflection/edit patterns from teacher data | Optimization of reflection quality via reward signals |
| **Reward signals** | None — uses ground-truth targets | 8-dimensional: plan, reflection, format, tool, result, pointwise, pairwise, diffusion |
| **Trajectory source** | Offline datasets (UniCoT, Echo-4o, Midjourney prompts) | Online rollout via agentic loop |
| **Model scope** | BAGEL-specific (BagelForSFT) | Model-agnostic (any agent LLM + any frozen diffusion tool) |
| **Reflection-plan asymmetry** | Not addressed (SFT does not have this problem) | Addressed via RPCO staged training (§5) |
| **What it explicitly excludes** | "Online multi-turn RL or reward-model changes" | SFT primitives, BAGEL-specific model refactoring |

**Complementary relationship:** #295's SFT output can serve as the cold-start initialization for this RFC's RL training — mirroring the two-stage paradigm (SFT cold-start → agentic RL) proven by GenAgent. #295 teaches the model *how* to reflect; this RFC teaches it *when* and *how well* to reflect via reward-driven optimization.

## 4. Scope

### 4.1. What this RFC solves

1. **Multi-turn trajectory format** for agent LLM + frozen diffusion tool interaction, with per-turn prompt rewriting and loss masking on environment observations
2. **Mode (2a) agentic LLM RL** — GRPO on agent LLM weights, optimizing the agent's prompt rewriting ability with diffusion model as frozen tool
3. **Multi-dimensional reward computation** — 8 reward dimensions with per-turn granularity and pairwise monotonic-improvement signals
4. **RPCO staged training** — decouple-then-fuse strategy to mitigate the reflection-plan optimization asymmetry
5. **Trajectory resampling** — oversample + resample by interaction turns for diverse trajectory patterns

### 4.2. What this RFC does NOT solve

1. **Mode (1) multi-stage RL** — This's overambitious at this point, due to the resources/compute constraint. Coordinated gradient computation across agent LLM and diffusion model simultaneously across stages. Requires cross-model gradient orchestration not supported by current FSDP2/VeOmni backends. Deferred to future work.
2. **Mode (2b) agentic diffusion RL** — training the diffusion model with a frozen agent LLM providing prompt rewriting and per-turn judgments as intermediate reward signals. Acknowledged as a promising direction (see §2) but deferred to future work, not in initial PRs.
3. **Mode (0) single-stage RL** — this is the current FlowGRPO/GSPO capability, already supported. Not modified by this RFC.
4. **BAGEL-specific SFT primitives** — `t2i`, `reflect`, `edit` SFT objectives are #295's scope. This RFC consumes SFT checkpoints as initialization but does not implement SFT training itself.
5. **Whole-trajectory packing** — variable-length sequence packing for memory-efficient multi-turn training is an engineering optimization left for follow-up.
6. **Tokenizer vocabulary changes** — no special tokens or vocabulary modifications proposed.
7. **Full-weight training recipes** — initial implementation targets LoRA. Full-weight recipes are straightforward extensions but not included in the initial PRs.
8. **External agent harness trajectory capture** — verl-omni accepts multi-turn trajectories in the `AgenticTrajectory` format from any source, but building an API-proxy gateway for capturing trajectories from external harnesses is outside this RFC's scope. Trajectory capture infrastructure is expected to be provided by the complementary project, [`verl-project/uni-agent`](https://github.com/verl-project/uni-agent).
9. **Simulated environment mode** — training with fake tool outputs for cost-effective planning-logic optimization is a promising direction (see VisionCreator's VRL) but is not included in the initial PRs.
10. **Online RL on diffusion model weights** — this RFC trains the agent LLM via Mode (2a), not the diffusion model. The diffusion model is always frozen in all proposed modes. RL on diffusion weights remains Mode (0) single-stage FlowGRPO's domain.

## 5. Background

### 5.1. Current verl-omni architecture

```
Prompt → vLLM-Omni rollout (n SDE trajectories) → n images
                                                        ↓
                                              Reward scoring (per image)
                                                        ↓
                                              Group-relative advantage (GRPO)
                                                        ↓
                                              Policy update (diffusion model weights)
```

**Key characteristics:**
- **Policy = diffusion model** (the only trainable component)
- **Trajectory = SDE denoising path** (continuous, not token-based)
- **Reward = single scalar per image** (rule-based or model-based)
- **No agent LLM** in the training loop
- **No multi-turn interaction** — strictly one-shot
- **Reward is computed after generation completes** — no feedback into the generation process

### 5.2. What needs to change

The agentic setting introduces a fundamentally different trajectory structure:

```
Turn 1: agent LLM generates (reasoning + initial prompt/tool call) → diffusion model generates image → observation
Turn 2: agent VLM generates (judgment + reflection) → decision: retry or terminate
Turn 3: agent VLM generates (reflected reasoning + rewritten prompt/tool call) → diffusion model generates image → observation
...
Final: reward computed on full trajectory (multi-dimensional)
```

The key agentic behavior is **prompt rewriting**: at each turn, the agent LLM analyzes the previous generation's failures and produces a rewritten prompt for the next diffusion model call. This is what the RL optimizes — not the diffusion model's weights, but the agent's ability to write better prompts through reasoning and reflection.

This requires:
1. A new trajectory format that captures multi-turn LLM-diffusion interactions, including rewritten prompts at each turn
2. Support for training the agent LLM (not just the diffusion model) — the agent's prompt rewriting ability is the trainable policy
3. Multi-dimensional rewards that evaluate reasoning and prompt rewriting quality, not just final image quality

### 5.3. Prevalent Omni-Agentic Benchmarks/Datasets
In #295, the three datasets are:

| Public dataset | Role | Construction |
| --- | --- | --- |
| [Fr0zencr4nE/UniCoT-Self-Reflection-6K](https://huggingface.co/datasets/Fr0zencr4nE/UniCoT-Self-Reflection-6K) | Native multi-turn reflection data | Directly parse its image states, evaluations, edits, and terminal decisions. |
| [Yejy53/Echo-4o-Image](https://huggingface.co/datasets/Yejy53/Echo-4o-Image), `Instruction-Following-Image` | Synthesized 0/1-turn data | Generate a repo-native BAGEL draft from each public T2I pair and use a VLM to construct and verify a zero- or one-edit trajectory. |
| [brivangl/midjourney-v6-llava](https://huggingface.co/datasets/brivangl/midjourney-v6-llava) | Synthesized k-turn data | Use only the public prompt field and iteratively apply repo-native BAGEL generation, VLM reflection, and an edit model. |

*Note:* `Echo-4o-Image` is 0/1-turn by design, so it does **not** independently satisfy criterion 2 — it is kept only because you selected it as a baseline example. `UniCoT-Self-Reflection-6K` and `midjourney-v6-llava` do satisfy all three.

Beyond these three, some other multi-turn image generation datasets include:

| Public dataset | Role | Construction |
| --- | --- | --- |
| [Fr0zencr4nE/UniCoT-Breakdown-3K](https://huggingface.co/datasets/Fr0zencr4nE/UniCoT-Breakdown-3K) | BAGEL-native multi-turn planning + self-reflection | Companion release to UniCoT-6K. Parses **macro-level CoT** (task breakdown / planning) and **micro-level CoT** (self-reflection) over evolving image states. Same BAGEL-7B-MoT interleaved schema; directly extends the UniCoT trajectory format with the missing *planning* step. Multi-turn, with reflection at the micro level → meets 1+2+3. |
| [appletea233/EditThinker · ThinkEdit-140K](https://github.com/appletea233/EditThinker) (HF dataset linked from repo) | Multi-turn think-while-edit **with critique score + binary stop flag** | Each round: the editor produces an image, then the Thinker emits a **critique score `S_t`**, a reasoning trace `R_t`, and a refined instruction `T_t`; the loop repeats until `S_t` passes a threshold (binary stop). 140k samples; ships `sft_train.json` + `rl_train.jsonl` (RL-ready). The critic/editor split is exactly the **Decoupled Agent–Tool Policy** of the RFC. Critique score = explicit `R_reflect` / reward flag → meets 1+2+3. (2025-12; Meituan + CUHK MMLab + Beihang + Tsinghua.) |
| [WeiChow/WEAVE (WEAVE-100k)](https://huggingface.co/datasets/WeiChow/WEAVE) | Multi-turn interleaved image-gen dialogue (visual memory) | 100,750 chats / ~600k dialogue turns / ~700k images spanning comprehension, editing, and generation that require reasoning over prior context (rollback, recall, fusion across turns). CVPR 2026; a finetuned **BAGEL** checkpoint is released. ⚠ **Reflection is NOT a stored field** — only the held-out **WEAVEBench** uses an external VLM-judge (Key-Point / Visual-Consistency / Image-Quality / Accuracy). So it meets criteria **1+2**, and criterion 3 only via an *external* judge, not an in-sample flag. Listed because it is the largest fresh (2026) multi-turn image-gen dialogue corpus; use it for PR1 trajectory format, not as a reward source. |

## 6. Proposed Design

### 6.1. Multi-Turn Trajectory Format (GenAgent, Agent-R1)

**Current format:**
```
trajectory = {prompt, SDE_denoising_path, generated_image, reward}
```

**Proposed format:**
```
trajectory = {
    prompt,
    turns: [
        {
            agent_tokens: [token_ids],      # LLM-generated reasoning + rewritten prompt/tool call
            agent_logprobs: [logprobs],       # for policy gradient
            tool_call: {tool_name, params},   # which generation tool, with what (rewritten) params
            tool_output: {image/video/...},  # observation from the tool
            loss_mask: [0/1],                 # 1 for agent-generated tokens, 0 for observations
        },
        ...
    ],
    rewards: {
        plan: float,
        reflection: float,
        format: float,
        tool: float,
        result: float,
        pointwise: float,
        pairwise: float,
    },
    metadata: {num_turns, terminated, ...}
}
```

**Key design decisions:**
- **Loss masking**: only agent-generated tokens are trainable; tool outputs (images, videos) are masked, consistent with GenAgent's approach.
- **Variable length**: trajectories may have 1 to N_max turns. Trajectories with different turn counts represent different reasoning patterns.
- **Step-level representation**: each turn is an atomic RL transition, enabling step-level credit assignment (as proposed by Agent-R1).

### 6.2. Decoupled Agent–Tool Policy (train agent LLM, freeze diffusion) (model-level RL: GenAgent, VisionCreator, VisionCreator-R1, GEMS)

**Current:** only the diffusion model is the policy — Mode (0) single-stage RL.

**Proposed:** extend to Mode (2a) and acknowledge Mode (1) and Mode (2b) as not implemented (see Terminology table above for full definitions).

**Mode (2a) agentic LLM RL** is the primary new capability. The rollout becomes:

```
1. Agent LLM generates reasoning + initial prompt/tool call (token-level, with logprobs)
2. Tool (frozen diffusion model via vLLM-Omni) executes → image/video
3. Agent LLM generates judgment + (reflection or termination)
4. If reflection: go to step 1 with rewritten prompt based on reflection analysis
5. If termination: compute multi-dimensional rewards on full trajectory
6. GRPO update on agent LLM weights (loss masked to agent-generated tokens only, including rewritten prompts)
```

The RL optimizes the agent's **prompt rewriting ability** — how well it analyzes visual feedback and produces improved prompts for the frozen diffusion tool across iterations.

### 6.3. Multi-Dimensional Reward System (VisionCreator-R1, GenAgent)

**Current:** single reward per image (rule-based or model-based scorer).

**Proposed:** multi-dimensional reward per trajectory, inspired by VisionCreator-R1 and GenAgent:

| Reward | Type | Range | Computed by | What it measures |
|--------|------|-------|------------|------------------|
| **R_plan** | Continuous | [0, 1] | LLM evaluator | Requirement completeness, logical coherence, tool-goal matching |
| **R_reflect** | Continuous | [0, 1] | VLM judge vs. checkpoints | Whether reflection correctly identified and fixed errors |
| **R_format** | Continuous | [0, 1] | Rule-based | Tag presence, order, non-emptiness (trajectory-level minimum) |
| **R_tool** | Discrete | {0, 0.1, 0.8, 1.0} | Rule-based | Tool call success rate, self-correction rate |
| **R_result** | Binary | {0, 1} | Rule-based | Exact output count/type match |
| **R_pointwise** | Continuous | [0, 1] | MLLM judge | Final output satisfies all conditions |
| **R_pairwise** | Binary | {0, 1} | MLLM judge | Each iteration's output is better than the previous |
| **R_diffusion** | Continuous | [0, 1] | Diffusion reward model (CLIP / aesthetic / rule) | FlowGRPO terminal reward $R(\boldsymbol{s}_t, \boldsymbol{a}_t) \triangleq r(\boldsymbol{x}_0, \boldsymbol{c})$: score the frozen diffusion tool's final denoising output $\boldsymbol{x}_0$ against the rewritten condition $\boldsymbol{c}$ (`tool_call.params`), evaluated only at the final denoising step |

**Diffusion-native terminal reward (FlowGRPO):** the GenAgent-derived rewards above (`R_result`, `R_pointwise`, `R_pairwise`) are MLLM/rule-based image-quality signals and are unrelated to FlowGRPO. The `R_diffusion` row adds a *separate* reward sourced from FlowGRPO's terminal reward $R(\boldsymbol{s}_t, \boldsymbol{a}_t) \triangleq r(\boldsymbol{x}_0, \boldsymbol{c})$ — a score on the frozen tool's final denoising output $\boldsymbol{x}_0$ under the rewritten condition $\boldsymbol{c}$, evaluated only at the final denoising step. In FlowGRPO (Mode (0)) this reward trains the diffusion model (gradient → diffusion weights). In Mode (2a) the diffusion is frozen, so the same $r(\boldsymbol{x}_0, \boldsymbol{c})$ is reused as a scalar reward dimension for the agent LLM (gradient → agent only). The two reward families are complementary, not the same signal: GenAgent rewards judge instruction-following and monotonic improvement, while `R_diffusion` judges intrinsic image quality under the condition with the diffusion-native reward model. Feasibility is high — verl-omni already runs FlowGRPO recipes that compute $r(\boldsymbol{x}_0, \boldsymbol{c})$; this PR only reroutes that scalar into the agent's reward vector instead of backpropagating into the frozen tool.

**Total reward:** `R_total = (1/|W|) * sum(w_i * R_i)` where W is the active reward set (configurable per training run).

**Key insight — reflection-plan asymmetry:** R_plan has negligible trajectory variance (deterministic LLM judge) so GRPO optimizes it stably. R_reflect depends on post-reflection visual outcomes from the stochastic diffusion process, introducing dominant trajectory variance that collapses the signal-to-noise ratio. This asymmetry must be addressed via staged training (see §5).

**HTTP scorer integration:** the existing HTTP scorer service in verl-omni can serve as the bridge for external reward computation. An external evaluation system can expose an HTTP endpoint that receives generated outputs and returns multi-dimensional reward scores. This enables reward computation by systems that have domain-specific evaluation logic not available within verl-omni.

### 6.4. Trajectory Resampling (GenAgent)

During rollout, oversample G' trajectories per prompt (G' > G; following GenAgent, which generates G' = 12 rollouts per prompt and downsamples to G = 8), then uniformly resample G trajectories based on the number of interaction turns. This ensures the training batch contains diverse trajectory patterns (1-turn, 2-turn, 3-turn) rather than collapsing to a single interaction depth.

**Rationale (from GenAgent):** trajectories with different turn counts represent fundamentally different reasoning traces. Without resampling, the policy may collapse to always using the same number of turns, losing the ability to adapt reasoning depth to task complexity.

### 6.5. Reflection-Plan Asymmetry Mitigation (RPCO, VisionCreator-R1)

**Problem:** when training with both R_plan and R_reflect via GRPO, the reflection reward signal is dominated by diffusion stochasticity, not by the quality of the reflection action. The gradient estimator's trajectory variance is much larger than the action variance, making reflection optimization intractable in multi-turn settings.

**Proposed mitigation — decouple-then-fuse (from VisionCreator-R1's RPCO):**

```
Stage 1: Isolate reflection on single-image tasks
  - Single-image tasks have minimal planning demands and low stochasticity
  - Train with R_reflect as dominant signal
  - Result: strong-reflection checkpoint

Stage 2: Advantage-complementary SFT
  - Construct SFT corpus combining:
    - Reflection-strong single-image trajectories (from Stage 1)
    - Planning-strong multi-image trajectories (from a strong planner)
  - This provides balanced initialization

Stage 3: Multi-task RL co-optimization
  - Train on both single-image and multi-image tasks
  - R_plan continues to improve (stable signal)
  - R_reflect is preserved from Stage 1/2 initialization
  - Expanded reward set: {reflection, plan, format, tool, result}
```

**Implementation:** this requires verl-omni to support staged training with configurable reward sets per stage, and the ability to load checkpoints between stages.

## 7. Implementation Plan

Two PRs in strict dependency order. Each PR is independently reviewable and testable.

**Evaluation dataset (shared by both PRs) — UniCoT-Self-Reflection-6K:** the only native multi-turn reflection dataset, and it is built on BAGEL-7B-MoT — the same backbone family as the PR's agent (BAGEL-7B understanding path). This makes it the natural behavioral reference and eval oracle. A source adapter maps each UniCoT state `i` to an agentic transition:

- current image = `input_image[i]`
- reflection = `eval_summary[i]`, or a cleaned/length-limited `eval[i]`
- action = `continue` iff `output_image[i]` is non-null; otherwise `stop`
- edit = `edit[i]` for `continue`; empty for `stop`
- next image = `output_image[i]` for `continue`

`Action` is derived from the transition structure, not from natural-language phrases. Every non-terminal `output_image[i]` must hash-match `input_image[i+1]`. The adapter **fails closed** on: length mismatches across the parallel lists, empty reflections, `continue` with an empty Edit, contradictory terminal rows, and missing images — those records are dropped. After filtering, hold out an eval split (e.g., by `data_id` hash) used only for evaluation; the remaining records, plus Echo-4o / Midjourney prompts, seed online rollout. The eval split doubles as (a) reference trajectories and (b) a structural oracle that reuses UniCoT's own fail-closed invariants.

### 7.1. PR 1 — Multi-Turn Agentic Trajectory Format and Decoupled Agent–Tool Policy (train agent LLM, freeze diffusion) Rollout

**Depends on:** none (foundation)
**Assignee:** Frederick Hong (HaFred)

**Scope:** §1 (trajectory format) + §2 (Decoupled Agent–Tool Policy (train agent LLM, freeze diffusion), Mode (2a) only)

**Implementation path:** PR 1 uses the omni token-training path rather than the
diffusion trainer. The current entrypoint is `verl.trainer.main_ppo`; it may move
to `verl_omni.trainer.main_omni` once the upstream omni entrypoint is available.
The policy uses verl's standard `grpo` advantage estimator and an
`AgenticLLMFSDPEngine` derived from `FSDPEngineWithLMHead`. This preserves
micro-batching, global data-parallel loss normalization, mixed-precision
scaling, checkpointing, and the existing `PolicyLoss` integration. The
generation path is selectively frozen and used only as the image-generation
tool. PR 1 does not add an `agentic_grpo` estimator to the diffusion registry or
route token-level policy optimization through `main_diffusion`.

**Model recipe:** BAGEL-7B/Lance-3B (below we use BAGEL as examples):

| | BAGEL-based |
|---|---|
| **Agent LLM (trained)** | BAGEL-7B understanding path (`lm_head`, ViT, connector) |
| **Frozen diffusion tool** | BAGEL-7B generation path (diffusion, frozen weights, served via vLLM-Omni) |
| **Algorithm** | GRPO on understanding-path weights; generation path frozen |
| **Why this model** | BAGEL is a unified understanding+generation model — using its own understanding path as the trainable agent and its own generation path as the frozen tool is the most natural setup. The agent and tool share a backbone, enabling tighter coupling between reasoning and generation. Aligns with #295's `BagelForSFT` infrastructure. |
| **Dependency** | Depends on #295's `BagelForSFT` exposing the understanding path (`lm_head`, ViT, connector, clean-source VAE encoder), OR PR 1 implements this exposure itself. BAGEL's generation path is already available via existing `BagelForTraining` / FlowGRPO infrastructure. |
| **Trade-off** | Agent and tool share the same model — enables shared representations but requires careful selective freezing. Tighter coupling with #295. |
| **Initial size** | 7B (BAGEL-7B, understanding path trained + generation path frozen) |

BAGEL-7B understanding path as the trainable agent plus BAGEL-7B generation path as the frozen tool. The unified understanding+generation backbone gives the tightest coupling between the agent's reasoning and the tool's generation, and aligns directly with #295's `BagelForSFT` infrastructure. The only external blocker is #295's understanding-path exposure; PR 1 can implement that exposure itself if #295 is not yet merged.

**What this PR delivers:**
- The `AgenticTrajectory` dataclass with per-turn token/logprob/loss_mask/tool_call/tool_output structure
- An agentic rollout worker that executes the multi-turn loop: agent LLM generates reasoning + prompt → tool call dispatched to vLLM-Omni → observation returned → agent reflects and rewrites prompt → next tool call → repeat until termination or max turns
- Loss masking: only agent-generated tokens receive policy gradient; tool outputs are masked
- Agent LLM as a trainable FSDP/FSDP2 policy using `AgenticLLMFSDPEngine(FSDPEngineWithLMHead)` (Mode (2a)), optimizing the agent's prompt rewriting ability with the diffusion model configured as a frozen tool
- Configuration options: `max_turns`, `early_termination`, trajectory length limits, co-located vs decoupled GPU pools
- Data pipeline: load multi-turn trajectory datasets from parquet with nested turn structure
- Integration with existing async reward computation (reward computed while next rollout batch proceeds)
- Regression tests: trajectory format round-trip, loss mask correctness, rollout-train logprob consistency, single-turn backward compatibility (existing FlowGRPO must still work)

**What this PR does NOT include:**
- Multi-dimensional rewards (single scalar reward only, using existing reward infrastructure)
- Trajectory resampling
- RPCO staged training

**Acceptance criteria:**
- `python3 -m verl.trainer.main_ppo` (or `python3 -m verl_omni.trainer.main_omni` once available) with the standard `algorithm.adv_estimator=grpo` completes a full training step on a toy multi-turn dataset where the agent rewrites prompts across iterations
- Agent LLM weights update; diffusion model weights remain frozen
- Existing single-turn FlowGRPO training is unaffected

**PR 1 merge bar:** Acceptance-only. The three acceptance criteria above are the
required upstream merge gate. They must be demonstrated with deterministic
unit/regression tests and a one-step toy multi-turn training smoke test; a
production-scale Lance/BAGEL training run is not required.

**Post-merge evaluation plan (non-blocking for PR 1):** Evaluate on the UniCoT
held-out split by checking structural correctness (well-formed
`AgenticTrajectory`, frozen diffusion, loss masks on agent tokens only) and
behavioral alignment to UniCoT — action/termination accuracy versus the
`output_image[i]` non-null label, reflection–edit consistency reusing UniCoT's
own fail-closed checks, plus edit semantic similarity and reflection quality
versus `eval_summary`. Confirm the single scalar reward (e.g., `R_diffusion`) on
trained-agent rollouts exceeds both the SFT cold-start and random baselines
within a short GRPO run, while existing single-turn FlowGRPO eval is unchanged.

---

### 7.2. PR 2 — Multi-Dimensional Reward System and RPCO Staged Training

**Depends on:** PR 1 (trajectory format + agentic rollout)
**Assignee:** Frederick Hong (HaFred)

**Scope:** §3 (multi-dim rewards) + §4 (trajectory resampling) + §5 (RPCO staged training)

**What this PR delivers:**
- Seven reward scorers: `R_plan` (LLM evaluator), `R_reflect` (VLM judge vs. checkpoints), `R_format` (rule-based tag validation), `R_tool` (discrete success/correction rates), `R_result` (binary count match), `R_pointwise` (MLLM final-output judge), `R_pairwise` (shuffled monotonic-improvement check)
- Per-turn reward computation (not just final outcome) — each turn can receive its own reward signal
- Reward weight configuration: `R_total = (1/|W|) * sum(w_i * R_i)` with configurable active set W per training run
- Extension of the Reward Loop Manager to dispatch to multiple concurrent scorers (LLM judge + VLM judge + rule-based) with async overlap
- Extension of the HTTP scorer protocol to accept and return multi-dimensional reward responses (JSON array of `{dimension, score, metadata}`)
- Trajectory resampling: oversample G' > G trajectories per prompt, uniformly resample G by turn count
- RPCO staged training: configurable reward sets per stage, checkpoint loading between stages, the decouple-then-fuse pipeline (Stage 1: isolate reflection on single-image tasks → Stage 2: advantage-complementary SFT init → Stage 3: multi-task RL co-optimization)
- Trajectory analysis tools: reflection quality classification (under/good/over-reflection) for evaluation
- Tests: each scorer in isolation, multi-scorer concurrency, pairwise shuffle correctness, RPCO stage transitions, reward weight configuration

**What this PR does NOT include:**
- External trajectory capture infrastructure (trajectories are generated internally via PR 1's rollout; consuming trajectories from external sources is left for complementary projects)

**Acceptance criteria:**
- A full RPCO training run completes: Stage 1 (single-image reflection RL) → checkpoint → Stage 3 (multi-task co-optimization) with expanded reward set
- All 8 reward dimensions compute correctly on a multi-turn trajectory
- HTTP scorer returns multi-dimensional rewards from an external endpoint
- Trajectory resampling produces a balanced distribution of turn counts

**Evaluation plan (PR 2):** Verify each of the 8 reward dimensions against UniCoT references on the held-out split (e.g., `R_reflect` versus `eval_summary`, `R_pairwise` using UniCoT's `output_image[i]` as the improved reference, `R_diffusion` as the FlowGRPO terminal reward versus the final image) and confirm RPCO staging shifts the reflection-quality classifier toward "good" after Stage 1 without regressing `R_plan`. Check that resampling yields a uniform turn-count distribution (entropy) and that the GRPO-trained BAGEL-7B agent matches or exceeds UniCoT-7B-MoT on action accuracy, edit similarity, and final-image quality. All 8 dimensions must compute with a configurable weighted total and the HTTP scorer returning the multi-dimensional array.

Help wanted for the PR2 specification, if there are certain academic/industrial cases needed to be considerred.

## 8. Impacts

### 8.1. For model training

- **New capability:** verl-omni becomes the first framework to support multi-turn agentic RL (Mode (2a)) for omni-modal generation models, with a clear path to Mode (2b) agentic diffusion RL and Mode (1) multi-stage RL.
- **Better generation quality:** agentic approaches (iterative prompt rewriting, reflection) have been shown to improve generation quality by 14-24% across benchmarks — bringing these gains to any model trainable via verl-omni.

### 8.2. For external systems

- **Upstream training engine:** an external harness system can use verl-omni as its model training backend. The harness captures execution trajectories in the `AgenticTrajectory` format, sends them to verl-omni for RL training, and receives updated model weights in return. The harness's own prompt rewriting and skill optimization and verl-omni's model training run in complementary cycles.

- **External reward computation:** an external evaluation system can serve as an HTTP scorer to verl-omni's reward pipeline. This allows domain-specific evaluation logic (creative quality, aesthetic assessment, spatial reasoning, long-text rendering) to be provided by the system that has the most domain expertise, without requiring the evaluation code to be integrated into verl-omni.

### 8.3. For the ecosystem

- **Closes the agentic RL gap:** currently, agentic RL for multimodal generation is only demonstrated in research papers with custom training infrastructure. verl-omni would make this capability available as a production-grade, reusable framework.
- **Enables complementary integration:** verl-omni accepts multi-turn trajectories in a standardized format from any source, enabling complementary projects (e.g., external agent harnesses, trajectory capture infrastructure) to feed training-ready data without tight coupling.
- **Extends the verl ecosystem:** verl handles text agent RL, verl-omni handles multimodal model RL — this proposal extends verl-omni to also handle agentic RL for multimodal generation, completing the coverage.

## 9. References

1. GenAgent: Scaling Text-to-Image Generation via Agentic Multimodal Reasoning (arXiv:2601.18543)
2. VisionCreator: A Native Visual-Generation Agentic Model (arXiv:2603.02681)
3. VisionCreator-R1: A Reflection-Enhanced Native Visual-Generation Agentic Model (arXiv:2603.08812)
4. GEMS: Agent-Native Multimodal Generation with Memory and Skills (arXiv:2603.28088)
5. Polar: Agentic RL on Any Harness at Scale (arXiv:2605.24220)
6. Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning (arXiv:2511.14460)
7. DramaDirector: Geometry-Guided Short Drama Generation (arXiv:2606.24107)
