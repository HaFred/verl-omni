# Shared Hydra CLI overrides for Mode (2a) Lance agentic GRPO.
# Sourced by run_lance_agentic_grpo.sh and tests/special_e2e/run_agentic_grpo_lance.sh.
# Uses stock verl ``ppo_trainer`` defaults (FlowGRPO-style: no custom recipe YAML).
#
# Includes:
#   1) PR1 Mode (2a) essentials (RFC §7.1)
#   2) Unified toy / not-OOM sizing for Lance_3B_hf_und on ~2 GPUs
# Override any field via launch-script CLI / "$@" as needed.

AGENTIC_GRPO_OVERRIDES=(
  # ── Mode (2a) essentials ──────────────────────────────────────────
  algorithm.adv_estimator=grpo

  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.calculate_log_probs=true

  # Distinct embed/lm_head weights; tying breaks FSDP lm_head unshard on Lance und.
  +actor_rollout_ref.model.override_config.tie_word_embeddings=false
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa

  actor_rollout_ref.rollout.multi_turn.enable=true
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
  actor_rollout_ref.rollout.multi_turn.max_user_turns=5
  actor_rollout_ref.rollout.multi_turn.function_tool_path=verl_omni/agent_loop/diffusion_tool.py

  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path=null

  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_reward
  reward.custom_reward_function.name=compute_score

  # ── Toy / not-OOM sizing (Lance-3B und, ~2 GPUs) ──────────────────
  data.train_batch_size=4
  data.max_prompt_length=512
  data.max_response_length=1024
  data.filter_overlong_prompts=true
  data.truncation=left

  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=32
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
  actor_rollout_ref.model.enable_gradient_checkpointing=true

  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true

  actor_rollout_ref.rollout.n=2
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2

  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true

  trainer.val_before_train=false
  trainer.nnodes=1
)
