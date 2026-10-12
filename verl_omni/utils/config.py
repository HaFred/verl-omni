# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fail-fast validation shared by VeRL-Omni trainer entrypoints."""

from __future__ import annotations

from typing import Any


def _select(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for part in path.split("."):
        if value is None:
            return default
        if hasattr(value, "get"):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
    return default if value is None else value


def _validate_delta_sharded(config: Any) -> None:
    """Fail-fast gates for the ``omni_delta_sharded`` checkpoint engine backend (RFC #38).

    The delta path syncs full weights to standalone rollout replicas and diffs each
    rank's shard against the last export, so it is only defined for full-weight
    separate_async runs; anything else raises here instead of degrading silently.
    """
    backend = _select(config, "actor_rollout_ref.rollout.checkpoint_engine.backend")
    if backend == "delta_sharded":
        raise ValueError(
            "checkpoint_engine.backend='delta_sharded' is verl's sglang-only delta backend; verl's "
            "CheckpointEngineWorker rejects it for the vllm_omni rollout. Use backend='omni_delta_sharded': "
            "verl-omni registers the same DeltaShardedCheckpointEngine under that name and consumes its "
            "flushes in the vllm-omni worker extension."
        )
    if backend != "omni_delta_sharded":
        return

    mode = _select(config, "trainer.v1.trainer_mode")
    valid_modes = ("separate_async", "omni_separate_async")
    if mode not in valid_modes:
        raise ValueError(
            f"actor_rollout_ref.rollout.checkpoint_engine.backend='omni_delta_sharded' requires "
            f"trainer.v1.trainer_mode in {list(valid_modes)} (got {mode!r})."
        )
    if mode == "separate_async" and not _select(config, "trainer.use_v1", False):
        raise ValueError(
            "checkpoint_engine.backend='omni_delta_sharded' requires the v1 diffusion trainer "
            "(trainer.use_v1=true); the legacy trainer does not drive the delta sync flow."
        )

    model_key = "actor_rollout_ref.model"
    lora_rank = _select(config, f"{model_key}.lora.rank", 0) or 0
    legacy_lora_rank = _select(config, f"{model_key}.lora_rank", 0) or 0
    lora_adapter_path = _select(config, f"{model_key}.lora_adapter_path")
    if lora_rank > 0 or legacy_lora_rank > 0 or lora_adapter_path is not None:
        raise ValueError(
            "checkpoint_engine.backend='omni_delta_sharded' requires full-weight training; LoRA is "
            "refused in either merge mode because the shard export's names and values differ "
            "from the adapter (merge=false) and merged (merge=true) full exports. "
            f"Got {model_key}.lora.rank={lora_rank}, {model_key}.lora_rank={legacy_lora_rank}, "
            f"{model_key}.lora_adapter_path={lora_adapter_path}."
        )

    for qat_path in ("actor_rollout_ref.actor.fsdp_config.qat.enable", "actor_rollout_ref.actor.megatron.qat.enable"):
        if _select(config, qat_path, False):
            raise ValueError(
                f"checkpoint_engine.backend='omni_delta_sharded' does not support QAT ({qat_path}=true): "
                "the quantized, fused full export does not share coordinates with the shard export."
            )


def validate_config(config: Any) -> None:
    """Validate configuration values that otherwise trigger silent fallbacks."""
    if _select(config, "actor_rollout_ref.actor.enable_timestep_staging", False):
        sp_size = _select(config, "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 1)
        if sp_size != 1:
            raise ValueError("Timestep staging requires ulysses_sequence_parallel_size=1.")

    _validate_delta_sharded(config)

    if _select(config, "actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation", False):
        strategy = _select(config, "actor_rollout_ref.actor.strategy")
        if strategy not in ("fsdp", "fsdp2"):
            raise ValueError(
                "actor.use_no_sync_for_gradient_accumulation=true requires actor.strategy "
                f"fsdp or fsdp2, got {strategy!r}."
            )

    resume_mode = _select(config, "trainer.resume_mode")
    valid_resume_modes = ("disable", "auto", "resume_path")
    if resume_mode not in valid_resume_modes:
        raise ValueError(f"Unknown trainer.resume_mode={resume_mode!r}. Available options: {list(valid_resume_modes)}.")
    if resume_mode == "resume_path" and not _select(config, "trainer.resume_from_path"):
        raise ValueError("trainer.resume_from_path must be set when trainer.resume_mode='resume_path'.")

    total_steps = _select(config, "trainer.total_training_steps")
    if total_steps is not None:
        try:
            total_steps = int(total_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.") from exc
        if total_steps <= 0:
            raise ValueError("trainer.total_training_steps must be a positive integer or null.")

    _validate_dynamic_resource_scheduling(config)
    validate_bagel_corl_config(config)


def _validate_dynamic_resource_scheduling(config: Any) -> None:
    """Refuse the fully_async_policy scheduler on v1/v0 entrypoints.

    ``async_training.use_dynamic_resource_scheduling`` drives verl's
    ``DynamicResourceController`` (verl#6556) on the MessageQueue
    ``fully_async_policy`` stack. ``main_omni`` / ``main_diffusion`` /
    ``main_diffusion_v1`` never construct that controller, so the flag would
    be a silent no-op. The v1 analog is ``hybrid_rollout.enable_switch``.
    """
    if not _select(config, "async_training.use_dynamic_resource_scheduling", False):
        return
    raise ValueError(
        "async_training.use_dynamic_resource_scheduling=true is the fully_async_policy "
        "DynamicResourceController (verl#6556) and is not wired on main_omni / "
        "main_diffusion / main_diffusion_v1. For v1 separate_async, set "
        "trainer.v1.separate_async.hybrid_rollout.enable_switch=true instead."
    )


def validate_bagel_corl_config(config: Any) -> None:
    """Fail-closed Bagel Co-RL (Joint-Training) recipe checks (sibling N, seeds S, LoRA, no Qwen UND).

    In-episode ``J`` (UND turns) and ``K`` (GEN calls) are runtime with ``J >= K``;
    they are **not** ``rollout.n`` / ``gen_samples_per_call``. Do not require ``N == 2S``.
    """
    mode = _select(config, "trainer.v1.trainer_mode")
    if mode != "bagel_corl_sync":
        return

    n = _select(config, "actor_rollout_ref.rollout.n")
    s = _select(config, "actor_rollout_ref.rollout.agent.gen_samples_per_call")
    if n is None or s is None:
        raise ValueError(
            "bagel_corl_sync requires actor_rollout_ref.rollout.n (sibling episodes N) and "
            "actor_rollout_ref.rollout.agent.gen_samples_per_call (seeds per call S)"
        )
    try:
        n_int = int(n)
        s_int = int(s)
    except (TypeError, ValueError) as exc:
        raise ValueError("rollout.n and gen_samples_per_call must be integers") from exc
    if n_int < 1:
        raise ValueError(f"bagel_corl_sync requires rollout.n >= 1 (sibling episodes N), got n={n_int}")
    # One image per ``generate_image`` call. Sibling episodes (``rollout.n`` > 1) are the
    # GRPO group; a 4-seed fan-out is not required for a within-group advantage.
    if s_int < 1:
        raise ValueError(f"bagel_corl_sync requires gen_samples_per_call >= 1 (images per call), got S={s_int}")

    max_passes = _select(config, "actor_rollout_ref.rollout.agent.max_generate_passes", default=3)
    if int(max_passes) < 1 or int(max_passes) > 3:
        raise ValueError(
            "bagel_corl_sync requires max_generate_passes in 1..3 "
            f"(in-episode K cap, matching the agentic loop), got {max_passes}"
        )

    lora_rank = _select(config, "actor_rollout_ref.model.lora_rank") or _select(
        config, "actor_rollout_ref.model.lora.rank", default=0
    )
    if int(lora_rank or 0) <= 0:
        raise ValueError("bagel_corl_sync PR1 requires LoRA (lora_rank > 0)")

    und_model = str(_select(config, "actor_rollout_ref.model.path") or "")
    if "qwen3-vl" in und_model.lower() or "Qwen3-VL" in und_model:
        raise ValueError("bagel_corl_sync forbids Qwen3-VL as the UND policy; use the published Bagel checkpoint")

    # One vLLM-Omni replica is AR xor Diffusion (strategy chosen at server init).
    # bagel_corl_deploy.yaml → bagel_single_stage → DiffusionStrategy: GEN works;
    # UND decode with AR sampling params hits "num_inference_steps must be set".
    # output_mode=ar alone would break GEN traj. Dual-role serving is the spike.
    output_mode = str(
        _select(config, "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode", default="diffusion")
        or "diffusion"
    )
    if output_mode == "ar":
        raise ValueError(
            "bagel_corl_sync refuses output_mode=ar alone: GEN FlowGRPO needs DiffusionStrategy. "
            "Need dual-role UND AR + GEN diffusion on Bagel (not Qwen)."
        )
    und_ready = _select(config, "actor_rollout_ref.rollout.agent.und_ar_serving_ready", default=False)
    if not bool(und_ready):
        raise ValueError(
            "bagel_corl_sync: dual-role UND AR serving is not ready. "
            "bagel_single_stage is GEN-only (DiffusionStrategy); UND _und_decode then fails with "
            "'num_inference_steps must be set for RL rollouts' and leaves TQ empty. "
            "Prove Hermes generate_image via spike_und_hermes.py / dual-role replica, then set "
            "actor_rollout_ref.rollout.agent.und_ar_serving_ready=True. "
            "Do not fall back to Qwen3-VL for UND."
        )
    und_deploy = _select(config, "actor_rollout_ref.rollout.agent.und_deploy_config")
    if not und_deploy:
        raise ValueError(
            "bagel_corl_sync with und_ar_serving_ready=True requires "
            "actor_rollout_ref.rollout.agent.und_deploy_config "
            "(AR bagel_think yaml). GEN keeps engine_kwargs.vllm_omni.deploy_config "
            "(bagel_single_stage)."
        )

    # RFC §4.4.2b: routing affinity exists *only* to pin an S-group to the replica
    # holding its conditioning. Turning it on with the cache off is incoherent —
    # the pins buy nothing and the run silently reports no reuse.
    pec_affinity = bool(
        _select(config, "actor_rollout_ref.rollout.enable_prompt_embed_cache_routing_affinity", default=False)
    )
    pec_enabled = bool(_select(config, "actor_rollout_ref.rollout.enable_prompt_embed_cache", default=False))
    if pec_affinity and not pec_enabled:
        raise ValueError(
            "bagel_corl_sync: enable_prompt_embed_cache_routing_affinity=True requires "
            "enable_prompt_embed_cache=True (RFC §4.4.2b) — affinity without a cache pins "
            "S-groups to replicas that hold no conditioning entry."
        )

    # RFC §4.4 launch contract: an episode's context is bounded by
    # ``rollout.prompt_length + rollout.response_length`` -- the shape every trajectory
    # is padded to, and the budget ``run_serial_episode`` stops the UND loop at -- while
    # the AR engine's ``max_model_len`` is derived from ``data.max_*``. If the two pairs
    # disagree, episodes are built for one width and decoded/padded against another, and
    # ``_pad_token_ids`` does not truncate: an over-long prompt or response comes back as
    # a *longer* tensor, so the batch silently loses its uniform shape.
    rollout_prompt = _select(config, "actor_rollout_ref.rollout.prompt_length")
    rollout_response = _select(config, "actor_rollout_ref.rollout.response_length")
    data_prompt = _select(config, "data.max_prompt_length")
    data_response = _select(config, "data.max_response_length")
    if None not in (rollout_prompt, rollout_response, data_prompt, data_response):
        length_mismatches = []
        if int(rollout_prompt) != int(data_prompt):
            length_mismatches.append(
                f"rollout.prompt_length={int(rollout_prompt)} vs data.max_prompt_length={int(data_prompt)}"
            )
        if int(rollout_response) != int(data_response):
            length_mismatches.append(
                f"rollout.response_length={int(rollout_response)} vs data.max_response_length={int(data_response)}"
            )
        if length_mismatches:
            raise ValueError(
                "bagel_corl_sync length contract (RFC §4.4) violated:\n  - "
                + "\n  - ".join(length_mismatches)
                + "\n  The agent loop decodes prompt+response, the trainer pads every trajectory to the "
                "rollout pair, and the UND AR engine's max_model_len is sized from the data pair. "
                "Pin actor_rollout_ref.rollout.prompt_length/response_length to "
                "data.max_prompt_length/data.max_response_length."
            )

    # RFC §4.8.4: enforce the launch divisibility the batch mechanics assume.
    failures = _bagel_corl_divisibility_failures(config, n=n_int)
    if failures:
        raise ValueError("bagel_corl_sync launch divisibility (RFC §4.8.4) violated:\n  - " + "\n  - ".join(failures))


def _bagel_corl_divisibility_failures(config: Any, *, n: int) -> list[str]:
    """RFC §4.8.4 launch divisibility, as human-readable failures.

    Split out from :func:`validate_bagel_corl_config` so the arithmetic is testable
    without assembling a whole recipe. ``pool`` is the width of the actor+GEN
    hybrid placement group. ``N`` siblings turn one task into ``N`` episodes, so
    divisibility has to be checked on the **episode** count (``mini x N``), not on
    the prompt count the knobs are written in — a ragged split is what silently
    drops or duplicates episodes rather than failing.

    Returns an empty list when everything divides, so the caller can decide the
    message; values that are absent are skipped rather than defaulted, since
    inventing a pool size here would defeat the check.
    """

    def _int(path: str, default: Any = None) -> int | None:
        raw = _select(config, path, default=default)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    pool = _int("trainer.n_gpus_per_node", default=1)
    train_bsz = _int("data.train_batch_size")
    mini = _int("actor_rollout_ref.actor.ppo_mini_batch_size")
    micro = _int("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu")
    rollout_tp = _int("actor_rollout_ref.rollout.tensor_model_parallel_size")
    reward_tp = _int("reward.reward_model.rollout.tensor_model_parallel_size")

    failures: list[str] = []
    if pool is None or pool <= 0:
        return ["trainer.n_gpus_per_node must be a positive integer to check divisibility."]

    if train_bsz is not None and mini:
        if train_bsz % mini != 0:
            failures.append(
                f"data.train_batch_size={train_bsz} % actor_rollout_ref.actor.ppo_mini_batch_size={mini} != 0"
            )
    if mini:
        episodes_per_update = mini * max(1, n)
        if episodes_per_update % pool != 0:
            failures.append(
                f"(ppo_mini_batch_size={mini} x rollout.n={max(1, n)}) = {episodes_per_update} "
                f"% trainer.n_gpus_per_node={pool} != 0"
            )
        elif micro and micro > 0 and (episodes_per_update // pool) % micro != 0:
            failures.append(
                f"(ppo_mini_batch_size={mini} x rollout.n={max(1, n)} / trainer.n_gpus_per_node={pool}) = "
                f"{episodes_per_update // pool} % ppo_micro_batch_size_per_gpu={micro} != 0"
            )
    for label, tp in (
        ("actor_rollout_ref.rollout.tensor_model_parallel_size (ROLLOUT_TP)", rollout_tp),
        ("reward.reward_model.rollout.tensor_model_parallel_size (REWARD_TP)", reward_tp),
    ):
        if tp and tp > 0 and pool % tp != 0:
            failures.append(f"trainer.n_gpus_per_node={pool} % {label}={tp} != 0")
    return failures
