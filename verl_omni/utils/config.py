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


def validate_config(config: Any) -> None:
    """Validate configuration values that otherwise trigger silent fallbacks."""
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

    validate_bagel_corl_config(config)


def validate_bagel_corl_config(config: Any) -> None:
    """Fail-closed Bagel Co-RL recipe checks (sibling N, seeds S, LoRA, no Qwen UND).

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
    if s_int < 2:
        raise ValueError(
            f"bagel_corl_sync requires gen_samples_per_call >= 2 (FlowGRPO seeds S), got S={s_int}"
        )

    max_passes = _select(config, "actor_rollout_ref.rollout.agent.max_generate_passes", default=1)
    if int(max_passes) != 1:
        raise ValueError("PR1 bagel_corl_sync requires max_generate_passes=1 (bounds in-episode K)")

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
