# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bagel UND+GEN Co-RL (Joint-Training) trainer: one post-gather composite update + weight publish."""

from __future__ import annotations

import logging
import os
from dataclasses import fields
from typing import Any

from omegaconf import OmegaConf, open_dict
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.agent_loop.bagel_corl_lib import aggregate_episode_metrics
from verl_omni.trainer.omni.bagel_corl_diff_v1 import DiffusionV1GenLane
from verl_omni.trainer.omni.bagel_corl_gen_adv import apply_gen_flowgrpo_advantage, build_gen_flowgrpo_proto
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.config.diffusion import DiffusionRolloutConfig, DiffusionSamplingConfig

logger = logging.getLogger(__name__)

# Bagel Co-RL (Joint-Training) agent knobs with no code default (RFC §5 "knob source of truth"):
# ``actor_rollout_ref.rollout.agent.*`` is the single SoT, and
# ``verl_omni/utils/config.py`` validates S >= 2 / max_generate_passes == 1 at
# launch. The trainer must not invent values, or a missing/bad recipe knob would
# be masked from that validation.
_BAGEL_AGENT_REQUIRED_KNOBS = ("gen_samples_per_call", "max_generate_passes", "max_und_turns")

# Fields accepted by DiffusionModelConfig (plus _target_). Omni YAML keys outside this
# set must be stripped before instantiate/omega_conf_to_dataclass.
_DIFFUSION_MODEL_KEYS = {
    "path",
    "architecture",
    "transformer_config",
    "algorithm",
    "local_path",
    "tokenizer_path",
    "local_tokenizer_path",
    "hf_config",
    "model_type",
    "composite_mode",
    "load_tokenizer",
    "tokenizer",
    "processor",
    "extra_tokenizers",
    "extra_tokenizer_map",
    "use_shm",
    "trust_remote_code",
    "custom_chat_template",
    "external_lib",
    "enable_gradient_checkpointing",
    "attn_backend",
    "lora_rank",
    "lora_alpha",
    "lora_init_weights",
    "lr_gen",
    "target_modules",
    "target_parameters",
    "exclude_modules",
    "lora",
    "lora_adapter_path",
    "policy_state_adapters",
    "lora_dtype",
    "mtp",
    "pipeline",
    "algo",
    "fsdp_layer_prefixes",
    "config_path",
    "transformer_subfolder",
}


def _bagel_role_histogram(tags) -> dict[str, int]:
    """Count a ``KVBatchMeta`` row-tag list by ``bagel_role`` (+ padding), for step diagnostics.

    A GEN stale in a sampled batch is invisible in the training metrics but fatal a few frames
    later (``assert len(output) == len(batch)``), so the batch's own composition is worth a line.
    """
    counts: dict[str, int] = {}
    for tag in tags or []:
        if not isinstance(tag, dict):
            continue
        role = str(tag.get("bagel_role") or "untagged")
        if tag.get("is_padding"):
            role += ":pad"
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def _normalize_tq_kv_get_result(data, keys: list[str]) -> dict[str, dict]:
    """Normalize ``transfer_queue.kv_batch_get`` output into ``{key: {"fields": row}}``.

    ``kv_batch_get`` returns a columnar TensorDict (one entry per requested key, in
    request order). It is **not** a plain dict. Fail-loud on any shape we do not
    recognize so a silent ``{}`` can never drop the GEN lane.
    """
    if data is None:
        return {}

    # Columnar TensorDict (the canonical TQ shape): each field is a stacked column.
    td = data
    if hasattr(td, "items") and hasattr(td, "get") and not isinstance(td, (dict, list)):
        # Iterate ``td[field]`` (not ``td.items()``): TensorDict ``__getitem__`` already
        # unwraps non-tensor columns to plain lists, while ``.items()`` yields wrapped
        # NonTensorData entries whose ``[i]`` does not index the underlying data.
        if hasattr(td, "keys") and callable(getattr(td, "keys", None)):
            try:
                field_names = [str(k) for k in td.keys()]
                out: dict[str, dict] = {}
                for i, key in enumerate(keys):
                    row: dict[str, Any] = {}
                    for field_name in field_names:
                        row[field_name] = _index_column(td[field_name], i)
                    out[str(key)] = {"fields": row}
                if out:
                    return out
            except (TypeError, KeyError, IndexError):
                pass

    if isinstance(data, dict):
        # Keyed dict {key: row} — only if every key maps back to a requested key.
        if all(k in keys for k in data.keys()) and keys:
            out = {}
            for key in keys:
                row = data.get(key)
                if row is None:
                    continue
                if isinstance(row, dict) and "fields" in row:
                    # Already row-shaped ({"fields": {...}}): pass through. Wrapping
                    # again would nest "fields" twice and every downstream
                    # ``row["<column>"]`` lookup (gen_group_uid, rollout_log_probs, …)
                    # would KeyError, silently dropping the whole GEN lane.
                    out[str(key)] = row
                else:
                    out[str(key)] = {"fields": row if isinstance(row, dict) else {"value": row}}
            return out
        # Columnar dict {field: [values...]} optionally carrying a "keys" column.
        got_keys = data.get("keys")
        if got_keys is None:
            got_keys = keys
        got_keys = [str(k) for k in got_keys]
        out = {}
        for i, key in enumerate(got_keys):
            row = {}
            for field_name, col in data.items():
                if field_name == "keys":
                    continue
                try:
                    row[field_name] = col[i]
                except (TypeError, IndexError, KeyError):
                    row[field_name] = None
            out[key] = {"fields": row}
        return out

    if isinstance(data, (list, tuple)):
        # Positional list: zip with requested keys.
        out = {}
        for i, (key, row) in enumerate(zip(keys, data)):
            if row is None:
                continue
            out[str(key)] = {"fields": row if isinstance(row, dict) else {"value": row}}
        return out

    raise ValueError(
        f"bagel_corl: unrecognized transfer_queue.kv_batch_get return type {type(data)!r}; "
        "refusing to silently drop the GEN lane"
    )


def _index_column(col, i: int):
    """Index a stacked TQ column (Tensor / NonTensorStack / list) at position ``i``.

    ``NonTensorStack.__getitem__`` yields a ``NonTensorData`` wrapper — unwrap it so
    field dicts carry plain values (a wrapper would silently break
    ``split_und_gen_metas`` iteration and ``build_gen_flowgrpo_proto``). Tensors are
    returned as-is (never touch ``Tensor.data``).
    """
    if col is None:
        return None
    try:
        value = col[i]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        from tensordict import NonTensorData
    except ImportError:  # pragma: no cover - tensordict is a hard dependency here
        return value
    if isinstance(value, NonTensorData):
        return value.data
    return value


def _actor_merges_lora(model) -> bool:
    """True when the recipe asks the *actor* to merge LoRA into the base weights.

    ``actor_rollout_ref.model.lora.merge=True`` (required on vllm-omni >= 0.24, see
    ``run_agentic_bagel_rpco_lora.sh``) makes the trainer publish **merged** full weights:
    ``EngineWorker._update_weights`` reports ``peft_config=None``
    (``self.peft_merge = model_config.lora.get("merge", False)`` in
    ``verl/workers/engine_workers.py``), so every rollout replica takes the standard
    full-weight update and **no adapter is ever added engine-side**.

    Tolerates both shapes this codebase hands around: the resolved ``DictConfig``
    (``.get``) and the converted dataclass (attribute access).
    """
    lora_cfg = model.get("lora") if hasattr(model, "get") else getattr(model, "lora", None)
    if lora_cfg is None:
        return False
    if hasattr(lora_cfg, "get"):
        return bool(lora_cfg.get("merge", False))
    return bool(getattr(lora_cfg, "merge", False))


@register_trainer("bagel_corl_sync")
class OmniBagelCoRLTrainerSync(OmniPPOTrainerSync):
    """Synchronous Bagel Co-RL (Joint-Training): serial J-episode gather, then one UND+GEN optimizer step.

    Weight sync runs once in ``on_step_end`` (inherited) to every replica. No mid-episode sync.
    Replay uses the sync ``ReplayBuffer`` even though ``trainer_mode`` is not the string ``sync``.

    ``main_omni`` defaults to omni ``OmniModelConfig`` / ``RolloutConfig``. Bagel Co-RL (Joint-Training) needs
    the diffusion model + rollout surfaces (``algorithm``, ``pipeline``, ``algo``,
    ``rollout_adapter``) for FlowGRPO GEN / weight sync while keeping the omni actor for UND
    ``ppo_loss``. Rewrite those configs and inject ``diffusion_loss`` before tokenizer /
    worker init.

    Do **not** only add ``rollout_adapter`` onto omni ``RolloutConfig``: ``init_model``
    instantiates via Hydra ``_target_``, and verl's ``RolloutConfig`` rejects that kwarg.

    Parent is intentional PPO V1 (``OmniPPOTrainerSync`` / ``TaskRunnerV1`` / token GRPO),
    not diffusion ``PolicyGradientDiffusionTrainerV1Sync.fit()`` and not OCR's legacy
    ``PolicyGradientRayTrainer``. GEN lane binds
    ``PolicyGradientDiffusionTrainerV1._compute_old_log_prob`` /
    ``_compute_advantage`` onto the same ``actor_rollout_wg``. Rewrite +
    ``ActorRolloutRefWorker`` unlock GEN *loss* on ``PPODiffusersFSDPEngine``.
    Composite ``forward_backward_batch`` (``bagel_corl_composite``) runs UND
    ``BagelForCoRL.compute_und_log_prob`` then GEN timestep FlowGRPO in one
    ``train_batch`` / one ``optimizer.step``. Same training step: UND token GRPO via
    ``super()._compute_advantage`` (TQ ``advantages``); GEN FlowGRPO via the bound
    diffusion V1 ``_compute_advantage`` into ``extra_info["bagel_corl_gen"]`` only
    (grouped by ``gen_group_uid``), including traj ``all_latents`` when stashed.
    Estimators are decoupled: UND uses ``algorithm.adv_estimator`` (omni default
    ``grpo``); GEN uses ``model.algorithm`` / ``flow_grpo``. Incomplete or traj-less
    GEN views skip GEN only; UND still updates.
    Inherited Omni ``resume_generation_replicas`` is a poor fit for ``bagel_single_stage``
    (diffusion V1 sleeps replicas instead) — revisit after GEN serving is live.
    """

    def _rewrite_bagel_corl_configs(self) -> None:
        """Retarget model/rollout to diffusion configs; inject actor ``diffusion_loss``."""
        raw = OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True) or {}
        filtered = {k: v for k, v in raw.items() if k in _DIFFUSION_MODEL_KEYS}
        filtered["_target_"] = "verl_omni.workers.config.diffusion.DiffusionModelConfig"
        filtered.setdefault("algorithm", "flow_grpo")
        filtered["model_type"] = "diffusion_model"
        filtered.setdefault("architecture", "OmniBagelForConditionalGeneration")
        filtered.setdefault("composite_mode", "bagel_corl")
        # UniGRPO per-expert LRs: the GEN (*_moe_gen) optimizer group runs at its
        # own LR; UND keeps the base actor.optim.lr. Recipe-visible, configurable.
        filtered.setdefault("lr_gen", 3e-5)
        filtered.setdefault("trust_remote_code", True)
        # Bagel Co-RL is defined by the *disjoint* UND/GEN LoRA split (RFC §4.0.2), so the
        # target list must be explicit. The omni model yaml default is the string
        # ``all-linear``; ``validate_disjoint_lora_targets`` would iterate it character by
        # character and abort model construction with the useless
        # "unknown names: ['-', 'a', 'e', 'i', 'l', 'n', 'r']". Fail loud here instead,
        # where the recipe author can act on it.
        lora_targets = filtered.get("target_modules")
        if not isinstance(lora_targets, (list, tuple)):
            raise ValueError(
                "bagel_corl_sync requires actor_rollout_ref.model.target_modules as an explicit "
                f"list of UND/GEN LoRA module names, got {lora_targets!r} (RFC §4.0.2 dual-lane LoRA)"
            )

        # Strip omni-only rollout keys (do_sample, over_sample_rate, …) before Hydra instantiate.
        # Keep agent / multi_turn / trace / response_length: AgentLoopWorker requires them.
        rollout_allowed = {f.name for f in fields(DiffusionRolloutConfig)}
        rollout_raw = OmegaConf.to_container(self.config.actor_rollout_ref.rollout, resolve=True) or {}
        rollout_filtered = {k: v for k, v in rollout_raw.items() if k in rollout_allowed}
        rollout_filtered["_target_"] = "verl_omni.workers.config.diffusion.DiffusionRolloutConfig"
        rollout_filtered.setdefault("rollout_adapter", "default")
        # Omni YAML can leave free_cache_engine=null; falsy disables AsyncOmni.sleep and the
        # first update_weights then OOMs (actor FSDP summon ~50GiB + live Omni ~30GiB).
        rollout_filtered["free_cache_engine"] = True
        rollout_filtered.setdefault("enable_sleep_mode", True)
        agent = dict(rollout_filtered.get("agent") or {})
        agent["_target_"] = "verl_omni.workers.config.omni.BagelCorlAgentLoopConfig"
        # RFC §5: ``gen_samples_per_call`` / ``max_generate_passes`` /
        # ``max_und_turns`` have a single SoT of
        # ``actor_rollout_ref.rollout.agent.*`` and **no code fallback** — the recipe
        # (or yaml) must set them, and ``verl_omni/utils/config.py`` validates S >= 2
        # and ``max_generate_passes == 1`` at launch. Inventing values here would
        # silently mask a missing/bad recipe knob from that validation.
        missing = [k for k in _BAGEL_AGENT_REQUIRED_KNOBS if agent.get(k) is None]
        if missing:
            raise ValueError(
                "bagel_corl_sync requires actor_rollout_ref.rollout.agent."
                f"{', '.join(missing)} (RFC §5 knob SoT, no code default)"
            )
        agent.setdefault(
            "agent_loop_manager_class",
            "verl_omni.agent_loop.bagel_corl_tq.BagelCorlAgentLoopManagerTQ",
        )
        agent.setdefault("default_agent_loop", "bagel_multiturn_agent")
        agent.setdefault("und_ar_serving_ready", False)
        rollout_filtered["agent"] = agent
        # Live GEN traj stash requires FlowGRPO SDE + logprobs (refuse ODE soft-skip).
        rollout_filtered["calculate_log_probs"] = True
        algo = dict(rollout_filtered.get("algo") or {})
        algo.setdefault("noise_level", 0.7)
        algo.setdefault("sde_window_size", 2)
        algo.setdefault("sde_window_range", [0, 7])
        algo.setdefault("sde_type", "sde")
        rollout_filtered["algo"] = algo
        pipeline = dict(rollout_filtered.get("pipeline") or {})
        pipeline.setdefault("num_inference_steps", 10)
        # UniGRPO (arXiv:2603.23500): CFG doubles per-step evaluations and branches
        # the rollout graph — prohibitive for multi-turn episodes. Joint-Training runs
        # CFG-free on BOTH sides; the recipe's eval block re-enables CFG if desired.
        pipeline["cfg_text_scale"] = 1.0
        # Neither the omni rollout config nor its yaml declares a ``pipeline`` node —
        # the recipe reaches it with ``+actor_rollout_ref.rollout.pipeline.*`` — so the
        # dict built here must carry ``_target_`` exactly like ``diffusion_rollout.yaml``
        # does. Without it Hydra hands a plain mapping to the typed
        # ``pipeline: DiffusionPipelineConfig`` field, and every ``pipeline.<attr>``
        # read (``diffusion_agent_loop``, ``VisualRewardManager``) breaks afterwards.
        pipeline["_target_"] = "verl_omni.workers.config.diffusion.DiffusionPipelineConfig"
        rollout_filtered["pipeline"] = pipeline

        # ``val_kwargs`` needs the same retarget for a different reason: unlike ``pipeline``,
        # the omni schema *does* declare this node -- but with the AR ``_target_``
        # ``verl.workers.config.SamplingConfig``, while ``DiffusionRolloutConfig`` types the
        # field as ``DiffusionSamplingConfig``. SamplingConfig has no ``pipeline``/``algo``
        # subtree, so the recipe's ``+…val_kwargs.pipeline.num_inference_steps=50`` (and the
        # ``algo.noise_level`` beside it) compose fine as new keys and then kill the first
        # ``init_model`` on every actor rank:
        #   InstantiationException: Error in call to target
        #   'verl.workers.config.rollout.SamplingConfig':
        #   TypeError("SamplingConfig.__init__() got an unexpected keyword argument 'pipeline'")
        #   full_key: actor_rollout_ref.rollout.val_kwargs
        # ``do_sample`` is AR-only and has to go with the target, so filter as well as retarget
        # -- exactly what the parent node above does. This is also what makes the validate
        # branch of ``composite_agent_loop`` work at all: it reads
        # ``config.val_kwargs.pipeline`` / ``.algo`` / ``.seed``, none of which exist on the
        # AR SamplingConfig.
        val_allowed = {f.name for f in fields(DiffusionSamplingConfig)}
        val_kwargs = {
            k: v for k, v in (rollout_filtered.get("val_kwargs") or {}).items() if k in val_allowed and k != "_target_"
        }
        val_kwargs["_target_"] = "verl_omni.workers.config.diffusion.DiffusionSamplingConfig"
        rollout_filtered["val_kwargs"] = val_kwargs

        # ``cfg_text_scale`` is a transition-kernel knob, not a style knob: the
        # GEN importance ratio is only unbiased when training and rollout use the
        # SAME value (RFC §4.8.5 / §5). The training side reads
        # ``model.pipeline.cfg_text_scale`` via
        # ``BagelDiffusion._get_cfg_params``, which falls back to
        # ``BAGEL_FLOWGRPO_CFG_DEFAULTS`` = 4.0 — so leaving it unset here silently
        # trained with CFG while rollout ran CFG-free.
        #
        # ``omni_model.yaml`` declares no ``pipeline`` node at all, so anything set
        # here has to be the whole node, not a patch onto an existing one. Mirror what
        # ``diffusion_model.yaml`` does for the stock diffusion path and make the model
        # node the rollout pipeline itself (rollout is the knob SoT, RFC §5): that keeps
        # ``num_inference_steps`` / ``height`` / ``width`` — used by the training adapter
        # for the sigma schedule and latent position ids — in lockstep with rollout, and
        # carries the ``_target_`` that turns this into a real ``DiffusionPipelineConfig``.
        # A bare ``{"cfg_text_scale": 1.0}`` dict here would instead replace the dataclass
        # default and then fail every ``model_config.pipeline.<attr>`` read.
        filtered["pipeline"] = dict(pipeline)

        with open_dict(self.config):
            self.config.actor_rollout_ref.model = OmegaConf.create(filtered)
            self.config.actor_rollout_ref.rollout = OmegaConf.create(rollout_filtered)
            actor = self.config.actor_rollout_ref.actor
            if actor.get("diffusion_loss") is None:
                actor.diffusion_loss = OmegaConf.create(
                    {
                        "_target_": "verl_omni.workers.config.diffusion.DiffusionLossConfig",
                        "loss_mode": filtered.get("algorithm", "flow_grpo"),
                    }
                )
            elif actor.diffusion_loss.get("loss_mode") is None:
                actor.diffusion_loss.loss_mode = filtered.get("algorithm", "flow_grpo")
            # verl AgentLoopBase always reads data.continuous_token (struct). Omni data
            # schemas may omit it; missing key aborts every episode → empty TQ →
            # "no materializable trajectories".
            data = self.config.get("data")
            if data is None:
                self.config.data = data = OmegaConf.create({})
            if data.get("continuous_token") is None:
                data.continuous_token = OmegaConf.create({"enable": False, "model_family": "auto"})
            elif data.continuous_token.get("enable") is None:
                data.continuous_token.enable = False

        # RFC §5: ``cfg_text_scale`` is a transition-kernel knob whose single SoT is
        # ``model.pipeline`` AND ``rollout.pipeline``. A mismatch is NON-CONFORMANT —
        # the GEN importance ratio would be computed under a different kernel than
        # the rollout that produced the trajectory. Assert it at launch, not later.
        model_cfg = float(self.config.actor_rollout_ref.model.pipeline.cfg_text_scale)
        rollout_cfg = float(self.config.actor_rollout_ref.rollout.pipeline.cfg_text_scale)
        if model_cfg != rollout_cfg:
            raise ValueError(
                "bagel_corl_sync CFG parity violated: "
                f"model.pipeline.cfg_text_scale={model_cfg} != "
                f"rollout.pipeline.cfg_text_scale={rollout_cfg}. Training and rollout "
                "must share one transition kernel (RFC §5); set both to 1 (CFG-free) "
                "as §4.8.5 prescribes."
            )

    def _init_tokenizer(self):
        self._rewrite_bagel_corl_configs()
        model_config: DiffusionModelConfig = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.model, DiffusionModelConfig
        )
        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor

    def _build_replay_buffer(self):
        original_mode = self.trainer_mode
        self.trainer_mode = "sync"
        try:
            return super()._build_replay_buffer()
        finally:
            self.trainer_mode = original_mode

    def _init_resource_pool_mgr(self):
        """Use VeRL-Omni ``ActorRolloutRefWorker`` so Bagel FSDP + composite loss are available."""
        import ray

        super()._init_resource_pool_mgr()
        from verl_omni.workers.engine_workers import ActorRolloutRefWorker as OmniActorRolloutRefWorker

        remote_cls = ray.remote(OmniActorRolloutRefWorker)
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = remote_cls

    def _expected_s(self) -> int:
        """Seeds per ``generate_image`` call (RFC §5 knob; no silent default).

        The recipe sets ``agent.gen_samples_per_call``; a fallback here would make
        the dual-lane gate accept a group size nobody configured, so fail loud.
        """
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        raw = agent.get("gen_samples_per_call")
        if raw is None:
            raise ValueError(
                "bagel_corl_sync requires actor_rollout_ref.rollout.agent.gen_samples_per_call; "
                "the dual-lane GEN seed gate cannot infer S"
            )
        return int(raw)

    def _gen_adv_estimator(self) -> str:
        """GEN FlowGRPO estimator — never reuse ``algorithm.adv_estimator`` (UND token GRPO)."""
        model = self.config.actor_rollout_ref.model
        algo = self.config.algorithm
        # Prefer diffusion model.algorithm / explicit gen override; omni default adv_estimator is token grpo.
        for candidate in (
            algo.get("gen_adv_estimator"),
            model.get("algorithm"),
        ):
            if candidate:
                return str(candidate)
        return "flow_grpo"

    @staticmethod
    def _extra_info(batch) -> dict:
        extra = getattr(batch, "extra_info", None)
        if extra is None:
            extra = {}
            if hasattr(batch, "extra_info"):
                batch.extra_info = extra
        return extra

    def _diffusion_v1_gen_lane(self) -> DiffusionV1GenLane:
        """GEN hooks from ``PolicyGradientDiffusionTrainerV1`` on this job's workers.

        The lane carries its own GEN estimator so ``_compute_advantage`` never reads
        the UND token ``algorithm.adv_estimator`` (RFC §4.4).
        """
        lane = getattr(self, "_diff_v1_gen_lane", None)
        wg = getattr(self, "actor_rollout_wg", None)
        gen_estimator = self._gen_adv_estimator()
        if (
            lane is None
            or getattr(lane, "actor_rollout_wg", None) is not wg
            or getattr(lane, "gen_adv_estimator", None) != gen_estimator
        ):
            self._diff_v1_gen_lane = DiffusionV1GenLane(self, gen_adv_estimator=gen_estimator)
        return self._diff_v1_gen_lane

    def _compute_old_log_prob(self, batch, metrics: dict):
        """UND: PPO V1 token old-logprob. GEN: diffusion V1 ``infer_actor_batch`` (required when K>0)."""
        extra = self._extra_info(batch)
        gen_rows = self._gen_batch_from_step(batch)
        logger.info(
            "bagel_corl old_log_prob batch_rows=%d roles=%s gen_rows=%d",
            len(batch),
            _bagel_role_histogram(getattr(batch, "tags", None)),
            len(gen_rows),
        )
        proto = build_gen_flowgrpo_proto(gen_rows)
        if proto is None:
            return super()._compute_old_log_prob(batch, metrics)
        if "all_latents" not in proto.batch.keys():
            raise RuntimeError(
                "bagel_corl GEN old_log_prob requires all_latents; refuse rollout_log_probs-only recompute skip."
            )
        if getattr(self, "actor_rollout_wg", None) is None:
            # CPU unit tests: no worker group. Packed traj logprobs stay on proto for GEN advantage.
            metrics["gen/old_log_prob_recomputed"] = 0.0
            return super()._compute_old_log_prob(batch, metrics)
        old = self._diffusion_v1_gen_lane()._compute_old_log_prob(proto)
        extra["bagel_corl_gen_old"] = old.batch
        metrics["gen/old_log_prob_recomputed"] = 1.0
        return super()._compute_old_log_prob(batch, metrics)

    def _gen_batch_from_step(self, batch) -> list:
        """GEN rows for FlowGRPO: dual-lane ``child_gen_keys`` or ``extra['gen_batch']`` only."""
        extra = self._extra_info(batch)
        if extra.get("gen_batch") is not None:
            return list(extra["gen_batch"])

        from verl_omni.agent_loop.bagel_corl_tq import split_und_gen_metas

        und_records = self._und_records_from_batch(batch)
        if und_records and any("child_gen_keys" in (r.get("fields") or r) for r in und_records):
            gen_by_key = self._fetch_gen_records_by_keys(batch, und_records)
            _, gen_batch = split_und_gen_metas(und_records, gen_by_key)
            # Batch aggregation (mean J/K over siblings, summed dropped groups,
            # fraction without image credit) — first-row-wins collapse hid every
            # episode but one (audit module A).
            aggregated = aggregate_episode_metrics(und_records)
            for key, value in aggregated.items():
                if extra.get(key) is None:
                    extra[key] = value
            extra["gen_batch"] = gen_batch
            return gen_batch

        bagel = extra.get("bagel_corl") if isinstance(extra.get("bagel_corl"), dict) else {}
        meta = getattr(batch, "meta_info", None) or {}
        bagel_meta = meta.get("bagel_corl") if isinstance(meta, dict) else {}
        if bagel.get("gen_batch") or (isinstance(bagel_meta, dict) and bagel_meta.get("gen_batch")):
            raise RuntimeError(
                "bagel_corl: GEN rows found only on nested bagel_corl.gen_batch / meta_info; "
                "dual-lane extra['gen_batch'] or child_gen_keys is required (legacy nested path removed)."
            )
        # Loud on the driver side (Ray only forwards WARNING+ from the agent-loop workers, so the
        # worker's own dual-lane pack log never reaches this log). Without it a skipped GEN lane and
        # a genuinely empty one are indistinguishable in the step metrics.
        logger.info(
            "bagel_corl gen_batch empty (GEN lane will skip): und_records=%d has_child_gen_keys=%s "
            "has_tq_handle=%s ntb_keys=%s batch=%s",
            len(und_records),
            bool(und_records) and any("child_gen_keys" in (r.get("fields") or {}) for r in und_records),
            hasattr(batch, "keys") and hasattr(batch, "partition_id"),
            sorted((getattr(batch, "non_tensor_batch", None) or {}).keys()),
            type(batch).__name__,
        )
        return []

    def _und_records_from_batch(self, batch) -> list[dict]:
        extra = self._extra_info(batch)
        if extra.get("und_batch"):
            return [{"fields": row} for row in extra["und_batch"]]
        ntb = getattr(batch, "non_tensor_batch", None) or {}
        if ntb.get("child_gen_keys") is not None:
            length = len(ntb["child_gen_keys"])
            records = []
            for i in range(length):
                fields = {}
                for key, col in ntb.items():
                    try:
                        fields[key] = col[i]
                    except (TypeError, IndexError, KeyError):
                        continue
                records.append({"fields": fields})
            return records
        # TensorDict carriers (the replay-buffer / balanced-batch shape) keep the dual-lane
        # bookkeeping in ``extra_fields`` instead of ``child_gen_keys``: the v1 advantage phase can
        # be handed a TensorDict rather than the rollout ``KVBatchMeta``, and ``_extra_fields_from_tq``
        # returns ``None`` for it (no ``partition_id``). Without this branch the UND records came
        # back with no ``child_gen_keys``, ``_gen_batch_from_step`` returned ``[]``, and the GEN lane
        # was skipped for the whole step even though every episode had a complete S-group.
        if ntb.get("extra_fields") is not None:
            extras = list(ntb["extra_fields"])
            child_col = ntb.get("child_gen_keys")
            child_col = list(child_col) if child_col is not None else None
            tags = list(getattr(batch, "tags", None) or [])
            indices = (
                [i for i, tag in enumerate(tags) if not tag.get("is_padding", False)]
                if len(tags) == len(extras)
                else range(len(extras))
            )
            records = []
            for idx in indices:
                mapping = extras[idx] if isinstance(extras[idx], dict) else {}
                fields = dict(mapping)
                if "child_gen_keys" not in fields and isinstance(mapping.get("extra_fields"), dict):
                    fields.update(mapping["extra_fields"])
                if child_col is not None and idx < len(child_col):
                    fields = self._merge_child_gen_keys(fields, child_col[idx])
                records.append({"fields": fields})
            return records
        extras = self._extra_fields_from_tq(batch)
        if extras is None:
            return []
        keys = list(getattr(batch, "keys", []) or [])
        if keys and len(extras) != len(keys):
            raise RuntimeError(
                f"bagel_corl UND extra_fields fetch returned {len(extras)} row(s) for {len(keys)} key(s); "
                "row↔key alignment is what the GEN gather and the J/K aggregation rely on"
            )
        # ``upsample_batch_to_divisible_size`` builds its synthetic rows by deep-copying the first
        # row's fields *and* tag, so a padding row still carries its template's ``child_gen_keys``,
        # ``extra_fields`` and J/K. Counting it as an episode would fetch the template's GEN rows
        # twice and fold its J/K into the batch mean a second time, so drop it here. The tags live
        # on the KVBatchMeta in key order, which is exactly the fetch order.
        tags = list(getattr(batch, "tags", None) or [])
        indices = (
            [i for i, tag in enumerate(tags) if not tag.get("is_padding", False)]
            if len(tags) == len(extras)
            else range(len(extras))
        )
        records = []
        for idx in indices:
            extra_row = extras[idx]
            mapping = extra_row if isinstance(extra_row, dict) else {}
            fields = dict(mapping)
            if "child_gen_keys" not in fields and isinstance(mapping.get("extra_fields"), dict):
                fields.update(mapping["extra_fields"])
            records.append({"fields": fields})
        return records

    def _fetch_gen_records_by_keys(self, batch, und_records: list[dict]) -> dict[str, dict]:
        """Lookup GEN TQ rows by ``child_gen_keys``. Returns key → {fields: ...}.

        ``transfer_queue.kv_batch_get`` returns a **columnar TensorDict** (one entry
        per requested key, in request order), not a keyed dict. Treating it as a dict
        silently yields ``{}`` and drops the whole GEN lane. Normalize every known
        return shape here and fail-loud on an unrecognized one instead of silently
        skipping GEN.
        """
        extra = self._extra_info(batch)
        if isinstance(extra.get("gen_by_key"), dict):
            return {str(k): ({"fields": v} if "fields" not in v else v) for k, v in extra["gen_by_key"].items()}

        keys: list[str] = []
        for rec in und_records:
            fields = rec.get("fields") or rec
            for child in fields.get("child_gen_keys") or []:
                # Order-preserving dedupe: TQ's KVBatchMeta rejects duplicate keys, and a batch
                # may legitimately point at the same seed row from more than one gather path.
                if str(child) not in keys:
                    keys.append(str(child))
        if not keys:
            return {}
        if not hasattr(batch, "partition_id"):
            raise RuntimeError(
                f"bagel_corl: batch has child_gen_keys={len(keys)} but no partition_id; cannot fetch GEN TQ rows"
            )

        import transfer_queue as tq

        data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id)
        out = _normalize_tq_kv_get_result(data, keys)
        missing = [k for k in keys if k not in out]
        if missing:
            raise RuntimeError(
                f"bagel_corl GEN TQ fetch incomplete: {len(out)}/{len(keys)} rows "
                f"(first missing: {missing[:3]})"
            )
        return out

    @staticmethod
    def _extra_fields_from_tq(batch):
        if not hasattr(batch, "keys") or not hasattr(batch, "partition_id"):
            return None
        try:
            import transfer_queue as tq

            data = tq.kv_batch_get(
                keys=list(batch.keys), partition_id=batch.partition_id, select_fields=["extra_fields", "child_gen_keys"]
            )
        except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("bagel_corl UND extra_fields TQ fetch failed") from exc
        if isinstance(data, dict):
            if "extra_fields" in data:
                return data["extra_fields"]
            return self._rows_in_batch_key_order(batch, data)
        # ``kv_batch_get`` hands back a **columnar TensorDict** (one entry per requested key, in
        # request order), not a dict and not a keyed mapping. Iterating it yields *field names*, so
        # the old ``return data`` made ``_und_records_from_batch`` build one empty record per
        # *field* -- no ``child_gen_keys`` anywhere, PROTO None, and the GEN lane silently vanished
        # (``gen/num_rows`` > 0 at pack time, GEN never trained). Read the column instead, exactly
        # like ``ReplayBuffer._dapo_filtered_keys`` does.
        if "extra_fields" in data.keys():
            rows = list(data["extra_fields"])
            # ``extra_fields`` and ``child_gen_keys`` come back as **sibling columns**, and the
            # dual-lane put writes ``child_gen_keys`` both inside the ``extra_fields`` blob and as
            # its own column. Returning only the first column silently dropped the second for any
            # row whose blob did not repeat it -- the same "GEN lane vanishes" failure the comment
            # above documents, one layer deeper. Re-attach the column before handing rows back.
            if "child_gen_keys" in data.keys():
                child_col = list(data["child_gen_keys"])
                rows = [
                    OmniBagelCoRLTrainerSync._merge_child_gen_keys(
                        row, child_col[i] if i < len(child_col) else None
                    )
                    for i, row in enumerate(rows)
                ]
            return rows
        if "child_gen_keys" in data.keys():
            return list(data["child_gen_keys"])
        return data

    @staticmethod
    def _rows_in_batch_key_order(batch, data: dict):
        """Normalize a per-key mapping (``{tq_key: row}``) into batch-key order."""
        keys = [str(k) for k in (getattr(batch, "keys", None) or [])]
        if keys and all(k in data for k in keys):
            return [data[k] for k in keys]
        return list(data.values())

    @staticmethod
    def _merge_child_gen_keys(row, child_keys):
        """Give a fetched row the ``child_gen_keys`` column when its own blob omitted it."""
        if not isinstance(row, dict) or not child_keys:
            return row
        if row.get("child_gen_keys"):
            return row
        merged = dict(row)
        merged["child_gen_keys"] = list(child_keys)
        return merged

    def _compute_advantage(self, batch, metrics: dict):
        """Same step, separate tensors: GEN FlowGRPO → ``bagel_corl_gen``; UND token GRPO → TQ via ``super``.

        Do **not** write FlowGRPO into TQ ``advantages`` (that field is UND token GRPO only).
        Do **not** pass ``algorithm.adv_estimator`` (omni default ``grpo``) into GEN.
        """
        extra = self._extra_info(batch)
        # NOTE: the dual-lane metrics (``episode/J``, ``episode/K``, ``gen/dropped_incomplete_groups``,
        # ``und/no_image_credit``) do not exist on ``extra`` yet -- ``_gen_batch_from_step`` below is
        # what runs ``aggregate_episode_metrics`` and populates them. The copy therefore has to happen
        # *after* the gather, otherwise every one of them stays absent from the step log (measured:
        # ``advantage ... J=None K=None`` on every step of the 20260922 runs, which is what made the
        # GEN lane look like it had no data at all).

        algo = self.config.algorithm
        gen_estimator = self._gen_adv_estimator()
        und_estimator = str(algo.get("adv_estimator", "grpo"))
        if gen_estimator == und_estimator and und_estimator in {"grpo", "gspo", "gae", "rloo", "reinforce_plus_plus"}:
            raise ValueError(
                f"Bagel Co-RL (Joint-Training) GEN adv_estimator={gen_estimator!r} collides with UND token estimator; "
                "set actor_rollout_ref.model.algorithm=flow_grpo (GEN) and leave algorithm.adv_estimator for UND"
            )
        proto = build_gen_flowgrpo_proto(self._gen_batch_from_step(batch))
        gen_proto = None
        gen_metrics: dict = {
            "gen/skipped_no_groups": 1.0,
            "gen/num_rows": 0.0,
            "has_complete_gen_groups": 0.0,
        }
        if proto is not None:
            old_batch = extra.get("bagel_corl_gen_old")
            if old_batch is not None and "old_log_probs" in old_batch.keys():
                proto.batch["old_log_probs"] = old_batch["old_log_probs"]
                if "old_prev_sample_mean" in old_batch.keys():
                    proto.batch["old_prev_sample_mean"] = old_batch["old_prev_sample_mean"]
            if getattr(self, "actor_rollout_wg", None) is not None:
                gen_proto = self._diffusion_v1_gen_lane()._compute_advantage(proto)
                if "all_latents" not in gen_proto.batch.keys():
                    raise RuntimeError(
                        "bagel_corl GEN V1 advantage returned no all_latents; refuse skip GEN loss."
                    )
                gen_metrics = {
                    "gen/skipped_no_groups": 0.0,
                    "gen/num_rows": float(len(gen_proto)),
                    "gen/num_usable_rows": float(len(gen_proto)),
                    "has_complete_gen_groups": 1.0,
                    "gen/has_traj": 1.0,
                }
            else:
                # CPU tests without a worker group: same FlowGRPO helper, still requires traj.
                gen_proto, gen_metrics = apply_gen_flowgrpo_advantage(
                    self._gen_batch_from_step(batch),
                    adv_estimator=gen_estimator,
                    norm_adv_by_std_in_grpo=bool(algo.get("norm_adv_by_std_in_grpo", True)),
                    global_std=bool(algo.get("global_std", True)),
                    algo_config=algo,
                )
        metrics.update(gen_metrics)
        # Now that the gather has run, the dual-lane bookkeeping is on ``extra`` (see the note at the
        # top of this method); publish it so a skipped GEN lane is attributable from the step log.
        for metric_key in (
            "und/no_image_credit",
            "gen/dropped_incomplete_groups",
            "episode/J",
            "episode/K",
        ):
            if extra.get(metric_key) is not None:
                metrics.setdefault(metric_key, extra[metric_key])
        has_complete = bool(gen_metrics.get("has_complete_gen_groups"))
        extra["has_complete_gen_groups"] = has_complete
        extra["skip_gen"] = not has_complete
        extra["num_gen_rows"] = int(gen_metrics.get("gen/num_usable_rows") or gen_metrics.get("gen/num_rows") or 0)
        # FlowGRPO advantages live only here — never merge into TQ UND advantages.
        extra["bagel_corl_gen"] = None if gen_proto is None else gen_proto.batch
        skipped = gen_metrics.get("gen/skipped_no_groups", 1.0 if not has_complete else 0.0)
        metrics["gen/skipped_no_groups"] = float(skipped)
        logger.info(
            "bagel_corl_sync advantage und=%s gen=%s skip_gen=%s J=%s K=%s",
            und_estimator,
            gen_estimator,
            extra["skip_gen"],
            metrics.get("episode/J"),
            metrics.get("episode/K"),
        )
        # UND token GRPO writes advantages/returns onto episode TQ keys.
        return super()._compute_advantage(batch, metrics)

    def _update_actor(self, batch, metrics: dict):
        """Single composite ``update_actor`` after the N-sibling gather. Skip GEN when no complete S-groups."""
        from verl.utils import tensordict_utils as tu

        extra = self._extra_info(batch)
        has_complete = bool(extra.get("has_complete_gen_groups"))
        extra["skip_gen"] = not has_complete
        extra["has_complete_gen_groups"] = has_complete
        if not has_complete:
            metrics["gen/skipped_no_groups"] = 1.0
        for key in ("episode/J", "episode/K", "und/no_image_credit"):
            if extra.get(key) is not None:
                metrics[key] = extra[key]
        # The flags have to ride on whatever carrier this trainer's bus uses, and ``_extra_info``
        # above already put them where the actor will look: the v1 bus hands us a ``KVBatchMeta``,
        # whose ``extra_info`` dict the TQ dispatch layer copies onto the actor's TensorDict as
        # non-tensor data (``verl/utils/transferqueue_utils.py:160-177``) -- which is exactly where
        # the composite reads them back (``bagel_corl_composite.py:129,131`` and
        # ``diffusers_impl.py:954,956``). ``tu.assign_non_tensor_data`` is only valid for a real
        # TensorDict; calling it on the meta asserts (``tensordict_utils.py:44``), and
        # ``hasattr(batch, "keys")`` is *not* a TensorDict test -- ``KVBatchMeta.keys`` is its list
        # of TQ keys. Measured 2026-09-18 on `hk01dgx012` (devices 4-7), the first step that
        # reached the actor: ``AssertionError: input dict must be a TensorDict``.
        from tensordict import TensorDict

        if isinstance(batch, TensorDict):
            tu.assign_non_tensor_data(batch, "skip_gen", extra["skip_gen"])
            tu.assign_non_tensor_data(batch, "has_complete_gen_groups", has_complete)
            tu.assign_non_tensor_data(batch, "num_gen_rows", int(extra.get("num_gen_rows") or 0))
            if extra.get("bagel_corl_gen") is not None:
                tu.assign_non_tensor_data(batch, "bagel_corl_gen", extra["bagel_corl_gen"])
        logger.info(
            "bagel_corl_sync update_actor skip_gen=%s policy_version=%s J=%s K=%s",
            extra.get("skip_gen"),
            getattr(self, "global_steps", None),
            metrics.get("episode/J"),
            metrics.get("episode/K"),
        )
        return super()._update_actor(batch, metrics)

    def _und_deploy_config_path(self) -> str | None:
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        path = agent.get("und_deploy_config")
        if path:
            return str(path)
        ek = self.config.actor_rollout_ref.rollout.get("engine_kwargs") or {}
        vo = (ek.get("vllm_omni") or {}) if hasattr(ek, "get") else {}
        path = vo.get("und_deploy_config") if hasattr(vo, "get") else None
        return str(path) if path else None

    @staticmethod
    def _und_ar_stage_ids(deploy_config: str) -> tuple[int, ...]:
        """Stage ids declared by the UND AR deploy config (``bagel_think``: 0 + 1).

        The UND AR replica declares its width through
        ``tensor_model_parallel_size`` (see ``_build_und_ar_entrypoint_config``),
        and a plain top-level engine arg is copied into *every* stage by the
        deploy-config path (``build_stage_runtime_overrides`` applies non-
        orchestrator keys to each stage; for the DiT stage it is then folded into
        ``parallel_config``). Each stage therefore has to be pinned back to one
        rank, or a stage holding a single ``devices`` entry is asked for two.
        Read the ids from the deploy config instead of hard-coding them so a stage
        added or collapsed upstream cannot silently lose the pin.
        """
        if not os.path.isfile(deploy_config):
            raise ValueError(
                f"bagel_corl_sync cannot read agent.und_deploy_config={deploy_config!r} to derive the "
                "UND AR stage ids. Without them the per-stage tensor_parallel_size pin cannot be "
                "emitted, and every stage would inherit the replica-width value."
            )
        stages = OmegaConf.load(deploy_config).get("stages") or []
        stage_ids = tuple(int(stage["stage_id"]) for stage in stages if "stage_id" in stage)
        if not stage_ids:
            raise ValueError(f"bagel_corl_sync: {deploy_config!r} declares no stages to pin")
        return stage_ids

    @staticmethod
    def _und_stage_device_indices(deploy_config: str) -> dict[int, int]:
        """Highest logical device index each stage of the UND AR deploy config asks for.

        Stage ``devices`` are logical indices into the replica's own
        ``CUDA_VISIBLE_DEVICES`` (``run_bagel_und_ar_serve.sh`` relies on the same
        convention), so the pool width has to cover the widest one or the stage's
        engine core dies during init with an unrelated-looking error.
        """
        stages = OmegaConf.load(deploy_config).get("stages") or []
        widest: dict[int, int] = {}
        for stage in stages:
            if "stage_id" not in stage:
                continue
            runtime = stage.get("runtime") or {}
            raw = runtime.get("devices", stage.get("devices"))
            if raw is None:
                continue
            indices = [int(token) for token in str(raw).replace(",", " ").split() if token.strip()]
            if indices:
                widest[int(stage["stage_id"])] = max(indices)
        return widest

    @staticmethod
    def _und_stage_max_model_len(deploy_config: str) -> int | None:
        """Widest ``max_model_len`` the UND AR deploy config declares across its stages.

        The entrypoint's ``max_model_len`` is a plain (non-orchestrator) engine arg, so
        the deploy-config path copies it into *every* stage and it therefore **wins**
        over each stage's own value. The builder is thus the single source of truth for
        the AR replica's context, and its value has to cover what ``_und_decode``
        actually feeds the engine: the decode prompt of UND turn ``j`` is
        ``prompt_ids + response_ids``, i.e. the *whole episode so far*.

        Sizing it as ``max_prompt + max_resp`` leaves exactly zero headroom for the
        turn being generated once the episode's response budget is spent, which is the
        measured failure:

            ValueError: Prompt length (2048) meets or exceeds the model's maximum
            context length (2048), leaving no space for generation.

        Read the value back instead of hard-coding it, so the yaml's documented budget
        (16384: the ~8625-token-per-image MM-encoder floor, sized so
        ``max_num_batched_tokens == max_num_seqs * max_model_len``) is what the engine
        actually gets and the two cannot drift.
        """
        if not os.path.isfile(deploy_config):
            return None
        stages = OmegaConf.load(deploy_config).get("stages") or []
        declared = [int(stage["max_model_len"]) for stage in stages if stage.get("max_model_len") is not None]
        return max(declared) if declared else None

    # Bagel publishes weights only: its top-level ``config.json`` is
    # ``model_type: bagel`` with no ``auto_map`` and no modeling code, so
    # ``transformers.AutoConfig`` cannot resolve it. The MoT sub-configs ship
    # alongside in the snapshot; the UND AR (Thinker) replica is an LLM and needs
    # the language sub-config.
    UND_HF_SUBCONFIG = "llm_config.json"

    def _resolve_und_hf_config_path(self, model, agent) -> str | None:
        """HF config path for the UND AR replica (RFC §4.7).

        ``OmniModelConfig.__post_init__`` calls ``AutoConfig.from_pretrained`` on
        ``hf_config_path``, falling back to ``path``. For a Bagel checkpoint that
        fallback raises ``ValueError: ... model type ``bagel`` but Transformers
        does not recognize this architecture``. Prefer the explicit
        ``agent.und_hf_config_path`` knob; otherwise derive the LLM sub-config
        from the snapshot so no recipe has to hand-set it. Non-Bagel checkpoints
        (e.g. Qwen3-Omni) resolve from ``path`` directly and get ``None``.
        """
        explicit = agent.get("und_hf_config_path")
        if explicit:
            return str(explicit)
        path = model.get("path")
        if not path:
            return None
        try:
            from verl_omni.utils.fs import resolve_model_local_dir

            local_dir = resolve_model_local_dir(str(path))
        except Exception as exc:  # noqa: BLE001 - optional derivation only
            logger.warning("bagel_corl_sync cannot locate model snapshot %s: %s", path, exc)
            return None
        candidate = os.path.join(local_dir, self.UND_HF_SUBCONFIG)
        return candidate if os.path.isfile(candidate) else None

    def _build_und_ar_entrypoint_config(self):
        """Clone entry config for a standalone UND AR ``LLMServerManager``.

        GEN keeps the rewritten diffusion hybrid stack. UND needs Omni model +
        ``output_mode=ar`` + bagel_think deploy — never ``bagel_single_stage``.
        """
        from omegaconf import OmegaConf, open_dict

        und_deploy = self._und_deploy_config_path()
        if not und_deploy:
            raise ValueError("bagel_corl_sync dual-role requires agent.und_deploy_config")

        model = self.config.actor_rollout_ref.model
        rollout = self.config.actor_rollout_ref.rollout
        agent = rollout.get("agent") or {}
        und_hf_config_path = self._resolve_und_hf_config_path(model, agent)
        und_n_gpus = int(agent.get("und_n_gpus") or 1)
        und_util = float(agent.get("und_gpu_memory_utilization") or 0.40)
        max_prompt = int(self.config.data.get("max_prompt_length") or rollout.get("prompt_length") or 1024)
        max_resp = int(self.config.data.get("max_response_length") or rollout.get("response_length") or max_prompt)
        # The AR replica's context has to be able to serve any turn the UND loop is
        # allowed to start. ``run_serial_episode`` stops at ``max_prompt + max_resp``
        # (the episode's whole context: ``_und_decode`` passes ``prompt_ids +
        # response_ids`` as the decode prompt), so an engine context below that asks for
        # a decode with no room for even one token, which the AR strategy refuses:
        #   ValueError: Prompt length (2048) meets or exceeds the model's maximum
        #   context length (2048), leaving no space for generation.
        # Measured 2026-09-17 09:30 on a 1024+1024 episode budget, where this line used to
        # say ``max_prompt + max_resp`` (2048) and contradicted both the deploy config
        # it ships and the RFC's stated 16384.
        # Prefer ``agent.und_max_model_len``, else the deploy config's own declared
        # budget (``bagel_corl_deploy_ar.yaml``: 16384), else the bare episode size.
        und_max_model_len = int(
            agent.get("und_max_model_len") or self._und_stage_max_model_len(und_deploy) or (max_prompt + max_resp)
        )
        if und_max_model_len < max_prompt + max_resp:
            raise ValueError(
                f"bagel_corl_sync: UND AR max_model_len={und_max_model_len} is smaller than the episode "
                f"context the loop has to serve (max_prompt_length={max_prompt} + "
                f"max_response_length={max_resp} = {max_prompt + max_resp}). The loop starts a turn while "
                "len(prompt_ids)+len(response_ids) is still below that budget, and the AR strategy rejects "
                "a prompt that leaves no room for generation ('Prompt length (...) meets or exceeds the "
                "model's maximum context length'). Lower data.max_prompt_length/data.max_response_length, "
                "or raise agent.und_max_model_len / the deploy config's max_model_len."
            )

        und_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        # ``actor_rollout_ref.model`` is the *actor's* config, and the recipe runs it with
        # ``lora.merge=True`` (mandatory on vllm-omni >= 0.24) -- i.e. the UND replica below
        # receives merged full weights and never an adapter. Its engine is launched by
        # ``verl/workers/rollout/vllm_rollout/vllm_async_server.py``, which enables LoRA
        # whenever the *effective* rank is > 0: it reads ``model.lora.rank``, falls back to
        # the top-level ``model.lora_rank`` ("FIXME: fallback to lora_rank for now") and only
        # zeroes that fallback when ``model.lora.merge`` is set. This builder copies
        # ``lora_rank``/``lora_alpha`` but not ``lora.merge``, so the AR engine used to be
        # launched with ``enable_lora=True, max_loras=1`` with nothing to apply.
        #
        # Measured 2026-09-20 19:44:45 on hk01dgx039 (devices 3,5,6,7): the *first* UND decode
        # of step 0 never returned. The AR worker's main thread (pid 1458872) sat in
        # ``vllm/lora/punica_wrapper/punica_gpu.py:add_lora_linear`` / ``add_shrink``, reached
        # from ``RowParallelLinear.forward -> LoRA apply`` of ``qwen2.py`` -- the Punica
        # kernels were launched for a model whose ``lora_a_stacked`` was never populated (no
        # ``add_lora`` ever ran, because the actor syncs merged weights) and never finished,
        # so the stage-0 engine core blocked on that forward forever and the two in-flight
        # requests were only failed 8 minutes later when the worker was killed. The
        # step-0 rollout, the whole training step and the validation that followed all
        # produced no rows, i.e. it surfaced as ``ValueError: Received an empty list as
        # keys.`` from the trainer's TQ read.
        #
        # Pinning the AR replica's declared rank to 0 restores the intended "full weights, no
        # engine-side adapter" state -- the same state the GEN replica reaches through
        # ``lora_as_adapter=False`` in ``vllm_omni_async_server.py`` -- and keeps the AR engine
        # on the plain ``load_weights`` path that the merged sync uses.
        und_lora_rank = 0 if _actor_merges_lora(model) else int(model.get("lora_rank") or 0)
        with open_dict(und_cfg):
            und_cfg.actor_rollout_ref.model = OmegaConf.create(
                {
                    "_target_": "verl_omni.workers.config.omni.OmniModelConfig",
                    "path": model.get("path"),
                    "tokenizer_path": model.get("tokenizer_path") or model.get("path"),
                    "model_type": "omni_model",
                    "architecture": model.get("architecture") or "OmniBagelForConditionalGeneration",
                    "trust_remote_code": bool(model.get("trust_remote_code", True)),
                    "composite_mode": "bagel_corl",
                    "lora_rank": und_lora_rank,
                    "lora_alpha": int(model.get("lora_alpha") or 0),
                    # OmniModelConfig.__post_init__ runs AutoConfig.from_pretrained on
                    # hf_config_path (falling back to ``path``). Bagel publishes weights
                    # only: its config.json is model_type "bagel" with no auto_map and no
                    # modeling code, so transformers cannot resolve it. Use the explicit
                    # agent.und_hf_config_path when set, else the snapshot's LLM sub-config.
                    "hf_config_path": und_hf_config_path,
                    # No OmniModelBase adapter is registered for
                    # ("OmniBagelForConditionalGeneration", "thinker") — the registry only
                    # knows Qwen3-Omni — and OmniModelConfig.__post_init__ only consults it
                    # when load_tokenizer=True. The UND AR replica is served over HTTP by
                    # vllm-omni (tokenisation happens server-side) and never goes through a
                    # training engine, so skip client-side tokenizer/processor loading.
                    "load_tokenizer": False,
                }
            )
            # ``to_container`` raises on ``None``, so the fallback has to be applied first --
            # otherwise a config without ``checkpoint_engine`` (the documented "naive"
            # default the ``or`` implies) dies here with "Input cfg is not an OmegaConf
            # config object (NoneType)".
            _ckpt = rollout.get("checkpoint_engine")
            ckpt_engine = OmegaConf.to_container(_ckpt, resolve=True) if _ckpt is not None else None
            ckpt_engine = ckpt_engine or {"backend": "naive"}
            und_cfg.actor_rollout_ref.rollout = OmegaConf.create(
                {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "vllm_omni",
                    # ``tensor_model_parallel_size`` here is the replica's *width
                    # declaration*, not a stage-level parallelism knob. It is what makes
                    # the colocated server see more than one card:
                    # ``RolloutReplica.__init__`` derives ``world_size = TP * DP * PP``,
                    # slices ``gpus_per_replica_node = min(n_gpus_per_node, world_size)``
                    # workers out of the pool ``_ensure_dual_role_rollout`` hands to
                    # ``init_colocated`` (one worker per ``und_n_gpus``-wide slice), and
                    # ``vLLMReplica.launch_servers`` builds the server's
                    # ``CUDA_VISIBLE_DEVICES`` from exactly ``gpus_per_replica_node``
                    # workers (``assert len(self.workers) == world_size``). Declare a
                    # width of 1 and the server gets ``CUDA_VISIBLE_DEVICES=0`` only, and
                    # stage devices are *logical* indices into that string -- which is how
                    # both ``bagel_think`` stages silently ended up on GPU 0. Each stage
                    # loads its own ~28.2GiB copy of the checkpoint, so that card then
                    # held 73.6GiB (measured: AR Thinker 32.96 + AR DiT 28.86 + GEN
                    # residual 2.15 + actor) and the first ``update_weights`` died in
                    # ``cumem create_and_map`` when the GEN rank tried to re-map its
                    # 28.56GiB level-1 sleep ("Wake-up failed on Rank 0").
                    #
                    # Declaring the width as TP (rather than data_parallel_size) keeps the
                    # launch free of extra engine args: the DP path makes
                    # ``launch_servers`` inject ``data_parallel_size_local`` alongside DP,
                    # and that key is *not* in vLLM-Omni's ``OrchestratorArgs`` (so it is
                    # not filtered as an orchestrator field) while DP *is* a pipeline-wide
                    # field that reaches every stage -- a stage pinned back to DP=1 would
                    # then abort with "data_parallel_size_local (2) must be <=
                    # data_parallel_size (1)". TP alone carries no such companion arg.
                    # ``bagel_corl_deploy_ar.yaml`` then spreads the stages over the two
                    # cards; ``stage_overrides`` below pins both stages back to TP=1 (a
                    # plain top-level engine arg reaches every stage through the deploy
                    # config path, and a TP=2 stage with one ``devices`` entry cannot
                    # start). Mirrors the Qwen3-Omni AR recipe, which also declares its
                    # replica width through ``tensor_model_parallel_size`` and leaves the
                    # per-stage engine args to the stage config.
                    "tensor_model_parallel_size": und_n_gpus,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n_gpus_per_node": und_n_gpus,
                    "nnodes": 1,
                    "prompt_length": max_prompt,
                    "response_length": max_resp,
                    # From the deploy config, never ``max_prompt + max_resp`` -- see
                    # ``_und_stage_max_model_len`` and the invariant check above.
                    "max_model_len": und_max_model_len,
                    # Bagel registers as encoder-decoder in vLLM-Omni, which disables
                    # chunked MM input and pins the MM encoder budget to
                    # max_num_batched_tokens. One image is ~8625 tokens, so the
                    # RolloutConfig default (8192) aborts AR engine init with
                    # "max_tokens_per_mm_item (8625) is larger than max_num_batched_tokens".
                    # Mirror bagel_corl_deploy_ar.yaml stage 0.
                    "max_num_batched_tokens": int(agent.get("und_max_num_batched_tokens") or 16384),
                    "gpu_memory_utilization": und_util,
                    "enforce_eager": True,
                    "free_cache_engine": True,
                    "enable_sleep_mode": True,
                    "disable_log_stats": True,
                    "checkpoint_engine": ckpt_engine,
                    "disaggregation": {"enabled": False},
                    # Nested rollout sub-configs are only instantiated into their dataclass
                    # when the node carries ``_target_`` (``omega_conf_to_dataclass`` uses
                    # ``_convert_="partial"``). Without it the AR server dies on
                    # ``self.config.prometheus.enable`` in vllm_async_server.launch_server.
                    # Keep it disabled: the AR replica must not bind a second exporter.
                    "prometheus": {"_target_": "verl.workers.config.PrometheusConfig", "enable": False},
                    "engine_kwargs": {
                        "vllm_omni": {
                            "output_mode": "ar",
                            "deploy_config": und_deploy,
                            # ``tensor_model_parallel_size`` above only declares how wide
                            # the colocated replica is; ``--stage-overrides`` is the
                            # per-stage mechanism run_bagel_und_ar_serve.sh already uses
                            # for its two-card Thinker/DiT split. Verified against vLLM-Omni:
                            # a plain top-level engine arg reaches every stage through the
                            # deploy-config path, so both ``bagel_think`` stages would
                            # otherwise be asked for two ranks while
                            # ``bagel_corl_deploy_ar.yaml`` gives each of them a single
                            # ``devices`` entry (the DiT stage even folds it into its
                            # diffusion ``parallel_config``). Stage-scoped keys are applied
                            # after the plain ones, so these win, and the deploy yaml's
                            # per-stage ``devices`` stay authoritative.
                            "stage_overrides": {
                                str(stage_id): {"tensor_parallel_size": 1}
                                for stage_id in self._und_ar_stage_ids(und_deploy)
                            },
                        }
                    },
                    "agent": OmegaConf.to_container(agent, resolve=True) or {},
                }
            )
        return und_cfg

    @staticmethod
    def _validate_und_ar_pool(
        und_rollout, und_n_gpus: int, deploy_config: str, actor_world_size: int | None = None
    ) -> None:
        """Fail loud when the AR replica's pool and its deploy config cannot agree.

        Three invariants, all otherwise surfacing as unrelated deep-stack failures:

        * ``world_size = TP * DP * PP`` (declared in ``_build_und_ar_entrypoint_config``)
          must equal the ``und_n_gpus``-wide actor-pool slice handed to ``init_colocated``,
          or ``vLLMReplica.launch_servers`` trips ``assert len(self.workers) == world_size``.
        * the pool must be at least as wide as the widest logical stage device. Stage
          ``devices`` are indices into the replica's own ``CUDA_VISIBLE_DEVICES``, so a
          1-wide pool cannot resolve ``devices: "1"`` (the layout the AR deploy yaml
          ships); a run configured that way died as
          ``StageEngineCoreProc_stage0_replica0 ... ValueError: No available memory for the
          cache blocks`` -> ``Orchestrator initialization failed: ... Failed core proc(s): {}``.
        * the pool must **divide** the actor pool, because that is how it is carved out:
          ``_ensure_dual_role_rollout`` takes the first slice with
          ``split_resource_pool(actor_pool, split_size=und_n_gpus)``, which cannot return a
          ragged split. On a 4-card pool the valid widths are therefore {1, 2, 4}; a 3-wide
          AR window -- what an offset stage layout would need to dodge a co-tenant on the
          first card -- is not expressible there. Measured 2026-09-18 10:36 on ``hk01dgx012``
          (devices 4-7): a run with ``agent.und_n_gpus=3`` against a 4-card pool died ~8
          minutes in as ``AssertionError: split_size must be a divisor of world_size``.

        ``actor_world_size`` is the actor PG width when the caller can resolve it; ``None``
        (e.g. from a unit test that only exercises the config consistency) skips the
        divisibility check rather than guessing.
        """
        declared_world_size = (
            int(und_rollout.get("tensor_model_parallel_size") or 1)
            * int(und_rollout.get("data_parallel_size") or 1)
            * int(und_rollout.get("pipeline_model_parallel_size") or 1)
        )
        if declared_world_size != und_n_gpus:
            raise ValueError(
                f"bagel_corl_sync: the UND AR replica declares world_size={declared_world_size} "
                f"(TP*DP*PP) but agent.und_n_gpus={und_n_gpus} workers are handed to it. "
                "The replica's width declaration and the actor-pool slice must match."
            )
        if actor_world_size is not None and actor_world_size % und_n_gpus:
            raise ValueError(
                f"bagel_corl_sync: agent.und_n_gpus={und_n_gpus} does not divide the actor pool "
                f"(n_gpus_per_node={actor_world_size}). The UND AR slice is taken with "
                "split_resource_pool, which requires an exact split, so this would otherwise die "
                "as 'split_size must be a divisor of world_size'. Pick a divisor of the pool "
                "(on a 4-card pool: UND_N_GPUS in {1, 2, 4})."
            )
        stage_devices = OmniBagelCoRLTrainerSync._und_stage_device_indices(deploy_config)
        widest_stage_device = 1 + max(stage_devices.values(), default=0)
        if und_n_gpus < widest_stage_device:
            raise ValueError(
                f"bagel_corl_sync: agent.und_n_gpus={und_n_gpus} cannot cover the stage devices in "
                f"{deploy_config!r} (needs >= {widest_stage_device}). Stage devices are logical "
                "indices into the replica's CUDA_VISIBLE_DEVICES; either raise UND_N_GPUS or put "
                "every stage back on logical device 0 (1-GPU smoke test)."
            )

    def _und_actor_pool_size(self) -> int | None:
        """Width of the actor pool the UND AR replica is carved out of, or ``None``.

        Used only to feed ``_validate_und_ar_pool``'s divisibility check. Returns ``None``
        rather than raising when the pool cannot be resolved -- this is a pre-flight check,
        and ``_ensure_dual_role_rollout`` reports a missing pool itself.
        """
        try:
            actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
            actor_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        except Exception:  # noqa: BLE001 - a missing pool is reported by _ensure_dual_role_rollout
            return None
        # ``split_resource_pool`` reads exactly this attribute, so validating against it
        # checks the same quantity the assert that failed would have.
        world_size = getattr(actor_pool, "world_size", None)
        return int(world_size) if world_size else None

    def _ensure_dual_role_rollout(self) -> None:
        """Start UND AR colocated on the actor GPU pool; wrap GEN+UND behind one client.

        Standalone UND fails when ``trainer.n_gpus_per_node`` already claims every
        visible card (``Total available GPUs 0``). Reward/teacher use the same
        ``init_colocated`` pattern on the actor placement group.
        """
        from verl.single_controller.ray.base import split_resource_pool
        from verl.trainer.ppo.utils import Role
        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.ray_utils import auto_await
        from verl.workers.rollout.llm_server import (
            DEFAULT_ROUTING_CACHE_SIZE,
            GlobalRequestLoadBalancer,
            LLMServerClient,
        )
        from verl.workers.rollout.replica import get_rollout_replica_class

        from verl_omni.workers.rollout.bagel_dual_role_llm_server import BagelDualRoleLLMServerClient

        if getattr(self, "_bagel_dual_role_ready", False):
            return
        if not bool((self.config.actor_rollout_ref.rollout.get("agent") or {}).get("und_ar_serving_ready")):
            raise RuntimeError(
                "bagel_corl_sync dual-role init requires agent.und_ar_serving_ready=True "
                "(prove spike_und_hermes.py first)"
            )

        gen_manager = self.llm_server_manager
        und_cfg = self._build_und_ar_entrypoint_config()
        und_rollout = und_cfg.actor_rollout_ref.rollout
        und_model = und_cfg.actor_rollout_ref.model
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        und_n_gpus = int(agent.get("und_n_gpus") or 1)
        if und_n_gpus < 1:
            raise ValueError(f"agent.und_n_gpus must be >= 1, got {und_n_gpus}")

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        # One UND AR replica: take the first ``und_n_gpus``-wide slice of the actor PG. Both
        # the declared width and the stage devices are validated against it first (see
        # ``_validate_und_ar_pool``), so a stale ``UND_N_GPUS`` reports itself here instead
        # of dying inside vLLM's engine core. ``und_deploy`` is non-``None`` because
        # ``_build_und_ar_entrypoint_config`` above already failed loud without it.
        und_deploy = self._und_deploy_config_path()
        self._validate_und_ar_pool(und_rollout, und_n_gpus, und_deploy, self._und_actor_pool_size())
        split_pools = split_resource_pool(actor_pool, split_size=und_n_gpus)
        if not split_pools:
            raise RuntimeError("bagel_corl_sync: actor resource pool is empty; cannot colocate UND AR")
        und_pool = split_pools[0]

        hybrid_n = len(gen_manager.rollout_replicas)
        replica_cls = get_rollout_replica_class(str(und_rollout.get("name") or "vllm_omni"))
        und_replica = replica_cls(
            replica_rank=hybrid_n,
            config=und_rollout,
            model_config=und_model,
            gpus_per_node=int(und_rollout.get("n_gpus_per_node") or und_n_gpus),
            name_suffix="bagel_und_ar",
        )
        logger.info(
            "bagel_corl_sync colocating UND AR on actor pool start_rank=%s und_n_gpus=%s und_deploy=%s",
            hybrid_n,
            und_n_gpus,
            und_deploy,
        )
        # ``vllm_async_server`` turns a nonzero ``model_config.lora_rank`` into
        # ``enable_lora=True`` for this engine. With the actor merging its LoRA the AR
        # replica must stay at 0 (see ``_build_und_ar_entrypoint_config``): a nonzero rank
        # here means the AR engine was launched with Punica LoRA layers and *no* adapter,
        # which hung the first decode of step 0 inside ``punica_gpu.add_lora_linear``.
        logger.info(
            "bagel_corl_sync UND AR engine lora_rank=%s actor_lora_merge=%s "
            "(0 => plain full-weight sync, no engine-side adapter)",
            int(und_model.get("lora_rank", 0) or 0),
            _actor_merges_lora(self.config.actor_rollout_ref.model),
        )

        @auto_await
        async def _init_und():
            await und_replica.init_colocated(und_pool)

        _init_und()

        und_replicas = [und_replica]
        self.und_rollout_replicas = und_replicas
        # RFC §4.11: ONE publication path. The UND AR replica joins the shared
        # ``checkpoint_manager`` so ``on_step_end`` publishes to it alongside the GEN
        # hybrids — a second ``CheckpointEngineManager`` here was dead code (never
        # invoked) and would have been a no-op anyway: the V1 trainer forces
        # ``checkpoint_engine.backend="naive"`` (verl/trainer/ppo/v1/trainer_base.py),
        # and ``CheckpointEngineManager.update_weights`` early-returns for that
        # backend without touching ``self.replicas``.
        self.checkpoint_manager.add_replicas(und_replicas)
        logger.info(
            "bagel_corl_sync UND AR replica registered for weight publication "
            "replica=%s und_n_gpus=%s (shared checkpoint_manager, RFC §4.11)",
            getattr(und_replica, "replica_rank", None),
            und_n_gpus,
        )

        und_lb = GlobalRequestLoadBalancer.remote(
            servers={und_replica.server_address: und_replica.server_handle},
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=bool(getattr(und_rollout, "full_determinism", False)),
        )
        und_client = LLMServerClient(config=und_cfg, load_balancer_handle=und_lb)
        self._bagel_dual_client = BagelDualRoleLLMServerClient(
            config=self.config,
            und_client=und_client,
            gen_client=gen_manager.get_client(),
        )
        self._bagel_dual_role_ready = True

    def get_llm_client(self):
        """Return the dual-role client (UND AR + GEN diffusion)."""
        self._ensure_dual_role_rollout()
        return self._bagel_dual_client

    def _wake_und_rollout_replicas(self) -> None:
        """Wake the colocated UND AR replica(s) after ``on_sample_end`` slept them.

        ``checkpoint_manager.sleep_replicas()`` sleeps *every* registered replica,
        and ``_ensure_dual_role_rollout`` registers the UND AR replica there. Nothing
        in the naive sync path wakes it again: ``CheckpointEngineManager.update_weights``
        short-circuits for ``backend='naive'`` without touching ``self.replicas``, and the
        actor-side ``EngineWorker.update_weights`` only resumes its *own* colocated server
        (``self.rollout``, the GEN engine). The AR engine therefore stayed asleep with
        **both** tags still set and the first UND decode of the next rollout died in
        ``AsyncOmni.generate``:

            RuntimeError: Generation rejected: Engine is partially or fully asleep.
            Currently sleeping tags: ['weights', 'kv_cache'].
            Please perform a full wake_up before generating.

        measured 2026-09-20 16:19:54 on hk01dgx039, which is the step-0 **validation**
        rollout: step 0's training rollout ran before any sleep, then ``on_sample_end``
        slept both pools, ``on_step_end`` published weights and woke only GEN, and
        every validation episode failed at its first ``_und_decode``. Validation then
        materialized no TQ rows and ``_validate`` died on the follow-on
        ``ValueError: Received an empty list as keys.`` from ``tq.kv_batch_get`` --
        i.e. the opaque second error is only a symptom of this one.

        Both tags are requested on purpose. ``vLLMOmniHttpServer.wake_up`` defaults to
        ``_get_wake_up_tags() == ["weights"]``, while ``AsyncOmni`` keeps its own
        ``_sleeping_tags`` and rejects generation while *any* tag is still sleeping;
        that is why the actor-side naive sync resumes weights and kv_cache in two
        separate calls. Asking for both in one wake clears the tag set outright, and is
        a no-op (``wake_up`` returns early on an already-warm engine) when the replica is
        awake, e.g. at ``on_init_end`` where ``init_colocated`` created it after
        ``_setup``'s sleep.
        """
        replicas = getattr(self, "und_rollout_replicas", None) or []
        if not replicas:
            return
        import asyncio

        from verl.utils.ray_utils import auto_await

        @auto_await
        async def _wake_all() -> None:
            await asyncio.gather(
                *(
                    server.wake_up.remote(tags=["weights", "kv_cache"])
                    for replica in replicas
                    for server in replica.servers
                )
            )

        _wake_all()
        logger.info(
            "bagel_corl_sync woke %s UND AR replica(s) (weights+kv_cache) after sleep",
            len(replicas),
        )

    def _bagel_rm_enabled(self) -> bool:
        reward_cfg = getattr(self.config, "reward", None)
        if reward_cfg is None:
            return False
        try:
            return bool(reward_cfg.get("reward_model", {}).get("enable", False))
        except (AttributeError, TypeError):
            return False

    def _ensure_reward_loop_manager(self):
        """Create the colocated ``OmniRewardLoopManager`` (RFC §4.2).

        Its worker handles serve both consumers: the inherited episode-reward
        ``_compute_score`` and the mid-loop GEN scoring adapter
        (``bagel_corl_rm``). Creation mirrors the diffusion V1 trainer.
        """
        if getattr(self, "reward_loop_manager", None) is not None:
            return self.reward_loop_manager
        from verl.trainer.ppo.utils import Role

        from verl_omni.reward_loop import OmniRewardLoopManager

        resource_pool = None
        if getattr(self, "use_rm", False) and getattr(self, "resource_pool_manager", None) is not None:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        self.reward_loop_manager = OmniRewardLoopManager(config=self.config, rm_resource_pool=resource_pool)
        return self.reward_loop_manager

    def get_reward_handles(self):
        """Handles the pinned ``TaskRunnerV1`` forwards into the agent-loop manager.

        The manager passes them to every ``BagelCorlAgentLoopWorkerTQ``, which
        binds the GEN-side handle for in-loop GEN scoring.

        Normalised to ``None`` when there are none. ``RewardLoopManager.
        reward_loop_worker_handles`` returns ``self.reward_loop_workers`` -- an *empty
        list*, not ``None`` -- whenever ``reward_model.enable=False``, which the Co-RL
        recipe sets alongside ``reward.num_workers=0`` (the in-loop Bagel reward is the
        only scorer). The agent loop then tests ``is not None`` and calls
        ``random.choice(...)`` on it, so any episode that finishes with
        ``reward_score is None`` kills the worker with

            IndexError: Cannot choose from an empty sequence

        which masks the real failure. Measured 2026-09-17 09:30, 29 ms behind the AR
        context-length ``ValueError`` in the same worker. Returning ``None`` makes the
        agent loop skip that branch and lets the genuine error surface.
        """
        manager = getattr(self, "reward_loop_manager", None)
        if manager is not None:
            workers = getattr(manager, "reward_loop_workers", None)
            if workers:
                return workers
        getter = getattr(super(), "get_reward_handles", None)
        handles = getter() if callable(getter) else None
        return list(handles) if handles else None

    def on_init_end(self):
        # Build UND AR before the first weight publish so both pools see step-0 weights.
        self._ensure_dual_role_rollout()
        if self._bagel_rm_enabled():
            self._ensure_reward_loop_manager()
        super().on_init_end()
        # ``_setup`` slept the GEN replica(s) to load the checkpoint and the parent hook
        # woke them through the actor; the AR replica is created after that sleep, so this
        # is a no-op today and a guard if the creation order ever moves.
        self._wake_und_rollout_replicas()

    def on_step_end(self):
        # Parent updates weights for every replica registered on checkpoint_manager
        # (GEN hybrid + UND standalone via add_replicas, RFC §4.11). The UND AR
        # replica must NOT stay on step-0 weights, so record which replicas were
        # published rather than trusting the registration implicitly.
        super().on_step_end()
        if getattr(self, "und_rollout_replicas", None):
            registered = getattr(getattr(self, "checkpoint_manager", None), "replicas", None) or []
            missing = [r for r in self.und_rollout_replicas if r not in registered]
            if missing:
                raise RuntimeError(
                    f"bagel_corl_sync: {len(missing)} UND AR replica(s) are not registered on "
                    "checkpoint_manager; they would keep step-0 weights (RFC §4.11)"
                )
            logger.info(
                "bagel_corl_sync step=%s: %s rollout replica(s) registered incl. UND AR "
                "(naive backend publishes through the actor to its own GEN server only)",
                getattr(self, "global_steps", None),
                len(registered),
            )
        # The parent hook's naive publish only resumed the actor's own GEN server, so the
        # AR engine is still asleep here and every rollout since would fail on its first
        # UND decode. See ``_wake_und_rollout_replicas``.
        self._wake_und_rollout_replicas()

    def on_validate_end(self):
        """Re-wake the AR replica after ``_validate``'s colocated-reward sleep.

        ``_validate`` sleeps **all** replicas when ``reward_loop_manager.
        reward_loop_worker_handles is None`` (a colocated RM). Today's recipe runs with
        ``reward.reward_model.enable=False`` + ``num_workers=0``, whose handle list is empty
        -- not ``None`` -- so that branch is skipped; ``ENABLE_RM=1`` flips it on. Its
        follow-up ``update_weights()`` is the naive short-circuit that only wakes the actor's
        GEN server, so without this the AR engine would be asleep for the *next* step's
        rollout: the same failure ``_wake_und_rollout_replicas`` fixes, one step later. Also
        covers ``val_before_train=True``, where ``on_step_end`` has not run yet.

        Harmless when nothing slept: ``AsyncOmni.wake_up`` logs "already warm" and returns
        without touching the engine.
        """
        super().on_validate_end()
        self._wake_und_rollout_replicas()

    def _validate(self):
        """V1 validation, plus the RFC §8.2 evidence record.

        Was NON-CONFORMANT: ``test_freq`` unset meant the required validation
        evidence had no execution path. The cadence is restored by the recipe
        (``trainer.test_freq``); this hook records what the validated episodes
        actually exercise — in-episode J/K, pattern coverage, and whether UND got
        image credit — because §8.2 gates PR1 on seeing pattern-1/2/3 episodes.
        """
        val_metrics = super()._validate()
        evidence = {
            key: float(value)
            for key, value in (val_metrics or {}).items()
            if isinstance(value, (int, float))
            and (
                "episode/J" in str(key)
                or "episode/K" in str(key)
                or "pattern" in str(key).lower()
                or "no_image_credit" in str(key)
                or "skipped_no_groups" in str(key)
            )
        }
        val_metrics = dict(val_metrics or {})
        val_metrics["val/rfc82_evidence_keys"] = float(len(evidence))
        if "episode/K" in evidence and evidence.get("episode/J") == 0 and evidence.get("episode/K") == 0:
            logger.warning(
                "bagel_corl_sync §8.2: validation produced no dual-lane episode evidence "
                "(J=0, K=0); the validation batch did not run the bagel_multiturn_agent loop."
            )
        logger.info("bagel_corl_sync §8.2 validation evidence: %s", evidence or "<none>")
        return val_metrics
