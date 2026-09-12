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
"""Bagel UND+GEN Co-RL trainer: one post-gather composite update + weight publish."""

from __future__ import annotations

import logging
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
from verl_omni.workers.config.diffusion import DiffusionRolloutConfig

logger = logging.getLogger(__name__)

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
                if row is not None:
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


@register_trainer("bagel_corl_sync")
class OmniBagelCoRLTrainerSync(OmniPPOTrainerSync):
    """Synchronous Bagel Co-RL: serial J-episode gather, then one UND+GEN optimizer step.

    Weight sync runs once in ``on_step_end`` (inherited) to every replica. No mid-episode sync.
    Replay uses the sync ``ReplayBuffer`` even though ``trainer_mode`` is not the string ``sync``.

    ``main_omni`` defaults to omni ``OmniModelConfig`` / ``RolloutConfig``. Bagel Co-RL needs
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
        agent.setdefault("gen_samples_per_call", 4)
        agent.setdefault("max_generate_passes", 1)
        agent.setdefault("max_und_turns", 8)
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
        # the rollout graph — prohibitive for multi-turn episodes. Training runs
        # CFG-free; the recipe's eval block re-enables CFG if desired.
        pipeline["cfg_text_scale"] = 1.0
        rollout_filtered["pipeline"] = pipeline

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
        agent = self.config.actor_rollout_ref.rollout.get("agent") or {}
        return int(agent.get("gen_samples_per_call") or 4)

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
        """GEN hooks from ``PolicyGradientDiffusionTrainerV1`` on this job's workers."""
        lane = getattr(self, "_diff_v1_gen_lane", None)
        wg = getattr(self, "actor_rollout_wg", None)
        if lane is None or getattr(lane, "actor_rollout_wg", None) is not wg:
            self._diff_v1_gen_lane = DiffusionV1GenLane(self)
        return self._diff_v1_gen_lane

    def _compute_old_log_prob(self, batch, metrics: dict):
        """UND: PPO V1 token old-logprob. GEN: diffusion V1 ``infer_actor_batch`` (required when K>0)."""
        extra = self._extra_info(batch)
        proto = build_gen_flowgrpo_proto(self._gen_batch_from_step(batch))
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
        extras = self._extra_fields_from_tq(batch)
        if extras is None:
            return []
        records = []
        for extra_row in extras:
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
            keys.extend(str(k) for k in (fields.get("child_gen_keys") or []))
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
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields", "child_gen_keys"]
            )
            if isinstance(data, dict) and "extra_fields" in data:
                return data["extra_fields"]
            return data
        except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("bagel_corl UND extra_fields TQ fetch failed") from exc

    def _compute_advantage(self, batch, metrics: dict):
        """Same step, separate tensors: GEN FlowGRPO → ``bagel_corl_gen``; UND token GRPO → TQ via ``super``.

        Do **not** write FlowGRPO into TQ ``advantages`` (that field is UND token GRPO only).
        Do **not** pass ``algorithm.adv_estimator`` (omni default ``grpo``) into GEN.
        """
        extra = self._extra_info(batch)
        for metric_key in (
            "und/no_image_credit",
            "gen/dropped_incomplete_groups",
            "episode/J",
            "episode/K",
            "gen/skipped_no_groups",
        ):
            if extra.get(metric_key) is not None:
                metrics[metric_key] = extra[metric_key]

        algo = self.config.algorithm
        gen_estimator = self._gen_adv_estimator()
        und_estimator = str(algo.get("adv_estimator", "grpo"))
        if gen_estimator == und_estimator and und_estimator in {"grpo", "gspo", "gae", "rloo", "reinforce_plus_plus"}:
            raise ValueError(
                f"Bagel Co-RL GEN adv_estimator={gen_estimator!r} collides with UND token estimator; "
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
        # Ensure GEN FlowGRPO view + skip flags ride on the actor TensorDict / extra_info.
        if hasattr(batch, "keys") and callable(getattr(tu, "assign_non_tensor_data", None)):
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
        und_n_gpus = int(agent.get("und_n_gpus") or 1)
        und_util = float(agent.get("und_gpu_memory_utilization") or 0.40)
        max_prompt = int(self.config.data.get("max_prompt_length") or rollout.get("prompt_length") or 1024)
        max_resp = int(self.config.data.get("max_response_length") or rollout.get("response_length") or max_prompt)

        und_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
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
                    "lora_rank": int(model.get("lora_rank") or 0),
                    "lora_alpha": int(model.get("lora_alpha") or 0),
                }
            )
            ckpt_engine = OmegaConf.to_container(rollout.get("checkpoint_engine"), resolve=True) or {
                "backend": "naive"
            }
            und_cfg.actor_rollout_ref.rollout = OmegaConf.create(
                {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "vllm_omni",
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "n_gpus_per_node": und_n_gpus,
                    "nnodes": 1,
                    "prompt_length": max_prompt,
                    "response_length": max_resp,
                    "max_model_len": max_prompt + max_resp,
                    "gpu_memory_utilization": und_util,
                    "enforce_eager": True,
                    "free_cache_engine": True,
                    "enable_sleep_mode": True,
                    "disable_log_stats": True,
                    "checkpoint_engine": ckpt_engine,
                    "disaggregation": {"enabled": False},
                    "prometheus": {"enable": False},
                    "engine_kwargs": {
                        "vllm_omni": {
                            "output_mode": "ar",
                            "deploy_config": und_deploy,
                        }
                    },
                    "agent": OmegaConf.to_container(agent, resolve=True) or {},
                }
            )
        return und_cfg

    def _ensure_dual_role_rollout(self) -> None:
        """Start UND AR colocated on the actor GPU pool; wrap GEN+UND behind one client.

        Standalone UND fails when ``trainer.n_gpus_per_node`` already claims every
        visible card (``Total available GPUs 0``). Reward/teacher use the same
        ``init_colocated`` pattern on the actor placement group.
        """
        from verl.checkpoint_engine.base import CheckpointEngineManager
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
        # One UND AR replica (TP=1): take the first ``und_n_gpus``-wide slice of the actor PG.
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
            self._und_deploy_config_path(),
        )

        @auto_await
        async def _init_und():
            await und_replica.init_colocated(und_pool)

        _init_und()

        und_replicas = [und_replica]
        self.und_rollout_replicas = und_replicas
        self.checkpoint_manager.add_replicas(und_replicas)

        und_ckpt_config = omega_conf_to_dataclass(und_rollout.checkpoint_engine)
        und_ckpt_config.backend = "naive"
        self.und_checkpoint_manager = CheckpointEngineManager(
            config=und_ckpt_config,
            actor_wg=self.actor_rollout_wg,
            replicas=und_replicas,
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
        binds the DiT-side handle for in-loop GEN scoring.
        """
        manager = getattr(self, "reward_loop_manager", None)
        if manager is not None:
            workers = getattr(manager, "reward_loop_workers", None)
            if workers:
                return workers
        getter = getattr(super(), "get_reward_handles", None)
        return getter() if callable(getter) else None

    def on_init_end(self):
        # Build UND AR before the first weight publish so both pools see step-0 weights.
        self._ensure_dual_role_rollout()
        if self._bagel_rm_enabled():
            self._ensure_reward_loop_manager()
        super().on_init_end()

    def on_step_end(self):
        # Parent updates weights for every replica registered on checkpoint_manager
        # (GEN hybrid + UND standalone via add_replicas).
        super().on_step_end()
