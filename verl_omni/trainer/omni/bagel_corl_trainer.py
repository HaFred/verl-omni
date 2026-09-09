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

from omegaconf import OmegaConf, open_dict
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.trainer.omni.bagel_corl_gen_adv import apply_gen_flowgrpo_advantage
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

    Parent is intentional AR V1 (``OmniPPOTrainerSync`` / token GRPO), not diffusion
    ``PolicyGradientDiffusionTrainerV1Sync`` and not OCR's legacy ``PolicyGradientRayTrainer``.
    Rewrite + ``ActorRolloutRefWorker`` swap unlock GEN *loss* on ``DiffusersFSDPEngine``.
    Same training step: UND token GRPO via ``super()._compute_advantage`` (TQ
    ``advantages``); GEN FlowGRPO via ``apply_gen_flowgrpo_advantage`` into
    ``extra_info["bagel_corl_gen"]`` only (grouped by ``gen_group_uid``). Estimators
    are decoupled: UND uses ``algorithm.adv_estimator`` (omni default ``grpo``);
    GEN uses ``model.algorithm`` / ``flow_grpo``. One ``update_actor`` /
    ``optimizer.step`` / weight publish. Incomplete or traj-less GEN views skip GEN
    only; UND still updates.
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
        rollout_filtered["agent"] = agent

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

    def _expected_k(self) -> int:
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

    def _gen_batch_from_step(self, batch) -> list:
        """GEN rows for FlowGRPO: dual-lane ``child_gen_keys`` first; nested extras are legacy only."""
        extra = self._extra_info(batch)
        if extra.get("gen_batch") is not None:
            return list(extra["gen_batch"])

        from verl_omni.agent_loop.bagel_corl_tq import split_und_gen_metas

        und_records = self._und_records_from_batch(batch)
        if und_records and any("child_gen_keys" in (r.get("fields") or r) for r in und_records):
            gen_by_key = self._fetch_gen_records_by_keys(batch, und_records)
            _, gen_batch = split_und_gen_metas(und_records, gen_by_key)
            for rec in und_records:
                fields = rec.get("fields") or rec
                metrics = fields.get("bagel_corl_metrics") or {}
                for key in (
                    "episode/J",
                    "episode/K",
                    "und/no_image_credit",
                    "gen/skipped_no_groups",
                    "gen/dropped_incomplete_groups",
                ):
                    if key in metrics and extra.get(key) is None:
                        extra[key] = metrics[key]
                if fields.get("episode_J") is not None and extra.get("episode/J") is None:
                    extra["episode/J"] = float(fields["episode_J"])
                if fields.get("episode_K") is not None and extra.get("episode/K") is None:
                    extra["episode/K"] = float(fields["episode_K"])
            extra["gen_batch"] = gen_batch
            return gen_batch

        bagel = extra.get("bagel_corl") if isinstance(extra.get("bagel_corl"), dict) else {}
        if bagel.get("gen_batch"):
            logger.warning("bagel_corl: using legacy bagel_corl.gen_batch; prefer dual-lane child_gen_keys")
            return list(bagel["gen_batch"])
        meta = getattr(batch, "meta_info", None) or {}
        bagel_meta = meta.get("bagel_corl") if isinstance(meta, dict) else {}
        if isinstance(bagel_meta, dict) and bagel_meta.get("gen_batch"):
            logger.warning("bagel_corl: using legacy meta_info bagel_corl.gen_batch")
            extra.update({k: v for k, v in bagel_meta.items() if k != "gen_batch"})
            extra["gen_batch"] = list(bagel_meta["gen_batch"])
            return extra["gen_batch"]
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
        """Lookup GEN TQ rows by ``child_gen_keys``. Returns key → {fields: ...}."""
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
            return {}
        try:
            import transfer_queue as tq

            data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id)
        except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("bagel_corl GEN key fetch skipped: %s", exc)
            return {}

        out: dict[str, dict] = {}
        if isinstance(data, dict):
            # transfer_queue may return {key: fields} or columnar {field: [..]}
            if all(isinstance(k, str) and k in keys for k in data.keys()) and keys and keys[0] in data:
                for key in keys:
                    row = data.get(key)
                    if row is not None:
                        out[key] = {"fields": row if isinstance(row, dict) else {"value": row}}
                return out
            # Columnar: zip by position if "keys" present
            got_keys = data.get("keys") or keys
            n = len(got_keys)
            for i, key in enumerate(got_keys):
                fields = {}
                for field_name, col in data.items():
                    if field_name == "keys":
                        continue
                    try:
                        fields[field_name] = col[i]
                    except (TypeError, IndexError, KeyError):
                        continue
                out[str(key)] = {"fields": fields}
            if out:
                return out
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
        except (KeyError, AttributeError, RuntimeError, TypeError, ValueError):
            return None

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
            try:
                tu.assign_non_tensor_data(batch, "skip_gen", extra["skip_gen"])
                tu.assign_non_tensor_data(batch, "has_complete_gen_groups", has_complete)
                tu.assign_non_tensor_data(batch, "num_gen_rows", int(extra.get("num_gen_rows") or 0))
                if extra.get("bagel_corl_gen") is not None:
                    tu.assign_non_tensor_data(batch, "bagel_corl_gen", extra["bagel_corl_gen"])
            except (TypeError, AttributeError, ValueError) as exc:
                logger.debug("bagel_corl actor non-tensor attach skipped: %s", exc)
        logger.info(
            "bagel_corl_sync update_actor skip_gen=%s policy_version=%s J=%s K=%s",
            extra.get("skip_gen"),
            getattr(self, "global_steps", None),
            metrics.get("episode/J"),
            metrics.get("episode/K"),
        )
        return super()._update_actor(batch, metrics)
