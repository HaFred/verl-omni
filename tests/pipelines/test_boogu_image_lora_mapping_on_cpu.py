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
"""CPU tests for Boogu-Image LoRA rollout name mapping.

Regression cover for https://github.com/verl-project/verl-omni/issues/658: the
attention output projection used to be trained on the actor and silently
dropped on the rollout, because diffusers names it ``attn.to_out.0`` while the
vllm-omni transformer exposes ``attn.to_out``.
"""

import pytest
import torch

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    BOOGU_LORA_TARGETS,
    rename_boogu_lora_name,
    validate_boogu_lora_targets,
)

RANK = 4
HIDDEN = 8

# Both shipped Boogu recipes:
#   examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora.sh
#   examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh
RECIPE_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "img_to_q",
    "img_to_k",
    "img_to_v",
    "img_out",
    "instruct_to_q",
    "instruct_to_k",
    "instruct_to_v",
    "instruct_out",
    "feed_forward.linear_1",
    "feed_forward.linear_2",
    "feed_forward.linear_3",
    "img_feed_forward.linear_1",
    "img_feed_forward.linear_2",
    "img_feed_forward.linear_3",
]


def _mapper():
    """The mapper is a pure function of (tensors, config); no engine state is read."""
    pytest.importorskip("vllm_omni")
    from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import (
        BooguImagePipelineWithLogProb,
    )

    return BooguImagePipelineWithLogProb.__new__(BooguImagePipelineWithLogProb)


# Block-local module names as the actor holds them, taken from the FSDP shard keys
# of this recipe's own checkpoint (e.g.
# ``double_stream_layers.0.img_instruct_attn.to_out.0.lora_A.default.weight``).
DOUBLE_STREAM_MODULES = (
    "img_instruct_attn.to_out.0",
    "img_instruct_attn.instruct_out",
    "img_self_attn.to_q",
    "img_feed_forward.linear_3",
)
BASE_BLOCK_MODULES = ("attn.to_q", "attn.to_out.0", "feed_forward.linear_1")


def _trainer_lora_tensors(
    block: str = "double_stream_layers.0", modules=DOUBLE_STREAM_MODULES
) -> dict[str, torch.Tensor]:
    """Build a PEFT-style Boogu LoRA state dict as the actor exports it."""
    tensors = {}
    for module in modules:
        tensors[f"{block}.{module}.lora_A.weight"] = torch.randn(RANK, HIDDEN)
        tensors[f"{block}.{module}.lora_B.weight"] = torch.randn(HIDDEN, RANK)
    return tensors


def _peft_config() -> dict:
    return {"r": RANK, "lora_alpha": 8, "target_modules": list(RECIPE_TARGETS)}


class TestNameTranslation:
    @pytest.mark.parametrize(
        ("diffusers_name", "vllm_name"),
        [
            ("attn.to_out.0", "attn.to_out"),
            ("img_instruct_attn.to_out.0", "img_instruct_attn.to_out"),
        ],
    )
    def test_output_projection_is_unwrapped(self, diffusers_name, vllm_name):
        assert rename_boogu_lora_name(diffusers_name) == vllm_name

    def test_verbatim_targets_are_untouched(self):
        # Everything except the o-proj already matches, so the mapper must not
        # disturb it -- a broad rewrite is how the q/k/v half got broken before.
        for target in RECIPE_TARGETS:
            if target != "to_out.0":
                assert rename_boogu_lora_name(target) == target

    def test_lora_tensor_keys_are_renamed(self):
        mapped, _ = _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        modules = {name.rsplit(".lora_", 1)[0] for name in mapped}
        assert "double_stream_layers.0.img_instruct_attn.to_out" in modules
        assert "double_stream_layers.0.img_self_attn.to_q" in modules
        assert not any(".to_out.0" in module for module in modules)

    def test_base_block_output_projection_is_renamed_too(self):
        mapped, _ = _mapper().map_lora_update_to_engine(
            _trainer_lora_tensors("context_refiner.1", BASE_BLOCK_MODULES), _peft_config()
        )
        modules = {name.rsplit(".lora_", 1)[0] for name in mapped}
        assert "context_refiner.1.attn.to_out" in modules
        assert "context_refiner.1.attn.to_q" in modules

    def test_component_prefix_is_preserved(self):
        # Pushed keys may or may not carry a component prefix; the rename is a leaf
        # change and must not depend on it.
        tensors = {"transformer.context_refiner.0.attn.to_out.0.lora_A.weight": torch.randn(RANK, HIDDEN)}
        mapped, _ = _mapper().map_lora_update_to_engine(tensors, _peft_config())
        assert list(mapped) == ["transformer.context_refiner.0.attn.to_out.lora_A.weight"]

    def test_non_lora_names_pass_through(self):
        tensors = {"some_unrelated.weight": torch.randn(2, 2)}
        mapped, _ = _mapper().map_lora_update_to_engine(tensors, _peft_config())
        assert list(mapped) == ["some_unrelated.weight"]

    def test_colliding_renames_are_rejected(self):
        # A half-renamed state dict would make one delta overwrite the other.
        tensors = {
            "context_refiner.0.attn.to_out.0.lora_A.weight": torch.randn(RANK, HIDDEN),
            "context_refiner.0.attn.to_out.lora_A.weight": torch.randn(RANK, HIDDEN),
        }
        with pytest.raises(ValueError, match="collapsed distinct tensors"):
            _mapper().map_lora_update_to_engine(tensors, _peft_config())


class TestTargetValidation:
    def test_recipe_targets_are_accepted_and_translated(self):
        translated = validate_boogu_lora_targets(RECIPE_TARGETS)
        assert len(translated) == len(RECIPE_TARGETS)
        assert "to_out" in translated
        assert "to_out.0" not in translated

    def test_mapper_rewrites_target_modules_in_config(self):
        _, config = _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        assert "to_out" in config["target_modules"]
        assert "to_out.0" not in config["target_modules"]
        assert config["r"] == RANK  # other fields untouched

    def test_input_config_is_not_mutated(self):
        config = _peft_config()
        _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), config)
        assert "to_out.0" in config["target_modules"]

    @pytest.mark.parametrize("target_modules", ["all-linear", ["to_q", "adaln_proj.linear"], ["to_q", "norm_q"]])
    def test_unbindable_targets_raise_instead_of_being_dropped(self, target_modules):
        config = {**_peft_config(), "target_modules": target_modules}
        with pytest.raises(ValueError, match="unsupported targets"):
            _mapper().map_lora_update_to_engine({}, config)

    def test_empty_targets_are_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_boogu_lora_targets([])

    def test_missing_target_modules_are_rejected(self):
        with pytest.raises(ValueError, match="explicit target_modules"):
            validate_boogu_lora_targets(None)


class TestVllmManagerInterplay:
    """Guard against the original failure: the o-proj target matched no vllm module."""

    # Leaf modules as they appear on the vllm-omni Boogu transformer
    # (boogu_image_transformer.py: BooguImageSelfAttention, BooguImageJointAttention,
    # LuminaFeedForward).
    VLLM_MODULES = [
        "noise_refiner.0.attn.to_q",
        "noise_refiner.0.attn.to_k",
        "noise_refiner.0.attn.to_v",
        "noise_refiner.0.attn.to_out",
        "noise_refiner.0.feed_forward.linear_1",
        "noise_refiner.0.feed_forward.linear_2",
        "noise_refiner.0.feed_forward.linear_3",
        "double_stream_layers.0.img_instruct_attn.img_to_q",
        "double_stream_layers.0.img_instruct_attn.img_to_k",
        "double_stream_layers.0.img_instruct_attn.img_to_v",
        "double_stream_layers.0.img_instruct_attn.instruct_to_q",
        "double_stream_layers.0.img_instruct_attn.instruct_to_k",
        "double_stream_layers.0.img_instruct_attn.instruct_to_v",
        "double_stream_layers.0.img_instruct_attn.instruct_out",
        "double_stream_layers.0.img_instruct_attn.img_out",
        "double_stream_layers.0.img_instruct_attn.to_out",
        "double_stream_layers.0.img_self_attn.to_out",
        "double_stream_layers.0.img_feed_forward.linear_1",
        "double_stream_layers.0.img_feed_forward.linear_2",
        "double_stream_layers.0.img_feed_forward.linear_3",
        "single_stream_layers.0.attn.to_q",
        "single_stream_layers.0.attn.to_out",
        "single_stream_layers.0.feed_forward.linear_1",
    ]

    @pytest.fixture()
    def match(self):
        pytest.importorskip("vllm_omni")
        from vllm_omni.diffusion.lora.utils import _match_target_modules

        return _match_target_modules

    def test_untranslated_target_matches_nothing_on_vllm(self, match):
        """This is the bug: ``to_out.0`` binds no layer on the Boogu transformer."""
        assert not any(match(module, ["to_out.0"]) for module in self.VLLM_MODULES)

    def test_translated_target_matches_the_o_proj_modules(self, match):
        matched = {module for module in self.VLLM_MODULES if match(module, ["to_out"])}
        assert matched == {
            "noise_refiner.0.attn.to_out",
            "double_stream_layers.0.img_instruct_attn.to_out",
            "double_stream_layers.0.img_self_attn.to_out",
            "single_stream_layers.0.attn.to_out",
        }

    def test_every_recipe_target_binds_at_least_one_module(self, match):
        """No target in the shipped recipes may be silently unbindable."""
        for target in validate_boogu_lora_targets(RECIPE_TARGETS):
            assert any(match(module, [target]) for module in self.VLLM_MODULES), (
                f"target {target!r} matches no Boogu transformer module; "
                "it would be shipped and silently dropped by the rollout"
            )

    def test_pushed_key_matches_a_binding_candidate(self):
        """Matching alone is not enough: the manager must also *bind* the tensor.

        Mirrors ``DiffusionLoRAManager._get_lora_weights``, which tries the full name,
        the component-relative name and the bare suffix. With the original
        ``...to_out.0`` key none of those candidates matched, so even a wrapped layer
        would have stayed unbound.
        """
        mapped, _ = _mapper().map_lora_update_to_engine(
            {"context_refiner.1.attn.to_out.0.lora_A.weight": torch.randn(RANK, HIDDEN)}, _peft_config()
        )
        (key,) = mapped
        module = "context_refiner.1.attn.to_out"
        candidates = {f"transformer.{module}", module, module.split(".")[-1]}
        assert key.removesuffix(".lora_A.weight") in candidates

    def test_whitelist_covers_what_the_recipes_request(self):
        assert {rename_boogu_lora_name(target) for target in RECIPE_TARGETS} <= BOOGU_LORA_TARGETS


class TestRegisteredPipelinesExposeTheMapper:
    """The hijack only calls the mapper when the *registered* pipeline class defines it.

    ``VLLMOmniHijack`` does ``getattr(self.pipeline, "map_lora_update_to_engine", None)``
    and skips translation when it is absent, so a mapper that never reaches the
    registered class is indistinguishable from having no mapper at all. Boogu
    registers one rollout class per algorithm, and the DiffusionNFT class derives
    from the FlowGRPO one, so both must expose it or one recipe silently regresses.
    """

    ALGORITHMS = ("flow_grpo", "diffusion_nft")

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_registered_pipeline_defines_the_mapper(self, algorithm):
        pytest.importorskip("vllm_omni")
        import verl_omni.pipelines  # noqa: F401  (populates the registry)
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        pipeline_cls = VllmOmniPipelineBase.get_class("BooguImagePipeline", algorithm)
        assert pipeline_cls is not None, f"no Boogu rollout pipeline registered for {algorithm!r}"
        assert callable(getattr(pipeline_cls, "map_lora_update_to_engine", None)), (
            f"{pipeline_cls.__name__} does not expose map_lora_update_to_engine, so the "
            "rollout would silently drop every to_out.0 delta"
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_registered_pipeline_mapper_translates_the_recipe(self, algorithm):
        pytest.importorskip("vllm_omni")
        import verl_omni.pipelines  # noqa: F401
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        pipeline_cls = VllmOmniPipelineBase.get_class("BooguImagePipeline", algorithm)
        # The mapper reads no instance state, so it is callable without an engine.
        mapped, config = pipeline_cls.__new__(pipeline_cls).map_lora_update_to_engine(
            _trainer_lora_tensors(), _peft_config()
        )
        assert "double_stream_layers.0.img_instruct_attn.to_out.lora_A.weight" in mapped
        assert "to_out" in config["target_modules"]
        assert "to_out.0" not in config["target_modules"]
