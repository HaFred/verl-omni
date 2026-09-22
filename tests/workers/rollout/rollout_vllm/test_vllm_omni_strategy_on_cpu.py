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

from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch
import yaml

from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_ar_strategy as ar_strategy_module
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as server_module
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_diffusion_strategy as diffusion_strategy_module
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


@pytest.mark.parametrize(
    ("output_mode", "expected_kind"),
    [("ar", "ar"), ("diffusion", "diffusion"), (None, "diffusion")],
)
def test_server_selects_one_generation_strategy(monkeypatch, output_mode, expected_kind):
    class _Strategy:
        def __init__(self, server, kind):
            self.server = server
            self.kind = kind

        def init_config(self, config):
            return SimpleNamespace(kind=self.kind, seed=config.seed)

        def init_model_config(self, model_config):
            return (self.kind, model_config)

    monkeypatch.setattr(server_module, "ARStrategy", lambda server: _Strategy(server, "ar"))
    monkeypatch.setattr(server_module, "DiffusionStrategy", lambda server: _Strategy(server, "diffusion"))

    omni_kwargs = {} if output_mode is None else {"output_mode": output_mode}
    server = object.__new__(server_module.vLLMOmniHttpServer)
    rollout_config = server._init_config(SimpleNamespace(engine_kwargs={"vllm_omni": omni_kwargs}, seed=None))
    model_config = server._init_model_config("model-config")

    assert server._generate_strategy.kind == expected_kind
    assert rollout_config.kind == expected_kind
    assert rollout_config.seed == 42
    assert model_config == (expected_kind, "model-config")


@pytest.mark.asyncio
async def test_server_generate_delegates_without_changing_rpc_arguments():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server._generate_strategy = SimpleNamespace(generate=AsyncMock(return_value="result"))
    prompt_mask = torch.tensor([True, True])

    result = await server.generate(
        prompt_ids=[1, 2],
        sampling_params={"temperature": 0.7},
        request_id="request-1",
        image_data=["image"],
        video_data=["video"],
        audio_data=["audio"],
        mm_processor_kwargs={"fps": 8},
        negative_prompt_ids=[3, 4],
        prompt_mask=prompt_mask,
        extra_prompt_ids={"encoder": [5]},
        negative_extra_prompt_ids={"encoder": [6]},
        priority=2,
    )

    assert result == "result"
    server._generate_strategy.generate.assert_awaited_once_with(
        prompt_ids=[1, 2],
        sampling_params={"temperature": 0.7},
        request_id="request-1",
        image_data=["image"],
        video_data=["video"],
        audio_data=["audio"],
        mm_processor_kwargs={"fps": 8},
        negative_prompt_ids=[3, 4],
        prompt_mask=prompt_mask,
        extra_prompt_ids={"encoder": [5]},
        negative_extra_prompt_ids={"encoder": [6]},
        priority=2,
    )


def test_strategies_preserve_mode_specific_quantization(monkeypatch):
    monkeypatch.setattr(
        ar_strategy_module.vLLMHttpServer,
        "_apply_quantization",
        lambda server: ("fp8", {"source": server}),
    )
    server = SimpleNamespace()

    assert ARStrategy(server).apply_quantization() == ("fp8", {"source": server})
    assert DiffusionStrategy(server).apply_quantization() == (None, {})


def test_strategies_preserve_platform_worker_extensions():
    server = SimpleNamespace()

    assert ARStrategy(server).worker_extension_cls("npu").endswith("vLLMOmniColocateWorkerExtension")
    assert DiffusionStrategy(server).worker_extension_cls("cuda").endswith("vLLMOmniColocateWorkerExtension")
    assert DiffusionStrategy(server).worker_extension_cls("npu").endswith("vLLMOmniNPUColocateWorkerExtension")


def test_ar_strategy_preserves_prompt_and_sampling_preprocessing():
    processor = SimpleNamespace(dedup_pad_tokens=lambda token_ids: token_ids[:2])
    server = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=8,
            prompt_length=4,
            response_length=4,
            repetition_penalty=1.1,
        ),
        model_config=SimpleNamespace(processor=processor),
    )
    strategy = ARStrategy(server)
    sampling_params = {"max_new_tokens": 10, "logprobs": True}

    prompt, params = strategy.preprocess_input(
        prompt_ids=[1, 2, 3],
        sampling_params=sampling_params,
        multi_modal_data={"image": ["image"]},
        lora_request=None,
        negative_prompt_ids=None,
        mm_processor_kwargs={"size": 224},
    )

    assert prompt == {
        "prompt_token_ids": [1, 2],
        "multi_modal_data": {"image": ["image"]},
        "mm_processor_kwargs": {"size": 224},
    }
    assert params.max_tokens == 6
    assert params.logprobs == 0
    assert params.repetition_penalty == 1.1


def test_ar_strategy_preserves_output_conversion():
    server = SimpleNamespace(global_steps=12)
    strategy = ARStrategy(server)
    completion = SimpleNamespace(
        token_ids=[7, 8],
        logprobs=[
            {7: SimpleNamespace(logprob=-0.25)},
            {8: SimpleNamespace(logprob=-0.5)},
        ],
        finish_reason="length",
        num_preempted=2,
    )
    final_res = SimpleNamespace(request_output=SimpleNamespace(outputs=[completion]))

    output = strategy.process_output(
        final_res,
        params=SimpleNamespace(logprobs=0),
        sampling_params={},
    )

    assert output.token_ids == [7, 8]
    assert output.log_probs == [-0.25, -0.5]
    assert output.stop_reason == "completed"
    assert output.num_preempted == 2
    assert output.extra_fields == {"global_steps": 12}


def test_ar_strategy_preserves_engine_kwarg_normalization(monkeypatch):
    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: None)
    strategy = ARStrategy(SimpleNamespace())
    engine_kwargs = {
        "output_mode": "ar",
        "custom_pipeline": "unused",
        "pipeline_name": "missing",
        "stage_init_timeout": 45,
        "stage_overrides": {"stage": {}},
    }

    strategy.preprocess_engine_kwargs(engine_kwargs)

    assert engine_kwargs == {
        "stage-init-timeout": 45,
        "stage-overrides": {"stage": {}},
        "init-timeout": 600,
    }


def test_ar_strategy_preserves_engine_argument_normalization():
    server = SimpleNamespace(config=SimpleNamespace(logprobs_mode="raw_logprobs"))
    strategy = ARStrategy(server)
    engine_args = {
        "compilation_config": {
            "keep": 1,
            "drop": None,
            "nested": {"keep": 2, "drop": None},
        }
    }

    strategy.prepare_engine_args(
        engine_args,
        Namespace(stage_init_timeout="45", init_timeout=None),
    )

    assert engine_args == {
        "stage_init_timeout": 45,
        "logprobs_mode": "raw_logprobs",
        "compilation_config": {"keep": 1, "nested": {"keep": 2}},
    }


def test_diffusion_strategy_preserves_engine_argument_preparation(monkeypatch):
    imported = []
    monkeypatch.setattr(diffusion_strategy_module, "import_external_libs", imported.append)
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_pipeline_path",
        staticmethod(lambda **kwargs: "package.Adapter"),
    )
    pipeline_cls = SimpleNamespace(__name__="Adapter", supports_request_batch=False)
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_class",
        staticmethod(lambda **kwargs: pipeline_cls),
    )
    server = SimpleNamespace(
        config=SimpleNamespace(
            external_lib=["extension"],
            step_execution=False,
            enable_prompt_embed_cache=True,
            prompt_embed_cache_size=16,
        ),
        model_config=SimpleNamespace(architecture="Architecture", algorithm="Algorithm"),
    )
    strategy = DiffusionStrategy(server)
    engine_args = {"max_num_seqs": 4}

    strategy.prepare_engine_args(engine_args, Namespace())

    assert imported == [["extension"]]
    assert engine_args == {
        "max_num_seqs": 1,
        "enable_dummy_pipeline": True,
        "custom_pipeline_args": {"pipeline_class": "package.Adapter"},
        "enable_prompt_embed_cache": True,
        "prompt_embed_cache_size": 16,
    }


def test_diffusion_strategy_preserves_multistage_prompt_shape():
    server = SimpleNamespace(engine=SimpleNamespace(default_sampling_params_list=["ar-stage", "diffusion-stage"]))
    strategy = DiffusionStrategy(server)
    prompt_mask = torch.tensor([True, False])

    prompt, params = strategy.preprocess_input(
        prompt_ids=[1, 2],
        sampling_params={"pipeline_private_arg": 7},
        multi_modal_data={"image": ["image"]},
        lora_request=None,
        negative_prompt_ids=[3, 4],
        prompt_mask=prompt_mask,
        extra_prompt_ids={"encoder": [5]},
        negative_extra_prompt_ids={"encoder": [6]},
    )

    assert prompt["prompt_token_ids"] == [1, 2]
    assert prompt["prompt_mask"] is prompt_mask
    assert prompt["modalities"] == ["image"]
    assert prompt["negative_prompt_ids"] == [3, 4]
    assert prompt["extra_prompt_ids"] == {"encoder": [5]}
    assert prompt["negative_extra_prompt_ids"] == {"encoder": [6]}
    assert prompt["multi_modal_data"] == {"image": ["image"]}
    assert prompt["extra_args"] == {"multi_modal_data": {"image": ["image"]}}
    assert params[0] == "ar-stage"
    assert params[-1].extra_args == {"pipeline_private_arg": 7}


def _joint_video_audio_final_res():
    video = torch.zeros(1, 3, 2, 8, 8, dtype=torch.uint8)
    audio = torch.zeros(1, 16)
    final_res = SimpleNamespace(
        images=[(video, audio)],
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        multimodal_output=None,
        request_output=None,
    )
    return final_res, audio


def test_diffusion_strategy_uses_declared_audio_sample_rate(monkeypatch):
    pipeline_cls = SimpleNamespace(
        diffusion_io_spec=DiffusionIOSpec(
            primary=MediaSpec("video"),
            auxiliary=(MediaSpec("audio", sample_rate=48000),),
        )
    )
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_class",
        staticmethod(lambda **kwargs: pipeline_cls),
    )
    server = SimpleNamespace(
        global_steps=1,
        model_config=SimpleNamespace(architecture="Architecture", algorithm="Algorithm"),
    )
    final_res, audio = _joint_video_audio_final_res()

    processed = DiffusionStrategy(server).process_output(final_res, None, {"output_type": "pt", "logprobs": False})

    # The audio sample rate comes from the adapter-declared DiffusionIOSpec,
    # not the strategy's legacy 32 kHz fallback.
    assert processed.extra_fields["audio_sample_rate"] == 48000
    # process_output unbatches the leading dimension of auxiliary media.
    torch.testing.assert_close(processed.extra_fields["audio"], audio[0])


def test_diffusion_strategy_omits_audio_sample_rate_without_declared_spec(monkeypatch):
    # Without an adapter-declared DiffusionIOSpec the strategy still surfaces the
    # auxiliary audio stream but attaches no model-specific sample-rate default.
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_class",
        staticmethod(lambda **kwargs: SimpleNamespace()),
    )
    server = SimpleNamespace(
        global_steps=1,
        model_config=SimpleNamespace(architecture="Architecture", algorithm="Algorithm"),
    )
    final_res, audio = _joint_video_audio_final_res()

    processed = DiffusionStrategy(server).process_output(final_res, None, {"output_type": "pt", "logprobs": False})

    torch.testing.assert_close(processed.extra_fields["audio"], audio[0])
    assert "audio_sample_rate" not in processed.extra_fields


@pytest.mark.parametrize(
    ("strategy_cls", "expected_extra_keys", "sampling_kwarg"),
    [
        # The AR strategy hands the Thinker's single ``SamplingParams`` to the engine as the
        # stage-0 ``sampling_params`` entry and pins the e2e final output to the text stage; the
        # diffusion strategy has already expanded its own params into a full per-stage list in
        # ``preprocess_input`` and passes that list.
        (ARStrategy, {"lora_request", "priority", "output_modalities"}, "sampling_params"),
        (DiffusionStrategy, set(), "sampling_params_list"),
    ],
)
@pytest.mark.asyncio
async def test_strategy_preserves_mode_specific_engine_call(strategy_cls, expected_extra_keys, sampling_kwarg):
    class _Engine:
        def generate(self, **kwargs):
            self.kwargs = kwargs

            async def _outputs():
                yield "first"
                yield "last"

            return _outputs()

    engine = _Engine()
    strategy = strategy_cls(SimpleNamespace(engine=engine))

    result = await strategy.run_generation(
        prompt={"prompt_token_ids": [1]},
        params=["params"],
        request_id="request-1",
        lora_request="lora",
        priority=3,
    )

    assert result == "last"
    assert engine.kwargs["request_id"] == "request-1"
    assert engine.kwargs[sampling_kwarg] == ["params"]
    assert set(engine.kwargs) - {"prompt", "request_id", sampling_kwarg} == expected_extra_keys


def _resolve_like_async_omni(
    num_stages: int,
    *,
    sampling_params: Any = None,
    sampling_params_list: Any = None,
) -> list[Any]:
    """Mirror the pinned engine's stage-0 splice plus ``resolve_sampling_params_list`` validation.

    Transcribed from ``vllm_omni/entrypoints/async_omni.py`` (the ``sampling_params`` forward) and
    ``vllm_omni/entrypoints/omni_base.py`` (the single-object rejection), so a candidate call shape
    is judged by the same rule the server uses.
    """
    if sampling_params_list is None and sampling_params is not None:
        if num_stages == 1:
            sampling_params_list = [sampling_params]
        else:
            default = [f"stage{i}-default" for i in range(num_stages)]
            default[0] = sampling_params
            sampling_params_list = default
    if sampling_params_list is None:
        return [f"stage{i}-default" for i in range(num_stages)]
    if isinstance(sampling_params_list, Sequence) and not isinstance(sampling_params_list, (str, bytes)):
        resolved = list(sampling_params_list)
    elif num_stages == 1:
        resolved = [sampling_params_list]
    else:
        raise ValueError(f"Expected {num_stages} sampling params, got a single sampling params object")
    if len(resolved) != num_stages:
        raise ValueError(f"Expected {num_stages} sampling params, got {len(resolved)}")
    return resolved


def _und_ar_stage_count() -> int:
    """Stages declared by the UND AR replica's deploy config, read from the recipe's own yaml."""
    deploy = Path(__file__).resolve().parents[4] / "examples/agenticllmgrpo_trainer/bagel/bagel_corl_deploy_ar.yaml"
    return len(yaml.safe_load(deploy.read_text())["stages"])


@pytest.mark.asyncio
async def test_ar_strategy_survives_the_multistage_und_ar_replica():
    """The UND AR replica is multi-stage, and a single-object ``sampling_params_list`` breaks it.

    ``bagel_corl_deploy_ar.yaml`` is ``pipeline: bagel_think`` -- a Thinker stage plus a DiT stage
    on the replica's second card -- so the AR engine reports ``num_stages == 2``. Routing the
    Thinker's single ``SamplingParams`` through ``sampling_params_list`` made the engine raise

        ValueError: Expected 2 sampling params, got a single sampling params object

    on every UND tool-call decode. That left each episode with no materializable trajectory and
    aborted the step in the sync replay buffer with "selected terminal groups with no materializable
    trajectories" -- a rollout-shaped failure that looks nothing like its sampling-params cause.
    """
    stages = _und_ar_stage_count()
    assert stages > 1, (
        "bagel_corl_deploy_ar.yaml is single-stage, so this guard's premise (a multi-stage AR "
        "replica) no longer holds; re-check how the AR strategy hands params to the engine."
    )

    class _Engine:
        num_stages = stages

        def generate(self, **kwargs):
            self.kwargs = kwargs
            # Resolve exactly the way the engine does, so an unexpanded single object raises here
            # the way it did on the real run instead of silently passing the guard.
            self.resolved = _resolve_like_async_omni(
                self.num_stages,
                sampling_params=kwargs.get("sampling_params"),
                sampling_params_list=kwargs.get("sampling_params_list"),
            )

            async def _outputs():
                yield "last"

            return _outputs()

    engine = _Engine()
    strategy = ARStrategy(SimpleNamespace(engine=engine))

    result = await strategy.run_generation(
        prompt={"prompt_token_ids": [1]},
        params="THINKER-PARAMS",
        request_id="request-1",
        lora_request=None,
        priority=0,
    )

    assert result == "last"
    # Stage 0 is the Thinker and receives our params; the later stages keep the deploy config's
    # own defaults rather than being forced to the Thinker's decode settings.
    assert engine.resolved == ["THINKER-PARAMS", "stage1-default"]
    assert "sampling_params_list" not in engine.kwargs
    # ...and the e2e final stage is pinned to the Thinker, so the DiT output cannot become the
    # request's "final" one (see the modality test below).
    assert engine.kwargs["output_modalities"] == ["text"]

    # The shape this test exists to forbid: the old single-object list is rejected even though it
    # was a list-like payload, because one object is not one-params-per-stage.
    with pytest.raises(ValueError, match="Expected 2 sampling params, got a single sampling params object"):
        _resolve_like_async_omni(2, sampling_params_list="THINKER-PARAMS")


def _think_final_output_types() -> list[str]:
    """``final_output_type`` of each ``final_output`` stage of the AR replica's pipeline.

    Preferred source is vLLM-Omni's own registry, so this guard breaks if the pipeline gains or
    reorders stages. Falls back to the pair ``bagel_think`` is documented with (Thinker, then DiT)
    when the registry cannot be imported.
    """
    try:
        from vllm_omni.config.pipeline_registry import BAGEL_THINK_PIPELINE

        types = [s.final_output_type for s in BAGEL_THINK_PIPELINE.stages if getattr(s, "final_output", False)]
        if types:
            return types
    except Exception:  # pragma: no cover - registry unavailable in a stripped environment
        pass
    return ["text", "image"]


def _final_stage_like_engine(final_output_types: list[str], output_modalities: list[str]) -> int:
    """Mirror ``get_final_stage_id_for_e2e``: the *last* stage whose type was requested."""
    for sid in range(len(final_output_types) - 1, -1, -1):
        if final_output_types[sid] in output_modalities:
            return sid
    return len(final_output_types) - 1


@pytest.mark.asyncio
async def test_the_text_modality_pin_keeps_the_dit_output_out_of_the_und_lane():
    """Without the modality pin the UND lane receives the DiT stage's token-less output.

    ``bagel_think`` declares **both** stages ``final_output`` (stage 0 text, stage 1 image), and
    the engine's default modality list is every stage's type. ``get_final_stage_id_for_e2e`` scans
    backwards from the last stage, so the default lands on the DiT -- whose output carries no
    ``outputs`` at all. ``ARStrategy.process_output`` then fails on

        RuntimeError: AR mode expects outputs with token IDs, but got None or empty.

    for every UND tool-call decode, which is why the UND lane produced no materializable
    trajectory and the sync replay buffer aborted the step.
    """
    final_output_types = _think_final_output_types()
    assert final_output_types[0] == "text" and final_output_types[-1] != "text", (
        "this guard assumes the AR replica's pipeline ends in a non-text (DiT) final stage; "
        f"got {final_output_types}. Re-derive the failure mode before trusting it."
    )

    # The default (no modality requested) is every final-output type -> the DiT wins.
    assert _final_stage_like_engine(final_output_types, final_output_types) != 0
    # Pinning text is what makes the Thinker the e2e final stage and its tokens the delivered.
    assert _final_stage_like_engine(final_output_types, ["text"]) == 0

    class _Engine:
        def generate(self, **kwargs):
            self.kwargs = kwargs

            async def _outputs():
                yield "output"

            return _outputs()

    engine = _Engine()
    await ARStrategy(SimpleNamespace(engine=engine)).run_generation(
        prompt={"prompt_token_ids": [1]},
        params="THINKER-PARAMS",
        request_id="request-1",
        lora_request=None,
        priority=0,
    )
    assert engine.kwargs["output_modalities"] == ["text"]
