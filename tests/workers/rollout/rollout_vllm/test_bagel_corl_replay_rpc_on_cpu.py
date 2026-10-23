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

import ast
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "verl_omni/workers/rollout/vllm_rollout/bagel_corl_replay.py"


def load():
    spec = importlib.util.spec_from_file_location("replay", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_und_scores_exact_fixed_tokens_on_serving_engine(monkeypatch):
    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Params))
    seen = []

    class Engine:
        async def generate(self, **kwargs):
            seen.append(kwargs)
            yield SimpleNamespace(
                finished=True,
                prompt_token_ids=[1, 2, 3, 4],
                prompt_logprobs=[
                    None,
                    {2: SimpleNamespace(logprob=-0.2)},
                    {3: SimpleNamespace(logprob=-0.3)},
                    {4: SimpleNamespace(logprob=-0.4)},
                ],
                stage_id=0,
                replica_id=9,
                error=None,
            )

    server = SimpleNamespace(
        engine=Engine(),
        replica_rank=3,
        node_rank=0,
        config=SimpleNamespace(max_model_len=32, logprobs_mode="raw_logprobs"),
    )
    request = dict(
        input_ids=[1, 2, 3, 4], target_mask=[0, 1, 0, 1], weight_version="caller-cannot-attest-loaded-weights"
    )
    result = asyncio.run(load().score_und(server, request))
    assert result["log_probs"] == [None, -0.2, None, -0.4]
    assert result["server_replica_id"] == 3 and result["stage_replica_id"] == 9
    assert result["weight_version"] is None and not result["publication_verified"]
    assert seen[0]["prompt"] == {"prompt_token_ids": [1, 2, 3, 4]}
    assert seen[0]["sampling_params"].prompt_logprobs == 0
    assert seen[0]["sampling_params"].max_tokens == 1
    assert seen[0]["output_modalities"] == ["text"]
    assert request["target_mask"] == [0, 1, 0, 1]


@pytest.mark.parametrize("change", ["multimodal", "too_long", "first_token", "missing_logp"])
def test_und_rejects_unsupported_or_incomplete_replay(monkeypatch, change):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))

    async def generate(**kwargs):
        yield SimpleNamespace(
            finished=True, prompt_token_ids=[1, 2], prompt_logprobs=[None, {}], stage_id=0, replica_id=0, error=None
        )

    server = SimpleNamespace(
        engine=SimpleNamespace(generate=generate),
        replica_rank=0,
        node_rank=0,
        config=SimpleNamespace(max_model_len=16, logprobs_mode="raw_logprobs"),
    )
    request = dict(input_ids=[1, 2], target_mask=[0, 1])
    if change == "multimodal":
        request["multi_modal_inputs"] = {"pixels": [1]}
    if change == "too_long":
        request["input_ids"] *= 5000
    if change == "first_token":
        request["target_mask"][0] = 1
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(load().score_und(server, request))


def gen_request():
    return dict(
        prompt="a cube",
        latents=torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]),
        timesteps=torch.tensor([1.0, 0.75]),
        kernel=dict(
            height=512,
            width=512,
            num_inference_steps=3,
            timestep_shift=3.0,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_interval=[0.0, 1.0],
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
            noise_level=0.8,
            sde_type="sde",
            include_logprob_normalizer=False,
        ),
    )


@pytest.mark.parametrize("real_scheduler", [False, True])
@pytest.mark.parametrize("execution_failure", [False, True])
def test_gen_uses_actual_adapter_fixed_next_sample_and_restores_state(monkeypatch, real_scheduler, execution_failure):
    calls = []

    class Inner:
        def set_begin_index(self, index):
            calls.append(("index", index))

        def step(self, **kwargs):
            calls.append(("scheduler", kwargs["sample"].clone(), kwargs["prev_sample"].clone()))
            assert kwargs["include_logprob_normalizer"] is False
            # Return a score derived from the supplied transition, not a stored rollout value.
            return kwargs["prev_sample"], (kwargs["prev_sample"] - kwargs["sample"]).square().mean().reshape(1)

    tree = ast.parse(
        (ROOT / "verl_omni/pipelines/bagel_flow_grpo/vllm_omni_rollout_adapter.py").read_text(encoding="utf-8")
    )
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_BagelSchedulerAdapter")
    namespace = dict(
        torch=torch,
        Optional=__import__("typing").Optional,
        FlowMatchSDEDiscreteScheduler=object,
        _AdapterStepOutput=lambda **kwargs: SimpleNamespace(**kwargs),
        _tp_tensor_parallel_group=lambda: None,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<real-scheduler-adapter>", "exec"), namespace)
    common_namespace = dict(torch=torch, BAGEL_TIMESTEP_SHIFT=3.0, FlowMatchSDEDiscreteScheduler=object)
    if real_scheduler:
        # Load the real production SDE methods without importing optional SciPy.
        # The harness supplies only diffusers' schedule bookkeeping API.
        class Schedule:
            def __init__(self):
                self._step_index = None

            @property
            def step_index(self):
                return self._step_index

            def set_shift(self, value):
                pass

            def set_begin_index(self, value):
                self._begin_index = value

            def set_timesteps(self, sigmas, timesteps, device=None):
                self.sigmas = torch.tensor([*sigmas, 0.0], device=device)
                self.timesteps = torch.tensor(timesteps, device=device)

            def _init_step_index(self, timestep):
                self._step_index = self._begin_index

        def unexpected_noise(*args, **kwargs):
            raise AssertionError("Fixed replay sampled replacement noise")

        sched_namespace = dict(
            torch=torch,
            FlowMatchEulerDiscreteScheduler=Schedule,
            Optional=__import__("typing").Optional,
            Literal=__import__("typing").Literal,
            math=__import__("math"),
            randn_tensor=unexpected_noise,
            FlowMatchSDEDiscreteSchedulerOutput=SimpleNamespace,
        )
        sched_tree = ast.parse((ROOT / "verl_omni/pipelines/schedulers/flow_match_sde.py").read_text(encoding="utf-8"))
        sched_cls = next(
            n for n in sched_tree.body if isinstance(n, ast.ClassDef) and n.name == "FlowMatchSDEDiscreteScheduler"
        )
        exec(compile(ast.Module(body=[sched_cls], type_ignores=[]), "<real-sde-methods>", "exec"), sched_namespace)
        sched_module = SimpleNamespace(FlowMatchSDEDiscreteScheduler=sched_namespace["FlowMatchSDEDiscreteScheduler"])
        inner = sched_module.FlowMatchSDEDiscreteScheduler()
        common_tree = ast.parse((ROOT / "verl_omni/pipelines/bagel_flow_grpo/common.py").read_text(encoding="utf-8"))
        funcs = [
            n
            for n in common_tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("setup_bagel_sigmas", "bagel_time_shift")
        ]
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<real-bagel-schedule>", "exec"), common_namespace)
        setup = common_namespace["setup_bagel_sigmas"]
    else:
        inner = Inner()
        setup = lambda scheduler, count, device=None, shift=3.0: None
    adapter = namespace["_BagelSchedulerAdapter"](inner)

    class State:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Batch:
        @staticmethod
        def make_batch(states):
            return SimpleNamespace(states=states, latents=states[0].latents)

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.inputs.data",
        SimpleNamespace(OmniDiffusionSamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.worker.utils", SimpleNamespace(StepRequestState=State))
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.worker.input_batch", SimpleNamespace(InputBatch=Batch))
    monkeypatch.setitem(
        sys.modules, "verl_omni.pipelines.bagel_flow_grpo.common", SimpleNamespace(setup_bagel_sigmas=setup)
    )

    class Pipeline:
        device = torch.device("cpu")
        od_config = SimpleNamespace(dtype=torch.float32)
        bagel = SimpleNamespace(
            _sp_size=1,
            parallel_config=SimpleNamespace(cfg_parallel_size=1),
            prepare_prompts=lambda **kwargs: ({"packed_text_ids": torch.tensor([7, 8, 9])}, [3], [3]),
        )
        tokenizer = object()
        new_token_ids = {"bos_token_id": 7, "eos_token_id": 9}
        scheduler = adapter
        scheduler_kwargs = {"sentinel": "preserved"}

        def prepare_encode(self, state):
            calls.append("prepare")
            torch.rand(1)
            state.latents = torch.zeros(1, 2)
            state.timesteps = torch.tensor([1.0, 0.75])
            state.extra = {"bagel_dts": torch.tensor([0.25, 0.75])}
            return state

        def denoise_step(self, batch, states):
            calls.append(("denoise", states[0].step_index, batch.latents.clone()))
            if execution_failure:
                raise RuntimeError("serving denoise failed")
            return torch.ones_like(batch.latents)

    pipeline = Pipeline()
    original_scheduler, original_kwargs = pipeline.scheduler, pipeline.scheduler_kwargs
    request = gen_request()
    original_latents = request["latents"].clone()
    rng = torch.get_rng_state().clone()
    if execution_failure:
        with pytest.raises(RuntimeError, match="serving denoise failed"):
            load().score_gen_pipeline(pipeline, request)
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(request["latents"], original_latents)
        assert pipeline.scheduler is original_scheduler and pipeline.scheduler_kwargs is original_kwargs
        return
    result = load().score_gen_pipeline(pipeline, request)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(request["latents"], original_latents)
    assert pipeline.scheduler is original_scheduler and pipeline.scheduler_kwargs is original_kwargs
    if real_scheduler:
        reference = sched_module.FlowMatchSDEDiscreteScheduler()
        setup(reference, 3, shift=3.0)
        expected = []
        for index, timestep in enumerate((1.0, 0.75)):
            reference.set_begin_index(index)
            reference._step_index = None
            out = reference.step(
                model_output=torch.ones(1, 1, 2),
                timestep=torch.tensor(timestep),
                sample=original_latents[index].unsqueeze(0),
                prev_sample=original_latents[index + 1].unsqueeze(0),
                noise_level=0.8,
                include_logprob_normalizer=False,
            )
            expected.append(float(out.log_prob))
        assert result["log_probs"] == pytest.approx(expected)
    else:
        assert result["log_probs"] == [4.0, 4.0]
    assert calls[0] == "prepare"
    assert [call[1] for call in calls if isinstance(call, tuple) and call[0] == "denoise"] == [0, 1]
    actual = [call for call in calls if isinstance(call, tuple) and call[0] == "scheduler"]
    if not real_scheduler:
        assert torch.equal(actual[0][2], original_latents[1].unsqueeze(0))
        assert torch.equal(actual[1][2], original_latents[2].unsqueeze(0))


@pytest.mark.parametrize("change", ["nan", "shape", "too_many", "image", "think", "kv"])
def test_gen_rejects_unsupported_inputs_before_pipeline_execution(change):
    request = gen_request()
    if change == "nan":
        request["latents"][0, 0, 0] = float("nan")
    if change == "shape":
        request["latents"] = torch.ones(2, 1, 2)
    if change == "too_many":
        request["timesteps"] = torch.ones(100)
    if change == "image":
        request["image"] = "unsupported-image"
    if change == "think":
        request["think"] = True
    if change == "kv":
        request["past_key_values"] = {"cache": 1}
    with pytest.raises((ValueError, RuntimeError)):
        load().score_gen_pipeline(SimpleNamespace(), request)


def extract_method(file, class_name, method_name):
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == method_name)


def test_real_worker_bridge_uses_only_live_diffusion_pipeline(monkeypatch):
    module = load()
    calls = []

    class Context:
        def __init__(self, **kwargs):
            calls.append(("context", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Pipeline:
        pass

    pipeline = Pipeline()
    pipeline.od_config = SimpleNamespace(cache_backend=None)
    monkeypatch.setitem(
        sys.modules, "vllm_omni.diffusion.forward_context", SimpleNamespace(set_forward_context=Context)
    )
    monkeypatch.setitem(
        sys.modules,
        "verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter",
        SimpleNamespace(BagelPipelineWithLogProb=Pipeline),
    )
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.bagel_corl_replay", module)
    monkeypatch.setattr(
        module,
        "score_gen_pipeline",
        lambda live, request: calls.append(("score", live, request))
        or dict(supported=True, lane="gen", log_probs=[-0.5]),
    )
    node = extract_method(
        "verl_omni/workers/rollout/vllm_rollout/utils.py",
        "vLLMOmniColocateWorkerExtension",
        "score_bagel_corl_gen_fixed_inputs",
    )
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<worker-bridge>", "exec"), namespace)
    runner = SimpleNamespace(pipeline=pipeline, vllm_config="actual-vllm-config", od_config=pipeline.od_config)
    worker = SimpleNamespace(model_runner=runner)
    result = namespace[node.name](worker, {"probe": "fixed"})
    assert calls[0][0] == "context"
    assert calls[0][1]["vllm_config"] == "actual-vllm-config"
    assert calls[1] == ("score", pipeline, {"probe": "fixed"})
    assert result["log_probs"] == [-0.5]
    worker.model_runner.pipeline = object()
    with pytest.raises(RuntimeError, match="BAGEL"):
        namespace[node.name](worker, {})


def test_real_server_bridge_routes_gen_to_all_workers_and_rejects_unsupported_reply(monkeypatch):
    module = load()
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.bagel_corl_replay", module)

    class AR:
        pass

    class Diffusion:
        pass

    calls = []

    async def rpc(**kwargs):
        calls.append(kwargs)
        return [
            [
                dict(
                    supported=True,
                    lane="gen",
                    log_probs=[-0.5],
                    context_hash="ctx",
                    kernel_config_hash="kernel",
                    serving_process={"hostname": "host", "pid": 1},
                ),
                dict(
                    supported=True,
                    lane="gen",
                    log_probs=[-0.5],
                    context_hash="ctx",
                    kernel_config_hash="kernel",
                    serving_process={"hostname": "host", "pid": 2},
                ),
            ]
        ]

    node = extract_method(
        "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py",
        "vLLMOmniHttpServer",
        "score_bagel_corl_fixed_inputs",
    )
    namespace = dict(ARStrategy=AR, DiffusionStrategy=Diffusion, asyncio=asyncio)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<server-bridge>", "exec"), namespace)
    server = SimpleNamespace(
        _generate_strategy=Diffusion(),
        replica_rank=7,
        node_rank=0,
        replica_world_size=2,
        engine=SimpleNamespace(collective_rpc=rpc, default_sampling_params_list=[object()], request_states={}),
    )
    result = asyncio.run(namespace[node.name](server, {"lane": "gen"}))
    assert result["server_replica_id"] == 7
    assert len(result["worker_results"]) == 2
    assert calls[0]["method"] == "score_bagel_corl_gen_fixed_inputs"
    assert calls[0]["stage_ids"] == [0]

    async def unsupported(**kwargs):
        return [{"supported": False, "error": "missing method"}]

    server.engine.collective_rpc = unsupported
    with pytest.raises(RuntimeError, match="unsupported"):
        asyncio.run(namespace[node.name](server, {"lane": "gen"}))
    server.engine.request_states = {"busy": object()}
    with pytest.raises(RuntimeError, match="quiescent"):
        asyncio.run(namespace[node.name](server, {"lane": "gen"}))


@pytest.mark.parametrize(
    "field",
    [
        "negative_prompt",
        "negative_prompt_ids",
        "negative_prompt_embeds",
        "cfg_text_scale",
        "cfg_branch_past_key_values",
        "cfg_branch_kv_metadata",
        "cfg_active_branch",
        "cfg_branch_roles",
        "cfg_text_past_key_values",
        "cfg_img_past_key_values",
        "cfg_text_kv_metadata",
        "cfg_img_kv_metadata",
        "kv_transfer_params",
        "kv_computed_tokens",
        "diffusion_kv_metadata",
        "prompt_embeds",
        "additional_information",
        "unknown_context_override",
    ],
)
def test_gen_never_silently_ignores_conditioning_or_cache_overrides(field):
    request = gen_request()
    request[field] = "unsupported-context"
    with pytest.raises(ValueError, match="Unsupported"):
        load().score_gen_pipeline(SimpleNamespace(), request)


def test_probe_excludes_normal_generate_and_restores_admission_after_failure(monkeypatch):
    module = load()
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.bagel_corl_replay", module)

    class AR:
        pass

    class Diffusion:
        async def generate(self, **kwargs):
            return "normal-generation"

    names = ("score_bagel_corl_fixed_inputs", "generate")
    methods = [
        extract_method("verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py", "vLLMOmniHttpServer", name)
        for name in names
    ]
    namespace = dict(
        ARStrategy=AR,
        DiffusionStrategy=Diffusion,
        asyncio=asyncio,
        Optional=__import__("typing").Optional,
        Any=__import__("typing").Any,
        torch=torch,
        DiffusionOutput=object,
        TokenOutput=object,
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<server-admission>", "exec"), namespace)

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def rpc(**kwargs):
            started.set()
            await release.wait()
            raise RuntimeError("serving RPC failed")

        server = SimpleNamespace(
            _generate_strategy=Diffusion(),
            replica_rank=7,
            node_rank=0,
            replica_world_size=2,
            _inflight_generations=0,
            engine=SimpleNamespace(collective_rpc=rpc, default_sampling_params_list=[object()], request_states={}),
        )
        task = asyncio.create_task(namespace[names[0]](server, {"lane": "gen"}))
        await started.wait()
        assert server._bagel_corl_fixed_scoring is True
        with pytest.raises(RuntimeError, match="fixed scoring"):
            await namespace["generate"](server, prompt_ids=[1], sampling_params={}, request_id="normal")
        assert server._inflight_generations == 0
        release.set()
        with pytest.raises(RuntimeError, match="serving RPC failed"):
            await task
        assert server._bagel_corl_fixed_scoring is False
        assert (
            await namespace["generate"](server, prompt_ids=[1], sampling_params={}, request_id="normal")
            == "normal-generation"
        )
        assert server._inflight_generations == 0
        server._inflight_generations = 1
        with pytest.raises(RuntimeError, match="quiescent"):
            await namespace[names[0]](server, {"lane": "gen"})
        server._inflight_generations = 0
        del server.engine.request_states
        with pytest.raises(RuntimeError, match="observable"):
            await namespace[names[0]](server, {"lane": "gen"})

    asyncio.run(scenario())
