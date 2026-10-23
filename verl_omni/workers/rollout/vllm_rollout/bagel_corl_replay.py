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

"""Bounded serving-side BAGEL scoring probes for the installed vLLM 0.30 APIs.

These probes score supplied inputs using live serving weights. They do not attest
publication versions: weight_version/publication_index remain null until a
verified receiver-side publication receipt is wired. Never use this result alone
as a frozen-policy publication gate. Multimodal, injected-KV, think, paged-cache,
and CFG/sequence-parallel routes are deliberately unsupported. GEN TP is executed
through collective_rpc on every worker, never by invoking the trained actor.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import uuid

import torch
from verl.utils.device import get_device_id, get_device_name

MAX_TOKENS = 4096
MAX_TRANSITIONS = 32
MAX_LATENT_ELEMENTS = 4_194_304


def _hash(value):
    def encode(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            return {"dtype": str(data.dtype), "shape": list(data.shape), "values": data.tolist()}
        raise TypeError(type(item).__name__)

    return hashlib.sha256(json.dumps(value, default=encode, sort_keys=True, allow_nan=False).encode()).hexdigest()


def observation():
    versions = {}
    for package in ("torch", "vllm", "vllm-omni"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(
        weight_version=None,
        publication_index=None,
        publication_verified=False,
        package_versions=versions,
        serving_process={"hostname": socket.gethostname(), "pid": os.getpid()},
        scope="Live serving scores only; loaded weight version and publication coverage are unverified",
        admission_scope="Server-actor generate admission only; direct HTTP/engine traffic requires external isolation",
    )


def _reject_features(request, allowed):
    unsupported = set(request) - allowed
    if unsupported:
        raise ValueError(f"Unsupported fixed-input replay fields: {sorted(unsupported)}")


async def score_und(server, request):
    """Use scheduled prompt scoring, preserving the complete exact text transcript."""
    _reject_features(
        request,
        {
            "lane",
            "input_ids",
            "target_mask",
            "position_ids",
            "attention_mask",
            "weight_version",
            "publication_index",
        },
    )
    ids, mask = request.get("input_ids"), request.get("target_mask")
    if not isinstance(ids, list) or not 2 <= len(ids) <= MAX_TOKENS:
        raise ValueError("UND replay requires 2..4096 exact input_ids")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("UND replay token IDs must be nonnegative integers")
    if not isinstance(mask, list) or len(mask) != len(ids) or any(value not in (0, 1) for value in mask):
        raise ValueError("UND replay requires a binary target_mask aligned with input_ids")
    if mask[0] or not any(mask):
        raise ValueError("Cannot score first token without a causal prefix, or an empty target mask")
    if len(ids) >= server.config.max_model_len or server.node_rank != 0:
        raise ValueError("UND replay exceeds engine context limit or targets a non-head server")
    if getattr(server.config, "logprobs_mode", None) != "raw_logprobs":
        raise ValueError("UND fixed prompt replay requires verified raw_logprobs serving configuration")
    attention_mask = request.get("attention_mask")
    if attention_mask is not None and attention_mask != [1] * len(ids):
        raise ValueError("Custom UND attention masks are unsupported by text prompt replay")
    positions = request.get("position_ids")
    if positions is not None and positions != list(range(len(ids))):
        raise ValueError("Custom UND position_ids are unsupported by text prompt replay")
    from vllm import SamplingParams

    params = SamplingParams(prompt_logprobs=0, max_tokens=1, temperature=0.0, ignore_eos=True)
    final = None
    async for output in server.engine.generate(
        prompt={"prompt_token_ids": list(ids)},
        sampling_params=params,
        request_id="bagel-fixed-und-" + uuid.uuid4().hex,
        output_modalities=["text"],
    ):
        if getattr(output, "error", None):
            raise RuntimeError(f"UND serving replay failed: {output.error}")
        if output.finished:
            final = output
    if final is None or final.prompt_token_ids != ids:
        raise RuntimeError("UND serving replay did not preserve exact input tokens")
    if final.stage_id is None or final.replica_id is None:
        raise RuntimeError("UND serving replay omitted actual stage/replica identity")
    entries = final.prompt_logprobs
    if entries is None or len(entries) != len(ids):
        raise RuntimeError("UND serving replay omitted complete prompt logprobs")
    logps = [None] * len(ids)
    for index, active in enumerate(mask):
        if active:
            entry = entries[index]
            if not isinstance(entry, dict) or ids[index] not in entry:
                raise RuntimeError(f"UND serving replay omitted target token at position {index}")
            logps[index] = float(entry[ids[index]].logprob)
            if not math.isfinite(logps[index]):
                raise RuntimeError("Nonfinite UND serving replay logprob")
    kernel = dict(
        route="vllm_prompt_logprobs",
        logprob_convention="raw_policy",
        max_model_len=server.config.max_model_len,
        logprobs_mode=server.config.logprobs_mode,
    )
    return dict(
        supported=True,
        lane="und",
        server_replica_id=server.replica_rank,
        stage_id=final.stage_id,
        stage_replica_id=final.replica_id,
        input_ids=list(ids),
        target_mask=list(mask),
        log_probs=logps,
        context_hash=_hash({"input_ids": ids, "target_mask": mask}),
        kernel=kernel,
        kernel_config_hash=_hash(kernel),
        **observation(),
    )


def score_gen_pipeline(pipeline, request):
    """Score fixed stochastic transitions through BAGEL's serving denoiser and scheduler."""
    _reject_features(
        request,
        {
            "lane",
            "prompt",
            "latents",
            "timesteps",
            "kernel",
            "conditioning_token_ids",
            "weight_version",
            "publication_index",
        },
    )
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt or len(prompt.encode()) > 32768:
        raise ValueError("GEN replay requires one bounded exact text prompt")
    latents = torch.as_tensor(request.get("latents"))
    times = torch.as_tensor(request.get("timesteps"))
    if latents.ndim != 3 or times.ndim != 1 or not 1 <= times.numel() <= MAX_TRANSITIONS:
        raise ValueError("GEN replay requires [T+1,L,D] latents and bounded [T] timesteps")
    if latents.shape[0] != times.numel() + 1 or latents.numel() > MAX_LATENT_ELEMENTS or min(latents.shape) < 1:
        raise ValueError("Invalid or oversized GEN latent shape")
    if latents.dtype != torch.float32 or not torch.isfinite(latents).all() or not torch.isfinite(times).all():
        raise ValueError("GEN fixed latents must be finite float32; timesteps must be finite")
    if (times <= 0).any():
        raise ValueError("GEN stochastic timesteps must be positive")
    kernel = copy.deepcopy(request.get("kernel") or {})
    required = {
        "height",
        "width",
        "num_inference_steps",
        "timestep_shift",
        "cfg_text_scale",
        "cfg_img_scale",
        "cfg_interval",
        "cfg_renorm_type",
        "cfg_renorm_min",
        "noise_level",
        "sde_type",
        "include_logprob_normalizer",
    }
    if set(kernel) != required:
        raise ValueError("GEN replay requires the explicit supported kernel configuration only")
    _hash(kernel)  # Reject nonfinite/non-JSON configuration before model execution.
    if kernel["sde_type"] != "sde" or kernel["include_logprob_normalizer"] is not False:
        raise ValueError("GEN replay supports BAGEL quadratic SDE logprobs only")
    for key in ("timestep_shift", "cfg_text_scale", "cfg_img_scale", "cfg_renorm_min", "noise_level"):
        if type(kernel[key]) not in (int, float) or not math.isfinite(kernel[key]):
            raise ValueError(f"Invalid numeric GEN kernel field: {key}")
    interval = kernel["cfg_interval"]
    if (
        not isinstance(interval, list | tuple)
        or len(interval) != 2
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in interval)
        or not 0 <= interval[0] <= interval[1] <= 1
    ):
        raise ValueError("Invalid GEN CFG interval")
    if kernel["timestep_shift"] != 3.0:
        raise ValueError("Fixed GEN replay supports only the serving adapter's actual timestep_shift=3")
    if (
        kernel["timestep_shift"] <= 0
        or kernel["cfg_text_scale"] <= 0
        or kernel["cfg_img_scale"] <= 0
        or not 0 <= kernel["cfg_renorm_min"] <= 1
        or kernel["cfg_renorm_type"] not in ("global", "channel", "text_channel")
    ):
        raise ValueError("Invalid GEN CFG or timestep-shift configuration")
    if (
        not 0 < kernel["noise_level"] <= 1
        or type(kernel["num_inference_steps"]) is not int
        or not 2 <= kernel["num_inference_steps"] <= 100
    ):
        raise ValueError("Invalid GEN noise level or schedule length")
    if any(
        type(kernel[key]) is not int or not 16 <= kernel[key] <= 1024 or kernel[key] % 16 for key in ("height", "width")
    ):
        raise ValueError("Invalid GEN image geometry")
    if (
        getattr(pipeline.bagel, "_sp_size", 1) != 1
        or getattr(pipeline.bagel.parallel_config, "cfg_parallel_size", 1) != 1
    ):
        raise ValueError("GEN fixed replay does not support SP or CFG parallel execution")
    for method in ("prepare_encode", "denoise_step"):
        if not callable(getattr(pipeline, method, None)):
            raise RuntimeError(f"Installed BAGEL pipeline lacks {method}")
    if not hasattr(pipeline.scheduler, "_inner"):
        raise RuntimeError("GEN replay requires the BAGEL FlowGRPO serving scheduler adapter")

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from verl_omni.pipelines.bagel_flow_grpo.common import setup_bagel_sigmas

    clean_prompt = prompt.removeprefix("<|im_start|>").removesuffix("<|im_end|>")
    prepared_text, _, _ = pipeline.bagel.prepare_prompts(
        curr_kvlens=[0],
        curr_rope=[0],
        prompts=[clean_prompt],
        tokenizer=pipeline.tokenizer,
        new_token_ids=pipeline.new_token_ids,
    )
    conditioning_ids = prepared_text["packed_text_ids"].tolist()
    if len(conditioning_ids) > MAX_TOKENS:
        raise ValueError("GEN conditioning exceeds the bounded token limit")
    expected_ids = request.get("conditioning_token_ids")
    if expected_ids is not None and expected_ids != conditioning_ids:
        raise ValueError("GEN conditioning token IDs differ from the exact supplied context")

    extra_args = {
        key: value
        for key, value in kernel.items()
        if key not in ("height", "width", "num_inference_steps", "include_logprob_normalizer")
    }
    sampling = OmniDiffusionSamplingParams(
        height=kernel["height"],
        width=kernel["width"],
        num_inference_steps=kernel["num_inference_steps"],
        extra_args=extra_args,
    )
    saved_scheduler, saved_kwargs = pipeline.scheduler, pipeline.scheduler_kwargs
    device = torch.device(pipeline.device)
    devices = (
        [device.index if device.index is not None else get_device_id()] if device.type == get_device_name() else []
    )
    scores = []
    try:
        # Preparation draws initial noise; preserve global CPU/device RNG and discard
        # that noise. All model evaluations below use only caller-supplied latents.
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            pipeline.scheduler = type(saved_scheduler)(copy.deepcopy(saved_scheduler._inner))
            pipeline.scheduler_kwargs = {}
            setup_bagel_sigmas(
                pipeline.scheduler._inner, kernel["num_inference_steps"], device=device, shift=kernel["timestep_shift"]
            )
            state = StepRequestState(
                request_id="bagel-fixed-gen-" + uuid.uuid4().hex, sampling=sampling, prompt={"prompt": prompt}
            )
            state = pipeline.prepare_encode(state)
            if state.latents is None or tuple(state.latents.shape) != tuple(latents.shape[1:]):
                raise ValueError("Recorded GEN latent geometry differs from the serving conditioning")
            schedule = state.timesteps
            selected = []
            for timestep in times:
                matches = torch.nonzero(
                    torch.isclose(schedule.float().cpu(), timestep.float().cpu(), rtol=0.0, atol=1e-7)
                ).reshape(-1)
                if matches.numel() != 1:
                    raise ValueError("Recorded GEN timestep does not match the serving schedule")
                selected.append(int(matches[0]))
            if any(right != left + 1 for left, right in zip(selected, selected[1:], strict=False)):
                raise ValueError("GEN replay requires contiguous recorded stochastic transitions")
            for index, step in enumerate(selected):
                state.step_index = step
                state.latents = latents[index].to(device).clone()
                batch = InputBatch.make_batch([state])
                velocity = pipeline.denoise_step(batch, states=[state])
                if velocity.shape != state.latents.shape or not torch.isfinite(velocity).all():
                    raise RuntimeError("Serving denoiser returned invalid velocity")
                scheduler = pipeline.scheduler
                scheduler.begin_forward(
                    sde_window=(step, step + 1), noise_level=kernel["noise_level"], return_logprobs=True
                )
                scheduler._step_counter = step
                scheduler._inner.set_begin_index(step)
                scheduler._inner._step_index = None
                out = scheduler.step(
                    velocity,
                    schedule[step],
                    state.latents,
                    state.extra["bagel_dts"][step],
                    prev_sample=latents[index + 1].to(device).clone(),
                    sde_type=kernel["sde_type"],
                    include_logprob_normalizer=False,
                )
                if out.log_prob is None or not torch.isfinite(out.log_prob).all():
                    raise RuntimeError("Serving scheduler omitted finite fixed-transition logprobs")
                scores.append(float(out.log_prob))
    finally:
        pipeline.scheduler, pipeline.scheduler_kwargs = saved_scheduler, saved_kwargs
    kernel["serving_dtype"] = str(pipeline.od_config.dtype)
    return dict(
        supported=True,
        lane="gen",
        log_probs=scores,
        timesteps=times.tolist(),
        conditioning_token_ids=conditioning_ids,
        context_hash=_hash(
            {"prompt": prompt, "conditioning_token_ids": conditioning_ids, "latents": latents, "timesteps": times}
        ),
        kernel=kernel,
        kernel_config_hash=_hash(kernel),
        **observation(),
    )


def validate_gen_worker_results(results, expected_workers):
    """Reject best-effort TODO/error RPC responses instead of treating them as scores."""
    flat = []

    def visit(value):
        if isinstance(value, list | tuple):
            for item in value:
                visit(item)
        elif isinstance(value, dict) and value.get("supported") is True and value.get("lane") == "gen":
            scores = value.get("log_probs")
            if not scores or not all(isinstance(item, int | float) and math.isfinite(item) for item in scores):
                raise RuntimeError("GEN replay returned invalid serving scores")
            if not value.get("context_hash") or not value.get("kernel_config_hash"):
                raise RuntimeError("GEN replay omitted context/kernel identity")
            flat.append(value)
        else:
            raise RuntimeError(f"GEN serving replay unsupported or incomplete: {value!r}")

    visit(results)
    if len(flat) != expected_workers:
        raise RuntimeError("GEN replay did not return every physical TP worker result")
    identities = [value.get("serving_process") or {} for value in flat]
    if any(not item.get("hostname") or not isinstance(item.get("pid"), int) for item in identities):
        raise RuntimeError("GEN replay omitted observed worker process identity")
    if len({(item["hostname"], item["pid"]) for item in identities}) != len(flat):
        raise RuntimeError("GEN replay duplicated a worker process response")
    if len({(value["context_hash"], value["kernel_config_hash"]) for value in flat}) != 1:
        raise RuntimeError("GEN TP workers scored different contexts or kernels")
    return flat
