#!/usr/bin/env python3
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
"""Small frozen Qwen-Image HTTP service for the agentic GRPO tool loop.

Memory modes:
  full               BF16 pipeline resident on one large GPU.
  balanced           Split BF16 modules across all visible CUDA devices
                     (``device_map="balanced"``). Use when one GPU OOMs.
  model_offload      Component-level CPU offload; practical on smaller GPUs.
  sequential_offload Layer-level CPU offload; lowest VRAM, much slower.
  mmdit_nf4          NF4-quantize only Qwen-Image's MMDiT transformer and
                     component-offload the remaining frozen modules.

The MMDiT cannot generate by itself: the frozen text encoder, VAE, and scheduler
are still required. ``mmdit_nf4`` is therefore the useful low-memory
interpretation of "run its MMDiT", rather than an incomplete transformer-only
service.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from diffusers import BitsAndBytesConfig, PipelineQuantizationConfig, QwenImagePipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("qwen_image_tool_server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_ID = os.getenv("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
MEMORY_MODE = os.getenv("QWEN_IMAGE_MEMORY_MODE", "model_offload").strip().lower()
ALLOWED_MEMORY_MODES = {"full", "balanced", "model_offload", "sequential_offload", "mmdit_nf4"}

_pipe: QwenImagePipeline | None = None
_generate_lock = threading.Lock()


def _generator_device() -> str:
    """Pick a CUDA device for the RNG; balanced maps may leave cuda:0 empty."""
    if not torch.cuda.is_available():
        return "cpu"
    if MEMORY_MODE == "balanced" and torch.cuda.device_count() > 1:
        return "cuda:0"
    return "cuda"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    negative_prompt: str | None = None
    width: int | None = Field(default=None, ge=256, le=2048, multiple_of=16)
    height: int | None = Field(default=None, ge=256, le=2048, multiple_of=16)
    num_inference_steps: int | None = Field(default=None, ge=1, le=100)
    true_cfg_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None


class GenerateResponse(BaseModel):
    image_base64: str
    text: str
    reward: float = 0.0
    backend: Literal["qwen_image"] = "qwen_image"
    model: str
    memory_mode: str
    seed: int
    width: int
    height: int
    latency_s: float


def _load_pipeline() -> QwenImagePipeline:
    if MEMORY_MODE not in ALLOWED_MEMORY_MODES:
        raise ValueError(
            f"Unknown QWEN_IMAGE_MEMORY_MODE={MEMORY_MODE!r}; choose one of {sorted(ALLOWED_MEMORY_MODES)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image tool service requires a CUDA GPU")

    load_kwargs: dict = {"torch_dtype": torch.bfloat16}
    if MEMORY_MODE == "mmdit_nf4":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "QWEN_IMAGE_MEMORY_MODE=mmdit_nf4 requires bitsandbytes; "
                "install it in the active environment or use sequential_offload"
            ) from exc
        transformer_quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # These output/modulation layers are quality-sensitive.
            llm_int8_skip_modules=[
                "time_text_embed",
                "img_in",
                "norm_out",
                "proj_out",
                "img_mod",
                "txt_mod",
            ],
        )
        load_kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_mapping={"transformer": transformer_quant}
        )

    n_visible = torch.cuda.device_count()
    logger.info(
        "Loading frozen %s with memory_mode=%s (visible_cuda=%d)",
        MODEL_ID,
        MEMORY_MODE,
        n_visible,
    )
    if MEMORY_MODE == "balanced":
        if n_visible < 2:
            raise RuntimeError(
                "QWEN_IMAGE_MEMORY_MODE=balanced needs >=2 visible GPUs; "
                "set CUDA_VISIBLE_DEVICES to two free ids (e.g. 0,1)"
            )
        # Spread text encoder / transformer / VAE across the visible devices.
        load_kwargs["device_map"] = "balanced"
        pipe = QwenImagePipeline.from_pretrained(MODEL_ID, **load_kwargs)
    else:
        pipe = QwenImagePipeline.from_pretrained(MODEL_ID, **load_kwargs)
        if MEMORY_MODE == "full":
            pipe.to("cuda")
        elif MEMORY_MODE == "sequential_offload":
            pipe.enable_sequential_cpu_offload()
        else:
            # model_offload and mmdit_nf4
            pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _pipe
    _pipe = _load_pipeline()
    yield
    _pipe = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Frozen Qwen-Image Tool", version="1", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "ok": _pipe is not None,
        "model": MODEL_ID,
        "memory_mode": MEMORY_MODE,
        "cuda": torch.cuda.is_available(),
        "visible_cuda": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Qwen-Image pipeline is not loaded")

    width = request.width or int(os.getenv("QWEN_IMAGE_WIDTH", "512"))
    height = request.height or int(os.getenv("QWEN_IMAGE_HEIGHT", "512"))
    steps = request.num_inference_steps or int(os.getenv("QWEN_IMAGE_STEPS", "20"))
    cfg = request.true_cfg_scale
    if cfg is None:
        cfg = float(os.getenv("QWEN_IMAGE_TRUE_CFG_SCALE", "4.0"))
    seed = request.seed
    if seed is None:
        seed = int(os.getenv("QWEN_IMAGE_SEED", "42"))
    negative = request.negative_prompt
    if negative is None:
        negative = os.getenv(
            "QWEN_IMAGE_NEGATIVE_PROMPT",
            "blurry, low quality, distorted anatomy, duplicate objects, unreadable text",
        )

    started = time.perf_counter()
    try:
        # Single pipeline: serialize generates. Concurrent agent workers queue here,
        # which is why GPU util flickers and HTTP clients look "slow to refresh".
        with _generate_lock, torch.inference_mode():
            queue_wait_s = time.perf_counter() - started
            if queue_wait_s > 1.0:
                logger.info("generate queued %.1fs before acquiring pipeline lock", queue_wait_s)
            gen_started = time.perf_counter()
            result = _pipe(
                prompt=request.prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                generator=torch.Generator(device=_generator_device()).manual_seed(seed),
            )
            image = result.images[0]
            infer_s = time.perf_counter() - gen_started
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "generate ok wait=%.1fs infer=%.1fs steps=%d size=%dx%d prompt=%r",
                queue_wait_s,
                infer_s,
                steps,
                width,
                height,
                request.prompt[:80],
            )
    except Exception as exc:  # noqa: BLE001 - convert model failure to HTTP diagnostics
        logger.exception("Qwen-Image generation failed")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    elapsed = time.perf_counter() - started
    return GenerateResponse(
        image_base64=encoded,
        text="Frozen Qwen-Image generated a candidate. Inspect the attached image before revising the prompt.",
        model=MODEL_ID,
        memory_mode=MEMORY_MODE,
        seed=seed,
        width=image.width,
        height=image.height,
        latency_s=elapsed,
    )
