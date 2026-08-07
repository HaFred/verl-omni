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
"""Frozen Qwen3-VL reflect sidecar: correctness + aesthetics vs user request.

Loads the Instruct VL weights once at process start and keeps them resident.
``judge_image`` in the agent loop calls ``POST /reflect``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("qwen_vl_reflect_server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_ID = os.getenv("AGENTIC_REFLECT_VLM_PATH") or os.getenv("MODEL_PATH") or "Qwen/Qwen3-VL-2B-Instruct"
GOOD_ENOUGH_THRESHOLD = float(os.getenv("AGENTIC_REFLECT_GOOD_ENOUGH", "0.72"))

_model = None
_processor = None
_infer_lock = threading.Lock()

CORRECTNESS_QUESTIONS = {
    "subject_entities": "Are the requested primary subjects/entities visibly present and recognizable?",
    "attributes": "Are requested attributes such as color, count, material, text, and identity correct?",
    "relations_layout": "Are requested actions, spatial relations, and layout/composition constraints correct?",
    "scene_context": "Does the environment, setting, style, and overall scene match the request?",
    "completeness": "Is the request fully satisfied without missing requested details or contradictory extras?",
}
AESTHETICS_QUESTIONS = {
    "composition": "Is the composition balanced with a clear focal hierarchy and intentional framing?",
    "lighting": "Are lighting, exposure, contrast, and depth visually effective?",
    "color": "Are color harmony, saturation, and tonal relationships pleasing and coherent?",
    "fidelity": "Is the image sharp and spatially coherent, without obvious generation artifacts or distortions?",
    "appeal": "Does the image have strong overall visual appeal and professional finish?",
}


class ReflectRequest(BaseModel):
    user_request: str = Field(min_length=1, max_length=4096)
    image_prompt: str = Field(default="", max_length=4096)
    image_path: str | None = None
    image_base64: str | None = None
    notes: str = ""


class ReflectResponse(BaseModel):
    correctness: float
    aesthetics: float
    correctness_scores: dict[str, float]
    aesthetics_scores: dict[str, float]
    good_enough: bool
    findings: str
    suggested_fixes: str
    match: float
    backend: str = "qwen3_vl"
    model: str
    latency_s: float
    raw: str = ""


def _load_model() -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL reflect service requires a CUDA GPU")
    logger.info("Loading frozen reflect VLM once: %s", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    return model, processor


def _decode_image(request: ReflectRequest) -> Image.Image:
    if request.image_path:
        path = Path(request.image_path)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"image_path not found: {request.image_path}")
        return Image.open(path).convert("RGB")
    if request.image_base64:
        raw = base64.b64decode(request.image_base64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise HTTPException(status_code=400, detail="provide image_path or image_base64")


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _json_objects(blob: str) -> list[dict[str, Any]]:
    """Decode JSON objects from prose/fences, including nested objects."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(blob):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(blob[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _dimension_scores(
    data: dict[str, Any],
    *,
    field: str,
    aggregate_field: str,
    questions: dict[str, str],
) -> dict[str, float] | None:
    raw = data.get(field)
    if isinstance(raw, dict):
        return {key: _clamp_score(raw.get(key), 0.0) for key in questions}
    # Backward-compatible parsing for the old two-scalar response.
    if aggregate_field in data:
        scalar = _clamp_score(data.get(aggregate_field), 0.0)
        return {key: scalar for key in questions}
    return None


def _parse_scores(text: str) -> dict[str, Any]:
    """Extract ten rubric scores and aggregate correctness/aesthetics."""
    blob = (text or "").strip()
    for data in reversed(_json_objects(blob)):
        correctness_scores = _dimension_scores(
            data,
            field="correctness_scores",
            aggregate_field="correctness",
            questions=CORRECTNESS_QUESTIONS,
        )
        aesthetics_scores = _dimension_scores(
            data,
            field="aesthetics_scores",
            aggregate_field="aesthetics",
            questions=AESTHETICS_QUESTIONS,
        )
        if correctness_scores is None or aesthetics_scores is None:
            continue
        correctness = sum(correctness_scores.values()) / len(correctness_scores)
        aesthetics = sum(aesthetics_scores.values()) / len(aesthetics_scores)
        findings = str(data.get("findings", data.get("reason", "")) or "").strip()
        fixes = str(data.get("suggested_fixes", data.get("fixes", "")) or "").strip()
        return {
            "correctness": correctness,
            "aesthetics": aesthetics,
            "correctness_scores": correctness_scores,
            "aesthetics_scores": aesthetics_scores,
            "findings": findings[:400],
            "suggested_fixes": fixes[:400],
        }
    # Regex fallback if the model emitted bare numbers.
    c_m = re.search(r"correctness\s*[:=]\s*([01](?:\.\d+)?)", blob, re.I)
    a_m = re.search(r"aesthetics?\s*[:=]\s*([01](?:\.\d+)?)", blob, re.I)
    if c_m and a_m:
        correctness = _clamp_score(c_m.group(1))
        aesthetics = _clamp_score(a_m.group(1))
        return {
            "correctness": correctness,
            "aesthetics": aesthetics,
            "correctness_scores": {key: correctness for key in CORRECTNESS_QUESTIONS},
            "aesthetics_scores": {key: aesthetics for key in AESTHETICS_QUESTIONS},
            "findings": blob[:200],
            "suggested_fixes": "",
        }
    return {
        "correctness": 0.4,
        "aesthetics": 0.4,
        "correctness_scores": {key: 0.4 for key in CORRECTNESS_QUESTIONS},
        "aesthetics_scores": {key: 0.4 for key in AESTHETICS_QUESTIONS},
        "findings": "vlm parse fallback; could not read structured scores",
        "suggested_fixes": "clearer subject match to user request; improve composition and lighting",
    }


def _rubric_good_enough(
    correctness_scores: dict[str, float],
    aesthetics_scores: dict[str, float],
) -> bool:
    """Require every correctness/aesthetics facet to clear the threshold."""
    if not correctness_scores or not aesthetics_scores:
        return False
    return (
        min(correctness_scores.values()) >= GOOD_ENOUGH_THRESHOLD
        and min(aesthetics_scores.values()) >= GOOD_ENOUGH_THRESHOLD
    )


def _judge(image: Image.Image, user_request: str, image_prompt: str, notes: str) -> tuple[dict[str, Any], str]:
    assert _model is not None and _processor is not None
    correctness_schema = ",\n".join(f'    "{key}": 0.0' for key in CORRECTNESS_QUESTIONS)
    aesthetics_schema = ",\n".join(f'    "{key}": 0.0' for key in AESTHETICS_QUESTIONS)
    rubric = "\n".join(
        [
            "CORRECTNESS QUESTIONS:",
            *[f"- {key}: {question}" for key, question in CORRECTNESS_QUESTIONS.items()],
            "AESTHETICS QUESTIONS:",
            *[f"- {key}: {question}" for key, question in AESTHETICS_QUESTIONS.items()],
        ]
    )
    prompt = (
        "You are a strict, calibrated visual reward judge. Inspect the pixels, not merely the "
        "diffusion prompt. Independently answer all ten rubric questions.\n"
        f"User request: {user_request}\n"
        f"Diffusion prompt used (context only; never treat it as visual evidence): {image_prompt or '(none)'}\n"
        f"Notes: {notes or '(none)'}\n\n"
        f"{rubric}\n\n"
        "CALIBRATION (HARSH — default LOW):\n"
        "  0.00 = absent, broken, or completely wrong\n"
        "  0.20 = severely deficient, majority of requested elements missing or wrong\n"
        "  0.40 = several elements present but at least half are wrong, blurry, or misplaced\n"
        "  0.55 = mostly correct BUT one or two notable flaws (missing entity, wrong color, bad layout)\n"
        "  0.70 = clearly good, all key elements present, minor aesthetic issues only\n"
        "  0.85+ = near-flawless, every detail exactly as requested (RARELY given)\n"
        "MANDATORY ANTI-INFLATION RULES:\n"
        "- For EVERY distinct entity/attribute/detail from the user request NOT visibly "
        "confirmed in the pixels, score the relevant dimension ≤0.40.\n"
        "- For complex user requests with 8+ distinct elements, the DEFAULT correctness "
        "score is 0.35 unless the image clearly shows ALL of them.\n"
        '- List at least THREE specific flaws, missing elements, or defects in "findings". '
        "If you cannot find three, look harder — there are always flaws in generated images.\n"
        "- Every score ≥0.70 MUST be justified with pixel-level evidence for every "
        "sub-requirement in that dimension. If you hesitate, score ≤0.55.\n"
        "- The median score across all ten dimensions should be ≤0.55. "
        "A median above 0.60 means you are being too generous.\n"
        "- Generated images are ALWAYS flawed. Start from 0.40 and require evidence to go UP, "
        "not from 1.0 and deduct.\n"
        "Return ONLY this JSON shape (replace every 0.0 with an independently judged score):\n"
        "{\n"
        '  "correctness_scores": {\n'
        f"{correctness_schema}\n"
        "  },\n"
        '  "aesthetics_scores": {\n'
        f"{aesthetics_schema}\n"
        "  },\n"
        '  "findings": "specific visual evidence for the lowest scores (at least three flaws)",\n'
        '  "suggested_fixes": "specific prompt rewrite hints"\n'
        "}\n"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[chat], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = _model.generate(**inputs, max_new_tokens=384, do_sample=False)
    # Strip prompt tokens.
    gen = out_ids[:, inputs["input_ids"].shape[-1] :]
    text = _processor.batch_decode(gen, skip_special_tokens=True)[0]
    return _parse_scores(text), text


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _processor
    _model, _processor = _load_model()
    yield
    _model = None
    _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Frozen Qwen3-VL Reflect", version="1", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "ok": _model is not None,
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "visible_cuda": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "good_enough_threshold": GOOD_ENOUGH_THRESHOLD,
        "good_enough_mode": "min_of_five_per_dimension",
    }


@app.post("/reflect", response_model=ReflectResponse)
def reflect(request: ReflectRequest) -> ReflectResponse:
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Reflect VLM is not loaded")
    image = _decode_image(request)
    started = time.perf_counter()
    try:
        with _infer_lock:
            scores, raw = _judge(image, request.user_request, request.image_prompt, request.notes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("reflect VLM failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    correctness = float(scores["correctness"])
    aesthetics = float(scores["aesthetics"])
    correctness_scores = dict(scores["correctness_scores"])
    aesthetics_scores = dict(scores["aesthetics_scores"])
    match = 0.55 * correctness + 0.45 * aesthetics
    good_enough = _rubric_good_enough(correctness_scores, aesthetics_scores)
    findings = scores.get("findings") or ""
    fixes = scores.get("suggested_fixes") or ""
    if good_enough:
        fixes = fixes or "none"
    elapsed = time.perf_counter() - started
    logger.info(
        "reflect ok=1 correctness=%.2f aesthetics=%.2f good_enough=%s latency=%.2fs",
        correctness,
        aesthetics,
        good_enough,
        elapsed,
    )
    return ReflectResponse(
        correctness=correctness,
        aesthetics=aesthetics,
        correctness_scores=correctness_scores,
        aesthetics_scores=aesthetics_scores,
        good_enough=good_enough,
        findings=findings,
        suggested_fixes=fixes,
        match=match,
        model=MODEL_ID,
        latency_s=elapsed,
        raw=raw[:800],
    )
