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

"""Frozen image-generation function tool for verl's stock ``ToolAgentLoop``.

Mode (2a) keeps image generation **outside** the actor optimizer. GRPO trains
the actor as the visual agent while a frozen Qwen-Image pipeline generates
candidate images. Generated pixels are attached so the actor can self-reflect
and either finish with Done. or rewrite + call generate_image again.


Backends (first match wins):
  1. ``AGENTIC_QWEN_IMAGE_URL`` — bundled Qwen-Image HTTP service
     (POST ``{"prompt"}`` → base64 image JSON).
  2. ``AGENTIC_DIFFUSION_TOOL_URL`` — generic service with the same response
     contract.
  3. ``AGENTIC_LANCE_SERVER_URL`` — legacy OpenAI-compatible Lance Omni serve
     (``/v1/chat/completions``, ``modalities=["image"]``).
  4. Else text-only stub (acceptance smoke when no gen service is up).

Observation modality:
  Set ``AGENTIC_DIFFUSION_ATTACH_IMAGE=1`` for a VLM actor. Stock
  ``ToolAgentLoop`` then adds the generated PIL image to the next model turn.
  Set it to 0 only for a text-only actor or diagnostics.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image
from verl.tools.function_tool import function_tool
from verl.tools.schemas import ToolResponse

# Trajectory binding for artifact paths (no monkey-patch; agent loop sets ContextVars).
from verl_omni.agent_loop.agentic_trajectory_context import (  # noqa: F401
    get_active_call_provenance,
    get_active_trajectory_relpath,
    get_active_user_prompt,
    get_good_enough_yes_reached,
    get_latest_tool_image_path,
    register_tool_artifact,
    resolve_tool_image_path,
    set_active_call_provenance,
    set_active_trajectory_name,
    set_active_trajectory_relpath,
    set_active_user_prompt,
    set_good_enough_yes_reached,
    set_latest_tool_image_path,
)
from verl_omni.utils.judge_parse import (
    build_judge_prompt,
    format_judge_observation,
    format_judge_parse_error,
    parse_judge_json,
)

logger = logging.getLogger(__file__)


DIFFUSION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Generate an image with the frozen diffusion model. After each generation, "
            "inspect the attached image yourself: write a brief reflection, then either "
            "finish with Done. if good enough, or rewrite the prompt and call "
            "generate_image again in the same assistant turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The complete prompt to send to the diffusion model.",
                }
            },
            "required": ["prompt"],
        },
    },
}


def _decode_images(payload: dict) -> list[Image.Image]:
    """Decode the endpoint's optional base64 image fields."""
    encoded = payload.get("images_base64")
    if encoded is None and payload.get("image_base64") is not None:
        encoded = [payload["image_base64"]]
    if not encoded:
        return []
    if not isinstance(encoded, list):
        raise ValueError("diffusion tool response 'images_base64' must be a list")
    return [Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB") for item in encoded]


def _attach_images_enabled() -> bool:
    return os.getenv("AGENTIC_DIFFUSION_ATTACH_IMAGE", "0").strip().lower() in {"1", "true", "yes"}


def _e2e_run_root() -> Path:
    """Per-run artifact root: ``outputs/e2e/<experiment_name>/`` (or env override)."""
    explicit = os.getenv("AGENTIC_DIFFUSION_IMAGE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    run = os.getenv("AGENTIC_E2E_RUN_NAME", "").strip() or "default"
    repo_out = os.getenv("AGENTIC_E2E_ROOT", "").strip()
    if repo_out:
        return Path(repo_out) / run / "rollout_images"
    return Path("/tmp/agentic_qwen_image_t2i") / run / "rollout_images"


def _next_call_dir(root: Path) -> Path:
    """Fallback when no trajectory is bound: ``call_<ts>_<uuid>/``."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    call_dir = root / f"call_{stamp}_{uuid.uuid4().hex[:10]}"
    call_dir.mkdir(parents=True, exist_ok=True)
    return call_dir


def _next_image_index(traj_dir: Path) -> int:
    """Next ``image_XX`` index under a trajectory folder."""
    idxs: list[int] = []
    for path in traj_dir.glob("image_*.png"):
        try:
            idxs.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(idxs) + 1) if idxs else 0


def _call_meta_fields(prompt: str, *, user_prompt: str) -> dict:
    """Explicit reflection→rewrite provenance for meta.json call entries."""
    prov = dict(get_active_call_provenance() or {})
    controlled = bool(prov.get("controlled_by_reflection"))
    call_role = prov.get("call_role") or ("reflection_rewrite" if controlled else "initial")
    reflection = prov.get("reflection") or ""
    prev_prompt = prov.get("prev_tool_prompt") or ""
    source_image = prov.get("source_image") or ""
    rewritten = prov.get("rewritten_prompt") or (prompt if controlled else "")
    return {
        "call_role": call_role,
        "controlled_by_reflection": controlled,
        "reflection": reflection,
        "prev_tool_prompt": prev_prompt,
        "source_image_for_reflection": source_image,
        "rewritten_prompt": rewritten if controlled else "",
        # Explicit: this PNG was generated from the reflected/rewritten prompt.
        "image_generated_from_reflected_prompt": bool(controlled and prompt == rewritten),
        "tool_prompt_equals_rewritten_prompt": bool(controlled and prompt == rewritten),
        "content_source": prov.get("content_source") or ("teacher" if controlled else "initial"),
        "llm_reflection": prov.get("llm_reflection") or "",
        "llm_prompt": prov.get("llm_prompt") or "",
        "model_decode": prov.get("model_decode") or "",
        "user_prompt": user_prompt,
        "tool_prompt": prompt,
    }


def _update_traj_meta(traj_dir: Path, entry: dict) -> None:
    meta_path = traj_dir / "meta.json"
    meta: dict
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    else:
        meta = {}
    meta.setdefault("trajectory", traj_dir.name)
    meta.setdefault("experiment", os.getenv("AGENTIC_E2E_RUN_NAME", ""))
    user_prompt = entry.get("user_prompt") or get_active_user_prompt() or ""
    if user_prompt:
        meta["user_prompt"] = user_prompt
        entry.setdefault("user_prompt", user_prompt)
    calls = list(meta.get("calls") or [])
    calls.append(entry)
    meta["calls"] = calls
    meta["num_images"] = len(calls)
    meta["reflection_controlled_image_files"] = [c.get("file") for c in calls if c.get("controlled_by_reflection")]
    meta["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def _save_images(images: list[Image.Image], prompt: str, *, backend: str, tool_stubbed: bool) -> list[str]:
    """Persist tool images under the active stock-loop request path.

    When no trajectory is bound (e.g. standalone smoke), falls back to
    ``rollout_images/call_<ts>_<uuid>/``.
    """
    root = _e2e_run_root()
    root.mkdir(parents=True, exist_ok=True)
    relpath = get_active_trajectory_relpath()
    user_prompt = get_active_user_prompt() or ""
    provenance = _call_meta_fields(prompt, user_prompt=user_prompt)
    paths: list[str] = []

    def _entry(idx: int, path: Path, *, stubbed: bool) -> dict:
        return {
            "index": idx,
            "file": path.name,
            "path": str(path),
            "prompt": prompt,
            "backend": backend,
            "tool_stubbed": stubbed,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **provenance,
        }

    if relpath:
        traj_dir = root / relpath
        traj_dir.mkdir(parents=True, exist_ok=True)
        start_idx = _next_image_index(traj_dir)
        if images:
            for offset, img in enumerate(images):
                idx = start_idx + offset
                path = traj_dir / f"image_{idx:02d}.png"
                img.save(path)
                paths.append(str(path))
                _update_traj_meta(traj_dir, _entry(idx, path, stubbed=tool_stubbed))
        else:
            stub_path = traj_dir / f"STUB_NO_IMAGE_{start_idx:02d}.txt"
            stub_path.write_text(
                "No PNG produced (text stub or empty tool response).\n"
                f"user_prompt={user_prompt!r}\n"
                f"tool_prompt={prompt!r}\n"
                f"controlled_by_reflection={provenance.get('controlled_by_reflection')}\n"
                f"reflection={provenance.get('reflection')!r}\n"
                f"backend={backend}\n"
                "Set AGENTIC_QWEN_IMAGE_URL to a running Qwen-Image service for real images.\n"
            )
            paths.append(str(stub_path))
            _update_traj_meta(traj_dir, _entry(start_idx, stub_path, stubbed=True))
        logger.info(
            "diffusion tool artifacts (%d image(s), stub=%s, reflect_ctrl=%s) -> %s",
            len(images),
            tool_stubbed,
            provenance.get("controlled_by_reflection"),
            traj_dir,
        )
        register_tool_artifact(prompt=prompt, paths=paths, backend=backend, tool_stubbed=tool_stubbed)
        return paths

    # Legacy fallback (no active trajectory context).
    call_dir = _next_call_dir(root)
    meta = {
        **provenance,
        "backend": backend,
        "tool_stubbed": tool_stubbed,
        "num_images": len(images),
        "experiment": os.getenv("AGENTIC_E2E_RUN_NAME", ""),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for i, img in enumerate(images):
        path = call_dir / f"image_{i:02d}.png"
        img.save(path)
        paths.append(str(path))
    if not images:
        stub_path = call_dir / "STUB_NO_IMAGE.txt"
        stub_path.write_text(
            "No PNG produced (text stub or empty tool response).\n"
            f"user_prompt={user_prompt!r}\n"
            f"tool_prompt={prompt!r}\n"
            f"backend={backend}\n"
            "Set AGENTIC_QWEN_IMAGE_URL to a running Qwen-Image service for real images.\n"
        )
        paths.append(str(stub_path))
    (call_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    logger.info(
        "diffusion tool artifacts (%d image(s), stub=%s) -> %s",
        len(images),
        tool_stubbed,
        call_dir,
    )
    register_tool_artifact(prompt=prompt, paths=paths, backend=backend, tool_stubbed=tool_stubbed)
    return paths


def _image_vis_summary(image_path: str | None) -> str:
    """Compact measurable facts that complement the VLM's pixel observation."""
    if not image_path or not str(image_path).endswith(".png"):
        return ""
    try:
        from verl_omni.agent_loop.agentic_image_reflection import analyze_image

        stats = analyze_image(image_path)
    except Exception:  # noqa: BLE001 - never break the tool on viz failures
        return ""
    if not stats.get("ok"):
        return ""
    bright = float(stats.get("brightness") or 0.0)
    contrast = float(stats.get("contrast") or 0.0)
    edges = float(stats.get("edge_strength") or 0.0)
    color = float(stats.get("colorfulness") or 0.0)
    edge_tag = "soft" if edges < 0.04 else ("medium" if edges < 0.08 else "sharp")
    color_tag = "muted" if color < 0.08 else ("moderate" if color < 0.16 else "rich")
    luma_tag = "dark" if bright < 0.32 else ("bright" if bright > 0.82 else "mid")
    bits = [
        f"{stats.get('width')}x{stats.get('height')}",
        f"mean_luma={int(round(bright * 255))}",
        f"luma={luma_tag}",
        f"edges={edge_tag}",
        f"contrast={contrast:.2f}",
        f"colors={color_tag}",
    ]
    return "image_vis=" + " ".join(str(b) for b in bits)


def _pack_response(
    prompt: str,
    text: str,
    images: list[Image.Image],
    reward: float,
    *,
    backend: str,
    tool_stubbed: bool,
) -> tuple[ToolResponse, float, dict]:
    paths = _save_images(images, prompt, backend=backend, tool_stubbed=tool_stubbed)
    metrics: dict = {
        "tool_stubbed": tool_stubbed,
        "diffusion_backend": backend,
        "image_paths": paths,
        "num_images": len(images),
        "prompt": prompt,
        "artifact_dir": str(Path(paths[0]).parent) if paths else "",
    }
    ok = 1 if (images and not tool_stubbed) else 0
    # Machine-readable markers for agentic_reward (R_tool / R_result) — survive
    # decode of the multi-turn response including tool-obs tokens.
    prompt_snip = (prompt or "").replace("\n", " ")[:240]
    marker = (
        f"agentic_tool ok={ok} stub={1 if tool_stubbed else 0} images={len(images)} "
        f"backend={backend} prompt={prompt_snip!r}"
    )
    png0 = next((p for p in paths if str(p).endswith(".png")), None)
    if png0:
        set_latest_tool_image_path(png0)
    vis = _image_vis_summary(png0)
    if paths and "path=" not in text:
        text = f"{text} path={paths[0]}"
    if vis and "image_vis=" not in text:
        text = f"{text} {vis}"
        metrics["image_vis"] = vis
    text = f"{text} {marker}"
    if images and _attach_images_enabled():
        return ToolResponse(text=text, image=images), reward, metrics
    # Text-only fallback for an actor without image_processor or diagnostics.
    return ToolResponse(text=text), reward, metrics


def _call_generic_http(
    prompt: str,
    endpoint: str,
    *,
    backend: str = "http",
) -> tuple[ToolResponse, float, dict]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("AGENTIC_DIFFUSION_TOOL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        endpoint,
        data=json.dumps({"prompt": prompt}).encode(),
        headers=headers,
        method="POST",
    )
    # Offloaded Qwen-Image requests may queue behind other rollout workers on
    # the single frozen-tool GPU.
    timeout = float(os.getenv("AGENTIC_DIFFUSION_TOOL_TIMEOUT", "900"))
    try:
        with urlopen(request, timeout=timeout) as result:  # noqa: S310 - endpoint is operator-configured
            payload = json.loads(result.read())
    except Exception as exc:  # noqa: BLE001 - return failure as an observable tool result
        err = f"{backend} request failed: {exc}"
        logger.error(err)
        return _pack_response(
            prompt,
            err,
            images=[],
            reward=0.0,
            backend=f"{backend}_error",
            tool_stubbed=True,
        )

    images = _decode_images(payload)
    text = payload.get("text") or "The frozen diffusion tool generated the requested image."
    reward = float(payload.get("reward", 0.0))
    if not images:
        return _pack_response(
            prompt,
            text or f"{backend} returned no image",
            images=[],
            reward=0.0,
            backend=f"{backend}_empty",
            tool_stubbed=True,
        )
    return _pack_response(prompt, text, images, reward, backend=backend, tool_stubbed=False)


def _call_lance_omni(prompt: str, server_url: str) -> tuple[ToolResponse, float, dict]:
    """Call vLLM-Omni Lance OpenAI-compatible ``/v1/chat/completions`` (text2img)."""
    base = server_url.rstrip("/")
    height = int(os.getenv("AGENTIC_LANCE_HEIGHT", "512"))
    width = int(os.getenv("AGENTIC_LANCE_WIDTH", "512"))
    steps = int(os.getenv("AGENTIC_LANCE_STEPS", "30"))
    seed = os.getenv("AGENTIC_LANCE_SEED")
    # Default cfg_text_scale=1.0: vllm-omni Lance mRoPE + CFG batching
    # (torch.cat of (3,S) pids → (6,S)) crashes with "tensor a (3) vs b (6)".
    # Pass via extra_args (LancePipeline is not in model_extras registry, so
    # top-level cfg_text_scale is silently dropped).
    cfg_text_scale = float(os.getenv("AGENTIC_LANCE_CFG_TEXT_SCALE", "1.0"))
    payload: dict = {
        "messages": [
            {
                "role": "user",
                # Match vLLM-Omni Lance online client formatting.
                "content": [{"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}],
            }
        ],
        "modalities": ["image"],
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "extra_args": {"cfg_text_scale": cfg_text_scale},
    }
    if seed is not None and seed != "":
        payload["seed"] = int(seed)

    headers = {"Content-Type": "application/json"}
    token = os.getenv("AGENTIC_DIFFUSION_TOOL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = float(os.getenv("AGENTIC_DIFFUSION_TOOL_TIMEOUT", "300"))
    request = Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as result:  # noqa: S310 - operator-configured
            data = json.loads(result.read())
    except Exception as exc:  # noqa: BLE001 - surface server errors as tool obs + artifacts
        err = f"Lance Omni request failed: {exc}"
        logger.error(err)
        return _pack_response(prompt, err, images=[], reward=0.0, backend="lance_omni_error", tool_stubbed=True)

    images: list[Image.Image] = []
    text_bits: list[str] = []
    for choice in data.get("choices") or []:
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                img_url = (item.get("image_url") or {}).get("url") or ""
                if img_url.startswith("data:image"):
                    _, b64_data = img_url.split(",", 1)
                    images.append(Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB"))
                elif item.get("type") == "text" and item.get("text"):
                    text_bits.append(str(item["text"]))
        elif isinstance(content, str) and content:
            text_bits.append(content)

    if not images:
        err = f"Lance Omni server returned no image for prompt={prompt!r}"
        logger.error("%s; response keys=%s", err, list(data) if isinstance(data, dict) else type(data))
        return _pack_response(prompt, err, images=[], reward=0.0, backend="lance_omni_empty", tool_stubbed=True)

    text = " ".join(text_bits) if text_bits else "Lance frozen MoT tool generated the requested image."
    return _pack_response(prompt, text, images, 0.0, backend="lance_omni", tool_stubbed=False)


def _block_generate_after_yes_enabled() -> bool:
    return os.getenv("AGENTIC_BLOCK_GENERATE_AFTER_YES", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _blocked_generate_after_yes(prompt: str) -> tuple[ToolResponse, float, dict]:
    """Env hard-stop: refuse generate_image after good_enough=YES (no diffusion call)."""
    prompt_snip = (prompt or "").replace("\n", " ")[:240]
    text = (
        "generate_image blocked: a prior judge_image already returned good_enough=YES. "
        "Emit Reflection summarizing the VL feedback and end with Done. — do not rewrite. "
        f"agentic_block_generate_after_yes=1 agentic_tool ok=0 stub=0 images=0 "
        f"backend=blocked_after_yes prompt={prompt_snip!r}"
    )
    metrics = {
        "tool_stubbed": False,
        "diffusion_backend": "blocked_after_yes",
        "image_paths": [],
        "num_images": 0,
        "prompt": prompt,
        "blocked_after_yes": 1,
    }
    logger.info("Blocked generate_image after good_enough=YES (prompt=%r)", prompt_snip[:120])
    return ToolResponse(text=text), 0.0, metrics


@function_tool("generate_image", schema=DIFFUSION_TOOL_SCHEMA)
def generate_image(prompt: str) -> tuple[ToolResponse, float, dict]:
    """Generate an image with a frozen external Qwen-Image service.

    Args:
        prompt: Complete text prompt for the diffusion model.
    """
    if _block_generate_after_yes_enabled() and get_good_enough_yes_reached():
        return _blocked_generate_after_yes(prompt)

    # vLLM-omni (continuous batching) — preferred.
    vllm_omni_url = os.getenv("AGENTIC_VLLM_OMNI_URL", "").strip()
    if vllm_omni_url:
        return _call_vllm_omni(prompt, vllm_omni_url)

    qwen_image_url = os.getenv("AGENTIC_QWEN_IMAGE_URL", "").strip()
    if qwen_image_url:
        return _call_generic_http(prompt, qwen_image_url, backend="qwen_image")

    endpoint = os.getenv("AGENTIC_DIFFUSION_TOOL_URL", "").strip()
    if endpoint:
        return _call_generic_http(prompt, endpoint)

    # Retained only so older runs remain reproducible.
    lance_url = os.getenv("AGENTIC_LANCE_SERVER_URL", "").strip()
    if lance_url:
        return _call_lance_omni(prompt, lance_url)

    logger.warning(
        "AGENTIC_QWEN_IMAGE_URL / AGENTIC_VLLM_OMNI_URL unset; "
        "using text-only stub diffusion tool (acceptance smoke only)"
    )
    text = f"[stub diffusion result] No image service is configured. The requested prompt was: {prompt}"
    return _pack_response(prompt, text, images=[], reward=0.0, backend="stub", tool_stubbed=True)


def _call_vllm_omni(
    prompt: str,
    vllm_omni_url: str,
) -> tuple[ToolResponse, float, dict]:
    """Call vLLM-Omni's OpenAI-compatible image-generation endpoint."""
    base = vllm_omni_url.rstrip("/")
    height = int(os.getenv("QWEN_IMAGE_HEIGHT", "512"))
    width = int(os.getenv("QWEN_IMAGE_WIDTH", "512"))
    steps = int(os.getenv("QWEN_IMAGE_STEPS", "20"))
    cfg = float(os.getenv("QWEN_IMAGE_TRUE_CFG_SCALE", "4.0"))
    seed = os.getenv("QWEN_IMAGE_SEED")

    payload: dict = {
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
        "num_inference_steps": steps,
        "true_cfg_scale": cfg,
    }
    if seed is not None and seed != "":
        payload["seed"] = int(seed)

    headers = {"Content-Type": "application/json"}
    token = os.getenv("AGENTIC_DIFFUSION_TOOL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = float(os.getenv("AGENTIC_DIFFUSION_TOOL_TIMEOUT", "900"))
    try:
        req = Request(
            f"{base}/v1/images/generations",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=timeout) as result:  # noqa: S310
            data = json.loads(result.read().decode())
    except Exception as exc:  # noqa: BLE001
        err = f"vLLM-omni request failed: {exc}"
        logger.error(err)
        return _pack_response(prompt, err, images=[], reward=0.0, backend="vllm_omni_error", tool_stubbed=True)

    images: list[Image.Image] = []
    for item in data.get("data") or []:
        if not isinstance(item, dict):
            continue
        b64_data = item.get("b64_json")
        if b64_data:
            images.append(Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB"))

    if not images:
        err = f"vLLM-omni returned no image for prompt={prompt!r}"
        logger.error("%s", err)
        return _pack_response(prompt, err, images=[], reward=0.0, backend="vllm_omni_empty", tool_stubbed=True)

    text = "vLLM-Omni generated the requested image."
    return _pack_response(prompt, text, images, 0.0, backend="vllm_omni", tool_stubbed=False)


JUDGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "judge_image",
        "description": (
            "Call a frozen vision model to judge the LAST generated image. "
            "Returns structured feedback: correctness/aesthetics scores per dimension, "
            "specific findings, suggested prompt fixes, and a good_enough verdict. "
            "Call this AFTER every generate_image — the VL feedback tells you whether "
            "to finish (Done.) or rewrite and generate again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": "The original user request/task for the vision model to compare against.",
                },
                "image_prompt": {
                    "type": "string",
                    "description": "The exact diffusion prompt used to generate the image being evaluated.",
                },
            },
            "required": ["user_request", "image_prompt"],
        },
    },
}


def _call_judge_vlm(
    user_request: str,
    image_prompt: str,
) -> tuple[str, dict]:
    """Call the frozen Qwen3-VL sidecar to judge the last generated image.

    When ``AGENTIC_VLLM_URL`` is set, uses vLLM's OpenAI-compatible
    ``/v1/chat/completions`` with continuous batching. Otherwise falls back
    to the custom FastAPI ``/reflect`` endpoint.

    Returns ``(text, meta)`` where *text* is formatted for the agent to read
    and *meta* carries per-dimension scores for logging.
    """
    vllm_url = os.getenv("AGENTIC_VLLM_URL", "").strip()
    if vllm_url:
        return _call_judge_vllm(user_request, image_prompt, vllm_url)
    return _call_judge_custom(user_request, image_prompt)


# ── vLLM judge path (OpenAI /v1/chat/completions, continuous batching) ──────


def _post_vllm_chat(
    *,
    vllm_url: str,
    image_b64: str,
    prompt_text: str,
    max_tokens: int,
) -> tuple[str | None, str | None]:
    """Returns ``(raw_text, error)``. Exactly one is non-None on success/failure."""
    payload: dict = {
        "model": os.getenv("AGENTIC_VLLM_MODEL", "").strip() or "",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
    }
    if not payload["model"]:
        del payload["model"]
    timeout = float(os.getenv("AGENTIC_REFLECT_VLM_TIMEOUT", "120"))
    try:
        req = Request(
            f"{vllm_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    choices = data.get("choices") or []
    raw_text = ""
    if choices:
        raw_text = str(choices[0].get("message", {}).get("content", "") or "")
    if not raw_text:
        return None, "empty_response"
    return raw_text, None


def _call_judge_vllm(
    user_request: str,
    image_prompt: str,
    vllm_url: str,
) -> tuple[str, dict]:
    """Judge via vLLM's OpenAI-compatible ``/v1/chat/completions`` with parse retry."""
    image_path = resolve_tool_image_path(image_prompt=image_prompt)
    if not image_path:
        msg = (
            "[judge error] no image on disk for this generate_image call "
            f"(image_prompt={image_prompt[:120]!r}). "
            "Refusing to call the VL sidecar without pixels."
        )
        logger.error("judge_image aborted: missing image path (prompt=%r)", image_prompt[:160])
        return msg, {"error": "missing_image_path", "parse_ok": 0}

    try:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    except OSError as exc:
        msg = f"[judge error] cannot read image at {image_path}: {exc}"
        logger.error("%s", msg)
        return msg, {"error": str(exc), "image_path": image_path, "parse_ok": 0}

    base_tokens = int(os.getenv("AGENTIC_REFLECT_MAX_NEW_TOKENS", "1024"))
    max_retries = max(0, int(os.getenv("AGENTIC_JUDGE_PARSE_RETRIES", "1")))

    last_raw = ""
    last_err = None
    for attempt in range(max_retries + 1):
        strict = attempt > 0
        prompt_text = build_judge_prompt(user_request, image_prompt, strict_json=strict)
        # Give truncated JSON more room on retry.
        tokens = base_tokens if attempt == 0 else max(base_tokens, 1536)
        raw_text, err = _post_vllm_chat(
            vllm_url=vllm_url,
            image_b64=image_b64,
            prompt_text=prompt_text,
            max_tokens=tokens,
        )
        if err is not None:
            last_err = err
            logger.warning("vLLM judge call failed (attempt=%d): %s", attempt, err)
            continue
        assert raw_text is not None
        last_raw = raw_text
        parsed = parse_judge_json(raw_text)
        if parsed is not None:
            return format_judge_observation(
                image_path=image_path,
                parsed=parsed,
                backend="vllm",
                parse_retries=attempt,
            )
        logger.warning(
            "vLLM judge unparseable (attempt=%d/%d): %.200s",
            attempt,
            max_retries,
            raw_text,
        )

    if last_err and not last_raw:
        return (
            f"[judge error] vLLM request failed ({last_err}). "
            "Inspect the attached image yourself and decide Done. or rewrite.\n"
            f"  path={image_path}\n"
            f"  agentic_judge ok=0 parse_ok=0 stub=0 backend=vllm",
            {"error": last_err, "image_path": image_path, "parse_ok": 0, "parse_retries": max_retries},
        )
    return format_judge_parse_error(
        image_path=image_path,
        raw_text=last_raw,
        backend="vllm",
        parse_retries=max_retries,
    )


# ── Custom FastAPI fallback (original /reflect endpoint) ────────────────────


def _call_judge_custom(
    user_request: str,
    image_prompt: str,
) -> tuple[str, dict]:
    """Fallback: call the custom FastAPI ``/reflect`` endpoint."""
    image_path = resolve_tool_image_path(image_prompt=image_prompt)
    endpoint = os.getenv("AGENTIC_REFLECT_VLM_URL", "").strip()
    if not endpoint:
        return (
            "[judge stub] AGENTIC_REFLECT_VLM_URL / AGENTIC_VLLM_URL unset — inspect the attached image yourself.",
            {"stub": True},
        )

    if not image_path:
        msg = (
            "[judge error] no image on disk for this generate_image call "
            f"(image_prompt={image_prompt[:120]!r}). "
            "Refusing to call the VL sidecar without pixels. "
            "Rewrite/generate again only if a prior generate_image succeeded."
        )
        logger.error("judge_image aborted: missing image path (prompt=%r)", image_prompt[:160])
        return msg, {"error": "missing_image_path"}

    payload: dict = {
        "user_request": user_request,
        "image_prompt": image_prompt,
        "notes": "",
        "image_path": image_path,
    }
    try:
        payload["image_base64"] = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    except OSError as exc:
        msg = f"[judge error] cannot read image at {image_path}: {exc}"
        logger.error("%s", msg)
        return msg, {"error": str(exc), "image_path": image_path}

    timeout = float(os.getenv("AGENTIC_REFLECT_VLM_TIMEOUT", "120"))
    try:
        req = Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge VLM call failed: %s", exc)
        return (
            f"[judge error] VL sidecar request failed ({exc}). "
            "Inspect the attached image yourself and decide Done. or rewrite.",
            {"error": str(exc), "image_path": image_path},
        )

    if not isinstance(data, dict):
        return (
            "[judge error] VLM returned unexpected response — inspect the attached image yourself.",
            {"error": "not a dict", "image_path": image_path},
        )

    correctness = float(data.get("correctness", 0.0))
    aesthetics = float(data.get("aesthetics", 0.0))
    c_scores = data.get("correctness_scores") or {}
    a_scores = data.get("aesthetics_scores") or {}
    findings = str(data.get("findings") or "no specific findings")
    fixes = str(data.get("suggested_fixes") or "none")
    good = bool(data.get("good_enough", False))
    findings_short = re.sub(r"\s+", " ", findings).strip()[:220]
    fixes_short = re.sub(r"\s+", " ", fixes).strip()[:160]
    text = (
        f"VL judge on the last generated image:\n"
        f"  path={image_path}\n"
        f"  correctness={correctness:.2f}\n"
        f"  aesthetics ={aesthetics:.2f}\n"
        f"  good_enough ={'YES' if good else 'NO'}\n"
        f"  findings: {findings_short}\n"
        f"  suggested_fixes: {fixes_short}\n"
        f"  agentic_judge ok=1 stub=0 backend=custom"
    )

    meta = {
        "correctness": correctness,
        "aesthetics": aesthetics,
        "good_enough": good,
        "findings": findings,
        "suggested_fixes": fixes,
        "image_path": image_path,
        "backend": "custom",
    }
    meta.update({f"correctness_{k}": float(v) for k, v in c_scores.items() if isinstance(v, int | float)})
    meta.update({f"aesthetics_{k}": float(v) for k, v in a_scores.items() if isinstance(v, int | float)})
    return text, meta


@function_tool("judge_image", schema=JUDGE_TOOL_SCHEMA)
def judge_image(user_request: str, image_prompt: str) -> tuple[ToolResponse, float, dict]:
    """Call frozen Qwen3-VL to judge the last generated image in-turn.

    Args:
        user_request: Original user task for the vision model to compare against.
        image_prompt: The diffusion prompt used to generate the image being judged.
    """
    text, meta = _call_judge_vlm(user_request, image_prompt)
    # Env hard-stop latch: after YES, later generate_image calls are refused.
    if meta.get("good_enough") and meta.get("parse_ok", 1) != 0 and not meta.get("error"):
        set_good_enough_yes_reached(True)
    metrics = {
        "tool": "judge_image",
        "judge_stub": meta.get("stub", False),
        "judge_error": meta.get("error", ""),
    }
    for key in (
        "correctness",
        "aesthetics",
        "good_enough",
        "findings",
        "suggested_fixes",
    ):
        if key in meta:
            metrics[f"judge_{key}"] = meta[key]
    return ToolResponse(text=text), 0.0, metrics
