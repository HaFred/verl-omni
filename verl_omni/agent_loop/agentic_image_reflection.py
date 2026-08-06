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

"""Image-grounded reflection → rewritten diffusion prompt (PR1 Stage-1 proxy).

Lance_3B_hf_und is text-only in this recipe (no image_processor), so we cannot
feed pixels into the actor for a true VLM judge. Instead we **inspect the PNG
produced by the frozen diffusion tool** with lightweight vision heuristics and
emit:

1. A ``Reflection:`` string that cites the observed image + defects.
2. A **new** ``generate_image`` prompt derived from that reflection (missing
   user attributes + quality fixes), not a generic suffix.

Full VLM-as-judge reflection remains PR2 / RFC Stage-1+.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

_STOP = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "to",
    "and",
    "with",
    "for",
    "from",
    "generate",
    "image",
    "create",
    "draw",
    "please",
    "make",
}


def _keywords(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOP]


def analyze_image(image_path: str | Path) -> dict[str, Any]:
    """Return simple visual stats for a saved tool PNG (or empty if unreadable)."""
    path = Path(image_path)
    out: dict[str, Any] = {
        "path": str(path),
        "ok": False,
        "width": 0,
        "height": 0,
        "brightness": 0.0,
        "contrast": 0.0,
        "edge_strength": 0.0,
        "colorfulness": 0.0,
    }
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return out
    try:
        img = Image.open(path).convert("RGB")
    except OSError:
        return out
    w, h = img.size
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0]) / 255.0
    contrast = float(stat.stddev[0]) / 255.0
    # Edge proxy: mean abs response of a simple blur residual.
    blur = gray.filter(ImageFilter.BoxBlur(2))
    edge_acc = 0.0
    n = 0
    # Subsample for speed on 512².
    step = max(1, min(w, h) // 64)
    g_px = gray.load()
    b_px = blur.load()
    for y in range(0, h, step):
        for x in range(0, w, step):
            edge_acc += abs(int(g_px[x, y]) - int(b_px[x, y]))
            n += 1
    edge_strength = (edge_acc / max(n, 1)) / 255.0
    # Colorfulness proxy: mean channel std across R/G/B.
    rgb_stat = ImageStat.Stat(img)
    colorfulness = float(sum(rgb_stat.stddev) / 3.0) / 255.0
    out.update(
        {
            "ok": True,
            "width": int(w),
            "height": int(h),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "edge_strength": round(edge_strength, 4),
            "colorfulness": round(colorfulness, 4),
        }
    )
    return out


def _quality_findings(stats: dict[str, Any], *, variant: int = 0) -> tuple[list[str], list[str]]:
    """Map image stats → (human findings, prompt fix phrases)."""
    findings: list[str] = []
    fixes: list[str] = []
    if not stats.get("ok"):
        findings.append("previous tool call produced no usable PNG (stub or missing file)")
        fixes.append("highly detailed, sharp focus, coherent composition")
        return findings, fixes

    findings.append(
        f"observed generated image {stats['width']}x{stats['height']} "
        f"(brightness={stats['brightness']:.2f}, contrast={stats['contrast']:.2f}, "
        f"edges={stats['edge_strength']:.2f}, color={stats['colorfulness']:.2f})"
    )
    # Rotate which defect we emphasize first for within-group GRPO diversity.
    checks = [
        (
            stats["brightness"] < 0.32,
            "the image looks underexposed / too dark",
            "brighter lighting, well-lit subject",
        ),
        (
            stats["brightness"] > 0.82,
            "the image looks overexposed / washed out",
            "balanced exposure, richer midtones",
        ),
        (
            stats["contrast"] < 0.12,
            "contrast is low; subject may look flat",
            "higher contrast, clearer subject separation",
        ),
        (
            stats["edge_strength"] < 0.04,
            "edges look soft / image may be blurry",
            "sharp focus, crisp details",
        ),
        (
            stats["colorfulness"] < 0.08,
            "colors look muted / desaturated",
            "richer colors, vivid but natural palette",
        ),
    ]
    ordered = checks[variant % len(checks) :] + checks[: variant % len(checks)]
    for cond, finding, fix in ordered:
        if cond:
            findings.append(finding)
            fixes.append(fix)
    if len(fixes) == 0:
        findings.append("basic exposure looks ok but fine detail / attributes can be stronger")
        suite = [
            "highly detailed, sharp focus, coherent composition",
            "improved lighting, richer colors, clearer subject",
            "studio quality, balanced framing, fine texture",
        ]
        fixes.append(suite[variant % len(suite)])
    return findings, fixes


def _missing_attributes(user_task: str, prev_prompt: str) -> list[str]:
    task_keys = _keywords(user_task)
    blob = (prev_prompt or "").lower()
    return [k for k in task_keys if k not in blob]


def rewrite_prompt_from_reflection(
    *,
    user_task: str,
    prev_prompt: str,
    fixes: list[str],
    missing_attrs: list[str],
) -> str:
    """Build the next diffusion prompt from reflection fixes (must differ from prev)."""
    task = (user_task or "").strip() or "a detailed visual scene"
    prev = (prev_prompt or "").strip()
    # Prefer the user request as the semantic backbone, then prior prompt extras.
    base = task
    if prev and prev.lower() != task.lower():
        # Keep non-overlapping content from the previous diffusion prompt.
        extra = []
        for tok in re.split(r"[,\n]", prev):
            piece = tok.strip()
            if not piece:
                continue
            if piece.lower() in base.lower():
                continue
            extra.append(piece)
        if extra:
            base = f"{base}, " + ", ".join(extra[:4])
    bits = [base]
    for attr in missing_attrs[:6]:
        if attr.lower() not in base.lower():
            bits.append(attr)
    for fix in fixes:
        if fix.lower() not in " ".join(bits).lower():
            bits.append(fix)
    prompt = ", ".join(bits)
    prompt = re.sub(r"\s+", " ", prompt).strip(" ,")
    if prompt.lower() == prev.lower():
        prompt = f"{prompt}, pass refine"
    return prompt[:500]


def reflect_on_generated_image(
    *,
    image_path: str | None,
    user_task: str,
    prev_prompt: str,
    variant: int = 0,
) -> tuple[str, str, dict[str, Any]]:
    """Inspect last tool PNG → ``(reflection_text, rewritten_prompt, debug_meta)``.

    The rewritten prompt is what the next forced ``generate_image`` must call.
    """
    stats = analyze_image(image_path) if image_path else {"ok": False, "path": image_path or ""}
    findings, fixes = _quality_findings(stats, variant=variant)
    missing = _missing_attributes(user_task, prev_prompt)
    if missing:
        findings.append("previous prompt under-specified task attributes: " + ", ".join(missing[:5]))
    reflect = (
        "Looking at the generated image from the frozen diffusion tool: "
        + "; ".join(findings)
        + ". Rewriting the diffusion prompt to address these issues."
    )
    new_prompt = rewrite_prompt_from_reflection(
        user_task=user_task,
        prev_prompt=prev_prompt,
        fixes=fixes,
        missing_attrs=missing,
    )
    meta = {
        "image_path": stats.get("path") or image_path or "",
        "image_ok": bool(stats.get("ok")),
        "stats": {
            k: stats[k]
            for k in ("width", "height", "brightness", "contrast", "edge_strength", "colorfulness")
            if k in stats
        },
        "missing_attrs": missing,
        "fixes": fixes,
        "rewritten_prompt": new_prompt,
        "reflection": reflect,
    }
    return reflect, new_prompt, meta
