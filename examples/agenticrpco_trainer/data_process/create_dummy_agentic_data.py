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
"""Tiny agentic GRPO parquet: generate_image + judge_image → agent reflection.

Supports both actor tool-call wire formats (must match multi_turn.format):
  --tool_call_format hermes      → Qwen3-VL  JSON inside <tool_call>
  --tool_call_format qwen3_coder → Qwen3.5   <function=...><parameter=...>
  --tool_call_format auto        → pick from MODEL_PATH (default)

Protocol (one logical turn):
  generate_image → (image obs attached)
  judge_image    → (VL feedback: scores, findings, fixes, good_enough)
  agent reflects & decides: Reflection: ... Done.  OR  Reflection: ... + rewritten generate_image

Three demonstration classes (concatenated for overfit, cycled otherwise):
0. Single-pass success  (comprehensive prompt → VL says YES)
1. Two-pass refine      (lazy prompt → VL says NO → rewrite → VL says YES)
2. Three-pass refine    (very lazy → NO → rewrite → NO → rewrite → YES)
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

DATA_SOURCE = "jpeg_compressibility"
ABILITY = "agentic_generate_self_reflect"

SYSTEM_PROMPT = """You are a visual creation agent with two tools:
1) generate_image — create an image from a complete diffusion prompt
2) judge_image — call a frozen VL judge on the LAST generated image to get
   structured feedback (scores, findings, suggested fixes, good_enough verdict)

Protocol (one logical turn = generate → judge → reflect & decide):
1. Call generate_image with a complete diffusion prompt.
2. After the image returns, call judge_image(user_request, image_prompt)
   to get structured VL feedback on that image.
3. Read the VL feedback (correctness, aesthetics, good_enough, findings,
   suggested_fixes). Then write your reflection and decide:
   - If good_enough=YES → "Reflection: <summary> Done."
   - If good_enough=NO  → "Reflection: <what's wrong> + rewritten generate_image"
     call in the SAME assistant turn, using the suggested_fixes.

HARD RULES (non-negotiable):
- ALWAYS call judge_image after EVERY generate_image before deciding.
- Never skip judge_image — you need the VL feedback to make an informed decision.
- Never call tools other than generate_image and judge_image.
- If you rewrite, the new prompt MUST differ from the previous one.

Fewshot demos above/below (if present) are ONLY examples of the tool protocol for
on-policy GRPO exploration. They are NOT supervised targets: do not continue,
imitate, or debate the demo trajectory. Always treat the latest user message as
a fresh task.

Brevity (mandatory):
- Keep any private thinking to AT MOST one short paragraph (≤4 sentences).
- Do not debate yourself, repeat the user request, or rehash prior turns.
- Prefer emitting the <tool_call> immediately; finish with a one-line Done when done.
- Stop on your own when the task is complete — do not ramble until a length limit.
"""

_BREVITY_TAIL = " Keep any private thinking to AT MOST one short paragraph (≤4 sentences)."

# Wire format for fewshot <tool_call> blocks. Must match the actor chat template:
#   hermes      → Qwen3-VL  {"name": ..., "arguments": {...}}
#   qwen3_coder → Qwen3.5   <function=...><parameter=...>
_TOOL_CALL_FORMAT = os.environ.get("TOOL_CALL_FORMAT", "hermes").strip().lower()


def set_tool_call_format(fmt: str) -> str:
    """Set fewshot tool-call wire format (``hermes`` or ``qwen3_coder``)."""
    global _TOOL_CALL_FORMAT
    key = (fmt or "hermes").strip().lower()
    if key in {"xml", "qwen35", "qwen3.5", "qwen3_5"}:
        key = "qwen3_coder"
    if key not in {"hermes", "qwen3_coder"}:
        raise ValueError(f"Unsupported tool_call_format={fmt!r}; use hermes|qwen3_coder")
    _TOOL_CALL_FORMAT = key
    return _TOOL_CALL_FORMAT


def resolve_tool_call_format(fmt: str | None = None, model_path: str | None = None) -> str:
    """Resolve ``auto`` / explicit format from CLI or MODEL_PATH."""
    raw = (fmt or os.environ.get("TOOL_CALL_FORMAT") or "auto").strip().lower()
    if raw in {"hermes", "qwen3_coder", "xml", "qwen35", "qwen3.5", "qwen3_5"}:
        return set_tool_call_format(raw)
    path = (model_path or os.environ.get("MODEL_PATH") or os.environ.get("AGENT_MODEL_PATH") or "").strip()
    if path:
        try:
            from transformers import AutoConfig

            model_type = str(getattr(AutoConfig.from_pretrained(path, trust_remote_code=True), "model_type", "") or "")
            if model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_coder"}:
                return set_tool_call_format("qwen3_coder")
        except Exception:
            # Fall through to path heuristic / default.
            lowered = path.lower()
            if "qwen3.5" in lowered or "qwen3_5" in lowered or "qwen3-coder" in lowered:
                return set_tool_call_format("qwen3_coder")
    return set_tool_call_format("hermes")


def _with_brevity(user_task: str) -> str:
    """Append the brevity reminder to a user-facing request."""
    task = (user_task or "").rstrip()
    if _BREVITY_TAIL.strip() in task:
        return task
    return task + _BREVITY_TAIL


def _tc(name: str, **params: str) -> str:
    """Emit a tool-call block in the active wire format (Hermes JSON or Qwen XML)."""
    if _TOOL_CALL_FORMAT == "qwen3_coder":
        parts = [f"<tool_call>\n<function={name}>"]
        for key, value in params.items():
            parts.append(f"<parameter={key}>\n{value}\n</parameter>")
        parts.append("</function>\n</tool_call>")
        return "\n".join(parts)
    # Hermes (Qwen3-VL): JSON object inside <tool_call> tags.
    payload = {"name": name, "arguments": dict(params)}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


USER_PROMPTS = [
    (
        "In a realistic and emotionally evocative pencil sketch style, the composition focuses on a "
        "heartwarming indoor scene. Under the dim glow of an oil lamp, a returned soldier son is "
        "showing his elderly mother a yellowed letter from home. The soldier, tall and dressed in a "
        "dusty military uniform with medals pinned to his chest, leans forward and points at the words "
        "on the letter. His mother, with silver hair and a face full of wrinkles, sits on a wooden "
        "chair, her eyes glistening with tears of emotion as she gently touches the letter. The "
        "soldier's kind-hearted wife stands behind her husband, her hand resting on his shoulder, "
        "smiling reassuringly at her mother-in-law. The warm light of the oil lamp illuminates the "
        "faces of the three and the letter in their hands, while a faded family portrait hangs on the "
        "wall. The entire scene is filled with dramatic lighting and a profound sense of family emotion."
    ),
    (
        "Epic fantasy scene, wide-angle shot. In the dim ancient ruins, a circle of runestones on the "
        "ground glows with mysterious light. An elderly white-haired wizard, clad in a deep blue robe "
        "adorned with stars, wears a solemn expression as he chants a spell with both hands outstretched. "
        "Before him hovers an open, glowing blue magic book. He is protecting a young and beautiful elf "
        "princess, who has pointed ears and golden hair, dressed in an emerald-green gown. She tightly "
        "grips a life staff topped with a shining green gem, watching the enemy nervously. Their foe is "
        "a dark knight clad in full black runic armor, his face unseen, with ominous red light seeping "
        "through the cracks in his armor. He raises a massive black runic sword, poised to strike. "
        "Dynamic poses, dramatic lighting, digital painting, intricate details, cinematic feel."
    ),
    # (
    #     "In a dimly lit ancient stone chamber, the flames danced in the fireplace. An elderly rune "
    #     # do NOT include this for now, for rollout_n=2, the upper two is ok
    #     "master, dressed in a dark robe with silver-white hair and beard, was holding a wooden staff "
    #     "and pointing at an unfolded, weathered parchment scroll, imparting ancient knowledge to a "
    #     "young Celtic priestess. The priestess wore a green linen dress adorned with Celtic knots, "
    #     "her red hair braided into intricate plaits, and she gazed intently at the complex Norse runes "
    #     "on the scroll. Beside them, a sharp-eyed Viking warrior clad in leather armor stood with his "
    #     "arms crossed, observing the scene with curiosity. In the background, a massive runestone stood "
    #     "upright. The composition is a mid-shot, with strong contrasts of light and shadow."
    # ),
]

OVERFIT_PROMPTS = USER_PROMPTS[:2]

# ── Shared task (same for all three demo classes) ────────────────────────────
_SHARED_TASK = USER_PROMPTS[0]
_SHARED_USER = _with_brevity(_SHARED_TASK)

# Fewshot demos — all three classes use the same task, differing only in the
# number of generate→reflect→decide passes and the VL feedback scores.
#
# Each logical turn = generate_image → judge_image → reflect & decide (Done / rewrite).
# The VL feedback tool_obs below matches the output format of the judge_image
# tool in diffusion_tool.py (_call_judge_vlm).

# ── Helper: format a judge_image tool observation ────────────────────────────


def _judge_obs(
    correctness: float,
    aesthetics: float,
    good_enough: bool,
    findings: str,
    suggested_fixes: str,
    c_detail: str = "",
    a_detail: str = "",
) -> str:
    """Format a fewshot judge_image tool observation.

    Live rollouts use ``diffusion_tool._call_judge_vlm`` (scores only; no
    ``REQUIRED NEXT ACTION`` boilerplate — ``agentic_tool_agent`` injects
    ``Reflection:`` / ``Done.``). This helper keeps the instructional line for
    fewshot demonstrations only.
    """
    c_part = f"  correctness={correctness:.2f}  ({c_detail})" if c_detail else f"  correctness={correctness:.2f}"
    a_part = f"  aesthetics ={aesthetics:.2f}  ({a_detail})" if a_detail else f"  aesthetics ={aesthetics:.2f}"
    return (
        "VL judge / reflection feedback on the last generated image:\n"
        f"{c_part}\n"
        f"{a_part}\n"
        f"  good_enough ={'YES' if good_enough else 'NO'}\n"
        f"  findings: {findings}\n"
        f"  suggested_fixes: {suggested_fixes}\n"
        "  agentic_judge ok=1 stub=0\n"
        "\n"
        "REQUIRED NEXT ACTION (same assistant turn after reading this): "
        'write "Reflection: <brief notes on the VL scores/findings>". '
        "If good_enough=YES, end with Done. "
        "If good_enough=NO, continue in the SAME message with a rewritten "
        "generate_image call that incorporates suggested_fixes."
    )


# --- Class 0: single-pass — comprehensive prompt, VL says YES immediately ----
C1_USER = _SHARED_USER
C1_GEN_PROMPT = (
    "realistic pencil sketch, heartwarming indoor scene under oil lamp glow, "
    "a tall soldier in dusty uniform with medals pointing at a yellowed letter, "
    "elderly silver-haired mother on wooden chair touching the letter with "
    "teary eyes, soldier's wife standing behind with hand on his shoulder "
    "smiling reassuringly, faded family portrait on wall, dramatic lighting"
)
C1_GEN_OBS = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c1_pencilsketch_00.png "
    "image_vis=512x512 mean_luma=142 edges=sharp scene=pencil_sketch "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C1_GEN_PROMPT}'"
)
C1_REFLECT_OBS = _judge_obs(
    correctness=0.82,
    aesthetics=0.78,
    good_enough=True,
    findings=(
        "all three figures confirmed — soldier with medals, silver-haired mother touching letter, "
        "wife behind with hand on shoulder; oil lamp glow illuminates faces; "
        "faded family portrait on wall; dramatic pencil shading present"
    ),
    suggested_fixes="none",
    c_detail="subject_entities=0.90, attributes=0.80, relations_layout=0.80, scene_context=0.85, completeness=0.80",
    a_detail="composition=0.80, lighting=0.85, color=0.75, fidelity=0.75, appeal=0.75",
)
C1_ASSISTANT_3 = (
    "Reflection: VL confirms all three figures, medals, oil lamp, and family portrait "
    "present with dramatic pencil shading. Done."
)

# --- Class 1: two-pass — lazy first prompt, VL says NO, rewrite fixes it ----
C2_USER = _SHARED_USER
C2_GEN1 = "a soldier showing a letter to an old woman by lamplight, sketch style"
C2_GEN2 = (
    "realistic pencil sketch of a tall soldier in dusty military uniform with "
    "medals pinned to his chest, pointing at a yellowed letter, silver-haired "
    "elderly mother seated on wooden chair touching the letter with tears in "
    "her eyes, soldier's wife standing behind him with hand resting on his "
    "shoulder smiling reassuringly, warm oil lamp glow lighting faces and letter, "
    "faded family portrait on wall in background"
)
C2_GEN_OBS1 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c2_pencilsketch_00.png "
    "image_vis=512x512 mean_luma=78 edges=soft scene=dim_interior "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C2_GEN1}'"
)
C2_REFLECT_OBS1 = _judge_obs(
    correctness=0.38,
    aesthetics=0.42,
    good_enough=False,
    findings=(
        "only two figures visible — soldier and elderly woman; wife entirely missing from scene; "
        "no medals visible on uniform; family portrait on wall not rendered; "
        "lighting is dim and flat, lacks dramatic oil lamp glow; letter detail indistinct"
    ),
    suggested_fixes=(
        "add wife figure standing behind soldier with hand on his shoulder, "
        "render medals pinned to soldier's chest, "
        "add glowing oil lamp as primary light source illuminating faces, "
        "include faded family portrait on background wall, "
        "increase contrast for dramatic pencil shading effect, "
        "ensure yellowed letter is clearly visible with soldier pointing at it"
    ),
    c_detail="subject_entities=0.40, attributes=0.30, relations_layout=0.45, scene_context=0.35, completeness=0.30",
    a_detail="composition=0.45, lighting=0.30, color=0.50, fidelity=0.40, appeal=0.35",
)
C2_REFLECT_REWRITE = (
    "Reflection: VL finds only 2 of 3 figures — wife missing, no medals, no "
    "portrait, dim lighting. Rewriting with full cast and dramatic oil lamp glow."
)
C2_GEN_OBS2 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c2_pencilsketch_01.png "
    "image_vis=512x512 mean_luma=138 edges=medium scene=pencil_sketch "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C2_GEN2}'"
)
C2_REFLECT_OBS2 = _judge_obs(
    correctness=0.80,
    aesthetics=0.76,
    good_enough=True,
    findings=(
        "all three figures now present — soldier with medals, "
        "silver-haired mother touching letter with tears, "
        "wife behind with hand on shoulder; oil lamp glow effective; "
        "family portrait visible on wall; dramatic pencil shading conveys emotion"
    ),
    suggested_fixes="none",
    c_detail="subject_entities=0.88, attributes=0.82, relations_layout=0.80, scene_context=0.80, completeness=0.78",
    a_detail="composition=0.78, lighting=0.82, color=0.72, fidelity=0.74, appeal=0.74",
)
C2_ASSISTANT_4 = (
    "Reflection: VL confirms all three figures present — soldier with medals, "
    "elderly mother with tears touching letter, wife behind with hand on shoulder. "
    "Oil lamp glow and family portrait now visible. Dramatic pencil shading. Done."
)

# --- Class 2: three-pass — progressive refinement across three iterations -----
C3_USER = _SHARED_USER
C3_GEN1 = "a soldier and an old woman indoors, pencil drawing"
C3_GEN2 = (
    "realistic pencil sketch of a soldier in uniform showing a yellowed letter "
    "to an elderly mother under oil lamp, indoor scene with dramatic shadows"
)
C3_GEN3 = (
    "realistic emotionally evocative pencil sketch, heartwarming indoor scene, "
    "under dim glow of oil lamp a tall returned soldier in dusty military uniform "
    "with medals on chest leans forward pointing at yellowed letter, his elderly "
    "mother with silver hair and wrinkled face sits on wooden chair with glistening "
    "teary eyes gently touching the letter, his kind-hearted wife stands behind "
    "him with hand resting on his shoulder smiling reassuringly at her "
    "mother-in-law, warm lamp light illuminates the three faces and the letter, "
    "faded family portrait hangs on wall in background, dramatic lighting, "
    "profound family emotion"
)
C3_GEN_OBS1 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c3_pencilsketch_00.png "
    "image_vis=512x512 mean_luma=65 edges=blurry scene=vague_interior "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C3_GEN1}'"
)
C3_REFLECT_OBS1 = _judge_obs(
    correctness=0.22,
    aesthetics=0.25,
    good_enough=False,
    findings=(
        "image is blurry and barely recognizable — only two vague figures in dim light; "
        "pencil sketch style not achieved; no uniform, no medals, no letter detail, "
        "no oil lamp, no wife, no portrait; scene looks like a generic dark room"
    ),
    suggested_fixes=(
        "use realistic pencil sketch style with dramatic lighting, "
        "add soldier in uniform showing yellowed letter, "
        "add elderly mother with silver hair, add oil lamp as light source, "
        "ensure indoor setting with shadows, "
        "include at least the primary two figures with clear attributes"
    ),
    c_detail="subject_entities=0.25, attributes=0.15, relations_layout=0.20, scene_context=0.20, completeness=0.15",
    a_detail="composition=0.25, lighting=0.15, color=0.30, fidelity=0.15, appeal=0.20",
)
C3_REFLECT_REWRITE_1 = (
    "Reflection: VL finds image blurry with only two vague figures — no pencil "
    "style, no uniform, no oil lamp. Rewriting with style, lighting, and character "
    "attributes."
)
C3_GEN_OBS2 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c3_pencilsketch_01.png "
    "image_vis=512x512 mean_luma=105 edges=medium scene=pencil_sketch_interior "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C3_GEN2}'"
)
C3_REFLECT_OBS2 = _judge_obs(
    correctness=0.48,
    aesthetics=0.52,
    good_enough=False,
    findings=(
        "pencil sketch style now visible; soldier in uniform and elderly woman present; "
        "oil lamp rendered; but wife figure still missing from scene; "
        "medals on soldier's chest absent; mother lacks silver hair and emotional tears; "
        "family portrait on wall not rendered; "
        "composition feels incomplete with only two of three requested figures"
    ),
    suggested_fixes=(
        "add wife figure standing behind soldier with hand on his shoulder, "
        "render medals pinned to soldier's chest, "
        "give mother silver hair and visible tears of emotion, "
        "add faded family portrait on background wall, "
        "include yellowed letter with soldier pointing at it, "
        "ensure all three faces are illuminated by warm oil lamp glow"
    ),
    c_detail="subject_entities=0.50, attributes=0.40, relations_layout=0.55, scene_context=0.50, completeness=0.40",
    a_detail="composition=0.50, lighting=0.55, color=0.55, fidelity=0.45, appeal=0.50",
)
C3_REFLECT_REWRITE_2 = (
    "Reflection: VL finds pencil style present and two figures ok, but wife still "
    "missing, no medals, no tears, no portrait. Adding full cast with emotional "
    "detail and all scene elements."
)
C3_GEN_OBS3 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c3_pencilsketch_02.png "
    "image_vis=512x512 mean_luma=140 edges=sharp scene=pencil_sketch "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C3_GEN3}'"
)
C3_REFLECT_OBS3 = _judge_obs(
    correctness=0.84,
    aesthetics=0.80,
    good_enough=True,
    findings=(
        "all three figures confirmed — soldier with medals leaning forward "
        "pointing at yellowed letter, silver-haired elderly mother with glistening tears "
        "touching letter, wife behind with hand on shoulder smiling; "
        "oil lamp warmly illuminates three faces; faded family portrait on wall; "
        "dramatic pencil shading conveys profound family emotion"
    ),
    suggested_fixes="none",
    c_detail="subject_entities=0.92, attributes=0.85, relations_layout=0.85, scene_context=0.82, completeness=0.82",
    a_detail="composition=0.82, lighting=0.85, color=0.78, fidelity=0.78, appeal=0.80",
)
C3_ASSISTANT_5 = (
    "Reflection: VL confirms all elements present — three figures with correct "
    "attributes, oil lamp illuminating faces, family portrait on wall, dramatic "
    "pencil shading with profound emotion. Done."
)


def _demo_messages(class_id: int) -> list[dict]:
    """Fewshot trajectory following the generate → judge → reflect & decide protocol.

    Each logical turn is:
      assistant: generate_image(prompt_k)
      tool:      gen observation (image + metadata)
      assistant: judge_image(user_request, prompt_k)
      tool:      VL feedback (scores, findings, fixes, good_enough)
      assistant: Reflection: ... Done.  OR  Reflection: ... + rewritten generate_image

    All three classes demonstrate the same shared task (soldier scene), differing
    only in the number of generate→judge passes and initial prompt quality.
    """
    if class_id % 3 == 0:
        # Single-pass: comprehensive prompt → VL says YES → Done.
        return [
            {"role": "user", "content": C1_USER},
            {"role": "assistant", "content": _tc("generate_image", prompt=C1_GEN_PROMPT)},
            {"role": "tool", "content": C1_GEN_OBS},
            {
                "role": "assistant",
                "content": _tc(
                    "judge_image",
                    user_request=_SHARED_TASK,
                    image_prompt=C1_GEN_PROMPT,
                ),
            },
            {"role": "tool", "content": C1_REFLECT_OBS},
            {"role": "assistant", "content": C1_ASSISTANT_3},
        ]
    if class_id % 3 == 1:
        # Two-pass: lazy first prompt → VL says NO → rewrite → VL says YES → Done.
        return [
            {"role": "user", "content": C2_USER},
            {"role": "assistant", "content": _tc("generate_image", prompt=C2_GEN1)},
            {"role": "tool", "content": C2_GEN_OBS1},
            {
                "role": "assistant",
                "content": _tc(
                    "judge_image",
                    user_request=_SHARED_TASK,
                    image_prompt=C2_GEN1,
                ),
            },
            {"role": "tool", "content": C2_REFLECT_OBS1},
            {
                "role": "assistant",
                "content": C2_REFLECT_REWRITE + "\n" + _tc("generate_image", prompt=C2_GEN2),
            },
            {"role": "tool", "content": C2_GEN_OBS2},
            {
                "role": "assistant",
                "content": _tc(
                    "judge_image",
                    user_request=_SHARED_TASK,
                    image_prompt=C2_GEN2,
                ),
            },
            {"role": "tool", "content": C2_REFLECT_OBS2},
            {"role": "assistant", "content": C2_ASSISTANT_4},
        ]
    # Three-pass: very lazy prompt → VL says NO → rewrite → VL says NO → rewrite → VL says YES → Done.
    return [
        {"role": "user", "content": C3_USER},
        {"role": "assistant", "content": _tc("generate_image", prompt=C3_GEN1)},
        {"role": "tool", "content": C3_GEN_OBS1},
        {
            "role": "assistant",
            "content": _tc(
                "judge_image",
                user_request=_SHARED_TASK,
                image_prompt=C3_GEN1,
            ),
        },
        {"role": "tool", "content": C3_REFLECT_OBS1},
        {
            "role": "assistant",
            "content": C3_REFLECT_REWRITE_1 + "\n" + _tc("generate_image", prompt=C3_GEN2),
        },
        {"role": "tool", "content": C3_GEN_OBS2},
        {
            "role": "assistant",
            "content": _tc(
                "judge_image",
                user_request=_SHARED_TASK,
                image_prompt=C3_GEN2,
            ),
        },
        {"role": "tool", "content": C3_REFLECT_OBS2},
        {
            "role": "assistant",
            "content": C3_REFLECT_REWRITE_2 + "\n" + _tc("generate_image", prompt=C3_GEN3),
        },
        {"role": "tool", "content": C3_GEN_OBS3},
        {
            "role": "assistant",
            "content": _tc(
                "judge_image",
                user_request=_SHARED_TASK,
                image_prompt=C3_GEN3,
            ),
        },
        {"role": "tool", "content": C3_REFLECT_OBS3},
        {"role": "assistant", "content": C3_ASSISTANT_5},
    ]


def _all_demo_messages() -> list[dict]:
    """All three demo classes concatenated — single-pass, two-pass, three-pass.

    Overfit mode uses this so every rollout in a group sees the identical fewshot
    context, giving GRPO a level playing field for within-group advantage computation.
    """
    msgs: list[dict] = []
    for cid in range(3):
        msgs.extend(_demo_messages(cid))
    return msgs


def build_prompt_messages(
    user_text: str,
    *,
    class_id: int = 1,
    all_demos: bool = False,
) -> list[dict]:
    """System + demonstration(s) + the live user turn (with brevity reminder)."""
    demos = _all_demo_messages() if all_demos else _demo_messages(class_id)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *demos,
        {"role": "user", "content": _with_brevity(user_text)},
    ]


def build_ground_truth(user_text: str, *, class_id: int = 1, overfit: bool = False) -> dict:
    """Weights for ``agentic_reward.compute_score`` (actor self-reflection protocol)."""
    # Done must be heavy enough that high frozen-judge C/A cannot plateau the
    # scalar without Reflection:+Done. (C/A are also gated in compute_score.)
    weights = {
        "w_tool_call": 0.10,
        "w_correctness": 0.35,
        "w_aesthetics": 0.35,
        "w_done": 0.20,
        "forced_consolation": 0.0,
    }
    if overfit:
        return {
            "user_request": user_text,
            "demo_class": "all",
            "expected_num_images": 2,
            **weights,
        }
    expected = 1 + (class_id % 3)
    return {
        "user_request": user_text,
        "demo_class": int(class_id % 3),
        "expected_num_images": expected,
        **weights,
    }


def build_rows(
    split: str,
    n: int,
    prompts: list[str] | None = None,
    *,
    overfit: bool = False,
) -> list[dict]:
    prompt_pool = prompts or USER_PROMPTS
    rows = []
    for i in range(n):
        if overfit:
            # First half of samples get prompt[0], second half get prompt[1], etc.
            chunk = max(1, n // len(prompt_pool))
            prompt_text = prompt_pool[min(i // chunk, len(prompt_pool) - 1)]
            class_id = -1  # all three fewshot classes included
        else:
            prompt_text = prompt_pool[i % len(prompt_pool)]
            class_id = i % 3
        gt = build_ground_truth(prompt_text, class_id=class_id, overfit=overfit)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": build_prompt_messages(prompt_text, class_id=class_id, all_demos=overfit),
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "raw_prompt": prompt_text,
                    "toy_agentic": True,
                    "overfit": overfit,
                    "demo_class": gt["demo_class"],
                    "expected_num_images": gt["expected_num_images"],
                    "native_tool_template": True,
                    "visual_tool_observation": True,
                    **{
                        k: gt[k]
                        for k in (
                            "w_tool_call",
                            "w_correctness",
                            "w_aesthetics",
                            "w_done",
                        )
                    },
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate agentic GRPO parquet (Qwen3-VL Hermes or Qwen3.5 XML fewshots)"
    )
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/agentic"))
    parser.add_argument("--train_size", type=int, default=64)
    parser.add_argument("--val_size", type=int, default=8)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Chunked prompt assignment + all-3-class fewshot for short overfit e2e",
    )
    parser.add_argument(
        "--tool_call_format",
        default=os.environ.get("TOOL_CALL_FORMAT", "auto"),
        choices=["auto", "hermes", "qwen3_coder", "xml", "qwen35"],
        help=(
            "Fewshot <tool_call> wire format. auto picks from MODEL_PATH "
            "(qwen3_coder for Qwen3.5, hermes for Qwen3-VL). "
            "Must match actor_rollout_ref.rollout.multi_turn.format."
        ),
    )
    parser.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH") or os.environ.get("AGENT_MODEL_PATH") or "",
        help="Optional actor checkpoint used when --tool_call_format=auto",
    )
    args = parser.parse_args()

    fmt = resolve_tool_call_format(args.tool_call_format, args.model_path or None)
    os.makedirs(args.local_save_dir, exist_ok=True)
    prompts = OVERFIT_PROMPTS if args.overfit else None
    train_n = args.train_size
    val_n = args.val_size
    train_df = pd.DataFrame(build_rows("train", train_n, prompts, overfit=args.overfit))
    val_df = pd.DataFrame(build_rows("val", val_n, prompts, overfit=args.overfit))
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "val.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")
    if args.overfit:
        print(f"overfit: all-3-class fewshot × {len(prompts)} prompts chunked; tool_call_format={fmt}")
    else:
        print(f"demo classes={{0:single,1:two-pass,2:three-pass}}; tool_call_format={fmt}")


if __name__ == "__main__":
    main()
