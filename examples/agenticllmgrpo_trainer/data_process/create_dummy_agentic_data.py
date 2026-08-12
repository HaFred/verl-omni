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
  generate_image → (text obs: path= + agentic_tool; PNGs stay on disk for the judge)
  judge_image    → (VL feedback: scores, findings, fixes, good_enough)
  agent reflects & decides: Reflection: ... Done.  OR  Reflection: ... + rewritten generate_image

Three demonstration classes (optional fewshot; overfit: Class-1 on soldier only):
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
2. After the image returns, call judge_image with SHORT args only:
   user_request="same as user message"
   image_prompt="last"
   Do NOT paste the full user task or re-paste long diffusion prompts into
   judge_image — the tool expands those from the live task and latest image.
3. Read the VL feedback (correctness, aesthetics, good_enough, findings,
   suggested_fixes). Then write your reflection and decide:
   - If good_enough=YES → "Reflection: <summary> Done."
   - If good_enough=NO  → "Reflection: <what's wrong> + rewritten generate_image"
     call in the SAME assistant turn, using the suggested_fixes.
   - After at most 3 successful generate_image calls, you MUST stop with
     "Reflection: <summary> Done." even if good_enough=NO. Do not keep rewriting.

HARD RULES (non-negotiable):
- ALWAYS call judge_image after EVERY generate_image before deciding.
- Never skip judge_image — you need the VL feedback to make an informed decision.
- Never call tools other than generate_image and judge_image.
- If you rewrite, the new prompt MUST differ from the previous one.
- Keep judge_image arguments compact (placeholders above). Long pasted args
  waste the response budget and truncate the tool call.

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
]

OVERFIT_PROMPTS = USER_PROMPTS[:2]  # soldier (idx=0) + fantasy wizard/elf (idx=1)

# Fewshot Class-1 demos are authored against prompt 0 only. Other overfit prompts
# get system+user (no baked trajectory) so the model does not copy soldier content.
_FEWSHOT_TASK = USER_PROMPTS[0]

# ── Shared task (same for all three demo classes) ────────────────────────────
_SHARED_TASK = _FEWSHOT_TASK
_SHARED_USER = _with_brevity(_SHARED_TASK)

# Compact judge args taught by fewshot (server expands to the live task + latest prompt).
_JUDGE_USER_ARG = "same as user message"
_JUDGE_PROMPT_ARG = "last"


def _gen_obs(path: str, prompt: str) -> str:
    """Fewshot generate_image tool obs — same shape as live ``_pack_response`` text."""
    prompt_snip = (prompt or "").replace("\n", " ")[:240]
    return (
        f"Frozen diffusion produced the image. path={path} "
        f"agentic_tool ok=1 stub=0 images=1 backend=fewshot prompt={prompt_snip!r}"
    )


def _judge_obs(
    *,
    path: str,
    correctness: float,
    aesthetics: float,
    good_enough: bool,
    findings: str,
    suggested_fixes: str,
) -> str:
    """Fewshot judge_image tool obs — same shape as live ``format_judge_observation``."""
    findings_short = " ".join((findings or "no specific findings").split())[:220]
    fixes_short = " ".join((suggested_fixes or "none").split())[:160]
    return (
        "VL judge on the last generated image:\n"
        f"  path={path}\n"
        f"  correctness={correctness:.2f}\n"
        f"  aesthetics ={aesthetics:.2f}\n"
        f"  good_enough ={'YES' if good_enough else 'NO'}\n"
        f"  findings: {findings_short}\n"
        f"  suggested_fixes: {fixes_short}\n"
        "  agentic_judge ok=1 parse_ok=1 stub=0 backend=fewshot parse_retries=0"
    )


# --- Class 1: single-pass — comprehensive prompt, VL says YES immediately ----
C1_USER = _SHARED_USER
C1_GEN_PROMPT = (
    "realistic pencil sketch, heartwarming indoor scene under oil lamp glow, "
    "a tall soldier in dusty uniform with medals pointing at a yellowed letter, "
    "elderly silver-haired mother on wooden chair touching the letter with "
    "teary eyes, soldier's wife standing behind with hand on his shoulder "
    "smiling reassuringly, faded family portrait on wall, dramatic lighting"
)
C1_PATH = "/tmp/fewshot/c1_pencilsketch_00.png"
C1_GEN_OBS = _gen_obs(C1_PATH, C1_GEN_PROMPT)
C1_REFLECT_OBS = _judge_obs(
    path=C1_PATH,
    correctness=0.82,
    aesthetics=0.78,
    good_enough=True,
    findings=(
        "all three figures confirmed — soldier with medals, silver-haired mother touching letter, "
        "wife behind with hand on shoulder; oil lamp glow illuminates faces; "
        "faded family portrait on wall; dramatic pencil shading present"
    ),
    suggested_fixes="none",
)
C1_ASSISTANT_3 = (
    "Reflection: VL confirms all three figures, medals, oil lamp, and family portrait "
    "present with dramatic pencil shading. Done."
)

# --- Class 2: compact two-pass demo on the same overfit soldier task ----------
C2_TASK = _SHARED_TASK
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
C2_PATH1 = "/tmp/fewshot/c2_pencilsketch_00.png"
C2_PATH2 = "/tmp/fewshot/c2_pencilsketch_01.png"
C2_GEN_OBS1 = _gen_obs(C2_PATH1, C2_GEN1)
C2_REFLECT_OBS1 = _judge_obs(
    path=C2_PATH1,
    correctness=0.38,
    aesthetics=0.42,
    good_enough=False,
    findings=(
        "only two figures visible — soldier and elderly woman; wife entirely missing; "
        "no medals or family portrait; lighting is flat and the letter is indistinct"
    ),
    suggested_fixes=(
        "add wife behind soldier with hand on shoulder, medals on his chest, "
        "glowing oil lamp, faded family portrait, and a clearly visible yellowed letter"
    ),
)
C2_REFLECT_REWRITE = (
    "Reflection: VL finds the wife, medals, portrait, and dramatic lamp lighting "
    "missing. Rewriting with the full cast and scene details."
)
C2_GEN_OBS2 = _gen_obs(C2_PATH2, C2_GEN2)
C2_REFLECT_OBS2 = _judge_obs(
    path=C2_PATH2,
    correctness=0.80,
    aesthetics=0.76,
    good_enough=True,
    findings=(
        "all three figures now present with correct relations; medals, yellowed "
        "letter, oil-lamp glow, family portrait, and dramatic pencil shading are visible"
    ),
    suggested_fixes="none",
)
C2_ASSISTANT_4 = (
    "Reflection: VL confirms all three figures, medals, letter, oil-lamp glow, "
    "family portrait, and dramatic pencil shading are present. Done."
)

# --- Class 3: three-pass — progressive refinement across three iterations -----
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
C3_PATH1 = "/tmp/fewshot/c3_pencilsketch_00.png"
C3_PATH2 = "/tmp/fewshot/c3_pencilsketch_01.png"
C3_PATH3 = "/tmp/fewshot/c3_pencilsketch_02.png"
C3_GEN_OBS1 = _gen_obs(C3_PATH1, C3_GEN1)
C3_REFLECT_OBS1 = _judge_obs(
    path=C3_PATH1,
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
)
C3_REFLECT_REWRITE_1 = (
    "Reflection: VL finds image blurry with only two vague figures — no pencil "
    "style, no uniform, no oil lamp. Rewriting with style, lighting, and character "
    "attributes."
)
C3_GEN_OBS2 = _gen_obs(C3_PATH2, C3_GEN2)
C3_REFLECT_OBS2 = _judge_obs(
    path=C3_PATH2,
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
)
C3_REFLECT_REWRITE_2 = (
    "Reflection: VL finds pencil style present and two figures ok, but wife still "
    "missing, no medals, no tears, no portrait. Adding full cast with emotional "
    "detail and all scene elements."
)
C3_GEN_OBS3 = _gen_obs(C3_PATH3, C3_GEN3)
C3_REFLECT_OBS3 = _judge_obs(
    path=C3_PATH3,
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
)
C3_ASSISTANT_5 = (
    "Reflection: VL confirms all elements present — three figures with correct "
    "attributes, oil lamp illuminating faces, family portrait on wall, dramatic "
    "pencil shading with profound emotion. Done."
)


def _demo_messages(class_id: int, *, omit_final_done: bool = False) -> list[dict]:
    """Fewshot trajectory following the generate → judge → reflect & decide protocol.

    ``omit_final_done=True`` (overfit): keep tool protocol, drop the terminal
    ``Reflection:…Done.`` assistant turn. Same-task demos that end in Done teach
    Qwen3-VL to skip tools and copy Done on the live turn.
    """
    if class_id % 3 == 0:
        msgs = [
            {"role": "user", "content": C1_USER},
            {"role": "assistant", "content": _tc("generate_image", prompt=C1_GEN_PROMPT)},
            {"role": "tool", "content": C1_GEN_OBS},
            {
                "role": "assistant",
                "content": _tc(
                    "judge_image",
                    user_request=_JUDGE_USER_ARG,
                    image_prompt=_JUDGE_PROMPT_ARG,
                ),
            },
            {"role": "tool", "content": C1_REFLECT_OBS},
        ]
        if not omit_final_done:
            msgs.append({"role": "assistant", "content": C1_ASSISTANT_3})
        return msgs
    if class_id % 3 == 1:
        msgs = [
            {"role": "user", "content": C2_USER},
            {"role": "assistant", "content": _tc("generate_image", prompt=C2_GEN1)},
            {"role": "tool", "content": C2_GEN_OBS1},
            {
                "role": "assistant",
                "content": _tc(
                    "judge_image",
                    user_request=_JUDGE_USER_ARG,
                    image_prompt=_JUDGE_PROMPT_ARG,
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
                    user_request=_JUDGE_USER_ARG,
                    image_prompt=_JUDGE_PROMPT_ARG,
                ),
            },
            {"role": "tool", "content": C2_REFLECT_OBS2},
        ]
        if not omit_final_done:
            msgs.append({"role": "assistant", "content": C2_ASSISTANT_4})
        return msgs
    msgs = [
        {"role": "user", "content": C3_USER},
        {"role": "assistant", "content": _tc("generate_image", prompt=C3_GEN1)},
        {"role": "tool", "content": C3_GEN_OBS1},
        {
            "role": "assistant",
            "content": _tc(
                "judge_image",
                user_request=_JUDGE_USER_ARG,
                image_prompt=_JUDGE_PROMPT_ARG,
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
                user_request=_JUDGE_USER_ARG,
                image_prompt=_JUDGE_PROMPT_ARG,
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
                user_request=_JUDGE_USER_ARG,
                image_prompt=_JUDGE_PROMPT_ARG,
            ),
        },
        {"role": "tool", "content": C3_REFLECT_OBS3},
    ]
    if not omit_final_done:
        msgs.append({"role": "assistant", "content": C3_ASSISTANT_5})
    return msgs


def _all_demo_messages() -> list[dict]:
    """All three demo classes concatenated — single-pass, two-pass, three-pass."""
    msgs: list[dict] = []
    for cid in range(3):
        msgs.extend(_demo_messages(cid))
    return msgs


def build_prompt_messages(
    user_text: str,
    *,
    class_id: int = 1,
    all_demos: bool = False,
    include_demos: bool = True,
    omit_final_done: bool = False,
) -> list[dict]:
    """System + optional demonstration(s) + the live user turn (with brevity reminder).

    Overfit fewshot should set ``omit_final_done=True`` so the demo teaches tools,
    not a copy-paste terminal Done on the same user request.
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if include_demos:
        if all_demos:
            demos: list[dict] = []
            for cid in range(3):
                demos.extend(_demo_messages(cid, omit_final_done=omit_final_done))
        else:
            demos = _demo_messages(class_id, omit_final_done=omit_final_done)
        msgs.extend(demos)
    msgs.append({"role": "user", "content": _with_brevity(user_text)})
    return msgs


def build_ground_truth(user_text: str, *, class_id: int = 1, overfit: bool = False) -> dict:
    """Weights for ``agentic_reward.compute_score`` (actor self-reflection protocol)."""
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
            "w_delta_c": 0.25,
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
    with_fewshot: bool = False,
) -> list[dict]:
    prompt_pool = prompts or USER_PROMPTS
    rows = []
    for i in range(n):
        if overfit:
            chunk = max(1, n // len(prompt_pool))
            prompt_index = min(i // chunk, len(prompt_pool) - 1)
            prompt_text = prompt_pool[prompt_index]
            class_id = 1  # compact two-pass Class 1 demo when fewshot is enabled
            # Same-task fewshot only: mismatch (e.g. soldier demo + fantasy live)
            # teaches the model to paste the wrong scene into generate_image.
            include_demos = bool(with_fewshot) and prompt_text == _FEWSHOT_TASK
        else:
            prompt_index = i % len(prompt_pool)
            prompt_text = prompt_pool[prompt_index]
            class_id = i % 3
            include_demos = True
        gt = build_ground_truth(prompt_text, class_id=class_id, overfit=overfit)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": build_prompt_messages(
                    prompt_text,
                    class_id=class_id,
                    all_demos=False,
                    include_demos=include_demos,
                    # Overfit same-task fewshot must not end in Done — that induced
                    # score=0 (Reflection+Done with zero tool calls) on Qwen3-VL.
                    omit_final_done=bool(overfit and include_demos),
                ),
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "prompt_index": prompt_index,
                    "raw_prompt": prompt_text,
                    "toy_agentic": True,
                    "overfit": overfit,
                    "with_fewshot": include_demos,
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
                        if k in gt
                    },
                    **({"w_delta_c": gt["w_delta_c"]} if "w_delta_c" in gt else {}),
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
        help="Chunked prompt assignment for short overfit e2e (GSM8K-style: system+user by default)",
    )
    parser.add_argument(
        "--with_fewshot",
        action="store_true",
        help=(
            "With --overfit, bake the compact Class 1 two-pass demonstration "
            "of the same overfit task into each prompt. "
            "Or set OVERFIT_FEWSHOT=1."
        ),
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
    with_fewshot = bool(args.with_fewshot) or os.environ.get("OVERFIT_FEWSHOT", "").strip() in {
        "1",
        "true",
        "yes",
        "on",
    }
    train_df = pd.DataFrame(build_rows("train", train_n, prompts, overfit=args.overfit, with_fewshot=with_fewshot))
    val_df = pd.DataFrame(build_rows("val", val_n, prompts, overfit=args.overfit, with_fewshot=with_fewshot))
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "val.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")
    if args.overfit:
        mode = (
            "compact Class 1 two-pass fewshot (same-task rows only)"
            if with_fewshot
            else "system+user only (no fewshot)"
        )
        print(f"overfit: {mode} × {len(prompts)} prompts chunked; tool_call_format={fmt}")
        for split_name, df in (("train", train_df), ("val", val_df)):
            counts: dict[int, int] = {}
            fewshot_counts: dict[int, int] = {}
            for extra in df["extra_info"].tolist():
                idx = int(extra.get("prompt_index", -1))
                counts[idx] = counts.get(idx, 0) + 1
                if extra.get("with_fewshot"):
                    fewshot_counts[idx] = fewshot_counts.get(idx, 0) + 1
            print(
                f"  {split_name} prompt_index counts={dict(sorted(counts.items()))} "
                f"fewshot={dict(sorted(fewshot_counts.items()))}"
            )
    else:
        print(f"demo classes={{0:single,1:two-pass,2:three-pass}}; tool_call_format={fmt}")


if __name__ == "__main__":
    main()
