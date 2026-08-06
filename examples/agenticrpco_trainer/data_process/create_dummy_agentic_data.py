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
"""Tiny agentic GRPO parquet: generate_image + actor self-reflection.

Supports both actor tool-call wire formats (must match multi_turn.format):
  --tool_call_format hermes      → Qwen3-VL  JSON inside <tool_call>
  --tool_call_format qwen3_coder → Qwen3.5   <function=...><parameter=...>
  --tool_call_format auto        → pick from MODEL_PATH (default)

Protocol (gated):
  generate_image → (image obs attached)
  actor inspects the image and either:
    brief reflection + Done.  OR
    brief reflection + rewritten generate_image  (same assistant turn)

Frozen Qwen3-VL is used only as the reward judge (not an agent tool).

Three demonstration classes (cycled by sample index):
0. Single-pass success
1. Two-pass refine
2. Three-pass refine
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

DATA_SOURCE = "jpeg_compressibility"
ABILITY = "agentic_generate_self_reflect"

SYSTEM_PROMPT = """You are a visual creation agent with one tool:
1) generate_image — create an image from a complete diffusion prompt

After generate_image returns, an image is attached to the observation. You must
inspect that image yourself (do not invent a separate judge tool).

HARD RULE (non-negotiable):
- After EVERY generate_image observation, write a brief reflection on what you see
  (colors, lighting, subject, defects) in the same assistant turn as your next action.
- If the image is good enough, end that turn with Done.
- If not, rewrite the diffusion prompt based on your reflection and call
  generate_image again in that same assistant message (reflection text + tool call).
- Never call tools other than generate_image.

Fewshot demos above/below (if present) are ONLY examples of the tool protocol for
on-policy GRPO exploration. They are NOT supervised targets: do not continue,
imitate, or debate the demo trajectory. Always treat the latest user message as
a fresh task.

Brevity (mandatory):
- Keep any private thinking to AT MOST one short paragraph (≤4 sentences).
- Do not debate yourself, repeat the user request, or rehash prior turns.
- Prefer emitting the <tool_call> immediately; finish with a one-line Done when done.
- Stop on your own when the task is complete — do not ramble until a length limit.

Protocol (follow exactly):
- Call generate_image with a complete diffusion prompt.
- After the tool returns an image, inspect it and either:
  (a) Reflection: <brief visual notes> Done.
  (b) Reflection: <brief visual notes / what to fix> + generate_image with a rewritten prompt.
Never copy path=/agentic_* metadata into your reply except by calling the tools.
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


# --- Class 0: one generate → reflection + Done --------------------------------
C1_TASK = "Generate an image of a bright red apple on a white table"
C1_USER = _with_brevity(C1_TASK)
C1_GEN_PROMPT = "a bright red apple on a white table, soft studio lighting, sharp focus"
C1_TOOL_1 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c1_apple_00.png "
    "image_vis=512x512 mean_luma=168 edges=sharp colors=red_rich "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C1_GEN_PROMPT}'"
)
C1_ASSISTANT_2 = "Reflection: bright red apple on white table, sharp edges, rich color. Done."

# --- Class 1: gen → reflection+rewrite gen → reflection+Done ------------------
C2_TASK = "Generate an image of a bright red apple on a white table"
C2_USER = _with_brevity(C2_TASK)
C2_GEN1 = "a red apple on a white table, soft lighting"
C2_GEN2 = (
    "a bright red apple on a white table, strong studio lighting, "
    "highly detailed, sharp focus, richer reds, coherent composition"
)
C2_TOOL_1 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c2_apple_00.png "
    "image_vis=512x512 mean_luma=92 edges=soft colors=red_muted "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C2_GEN1}'"
)
C2_TOOL_2 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c2_apple_01.png "
    "image_vis=512x512 mean_luma=155 edges=medium colors=red_rich "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C2_GEN2}'"
)
C2_REFLECT_REWRITE = "Reflection: apple present but muted reds and soft edges; rewrite for brighter lighting."
C2_ASSISTANT_3 = "Reflection: bright red apple now matches; richer color and sharper focus. Done."

# --- Class 2: three-pass same pattern ----------------------------------------
C3_TASK = "Create a vivid sunset over snowy mountains with a red cabin"
C3_USER = _with_brevity(C3_TASK)
C3_GEN1 = "a sunset over mountains with a cabin"
C3_GEN2 = "a vivid sunset over snowy mountains with a red cabin, soft light"
C3_GEN3 = (
    "a vivid golden-hour sunset over snow-capped mountains with a rustic red cabin "
    "in the foreground, high contrast, sharp details, rich warm colors"
)
C3_TOOL_1 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c3_sunset_00.png "
    "image_vis=512x512 mean_luma=88 edges=soft colors=muted "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C3_GEN1}'"
)
C3_TOOL_2 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c3_sunset_01.png "
    "image_vis=512x512 mean_luma=110 edges=soft colors=moderate "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C3_GEN2}'"
)
C3_TOOL_3 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/c3_sunset_02.png "
    "image_vis=512x512 mean_luma=148 edges=medium colors=rich "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    f"prompt='{C3_GEN3}'"
)
C3_REFLECT_REWRITE_1 = "Reflection: missing snowy mountains and red cabin; muted vs vivid request."
C3_REFLECT_REWRITE_2 = "Reflection: attributes mostly present but vividness and contrast still weak."
C3_ASSISTANT_4 = "Reflection: vivid sunset, snowy mountains, and red cabin match the request. Done."


def _demo_messages(class_id: int) -> list[dict]:
    """Fewshot trajectory; assistant tool calls use the active ``_TOOL_CALL_FORMAT``."""
    if class_id % 3 == 0:
        return [
            {"role": "user", "content": C1_USER},
            {"role": "assistant", "content": _tc("generate_image", prompt=C1_GEN_PROMPT)},
            {"role": "tool", "content": C1_TOOL_1},
            {"role": "assistant", "content": C1_ASSISTANT_2},
        ]
    if class_id % 3 == 1:
        return [
            {"role": "user", "content": C2_USER},
            {"role": "assistant", "content": _tc("generate_image", prompt=C2_GEN1)},
            {"role": "tool", "content": C2_TOOL_1},
            {
                "role": "assistant",
                "content": C2_REFLECT_REWRITE + "\n" + _tc("generate_image", prompt=C2_GEN2),
            },
            {"role": "tool", "content": C2_TOOL_2},
            {"role": "assistant", "content": C2_ASSISTANT_3},
        ]
    return [
        {"role": "user", "content": C3_USER},
        {"role": "assistant", "content": _tc("generate_image", prompt=C3_GEN1)},
        {"role": "tool", "content": C3_TOOL_1},
        {
            "role": "assistant",
            "content": C3_REFLECT_REWRITE_1 + "\n" + _tc("generate_image", prompt=C3_GEN2),
        },
        {"role": "tool", "content": C3_TOOL_2},
        {
            "role": "assistant",
            "content": C3_REFLECT_REWRITE_2 + "\n" + _tc("generate_image", prompt=C3_GEN3),
        },
        {"role": "tool", "content": C3_TOOL_3},
        {"role": "assistant", "content": C3_ASSISTANT_4},
    ]


USER_PROMPTS = [
    "In a realistic and emotionally evocative pencil sketch style, the composition focuses on a heartwarming indoor scene. Under the dim glow of an oil lamp, a returned soldier son is showing his elderly mother a yellowed letter from home. The soldier, tall and dressed in a dusty military uniform with medals pinned to his chest, leans forward and points at the words on the letter. His mother, with silver hair and a face full of wrinkles, sits on a wooden chair, her eyes glistening with tears of emotion as she gently touches the letter. The soldier's kind-hearted wife stands behind her husband, her hand resting on his shoulder, smiling reassuringly at her mother-in-law. The warm light of the oil lamp illuminates the faces of the three and the letter in their hands, while a faded family portrait hangs on the wall. The entire scene is filled with dramatic lighting and a profound sense of family emotion.",
    "Epic fantasy scene, wide-angle shot. In the dim ancient ruins, a circle of runestones on the ground glows with mysterious light. An elderly white-haired wizard, clad in a deep blue robe adorned with stars, wears a solemn expression as he chants a spell with both hands outstretched. Before him hovers an open, glowing blue magic book. He is protecting a young and beautiful elf princess, who has pointed ears and golden hair, dressed in an emerald-green gown. She tightly grips a life staff topped with a shining green gem, watching the enemy nervously. Their foe is a dark knight clad in full black runic armor, his face unseen, with ominous red light seeping through the cracks in his armor. He raises a massive black runic sword, poised to strike. Dynamic poses, dramatic lighting, digital painting, intricate details, cinematic feel.",
    "In a dimly lit ancient stone chamber, the flames danced in the fireplace. An elderly rune master, dressed in a dark robe with silver-white hair and beard, was holding a wooden staff and pointing at an unfolded, weathered parchment scroll, imparting ancient knowledge to a young Celtic priestess. The priestess wore a green linen dress adorned with Celtic knots, her red hair braided into intricate plaits, and she gazed intently at the complex Norse runes on the scroll. Beside them, a sharp-eyed Viking warrior clad in leather armor stood with his arms crossed, observing the scene with curiosity. In the background, a massive runestone stood upright. The composition is a mid-shot, with strong contrasts of light and shadow.",
]

OVERFIT_PROMPTS = USER_PROMPTS[:3]


def build_prompt_messages(user_text: str, *, class_id: int = 1) -> list[dict]:
    """System + one class demonstration + the live user turn (with brevity reminder)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_demo_messages(class_id),
        {"role": "user", "content": _with_brevity(user_text)},
    ]


def build_ground_truth(user_text: str, *, class_id: int = 1, overfit: bool = False) -> dict:
    """Weights for ``agentic_reward.compute_score`` (actor self-reflection protocol)."""
    del overfit
    expected = 1 + (class_id % 3)
    return {
        "user_request": user_text,
        "demo_class": int(class_id % 3),
        "expected_num_images": expected,
        # Qwen3.5 already calls tools reliably. Frozen-VL correctness/aesthetics
        # are 60% of the signal; protocol remains a gate, not the easy objective.
        "w_tool_call": 0.05,
        "w_brevity": 0.05,
        "w_format": 0.05,
        "w_reflect": 0.10,
        "w_tool": 0.10,
        "w_result": 0.05,
        "w_correctness": 0.30,
        "w_aesthetics": 0.30,
        "forced_consolation": 0.0,
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
        prompt_text = prompt_pool[i % len(prompt_pool)]
        class_id = i % 3
        gt = build_ground_truth(prompt_text, class_id=class_id, overfit=overfit)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": build_prompt_messages(prompt_text, class_id=class_id),
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "raw_prompt": prompt_text,
                    "toy_agentic": True,
                    "overfit": overfit,
                    "demo_class": class_id,
                    "expected_num_images": gt["expected_num_images"],
                    "native_tool_template": True,
                    "visual_tool_observation": True,
                    **{
                        k: gt[k]
                        for k in (
                            "w_tool_call",
                            "w_brevity",
                            "w_format",
                            "w_reflect",
                            "w_tool",
                            "w_result",
                            "w_correctness",
                            "w_aesthetics",
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
        help="Repeat 3 prompts (one per demo class) for short overfit e2e",
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
    train_n = 9 if args.overfit and args.train_size > 9 else args.train_size
    val_n = 3 if args.overfit and args.val_size > 3 else args.val_size
    train_df = pd.DataFrame(build_rows("train", train_n, prompts, overfit=args.overfit))
    val_df = pd.DataFrame(build_rows("val", val_n, prompts, overfit=args.overfit))
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "val.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")
    print(f"demo classes={{0:single,1:two-pass,2:three-pass}}; overfit={args.overfit}; tool_call_format={fmt}")


if __name__ == "__main__":
    main()
