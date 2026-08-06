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
"""Toy / overfit parquet for Mode (2a) agentic GRPO (Lance und + frozen tool).

Each sample seeds a **full** Hermes trajectory so cold und can imitate:

  generate_image → tool obs → Reflection (on the image) → rewritten
  generate_image → tool obs → short final confirmation

Use ``--overfit`` for the 100-step e2e (1–2 prompts repeated).
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

DATA_SOURCE = "jpeg_compressibility"
ABILITY = "agentic_prompt_rewrite"
EXPECTED_NUM_IMAGES = 2

SYSTEM_PROMPT = """You are a visual creation agent that improves images by reflection.

Workflow (required, every task):
1) Emit Hermes tool call generate_image with a detailed first prompt.
2) After the tool observation, write one short line starting with \
"Reflection:" that names what to improve in the *generated image* \
(detail, lighting, attributes). Ground the critique in the tool result.
3) Emit a SECOND Hermes generate_image call with a REWRITTEN prompt \
(must differ from the first; add missing attributes / quality cues).
4) After the second tool result, give a short final confirmation (no tool call).

Exact tool format:
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "<complete image prompt>"}}
</tool_call>

Never copy or paraphrase tool observations (no "Lance frozen MoT", no \
"agentic_tool ok=", no path= lines). Never emit bare JSON without <tool_call> \
XML. First assistant turn must be a Hermes tool call only.
"""

# Full demo: call → image obs → reflect-on-image → rewrite call → obs → done.
FEWSHOT_USER = "Generate an image of a red apple"
FEWSHOT_ASSISTANT_1 = (
    "<tool_call>\n"
    '{"name": "generate_image", "arguments": {"prompt": '
    '"a bright red apple on a white table, soft studio lighting"}}\n'
    "</tool_call>"
)
FEWSHOT_TOOL_1 = "agentic_tool ok=1 images=1 path=/tmp/example/image_00.png"
FEWSHOT_ASSISTANT_2 = (
    "Reflection: looking at the generated image, edges are soft and reds "
    "look muted; rewrite for sharper detail and richer color.\n"
    "<tool_call>\n"
    '{"name": "generate_image", "arguments": {"prompt": '
    '"a bright red apple on a white table, soft studio lighting, '
    'highly detailed, sharp focus, richer reds, coherent composition"}}\n'
    "</tool_call>"
)
FEWSHOT_TOOL_2 = "agentic_tool ok=1 images=1 path=/tmp/example/image_01.png"
FEWSHOT_ASSISTANT_3 = "Done. Refined the apple image after reflecting on the first generation."

USER_PROMPTS = [
    "Generate an image of a cat wearing a blue hat",
    "Create a sunset over snowy mountains with a red cabin",
    "Draw a silver robot painting a colorful landscape on an easel",
    "A glass of orange juice next to three green apples on a wooden table",
    "A yellow bicycle leaning against a blue brick wall in soft morning light",
    "A small brown dog wearing red sunglasses sitting on a white sofa",
    "An astronaut holding a purple umbrella on the surface of Mars",
    "A vintage typewriter with the word HELLO typed in bold letters",
]

# Tiny pool for 100-step overfit (repeat these only).
OVERFIT_PROMPTS = USER_PROMPTS[:2]


def build_prompt_messages(user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT_1},
        {"role": "tool", "content": FEWSHOT_TOOL_1},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT_2},
        {"role": "tool", "content": FEWSHOT_TOOL_2},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT_3},
        {"role": "user", "content": user_text},
    ]


def build_ground_truth(user_text: str, *, overfit: bool = False) -> dict:
    """Weights for ``agentic_reward.compute_score`` (hard-gated protocol)."""
    if overfit:
        # Hard gate in agentic_reward zeros incomplete trajs; weights only rank
        # complete gen→reflect→rewrite trajectories. No consolation floor.
        return {
            "user_request": user_text,
            "expected_num_images": EXPECTED_NUM_IMAGES,
            "w_format": 0.15,
            "w_reflect": 0.35,
            "w_tool": 0.35,
            "w_result": 0.15,
            "forced_consolation": 0.0,
        }
    return {
        "user_request": user_text,
        "expected_num_images": EXPECTED_NUM_IMAGES,
        "w_format": 0.15,
        "w_reflect": 0.35,
        "w_tool": 0.35,
        "w_result": 0.15,
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
        gt = build_ground_truth(prompt_text, overfit=overfit)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": build_prompt_messages(prompt_text),
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "raw_prompt": prompt_text,
                    "toy_agentic": True,
                    "overfit": overfit,
                    "expected_num_images": EXPECTED_NUM_IMAGES,
                    "require_multiturn_tools": True,
                    "require_reflection_between_tools": True,
                    **{k: gt[k] for k in ("w_format", "w_reflect", "w_tool", "w_result")},
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy/overfit agentic GRPO parquet (reflect → 2nd tool call)")
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/agentic"))
    parser.add_argument("--train_size", type=int, default=64)
    parser.add_argument("--val_size", type=int, default=8)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Repeat 2 prompts only + reflection-heavy reward weights (100-step e2e)",
    )
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)
    prompts = OVERFIT_PROMPTS if args.overfit else None
    train_n = 8 if args.overfit and args.train_size > 8 else args.train_size
    val_n = 2 if args.overfit and args.val_size > 2 else args.val_size
    train_df = pd.DataFrame(build_rows("train", train_n, prompts, overfit=args.overfit))
    val_df = pd.DataFrame(build_rows("val", val_n, prompts, overfit=args.overfit))
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "val.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    gt0 = build_ground_truth("x", overfit=args.overfit)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")
    print(
        f"full few-shot (call→reflect→2nd call→done); overfit={args.overfit}; "
        f"w_reflect={gt0['w_reflect']} w_tool={gt0['w_tool']}"
    )


if __name__ == "__main__":
    main()
