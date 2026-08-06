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
"""Tiny Qwen3-VL overfit parquet for visual reflection + prompt rewriting.

Qwen3-VL's native chat template supplies the Hermes tool schema and the model
already knows ``<tool_call>``. One demonstration teaches only the task-specific
policy: generate → inspect attached image → reflect → rewrite → generate.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

DATA_SOURCE = "jpeg_compressibility"
ABILITY = "agentic_hermes_tool_call"
EXPECTED_NUM_IMAGES = 2

# The model's native template defines the wire format. This prompt defines the
# visual-refinement behavior, without teacher-generating any runtime action.
SYSTEM_PROMPT = """You are a visual creation agent with a generate_image tool.
Call it once with a complete prompt. Inspect the returned image itself and its
image_vis measurements. In your next assistant turn, write one concise line
starting with "Reflection:" that identifies a visible shortcoming, then call
generate_image again with a materially rewritten prompt that addresses it.
After inspecting the second image, briefly finish. Never invent tool results or
copy path= and agentic_tool metadata into your reply.
"""

# --- One two-call visual-refinement demonstration ---------------------------
FS2_USER = "Generate an image of a red apple"
FS2_ASSISTANT_1 = (
    "<tool_call>\n"
    '{"name": "generate_image", "arguments": {"prompt": '
    '"a bright red apple on a white table, soft studio lighting"}}\n'
    "</tool_call>"
)
FS2_TOOL_1 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/apple_00.png "
    "image_vis=512x512 mean_luma=92 edges=soft colors=red_muted "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    "prompt='a bright red apple on a white table, soft studio lighting'"
)
FS2_ASSISTANT_2 = (
    "Reflection: image_vis shows mean_luma=92 and colors=red_muted with soft edges; "
    "rewrite for brighter reds and sharper detail.\n"
    "<tool_call>\n"
    '{"name": "generate_image", "arguments": {"prompt": '
    '"a bright red apple on a white table, soft studio lighting, '
    'highly detailed, sharp focus, richer reds, coherent composition"}}\n'
    "</tool_call>"
)
FS2_TOOL_2 = (
    "Frozen diffusion produced the image. path=/tmp/fewshot/apple_01.png "
    "image_vis=512x512 mean_luma=140 edges=medium colors=red_rich "
    "agentic_tool ok=1 stub=0 images=1 backend=fewshot "
    "prompt='a bright red apple on a white table, soft studio lighting, "
    "highly detailed, sharp focus, richer reds, coherent composition'"
)

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

OVERFIT_PROMPTS = USER_PROMPTS[:2]


def build_prompt_messages(user_text: str) -> list[dict]:
    """System + one visual-refinement demonstration + the live user turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Demonstrate the objective, not basic tool syntax discovery.
        {"role": "user", "content": FS2_USER},
        {"role": "assistant", "content": FS2_ASSISTANT_1},
        {"role": "tool", "content": FS2_TOOL_1},
        {"role": "assistant", "content": FS2_ASSISTANT_2},
        {"role": "tool", "content": FS2_TOOL_2},
        # Live task
        {"role": "user", "content": user_text},
    ]


def build_ground_truth(user_text: str, *, overfit: bool = False) -> dict:
    """Tool-call-first weights for ``agentic_reward.compute_score``."""
    del overfit
    return {
        "user_request": user_text,
        "expected_num_images": EXPECTED_NUM_IMAGES,
        "w_format": 0.35,
        "w_reflect": 0.20,
        "w_tool": 0.35,
        "w_result": 0.10,
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
                    "native_tool_template": True,
                    "visual_tool_observation": True,
                    **{k: gt[k] for k in ("w_format", "w_reflect", "w_tool", "w_result")},
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3-VL visual-reflection GRPO overfit parquet")
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/agentic"))
    parser.add_argument("--train_size", type=int, default=64)
    parser.add_argument("--val_size", type=int, default=8)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Repeat 2 prompts only (short overfit e2e)",
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
    n_msgs = len(build_prompt_messages("x"))
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")
    print(f"native-tool visual-reflection messages/sample={n_msgs}; overfit={args.overfit}")


if __name__ == "__main__":
    main()
