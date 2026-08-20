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
"""UniCoT → agentic RL parquet builder (PR 2 / RPCO stage 3).

This is the GRPO application of the UniCoT parsers, not a generic dataset
loader. Invoke as::

    python -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl

Builds the mixed single-image + multi-image training set for multi-task RL
co-optimization (§6.5 stage 3 of RFC #302) from the local UniCoT snapshots:

- UniCoT-Self-Reflection-6K → ``task_type=reflect`` rows (single-image,
  reference states carry the reflection summaries and the continue/stop
  transition structure).
- UniCoT-Breakdown-3K → ``task_type=plan`` rows (reference subtasks) and
  ``task_type=reflect`` rows ("No breakdown needed." → single image).

UniCoT fields are reward ground truth only — they are never baked into the
prompt as fewshot. Prompt rows are system (per task type) + user + brevity
tail, with the agentic RL row schema (``data_source``, ``prompt`` messages,
``ability``, ``reward_model.ground_truth``, ``extra_info``).

Without ``--train_size``/``--val_size`` every parsed row is used (full-dataset
training; hash-based train/val split at ``--val_ratio``). The sizes exist only
for smoke runs, where ``--mix_ratio`` bounds the reflect/plan sampling.

Image files are not required: validation is structural/text-only. The
full hash-audited image path (``LocalImageResolver``) remains available for
evaluation and future image-backed rewards.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from verl_omni.utils.dataset.visual_reflection import VisualReflectionDataError
from verl_omni.utils.dataset.visual_reflection.contracts import derive_prompt_source_dedup_key
from verl_omni.utils.dataset.visual_reflection.partition import assign_source_splits
from verl_omni.utils.dataset.visual_reflection.unicot import (
    UNICOT_DATASET_ID,
    parse_unicot_record,
)
from verl_omni.utils.dataset.visual_reflection.unicot_breakdown import (
    UNICOT_BREAKDOWN_DATASET_ID,
    parse_unicot_breakdown_record,
)

REFLECT_ABILITY = "agentic_generate_self_reflect"
PLAN_ABILITY = "agentic_plan_generate"
REFLECT_DATA_SOURCE = "unicot_reflection"
BREAKDOWN_DATA_SOURCE = "unicot_breakdown"
DIMS = ("reflect", "plan", "format", "tool_call", "result")
MANIFEST_ID = "agentic_rpco_stage3"

# Reflect protocol (self-contained; no dependency on ``examples/``).
REFLECT_SYSTEM_PROMPT = """You are a visual creation agent with two tools:
1) generate_image — create an image from a complete diffusion prompt
2) judge_image — call a frozen VL judge on the LAST generated image to get
   structured feedback (scores, findings, suggested fixes, good_enough verdict)

Protocol (one logical turn = generate → judge → reflect & decide):
1. Call generate_image with a complete diffusion prompt.
2. After the image returns, call judge_image with SHORT args only:
   user_request="same as user message"
   image_prompt="last"
   The tool judges the latest image against the ORIGINAL user request. Rewritten
   diffusion prompts may improve pixels but never replace the evaluation target.
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


def _with_brevity(user_task: str) -> str:
    """Append the brevity reminder to a user-facing request."""
    task = (user_task or "").rstrip()
    if _BREVITY_TAIL.strip() in task:
        return task
    return task + _BREVITY_TAIL


# Plan task protocol: decompose, generate per subtask, judge only the final
# image, then reflect and stop. Mirrors the reflect SYSTEM_PROMPT conventions.
PLAN_SYSTEM_PROMPT = """You are a visual creation agent with two tools:
1) generate_image — create an image from a complete diffusion prompt
2) judge_image — call a frozen VL judge on the LAST generated image to get
   structured feedback (scores, findings, suggested fixes, good_enough verdict)

Protocol (plan → generate subtasks → judge final → reflect & stop):
1. Read the user request. If it needs multiple subtask images, write a short
   plan: a numbered list of subtask prompts, one per image to generate (at
   most 3). Each subtask prompt must be a complete diffusion prompt.
2. Call generate_image once per subtask, in order. Do NOT call judge_image
   between subtasks.
3. After the LAST subtask image, call judge_image with SHORT args only:
   user_request="same as user message"
   image_prompt="last"
   The latest image is evaluated against the original complete user request,
   not only the final subtask/rewrite prompt.
4. Read the VL feedback, then write your reflection and end with Done. — do
   not generate more images than the plan listed.

HARD RULES (non-negotiable):
- Call generate_image exactly once per planned subtask, in order.
- ALWAYS call judge_image after the final image before deciding Done.
- Never call tools other than generate_image and judge_image.
- Keep judge_image arguments compact (placeholders above). Long pasted args
  waste the response budget and truncate the tool call.

Brevity (mandatory):
- Keep any private thinking to AT MOST one short paragraph (≤4 sentences).
- Do not debate yourself, repeat the user request, or rehash prior turns.
- Prefer emitting the <tool_call> immediately; finish with a one-line Done when done.
- Stop on your own when the task is complete — do not ramble until a length limit.
"""


class _TextOnlyImageResolver:
    """Structural-only image resolver: no pixel reads, no hash auditing.

    Transition hash checks in ``parse_unicot_record`` compare ``sha256`` of
    ``output_image[i]`` against ``input_image[i+1]``; an empty digest passes
    trivially, so structure is validated without materializing ``images.zip``.
    """

    def __call__(self, value: Any, *, field: str = "", index: int = 0, source_record_id: str | None = None) -> dict:
        del field, index
        uri = str(value).strip() if value is not None else ""
        if not uri:
            uri = f"<no-image>:{source_record_id or 'unknown'}"
        # Well-formed constant digest: transition hash checks pass trivially and
        # the format validates, without reading pixels from ``images.zip``.
        return {"uri": uri, "sha256": "0" * 64}


def _env_weight(dim: str) -> float:
    """Read RPCO_W_<DIM> (default 1.0 — VisionCreator-R1 sets all weights to 1)."""
    env_key = f"RPCO_W_{dim.upper()}"
    raw = os.environ.get(env_key)
    # Alias kept so existing launchers that export RPCO_W_TOOL still apply.
    if raw is None and dim == "tool_call":
        raw = os.environ.get("RPCO_W_TOOL")
    if raw is None:
        raw = "1.0"
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise ValueError(f"{env_key} must be a float, got {raw!r}") from None


def _weights() -> dict[str, float]:
    return {f"w_{dim}": _env_weight(dim) for dim in DIMS}


def _load_metadata(dataset_dir: str | None, dataset_id: str) -> list[dict]:
    if not dataset_dir:
        return []
    snapshots = sorted((Path(dataset_dir).expanduser() / "snapshots").glob("*/"))
    for snapshot in snapshots:
        meta = snapshot / "metadata.json"
        if meta.is_file():
            with meta.open() as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"{dataset_id}: metadata.json must be a JSON list, got {type(data).__name__}")
            return data
    raise FileNotFoundError(f"{dataset_id}: no snapshot with metadata.json under {dataset_dir}")


def _parse_reflection_rows(metadata: list[dict]) -> tuple[list[dict], list[dict]]:
    """Parse Self-Reflection rows into agentic RL rows + rejections."""
    rows: list[dict] = []
    rejections: list[dict] = []
    weights = _weights()
    resolver = _TextOnlyImageResolver()
    for record in metadata:
        data_id = str(record.get("data_id") or "")
        try:
            trajectory = parse_unicot_record(
                record,
                manifest_id=MANIFEST_ID,
                image_resolver=resolver,
            )
        except VisualReflectionDataError as error:
            rejections.append({"data_id": data_id, "reason": error.reason.value, "field": error.field})
            continue
        task = trajectory["prompt"]
        expected = len(trajectory["steps"])
        ground_truth = {
            "user_request": task,
            "task_type": "reflect",
            "expected_num_images": expected,
            "reference_steps": trajectory["steps"],
            **weights,
        }
        rows.append(
            {
                "data_id": data_id,
                "task_type": "reflect",
                "prompt_text": task,
                "expected_num_images": expected,
                "ground_truth": ground_truth,
                "source_dataset": UNICOT_DATASET_ID,
                "split_record": {
                    "source_dataset": UNICOT_DATASET_ID,
                    "source_record_id": data_id,
                    "pipeline_variant": "prompt_k_turn",
                    "prompt": task,
                    "dedup_key": derive_prompt_source_dedup_key(task),
                },
            }
        )
    return rows, rejections


def _parse_breakdown_rows(metadata: list[dict]) -> tuple[list[dict], list[dict]]:
    """Parse Breakdown rows into agentic RL rows + rejections."""
    rows: list[dict] = []
    rejections: list[dict] = []
    weights = _weights()
    for record in metadata:
        data_id = str(record.get("data_id") or "")
        try:
            parsed = parse_unicot_breakdown_record(record, manifest_id=MANIFEST_ID)
        except VisualReflectionDataError as error:
            rejections.append({"data_id": data_id, "reason": error.reason.value, "field": error.field})
            continue
        task = parsed.prompt
        if parsed.task_type == "plan":
            ground_truth = {
                "user_request": task,
                "task_type": "plan",
                "expected_num_images": parsed.expected_num_images,
                "reference_subtasks": list(parsed.subtasks),
                "plan_expected": True,
                **weights,
            }
        else:
            ground_truth = {
                "user_request": task,
                "task_type": "reflect",
                "expected_num_images": parsed.expected_num_images,
                "plan_expected": False,
                **weights,
            }
        rows.append(
            {
                "data_id": data_id,
                "task_type": parsed.task_type,
                "prompt_text": task,
                "expected_num_images": parsed.expected_num_images,
                "ground_truth": ground_truth,
                "source_dataset": UNICOT_BREAKDOWN_DATASET_ID,
                "split_record": {
                    "source_dataset": UNICOT_BREAKDOWN_DATASET_ID,
                    "source_record_id": data_id,
                    "pipeline_variant": "prompt_k_turn",
                    "prompt": task,
                    "dedup_key": derive_prompt_source_dedup_key(task),
                },
            }
        )
    return rows, rejections


def _system_prompt(task_type: str) -> str:
    return PLAN_SYSTEM_PROMPT if task_type == "plan" else REFLECT_SYSTEM_PROMPT


def _build_parquet_row(row: dict, *, split: str, index: int) -> dict:
    gt = dict(row["ground_truth"])
    task = row["prompt_text"]
    return {
        "data_source": REFLECT_DATA_SOURCE if row["source_dataset"] == UNICOT_DATASET_ID else BREAKDOWN_DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": _system_prompt(row["task_type"])},
            {"role": "user", "content": _with_brevity(task)},
        ],
        "ability": PLAN_ABILITY if row["task_type"] == "plan" else REFLECT_ABILITY,
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "index": index,
            "data_id": row["data_id"],
            "task_type": row["task_type"],
            "expected_num_images": row["expected_num_images"],
            "raw_prompt": task,
            "unicot_source": row["source_dataset"],
            "plan_expected": bool(gt.get("plan_expected", False)),
            **{key: gt[key] for key in (f"w_{dim}" for dim in DIMS) if key in gt},
        },
    }


def _assign_splits(rows: list[dict], *, seed: int, val_ratio: float) -> dict[str, dict[str, str]]:
    """Hash-based train/validation assignment per source record."""
    assignments = assign_source_splits(
        [row["split_record"] for row in rows],
        ratios={"train": 1.0 - val_ratio, "validation": val_ratio, "test": 0.0},
        seed=seed,
    )
    split_by_id: dict[str, dict[str, str]] = {}
    for identity, assignment in assignments.items():
        split = "val" if assignment["split"] == "validation" else "train"
        split_by_id[f"{identity[0]}\0{identity[1]}"] = {"split": split, "partition_id": assignment["partition_id"]}
    return split_by_id


def _sample_pool(pool: list[dict], *, split: str) -> list[dict]:
    return [row for row in sorted(pool, key=lambda row: row["data_id"]) if row["_split"] == split]


def _mix_pools(
    reflect: list[dict], plan: list[dict], *, n: int | None, mix_ratio: float, rng: random.Random
) -> list[dict]:
    """Select rows for one split.

    ``n=None`` (real training): use the **entire** pool — full dataset
    utilization, natural reflect:plan ratio (mix_ratio is ignored).
    ``n`` given (smoke runs): sample ``round(n * mix_ratio)`` reflect rows and
    the remainder plan rows, capped by availability.
    """
    if n is None:
        return reflect + plan
    n_reflect = min(len(reflect), round(n * mix_ratio))
    n_plan = min(len(plan), max(0, n - n_reflect))
    chosen = rng.sample(reflect, min(n_reflect, len(reflect))) + rng.sample(plan, min(n_plan, len(plan)))
    return sorted(chosen, key=lambda row: row["data_id"])


def build_rows(
    rows: list[dict],
    split_by_id: dict[str, dict[str, str]],
    *,
    split: str,
    n: int | None,
    mix_ratio: float,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    reflect_pool, plan_pool = [], []
    for row in rows:
        identity = f"{row['source_dataset']}\0{row['data_id']}"
        row["_split"] = split_by_id[identity]["split"]
        (reflect_pool if row["task_type"] == "reflect" else plan_pool).append(row)
    reflect_pool = _sample_pool(reflect_pool, split=split)
    plan_pool = _sample_pool(plan_pool, split=split)
    chosen = _mix_pools(reflect_pool, plan_pool, n=n, mix_ratio=mix_ratio, rng=random.Random(seed))
    counts = {"reflect": 0, "plan": 0}
    parquet_rows = []
    for index, row in enumerate(chosen):
        counts[row["task_type"]] += 1
        parquet_rows.append(_build_parquet_row(row, split=split, index=index))
    return parquet_rows, counts


def main_cli(
    *,
    reflection_dir: str,
    breakdown_dir: str,
    local_save_dir: str,
    train_size: int | None,
    val_size: int | None,
    mix_ratio: float,
    seed: int,
    val_ratio: float,
) -> None:
    """Build train/val parquet from UniCoT snapshots (also the test entry point)."""
    if not breakdown_dir and not reflection_dir:
        raise SystemExit("provide at least one of breakdown_dir / reflection_dir (or UNICOT_*_DIR)")
    if not 0.0 < mix_ratio < 1.0:
        raise SystemExit("mix_ratio must be in (0, 1)")
    if not 0.0 < val_ratio < 1.0:
        raise SystemExit("val_ratio must be in (0, 1)")

    all_rows: list[dict] = []
    rejections: list[dict] = []
    if reflection_dir:
        metadata = _load_metadata(reflection_dir, UNICOT_DATASET_ID)
        parsed, rejected = _parse_reflection_rows(metadata)
        all_rows.extend(parsed)
        rejections.extend(rejected)
    if breakdown_dir:
        metadata = _load_metadata(breakdown_dir, UNICOT_BREAKDOWN_DATASET_ID)
        parsed, rejected = _parse_breakdown_rows(metadata)
        all_rows.extend(parsed)
        rejections.extend(rejected)

    split_by_id = _assign_splits(all_rows, seed=seed, val_ratio=val_ratio)
    save_dir = Path(local_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report = {"rejections": rejections, "rejection_count": len(rejections), "splits": {}}
    for split, size in (("train", train_size), ("val", val_size)):
        parquet_rows, counts = build_rows(all_rows, split_by_id, split=split, n=size, mix_ratio=mix_ratio, seed=seed)
        df = pd.DataFrame(parquet_rows)
        df.to_parquet(save_dir / f"{split}.parquet")
        report["splits"][split] = {**counts, "total": len(df)}
        print(f"[INFO] {split}: wrote {len(df)} rows ({counts}) to {save_dir / f'{split}.parquet'}")
    (save_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[INFO] rejected {len(rejections)} source rows; see {save_dir / 'build_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build UniCoT agentic RL parquet (python -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl)"
    )
    parser.add_argument("--breakdown_dir", default=os.environ.get("UNICOT_BREAKDOWN_DIR", ""))
    parser.add_argument("--reflection_dir", default=os.environ.get("UNICOT_REFLECTION_DIR", ""))
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/agentic_unicot"))
    parser.add_argument(
        "--train_size", type=int, default=None, help="Total train rows (None = full dataset)"
    )
    parser.add_argument(
        "--val_size", type=int, default=None, help="Total val rows (None = full val split)"
    )
    parser.add_argument(
        "--mix_ratio",
        type=float,
        default=float(os.environ.get("UNICOT_MIX_RATIO", "0.5")),
        help="Reflect-row fraction, applied only when --train_size/--val_size cap the pools",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=float(os.environ.get("UNICOT_VAL_RATIO", "0.05")),
        help="Hash-based validation split fraction",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("UNICOT_SPLIT_SEED", "42")))
    args = parser.parse_args()
    main_cli(
        reflection_dir=args.reflection_dir,
        breakdown_dir=args.breakdown_dir,
        local_save_dir=args.local_save_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        mix_ratio=args.mix_ratio,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
