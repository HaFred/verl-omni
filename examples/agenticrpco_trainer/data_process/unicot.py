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
"""
Preprocess the UniCoT-Self-Reflection-6K dataset for PR1 agentic RL training.

Downloads from HuggingFace (Fr0zencr4nE/UniCoT-Self-Reflection-6K), applies
fail-closed validation per RFC S7, and writes train/val parquet splits in
verl-compatible format.  Each output row is a prompt seed for the multi-turn
agentic rollout loop.

Usage::

    python examples/agenticrpco_trainer/data_process/unicot.py \
        --local_save_dir ~/data/agentic \
        --eval_ratio 0.1

Output::

    ~/data/agentic/train.parquet    # 90 % of valid trajectories
    ~/data/agentic/val.parquet      # 10 % hold-out split (by data_id hash)

Parquet schema::

    raw_prompt       str     Prompt text for the agentic rollout (first-turn
                             edit or task description).
    data_source      str     ``"unicot_self_reflection_6k"``.
    reward_model     dict    ``{"ground_truth": ground_truth_dict}`` where
                             ground_truth_dict contains reference reflection
                             steps (eval_summary, action, edit) and image
                             URIs for reward computation.
    extra_info       dict    Provenance fields (trajectory_id, num_turns,
                             source_dataset, pipeline_variant).
    num_turns        int     Number of reflection steps in the reference
                             trajectory (informational, not used in training).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from typing import Optional

import pandas as pd

# -- add repo root so we can import the adapter without installing --
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from verl_omni.agent_loop.agentic_trajectory import (  # noqa: E402
    ImageRef,
    ReflectionStep,
    VisualReflectionTrajectory,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("unicot_preprocess")


# ---------------------------------------------------------------------------
# UniCoT adapter — mirrors verl_omni.utils.dataset.unicot_adapter
# ---------------------------------------------------------------------------


def _adapt_group(
    data_id: int, group: pd.DataFrame
) -> Optional[VisualReflectionTrajectory]:
    """Adapt a single UniCoT data_id group to VisualReflectionTrajectory.

    Returns None on fail-closed validation (length mismatches, empty
    reflections, continue-with-empty-edit, contradictory terminals).
    """
    rows = group.sort_values("state_index")
    input_imgs = rows["input_image"].tolist()
    output_imgs = rows["output_image"].tolist()
    refs = rows["eval_summary"].tolist()
    edits = rows["edit"].tolist()
    n = len(rows)

    if not (len(input_imgs) == len(output_imgs) == len(refs) == len(edits)):
        return None

    images: list[ImageRef] = []
    steps: list[ReflectionStep] = []

    for i in range(n):
        ref_text = str(refs[i] or "").strip()
        if not ref_text:
            return None  # fail-closed: empty reflection

        has_next = (i + 1 < n) and (output_imgs[i] is not None)

        if has_next:
            edit_text = str(edits[i] or "").strip()
            if not edit_text:
                return None  # fail-closed: continue with empty edit
            action: str = "continue"
        else:
            edit_text = ""
            action = "stop"

        img_uri = str(input_imgs[i]) if input_imgs[i] is not None else ""
        images.append(ImageRef(uri=img_uri))

        steps.append(
            ReflectionStep(reflection=ref_text, action=action, edit=edit_text)
        )

    # Build a meaningful prompt from the first turn's edit
    first_edit = steps[0].edit if steps else ""
    prompt_text = first_edit if first_edit else f"uniCoT trajectory {data_id}"

    return VisualReflectionTrajectory(
        trajectory_id=str(data_id),
        prompt=prompt_text,
        images=images,
        steps=steps,
        source_dataset="UniCoT-Self-Reflection-6K",
        source_record_id=str(data_id),
        pipeline_variant="direct_parse",
    )


def _trajectory_to_row(traj: VisualReflectionTrajectory) -> dict:
    """Convert a VisualReflectionTrajectory to a verl-compatible parquet row."""
    gt_steps = []
    for step in traj.steps:
        gt_steps.append(
            {"reflection": step.reflection, "action": step.action, "edit": step.edit}
        )

    ground_truth = {
        "steps": gt_steps,
        "num_turns": len(traj.steps),
        "image_uris": [img.uri for img in traj.images],
    }

    return {
        "raw_prompt": traj.prompt,
        "data_source": "unicot_self_reflection_6k",
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {
            "trajectory_id": traj.trajectory_id,
            "num_turns": len(traj.steps),
            "source_dataset": traj.source_dataset,
            "pipeline_variant": traj.pipeline_variant,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess UniCoT-Self-Reflection-6K for agentic RL training"
    )
    parser.add_argument(
        "--dataset",
        default="Fr0zencr4nE/UniCoT-Self-Reflection-6K",
        help="HuggingFace dataset ID (default: %(default)s)",
    )
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/agentic"),
        help="Output directory for train.parquet / val.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of valid trajectories to hold out for eval (default: %(default).1f)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility (default: %(default)d)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load from HuggingFace
    # ------------------------------------------------------------------
    logger.info("Loading dataset: %s", args.dataset)
    try:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split="train")
        df = ds.to_pandas()
    except ImportError:
        logger.error("`datasets` package not installed.  Run: pip install datasets")
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load dataset %s: %s", args.dataset, e)
        sys.exit(1)

    logger.info("Raw dataset: %d rows, %d columns", len(df), len(df.columns))
    logger.info("Columns: %s", list(df.columns))

    # ------------------------------------------------------------------
    # 2. Adapt to VisualReflectionTrajectory with fail-closed validation
    # ------------------------------------------------------------------
    trajectories: list[VisualReflectionTrajectory] = []
    dropped = 0

    for data_id, group in df.groupby("data_id"):
        traj = _adapt_group(data_id, group)
        if traj is not None:
            trajectories.append(traj)
        else:
            dropped += 1

    total = len(trajectories) + dropped
    pass_rate = 100 * len(trajectories) / total if total > 0 else 0
    logger.info(
        "Adapted: %d valid trajectories, %d dropped (%.1f%% pass rate)",
        len(trajectories),
        dropped,
        pass_rate,
    )

    if not trajectories:
        logger.error("No valid trajectories — check dataset format.")
        sys.exit(1)

    turn_counts = [len(t.steps) for t in trajectories]
    logger.info(
        "Turn distribution: min=%d, max=%d, mean=%.1f, median=%d",
        min(turn_counts),
        max(turn_counts),
        sum(turn_counts) / len(turn_counts),
        sorted(turn_counts)[len(turn_counts) // 2],
    )

    # ------------------------------------------------------------------
    # 3. Train / val split by trajectory_id hash
    # ------------------------------------------------------------------
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for traj in trajectories:
        h = int(hashlib.md5(traj.trajectory_id.encode()).hexdigest(), 16)
        row = _trajectory_to_row(traj)
        if (h % 100) < (args.eval_ratio * 100):
            val_rows.append(row)
        else:
            train_rows.append(row)

    logger.info(
        "Split: %d train, %d val (eval_ratio=%.2f)",
        len(train_rows),
        len(val_rows),
        args.eval_ratio,
    )

    # ------------------------------------------------------------------
    # 4. Write parquet
    # ------------------------------------------------------------------
    os.makedirs(args.local_save_dir, exist_ok=True)

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "val.parquet")

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    logger.info("Wrote train: %s (%d rows)", train_path, len(train_rows))
    logger.info("Wrote val:   %s (%d rows)", val_path, len(val_rows))


if __name__ == "__main__":
    main()
