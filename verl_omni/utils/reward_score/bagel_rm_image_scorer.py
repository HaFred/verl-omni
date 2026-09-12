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
"""Reward function for the Bagel Co-RL in-loop RM payload (RFC §4.2).

Wire as the custom reward function of the colocated RM pool — the SAME handles
serve mid-loop GEN scoring and post-hoc episode scoring, so ``compute_score``
dispatches on payload shape::

    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.bagel_rm_image_scorer
    reward.custom_reward_function.name=compute_score

- Mid-loop (``verl_omni.agent_loop.bagel_corl_rm.build_rm_score_payload``): the
  ``DataProto`` carries ``non_tensor_batch["bagel_rm_payload"]`` — one object row
  with ``image_paths`` / ``reference_paths`` / ``extra_info`` / ``scorer_knobs`` /
  ``image_prompt``. Each image is judged (C/A via the frozen VL judge) and the
  result carries per-image ``sample_scores`` / ``sample_good_enough`` aligned to
  ``image_paths`` — exactly what ``parse_rm_result`` consumes.
- Episode post-hoc (no payload): delegated to
  ``agentic_multidim_reward.compute_score`` (image-grounded when ``K >= 1``,
  multi-dim RPCO when ``K = 0``) — the authoritative token-GRPO scalar.

Fails loud when any image cannot be scored (missing file, empty judge URL, parse
failure): a silently-unscored GEN call would zero FlowGRPO signal for that group.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["compute_score", "extract_bagel_rm_payload"]


def extract_bagel_rm_payload(data: Any) -> dict[str, Any]:
    """Pull the single ``bagel_rm_payload`` row out of a DataProto-like object."""
    ntb = getattr(data, "non_tensor_batch", None)
    if ntb is None and isinstance(data, dict):
        ntb = data
    if ntb is None:
        raise ValueError("bagel_rm_image_scorer: input carries no non_tensor_batch")
    column = ntb.get("bagel_rm_payload") if hasattr(ntb, "get") else None
    if column is None and isinstance(ntb, dict):
        column = ntb.get("bagel_rm_payload")
    if column is None:
        raise ValueError("bagel_rm_image_scorer: missing non_tensor_batch['bagel_rm_payload']")
    rows = list(column)
    if len(rows) != 1:
        raise ValueError(f"bagel_rm_image_scorer: expected exactly 1 payload row, got {len(rows)}")
    payload = rows[0]
    if not isinstance(payload, dict):
        raise ValueError(f"bagel_rm_image_scorer: payload row must be a dict, got {type(payload)!r}")
    return payload


def _has_bagel_rm_payload(data: Any) -> bool:
    """True when the input carries a mid-loop ``bagel_rm_payload`` row."""
    ntb = getattr(data, "non_tensor_batch", None)
    if ntb is None and isinstance(data, dict):
        ntb = data
    if ntb is None or not hasattr(ntb, "get"):
        return False
    try:
        return ntb.get("bagel_rm_payload") is not None
    except (AttributeError, TypeError):
        return False


def compute_score(data: Any) -> dict[str, Any]:
    """Dispatch on payload shape: mid-loop image scoring vs post-hoc episode scoring.

    Args:
        data: DataProto. With a ``bagel_rm_payload`` row → judge each image
            (mid-loop GEN scoring). Without one → delegate to
            ``agentic_multidim_reward.compute_score`` (the authoritative episode
            scalar for token GRPO).

    Returns:
        ``{"reward_score": float, "reward_extra_info": {...}}``.

    Raises:
        KeyError: If ``good_enough_threshold`` is missing from the knobs.
        ValueError: If any image fails to score (fail-loud, no zero-fill).
    """
    if not _has_bagel_rm_payload(data):
        from verl_omni.utils.reward_score.agentic_multidim_reward import compute_score as episode_compute_score

        return episode_compute_score(data)

    from verl_omni.utils.reward_score.agentic_image_judge_client import call_reflect_vlm

    payload = extract_bagel_rm_payload(data)
    image_paths = [str(p) for p in payload.get("image_paths") or []]
    if not image_paths:
        raise ValueError("bagel_rm_image_scorer: payload has no image_paths")

    extra_info = dict(payload.get("extra_info") or {})
    knobs = dict(payload.get("scorer_knobs") or {})
    knobs.setdefault("good_enough_threshold", extra_info.get("good_enough_threshold"))
    if knobs.get("good_enough_threshold") is None:
        raise ValueError(
            "bagel_rm_image_scorer: good_enough_threshold missing from payload "
            "scorer_knobs/extra_info; the loop must stamp agentic scorer knobs"
        )
    knobs["good_enough_threshold"] = float(knobs["good_enough_threshold"])
    user_request = str(extra_info.get("user_prompt") or extra_info.get("raw_prompt") or "")
    image_prompt = str(payload.get("image_prompt") or "")
    reference_paths = [str(p) for p in payload.get("reference_paths") or []]
    if reference_paths:
        notes = f"reference images: {', '.join(reference_paths)}"
    else:
        notes = str(extra_info.get("judge_notes") or "")

    scores: list[float] = []
    flags: list[bool] = []
    per_image: list[dict[str, Any]] = []
    failed: list[str] = []
    for path in image_paths:
        judged = call_reflect_vlm(
            user_request=user_request,
            image_prompt=image_prompt,
            notes=notes,
            image_path=path,
            extra_info=dict(knobs),
        )
        if judged is None or not judged.get("ok"):
            failed.append(path)
            continue
        correctness = float(judged.get("correctness", 0.0))
        aesthetics = float(judged.get("aesthetics", 0.0))
        match = judged.get("match")
        facets = [correctness, aesthetics] + ([float(match)] if match is not None else [])
        score = sum(facets) / len(facets)
        scores.append(score)
        flags.append(bool(judged.get("good_enough", False)))
        per_image.append({"image_path": path, "score": score, "good_enough": flags[-1]})
    if failed:
        raise ValueError(
            f"bagel_rm_image_scorer: failed to score {len(failed)}/{len(image_paths)} image(s) "
            f"(first: {failed[0]}); refusing zero-fill for FlowGRPO"
        )

    reward_score = sum(scores) / len(scores)
    return {
        "reward_score": float(reward_score),
        "reward_extra_info": {
            "sample_scores": scores,
            "sample_good_enough": flags,
            "good_enough": all(flags),
            "per_image": per_image,
        },
    }
