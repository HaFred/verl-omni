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
"""Reward function for the Bagel Co-RL (Joint-Training) in-loop RM payload (RFC §4.2).

Wire as the custom reward function of the colocated RM pool — the SAME handles
serve mid-loop GEN scoring and post-hoc episode scoring, so ``compute_score``
dispatches on payload shape::

    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.bagel_rm_image_scorer
    reward.custom_reward_function.name=compute_score

Call contract (this is what the configured manager actually does)
----------------------------------------------------------------
``RewardLoopWorker.compute_score(data)`` → ``reward_manager.run_single(data)``.
``NaiveRewardManager`` (the repo default) reads ``data_source`` /
``reward_model["ground_truth"]`` and calls the reward function with exactly four
keywords — ``data_source``, ``solution_str``, ``ground_truth``, ``extra_info``
(``VisualRewardManager`` passes ``solution_image`` instead), and merges the
returned dict's keys into ``reward_extra_info``. A reward function that takes a
single positional ``data`` therefore fails with ``TypeError``/``KeyError``; so
does one that returns ``reward_score`` instead of ``score``.

The payload must ride a key the manager already forwards. ``extra_info`` is that
key: the in-loop builder
(``verl_omni.agent_loop.bagel_corl_rm._default_data_builder``) puts the payload
under ``extra_info["bagel_corl"]``.

- Mid-loop (``verl_omni.agent_loop.bagel_corl_rm.build_rm_score_payload``): each
  image is judged (C/A via the frozen VL judge) and the result carries per-image
  ``sample_scores`` / ``sample_good_enough`` aligned to ``image_paths`` — exactly
  what ``parse_rm_result`` consumes off ``reward_extra_info``.
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

__all__ = ["BAGEL_RM_EXTRA_INFO_KEY", "compute_score"]

# Wire key shared with the producer (verl_omni/agent_loop/bagel_corl_rm.py). Spelled
# out in both modules rather than imported so this module stays importable in
# isolation (the CPU test loads it by path without the agent_loop package).
BAGEL_RM_EXTRA_INFO_KEY = "bagel_corl"


def _mid_loop_payload(extra_info: Any) -> dict[str, Any] | None:
    """Return the in-loop payload riding ``extra_info``, or ``None`` for episodes."""
    if not isinstance(extra_info, dict):
        return None
    payload = extra_info.get(BAGEL_RM_EXTRA_INFO_KEY)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(
            f"bagel_rm_image_scorer: extra_info[{BAGEL_RM_EXTRA_INFO_KEY!r}] must be a dict, "
            f"got {type(payload)!r}"
        )
    return payload


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch on payload shape: mid-loop image scoring vs post-hoc episode scoring.

    Args:
        data_source: Unused; kept for the verl ``compute_score`` signature.
        solution_str: Decoded trajectory text (``NaiveRewardManager``). Unused —
            the in-loop payload carries image paths, not text.
        ground_truth: Episode ground truth, forwarded to the episode scorer.
        extra_info: Manager-forwarded metadata. Carries ``bagel_corl`` →
            mid-loop GEN scoring; absent → post-hoc episode scoring.

    Returns:
        Flat dict with ``score`` plus metric keys (``sample_scores``,
        ``sample_good_enough``, ``good_enough`` for the in-loop path). Flat, not
        nested under ``reward_extra_info``: the manager does the nesting.

    Raises:
        ValueError: If the payload is malformed, ``good_enough_threshold`` is
            missing, or any image fails to score (fail-loud, no zero-fill).
    """
    payload = _mid_loop_payload(extra_info)
    if payload is None:
        from verl_omni.utils.reward_score.agentic_multidim_reward import compute_score as episode_compute_score

        return episode_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    from verl_omni.utils.reward_score.agentic_image_judge_client import call_reflect_vlm

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
    # Flat, manager-shaped dict: NaiveRewardManager does result["score"] and merges
    # every key into reward_extra_info, which is where parse_rm_result looks for
    # sample_scores / sample_good_enough. Nesting these under reward_extra_info
    # here would KeyError on "score" before scoring ever returns.
    return {
        "score": float(reward_score),
        "sample_scores": scores,
        "sample_good_enough": flags,
        "good_enough": all(flags),
        "per_image": per_image,
    }
