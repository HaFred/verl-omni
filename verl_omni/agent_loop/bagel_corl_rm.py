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
"""Mid-episode RM scoring adapter for Bagel Co-RL (RFC §4.2).

The rollout worker holds the dual reward handles (``reward_loop_worker_handles``,
``[0]`` = DiT/GEN pool, ``[1]`` = LLM/AR pool — the ``CompositeAgentLoopWorker``
contract). ``bind_bagel_rm_handles`` stashes the DiT-side handle process-locally
and ``make_rm_score_fn`` turns it into the ``score_fn`` that
``bagel_corl_lib.run_serial_episode`` already consumes — GEN call samples get
``rm_score`` / ``good_enough`` **inside the episode**, which is what drives the
``good_enough`` stop/continue cue.

Payload contract for the reward-loop worker (``compute_score``): a verl
``DataProto`` whose ``non_tensor_batch`` carries one object row
``{"bagel_rm_payload": <dict>}`` with keys ``image_paths``, ``reference_paths``,
``extra_info``, ``scorer_knobs``. The reward function wired on the RM pool must
read that key (recipe-level wiring); ``reward_score`` comes back as a scalar or
per-image ``reward_extra_info["sample_scores"]`` aligned with ``image_paths``.

This module stays torch-free at import time (verl is imported lazily inside the
closure) so the pure payload/parse logic is unit-testable without the training
stack.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RMScoringError",
    "bind_bagel_rm_handles",
    "build_rm_score_payload",
    "get_bagel_rm_dit_handle",
    "make_rm_score_fn",
    "parse_rm_result",
]

_RM_HANDLES_KEY = "_bagel_corl_rm_handles"
# [0] is the DiT/GEN pool under the CompositeAgentLoopWorker handle order.
_RM_DIT_HANDLE_INDEX = 0


class RMScoringError(RuntimeError):
    """Raised when the in-loop RM handle is missing, malformed, or fails."""


def bind_bagel_rm_handles(handles: list[Any] | None) -> None:
    """Stash the reward-loop handles process-locally (worker ``__init__`` time).

    Args:
        handles: ``[dit_handle, ar_handle]`` per the composite contract, or
            ``None`` when no RM pool is wired for this worker.
    """
    import sys

    module = sys.modules[__name__]
    if handles is None:
        setattr(module, _RM_HANDLES_KEY, None)
        return
    if not isinstance(handles, (list, tuple)):
        raise RMScoringError(f"bagel RM handles must be a list, got {type(handles)!r}")
    if len(handles) < _RM_DIT_HANDLE_INDEX + 1:
        raise RMScoringError(
            f"bagel RM handles need at least {_RM_DIT_HANDLE_INDEX + 1} entries (DiT pool first), got {len(handles)}"
        )
    setattr(module, _RM_HANDLES_KEY, list(handles))


def get_bagel_rm_dit_handle() -> Any | None:
    """Return the DiT-side RM handle bound in this process, or ``None``."""
    import sys

    handles = getattr(sys.modules[__name__], _RM_HANDLES_KEY, None)
    if not handles:
        return None
    return handles[_RM_DIT_HANDLE_INDEX]


def build_rm_score_payload(
    image_paths: list[str],
    reference_paths: list[str] | None = None,
    extra_info: dict[str, Any] | None = None,
    scorer_knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the plain-dict payload handed to the RM worker's ``compute_score``."""
    if not image_paths:
        raise RMScoringError("bagel RM payload needs at least one image path")
    return {
        "image_paths": [str(p) for p in image_paths],
        "reference_paths": [str(p) for p in (reference_paths or [])],
        "extra_info": dict(extra_info or {}),
        "scorer_knobs": dict(scorer_knobs or {}),
    }


def parse_rm_result(
    result: Any,
    image_paths: list[str],
) -> tuple[list[float], list[bool | None]]:
    """Parse a ``compute_score`` result into per-image scores + per-image stop bits.

    Accepts ``reward_score`` as a scalar (broadcast to every image) or
    ``reward_extra_info["sample_scores"]`` aligned with ``image_paths``. The stop
    bit comes from ``reward_extra_info["sample_good_enough"]`` (per-image flags,
    preferred) or ``reward_extra_info["good_enough"]`` (broadcast).
    """
    if not isinstance(result, dict) or "reward_score" not in result:
        raise RMScoringError(f"bagel RM result missing 'reward_score': {type(result)!r}")
    extra = result.get("reward_extra_info") or {}
    if not isinstance(extra, dict):
        extra = {}
    sample_scores = extra.get("sample_scores")
    if sample_scores is not None:
        scores = [float(s) for s in sample_scores]
        if len(scores) != len(image_paths):
            raise RMScoringError(
                f"bagel RM sample_scores length {len(scores)} != image count {len(image_paths)}"
            )
    else:
        scores = [float(result["reward_score"])] * len(image_paths)
    sample_flags = extra.get("sample_good_enough")
    if sample_flags is not None:
        if len(sample_flags) != len(image_paths):
            raise RMScoringError(
                f"bagel RM sample_good_enough length {len(sample_flags)} != image count {len(image_paths)}"
            )
        flags = [None if flag is None else bool(flag) for flag in sample_flags]
    elif "good_enough" in extra and extra["good_enough"] is not None:
        flags = [bool(extra["good_enough"])] * len(image_paths)
    else:
        flags = [None] * len(image_paths)
    return scores, flags


def _default_data_builder(payload: dict[str, Any]) -> Any:
    """Wrap the payload in a one-row verl ``DataProto`` (the worker-side contract)."""
    from tensordict import TensorDict
    from verl.protocol import DataProto

    return DataProto(
        batch=TensorDict({}, batch_size=[len(payload["image_paths"])]),
        non_tensor_batch={"bagel_rm_payload": np.array([payload], dtype=object)},
    )


def make_rm_score_fn(
    handle: Any,
    *,
    timeout_s: float | None = None,
    get_reference_paths: Callable[[], list[str]] | None = None,
    get_extra_info: Callable[[], dict[str, Any]] | None = None,
    get_scorer_knobs: Callable[[], dict[str, Any]] | None = None,
    get_image_prompt: Callable[[], str] | None = None,
    data_builder: Callable[[dict[str, Any]], Any] | None = None,
) -> Callable[[list[Any]], Any]:
    """Build the ``score_fn(samples) -> samples`` hook consumed by ``run_serial_episode``.

    The handle is duck-typed so tests can pass a plain object: either an async
    Ray actor (``await handle.compute_score.remote(data)`` — the
    ``CompositeAgentLoopWorker`` idiom) or a plain async/sync callable. ``verl``
    is imported lazily so this module imports without the training stack.
    """
    if handle is None:
        raise RMScoringError(
            "bagel in-loop RM scoring requires a reward-loop handle; "
            "wire agent-loop reward handles (trainer.get_reward_handles) or "
            "disable agent.rm_scoring explicitly"
        )

    async def _invoke(data: Any) -> Any:
        method = getattr(handle, "compute_score", None)
        if method is None:
            raise RMScoringError(f"bagel RM handle {type(handle)!r} has no compute_score")
        if hasattr(method, "remote"):
            result = method.remote(data)
        else:
            result = method(data)
        result = await result if asyncio.iscoroutine(result) else result
        if timeout_s is not None and hasattr(result, "get") and not asyncio.iscoroutine(result):
            # Ray ObjectRef path: enforce the timeout via ray.get when available.
            import ray

            return ray.get(result, timeout=timeout_s)
        return result

    async def score(samples: list[Any]) -> list[Any]:
        scored = [s.image_path for s in samples if getattr(s, "valid", False) and getattr(s, "image_path", None)]
        if not scored:
            return samples
        payload = build_rm_score_payload(
            scored,
            reference_paths=get_reference_paths() if get_reference_paths is not None else [],
            extra_info=get_extra_info() if get_extra_info is not None else {},
            scorer_knobs=get_scorer_knobs() if get_scorer_knobs is not None else {},
        )
        if get_image_prompt is not None:
            payload["image_prompt"] = str(get_image_prompt() or "")
        data = (data_builder or _default_data_builder)(payload)
        scores, flags = parse_rm_result(await _invoke(data), payload["image_paths"])
        by_path = dict(zip(payload["image_paths"], scores, strict=True))
        flag_by_path = dict(zip(payload["image_paths"], flags, strict=True))
        for sample in samples:
            if not getattr(sample, "valid", False) or not getattr(sample, "image_path", None):
                continue
            sample.rm_score = by_path[str(sample.image_path)]
            flag = flag_by_path.get(str(sample.image_path))
            if flag is not None:
                sample.good_enough = flag
        return samples

    return score
