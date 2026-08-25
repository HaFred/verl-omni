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

"""Optional ``rl-insight`` online-observability hooks for the agentic pipeline.

`rl-insight <https://github.com/verl-project/rl-insight>`_ is an online
observability layer for RL training: training code emits Prometheus metrics and
OTLP trace spans through a Ray monitor-hub actor, which the RL-Insight server
aggregates into Grafana dashboards with **RL state timelines** (swim lanes per
rollout / worker / replica).

This module is a thin, dependency-optional facade:

- If ``rl_insight`` is not installed, or ``RL_INSIGHT_SERVER_URL`` is unset,
  every call is a silent no-op, so normal training is completely unaffected.
- ``init_rl_insight`` is idempotent (``rl_insight.init`` itself ignores repeat
  calls) and safe to invoke from the trainer driver and every agent-loop worker.
- The emit helpers never raise: observability must never break training.

The state lanes here target the multi-turn image-generation bubble that
``torch.profiler``/``nsys`` cannot see (they capture GPU/CUDA timelines, not the
I/O-bound ``policy-decode -> sidecar tool call -> tool-obs`` loop).  Each rollout
gets one lane (``state_lane_id = trajectory_relpath``) on which we paint:

    decode           policy decode (vLLM actor replica)
    generate_image   blocking HTTP round-trip to the frozen diffusion sidecar
    judge_image      blocking HTTP round-trip to the frozen VL-judge sidecar

so the sidecar bubbles appear directly as gaps between ``decode`` spans.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any

logger = logging.getLogger(__name__)

try:
    import rl_insight
except Exception:  # noqa: BLE001 - optional dependency; never fatal
    rl_insight = None


def _server_url() -> str:
    return os.getenv("RL_INSIGHT_SERVER_URL", "").strip()


def _available() -> bool:
    return rl_insight is not None and bool(_server_url())


def init_rl_insight(
    project: str | None = None,
    experiment_name: str | None = None,
    config: Any = None,
) -> None:
    """Initialize rl-insight for this process (idempotent; no-op if unavailable).

    Call once per process that emits metrics/traces: the trainer driver and each
    agent-loop worker. ``rl_insight.init`` ignores repeat calls with a warning;
    we suppress that by calling through the facade only once per process state.
    """
    if rl_insight is None:
        return
    if not _server_url():
        return
    try:
        rl_insight.init(project=project, experiment_name=experiment_name, config=config)
    except Exception as exc:  # noqa: BLE001 - observability must never break training
        logger.warning("[rl-insight] init skipped: %s", exc)


def trace_state(name: str, **kwargs: Any):
    """Swim-lane state interval (context manager); ``nullcontext`` when disabled.

    Args:
        name: Span name / human-readable state label (e.g. ``"decode"``).
        state_lane_id: Lane id grouping intervals in trace UIs (per rollout).
        **kwargs: Extra span attributes (``step``, ``rollout_n``, ...).

    Returns:
        A context manager; the span is emitted on exit. Usable with ``with`` in
        both sync and async code (it is a plain, not async, context manager).
    """
    if not _available():
        return nullcontext()
    try:
        return rl_insight.trace_state(name, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[rl-insight] trace_state(%s) skipped: %s", name, exc)
        return nullcontext()


def metric_histogram(name: str, value: float, documentation: str = "", **labels: Any) -> None:
    """Record one sample into a Prometheus histogram (latency distributions)."""
    if not _available():
        return
    try:
        rl_insight.metric_histogram(name, value, documentation=documentation, **labels)
    except Exception:  # noqa: BLE001
        pass


def metric_count(name: str, amount: float = 1.0, documentation: str = "", **labels: Any) -> None:
    """Record a counter increment (call / parse-failure counts)."""
    if not _available():
        return
    try:
        rl_insight.metric_count(name, amount, documentation=documentation, **labels)
    except Exception:  # noqa: BLE001
        pass


def metric_gauge(name: str, value: float, documentation: str = "", **labels: Any) -> None:
    """Record the latest value of a gauge (e.g. rollout turn count)."""
    if not _available():
        return
    try:
        rl_insight.metric_gauge(name, value, documentation=documentation, **labels)
    except Exception:  # noqa: BLE001
        pass
