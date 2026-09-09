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

"""Run-dir roots, artifact ids, and ``step_*/sample_*.*`` relpaths."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from verl_omni.tools.trajectory.hydra_env import agentic_get

__all__ = [
    "bind_run_artifacts",
    "build_artifact_id",
    "build_trajectory_relpath",
    "clear_run_artifacts",
    "get_run_name",
    "resolve_rollout_images_root",
    "resolve_run_dir",
    "rollout_id_from_relpath",
]

_run_name: str = "agentic_run"
_e2e_root: Path | None = None
_diffusion_image_dir: Path | None = None


def default_e2e_root() -> Path:
    """Repo ``outputs/e2e`` (absolute). Prefer ``VERLOMNI_ROOT`` when set."""
    verlomni = os.getenv("VERLOMNI_ROOT", "").strip()
    if verlomni:
        return Path(verlomni).expanduser().resolve() / "outputs" / "e2e"
    # <repo>/verl_omni/tools/trajectory/this_file.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "outputs" / "e2e"


def clear_run_artifacts() -> None:
    """Reset process-local run-dir bindings (tests)."""
    global _run_name, _e2e_root, _diffusion_image_dir
    _run_name = "agentic_run"
    _e2e_root = None
    _diffusion_image_dir = None


def get_run_name() -> str:
    """Bound experiment / run name (from ``trainer.experiment_name``)."""
    return _run_name or "agentic_run"


def resolve_e2e_root() -> Path:
    """Shared e2e artifact root for traj / images / hermes_actions."""
    if _e2e_root is not None:
        return _e2e_root
    return default_e2e_root()


def resolve_run_dir() -> Path:
    """Per-run dir: ``<e2e_root>/<experiment_name>/``."""
    if _diffusion_image_dir is not None:
        return _diffusion_image_dir.parent
    return resolve_e2e_root() / get_run_name()


def resolve_rollout_images_root() -> Path:
    """``<run_dir>/rollout_images`` (or explicit diffusion image dir override)."""
    if _diffusion_image_dir is not None:
        return _diffusion_image_dir
    return resolve_run_dir() / "rollout_images"


def bind_run_artifacts(config: Any) -> None:
    """Bind run-dir knobs from Hydra so driver + Ray workers share one layout.

    ``RUN_NAME`` comes from ``trainer.experiment_name``. ``ROOT`` comes from
    Hydra ``agentic_image_gen.e2e_root`` when set, else ``outputs/e2e``.

    Must run on each AgentLoop worker before ``generate_image``.
    """
    global _run_name, _e2e_root, _diffusion_image_dir
    # Drop stale explicit image-dir overrides from a previous bind/test.
    _diffusion_image_dir = None
    if config is None:
        _e2e_root = default_e2e_root()
        return

    try:
        experiment_name = config.trainer.get("experiment_name")
    except Exception:  # noqa: BLE001
        experiment_name = None
    if experiment_name:
        _run_name = str(experiment_name)

    e2e_root = None
    try:
        node = config.get("agentic_image_gen")
    except Exception:  # noqa: BLE001
        node = getattr(config, "agentic_image_gen", None)
    if node is not None:
        try:
            e2e_root = node.get("e2e_root")
        except Exception:  # noqa: BLE001
            e2e_root = getattr(node, "e2e_root", None)
    if e2e_root is None:
        e2e_root = agentic_get("e2e_root")
    if e2e_root:
        _e2e_root = Path(str(e2e_root)).expanduser().resolve()
    else:
        _e2e_root = default_e2e_root()


def rollout_id_from_relpath(relpath: str | None) -> str | None:
    """Short stable id for a trajectory folder (``sha256(relpath)[:16]``)."""
    text = (relpath or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_artifact_id(*, relpath: str, index: int, prompt: str) -> str:
    """Identity hash for one generate_image save (not a pixel content hash)."""
    blob = f"{relpath}\0{int(index)}\0{(prompt or '').strip()}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _sanitize_sample_index(sample_index: object | None) -> str:
    if sample_index is None:
        return "unknown"
    try:
        return str(int(sample_index))
    except (TypeError, ValueError):
        raw = str(sample_index)
        return re.sub(r"[^\w.\-]+", "_", raw)[:64] or "unknown"


def build_trajectory_relpath(*, step: int | None, sample_index: object | None, rollout_n: int) -> str:
    """Build ``step_XXXXXX/sample_{index}.{rollout_n:02d}``."""
    try:
        step_i = int(step) if step is not None else -1
    except (TypeError, ValueError):
        step_i = -1
    step_part = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
    sample_part = f"sample_{_sanitize_sample_index(sample_index)}.{int(rollout_n):02d}"
    return f"{step_part}/{sample_part}"
