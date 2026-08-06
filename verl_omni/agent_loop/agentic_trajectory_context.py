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

"""Per-task binding shared by the agent loop and diffusion tool.

Kept in a tiny module (no ``@function_tool``) so ``AgenticForceToolAgentLoop`` can
import it without re-loading ``diffusion_tool`` under a second module name,
which would double-register ``generate_image``.

Artifact layout (under ``rollout_images`` / ``rollout_trajectories``)::

    step_{global_step:06d}/sample_{index}.{rollout_n:02d}/
        image_00.png ...
        meta.json
    step_{global_step:06d}/sample_{index}.{rollout_n:02d}.json

``global_steps`` comes from stock vLLM generate ``extra_fields`` (no custom
worker). ``rollout_n`` is allocated with an exclusive marker under the step
folder so concurrent Ray workers stay unique without trainer metadata.
"""

from __future__ import annotations

import contextvars
import os
import re
import threading
from pathlib import Path

# Relative path under the images/trajectories roots.
_active_trajectory_relpath: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_trajectory_relpath", default=None
)
# Dataset / task user request for the active trajectory (written into meta.json).
_active_user_prompt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_user_prompt", default=None
)
# Provenance for the *next* generate_image call (reflection → rewrite linkage).
_active_call_provenance: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "agentic_active_call_provenance", default=None
)

_rollout_alloc_lock = threading.Lock()


def set_active_trajectory_relpath(relpath: str | None) -> contextvars.Token:
    """Bind (or clear) the relative artifact path for subsequent tool saves."""
    return _active_trajectory_relpath.set(relpath)


def get_active_trajectory_relpath() -> str | None:
    return _active_trajectory_relpath.get()


def set_active_user_prompt(prompt: str | None) -> contextvars.Token:
    """Bind the dataset user request so ``meta.json`` can record it per call."""
    return _active_user_prompt.set(prompt)


def get_active_user_prompt() -> str | None:
    return _active_user_prompt.get()


def set_active_call_provenance(meta: dict | None) -> contextvars.Token:
    """Bind per-call reflection/rewrite provenance for the next tool save."""
    return _active_call_provenance.set(meta)


def get_active_call_provenance() -> dict | None:
    return _active_call_provenance.get()


# Back-compat aliases used by earlier AgenticForceToolAgentLoop / smoke tests.
set_active_trajectory_name = set_active_trajectory_relpath
get_active_trajectory_name = get_active_trajectory_relpath


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


def allocate_rollout_n(*, artifacts_root: Path | str, step: int | None, sample_index: object | None) -> int:
    """Allocate a unique rollout index under ``step_*/`` via exclusive markers."""
    try:
        step_i = int(step) if step is not None else -1
    except (TypeError, ValueError):
        step_i = -1
    step_part = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
    sample_key = _sanitize_sample_index(sample_index)
    step_dir = Path(artifacts_root) / step_part
    step_dir.mkdir(parents=True, exist_ok=True)

    with _rollout_alloc_lock:
        n = 0
        while n < 10_000:
            marker = step_dir / f".alloc_sample_{sample_key}.{n:02d}"
            try:
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return n
            except FileExistsError:
                n += 1
    raise RuntimeError(f"exhausted rollout_n allocation under {step_dir} for sample {sample_key}")
