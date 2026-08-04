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

"""Per-task trajectory binding for rollout artifact paths.

Kept in a tiny module (no ``@function_tool``) so ``ForceToolAgentLoop`` can
import it without re-loading ``diffusion_tool`` under a second module name,
which would double-register ``generate_image``.

Artifact layout (under ``rollout_images`` / ``rollout_trajectories``)::

    step_{global_step:06d}/g{sample_index}_n{rollout_n:02d}_{rid}/
        image_00.png ...
        meta.json   # user_prompt + per-call tool_prompt
    step_{...}/g{...}_n{...}_{rid}.json

One training step can contain several GRPO groups (``train_batch_size`` prompts)
and each group has ``rollout.n`` samples. We keep a single folder level under
``step_*``: the group id and within-group rollout index are encoded in the name
(``g3_n00_…``, ``g3_n01_…``) instead of nested ``group_*/rollout_*/`` dirs.
"""

from __future__ import annotations

import contextvars
import re

# Relative path under the images/trajectories roots, e.g.
# ``step_000020/g3_n01_abc123def456``.
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


# Back-compat aliases used by earlier ForceToolAgentLoop / smoke tests.
set_active_trajectory_name = set_active_trajectory_relpath
get_active_trajectory_name = get_active_trajectory_relpath


def build_trajectory_relpath(
    *,
    step: int | None,
    sample_index: int | str | None,
    rollout_n: int | None,
    rollout_id: str,
) -> str:
    """Build ``step_XXXXXX/g{sample}_n{rollout}_{rid}`` relative path."""
    try:
        step_i = int(step) if step is not None else -1
    except (TypeError, ValueError):
        step_i = -1
    step_part = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"

    if sample_index is None:
        sample_part = "unknown"
    else:
        raw = str(sample_index)
        sample_part = re.sub(r"[^\w.\-]+", "_", raw)[:64] or "unknown"

    try:
        rn = int(rollout_n) if rollout_n is not None else 0
    except (TypeError, ValueError):
        rn = 0
    rid = (rollout_id or "x")[:12]
    sample_dir = f"g{sample_part}_n{rn:02d}_{rid}"
    return f"{step_part}/{sample_dir}"
