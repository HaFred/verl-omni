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

"""Artifact path helpers and optional per-task diffusion-tool bindings.

Kept in a tiny module with no ``@function_tool`` registration so path helpers
can be shared by the stock-loop manager and diffusion tool safely.

Artifact layout (under ``rollout_images`` / ``rollout_trajectories``)::

    step_{global_step:06d}/sample_{index}.{rollout_n:02d}/
        image_00.png ...
        meta.json
    step_{global_step:06d}/sample_{index}.{rollout_n:02d}.json

The manager uses stock rollout metadata for trajectory JSON/text dumps. The
context bindings remain optional for standalone/custom tool callers.
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

# Live tool saves register here so the post-hoc manager can place image_00/01
# under step_*/sample_* even when stock ToolAgentLoop never binds contextvars
# and attached vision tokens truncate later ``path=`` markers from the response.
_artifact_registry_lock = threading.Lock()
_artifact_registry: list[dict] = []


def set_active_trajectory_relpath(relpath: str | None) -> contextvars.Token:
    """Bind (or clear) the relative artifact path for subsequent tool saves."""
    return _active_trajectory_relpath.set(relpath)


def reset_active_trajectory_relpath(token: contextvars.Token) -> None:
    """Restore the trajectory path binding that preceded ``token``."""
    _active_trajectory_relpath.reset(token)


def get_active_trajectory_relpath() -> str | None:
    return _active_trajectory_relpath.get()


def set_active_user_prompt(prompt: str | None) -> contextvars.Token:
    """Bind the dataset user request so ``meta.json`` can record it per call."""
    return _active_user_prompt.set(prompt)


def reset_active_user_prompt(token: contextvars.Token) -> None:
    """Restore the user-prompt binding that preceded ``token``."""
    _active_user_prompt.reset(token)


def get_active_user_prompt() -> str | None:
    return _active_user_prompt.get()


def set_active_call_provenance(meta: dict | None) -> contextvars.Token:
    """Bind per-call reflection/rewrite provenance for the next tool save."""
    return _active_call_provenance.set(meta)


def get_active_call_provenance() -> dict | None:
    return _active_call_provenance.get()


def register_tool_artifact(
    *,
    prompt: str,
    paths: list[str],
    backend: str = "",
    tool_stubbed: bool = False,
) -> None:
    """Record a live generate_image save for later step/sample materialization."""
    entry = {
        "prompt": (prompt or "").strip(),
        "paths": [str(p) for p in paths],
        "backend": backend,
        "tool_stubbed": bool(tool_stubbed),
        "claimed": False,
        # Thread id helps judge_image resolve the right PNG when several
        # same-prompt rollouts interleave in one process (overfit GRPO).
        "thread_id": threading.get_ident(),
    }
    with _artifact_registry_lock:
        _artifact_registry.append(entry)


def claim_tool_artifacts_for_prompts(prompts: list[str]) -> list[dict]:
    """FIFO-claim unclaimed registry rows matching each prompt (exact, then stripped).

    Returns one dict per successfully claimed prompt (may be shorter than ``prompts``).
    """
    claimed: list[dict] = []
    with _artifact_registry_lock:
        for prompt in prompts:
            want = (prompt or "").strip()
            if not want:
                continue
            hit = None
            for entry in _artifact_registry:
                if entry.get("claimed"):
                    continue
                got = (entry.get("prompt") or "").strip()
                if got == want:
                    hit = entry
                    break
            if hit is None:
                # Soft fallback: allow trailing punctuation / whitespace drift.
                for entry in _artifact_registry:
                    if entry.get("claimed"):
                        continue
                    got = (entry.get("prompt") or "").strip().rstrip(".,; ")
                    if got == want.rstrip(".,; "):
                        hit = entry
                        break
            if hit is None:
                continue
            hit["claimed"] = True
            claimed.append(dict(hit))
    return claimed


def clear_tool_artifact_registry() -> None:
    """Test helper."""
    with _artifact_registry_lock:
        _artifact_registry.clear()
    set_latest_tool_image_path(None)
    _latest_tool_image_tls.path = None
    clear_good_enough_yes_reached()


_latest_tool_image_path: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_latest_tool_image_path", default=None
)
# Stock ToolAgentLoop often drops ContextVars across tool calls; thread-local
# survives same-thread generate → judge. Prefer resolve_tool_image_path().
_latest_tool_image_tls = threading.local()

# After judge_image returns good_enough=YES, further generate_image is blocked
# (env hard-stop — not token force). Thread-local + ContextVar like image path.
_good_enough_yes_reached: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agentic_good_enough_yes_reached", default=False
)
_good_enough_yes_tls = threading.local()


def set_latest_tool_image_path(path: str | None) -> contextvars.Token:
    """Remember the most recent generate_image PNG for judge_image."""
    _latest_tool_image_tls.path = path
    return _latest_tool_image_path.set(path)


def get_latest_tool_image_path() -> str | None:
    path = _latest_tool_image_path.get()
    if path:
        return path
    return getattr(_latest_tool_image_tls, "path", None)


def set_good_enough_yes_reached(reached: bool) -> contextvars.Token:
    """Mark that a live judge returned good_enough=YES on this rollout thread."""
    _good_enough_yes_tls.reached = bool(reached)
    return _good_enough_yes_reached.set(bool(reached))


def get_good_enough_yes_reached() -> bool:
    if _good_enough_yes_reached.get():
        return True
    return bool(getattr(_good_enough_yes_tls, "reached", False))


def clear_good_enough_yes_reached() -> None:
    """Reset YES latch (call when a new trajectory starts)."""
    _good_enough_yes_tls.reached = False
    _good_enough_yes_reached.set(False)


def _first_existing_png(paths: list[str] | None) -> str | None:
    for path in paths or []:
        text = str(path)
        if text.endswith(".png") and Path(text).is_file():
            return text
    return None


def _normalize_prompt(text: str | None) -> str:
    """Collapse whitespace/punctuation drift between generate and judge args."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _prompts_match(a: str | None, b: str | None) -> bool:
    na, nb = _normalize_prompt(a), _normalize_prompt(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Agent often lightly rewrites commas/wording when echoing image_prompt.
    return na[:96] == nb[:96] or na in nb or nb in na


def resolve_tool_image_path(*, image_prompt: str | None = None) -> str | None:
    """Resolve the PNG that ``judge_image`` should score.

    Order:
      1. ContextVar / thread-local set by the last ``generate_image`` on this thread
      2. Artifact registry: same thread + fuzzy-matching ``image_prompt``
      3. Artifact registry: same thread (most recent) — covers prompt wording drift
      4. Artifact registry: fuzzy-matching ``image_prompt`` (most recent, any thread)
    """
    direct = get_latest_tool_image_path()
    if direct and Path(direct).is_file():
        return direct

    want = image_prompt or ""
    tid = threading.get_ident()
    prompt_any_thread: str | None = None
    same_thread_any: str | None = None
    with _artifact_registry_lock:
        for entry in reversed(_artifact_registry):
            png = _first_existing_png(entry.get("paths"))
            if not png:
                continue
            same_thread = entry.get("thread_id") == tid
            same_prompt = _prompts_match(entry.get("prompt"), want)
            if same_thread and same_prompt:
                return png
            if same_thread and same_thread_any is None:
                same_thread_any = png
            if same_prompt and prompt_any_thread is None:
                prompt_any_thread = png
    # Prefer same-thread recent over cross-thread prompt match (overfit concurrency).
    if same_thread_any:
        return same_thread_any
    return prompt_any_thread


# Back-compat aliases used by earlier smoke tests.
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
