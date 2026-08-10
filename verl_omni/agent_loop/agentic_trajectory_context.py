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

Artifact layout (under the e2e run dir)::

    rollout_trajectories/step_{S:06d}/sample_{index}.{rollout_n:02d}.json
    rollout_images/step_{S:06d}/sample_{index}.{rollout_n:02d}/
        image_00_<artifact_id>.png ...
        meta.json

``artifact_id`` is ``sha256(relpath\\0index\\0prompt)[:12]`` — identity of the
generate call, not pixel content (overfit often reuses identical PNGs).

Judge lookup is **rollout-scoped** only. Cross-thread / cross-step fuzzy prompt
matching is intentionally removed: concurrent GRPO + identical overfit prompts
previously caused ``judge_image`` to score another rollout's (even previous
step's) PNG, corrupting live C/A rewards.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import re
import threading
from pathlib import Path

# Relative path under the images/trajectories roots.
_active_trajectory_relpath: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_trajectory_relpath", default=None
)
# Stable short id derived from trajectory_relpath (copied into asyncio.to_thread).
_active_rollout_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_rollout_id", default=None
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

# Live tool saves register here for judge lookup + optional materialization.
_artifact_registry_lock = threading.Lock()
_artifact_registry: list[dict] = []
# Direct artifact_id → png path (survives thread hops better than prompt match).
_artifact_by_id: dict[str, str] = {}
# Last PNG for each rollout_id (replaces bare thread-local latest-path).
_latest_image_by_rollout: dict[str, str] = {}


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


def set_active_trajectory_relpath(relpath: str | None) -> contextvars.Token:
    """Bind (or clear) the relative artifact path for subsequent tool saves."""
    rid = rollout_id_from_relpath(relpath)
    _active_rollout_id.set(rid)
    return _active_trajectory_relpath.set(relpath)


def reset_active_trajectory_relpath(token: contextvars.Token) -> None:
    """Restore the trajectory path binding that preceded ``token``."""
    _active_trajectory_relpath.reset(token)


def get_active_trajectory_relpath() -> str | None:
    return _active_trajectory_relpath.get()


def get_active_rollout_id() -> str | None:
    rid = _active_rollout_id.get()
    if rid:
        return rid
    return rollout_id_from_relpath(get_active_trajectory_relpath())


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
    artifact_id: str | None = None,
    trajectory_relpath: str | None = None,
    rollout_id: str | None = None,
) -> None:
    """Record a live generate_image save for judge lookup (rollout-scoped)."""
    relpath = trajectory_relpath or get_active_trajectory_relpath()
    rid = rollout_id or rollout_id_from_relpath(relpath) or get_active_rollout_id()
    png = _first_existing_png(paths)
    aid = (artifact_id or "").strip() or None
    if not aid and png:
        # Recover id from ``image_00_<hash>.png`` when callers omit it.
        m = re.search(r"image_\d+_([0-9a-f]{12})\.png$", Path(png).name, re.IGNORECASE)
        if m:
            aid = m.group(1)
    entry = {
        "prompt": (prompt or "").strip(),
        "paths": [str(p) for p in paths],
        "backend": backend,
        "tool_stubbed": bool(tool_stubbed),
        "claimed": False,
        "thread_id": threading.get_ident(),
        "trajectory_relpath": relpath,
        "rollout_id": rid,
        "artifact_id": aid,
    }
    with _artifact_registry_lock:
        _artifact_registry.append(entry)
        if aid and png:
            _artifact_by_id[aid] = png
        if rid and png:
            _latest_image_by_rollout[rid] = png


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
        _artifact_by_id.clear()
        _latest_image_by_rollout.clear()
    set_latest_tool_image_path(None)
    _latest_tool_image_tls.path = None
    clear_good_enough_yes_reached()


def count_live_generate_artifacts_for_active_rollout() -> int:
    """Count successful live ``generate_image`` PNGs for the active rollout.

    Used by ``AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES`` so the 4th+ generate is
    refused even when force-reflection is off for RL.
    """
    rid = get_active_rollout_id()
    n = 0
    with _artifact_registry_lock:
        for entry in _artifact_registry:
            if rid and entry.get("rollout_id") != rid:
                continue
            if entry.get("tool_stubbed"):
                continue
            if str(entry.get("backend") or "").lower() == "fewshot":
                continue
            if not _first_existing_png(entry.get("paths")):
                continue
            n += 1
    return n


def get_latest_generate_prompt_for_active_rollout() -> str | None:
    """Full diffusion prompt from the latest live artifact on this rollout."""
    rid = get_active_rollout_id()
    with _artifact_registry_lock:
        for entry in reversed(_artifact_registry):
            if rid and entry.get("rollout_id") != rid:
                continue
            prompt = str(entry.get("prompt") or "").strip()
            if prompt:
                return prompt
    return None


_latest_tool_image_path: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_latest_tool_image_path", default=None
)
# Kept only as a same-thread fallback for unit tests / smoke without rollout_id.
_latest_tool_image_tls = threading.local()

# After judge_image returns good_enough=YES, further generate_image is blocked
# (env hard-stop — not token force). Prefer asyncio-task scope so concurrent
# AgentLoopWorker gather() rollouts on one event-loop thread do not share a
# thread-local latch (that leak blocked the next sample's first generate).
_good_enough_yes_reached: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agentic_good_enough_yes_reached", default=False
)
_good_enough_yes_tls = threading.local()
_good_enough_yes_lock = threading.Lock()
_good_enough_yes_by_scope: dict[object, bool] = {}


def _rollout_scope_key() -> object:
    """Stable key for the active agent rollout (asyncio task, else thread)."""
    try:
        import asyncio

        task = asyncio.current_task()
        if task is not None:
            return ("task", id(task))
    except RuntimeError:
        pass
    return ("thread", threading.get_ident())


def set_latest_tool_image_path(path: str | None) -> contextvars.Token:
    """Remember the most recent generate_image PNG for judge_image (this rollout)."""
    rid = get_active_rollout_id()
    if rid:
        with _artifact_registry_lock:
            if path:
                _latest_image_by_rollout[rid] = str(path)
            else:
                _latest_image_by_rollout.pop(rid, None)
    _latest_tool_image_tls.path = path
    return _latest_tool_image_path.set(path)


def get_latest_tool_image_path() -> str | None:
    rid = get_active_rollout_id()
    if rid:
        with _artifact_registry_lock:
            scoped = _latest_image_by_rollout.get(rid)
        if scoped and Path(scoped).is_file():
            return scoped
    path = _latest_tool_image_path.get()
    if path and Path(path).is_file():
        return path
    tls = getattr(_latest_tool_image_tls, "path", None)
    if tls and Path(tls).is_file():
        return tls
    return None


def clear_latest_tool_image_for_active_rollout() -> None:
    """Drop the latest-image pointer for the active rollout_id."""
    set_latest_tool_image_path(None)


def set_good_enough_yes_reached(reached: bool) -> contextvars.Token:
    """Mark that a live judge returned good_enough=YES on this rollout scope."""
    flag = bool(reached)
    key = _rollout_scope_key()
    with _good_enough_yes_lock:
        if flag:
            _good_enough_yes_by_scope[key] = True
        else:
            _good_enough_yes_by_scope.pop(key, None)
    _good_enough_yes_tls.reached = flag
    return _good_enough_yes_reached.set(flag)


def get_good_enough_yes_reached() -> bool:
    if _good_enough_yes_reached.get():
        return True
    key = _rollout_scope_key()
    with _good_enough_yes_lock:
        if _good_enough_yes_by_scope.get(key, False):
            return True
    # Sync unit-test / no-running-task fallback only.
    if isinstance(key, tuple) and key[0] == "thread":
        return bool(getattr(_good_enough_yes_tls, "reached", False))
    return False


def clear_good_enough_yes_reached() -> None:
    """Reset YES latch for the current rollout scope (call at trajectory start/end)."""
    key = _rollout_scope_key()
    with _good_enough_yes_lock:
        _good_enough_yes_by_scope.pop(key, None)
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


def resolve_tool_image_path(
    *,
    image_prompt: str | None = None,
    artifact_id: str | None = None,
) -> str | None:
    """Resolve the PNG that ``judge_image`` should score.

    Strictly scoped to the **active rollout**. Order:
      1. Explicit ``artifact_id`` (from tool args / prior obs) if registered
      2. Latest PNG registered for this ``rollout_id``
      3. Same-rollout registry row with fuzzy-matching ``image_prompt``
      4. Same-rollout registry row (most recent)

    Never falls back to another rollout/thread/step via prompt match alone —
    that path corrupted live C/A rewards under concurrent overfit GRPO.
    """
    aid = (artifact_id or "").strip()
    if aid:
        with _artifact_registry_lock:
            png = _artifact_by_id.get(aid)
        if png and Path(png).is_file():
            return png

    rid = get_active_rollout_id()
    direct = get_latest_tool_image_path()
    if direct and Path(direct).is_file():
        # If we have a rollout_id, only accept the direct hit when it belongs
        # to this rollout (path contains the active trajectory folder).
        relpath = get_active_trajectory_relpath()
        if not rid or not relpath or relpath in direct.replace("\\", "/"):
            return direct

    if not rid:
        # Smoke / unbound tools: keep same-thread latest only (no cross-prompt).
        return direct if direct and Path(direct).is_file() else None

    want = image_prompt or ""
    prompt_hit: str | None = None
    latest_hit: str | None = None
    with _artifact_registry_lock:
        for entry in reversed(_artifact_registry):
            if entry.get("rollout_id") != rid:
                continue
            png = _first_existing_png(entry.get("paths"))
            if not png:
                continue
            if latest_hit is None:
                latest_hit = png
            if want and _prompts_match(entry.get("prompt"), want):
                prompt_hit = png
                break
    if prompt_hit:
        return prompt_hit
    return latest_hit


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
