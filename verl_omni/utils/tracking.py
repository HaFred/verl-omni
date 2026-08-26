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

"""Experiment-tracking helpers layered on verl.utils.tracking."""

import json
import logging
import os
import re
import subprocess
import tempfile
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl_omni.utils.reward_score.reward_utils import video_tensor_to_pil_frames

logger = logging.getLogger(__name__)


class AgenticValidationGenerationsLogger:
    """Log cumulative agentic validation images using verl's copy-table pattern.

    The agent loop supplies prompts and materialized image paths. This class
    owns all tracking-specific state: restoring prior rows after resume,
    constructing W&B media/tables, and avoiding out-of-order W&B steps.
    """

    DEFAULT_SAMPLE_TABLE_KEYS = {
        "sample_9001": "val/generations",
        "sample_9002": "val/generations_plan",
    }

    def __init__(
        self,
        run_dir: str | Path,
        max_turns: int | None = None,
        sample_table_keys: dict[str, str] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.max_turns = max_turns if max_turns is not None else self._max_turns_from_env()
        self.sample_table_keys = dict(sample_table_keys or self.DEFAULT_SAMPLE_TABLE_KEYS)
        self.history = self._restore_history()
        self.tables: dict[str, Any] = {}

    @staticmethod
    def _max_turns_from_env() -> int:
        try:
            return max(1, int(os.getenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "3")))
        except ValueError:
            return 3

    @staticmethod
    def pair_turns(prompts: Sequence[str], image_paths: Sequence[str]) -> list[tuple[str, str | None]]:
        """Zip generated-image prompts and paths in call order."""
        n = max(len(prompts), len(image_paths))
        return [
            (
                prompts[index] if index < len(prompts) else "",
                image_paths[index] if index < len(image_paths) else None,
            )
            for index in range(n)
        ]

    @staticmethod
    def effective_wandb_step(step) -> int | None:
        """Return the trainer global step for mid-validate soft W&B logs.

        Always the exact ``global_steps`` (no tip bump). Holdout tables and
        ``val_agentic_reward/*`` are logged with ``commit=False`` *before* the
        trainer's ``Tracking.log`` of ``val-core`` at the same step. Bumping to
        ``run.step + 1`` here used to finalize tip ``N+1`` first, so the later
        ``val-core`` commit at ``N`` was dropped (``Tried to log to step N <
        current step N+1``). Table ``row[0]`` also stores this same integer.
        """
        if step is None:
            return None
        try:
            return int(step)
        except (TypeError, ValueError):
            return None

    def _restore_history(self) -> dict[str, list[list[Any]]]:
        """Recover cumulative validation rows from on-disk image metadata."""
        history: dict[str, list[list[Any]]] = {}
        images_root = self.run_dir / "rollout_images"
        for sample_name, table_key in self.sample_table_keys.items():
            rows: list[list[Any]] = []
            for meta_path in images_root.glob(f"step_*/{sample_name}/meta.json"):
                match = re.fullmatch(r"step_(\d+)", meta_path.parent.parent.name)
                if match is None:
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                prompts = list(meta.get("tool_prompts") or [])
                paths = list(meta.get("image_paths") or [])
                row: list[Any] = [int(match.group(1))]
                for index in range(self.max_turns):
                    row.append(str(prompts[index]) if index < len(prompts) else "")
                    row.append(str(paths[index]) if index < len(paths) else "")
                rows.append(row)
            if rows:
                by_step = {int(row[0]): row for row in rows}
                history[table_key] = [by_step[step] for step in sorted(by_step)]
        return history

    def _build_table(self, step, table_key: str, turn_pairs: Sequence[tuple[str, str | None]]) -> Any:
        """Append one row and return a fresh cumulative W&B table."""
        import wandb

        if wandb.run is None:
            return None
        columns = ["step"] + sum(
            [[f"input_{index + 1}", f"output_{index + 1}"] for index in range(self.max_turns)], []
        )
        try:
            step_i = int(step) if step is not None else -1
        except (TypeError, ValueError):
            step_i = -1

        history_row: list[Any] = [step_i]
        for index in range(self.max_turns):
            if index < len(turn_pairs):
                prompt, path = turn_pairs[index]
                history_row.extend([prompt or "", str(path) if path else ""])
            else:
                history_row.extend(["", ""])
        history = self.history.setdefault(table_key, [])
        history[:] = [row for row in history if row[0] != step_i]
        history.append(history_row)
        history.sort(key=lambda row: row[0])

        previous = self.tables.get(table_key)
        if previous is None or len(getattr(previous, "data", [])) < len(history) - 1:
            table = wandb.Table(columns=columns)
            for saved_row in history[:-1]:
                row: list[Any] = [saved_row[0]]
                for index in range(self.max_turns):
                    prompt = saved_row[1 + 2 * index]
                    path = saved_row[2 + 2 * index]
                    row.extend([prompt, wandb.Image(path) if path and Path(path).is_file() else ""])
                table.add_data(*row)
        else:
            prior = [list(row) for row in previous.data if not row or row[0] != step_i]
            table = wandb.Table(columns=columns, data=prior)

        new_row: list[Any] = [history_row[0]]
        for index in range(self.max_turns):
            prompt = history_row[1 + 2 * index]
            path = history_row[2 + 2 * index]
            new_row.extend([prompt, wandb.Image(path) if path and Path(path).is_file() else ""])
        table.add_data(*new_row)
        self.tables[table_key] = table
        return table

    def log(self, step, table_rows: dict[str, Sequence[tuple[str, str | None]]]) -> None:
        """Soft-log cumulative validation tables at the trainer global step.

        Uses ``commit=False`` so the tip stays available for the trainer's later
        ``Tracking.log`` of ``val-core`` / step metrics at the same ``step``.
        """
        import wandb

        if wandb.run is None:
            return
        payload = {
            table_key: table
            for table_key, turn_pairs in table_rows.items()
            if (table := self._build_table(step, table_key, turn_pairs)) is not None
        }
        if payload:
            wandb.log(payload, step=self.effective_wandb_step(step), commit=False)


def batch_items(values: Any, batch_size: int, name: str) -> list[Any]:
    """Normalize an optional scalar or batched value to one item per sample."""
    if values is None:
        return [None] * batch_size
    if isinstance(values, torch.Tensor | np.ndarray):
        if values.ndim == 0:
            return [values] * batch_size
        if values.shape[0] == batch_size:
            return list(values)
        if batch_size == 1:
            return [values]
        raise ValueError(f"{name} batch size {values.shape[0]} does not match output batch size {batch_size}.")
    if isinstance(values, Sequence) and not isinstance(values, str | bytes):
        if len(values) != batch_size:
            raise ValueError(f"{name} batch size {len(values)} does not match output batch size {batch_size}.")
        return list(values)
    return [values] * batch_size


def _write_wav(audio: Any, sample_rate: Any, path: Path) -> None:
    waveform = torch.as_tensor(audio).detach().cpu().float()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected audio shape [T] or [C, T], got {tuple(waveform.shape)}.")
    if waveform.shape[0] > 8 and waveform.shape[1] <= 8:
        waveform = waveform.transpose(0, 1)
    if waveform.shape[0] > 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    sample_rate = int(torch.as_tensor(sample_rate).item())
    if sample_rate <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {sample_rate}.")
    pcm = (torch.nan_to_num(waveform).clamp(-1, 1).transpose(0, 1).numpy() * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _export_video(
    output: torch.Tensor,
    output_path: str,
    *,
    fps: int,
    audio: Any = None,
    audio_sample_rate: Any = None,
    video_exporter: Callable[..., str] | None = None,
    ffmpeg_exe: str | None = None,
) -> None:
    if video_exporter is None:
        from diffusers.utils import export_to_video

        video_exporter = export_to_video

    frames = video_tensor_to_pil_frames(output)
    if audio is None:
        video_exporter(frames, output_path, fps=fps)
        return
    if audio_sample_rate is None:
        raise ValueError("audio_sample_rate is required when logging a video with audio.")
    if ffmpeg_exe is None:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_exe = get_ffmpeg_exe()

    output_path = Path(output_path)
    silent_path = output_path.with_suffix(".silent.mp4")
    audio_path = output_path.with_suffix(".wav")
    try:
        video_exporter(frames, str(silent_path), fps=fps)
        _write_wav(audio, audio_sample_rate, audio_path)
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_path),
            ],
            check=True,
        )
    finally:
        silent_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


def wrap_val_samples_for_wandb(samples, fps=24, output_dir=None):
    """Wrap validation samples and prepare top-level ``wandb`` video media.

    Video outputs ``[T, C, H, W]`` are encoded to mp4 and passed to
    ``wandb.Video`` by path. Provide ``output_dir`` to keep the media available
    for asynchronous upload; otherwise a temp dir is returned for cleanup.
    Optional tuple elements four and five carry audio and its sample rate. The
    table stores a stable media key because offline ``wandb`` tables do not
    reliably persist nested videos. Other outputs become ``wandb.Image``.
    """
    import wandb

    video_dir = output_dir
    video_tmp_dir = None
    wrapped = []
    media_to_log = {}
    for sample in samples:
        inp, out, score = sample[:3]
        audio = sample[3] if len(sample) > 3 else None
        audio_sample_rate = sample[4] if len(sample) > 4 else None
        if hasattr(out, "ndim") and out.ndim == 4:
            if video_dir is None:
                video_tmp_dir = tempfile.mkdtemp(prefix="val_video_")
                video_dir = video_tmp_dir
            else:
                os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{len(wrapped)}.mp4")
            _export_video(out, video_path, fps=fps, audio=audio, audio_sample_rate=audio_sample_rate)
            media_key = f"val/videos/sample_{len(wrapped) + 1}"
            media_to_log[media_key] = wandb.Video(video_path, format="mp4")
            media = media_key
        else:
            if not isinstance(out, torch.Tensor) or out.dtype != torch.uint8:
                raise ValueError(f"Expected a uint8 image tensor, got {getattr(out, 'dtype', type(out))}.")
            media = wandb.Image(out, file_type="jpg", normalize=False)
        wrapped.append((inp, media, score))
    return wrapped, video_tmp_dir, media_to_log


def log_wandb_media(media: dict[str, Any], step: int) -> None:
    """Buffer top-level ``wandb`` media for the validation table log at ``step``."""
    if not media:
        return

    import wandb

    if wandb.run is not None:
        wandb.log(media, step=step, commit=False)
