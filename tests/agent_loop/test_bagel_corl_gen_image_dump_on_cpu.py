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
"""CPU regression: Bagel Co-RL (Joint-Training) GEN PNGs must land beside their trajectories.

Measured 2026-09-21 on hk01dgx039: ``outputs/e2e/bagel_corl_pr1/`` held
``rollout_trajectories/`` but **neither** ``rollout_images/`` nor ``hermes_actions/``,
while 42 PNGs sat in ``/tmp/bagel_corl_gen/`` and every ``gen_call`` trajectory row
pointed at that scratch dir (``image_paths: ["/tmp/bagel_corl_gen/gen_*.png"]``).

``dump_bagel_corl_episode_images`` resolved its episodes from a top-level
``gen_samples`` batch column or from the ``extra_fields`` *attribute*. ``bagel_corl``
emits neither shape: it publishes one ``extra_fields`` *row* per episode, each nesting
its own ``gen_samples`` list. Both lookups missed, ``episodes`` stayed empty and the
function fell through to ``return []`` **without raising** -- no log line, no
``rollout_images/`` directory, and a silent no-op in production.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

from verl_omni.tools.trajectory import paths as traj_paths
from verl_omni.utils.agentic.image_gen_rollout_dump import dump_bagel_corl_episode_images


def _seed_png(dirpath: Path, name: str) -> str:
    """A real file on disk (the dumper copies bytes; it does not decode them)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    return str(path)


def _sample(image_path: str):
    """Stand-in for ``GenSample``: the dumper only reads ``image_path``."""
    return types.SimpleNamespace(image_path=image_path)


def _output(rows):
    """Driver-side composite output -- ``extra_fields`` is a batch *column*, not an attribute."""
    return types.SimpleNamespace(non_tensor_batch={"extra_fields": rows}, meta_info={})


def _bind_root(monkeypatch, tmp_path: Path) -> Path:
    """Point ``resolve_rollout_images_root()`` at the test tree."""
    root = tmp_path / "rollout_images"
    monkeypatch.setattr(traj_paths, "_diffusion_image_dir", root)
    return root


def test_extra_fields_rows_are_materialized_beside_their_trajectory(tmp_path, monkeypatch):
    """The production shape: one ``extra_fields`` row per episode, samples nested inside."""
    root = _bind_root(monkeypatch, tmp_path)
    scratch = tmp_path / "scratch"
    rows = [
        {
            "gen_samples": [
                _sample(_seed_png(scratch, f"a{i}.png")),
                _sample(_seed_png(scratch, f"b{i}.png")),
            ],
            "trajectory_relpath": f"step_000001/sample_{i}.01",
        }
        for i in (0, 1)
    ]

    written = dump_bagel_corl_episode_images(_output(rows), step=1)

    assert len(written) == 4, "both seeds of both episodes must be copied"
    for i in (0, 1):
        target = root / "step_000001" / f"sample_{i}.01"
        assert sorted(p.name for p in target.glob("*.png")) == [f"a{i}.png", f"b{i}.png"]
        # The folder must sit beside the trajectory it belongs to -- not at a
        # synthesised ``sample_0.00`` that matches no trajectory on disk.
        meta = json.loads((target / "meta.json").read_text())
        assert meta["trajectory_relpath"] == f"step_000001/sample_{i}.01"
        # ``written`` accumulates across episodes, so meta.json has to be scoped
        # per-episode or every folder lists every earlier episode's images.
        assert sorted(Path(p).name for p in meta["image_paths"]) == [f"a{i}.png", f"b{i}.png"]


def test_top_level_gen_samples_column_still_materializes(tmp_path, monkeypatch):
    """Backward compatibility: the pre-existing hoisted-column shape must keep working."""
    root = _bind_root(monkeypatch, tmp_path)
    scratch = tmp_path / "scratch"
    samples = [_sample(_seed_png(scratch, "only.png"))]
    output = types.SimpleNamespace(non_tensor_batch={"gen_samples": samples}, meta_info={})

    written = dump_bagel_corl_episode_images(output, step=7, sample_index=3, rollout_n=0)

    assert len(written) == 1
    assert (root / "step_000007" / "sample_3.00" / "only.png").is_file()


def test_row_without_samples_creates_no_directory(tmp_path, monkeypatch):
    """A K=0 episode writes no GEN images, so it must not manufacture an empty folder."""
    root = _bind_root(monkeypatch, tmp_path)
    rows = [{"gen_samples": [], "trajectory_relpath": "step_000002/sample_5.01"}]

    written = dump_bagel_corl_episode_images(_output(rows), step=2)

    assert written == []
    assert not (root / "step_000002" / "sample_5.01").exists()


def test_missing_source_files_create_no_directory(tmp_path, monkeypatch):
    """Sample paths that no longer exist must not leave an empty folder behind.

    The GEN tool returns a scratch path and the copy happens later, so a file can be
    gone (scratch reaped) or never written (stub row). Creating the folder regardless
    would litter ``rollout_images/`` with one empty dir per rollout.
    """
    root = _bind_root(monkeypatch, tmp_path)
    rows = [
        {
            "gen_samples": [_sample(str(tmp_path / "gone" / "missing.png"))],
            "trajectory_relpath": "step_000003/sample_2.01",
        }
    ]

    written = dump_bagel_corl_episode_images(_output(rows), step=3)

    assert written == []
    assert not (root / "step_000003" / "sample_2.01").exists()
