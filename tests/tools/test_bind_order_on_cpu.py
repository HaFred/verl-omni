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
"""CPU tests for the agentic bind order (torch-free).

Regression for the audit blocker: ``bind_run_artifacts`` used to read
``agentic_get("e2e_root")`` while ``bind_agentic_image_gen`` had not run yet,
crashing worker/manager init with the default yaml (``e2e_root: null``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")
    sys.modules[name] = pkg


def _load_by_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_modules():
    root = Path(__file__).resolve().parents[2]
    omni = root / "verl_omni"
    _ensure_pkg("verl_omni", omni)
    _ensure_pkg("verl_omni.tools", omni / "tools")
    _ensure_pkg("verl_omni.tools.trajectory", omni / "tools" / "trajectory")
    hydra_env = _load_by_path(
        "verl_omni.tools.trajectory.hydra_env", omni / "tools" / "trajectory" / "hydra_env.py"
    )
    paths = _load_by_path("verl_omni.tools.trajectory.paths", omni / "tools" / "trajectory" / "paths.py")
    return hydra_env, paths


hydra_env, paths = _load_modules()


def _default_config():
    return OmegaConf.create(
        {
            "trainer": {"experiment_name": "bind-order"},
            "agentic_image_gen": {"e2e_root": None},
        }
    )


def test_bind_run_artifacts_before_agentic_bind_does_not_crash():
    """Production order regression: run-artifacts first must fall back to the
    documented default root instead of raising while unbound."""
    hydra_env.clear_agentic_image_gen()  # simulate a fresh worker process
    cfg = _default_config()
    paths.bind_run_artifacts(cfg)  # used to raise RuntimeError here
    assert paths.run_name == "bind-order"
    assert paths._e2e_root is not None and paths._e2e_root.name == "e2e"

    # Binding afterwards still works and keeps the default root.
    hydra_env.bind_agentic_image_gen(cfg)
    assert hydra_env.agentic_get("e2e_root") is None
    paths.bind_run_artifacts(cfg)
    assert paths._e2e_root is not None


def test_explicit_e2e_root_wins_when_bound():
    hydra_env.clear_agentic_image_gen()
    cfg = _default_config()
    merged = OmegaConf.merge(cfg, {"agentic_image_gen": {"e2e_root": "/tmp/custom_e2e"}})
    hydra_env.bind_agentic_image_gen(merged)
    paths.bind_run_artifacts(merged)
    assert str(paths._e2e_root) == str(Path("/tmp/custom_e2e").expanduser().resolve())
    hydra_env.clear_agentic_image_gen()


def test_bind_none_fails_loud():
    with pytest.raises(ValueError, match="silently serve yaml"):
        hydra_env.bind_agentic_image_gen(None)
    hydra_env.clear_agentic_image_gen()
