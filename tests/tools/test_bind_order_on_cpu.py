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
    """Register a transparent package stub so submodule imports skip heavy ``__init__``.

    A bare stub leaks for the rest of the pytest session: any later test file asking
    for a name the stub lacks (``from verl_omni.tools.trajectory import ...``, or a
    ``monkeypatch.setattr`` on ``verl_omni.tools.trajectory.hydra_env``) then dies with
    ImportError or AttributeError, depending on collection order. ``_missing`` answers
    those from the real package without making the heavy import eager.
    """
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")

    def _missing(attr: str):
        # PEP 562 hook, invoked only for names this stub lacks, so anything a test file
        # registered here still wins. Resolution mirrors a real package in two steps,
        # only the second of which executes an ``__init__``:
        #   1. a submodule of that name — ``getattr(verl_omni, "tools")``, and the
        #      ``verl_omni.tools.trajectory.hydra_env`` that dotted-path patching needs;
        #   2. otherwise the real ``__init__.py``, lazily — a re-export such as
        #      ``from verl_omni.tools.trajectory import active_trajectory_relpath``.
        if attr.startswith("__"):
            raise AttributeError(attr)
        try:
            child = importlib.import_module(f"{pkg.__name__}.{attr}")
        except ImportError:
            pass
        else:
            pkg.__dict__[attr] = child  # real packages expose submodules as attributes
            return child
        if path.is_dir() and not pkg.__dict__.get("_real_init_loaded"):
            try:
                spec = importlib.util.spec_from_file_location(
                    pkg.__name__, path / "__init__.py", submodule_search_locations=[str(path)]
                )
                real = importlib.util.module_from_spec(spec)
                # Execute the real ``__init__`` under the package name — that is what
                # makes its relative ``from .x import y`` resolve — while this stub stays
                # the canonical ``sys.modules`` entry, so the hook above keeps working.
                real.__package__ = pkg.__name__
                spec.loader.exec_module(real)
            except Exception:  # optional/heavy deps absent: stay a stub
                pass
            else:
                pkg.__dict__["_real_init_loaded"] = True
                pkg.__dict__.update(
                    {k: v for k, v in vars(real).items() if k not in {"__getattr__", "__dict__"}}
                )
        try:
            return pkg.__dict__[attr]
        except KeyError:
            raise AttributeError(f"module {pkg.__name__!r} has no attribute {attr!r}") from None

    pkg.__getattr__ = _missing
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


def test_explicit_run_dir_is_used_verbatim_without_the_experiment_name():
    """``run_dir`` names the run dir itself; appending ``experiment_name`` would nest a
    run inside the folder the caller just asked for.

    The bagel recipes pin ``agentic_image_gen.run_dir=$RUN_DIR`` so the three artifact
    trees land beside ``.hydra/`` and ``main_omni.log`` instead of under an extra
    ``e2e/<experiment_name>/``. Measured 2026-09-22 on ``bagel_corl_20260922_002349``:
    dumps landed at ``<run>/e2e/bagel_corl_pr1/rollout_images/...``.
    """
    hydra_env.clear_agentic_image_gen()
    cfg = OmegaConf.merge(_default_config(), {"agentic_image_gen": {"run_dir": "/tmp/run_20260922"}})
    hydra_env.bind_agentic_image_gen(cfg)
    paths.bind_run_artifacts(cfg)

    run_dir = paths.resolve_run_dir()
    assert str(run_dir) == str(Path("/tmp/run_20260922").resolve())
    assert "bind-order" not in str(run_dir), "experiment_name must not be appended to an explicit run_dir"
    assert str(paths.resolve_rollout_images_root()) == str(Path("/tmp/run_20260922").resolve() / "rollout_images")
    hydra_env.clear_agentic_image_gen()


def test_absent_run_dir_keeps_the_experiment_name_namespacing():
    """The default path is unchanged, so every caller relying on
    ``<e2e_root>/<experiment_name>/`` (and its per-run isolation) still gets it."""
    hydra_env.clear_agentic_image_gen()
    cfg = _default_config()
    hydra_env.bind_agentic_image_gen(cfg)
    paths.bind_run_artifacts(cfg)

    run_dir = paths.resolve_run_dir()
    assert run_dir.name == "bind-order"
    assert run_dir.parent == paths.resolve_e2e_root()
    hydra_env.clear_agentic_image_gen()


def test_run_dir_wins_over_e2e_root():
    """Both knobs may be set; the run dir is the more specific one and takes precedence,
    while ``e2e_root`` keeps its own value for anything reading it directly."""
    hydra_env.clear_agentic_image_gen()
    cfg = OmegaConf.merge(
        _default_config(),
        {"agentic_image_gen": {"e2e_root": "/tmp/custom_e2e", "run_dir": "/tmp/explicit_run"}},
    )
    hydra_env.bind_agentic_image_gen(cfg)
    paths.bind_run_artifacts(cfg)

    assert str(paths.resolve_run_dir()) == str(Path("/tmp/explicit_run").resolve())
    assert str(paths._e2e_root) == str(Path("/tmp/custom_e2e").resolve())
    hydra_env.clear_agentic_image_gen()


def test_clear_run_artifacts_drops_the_run_dir_override():
    """A stale override must not survive into the next bind (same class of leak the
    ``_diffusion_image_dir`` reset guards against)."""
    hydra_env.clear_agentic_image_gen()
    cfg = OmegaConf.merge(_default_config(), {"agentic_image_gen": {"run_dir": "/tmp/stale_run"}})
    hydra_env.bind_agentic_image_gen(cfg)
    paths.bind_run_artifacts(cfg)
    assert str(paths.resolve_run_dir()) == str(Path("/tmp/stale_run").resolve())

    paths.clear_run_artifacts()
    assert paths._run_dir_override is None
    assert paths.resolve_run_dir() == paths.resolve_e2e_root() / "agentic_run"
    hydra_env.clear_agentic_image_gen()
