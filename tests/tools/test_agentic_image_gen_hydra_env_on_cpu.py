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

"""CPU tests for Hydra ``agentic_image_gen`` → ``AGENTIC_*`` env bind.

Loads ``hydra_env`` via importlib so collection does not run
``verl_omni/__init__.py`` (pipelines → heavy deps).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_VERL_OMNI = Path(__file__).resolve().parents[2] / "verl_omni"
_HYDRA_ENV_PATH = _VERL_OMNI / "tools" / "trajectory" / "hydra_env.py"
_OMNI_CONFIG_DIR = str(_VERL_OMNI / "trainer" / "config")


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _load_hydra_env():
    _ensure_package("verl_omni", _VERL_OMNI)
    _ensure_package("verl_omni.tools", _VERL_OMNI / "tools")
    _ensure_package("verl_omni.tools.trajectory", _VERL_OMNI / "tools" / "trajectory")
    spec = importlib.util.spec_from_file_location(
        "verl_omni.tools.trajectory.hydra_env",
        _HYDRA_ENV_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_hydra_env = _load_hydra_env()
bind_agentic_image_gen_env = _hydra_env.bind_agentic_image_gen_env


def test_omni_trainer_composes_agentic_image_gen():
    with initialize_config_dir(config_dir=_OMNI_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="omni_trainer")
    assert "agentic_image_gen" in cfg
    assert cfg.agentic_image_gen.max_generate_image_passes == 3
    assert cfg.agentic_image_gen.block_generate_after_yes is True


def test_bind_agentic_image_gen_env_sets_diffusion_url(monkeypatch):
    monkeypatch.delenv("AGENTIC_DIFFUSION_TOOL_URL", raising=False)
    monkeypatch.delenv("AGENTIC_BLOCK_GENERATE_AFTER_YES", raising=False)
    monkeypatch.delenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", raising=False)

    cfg = OmegaConf.create(
        {
            "agentic_image_gen": {
                "diffusion_tool_url": "http://127.0.0.1:9999/generate",
                "diffusion_tool_token": None,
                "block_generate_after_yes": False,
                "max_generate_image_passes": 5,
            }
        }
    )
    bind_agentic_image_gen_env(cfg)

    assert os.environ["AGENTIC_DIFFUSION_TOOL_URL"] == "http://127.0.0.1:9999/generate"
    assert os.environ["AGENTIC_BLOCK_GENERATE_AFTER_YES"] == "0"
    assert os.environ["AGENTIC_MAX_GENERATE_IMAGE_PASSES"] == "5"
    assert "AGENTIC_DIFFUSION_TOOL_TOKEN" not in os.environ


def test_bind_skips_missing_agentic_image_gen(monkeypatch):
    monkeypatch.setenv("AGENTIC_DIFFUSION_TOOL_URL", "keep-me")
    bind_agentic_image_gen_env(OmegaConf.create({"trainer": {}}))
    assert os.environ["AGENTIC_DIFFUSION_TOOL_URL"] == "keep-me"
