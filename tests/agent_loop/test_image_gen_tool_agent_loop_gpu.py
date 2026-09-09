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

"""L2 GPU smoke: OmniAgentLoopManager + ImageGenToolAgentLoop.

Exercises Mode (2a) agent-loop wiring on a tiny AR checkpoint:
  - ``default_agent_loop=image_gen_tool_agent``
  - teacher-forced first ``generate_image`` (force-first curriculum)
  - local HTTP fake for diffusion tool pixels (no real DiT / VL sidecar)
  - rollout validity stamps + discard path in ``generate_sequences``

This is intentionally narrower than a full PPO recipe ([4/N]).
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from omegaconf import DictConfig, open_dict
from PIL import Image
from verl.protocol import DataProto
from verl.workers.rollout.llm_server import LLMServerManager

from tests.special_e2e.build_qwen3_omni_tiny_random import ensure_tiny_qwen3_omni_checkpoint
from tests.utils.gpu_test_topology import resolve_requested_num_gpus
from verl_omni.agent_loop.omni_agent_loop import OmniAgentLoopManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for L2 agent-loop smoke")


def _png_b64(color=(12, 34, 56)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _FakeDiffusionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003 - BaseHTTPRequestHandler API
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length) if length else b""
        body = json.dumps(
            {
                "text": "fake diffusion tool image",
                "images_base64": [_png_b64()],
                "reward": 0.0,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeDiffusionServer:
    """Local HTTP stand-in for ``AGENTIC_DIFFUSION_TOOL_URL`` (returns a tiny PNG)."""

    def __init__(self):
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str = ""

    def __enter__(self) -> str:
        self._httpd = HTTPServer(("127.0.0.1", 0), _FakeDiffusionHandler)
        host, port = self._httpd.server_address
        self.url = f"http://{host}:{port}/generate"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def __exit__(self, *exc) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


@pytest.fixture
def init_config(tmp_path_factory) -> tuple[DictConfig, Path]:
    from hydra import compose, initialize_config_dir

    requested_gpus = resolve_requested_num_gpus(default_num_gpus=2)
    if requested_gpus < 1:
        pytest.skip("No CUDA devices visible")
    tp_size = min(2, requested_gpus)

    model_path = ensure_tiny_qwen3_omni_checkpoint(
        os.path.expanduser("~/models/tiny-random/Qwen3-Omni"),
        skip_if_exists=True,
    )
    run_root = Path(tmp_path_factory.mktemp("agentic_gpu_smoke"))

    with initialize_config_dir(config_dir=os.path.abspath("verl_omni/trainer/config")):
        config = compose(config_name="omni_trainer")

    with open_dict(config):
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.model.tokenizer_path = model_path
        config.actor_rollout_ref.model.trust_remote_code = True
        # Thinker-only strip mirrors GSPO smoke (talker/visual unused for tool text loop).
        config.actor_rollout_ref.model.exclude_modules = (
            ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"
        )

        config.actor_rollout_ref.rollout.name = "vllm_omni"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.n = 1
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.35
        config.actor_rollout_ref.rollout.max_num_seqs = 4
        config.actor_rollout_ref.rollout.load_format = "safetensors"
        config.actor_rollout_ref.rollout.enable_prefix_caching = False
        config.actor_rollout_ref.rollout.prompt_length = 256
        config.actor_rollout_ref.rollout.response_length = 256
        config.actor_rollout_ref.rollout.max_model_len = 512
        config.actor_rollout_ref.rollout.temperature = 0.8
        config.actor_rollout_ref.rollout.calculate_log_probs = True

        if "engine_kwargs" not in config.actor_rollout_ref.rollout:
            config.actor_rollout_ref.rollout.engine_kwargs = {}
        if "vllm_omni" not in config.actor_rollout_ref.rollout.engine_kwargs:
            config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni = {}
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode = "ar"
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name = "qwen3_omni_moe"

        config.actor_rollout_ref.rollout.multi_turn.enable = True
        config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 4
        config.actor_rollout_ref.rollout.multi_turn.max_user_turns = 2
        config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
        config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length = 512
        config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
        # Leave function_tool_path unset — OmniAgentLoopWorker fills it under the gate.

        config.actor_rollout_ref.rollout.agent.num_workers = 1
        config.actor_rollout_ref.rollout.agent.default_agent_loop = "image_gen_tool_agent"

        config.data.max_prompt_length = 256
        config.data.max_response_length = 256

        config.trainer.n_gpus_per_node = requested_gpus
        config.trainer.nnodes = 1
        config.trainer.logger = ["console"]
        config.trainer.project_name = "verl_omni_gpu_smoke"
        config.trainer.experiment_name = "image_gen_tool_agent"
        config.trainer.default_local_dir = str(run_root / "ckpt")

        # Reward manager is unused for this generate_sequences-only smoke, but keep naive
        # so accidental custom agentic_reward wiring does not hit VisualRewardManager.
        if hasattr(config, "reward") and hasattr(config.reward, "reward_manager"):
            config.reward.reward_manager.name = "naive"

    return config, run_root


def test_image_gen_tool_agent_generate_sequences_stamps_validity(init_config, monkeypatch):
    config, run_root = init_config
    with _FakeDiffusionServer() as tool_url:
        monkeypatch.setenv("AGENTIC_DIFFUSION_TOOL_URL", tool_url)
        monkeypatch.delenv("AGENTIC_VLLM_OMNI_URL", raising=False)
        monkeypatch.delenv("AGENTIC_QWEN_IMAGE_URL", raising=False)
        monkeypatch.delenv("AGENTIC_VLLM_URL", raising=False)
        monkeypatch.setenv("AGENTIC_FORCE_FIRST_GENERATE", "1")
        monkeypatch.setenv("AGENTIC_FORCE_FIRST_WARMUP_STEPS", "100")
        monkeypatch.setenv("AGENTIC_FORCE_FIRST_END_STEP", "200")
        monkeypatch.setenv("AGENTIC_FORCE_REFLECTION_AFTER_JUDGE", "0")
        monkeypatch.setenv("AGENTIC_E2E_ROOT", str(run_root))
        monkeypatch.setenv("AGENTIC_E2E_RUN_NAME", "image_gen_tool_gpu_smoke")

        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "INFO",
                    "AGENTIC_DIFFUSION_TOOL_URL": tool_url,
                    "AGENTIC_FORCE_FIRST_GENERATE": "1",
                    "AGENTIC_FORCE_FIRST_WARMUP_STEPS": "100",
                    "AGENTIC_FORCE_FIRST_END_STEP": "200",
                    "AGENTIC_FORCE_REFLECTION_AFTER_JUDGE": "0",
                    "AGENTIC_E2E_ROOT": str(run_root),
                    "AGENTIC_E2E_RUN_NAME": "image_gen_tool_gpu_smoke",
                }
            }
        )
        try:
            llm_server_manager = LLMServerManager.create(config=config)
            agent_loop_manager = OmniAgentLoopManager.create(
                config=config,
                llm_client=llm_server_manager.get_client(),
            )

            raw_prompts = [
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an image-generation agent. Call generate_image with a diffusion "
                            "prompt, then judge_image, then Done. or rewrite."
                        ),
                    },
                    {"role": "user", "content": "a bright red apple on a white table"},
                ]
            ]
            batch = DataProto(
                non_tensor_batch={
                    "raw_prompt": np.array(raw_prompts, dtype=object),
                    "data_source": np.array(["agentic_smoke"] * len(raw_prompts), dtype=object),
                    "reward_model": np.array([{"style": "rule", "ground_truth": ""}] * len(raw_prompts), dtype=object),
                },
            )
            batch.meta_info["global_steps"] = 0
            result = agent_loop_manager.generate_sequences(prompts=batch)

            assert len(result) == len(raw_prompts)
            assert "responses" in result.batch
            assert "response_mask" in result.batch

            ntb = result.non_tensor_batch
            assert "num_generate_image_prompts" in ntb, (
                f"expected rollout-time generate stamp; keys={sorted(ntb.keys())}"
            )
            assert "rollout_valid" in ntb
            n_gen = int(np.asarray(ntb["num_generate_image_prompts"]).reshape(-1)[0])
            valid = int(np.asarray(ntb["rollout_valid"]).reshape(-1)[0])
            # Force-first + fake HTTP PNG → live ok=1 generate observation.
            assert n_gen >= 1, f"expected >=1 successful generate, got {n_gen}"
            assert valid == 1

            metrics = result.meta_info.get("agentic_metrics") or {}
            assert any(k.startswith("agentic_rollout/") for k in metrics), metrics
            print("image_gen_tool_agent GPU smoke passed:", {"n_gen": n_gen, "valid": valid, "metrics": metrics})
        finally:
            ray.shutdown()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
