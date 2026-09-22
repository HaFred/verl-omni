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
"""CPU contract for the BAGEL FlowGRPO SDE window.

Two properties matter for image quality and they are both about *where* the SDE
noise lands, not how much of it there is:

1. The window must stay inside the real denoise schedule.  ``setup_bagel_sigmas``
   installs ``num_inference_steps`` sigma points but only ``num_inference_steps - 1``
   denoise steps, and ``sde_window_range`` is written against the former (the
   recipes use ``[0, 7]`` while ``num_inference_steps`` may be as low as 4).  A
   window past the last step applies noise and records log-probs on *no* step,
   which empties the GEN trajectory silently instead of failing.

2. The window must not cover the terminal step (``sigma == 0``).  A bounded
   window is self-correcting -- later deterministic steps undo the injected
   noise exactly -- but noise injected *on* the terminal step has no later step
   to undo it, so it is baked into the decoded image.  ``_BagelSchedulerAdapter``
   only gates noise when a window is set, so a ``None`` window also falls back to
   the inner scheduler's default ``noise_level=0.7`` on every step, terminal step
   included.
"""

import pytest
import torch

from verl_omni.pipelines.bagel_flow_grpo.common import setup_bagel_sigmas
from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import (
    _BagelSchedulerAdapter,
    _pick_sde_window,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

# The recipes' ``rollout.algo`` defaults.
_RECIPE_WINDOW_SIZE = 2
_RECIPE_WINDOW_RANGE = (0, 7)
_SEEDS = range(64)


def _denoise_steps(num_inference_steps: int) -> int:
    """Denoise-step count ``setup_bagel_sigmas`` actually installs."""
    scheduler = FlowMatchSDEDiscreteScheduler()
    setup_bagel_sigmas(scheduler, num_inference_steps)
    return len(scheduler.timesteps)


@pytest.mark.parametrize("num_inference_steps", [4, 10, 15, 50])
def test_window_never_covers_the_terminal_sigma_zero_step(num_inference_steps):
    steps = _denoise_steps(num_inference_steps)
    for seed in _SEEDS:
        window = _pick_sde_window(_RECIPE_WINDOW_SIZE, _RECIPE_WINDOW_RANGE, seed, steps)
        assert window is not None
        begin, end = window
        assert begin >= 0
        assert end <= steps - 1, f"window {window} covers the terminal step of {steps}"


@pytest.mark.parametrize("num_inference_steps", [4, 10, 15, 50])
def test_window_range_past_the_schedule_is_clamped_to_a_real_step(num_inference_steps):
    """A range written against ``num_inference_steps`` may exceed the step count."""
    steps = _denoise_steps(num_inference_steps)
    for seed in _SEEDS:
        window = _pick_sde_window(_RECIPE_WINDOW_SIZE, _RECIPE_WINDOW_RANGE, seed, steps)
        begin, end = window
        assert end > begin, "empty window would record no log-probs at all"
        assert end <= steps, f"window {window} addresses no step of {steps}"


def test_single_denoise_step_window_shrinks_to_that_step():
    """``setup_bagel_sigmas`` accepts ``num_steps=1`` for warmup; the one step is
    the terminal one, so the window cannot avoid it -- it must still be non-empty
    so log-probs are recorded at all."""
    assert {_pick_sde_window(_RECIPE_WINDOW_SIZE, _RECIPE_WINDOW_RANGE, s, 1) for s in _SEEDS} == {(0, 1)}


def test_clamp_is_a_no_op_when_the_range_already_fits():
    """``num_inference_steps=10`` must keep the pre-clamp draw, so existing
    trajectories and dumps stay comparable."""
    steps = _denoise_steps(10)
    for seed in _SEEDS:
        assert _pick_sde_window(
            _RECIPE_WINDOW_SIZE, _RECIPE_WINDOW_RANGE, seed, steps
        ) == _pick_sde_window(_RECIPE_WINDOW_SIZE, _RECIPE_WINDOW_RANGE, seed)


def test_clamp_reaches_the_short_schedule_of_the_one_gpu_branch():
    """``num_inference_steps=4`` leaves 3 denoise steps, so only ``begin=0``
    is safe -- unclamped, ``[0, 7]`` could draw ``(5, 7)``, i.e. no step at all."""
    assert {_pick_sde_window(_RECIPE_WINDOW_SIZE, _RECIPE_WINDOW_RANGE, s, 3) for s in _SEEDS} == {(0, 2)}


def test_disabled_window_returns_none():
    assert _pick_sde_window(None, _RECIPE_WINDOW_RANGE, 0, 9) is None
    assert _pick_sde_window(0, _RECIPE_WINDOW_RANGE, 0, 9) is None


def test_missing_range_defaults_to_the_first_steps():
    assert _pick_sde_window(_RECIPE_WINDOW_SIZE, None, 0, 9) == (0, _RECIPE_WINDOW_SIZE)


def _oracle_decode(num_inference_steps, noise_level, window, seed=0):
    """Decode a known latent with an exact velocity field.

    The oracle velocity ``v = (x - x0) / sigma`` is the exact flow-matching
    velocity for the linear path, so the deterministic (``noise_level=0``) steps
    map the sample back onto ``x0`` exactly.  Whatever error remains is therefore
    noise the schedule could not undo -- which is what a grainy rollout image is.
    """
    torch.manual_seed(seed)
    height = width = 8
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    x0 = torch.stack([(x / width), (y / height), ((x + y) / (height + width))]).float()
    x0[:, :3, :3] = 0.5  # genuinely flat patch, i.e. where noise shows up first
    epsilon = torch.randn_like(x0)

    inner = FlowMatchSDEDiscreteScheduler()
    adapter = _BagelSchedulerAdapter(inner)
    setup_bagel_sigmas(inner, num_inference_steps)
    timesteps = inner.timesteps

    sample = (1 - float(timesteps[0])) * x0 + float(timesteps[0]) * epsilon
    adapter.begin_forward(sde_window=window, noise_level=noise_level, return_logprobs=True)
    for index, timestep in enumerate(timesteps):
        sigma = max(float(timestep), 1e-6)
        output = adapter.step(
            model_output=(sample - x0) / sigma,
            sigma=timestep,
            sample=sample,
            dt=0.0,
            generator=torch.Generator().manual_seed(1234 + index),
            include_logprob_normalizer=False,
        )
        sample = output.prev_sample
    return (sample - x0).std().item()


@pytest.mark.parametrize("window", [(0, 2), (2, 4), (5, 7)])
def test_bounded_window_noise_is_fully_undone_by_later_steps(window):
    """Every window the clamp can draw for ``num_inference_steps=10``, i.e. all
    of ``[0, 7]`` at size 2."""
    assert _oracle_decode(10, 0.7, window) < 1e-4


def test_terminal_step_noise_survives_into_the_image():
    """``(0, 9)`` covers the terminal step (9 denoise steps -> last index 8 is
    inside ``[0, 9)``), and the residual is what the user sees as grain."""
    assert _oracle_decode(10, 0.7, (0, 9)) > 0.05


def test_none_window_falls_back_to_noise_on_every_step():
    """The landmine: no window means the caller's ``noise_level`` runs on every
    step, i.e. the legacy flow_grpo behaviour.  That is only acceptable because
    the caller asked for it -- see the next test."""
    assert _oracle_decode(10, 0.7, None) > 0.05


def test_none_window_still_honours_a_deterministic_noise_level():
    """``noise_level=0.0`` must mean ODE even without a window.

    Kicking the override out of ``begin_forward``'s window branch made
    ``noise_level=0.0`` fall through to the inner scheduler's own default
    (``noise_level=0.7``), so a caller asking for a deterministic decode got
    maximum SDE noise on every step instead.
    """
    assert _oracle_decode(10, 0.0, None) < 1e-4


class _RecordingScheduler:
    """Stands in for ``FlowMatchSDEDiscreteScheduler`` and records what the
    adapter forwards per step."""

    def __init__(self):
        self.calls = []
        self.step_index = None

    def step(self, model_output, timestep, sample, return_dict=True, **kwargs):
        self.calls.append((kwargs.get("noise_level"), kwargs.get("return_logprobs")))
        return (sample, None, sample, torch.tensor(0.0))


def _gating(sde_window, noise_level, steps):
    inner = _RecordingScheduler()
    adapter = _BagelSchedulerAdapter(inner)
    adapter.begin_forward(sde_window=sde_window, noise_level=noise_level, return_logprobs=True)
    for _ in range(steps):
        adapter.step(model_output=torch.zeros(2), sigma=torch.tensor(0.5), sample=torch.zeros(2), dt=0.0)
    return inner.calls


def test_gating_applies_noise_and_logprobs_only_inside_the_window():
    calls = _gating((2, 4), 0.7, 6)
    assert [noise for noise, _ in calls] == [0.0, 0.0, 0.7, 0.7, 0.0, 0.0]
    assert [logprobs for _, logprobs in calls] == [False, False, True, True, False, False]


def test_gating_never_asks_for_logprobs_without_noise():
    """``std_dev_t == 0`` makes the Gaussian log-prob ``0/0``, so a
    ``noise_level=0.0`` decode must not request them even though the caller set
    ``calculate_log_probs=True``."""
    assert _gating(None, 0.0, 3) == [(0.0, False)] * 3


def test_gating_without_a_window_uses_the_callers_noise_level():
    assert _gating(None, 0.7, 3) == [(0.7, True)] * 3
