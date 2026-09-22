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
"""CPU tests for Bagel Co-RL (Joint-Training) IDs, flatten, GEN cap, and serial episode."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


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


def _load_lib():
    root = Path(__file__).resolve().parents[2]
    omni = root / "verl_omni"
    _ensure_pkg("verl_omni", omni)
    _ensure_pkg("verl_omni.agent_loop", omni / "agent_loop")
    # Bagel lib now shares the tools.trajectory layer (audit T1.6).
    _ensure_pkg("verl_omni.tools", omni / "tools")
    _ensure_pkg("verl_omni.tools.trajectory", omni / "tools" / "trajectory")
    _load_by_path("verl_omni.tools.trajectory.hydra_env", omni / "tools" / "trajectory" / "hydra_env.py")
    _load_by_path("verl_omni.tools.trajectory.paths", omni / "tools" / "trajectory" / "paths.py")
    _load_by_path("verl_omni.tools.trajectory.context", omni / "tools" / "trajectory" / "context.py")
    _load_by_path("verl_omni.tools.trajectory.artifacts", omni / "tools" / "trajectory" / "artifacts.py")
    _load_by_path("verl_omni.tools.trajectory.judge_latch", omni / "tools" / "trajectory" / "judge_latch.py")
    _load_by_path(
        "verl_omni.agent_loop.rpco_turn_protocol",
        omni / "agent_loop" / "rpco_turn_protocol.py",
    )
    return _load_by_path(
        "bagel_corl_lib_isolated",
        omni / "agent_loop" / "bagel_corl_lib.py",
    )


lib = _load_lib()
ctx = sys.modules["verl_omni.tools.trajectory.judge_latch"]


class _FakeTokenizer:
    """Minimal encode/decode for observation / reflection mask tests."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) + 1000 for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(chr(int(t) - 1000) for t in token_ids)


def test_hermes_generate_image_is_bagel_not_qwen():
    text = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "a cat"}}</tool_call>'
    assert lib.und_turn_kind(text) == "generate_image"
    assert lib.parse_hermes_tool_call(text)["name"] == "generate_image"
    assert lib.GENERATE_IMAGE_TOOL_SCHEMA["function"]["name"] == "generate_image"


def test_unsupported_tool_is_fail_closed():
    text = '<tool_call>{"name": "judge_image", "arguments": {}}</tool_call>'
    with pytest.raises(ValueError, match="unsupported tool"):
        lib.und_turn_kind(text)


def test_bare_payload_tool_call_is_accepted():
    """The published checkpoint samples the Hermes payload without the tags.

    Measured 2026-09-21 21:45 on hk01dgx039 with the ``<tools>`` block in the prompt:
    ``turn=1 ... text='{"name": "generate_image", "arguments": {"prompt": "Full body
    portrait of ...'`` and (one turn later) ``'ASSISTANT\\n {"name": "judge_image",
    ...}<|im_end|>'``. Neither carries ``<tool_call>``, so a tagged-only parse classified
    every turn as ``continue`` and the GEN lane never ran.
    """
    text = '{"name": "generate_image", "arguments": {"prompt": "a cat"}}'
    assert lib.und_turn_kind(text) == "generate_image"
    assert lib.parse_und_tool_call(text)["name"] == "generate_image"
    assert lib.parse_hermes_tool_call(text) is None  # tagged parse stays strict


def test_bare_payload_survives_role_echo_and_trailing_prose():
    text = 'ASSISTANT\n {"name": "generate_image", "arguments": {"prompt": "a {cat}"}}<|im_end|> and then it stops'
    call = lib.parse_und_tool_call(text)
    assert call is not None and call["arguments"]["prompt"] == "a {cat}"
    assert lib.und_turn_kind(text) == "generate_image"


def test_fenced_payload_tool_call_is_accepted():
    """Measured 2026-09-21 23:05 on hk01dgx039 (``run_bagel_diag.sh``, devices 3/5/6/7).

    The checkpoint fenced the same payload behind a label, so a tagged/bare-only parse saw
    ``continue`` on turn 1, the loop kept decoding as if the call were prose, and the model went
    on to role-play the *reply* (``'\\nuser\\nThanks for the feedback! ...'``) with K = 0.
    """
    text = (
        "Content Request: \n"
        '```\n{\n  "name": "generate_image",\n  "arguments": {\n'
        '    "prompt": "A full body portrait of a handsome bald male with a head tattoo"\n'
        "  }\n}\n```"
    )
    call = lib.parse_und_tool_call(text)
    assert call is not None and call["name"] == "generate_image"
    assert call["arguments"]["prompt"].startswith("A full body portrait")
    assert lib.und_turn_kind(text) == "generate_image"


def test_fenced_payload_with_language_tag_and_role_echo_is_accepted():
    text = 'assistant\nHere is the action:\n```json\n{"name": "generate_image", "arguments": {"prompt": "a cat"}}\n```'
    assert lib.und_turn_kind(text) == "generate_image"


def test_fenced_judge_image_is_inert():
    text = 'Content Request: \n```\n{"name": "judge_image", "arguments": {}}\n```'
    assert lib.parse_und_tool_call(text)["name"] == "judge_image"
    assert lib.und_turn_kind(text) == "continue"


def test_one_fenced_object_per_step_finds_the_call_in_a_later_fence():
    """Measured 2026-09-21 07:50 on hk01dgx039 (devices 2-5, ``bagel_corl_pr1`` step 1).

    The checkpoint numbered its steps and fenced one JSON object per step, so the turn's *first*
    fence is the plan (no ``name``) and the ``generate_image`` call is the second. Reading only
    the first block returned ``None``, ``und_turn_kind`` said ``continue`` on every turn and the
    GEN lane never ran (``gen/num_rows: 0``, ``gen/skipped_no_groups: 1``).
    """
    text = (
        "Sure, here's your request processed:\n"
        "\n"
        "1. Plan:\n"
        '```json\n{\n  "subtasks": [\n    {\n      "prompt": "A chaotic anime-style '
        'cartoon depiction of a destroyed school."\n    }\n  ]\n}\n```\n'
        "2. Generate Image:\n"
        '```json\n{\n  "name": "generate_image",\n  "arguments": {\n    "prompt": "A chaotic '
        'anime-style cartoon depiction of a destroyed school."\n  }\n}\n```\n'
        "3. Judge Image (Feedback will be provided after generating the image):\n"
        '```json\n{\n  "name": "judge_image"\n}\n```\n'
        "4. Reflection & Final Output:\nDone. The generated image aligns with the specified "
        "prompt.<|im_end|>"
    )
    call = lib.parse_und_tool_call(text)
    assert call is not None and call["name"] == "generate_image"
    assert call["arguments"]["prompt"].startswith("A chaotic anime-style cartoon")
    assert lib.und_turn_kind(text) == "generate_image"


def test_fenced_example_inside_a_prose_plan_is_not_a_tool_call():
    """The label bound is what keeps a *quoted* example from being executed.

    A call is the turn's whole content behind a short label; a plan is paragraphs of prose that
    happen to contain a fenced example.
    """
    text = (
        "1. Plan the scene carefully, thinking about the character, the lighting and the "
        "composition, and write the whole reasoning out before acting on anything.\n"
        '```\n{"name": "generate_image", "arguments": {"prompt": "x"}}\n```\n'
        "2. Only after that, emit the call."
    )
    assert lib.parse_und_tool_call(text) is None
    assert lib.und_turn_kind(text) == "continue"


def test_bare_judge_image_is_inert_but_tagged_is_fatal():
    """The recipe's prompt asks for a judge turn after the last image; the RM does the judging."""
    assert lib.und_turn_kind('ASSISTANT\n {"name": "judge_image", "arguments": {}}<|im_end|>') == "continue"
    with pytest.raises(ValueError, match="unsupported tool"):
        lib.und_turn_kind('<tool_call>{"name": "judge_image", "arguments": {}}</tool_call>')


def test_json_example_in_prose_is_not_a_tool_call():
    text = '1. Plan: emit {"name": "generate_image", "arguments": {"prompt": "x"}} for each subtask.'
    assert lib.parse_und_tool_call(text) is None
    assert lib.und_turn_kind(text) == "continue"


def test_tools_tagged_payload_behind_a_prose_plan_is_accepted():
    """Measured 2026-09-21 15:36 on hk01dgx039 (devices 2-5, ``bagel_corl_pr1`` step 1).

    The checkpoint re-uses the prompt's own schema tag for its calls, so the turn leads with prose
    and a ``<plan>`` block and only then emits ``<tools>{...}</tools>``. The payload neither starts
    the text (the bare path) nor sits in a fence (the fenced path), so every turn read as
    ``continue``: 0 GEN calls, no image observation, and a flat 0 reward on all 90 steps.

    The trailing ``<output>`` blocks are the model role-playing the tool's side of the exchange --
    it invented them precisely *because* no call was ever executed -- and must not be mistaken for
    calls themselves. The first real ``<tools>`` call wins.
    """
    text = (
        "Sure, I've got the plan ready!\n"
        "\n"
        "<plan>\n"
        "  1. Create an image depicting a destroyed school in an anime style.\n"
        "  2. Generate the image using the provided prompt.\n"
        "  3. Assess the image's quality and relevance to the original request.\n"
        "</plan>\n"
        "\n"
        "<tools>\n"
        "  {\n"
        '    "name": "generate_image",\n'
        '    "arguments": {\n'
        '      "prompt": "A chaotic anime-style cartoon depiction of a destroyed school"\n'
        "    }\n"
        "  }\n"
        "</tools>\n"
        "\n"
        "<output>\n"
        '  {"name": "generate_image", "arguments": {}}\n'
        "</output>\n"
        "\n"
        "<output>\n"
        '  {"name": "judge_image", "arguments": {}}\n'
        "</output>\n"
    )
    call = lib.parse_und_tool_call(text)
    assert call is not None and call["name"] == "generate_image"
    assert call["arguments"]["prompt"] == "A chaotic anime-style cartoon depiction of a destroyed school"
    assert lib.und_turn_kind(text) == "generate_image"


def test_tools_schema_echo_is_not_a_tool_call():
    """The prompt itself contains a ``<tools>`` block, so a turn can echo the *signature* back.

    A call carries ``arguments``; a signature carries ``parameters``/``description``. Without that
    distinction the schema echo parses as a call and ``run_serial_episode`` reads
    ``arguments.get("prompt", "")`` off the schema's ``parameters`` -- a GEN request with an empty
    prompt.
    """
    text = (
        "Sure, here are the tools I have available:\n"
        "<tools>\n"
        '{"type": "function", "function": {"name": "generate_image", "description": "Generate an '
        'image from a text prompt using the Bagel GEN pathway.", "parameters": {"type": "object", '
        '"properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}}}\n'
        "</tools>\n"
    )
    assert lib.parse_und_tool_call(text) is None
    assert lib.und_turn_kind(text) == "continue"


def test_tools_judge_image_is_inert():
    """``<tools>`` is a dialect of the checkpoint, not of Hermes, so the judge turn stays inert."""
    text = 'Plan done.\n<tools>\n{"name": "judge_image", "arguments": {}}\n</tools>\n'
    assert lib.parse_und_tool_call(text)["name"] == "judge_image"
    assert lib.und_turn_kind(text) == "continue"


def test_tools_tagged_call_without_arguments_is_not_a_call():
    """A ``<tools>`` span naming a tool but carrying no ``arguments`` is not a complete call."""
    text = 'Here you go.\n<tools>\n{"name": "generate_image"}\n</tools>\n'
    assert lib.parse_und_tool_call(text) is None
    assert lib.und_turn_kind(text) == "continue"


def test_jxk_ids_never_group_flowgrpo_across_und_prompts():
    a = lib.bind_episode_ids(dataset_task_uid="taskA", gen_call_id="callA")
    b = lib.bind_episode_ids(dataset_task_uid="taskB", gen_call_id="callB")
    assert a["und_group_uid"] == "taskA"
    assert b["und_group_uid"] == "taskB"
    assert a["gen_group_uid"] == "callA"
    assert a["gen_group_uid"] != b["gen_group_uid"]
    assert lib.gen_sample_uid("callA", 0) == "callA:0"


def _episode(*, uid: str, gens, used_image: bool):
    return lib.EpisodeRollout(
        und_group_uid=uid,
        episode_uid=uid + "-ep",
        policy_version=1,
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        turns=2,
        gen_samples=gens,
        used_image_credit=used_image,
    )


def test_flatten_reflection_only_has_zero_gen_rows():
    ep = _episode(uid="t0", gens=[], used_image=False)
    result = lib.flatten_multiturn_rollouts([ep], expected_s=2)
    assert len(result.und_batch) == 1
    assert result.gen_batch == []
    assert result.metrics["und/no_image_credit"] == 1.0
    assert result.metrics["gen/skipped_no_groups"] == 1.0


def test_flatten_drops_incomplete_k_groups():
    call = "g1"
    samples = [
        lib.GenSample(
            gen_sample_uid=lib.gen_sample_uid(call, 0),
            gen_group_uid=call,
            seed_index=0,
            valid=True,
            prompt_token_ids=[1],
            rm_score=1.0,
        )
    ]
    result = lib.flatten_multiturn_rollouts([_episode(uid="t0", gens=samples, used_image=True)], expected_s=2)
    assert result.gen_batch == []
    assert result.metrics["gen/dropped_incomplete_groups"] == 1.0


def test_flatten_complete_k_group_keeps_prompt_token_ids():
    call = "g1"
    samples = [
        lib.GenSample(
            gen_sample_uid=lib.gen_sample_uid(call, i),
            gen_group_uid=call,
            seed_index=i,
            valid=True,
            prompt_token_ids=[9, 8],
            rm_score=float(i + 1),
            all_latents="latents",
        )
        for i in range(2)
    ]
    result = lib.flatten_multiturn_rollouts([_episode(uid="t0", gens=samples, used_image=True)], expected_s=2)
    assert len(result.gen_batch) == 2
    assert result.gen_batch[0]["prompt_token_ids"] == [9, 8]
    assert result.und_batch[0]["token_level_scores"] == pytest.approx(1.5)
    stripped = lib.strip_pixels_for_actor({"prompt_embeds": 1, "prompt_token_ids": [1], "images": []})
    assert "prompt_embeds" not in stripped


def test_flatten_from_agent_output_reads_extra_fields():
    call = "g1"
    samples = [
        lib.GenSample(
            gen_sample_uid=lib.gen_sample_uid(call, i),
            gen_group_uid=call,
            seed_index=i,
            valid=True,
            prompt_token_ids=[2],
            rm_score=1.0,
            rollout_log_probs=[0.1, 0.2],
        )
        for i in range(2)
    ]
    output = types.SimpleNamespace(
        non_tensor_batch={
            "extra_fields": [
                {
                    "und_group_uid": "task",
                    "episode_uid": "ep0",
                    "prompt_ids": [1],
                    "response_ids": [3],
                    "response_mask": [1],
                    "turns": 2,
                    "used_image_credit": True,
                    "gen_samples": samples,
                }
            ]
        }
    )
    result = lib.flatten_from_agent_output(output, expected_s=2)
    assert len(result.gen_batch) == 2
    assert result.gen_batch[0]["gen_group_uid"] == "g1"
    assert result.metrics["gen/skipped_no_groups"] == 0.0


def test_max_generate_passes_one_refuses_second_gen():
    tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=1)

    async def _run():
        await tool(prompt="a", prompt_token_ids=[1], gen_call_id="c1", seeds=[0, 1])
        await tool(prompt="b", prompt_token_ids=[1], gen_call_id="c2", seeds=[0, 1])

    with pytest.raises(lib.GenerateImageCapError, match="second"):
        asyncio.run(_run())


def test_default_gen_seeds_are_not_the_constant_range_s():
    """The GEN group must vary per episode and per call, not be ``range(S)``.

    The bagel pipeline seeds its diffusion noise straight from
    ``sampling_params.seed`` (``torch.manual_seed`` in vllm_omni
    ``pipeline_bagel.py``), so a repeated seed is a repeated image. Measured
    2026-09-21 on hk01dgx039: ``/tmp/bagel_corl_gen/`` held 42 PNGs of which only 22
    were distinct -- eight groups of three byte-identical files, written by three
    separate runs hours apart. FlowGRPO cannot learn from a group with no variance.
    """
    seen: list[list[int]] = []

    async def gen_fn(**kwargs):
        seen.append(list(kwargs["seeds"]))
        return [{"valid": True, "image_path": "/tmp/seed.png"} for _ in kwargs["seeds"]]

    async def _run():
        for base in (5, 9):  # two episodes -> two different seed bases
            tool = lib.BagelGenerateImageTool(
                gen_samples_per_call=2,
                max_generate_passes=2,
                generate_fn=gen_fn,
                seed_base=base,
            )
            await tool(prompt="same prompt", prompt_token_ids=[1], gen_call_id=f"b{base}c1")
            await tool(prompt="same prompt", prompt_token_ids=[1], gen_call_id=f"b{base}c2")

    asyncio.run(_run())

    assert len(seen) == 4, "two episodes x two generate_image calls"
    for seeds in seen:
        assert len(seeds) == 2
        # Within one FlowGRPO group the S seeds must be distinct, else two of the
        # samples are the same noise draw and the group loses that much variance.
        assert len(set(seeds)) == 2, f"seeds collided inside one group: {seeds}"
    flat = [seed for group in seen for seed in group]
    # Across episodes and across calls: no seed may repeat, or those groups
    # re-denoise the same prompt to byte-identical images.
    assert len(set(flat)) == len(flat), f"seed groups overlapped: {seen}"
    assert all(group != [0, 1] for group in seen), "regressed to the constant range(S)"


def test_explicit_gen_seeds_are_passed_through_untouched():
    """Replay/tests pass ``seeds`` explicitly; the default derivation must not win."""
    seen: list[list[int]] = []

    async def gen_fn(**kwargs):
        seen.append(list(kwargs["seeds"]))
        return [{"valid": True, "image_path": "/tmp/x.png"} for _ in kwargs["seeds"]]

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, generate_fn=gen_fn, seed_base=123456)
        await tool(prompt="p", prompt_token_ids=[1], gen_call_id="c1", seeds=[7, 8])

    asyncio.run(_run())
    assert seen == [[7, 8]]


def test_gen_seed_count_must_match_s():
    """Fail closed when an explicit seed list does not match the configured group size."""
    tool = lib.BagelGenerateImageTool(gen_samples_per_call=2)

    async def _run():
        await tool(prompt="p", prompt_token_ids=[1], gen_call_id="c1", seeds=[7])

    with pytest.raises(ValueError, match="expected S=2"):
        asyncio.run(_run())


def test_serial_episode_await_und_then_gen():
    order: list[str] = []

    async def und_decode(**kwargs):
        order.append("und")
        return {
            "token_ids": [7],
            "text": '<tool_call>{"name": "generate_image", "arguments": {"prompt": "x"}}</tool_call>',
            "done_token_ids": [8],
        }

    async def gen_fn(**kwargs):
        order.append("gen")
        return [{"valid": True, "image_path": "/tmp/a.png"} for _ in kwargs["seeds"]]

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, generate_fn=gen_fn)
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=3,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
        )

    episode = asyncio.run(_run())
    assert order == ["und", "gen"]
    assert len(episode.gen_samples) == 2
    assert episode.response_mask[-1] == 1


def test_forced_reflection_masks_and_observation_encoding():
    tok = _FakeTokenizer()
    gen_calls = {"n": 0}

    async def und_decode(**kwargs):
        if "Reflection:" in tok.decode(kwargs["response_ids"]):
            return {"token_ids": tok.encode("Done."), "text": "Done."}
        return {
            "token_ids": tok.encode('<tool_call>{"name": "generate_image", "arguments": {"prompt": "cat"}}</tool_call>'),
            "text": '<tool_call>{"name": "generate_image", "arguments": {"prompt": "cat"}}</tool_call>',
        }

    async def gen_fn(**kwargs):
        gen_calls["n"] += 1
        return [{"valid": True, "image_path": "/tmp/obs.png"} for _ in kwargs["seeds"]]

    def score_fn(samples):
        for s in samples:
            s.rm_score = 0.9
            s.good_enough = True
        return samples

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=1, generate_fn=gen_fn)
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
            score_fn=score_fn,
            tokenizer=tok,
        )

    episode = asyncio.run(_run())
    assert gen_calls["n"] == 1
    assert episode.forced_reflection is True
    assert episode.stop_required is True
    assert episode.judge_text is not None
    assert "good_enough=YES" in episode.judge_text
    assert 0 in episode.response_mask
    assert episode.response_mask[-1] == 1
    obs_ids = tok.encode("path=/tmp/obs.png")
    joined = episode.response_ids
    obs_start = None
    for i in range(len(joined) - len(obs_ids) + 1):
        if joined[i : i + len(obs_ids)] == obs_ids:
            obs_start = i
            break
    assert obs_start is not None
    assert episode.response_mask[obs_start : obs_start + len(obs_ids)] == [0] * len(obs_ids)
    done_ids = tok.encode("Done.")
    assert joined[-len(done_ids) :] == done_ids
    assert episode.response_mask[-len(done_ids) :] == [1] * len(done_ids)
    assert episode.response_mask[-len(done_ids) - 1] == 0
    assert episode.und_reward == pytest.approx(0.9)
    flat = lib.flatten_multiturn_rollouts([episode], expected_s=2)
    assert flat.und_batch[0]["token_level_scores"] == pytest.approx(0.9)


def test_observation_and_forced_tokens_get_no_fabricated_rollout_log_prob():
    """π_rollout belongs to *sampled* tokens only; env tokens carry 0.0 and stay masked.

    A generate_image turn appends the observation (mask 0) and the forced reflection (mask 0)
    between two sampled decodes. Both must keep ``rollout_log_probs`` positionally aligned
    while contributing nothing to the ratio, i.e. the property to hold is

        mask[i] == 0  ->  logprob[i] == 0.0     (never read as a policy ratio)
        mask[i] == 1  ->  logprob[i] == sampled (the engine's own value)
    """
    tok = _FakeTokenizer()
    sampled = -0.5

    def step(text: str) -> dict:
        ids = tok.encode(text)
        return {"token_ids": ids, "text": text, "log_probs": [sampled] * len(ids)}

    async def und_decode(**kwargs):
        if "Reflection:" in tok.decode(kwargs["response_ids"]):
            return step("Done.")
        return step('<tool_call>{"name": "generate_image", "arguments": {"prompt": "cat"}}</tool_call>')

    async def gen_fn(**kwargs):
        return [{"valid": True, "image_path": "/tmp/obs.png"} for _ in kwargs["seeds"]]

    def score_fn(samples):
        for s in samples:
            s.rm_score = 0.9
            s.good_enough = True
        return samples

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=1, generate_fn=gen_fn)
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
            score_fn=score_fn,
            tokenizer=tok,
        )

    episode = asyncio.run(_run())
    assert 0 in episode.response_mask, "the observation must still be masked out"
    assert len(episode.rollout_log_probs) == len(episode.response_ids)
    for mask, logprob in zip(episode.response_mask, episode.rollout_log_probs, strict=True):
        assert logprob == pytest.approx(sampled if mask == 1 else 0.0)
    flat = lib.flatten_multiturn_rollouts([episode], expected_s=2)
    assert len(flat.und_batch[0]["rollout_log_probs"]) == len(episode.response_ids)


def test_good_enough_latch_blocks_second_generate():
    """Latch set after first GEN must block a second generate_image (max_passes=2)."""
    ctx.clear_good_enough_yes_reached()
    tok = _FakeTokenizer()
    gen_calls = {"n": 0}

    async def und_decode(**kwargs):
        return {
            "token_ids": tok.encode('<tool_call>{"name": "generate_image", "arguments": {"prompt": "p"}}</tool_call>'),
            "text": '<tool_call>{"name": "generate_image", "arguments": {"prompt": "p"}}</tool_call>',
        }

    async def gen_fn(**kwargs):
        gen_calls["n"] += 1
        return [{"valid": True, "image_path": f"/tmp/g{gen_calls['n']}.png"} for _ in kwargs["seeds"]]

    def score_fn(samples):
        for s in samples:
            s.rm_score = 0.95
            s.good_enough = True
        return samples

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=2, generate_fn=gen_fn)
        episode = await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
            score_fn=score_fn,
            tokenizer=tok,
            max_und_turns=6,
        )
        # Latch visibility is task-scoped (the unified layer deliberately dropped
        # the thread fallback): read it inside the episode's own context.
        latch_after = ctx.get_good_enough_yes_reached()
        return episode, latch_after

    episode, latch_after = asyncio.run(_run())
    assert gen_calls["n"] == 1
    assert latch_after is True
    assert episode.stop_required is True

    # Explicit latch-skip: with latch pre-set and clear disabled, GEN must not run.
    ctx.clear_good_enough_yes_reached()
    ctx.set_good_enough_yes_reached(True)
    blocked = {"n": 0}

    async def gen_blocked(**kwargs):
        blocked["n"] += 1
        return [{"valid": True, "image_path": "/tmp/blocked.png"} for _ in kwargs["seeds"]]

    original_clear = lib.clear_good_enough_yes_reached
    lib.clear_good_enough_yes_reached = lambda: None
    try:

        async def _blocked():
            tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=2, generate_fn=gen_blocked)
            return await lib.run_serial_episode(
                dataset_task_uid="blocked",
                policy_version=1,
                prompt_ids=[1],
                und_decode=und_decode,
                generate_tool=tool,
                tokenizer=tok,
                max_und_turns=3,
            )

        blocked_ep = asyncio.run(_blocked())
    finally:
        lib.clear_good_enough_yes_reached = original_clear
        ctx.clear_good_enough_yes_reached()

    assert blocked["n"] == 0
    assert blocked_ep.gen_samples == []


def _load_config_mod():
    root = Path(__file__).resolve().parents[2]
    _ensure_pkg("verl_omni", root / "verl_omni")
    _ensure_pkg("verl_omni.utils", root / "verl_omni" / "utils")
    return _load_by_path(
        "omni_config_isolated",
        root / "verl_omni" / "utils" / "config.py",
    )


def test_seeds_s_fail_closed_when_below_two():
    from omegaconf import OmegaConf

    cfg_mod = _load_config_mod()
    cfg = OmegaConf.create(
        {
            "trainer": {"resume_mode": "disable", "v1": {"trainer_mode": "bagel_corl_sync"}},
            "actor_rollout_ref": {
                "model": {"path": "/models/ByteDance-Seed/BAGEL-7B-MoT", "lora_rank": 64},
                "rollout": {"n": 8, "agent": {"gen_samples_per_call": 1, "max_generate_passes": 1}},
            },
        }
    )
    with pytest.raises(ValueError, match="gen_samples_per_call >= 2"):
        cfg_mod.validate_bagel_corl_config(cfg)


def _done_und_decode():
    async def und_decode(**kwargs):
        # Pattern 3 (K=0): policy emits Done. with no tool call.
        return {"token_ids": [7], "text": "Done."}

    return und_decode


def test_pattern3_non_image_reward_replaces_zero():
    """K=0 episodes must use the non-image UND scalar, not a hard-coded 0."""
    episode = asyncio.run(
        lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=_done_und_decode(),
            generate_tool=lib.BagelGenerateImageTool(gen_samples_per_call=2),
            non_image_reward=0.42,
        )
    )
    assert episode.num_gen_calls == 0
    assert episode.gen_samples == []
    assert episode.und_reward == pytest.approx(0.42)

    flat = lib.flatten_multiturn_rollouts([episode], expected_s=2)
    assert flat.und_batch[0]["token_level_scores"] == pytest.approx(0.42)
    assert flat.metrics["und/no_image_credit"] == 1.0


def test_pattern3_no_non_image_reward_keeps_zero():
    """Without a non-image scalar, K=0 reward stays 0 (backward compatible)."""
    episode = asyncio.run(
        lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=_done_und_decode(),
            generate_tool=lib.BagelGenerateImageTool(gen_samples_per_call=2),
        )
    )
    assert episode.und_reward == pytest.approx(0.0)


def test_flatten_propagates_episode_und_reward_for_no_image():
    """flatten_multiturn_rollouts must not zero out a pre-set non-image reward."""
    episode = lib.EpisodeRollout(
        und_group_uid="task",
        episode_uid="task-ep",
        policy_version=1,
        prompt_ids=[1],
        response_ids=[7],
        response_mask=[1],
        turns=1,
        gen_samples=[],
        used_image_credit=False,
        und_reward=0.73,
    )
    result = lib.flatten_multiturn_rollouts([episode], expected_s=2)
    assert result.und_batch[0]["token_level_scores"] == pytest.approx(0.73)


def test_k2_episode_keeps_both_calls():
    """RFC: every generate_image call contributes its S seeds — K=2 must yield 2xS
    GEN rows (the old code replaced gen_samples per call and silently dropped the
    first call's trajectories)."""
    ctx.clear_good_enough_yes_reached()
    tok = _FakeTokenizer()
    gen_calls = {"n": 0}

    async def und_decode(**kwargs):
        decoded = tok.decode(kwargs["response_ids"])
        rewrite_count = decoded.count("Reflection:")
        if rewrite_count == 0:
            text = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "v1"}}</tool_call>'
        elif rewrite_count == 1:
            text = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "v2"}}</tool_call>'
        else:
            text = "Done."
        return {"token_ids": tok.encode(text), "text": text}

    async def gen_fn(**kwargs):
        gen_calls["n"] += 1
        return [{"valid": True, "image_path": f"/tmp/k2_{gen_calls['n']}.png"} for _ in kwargs["seeds"]]

    call_index = {"n": 0}

    def score_fn(samples):
        call_index["n"] += 1
        for s in samples:
            s.rm_score = 0.2 if call_index["n"] == 1 else 0.9
            s.good_enough = call_index["n"] != 1
        return samples

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, max_generate_passes=2, generate_fn=gen_fn)
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=tool,
            score_fn=score_fn,
            tokenizer=tok,
        )

    episode = asyncio.run(_run())
    assert gen_calls["n"] == 2
    assert episode.num_gen_calls == 2
    assert len(episode.gen_samples) == 4
    groups = {s.gen_group_uid for s in episode.gen_samples}
    assert len(groups) == 2

    flat = lib.flatten_multiturn_rollouts([episode], expected_s=2)
    assert len(flat.gen_batch) == 4
    assert len({row["gen_group_uid"] for row in flat.gen_batch}) == 2
    assert flat.metrics["gen/dropped_incomplete_groups"] == 0.0
    assert flat.metrics["und/no_image_credit"] == 0.0
    # Episode scalar averages over both calls' seeds: (0.2*2 + 0.9*2) / 4.
    assert episode.und_reward == pytest.approx(0.55)


def test_judge_text_reduction_modes():
    """Episode-level stop bit: any (default best-of-S), all, and mean vs threshold."""
    from verl_omni.agent_loop import rpco_turn_protocol as protocol

    def _samples(flags):
        return [
            lib.GenSample(
                gen_sample_uid=f"g:{i}",
                gen_group_uid="g",
                seed_index=i,
                valid=True,
                prompt_token_ids=[1],
                rm_score=0.5,
                good_enough=flag,
            )
            for i, flag in enumerate(flags)
        ]

    assert "good_enough=YES" in lib.judge_text_from_gen_samples(_samples([True, False]))
    text = lib.judge_text_from_gen_samples(_samples([True, False]), reduction="all")
    assert "good_enough=NO" in text
    assert "good_enough=NO" in lib.judge_text_from_gen_samples(_samples([True, False]), reduction="mean", threshold=1.0)
    assert (
        "good_enough=YES"
        in lib.judge_text_from_gen_samples(_samples([True, False]), reduction="mean", threshold=0.5)
    )
    # Derived branch (no explicit flags) honours the threshold pass-through.
    # Fresh lists per assertion: judge_text_from_gen_samples writes the derived
    # stop bit back onto samples (episode latch input), so reuse would couple calls.
    def _unscored():
        return [
            lib.GenSample(
                gen_sample_uid=f"g:{i}",
                gen_group_uid="g",
                seed_index=i,
                valid=True,
                prompt_token_ids=[1],
                rm_score=0.75,
            )
            for i in range(2)
        ]

    assert "good_enough=YES" in lib.judge_text_from_gen_samples(_unscored(), threshold=0.7)
    assert "good_enough=NO" in lib.judge_text_from_gen_samples(_unscored(), threshold=0.9)
    assert protocol.derive_good_enough_from_scores(correctness=0.75, aesthetics=0.75, threshold=0.7) is True


def test_aggregate_episode_metrics_averages_over_siblings():
    """Step-level J/K metrics are batch means; first-row-wins collapse is gone."""
    records = [
        {
            "fields": {
                "episode_J": 2,
                "episode_K": 1,
                "bagel_corl_metrics": {
                    "episode/J": 2.0,
                    "episode/K": 1.0,
                    "gen/dropped_incomplete_groups": 1.0,
                    "und/no_image_credit": 0.0,
                    "gen/skipped_no_groups": 0.0,
                },
            }
        },
        {
            "fields": {
                "episode_J": 4,
                "episode_K": 0,
                "bagel_corl_metrics": {
                    "episode/J": 4.0,
                    "episode/K": 0.0,
                    "gen/dropped_incomplete_groups": 0.0,
                    "und/no_image_credit": 1.0,
                    "gen/skipped_no_groups": 1.0,
                },
            }
        },
    ]
    out = lib.aggregate_episode_metrics(records)
    assert out["episode/J"] == pytest.approx(3.0)
    assert out["episode/K"] == pytest.approx(0.5)
    assert out["gen/dropped_incomplete_groups"] == pytest.approx(1.0)
    assert out["und/no_image_credit"] == pytest.approx(0.5)
    assert out["gen/skipped_no_groups"] == pytest.approx(0.5)
    # Empty batch: no episode-derived keys, dropped stays a real 0 sum.
    empty = lib.aggregate_episode_metrics([])
    assert empty == {"gen/dropped_incomplete_groups": 0.0}


def test_aggregate_episode_metrics_sums_r2_counts_and_means_r2_ratios():
    """RFC §4.4.4: cache counters add up; the three ratios are batch means.

    Re-deriving the ratios from the summed counters would be wrong — a long episode
    would outweigh a short one — so each record contributes its own measured ratio,
    and a key only some episodes measured stays a mean over the episodes that did.
    """
    records = [
        {
            "fields": {
                "bagel_corl_metrics": {
                    "gen/cond_cache_hits": 3.0,
                    "gen/cond_cache_misses": 1.0,
                    "gen/cond_cache_bypassed": 0.0,
                    "gen/cond_recompute_ratio": 0.25,
                    "gen/prompt_embed_cache_hit_rate": 0.75,
                    "gen/cond_amortization": 4.0,
                }
            }
        },
        {
            "fields": {
                "bagel_corl_metrics": {
                    "gen/cond_cache_hits": 0.0,
                    "gen/cond_cache_misses": 0.0,
                    "gen/cond_cache_bypassed": 2.0,
                    "gen/cond_recompute_ratio": 1.0,
                    # 0 hits and 0 misses: no cache decision was made, so the engine
                    # published no hit rate and the aggregate must not invent one.
                    "gen/cond_amortization": 1.0,
                }
            }
        },
    ]
    out = lib.aggregate_episode_metrics(records)
    assert out["gen/cond_cache_hits"] == pytest.approx(3.0)
    assert out["gen/cond_cache_misses"] == pytest.approx(1.0)
    assert out["gen/cond_cache_bypassed"] == pytest.approx(2.0)
    assert out["gen/cond_recompute_ratio"] == pytest.approx((0.25 + 1.0) / 2)
    # Mean over the one episode that measured it — not 3/(3+1) re-derived from the sums.
    assert out["gen/prompt_embed_cache_hit_rate"] == pytest.approx(0.75)
    assert out["gen/cond_amortization"] == pytest.approx((4.0 + 1.0) / 2)

    # And a batch where nothing consulted the cache publishes no rate at all.
    quiet = lib.aggregate_episode_metrics(
        [{"fields": {"bagel_corl_metrics": {"gen/cond_recompute_ratio": 1.0, "gen/cond_cache_bypassed": 2.0}}}]
    )
    assert "gen/prompt_embed_cache_hit_rate" not in quiet


def test_turn_histogram_tail_frac_is_zero_without_a_tail():
    """Constant turn counts have no tail; the old >= p95 rule reported 1.0."""
    assert lib.turn_histogram([1, 1, 1, 1])["tail_frac"] == 0.0
    assert lib.turn_histogram([1])["tail_frac"] == 0.0
    hist = lib.turn_histogram([1, 1, 1, 1, 9])
    assert hist["tail_frac"] == pytest.approx(0.2)


def _context_budget_episode(max_context_tokens: int | None):
    """Drive ``run_serial_episode`` with an engine that refuses a spent-up prompt.

    Mirrors ``ARStrategy.preprocess_input``: it computes
    ``max_possible_tokens = max_model_len - len(prompt_ids)`` and raises when that is not
    positive, so a decode started with the episode's whole context already used up dies
    with the measured error rather than returning an empty output.
    """
    budget = 64
    prompt = [1] * 16
    decodes: list[int] = []

    def engine(**kwargs):
        length = len(kwargs["prompt_ids"]) + len(kwargs["response_ids"])
        decodes.append(length)
        if length >= budget:
            raise ValueError(
                f"Prompt length ({length}) meets or exceeds the model's maximum context "
                f"length ({budget}), leaving no space for generation."
            )
        return {"token_ids": [7] * 12, "text": "thinking"}

    async def gen_fn(**kwargs):
        return [{"valid": True, "image_path": "/tmp/a.png"} for _ in kwargs["seeds"]]

    async def _run():
        tool = lib.BagelGenerateImageTool(gen_samples_per_call=2, generate_fn=gen_fn)
        episode = await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=prompt,
            und_decode=engine,
            generate_tool=tool,
            max_und_turns=8,
            max_context_tokens=max_context_tokens,
        )
        return episode, decodes

    return _run


def test_serial_episode_stops_at_the_episode_context_budget():
    """The UND loop must not start a decode the AR engine cannot serve.

    ``_und_decode`` passes ``prompt_ids + response_ids`` as the decode prompt, so the
    context grows every turn while ``max_und_turns`` stays fixed. Asked for a turn with the
    whole budget spent, the engine refuses:

        ValueError: Prompt length (2048) meets or exceeds the model's maximum context
        length (2048), leaving no space for generation.

    Measured 2026-09-17 09:30 (1024+1024 budget, ``max_und_turns=8``), which killed every
    UND tool-call decode and left the step with no materializable trajectories. The loop has
    to stop at the budget and keep the turns that fit.
    """
    episode, decodes = asyncio.run(_context_budget_episode(max_context_tokens=64)())
    # 16 prompt + 12 per turn: decodes start at 16/28/40/52, then 64 exhausts the budget.
    assert decodes == [16, 28, 40, 52]
    assert episode.turns == 4
    assert len(episode.prompt_ids) + len(episode.response_ids) == 64
    assert episode.response_ids == [7] * 48


def test_without_the_context_budget_the_same_episode_hits_the_engine_wall():
    """The guard is load-bearing: unset, the very same episode reproduces the measured error."""
    with pytest.raises(ValueError, match="leaving no space for generation"):
        asyncio.run(_context_budget_episode(max_context_tokens=None)())


def _logged_episode(*, turn_log_probs: list[float | None]):
    """One UND turn per entry in ``turn_log_probs``; ``None`` = engine returned none."""
    steps = list(turn_log_probs)

    def engine(**kwargs):
        _ = kwargs
        value = steps.pop(0)
        step = {"token_ids": [7, 8], "text": "thinking"}
        if value is not None:
            step["log_probs"] = [value, value]
        return step

    async def _run():
        return await lib.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=engine,
            generate_tool=lib.BagelGenerateImageTool(gen_samples_per_call=2),
            max_und_turns=len(turn_log_probs),
        )

    return _run


def test_serial_episode_keeps_the_rollout_log_probs_aligned_with_the_response():
    """π_rollout has to arrive positionally aligned, because the trainer reads it per token.

    ``AgentLoopOutput.as_dict`` turns ``response_logprobs`` into the ``rollout_log_probs`` TQ
    field (``agent_loop.py:124``) which the v1 trainer pairs with our recomputed
    ``old_log_probs``. A drift here silently pairs a token with another token's log-prob.
    """
    episode = asyncio.run(_logged_episode(turn_log_probs=[-0.1, -0.2])())
    assert episode.response_ids == [7, 8, 7, 8]
    assert episode.rollout_log_probs == pytest.approx([-0.1, -0.1, -0.2, -0.2])
    assert len(episode.rollout_log_probs) == len(episode.response_ids)


def test_a_turn_without_log_probs_still_keeps_the_lists_aligned():
    """A decode that returned no log-probs must not shift every later position."""
    episode = asyncio.run(_logged_episode(turn_log_probs=[-0.5, None, -0.75])())
    assert len(episode.rollout_log_probs) == len(episode.response_ids) == 6
    # The unmeasured turn contributes zeros; its neighbours keep their own values.
    assert episode.rollout_log_probs == pytest.approx([-0.5, -0.5, 0.0, 0.0, -0.75, -0.75])


def test_flatten_publishes_the_rollout_log_probs_the_trainer_reads():
    """The flatten path is the row the trainer sees; π_rollout has to be on it."""
    episode = asyncio.run(_logged_episode(turn_log_probs=[-0.25])())
    flat = lib.flatten_multiturn_rollouts([episode], expected_s=2)
    assert flat.und_batch[0]["rollout_log_probs"] == pytest.approx([-0.25, -0.25])


def test_a_misaligned_rollout_log_prob_list_is_rejected():
    """Fail loud instead of publishing a ratio computed from the wrong positions."""
    with pytest.raises(ValueError, match="rollout_log_probs must align"):
        lib.EpisodeRollout(
            und_group_uid="task",
            episode_uid="task-ep",
            policy_version=1,
            prompt_ids=[1],
            response_ids=[7, 8],
            response_mask=[1, 1],
            rollout_log_probs=[-0.1],
            turns=1,
        )


# ---------------------------------------------------------------------------
# Per-turn UND token cap
# ---------------------------------------------------------------------------
# Without an explicit ``max_tokens`` the AR strategy caps a turn at
# ``min(response_length, prompt_length + response_length - len(prompt))``, which for the
# recipe's 1024+1024 budget is the whole episode budget. One degenerate turn then ends the
# episode: measured 2026-09-22 on hk01dgx039 (devices 2-5), turn 1 spent all 1024 tokens on
# ``assistant\n`` and the episode closed at ``response_tokens=1665`` with ``gen_calls=0``.
_RECIPE_BUDGET = 1024 + 1024


def test_turn_cap_leaves_room_for_later_turns():
    """The first turn of a fresh episode must not be allowed to take everything."""
    cap = lib.und_turn_max_tokens(context_used=383, max_context_tokens=_RECIPE_BUDGET)
    assert 0 < cap < _RECIPE_BUDGET - 383, "one turn must not be able to spend the episode"
    # ... and the remainder must still fit a full plan + call (measured ~120 tokens).
    assert _RECIPE_BUDGET - 383 - cap >= 120


def test_turn_cap_shrinks_with_the_remaining_budget():
    caps = [
        lib.und_turn_max_tokens(context_used=used, max_context_tokens=_RECIPE_BUDGET)
        for used in (383, 800, 1200, 1600)
    ]
    assert caps == sorted(caps, reverse=True), caps
    assert all(cap > 0 for cap in caps)


def test_turn_cap_never_exceeds_the_remaining_budget():
    for used in range(0, _RECIPE_BUDGET + 200, 137):
        cap = lib.und_turn_max_tokens(context_used=used, max_context_tokens=_RECIPE_BUDGET)
        assert cap <= max(_RECIPE_BUDGET - used, 0), (used, cap)


def test_a_spent_budget_yields_no_turn():
    assert lib.und_turn_max_tokens(context_used=_RECIPE_BUDGET, max_context_tokens=_RECIPE_BUDGET) == 0
    assert lib.und_turn_max_tokens(context_used=_RECIPE_BUDGET + 10, max_context_tokens=_RECIPE_BUDGET) == 0


def test_turn_cap_keeps_the_engine_default_when_disabled():
    """``fraction=1.0`` must reproduce the previous (uncapped) behaviour exactly."""
    assert lib.und_turn_max_tokens(context_used=383, max_context_tokens=_RECIPE_BUDGET, fraction=1.0) == _RECIPE_BUDGET - 383


def test_turn_cap_rejects_a_nonsense_fraction():
    with pytest.raises(ValueError, match="fraction"):
        lib.und_turn_max_tokens(context_used=0, max_context_tokens=100, fraction=0.0)


def test_und_stop_sequences_target_the_invented_tool_output():
    """``<output>`` is how the checkpoint role-plays the tool's side; stopping there cuts the
    spiral at the call boundary, and the template's own wrapper is ``<tool_response>``, so no
    legitimate content is truncated."""
    assert lib.UND_STOP_SEQUENCES == ("<output>",)


# ---------------------------------------------------------------------------
# AR-replica health canary
# ---------------------------------------------------------------------------
# The UND lane's worst failure mode is silent: ``LLM Worker.sleep(level=1)`` offloaded only the
# ``weights`` tag, so every other allocation was unmap_and_release'd with no CPU backup and the
# next wake re-mapped it as fresh ZEROED memory. The replica kept serving 200s -- it just stopped
# being a language model. Measured 2026-09-21 on hk01dgx039: ``/v1/chat/completions`` answered
# "Name three fruits." with ``' . . . . . . . . . . . .'`` while every UND turn in the rollout
# collapsed to ``assistant\n``. These cases pin the judgement that notices it.
def test_health_verdict_calls_a_repeated_token_a_loop():
    """The measured corrupted output: one token, over and over."""
    degenerate, share = lib.und_health_verdict(" . . . . . . . . . . . .")
    assert degenerate and share == 1.0


def test_health_verdict_calls_an_empty_answer_degenerate():
    """An empty answer is not a terse answer -- the replica produced nothing."""
    assert lib.und_health_verdict("") == (True, 1.0)
    assert lib.und_health_verdict("   ") == (True, 1.0)


def test_health_verdict_accepts_a_healthy_answer():
    healthy = "Apples, bananas, and cherries are three fruits."
    degenerate, share = lib.und_health_verdict(healthy)
    assert not degenerate
    assert share < lib.UND_HEALTH_REPEAT_SHARE


def test_health_verdict_ignores_case_and_punctuation_when_counting_repeats():
    """``'Ba Na na na na na'`` is the same loop as ``'na na na na'``."""
    degenerate, _ = lib.und_health_verdict("Ba na na na na na na na")
    assert degenerate


def test_health_verdict_refuses_to_judge_a_two_word_answer():
    """Too short to tell a terse model from a broken one, so it must not report healthy."""
    degenerate, _ = lib.und_health_verdict("Two words")
    assert degenerate


def test_health_canary_is_the_prompt_the_live_probe_used():
    """The canary that caught the zeroed replica, so the constant cannot drift from the evidence."""
    assert lib.UND_HEALTH_CANARY == "Name three fruits."


def test_health_verdict_thresholds_are_ordered_sensibly():
    """A share bound outside (0, 1) would make every answer (or none) degenerate."""
    assert 0.0 < lib.UND_HEALTH_REPEAT_SHARE < 1.0
    assert lib.UND_HEALTH_MIN_WORDS >= 1


# ---------------------------------------------------------------------------
# Degenerate-turn counter (the artifact-level oracle)
# ---------------------------------------------------------------------------
# An HTTP canary cannot detect this corruption: the checkpoint only behaves when the ``<tools>``
# schema is rendered into the system turn by the agent loop, and the omni chat endpoint rejects
# ``tools`` (HTTP 400). Measured 2026-09-22 against the *healthy* post-fix replica: a bare chat
# probe still answered ' . . . . . .'. The real turns are the honest signal, so that is what the
# dump counts.
def _und(record_kind, text):
    return {"record": "und_turn", "kind": record_kind, "text": text}


def test_degenerate_counter_counts_the_measured_assistant_loop():
    """The failure that cost a whole run: ``assistant\\n`` repeated, K stayed 0."""
    trace = [_und("continue", "assistant\n" * 300), _und("continue", "assistant\n" * 300)]
    assert lib.count_degenerate_und_turns(trace) == 2


def test_degenerate_counter_ignores_a_real_tool_call():
    """A ``generate_image`` turn is the success condition, never a loop -- even though its JSON
    repeats field-name tokens."""
    call = '{\n  "name": "generate_image",\n  "arguments": {\n    "prompt": "a red circle"\n  }\n}'
    assert lib.count_degenerate_und_turns([_und("generate_image", call)]) == 0


def test_degenerate_counter_ignores_a_short_continue_turn():
    """A terse plan is legitimate; it is only a loop when it keeps saying the same thing."""
    assert lib.count_degenerate_und_turns([_und("continue", "Let me think.")]) == 0


def test_degenerate_counter_ignores_a_prose_plan():
    trace = [_und("continue", "Sure, I have the plan ready! I will call the image tool now.")]
    assert lib.count_degenerate_und_turns(trace) == 0


def test_degenerate_counter_ignores_non_und_records():
    """GEN call records have their own shape and no UND text."""
    trace = [
        {"record": "gen_call", "kind": "generate_image", "prompt": "x " * 100},
        _und("continue", "assistant\n" * 200),
    ]
    assert lib.count_degenerate_und_turns(trace) == 1


def test_degenerate_counter_handles_a_missing_text_field():
    """A record without text must not raise inside the dump path."""
    assert lib.count_degenerate_und_turns([{"record": "und_turn", "kind": "continue"}]) == 0


def test_degenerate_counter_loop_threshold_is_positive():
    assert lib.UND_LOOP_MIN_TOKENS >= 1
