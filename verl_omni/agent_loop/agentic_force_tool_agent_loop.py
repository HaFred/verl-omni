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

"""ToolAgentLoop wrapper that forces multi-turn ``generate_image`` when Hermes is missing.

Lance_3B_hf_und does not reliably emit ``<tool_call>`` XML early in GRPO, so
stock ``tool_agent`` never reaches the frozen Lance server. This loop keeps the
stock ToolAgentLoop path, but synthesizes ``generate_image`` until
``AGENTIC_FORCE_MIN_TOOL_CALLS`` successful tool turns have run (default 2), so
rollouts are multi-turn and GRPO trains on tool/obs trajectories.

Note (fred): Temporary workaround until a tool-format SFT checkpoint emits
Hermes tool calls on its own — then switch back to stock ``tool_agent``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall

from verl_omni.agent_loop.agentic_image_reflection import reflect_on_generated_image
from verl_omni.agent_loop.agentic_trajectory_context import (
    allocate_rollout_n,
    build_trajectory_relpath,
    set_active_call_provenance,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TOOL_NAME = "generate_image"


def _force_enabled() -> bool:
    return os.getenv("AGENTIC_FORCE_GENERATE_IMAGE", "1").strip().lower() in {"1", "true", "yes"}


def _min_tool_calls() -> int:
    """Minimum generate_image executions before allowing a free-text finish."""
    try:
        return max(1, int(os.getenv("AGENTIC_FORCE_MIN_TOOL_CALLS", "2")))
    except ValueError:
        return 2


def _force_probability() -> float:
    """Configured probability of forcing the first missing tool call."""
    try:
        probability = float(os.getenv("AGENTIC_FORCE_PROB", "1.0"))
    except ValueError:
        probability = 1.0
    return min(1.0, max(0.0, probability))


def _effective_force_probability(*, executed: int, already_forced: bool) -> float:
    """Once a traj has any tool turn (or prior force), always finish min_tool_calls.

    Avoids the 0.25² lottery that zeroed images after the force-decay cliff.
    """
    if executed > 0 or already_forced:
        return 1.0
    return _force_probability()


def _teacher_force_hermes_enabled() -> bool:
    """Impose Hermes XML on non-Hermes rollouts (default ON for PR1 overfit).

    When the decode already contains a valid ``<tool_call>`` for generate_image,
    tokens are kept on-policy. Otherwise we wrap rollout content into Hermes
    (bare JSON → XML, caption → tool call) instead of discarding the span.
    """
    return os.getenv("AGENTIC_TEACHER_FORCE_HERMES", "1").strip().lower() in {"1", "true", "yes"}


def _prefer_llm_reflection_enabled() -> bool:
    """Prefer Lance und decode for Reflection + rewrite prompt (default ON)."""
    return os.getenv("AGENTIC_PREFER_LLM_REFLECTION", "1").strip().lower() in {"1", "true", "yes"}


def _overfit_stable_teacher_enabled() -> bool:
    """Use a fixed gen→Reflection→rewrite target (ignore caption wrap) for overfit.

    Cold Lance und rarely emits Hermes. ``wrap_caption`` then trains on whatever
    garbage caption was sampled, so GRPO never sees a stable fewshot-like target.
    When this flag is on, forced turns always use the teacher Hermes template with
    deterministic prompts derived from the user task.
    """
    return os.getenv("AGENTIC_OVERFIT_STABLE_TEACHER", "0").strip().lower() in {"1", "true", "yes"}


def _overfit_on_policy_frac() -> float:
    """Fraction of forced turns that keep raw decode tokens (no teacher replace).

    Needed so GRPO groups are not all ~reward=1 (advantages collapse to 0).
    Those on-policy failures stay low-reward garbage; teacher-replaced siblings
    get the high format/reflect/tool score → positive advantage toward the template.
    """
    try:
        frac = float(os.getenv("AGENTIC_OVERFIT_ON_POLICY_FRAC", "0.25"))
    except ValueError:
        frac = 0.25
    return min(0.9, max(0.0, frac))


def _task_to_visual_prompt(user_task: str) -> str:
    """Map 'Generate an image of X' → a stable diffusion caption for overfit BC."""
    task = (user_task or "a detailed visual scene").strip()
    low = task.lower()
    for prefix in (
        "generate an image of ",
        "generate a image of ",
        "create an image of ",
        "create a ",
        "draw ",
        "generate ",
    ):
        if low.startswith(prefix):
            task = task[len(prefix) :].strip()
            break
    task = task.rstrip(".")
    if not task:
        task = "a detailed visual scene"
    return f"{task}, soft studio lighting, coherent composition"


def _stable_overfit_prompt_and_reflection(
    *,
    user_task: str,
    executed: int,
    prev_prompts: list[str],
) -> tuple[str, str, dict]:
    """Fewshot-aligned stable target: call → Reflection + rewritten call."""
    task = (user_task or "a detailed visual scene").strip()
    if executed == 0:
        prompt = _task_to_visual_prompt(task)
        return prompt, "", {"stage": "overfit_initial", "image_ok": False, "content_source": "teacher"}
    prev = (prev_prompts[-1] if prev_prompts else _task_to_visual_prompt(task)).strip()
    reflect = (
        "looking at the generated image, edges are soft and details look muted; "
        "rewrite for sharper detail and richer color."
    )
    prompt = f"{prev}, highly detailed, sharp focus, richer colors, coherent composition"
    if prompt.lower() == prev.lower():
        prompt = f"{prev}, refined, higher quality"
    return (
        prompt,
        reflect,
        {
            "stage": "overfit_reflect_rewrite",
            "image_ok": True,
            "content_source": "teacher",
            "rewritten_prompt": prompt,
        },
    )


def _hermes_generate_image_text(prompt: str, *, reflect: str | None = None) -> str:
    """Emit optional ``Reflection:`` + Hermes ``generate_image``."""
    call = (
        "<tool_call>\n"
        f"{json.dumps({'name': DEFAULT_TOOL_NAME, 'arguments': {'prompt': prompt}}, ensure_ascii=False)}\n"
        "</tool_call>"
    )
    reflect = (reflect or "").strip()
    if not reflect:
        return call
    if reflect.lower().startswith("reflection:"):
        reflect = reflect.split(":", 1)[1].strip()
    return f"Reflection: {reflect}\n{call}"


def _has_valid_hermes_generate_image(text: str) -> bool:
    """True if decode already has a Hermes ``<tool_call>`` for generate_image."""
    for match in _TOOL_CALL_RE.finditer(text or ""):
        body = (match.group(1) or "").strip()
        if not body:
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            if "generate_image" in body.lower() and "prompt" in body.lower():
                return True
            continue
        if isinstance(obj, dict) and str(obj.get("name", "")).lower() == DEFAULT_TOOL_NAME:
            return True
    return False


def _extract_bare_json_generate_image(text: str) -> dict[str, Any] | None:
    """Parse a bare ``{"name":"generate_image",...}`` without Hermes XML wrappers."""
    raw = (text or "").strip()
    if not raw or "<tool_call>" in raw.lower():
        return None
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        start = raw.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        idx = end
        if not isinstance(obj, dict):
            continue
        if str(obj.get("name", "")).lower() != DEFAULT_TOOL_NAME:
            continue
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(args, dict) and args.get("prompt"):
            return {"name": DEFAULT_TOOL_NAME, "arguments": {"prompt": str(args["prompt"]).strip()}}
    return None


def _reflection_from_decode_only(decoded: str) -> str | None:
    """Use Reflection: from the rollout only — never the image-heuristic teacher blob."""
    m = _REFLECT_RE.search(decoded or "")
    if not m:
        return None
    cand = _strip_special_tokens(m.group(1))
    cand = re.split(r"[\{<]", cand, maxsplit=1)[0].strip(" \n\"'")
    if not cand or _looks_like_tool_echo(cand):
        return None
    # Reject teacher-style templates if the model somehow echoed them.
    low = cand.lower()
    if "observed generated image" in low and "brightness=" in low:
        return None
    if low.startswith("looking at the generated image from the frozen"):
        return None
    return cand[:400]


def _impose_hermes_on_rollout(
    decoded: str,
    *,
    prompt: str,
    reflect: str | None,
    require_reflection: bool = False,
) -> tuple[str, str, str]:
    """Keep / wrap rollout into Hermes; preserve rollout content in the training span.

    Returns ``(assistant_text, mode, tool_prompt)``:
      - ``keep``: decode already has valid Hermes (on-policy); tool_prompt from decode/prompt
      - ``wrap_json``: bare JSON → Hermes XML; **no** teacher Reflection (rollout-only)
      - ``wrap_caption``: caption → Hermes; Reflection only if present in decode
      - ``teacher``: echo/junk — only then use resolved ``prompt``/``reflect`` fallback

    Important: image-heuristic reflections must NOT be applied in wrap_* modes, or
    every traj gets the same ``Looking at the generated image...`` blob.
    """
    raw = decoded or ""
    if _has_valid_hermes_generate_image(raw):
        bare = _extract_bare_json_generate_image(raw)  # may be None
        p = prompt
        for match in _TOOL_CALL_RE.finditer(raw):
            try:
                obj = json.loads((match.group(1) or "").strip())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and str(obj.get("name", "")).lower() == DEFAULT_TOOL_NAME:
                args = obj.get("arguments") or {}
                if isinstance(args, dict) and args.get("prompt"):
                    p = str(args["prompt"]).strip() or p
                    break
        return raw, "keep", p

    bare = _extract_bare_json_generate_image(raw)
    if bare is not None:
        p = str(bare["arguments"]["prompt"]).strip() or prompt
        # Rollout-only reflection (if any). Do not inject teacher image heuristics.
        ref = _reflection_from_decode_only(raw)
        if require_reflection and not ref:
            ref = f"Rewrite the previous image toward this rollout refinement: {p}"
        return _hermes_generate_image_text(p, reflect=ref), "wrap_json", p

    if raw.strip() and not _looks_like_tool_echo(raw):
        p = (prompt or "").strip()
        if not p or _looks_like_tool_echo(p):
            p = _strip_special_tokens(raw)[:400]
        low = p.lower()
        if p and not _looks_like_tool_echo(p) and "```" not in p and 'type: "svg"' not in low:
            ref = _reflection_from_decode_only(raw)
            if require_reflection and not ref:
                ref = f"Rewrite the previous image toward this rollout refinement: {p}"
            return _hermes_generate_image_text(p, reflect=ref), "wrap_caption", p

    # True fallback: decode unusable → teacher template (may include image reflect).
    return _hermes_generate_image_text(prompt, reflect=reflect), "teacher", prompt


def _final_confirmation_text(user_task: str) -> str:
    return f"Done. Generated and refined the image for: {user_task}. No further tool calls."


def _extract_tool_prompt(tool_call: FunctionCall) -> str | None:
    try:
        args = json.loads(tool_call.arguments or "{}")
    except json.JSONDecodeError:
        return None
    if isinstance(args, dict) and args.get("prompt"):
        return str(args["prompt"]).strip() or None
    return None


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_REFLECT_RE = re.compile(r"reflection\s*:\s*(.+?)(?=<tool_call>|$)", re.IGNORECASE | re.DOTALL)
_PROMPT_KEY_RE = re.compile(r'["\']prompt["\']\s*:\s*["\']([^"\']+)["\']', re.IGNORECASE)
_REWRITE_HINT_RE = re.compile(
    r"(?:rewritten\s+prompt|new\s+prompt|improved\s+prompt|next\s+prompt)\s*[:=]\s*[\"']?(.+?)[\"']?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_special_tokens(text: str) -> str:
    out = text or ""
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        out = out.replace(tok, " ")
    return " ".join(out.split()).strip()


def _is_usable_diffusion_prompt(text: str, *, user_task: str, prev_prompt: str) -> bool:
    p = _strip_special_tokens(text)
    if len(p) < 8:
        return False
    if _looks_like_tool_echo(p):
        return False
    low = p.lower()
    if "generate_image" in low and "<tool_call>" not in low and "{" not in p:
        return False
    if low.startswith("<"):
        return False
    # Reject pure planning templates / teacher heuristic echoes.
    if low.startswith("planning the first generate_image"):
        return False
    if "observed generated image" in low and "brightness=" in low:
        return False
    if "```" in p or 'type: "svg"' in low:
        return False
    # Prefer prompts that still relate to the user task when possible.
    task_keys = [t for t in re.findall(r"[a-z0-9]+", (user_task or "").lower()) if len(t) >= 4]
    if task_keys:
        hits = sum(1 for t in task_keys if t in low)
        if hits == 0 and prev_prompt and p.lower() == prev_prompt.lower():
            return False
    return True


def _parse_llm_reflection_and_prompt(
    decoded: str,
    *,
    user_task: str,
    prev_prompt: str,
) -> tuple[str | None, str | None]:
    """Pull Reflection text + diffusion prompt from the Lance und decode.

    Returns ``(reflection, prompt)``; either may be None when unusable.
    """
    raw = decoded or ""
    if not raw.strip() or _looks_like_tool_echo(raw):
        return None, None

    reflect = None
    m = _REFLECT_RE.search(raw)
    if m:
        cand = _strip_special_tokens(m.group(1))
        # Keep reflection even without Hermes; reject tool-echo bodies.
        if cand and not _looks_like_tool_echo(cand) and len(cand) >= 8:
            # Drop trailing JSON / tool fragments from reflection body.
            cand = re.split(r"<tool_call>|\{", cand, maxsplit=1)[0].strip(" \n\"'")
            if cand and "brightness=" not in cand.lower():
                reflect = cand[:400]

    prompt = None
    for block in _TOOL_CALL_RE.finditer(raw):
        body = block.group(1)
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            pm = _PROMPT_KEY_RE.search(body)
            if pm and _is_usable_diffusion_prompt(pm.group(1), user_task=user_task, prev_prompt=prev_prompt):
                prompt = _strip_special_tokens(pm.group(1))[:500]
                break
            continue
        if isinstance(obj, dict):
            name = str(obj.get("name") or "")
            args = obj.get("arguments") or {}
            if name == DEFAULT_TOOL_NAME and isinstance(args, dict) and args.get("prompt"):
                cand = str(args["prompt"])
                if _is_usable_diffusion_prompt(cand, user_task=user_task, prev_prompt=prev_prompt):
                    prompt = _strip_special_tokens(cand)[:500]
                    break
    if prompt is None:
        pm = _PROMPT_KEY_RE.search(raw)
        if pm and _is_usable_diffusion_prompt(pm.group(1), user_task=user_task, prev_prompt=prev_prompt):
            prompt = _strip_special_tokens(pm.group(1))[:500]
    if prompt is None:
        hm = _REWRITE_HINT_RE.search(raw)
        if hm and _is_usable_diffusion_prompt(hm.group(1), user_task=user_task, prev_prompt=prev_prompt):
            prompt = _strip_special_tokens(hm.group(1))[:500]
    # Caption-like free text (often first call) without Hermes.
    if prompt is None and reflect is None and "<tool_call>" not in raw.lower() and not _looks_like_tool_echo(raw):
        caption = _strip_special_tokens(raw)
        if _is_usable_diffusion_prompt(caption, user_task=user_task, prev_prompt=prev_prompt) and len(caption) <= 400:
            judge_marks = ("looks", "blurry", "should", "need to", "too dark", "missing")
            if not any(j in caption.lower() for j in judge_marks):
                prompt = caption

    return reflect, prompt


def _resolve_force_prompt_and_reflection(
    *,
    decoded: str,
    user_task: str,
    executed: int,
    prev_prompts: list[str],
    variant: int,
    image_path: str | None,
) -> tuple[str, str, dict]:
    """Prefer LLM reflection/prompt; fall back to image-heuristic teacher.

    Returns ``(prompt, reflection, meta)`` with ``meta['content_source']`` in
    ``{llm, mixed, teacher}``.
    """
    prev = (prev_prompts[-1] if prev_prompts else user_task).strip()
    teacher_prompt, teacher_reflect, teacher_meta = _forced_diffusion_prompt(
        user_task=user_task,
        executed=executed,
        prev_prompts=prev_prompts,
        variant=variant,
        image_path=image_path,
    )
    teacher_meta = dict(teacher_meta)
    teacher_meta["content_source"] = "teacher"

    if not _prefer_llm_reflection_enabled():
        return teacher_prompt, teacher_reflect, teacher_meta

    llm_reflect, llm_prompt = _parse_llm_reflection_and_prompt(decoded, user_task=user_task, prev_prompt=prev)
    # Rewrite turns: require prompt to differ from previous when LLM provides one.
    if executed >= 1 and llm_prompt and prev and llm_prompt.lower() == prev.lower():
        llm_prompt = None

    used_llm_reflect = bool(llm_reflect)
    used_llm_prompt = bool(llm_prompt)
    prompt = llm_prompt or teacher_prompt
    reflect = llm_reflect or teacher_reflect

    meta = dict(teacher_meta)
    meta["llm_reflection"] = llm_reflect or ""
    meta["llm_prompt"] = llm_prompt or ""
    meta["teacher_reflection"] = teacher_reflect
    meta["teacher_prompt"] = teacher_prompt
    if used_llm_reflect and used_llm_prompt:
        meta["content_source"] = "llm"
        meta["stage"] = "llm_reflect_rewrite" if executed >= 1 else "llm_initial"
    elif used_llm_reflect or used_llm_prompt:
        meta["content_source"] = "mixed"
        meta["stage"] = "mixed_reflect_rewrite" if executed >= 1 else "mixed_initial"
    else:
        meta["content_source"] = "teacher"
    if used_llm_prompt:
        meta["rewritten_prompt"] = prompt
    return prompt, reflect, meta


def _record_diffusion_prompt(agent_data: AgentData, prompt: str) -> None:
    prompts = agent_data.extra_fields.setdefault("diffusion_prompts", [])
    prompts.append(prompt)
    agent_data.extra_fields["diffusion_prompts"] = prompts


def _teacher_force_replace_assistant_turn(agent_data: AgentData, tokenizer, text: str, *, response_length: int) -> None:
    """Replace the just-generated assistant tokens with ``text`` (mask=1, logprob stubs).

    Used only when the rollout lacks valid Hermes XML (see ``_impose_hermes_on_rollout``).
    Parent ``_handle_generating_state`` already appended model tokens to
    ``prompt_ids`` / ``response_mask``. We pop that span and append the
    imposed Hermes tokens so GRPO sees a valid reflection+tool-call format while
    preserving rollout prompt content whenever possible.
    """
    n_old = len(agent_data.response_ids or [])
    trimmed_logprobs = False
    if n_old > 0:
        agent_data.prompt_ids = agent_data.prompt_ids[:-n_old]
        agent_data.response_mask = agent_data.response_mask[:-n_old]
        if agent_data.response_logprobs and len(agent_data.response_logprobs) >= n_old:
            agent_data.response_logprobs = agent_data.response_logprobs[:-n_old]
            trimmed_logprobs = True

    new_ids = tokenizer.encode(text, add_special_tokens=False)
    remaining = max(0, int(response_length) - len(agent_data.response_mask))
    new_ids = new_ids[:remaining]
    agent_data.response_ids = new_ids
    agent_data.prompt_ids += new_ids
    agent_data.response_mask += [1] * len(new_ids)
    if trimmed_logprobs or agent_data.response_logprobs:
        # Rollout logprobs unknown for teacher tokens; actor recomputes old_log_prob.
        agent_data.response_logprobs += [0.0] * len(new_ids)


def _msg_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _looks_like_tool_echo(text: str) -> bool:
    """Reject model free-text that is just echoing tool observations."""
    t = (text or "").strip().lower()
    if not t:
        return True
    return (
        t.startswith("[tool_response]")
        or "[tool_response]" in t
        or "agentic_tool ok=" in t
        or "lance frozen mot tool" in t
        or t.startswith("path=")
    )


def _looks_like_garbage_assistant(text: str) -> bool:
    """Detect non-Hermes junk that should never enter the training span."""
    t = (text or "").strip()
    if not t:
        return True
    if _looks_like_tool_echo(t):
        return True
    low = t.lower()
    if "```" in t and "generate_image" not in low and "<tool_call>" not in low:
        return True
    if low.startswith("output:") or 'type: "svg"' in low or 'type: "object"' in low:
        return True
    if "<tool_call>" in low:
        return False
    # Long free-text without Hermes is not the PR1 target format.
    return len(t) > 40 and "reflection:" not in low


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the latest real user request (skip tool-response echoes)."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = _msg_text(message.get("content", ""))
        if text and not _looks_like_tool_echo(text):
            return text
    return "a detailed visual scene"


def _task_user_prompt(messages: list[dict[str, Any]]) -> str:
    """Dataset task request for meta.json (last non-echo user message)."""
    return _last_user_text(messages)


def _last_png_path(agent_data: AgentData) -> str | None:
    for path in reversed(list(agent_data.extra_fields.get("image_paths") or [])):
        p = str(path)
        if p.lower().endswith(".png") and Path(p).is_file():
            return p
    return None


def _refine_prompt(base: str, turn_idx: int, *, variant: int = 0) -> str:
    """Fallback refine when no image is available yet (should be rare after turn 1)."""
    base = (base or "").strip()
    suffixes = [
        "highly detailed, sharp focus, coherent composition",
        "improved lighting, richer colors, clearer subject",
        "studio quality, balanced framing, fine texture",
    ]
    idx = (max(turn_idx - 1, 0) + max(int(variant), 0)) % len(suffixes)
    suffix = suffixes[idx]
    if not base:
        return suffix
    if suffix.lower() in base.lower():
        return f"{base}, pass {turn_idx + 1}"
    return f"{base}, {suffix}"


def _forced_diffusion_prompt(
    *,
    user_task: str,
    executed: int,
    prev_prompts: list[str],
    variant: int,
    image_path: str | None = None,
) -> tuple[str, str, dict]:
    """Return ``(diffusion_prompt, reflection_text, reflect_meta)``.

    Turn 0: plan first call from the user request (no image yet).
    Turn 1+: **reflect on the last generated PNG** and rewrite the next prompt
    from that reflection (not a blind suffix).
    """
    task = (user_task or "a detailed visual scene").strip()
    if executed == 0:
        # First call: no image yet — do not invent a fake "Planning..." reflection.
        return task, "", {"stage": "initial", "image_ok": False}

    prev = (prev_prompts[-1] if prev_prompts else task).strip() or task
    reflect, prompt, meta = reflect_on_generated_image(
        image_path=image_path,
        user_task=task,
        prev_prompt=prev,
        variant=variant,
    )
    meta["stage"] = "image_reflect_rewrite"
    # Safety: never send an empty / identical-only prompt.
    if not prompt.strip():
        prompt = _refine_prompt(prev, executed, variant=variant)
    return prompt, reflect, meta


def _rollout_traj_dir() -> Path:
    image_dir = os.getenv("AGENTIC_DIFFUSION_IMAGE_DIR", "").strip()
    if image_dir:
        return Path(image_dir).parent / "rollout_trajectories"
    run = os.getenv("AGENTIC_E2E_RUN_NAME", "").strip() or "default"
    root = os.getenv("AGENTIC_E2E_ROOT", "").strip() or "/tmp/agentic_lance_t2i"
    return Path(root) / run / "rollout_trajectories"


def _hermes_actions_dir() -> Path:
    """``outputs/e2e/<run>/hermes_actions/`` — per-step impose action logs."""
    return _rollout_traj_dir().parent / "hermes_actions"


# Human-readable labels for e2e step dumps (match PR1 overfit curriculum table).
_HERMES_ACTION_LABELS: dict[str, str] = {
    "keep": "Keep tokens (on-policy GRPO)",
    "wrap_json": "Wrap with Hermes XML (content kept)",
    "wrap_caption": "Wrap as Hermes; prompt = rollout caption",
    "teacher": "Teacher template (only fallback)",
    "overfit_teacher": "Stable overfit teacher (fixed gen→reflect→rewrite)",
    "overfit_on_policy": "Overfit mix: keep raw decode (low-reward contrast)",
    "force_only": "Force tool only (teacher impose disabled)",
    "voluntary": "Model emitted Hermes tool_calls (no force)",
    "final_confirm": "Final confirmation (replaced garbage after tools)",
}


def _hermes_action_label(mode: str) -> str:
    return _HERMES_ACTION_LABELS.get(str(mode), f"Unknown mode={mode}")


def _record_hermes_action(
    agent_data: AgentData,
    *,
    mode: str,
    turn: int,
    decoded: str,
    used_text: str,
    forced: bool,
) -> None:
    """Accumulate per-turn Hermes impose decisions for e2e dumps."""
    used_reflection = _reflection_from_decode_only(used_text) or ""
    entry = {
        "turn": int(turn),
        "mode": str(mode),
        "action": _hermes_action_label(mode),
        "forced": bool(forced),
        "reflection": used_reflection,
        "decode_preview": (decoded or "")[:240].replace("\n", "\\n"),
        "used_preview": (used_text or "")[:240].replace("\n", "\\n"),
        "decode_has_tool_call": "<tool_call>" in (decoded or "").lower(),
        "decode_len": len(decoded or ""),
        "used_len": len(used_text or ""),
    }
    actions = agent_data.extra_fields.setdefault("_hermes_actions", [])
    actions.append(entry)
    agent_data.extra_fields["hermes_actions"] = list(actions)


def _dump_hermes_actions(agent_data: AgentData) -> None:
    """Write traj + step-level Hermes action tables under ``outputs/e2e/<run>/``."""
    try:
        actions = list(agent_data.extra_fields.get("_hermes_actions") or [])
        if not actions:
            return
        relpath = _ensure_trajectory_relpath(agent_data)
        step = agent_data.extra_fields.get("_train_step")
        sample_index = agent_data.extra_fields.get("_sample_index")
        rollout_n = agent_data.extra_fields.get("_rollout_n")
        try:
            step_i = int(step) if step is not None else -1
        except (TypeError, ValueError):
            step_i = -1
        step_tag = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"

        # Per-trajectory sidecar next to rollout_trajectories.
        traj_dir = _rollout_traj_dir() / Path(relpath).parent
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_base = traj_dir / Path(relpath).name
        traj_payload = {
            "trajectory_relpath": relpath,
            "request_id": agent_data.request_id,
            "step": step,
            "sample_index": sample_index,
            "rollout_n": rollout_n,
            "hermes_actions": actions,
        }
        traj_base.with_name(traj_base.name + ".hermes_actions.json").write_text(
            json.dumps(traj_payload, indent=2, ensure_ascii=False) + "\n"
        )

        # Step-level append log: one block per rollout under hermes_actions/step_XXXXXX.txt
        step_dir = _hermes_actions_dir()
        step_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"=== {Path(relpath).name}  sample={sample_index} rollout_n={rollout_n} ===",
            f"time={time.strftime('%Y-%m-%dT%H:%M:%S')}",
            "Rollout decode → Action",
        ]
        for a in actions:
            lines.append(
                f"  turn={a['turn']}  mode={a['mode']}"
                f"  forced={a['forced']}  decode_has_tool_call={a['decode_has_tool_call']}"
            )
            lines.append(f"    Action: {a['action']}")
            if a.get("reflection"):
                lines.append(f"    Reflection: {a['reflection']}")
            lines.append(f"    Decode: {a['decode_preview']}")
            if a.get("used_preview") and a["used_preview"] != a["decode_preview"]:
                lines.append(f"    Used:   {a['used_preview']}")
        lines.append("")
        step_path = step_dir / f"{step_tag}.txt"
        with step_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        jsonl_path = step_dir / f"{step_tag}.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "trajectory_relpath": relpath,
                        "request_id": agent_data.request_id,
                        "step": step,
                        "sample_index": sample_index,
                        "rollout_n": rollout_n,
                        "hermes_actions": actions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        logger.info(
            "hermes_actions step=%s traj=%s modes=%s -> %s",
            step_tag,
            Path(relpath).name,
            [a["mode"] for a in actions],
            step_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to dump hermes actions: %s", exc)


def _rollout_text_dir() -> Path:
    return _rollout_traj_dir().parent / "rollout_texts"


def _msg_content_as_str(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _resolve_train_step(agent_data: AgentData) -> int | None:
    """Prefer stock vLLM generate ``global_steps`` / ``max_global_steps``."""
    for key in ("_train_step", "global_steps", "max_global_steps", "min_global_steps"):
        raw = agent_data.extra_fields.get(key)
        if raw is None:
            continue
        try:
            step_i = int(raw)
        except (TypeError, ValueError):
            continue
        if step_i >= 0:
            agent_data.extra_fields["_train_step"] = step_i
            return step_i
    return None


def _ensure_trajectory_relpath(agent_data: AgentData) -> str:
    """Stable path: ``step_*/sample_{index}.{rollout_n:02d}``."""
    existing = agent_data.extra_fields.get("_trajectory_relpath") or agent_data.extra_fields.get("trajectory_relpath")
    step = _resolve_train_step(agent_data)
    if existing and not str(existing).startswith("step_unknown/"):
        return str(existing)
    if existing and step is None:
        return str(existing)
    # Upgrade provisional step_unknown paths once train step is known.
    if existing and str(existing).startswith("step_unknown/") and step is not None:
        agent_data.extra_fields.pop("_rollout_n", None)

    sample_index = agent_data.extra_fields.get("_sample_index")
    rollout_n = agent_data.extra_fields.get("_rollout_n")
    if rollout_n is None:
        rollout_n = allocate_rollout_n(
            artifacts_root=_rollout_traj_dir(),
            step=step,
            sample_index=sample_index,
        )
        agent_data.extra_fields["_rollout_n"] = int(rollout_n)

    relpath = build_trajectory_relpath(
        step=step,
        sample_index=sample_index,
        rollout_n=int(rollout_n),
    )
    agent_data.extra_fields["_trajectory_relpath"] = relpath
    agent_data.extra_fields["trajectory_relpath"] = relpath
    agent_data.extra_fields["_trajectory_name"] = Path(relpath).name
    return relpath


def _bind_artifact_context(agent_data: AgentData) -> str:
    relpath = _ensure_trajectory_relpath(agent_data)
    set_active_trajectory_relpath(relpath)
    user_prompt = agent_data.extra_fields.get("user_prompt") or _task_user_prompt(agent_data.messages)
    agent_data.extra_fields["user_prompt"] = user_prompt
    set_active_user_prompt(user_prompt)
    return relpath


def _conversation_init(agent_data: AgentData) -> None:
    """Seed a mutable conversation log from the dataset prompt (incl. few-shot)."""
    if "_rollout_conversation" in agent_data.extra_fields:
        # Re-bind contextvars for this asyncio task (worker may reuse the loop).
        _bind_artifact_context(agent_data)
        return
    seed = []
    for message in agent_data.messages:
        seed.append(
            {
                "role": message.get("role"),
                "content": _msg_content_as_str(message.get("content")),
                "source": "seed",
            }
        )
    agent_data.extra_fields["_rollout_conversation"] = seed
    # Capture task user request from seed messages before rollout appends.
    agent_data.extra_fields["user_prompt"] = _task_user_prompt(agent_data.messages)
    # Path bind waits for train step from the first generate when possible.
    if _resolve_train_step(agent_data) is not None:
        _bind_artifact_context(agent_data)


def _conversation_append(agent_data: AgentData, role: str, content: str, **meta: Any) -> None:
    _conversation_init(agent_data)
    entry = {"role": role, "content": content, "source": "rollout"}
    entry.update({k: v for k, v in meta.items() if v is not None})
    agent_data.extra_fields["_rollout_conversation"].append(entry)


def _dump_full_conversation(agent_data: AgentData) -> None:
    """Write the whole multi-turn chat once per trajectory (JSON + readable txt)."""
    try:
        _conversation_init(agent_data)
        conv = list(agent_data.extra_fields.get("_rollout_conversation") or [])
        if not conv:
            return
        # Avoid double-dump if terminate is hit twice.
        if agent_data.extra_fields.get("_rollout_dumped"):
            return
        agent_data.extra_fields["_rollout_dumped"] = True

        relpath = _ensure_trajectory_relpath(agent_data)
        out_dir = _rollout_traj_dir() / Path(relpath).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / Path(relpath).name

        image_root = Path(
            os.getenv("AGENTIC_DIFFUSION_IMAGE_DIR", "").strip() or (_rollout_traj_dir().parent / "rollout_images")
        )
        image_dir_for_traj = image_root / relpath

        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": getattr(agent_data, "request_id", ""),
            "trajectory_relpath": relpath,
            "trajectory_name": Path(relpath).name,
            "sample_id": agent_data.extra_fields.get("_sample_uid") or agent_data.extra_fields.get("_sample_index"),
            "step": agent_data.extra_fields.get("_train_step"),
            "sample_index": agent_data.extra_fields.get("_sample_index"),
            "rollout_n": agent_data.extra_fields.get("_rollout_n"),
            "user_prompt": agent_data.extra_fields.get("user_prompt") or "",
            "image_dir": str(image_dir_for_traj),
            "assistant_turns": agent_data.assistant_turns,
            "user_turns": agent_data.user_turns,
            "forced_tool_call": bool(agent_data.extra_fields.get("forced_tool_call")),
            "num_forced_tool_calls": int(agent_data.extra_fields.get("num_forced_tool_calls") or 0),
            "num_tool_calls_executed": int(agent_data.extra_fields.get("num_tool_calls_executed") or 0),
            "num_successful_images": int(agent_data.extra_fields.get("num_successful_images") or 0),
            "num_voluntary_hermes": int(agent_data.extra_fields.get("num_voluntary_hermes") or 0),
            "num_voluntary_successful_images": int(agent_data.extra_fields.get("num_voluntary_successful_images") or 0),
            "diffusion_prompts": list(agent_data.extra_fields.get("diffusion_prompts") or []),
            "image_paths": list(agent_data.extra_fields.get("image_paths") or []),
            "reflections": list(agent_data.extra_fields.get("reflections") or []),
            "hermes_actions": list(agent_data.extra_fields.get("_hermes_actions") or []),
            "messages": conv,
        }
        json_path = base.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

        lines = [
            f"relpath={relpath}",
            f"assistant_turns={payload['assistant_turns']} user_turns={payload['user_turns']}",
            f"forced={payload['forced_tool_call']} "
            f"num_forced={payload['num_forced_tool_calls']} "
            f"tool_exec={payload['num_tool_calls_executed']} "
            f"ok_images={payload['num_successful_images']}",
            "hermes_actions:",
        ]
        for a in payload["hermes_actions"]:
            lines.append(f"  turn={a.get('turn')} mode={a.get('mode')} → {a.get('action')}")
        lines.append("image_paths:")
        for p in payload["image_paths"]:
            lines.append(f"  - {p}")
        lines.append("---")
        for i, msg in enumerate(conv):
            role = msg.get("role", "?")
            src = msg.get("source", "")
            header = f"[{i}] {role}" + (f" ({src})" if src else "")
            if msg.get("forced"):
                header += " [FORCED_TOOL]"
            if msg.get("hermes_impose_mode"):
                mode = str(msg.get("hermes_impose_mode"))
                header += f" hermes={mode} ({_hermes_action_label(mode)})"
            if msg.get("content_source"):
                header += f" content_source={msg.get('content_source')}"
            lines.append(header)
            raw = msg.get("model_decode")
            if (
                raw is not None
                and str(raw).strip()
                and str(raw).strip() != _msg_content_as_str(msg.get("content")).strip()
            ):
                lines.append("--- model_decode (raw Lance und) ---")
                lines.append(str(raw))
                lines.append("--- used_for_training_and_tool ---")
            lines.append(_msg_content_as_str(msg.get("content")))
            if msg.get("image_paths"):
                lines.append("image_paths: " + ", ".join(msg["image_paths"]))
            lines.append("")
        base.with_suffix(".txt").write_text("\n".join(lines))
        agent_data.extra_fields["trajectory_path"] = str(json_path)
        agent_data.extra_fields["trajectory_relpath"] = relpath
        _dump_hermes_actions(agent_data)
        logger.info("saved full multi-turn trajectory -> %s", json_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to dump full conversation: %s", exc)


def _dump_assistant_text(text: str, *, forced: bool, turn: int) -> None:
    """Optional per-turn scrap (off by default; full traj is preferred)."""
    if os.getenv("AGENTIC_DUMP_ASSISTANT_TURNS", "0").strip().lower() not in {"1", "true", "yes"}:
        return
    try:
        out_dir = _rollout_text_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"asst_{stamp}_t{turn}_{uuid.uuid4().hex[:8]}.txt"
        path.write_text(
            f"forced_tool={forced}\nassistant_turn={turn}\ntime={time.strftime('%Y-%m-%dT%H:%M:%S')}\n---\n{text}\n"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to dump rollout text: %s", exc)


def _init_trajectory_fields(agent_data: AgentData) -> None:
    """Keys must exist on every sample so DataProto.concat stays aligned."""
    agent_data.extra_fields.setdefault("forced_tool_call", False)
    agent_data.extra_fields.setdefault("num_forced_tool_calls", 0)
    agent_data.extra_fields.setdefault("num_tool_calls_executed", 0)
    agent_data.extra_fields.setdefault("num_successful_images", 0)
    agent_data.extra_fields.setdefault("num_voluntary_hermes", 0)
    agent_data.extra_fields.setdefault("num_voluntary_successful_images", 0)
    agent_data.extra_fields.setdefault("diffusion_prompts", [])
    agent_data.extra_fields.setdefault("reflections", [])
    agent_data.extra_fields.setdefault("image_reflection_meta", [])
    agent_data.extra_fields.setdefault("hermes_actions", [])
    agent_data.extra_fields.setdefault("hermes_impose_mode", "none")
    agent_data.extra_fields.setdefault("_hermes_actions", [])
    agent_data.extra_fields.setdefault("image_paths", [])
    agent_data.extra_fields.setdefault("trajectory_path", "")
    agent_data.extra_fields.setdefault("trajectory_relpath", "")
    agent_data.extra_fields.setdefault("user_prompt", "")
    agent_data.metrics.setdefault("forced_tool_call", 0.0)
    agent_data.metrics.setdefault("num_forced_tool_calls", 0.0)
    agent_data.metrics.setdefault("num_tool_calls_executed", 0.0)
    agent_data.metrics.setdefault("num_successful_images", 0.0)
    agent_data.metrics.setdefault("num_voluntary_hermes", 0.0)
    agent_data.metrics.setdefault("num_voluntary_successful_images", 0.0)
    agent_data.metrics.setdefault("force_probability", 1.0)
    agent_data.metrics.setdefault("image_reflection", 0.0)


def _stamp_output_extra_fields(extra_fields: dict[str, Any]) -> None:
    """Guarantee concat-safe keys on the AgentLoopOutput (parent may drop unset)."""
    # Promote private hermes log before stripping `_` bookkeeping.
    if "_hermes_actions" in extra_fields and "hermes_actions" not in extra_fields:
        extra_fields["hermes_actions"] = list(extra_fields.get("_hermes_actions") or [])
    # Drop private bookkeeping so worker DataProto.concat keys stay aligned.
    for key in [k for k in list(extra_fields) if str(k).startswith("_")]:
        extra_fields.pop(key, None)
    extra_fields.setdefault("forced_tool_call", False)
    extra_fields.setdefault("num_forced_tool_calls", 0)
    extra_fields.setdefault("num_tool_calls_executed", 0)
    extra_fields.setdefault("num_successful_images", 0)
    extra_fields.setdefault("num_voluntary_hermes", 0)
    extra_fields.setdefault("num_voluntary_successful_images", 0)
    extra_fields.setdefault("diffusion_prompts", [])
    extra_fields.setdefault("reflections", [])
    extra_fields.setdefault("image_reflection_meta", [])
    extra_fields.setdefault("hermes_actions", [])
    extra_fields.setdefault("hermes_impose_mode", "none")
    extra_fields.setdefault("image_paths", [])
    extra_fields.setdefault("trajectory_path", "")
    extra_fields.setdefault("trajectory_relpath", "")
    extra_fields.setdefault("user_prompt", "")


def _capture_sample_id(agent_data: AgentData, kwargs: dict[str, Any]) -> None:
    """Keep dataset index for ``sample_{index}.{n}`` paths; uid is optional metadata."""
    if "index" in kwargs:
        agent_data.extra_fields["_sample_index"] = kwargs["index"]
    if "uid" in kwargs:
        agent_data.extra_fields["_sample_uid"] = str(kwargs["uid"])
    elif "_sample_index" not in agent_data.extra_fields and "index" not in kwargs:
        # Fallback only when dataset has neither field.
        pass


@register("agentic_force_tool_agent")
class AgenticForceToolAgentLoop(ToolAgentLoop):
    """Stock ToolAgentLoop + force multi-turn ``generate_image`` until min calls."""

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        self._pending_sample_kwargs = {k: kwargs[k] for k in ("uid", "index") if k in kwargs}
        try:
            output = await super().run(sampling_params, **kwargs)
            _stamp_output_extra_fields(output.extra_fields)
            return output
        finally:
            # Avoid leaking traj binding across asyncio tasks on the same worker.
            set_active_trajectory_relpath(None)
            set_active_user_prompt(None)
            set_active_call_provenance(None)
            self._pending_sample_kwargs = {}

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        _init_trajectory_fields(agent_data)
        _capture_sample_id(agent_data, getattr(self, "_pending_sample_kwargs", {}) or {})
        _conversation_init(agent_data)
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        """Capture untruncated image_paths from diffusion_tool metrics (before obs truncate)."""
        # Bind step/sample path before the tool writes images.
        _bind_artifact_context(agent_data)
        response, reward, metrics = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        paths: list[str] = []
        if isinstance(metrics, dict):
            raw = metrics.get("image_paths") or []
            if isinstance(raw, list):
                paths = [str(p) for p in raw if p]
            elif raw:
                paths = [str(raw)]
        queue = agent_data.extra_fields.setdefault("_pending_tool_image_paths", [])
        queue.append(paths)
        all_paths = agent_data.extra_fields.setdefault("image_paths", [])
        for p in paths:
            if p.endswith(".png"):
                all_paths.append(p)
        return response, reward, metrics

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        n_before = len(agent_data.tool_rewards)
        n_msgs_before = len(agent_data.messages)
        n_png_before = len([p for p in (agent_data.extra_fields.get("image_paths") or []) if str(p).endswith(".png")])
        this_turn_forced = bool(agent_data.extra_fields.pop("_this_turn_forced", False))
        agent_data.extra_fields["_pending_tool_image_paths"] = []
        state = await super()._handle_processing_tools_state(agent_data)
        pending = list(agent_data.extra_fields.get("_pending_tool_image_paths") or [])
        tool_msgs = [m for m in agent_data.messages[n_msgs_before:] if m.get("role") == "tool"]
        for i, msg in enumerate(tool_msgs):
            paths = pending[i] if i < len(pending) else []
            _conversation_append(
                agent_data,
                "tool",
                _msg_content_as_str(msg.get("content")),
                image_paths=paths or None,
            )
        n_new = max(0, len(agent_data.tool_rewards) - n_before)
        if n_new:
            executed = int(agent_data.extra_fields.get("num_tool_calls_executed") or 0) + n_new
            agent_data.extra_fields["num_tool_calls_executed"] = executed
            agent_data.metrics["num_tool_calls_executed"] = float(executed)
            pngs = [p for p in (agent_data.extra_fields.get("image_paths") or []) if str(p).endswith(".png")]
            agent_data.extra_fields["num_successful_images"] = len(pngs)
            agent_data.metrics["num_successful_images"] = float(len(pngs))
            new_pngs = max(0, len(pngs) - n_png_before)
            if new_pngs and not this_turn_forced:
                vol_img = int(agent_data.extra_fields.get("num_voluntary_successful_images") or 0) + new_pngs
                agent_data.extra_fields["num_voluntary_successful_images"] = vol_img
                agent_data.metrics["num_voluntary_successful_images"] = float(vol_img)
        return state

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        _init_trajectory_fields(agent_data)
        _conversation_init(agent_data)
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        # Stock generate attaches global_steps; finalize step_*/sample_*.** paths here.
        _resolve_train_step(agent_data)
        _bind_artifact_context(agent_data)

        turn = agent_data.assistant_turns
        decoded = ""
        if agent_data.response_ids:
            try:
                decoded = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
            except Exception:  # noqa: BLE001
                decoded = "<decode failed>"

        # Count model-emitted Hermes generate_image before any force injection.
        if agent_data.tool_calls:
            n_vol = sum(1 for tc in agent_data.tool_calls if getattr(tc, "name", "") == DEFAULT_TOOL_NAME)
            if n_vol:
                total = int(agent_data.extra_fields.get("num_voluntary_hermes") or 0) + n_vol
                agent_data.extra_fields["num_voluntary_hermes"] = total
                agent_data.metrics["num_voluntary_hermes"] = float(total)
                _record_hermes_action(
                    agent_data,
                    mode="voluntary",
                    turn=turn,
                    decoded=decoded,
                    used_text=decoded,
                    forced=False,
                )
            for tc in agent_data.tool_calls:
                if getattr(tc, "name", "") != DEFAULT_TOOL_NAME:
                    continue
                p = _extract_tool_prompt(tc)
                if p:
                    _record_diffusion_prompt(agent_data, p)

        executed = int(agent_data.extra_fields.get("num_tool_calls_executed") or 0)
        min_calls = _min_tool_calls()
        need_more = executed < min_calls
        already_forced = int(agent_data.extra_fields.get("num_forced_tool_calls") or 0) > 0
        force_prob = _effective_force_probability(
            executed=executed,
            already_forced=already_forced,
        )
        agent_data.metrics["force_probability"] = float(force_prob)

        will_force = (
            _force_enabled()
            and need_more
            and not agent_data.tool_calls
            and state == AgentState.TERMINATED
            # Allow force even at response_length: teacher replace truncates to fit.
            # Blocking here let spam-to-max-length escape tools (lazy score=0 path).
            and DEFAULT_TOOL_NAME in getattr(agent_data, "_active_tools", self.tools)
            and (not self.max_assistant_turns or turn < self.max_assistant_turns)
            and random.random() < force_prob
        )

        # Record live assistant turn (stock ToolAgentLoop never puts these in messages).
        asst_content = decoded
        force_args = None
        if will_force:
            user_task = _last_user_text(agent_data.messages)
            prev_prompts = list(agent_data.extra_fields.get("diffusion_prompts") or [])
            try:
                variant = int(agent_data.extra_fields.get("_rollout_n") or 0)
            except (TypeError, ValueError):
                variant = 0
            last_png = _last_png_path(agent_data) if executed >= 1 else None
            if _overfit_stable_teacher_enabled():
                # Stable fewshot-like target — do not wrap garbage captions into GRPO tokens.
                prompt, reflect, reflect_meta = _stable_overfit_prompt_and_reflection(
                    user_task=user_task,
                    executed=executed,
                    prev_prompts=prev_prompts,
                )
                synth = _hermes_generate_image_text(prompt, reflect=reflect if executed >= 1 else None)
                hermes_mode = "overfit_teacher"
                tool_prompt = prompt
            else:
                # Prefer Lance und Reflection + rewrite prompt; teacher heuristics only as fallback.
                prompt, reflect, reflect_meta = _resolve_force_prompt_and_reflection(
                    decoded=decoded,
                    user_task=user_task,
                    executed=executed,
                    prev_prompts=prev_prompts,
                    variant=variant,
                    image_path=last_png,
                )
                # Always classify/wrap via impose (never dump identical teacher Reflection
                # onto bare-JSON rollouts). TEACHER_FORCE only controls token replace.
                synth, hermes_mode, tool_prompt = _impose_hermes_on_rollout(
                    decoded,
                    prompt=prompt,
                    reflect=reflect,
                    require_reflection=executed >= 1,
                )
            # Tool + training span share the rollout prompt for wrap/keep.
            if hermes_mode in {"keep", "wrap_json", "wrap_caption"} and tool_prompt:
                prompt = tool_prompt
            force_args = {"name": DEFAULT_TOOL_NAME, "arguments": {"prompt": prompt}}
            asst_content = synth
            agent_data.metrics["hermes_impose_mode"] = {
                "keep": 0.0,
                "wrap_json": 0.25,
                "wrap_caption": 0.5,
                "teacher": 1.0,
                "overfit_teacher": 1.0,
                "overfit_on_policy": 0.0,
            }.get(hermes_mode, 1.0)
            do_token_replace = _teacher_force_hermes_enabled() and hermes_mode != "keep"
            # Overfit mix: leave some forced turns on raw decode so GRPO gets contrast.
            if do_token_replace and _overfit_stable_teacher_enabled() and random.random() < _overfit_on_policy_frac():
                do_token_replace = False
                hermes_mode = "overfit_on_policy"
                asst_content = decoded  # log honesty; tools still use stable prompt
                agent_data.metrics["hermes_impose_mode"] = 0.0
                agent_data.metrics["hermes_token_replace"] = 0.0
            if do_token_replace:
                _teacher_force_replace_assistant_turn(
                    agent_data, self.tokenizer, synth, response_length=self.response_length
                )
            elif hermes_mode == "keep":
                agent_data.metrics["hermes_kept_on_policy"] = 1.0
            else:
                # Still record impose mode for e2e logs; tokens stay as raw decode.
                agent_data.metrics["hermes_token_replace"] = 0.0
            _record_hermes_action(
                agent_data,
                mode=hermes_mode,
                turn=turn,
                decoded=decoded,
                used_text=asst_content if hermes_mode != "overfit_on_policy" else synth,
                forced=True,
            )
            _record_diffusion_prompt(agent_data, prompt)
            reflections = agent_data.extra_fields.setdefault("reflections", [])
            # Log the reflection that actually entered the training/tool span.
            used_reflect = _reflection_from_decode_only(synth) if hermes_mode != "teacher" else reflect
            reflections.append(used_reflect or "")
            agent_data.extra_fields.setdefault("image_reflection_meta", []).append(reflect_meta)
            agent_data.extra_fields["hermes_impose_mode"] = hermes_mode
            agent_data.metrics["image_reflection"] = 1.0 if reflect_meta.get("image_ok") else 0.0
            agent_data.metrics["llm_reflection_used"] = (
                1.0
                if reflect_meta.get("content_source") == "llm"
                else (0.5 if reflect_meta.get("content_source") == "mixed" else 0.0)
            )
            prev_tool = prev_prompts[-1] if prev_prompts else ""
            controlled = executed >= 1
            if hermes_mode in {"wrap_json", "wrap_caption", "keep"}:
                provenance_source = "rollout_wrap"
            elif hermes_mode == "overfit_on_policy":
                provenance_source = "overfit_on_policy"
            elif hermes_mode == "overfit_teacher":
                provenance_source = "overfit_teacher"
            else:
                provenance_source = reflect_meta.get("content_source", "teacher")
            set_active_call_provenance(
                {
                    "call_role": "reflection_rewrite" if controlled else "initial",
                    "controlled_by_reflection": controlled,
                    "reflection": used_reflect or "",
                    "prev_tool_prompt": prev_tool,
                    "source_image": last_png or "",
                    "rewritten_prompt": prompt if controlled else "",
                    "content_source": provenance_source,
                    "hermes_impose_mode": hermes_mode,
                    "hermes_action": _hermes_action_label(hermes_mode),
                    "llm_reflection": reflect_meta.get("llm_reflection", ""),
                    "llm_prompt": reflect_meta.get("llm_prompt", ""),
                    "model_decode": decoded,
                }
            )
        elif (
            state == AgentState.TERMINATED
            and executed >= min_calls
            and _teacher_force_hermes_enabled()
            and _looks_like_garbage_assistant(decoded)
        ):
            # Final turn after tools: replace tool-echo / SVG junk with a short confirmation.
            user_task = _last_user_text(agent_data.messages)
            confirm = _final_confirmation_text(user_task)
            asst_content = confirm
            _teacher_force_replace_assistant_turn(
                agent_data, self.tokenizer, confirm, response_length=self.response_length
            )
            _record_hermes_action(
                agent_data,
                mode="final_confirm",
                turn=turn,
                decoded=decoded,
                used_text=confirm,
                forced=False,
            )

        content_source = None
        hermes_mode_for_log = None
        model_decode_to_log = decoded if (decoded or will_force) else None
        if will_force:
            content_source = (agent_data.extra_fields.get("image_reflection_meta") or [{}])[-1].get(
                "content_source", "teacher"
            )
            hermes_mode_for_log = agent_data.extra_fields.get("hermes_impose_mode")

        if decoded or will_force or asst_content != decoded:
            _conversation_append(
                agent_data,
                "assistant",
                asst_content,
                forced=bool(will_force),
                assistant_turn=turn,
                forced_tool_args=force_args,
                diffusion_prompt=(force_args or {}).get("arguments", {}).get("prompt") if force_args else None,
                reflection=(agent_data.extra_fields.get("reflections") or [None])[-1] if will_force else None,
                model_decode=model_decode_to_log,
                content_source=content_source,
                hermes_impose_mode=hermes_mode_for_log,
                hermes_action=_hermes_action_label(str(hermes_mode_for_log)) if hermes_mode_for_log else None,
            )
            _dump_assistant_text(decoded or asst_content, forced=will_force, turn=turn)

        if not will_force:
            if state == AgentState.TERMINATED:
                _dump_full_conversation(agent_data)
            return state

        assert force_args is not None
        agent_data.tool_calls = [
            FunctionCall(
                name=DEFAULT_TOOL_NAME,
                arguments=json.dumps(force_args["arguments"], ensure_ascii=False),
            )
        ]
        agent_data.extra_fields["_this_turn_forced"] = True
        n_forced = int(agent_data.extra_fields.get("num_forced_tool_calls") or 0) + 1
        agent_data.extra_fields["forced_tool_call"] = True
        agent_data.extra_fields["num_forced_tool_calls"] = n_forced
        agent_data.metrics["forced_tool_call"] = 1.0
        agent_data.metrics["num_forced_tool_calls"] = float(n_forced)
        logger.info(
            "forcing %s (#%d, executed=%d/%d, force_p=%.3f) reflect+hermes prompt=%r",
            DEFAULT_TOOL_NAME,
            n_forced,
            executed,
            min_calls,
            force_prob,
            force_args["arguments"]["prompt"],
        )
        return AgentState.PROCESSING_TOOLS
