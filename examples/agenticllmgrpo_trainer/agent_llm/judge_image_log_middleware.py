#!/usr/bin/env python3
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
"""vLLM ASGI middleware: print parsed image-judge C/A scores per response.

Wire via ``vllm serve ... --middleware judge_image_log_middleware.judge_score_log_middleware``.
Keeps the sidecar tmux pane useful: access logs alone only show ``200 OK``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from starlette.responses import Response

logger = logging.getLogger("judge_image")

_USER_REQUEST_RE = re.compile(r"User request:\s*(.+?)(?:\nDiffusion prompt|\nNotes:|\n\n)", re.I | re.S)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _mean(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return sum(scores.values()) / float(len(scores))


def _parse_scores(text: str) -> dict[str, Any] | None:
    """Prefer shared ``parse_judge_json``; fall back to a tiny local parser."""
    try:
        from verl_omni.utils.agentic_image_judge_parse import parse_judge_json

        return parse_judge_json(text)
    except Exception:  # noqa: BLE001
        pass

    blob = re.sub(r"<think>[\s\S]*?</think>", " ", text or "", flags=re.I)
    blob = re.sub(r"```(?:json)?\s*", "", blob, flags=re.I).replace("```", "").strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(blob):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(blob[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        c_raw = data.get("correctness_scores") if isinstance(data.get("correctness_scores"), dict) else {}
        a_raw = data.get("aesthetics_scores") if isinstance(data.get("aesthetics_scores"), dict) else {}
        c_scores = {str(k): _safe_float(v) for k, v in c_raw.items() if isinstance(v, int | float)}
        a_scores = {str(k): _safe_float(v) for k, v in a_raw.items() if isinstance(v, int | float)}
        if c_scores and a_scores:
            return {
                "correctness": _mean(c_scores),
                "aesthetics": _mean(a_scores),
                "correctness_scores": c_scores,
                "aesthetics_scores": a_scores,
                "findings": str(data.get("findings") or ""),
                "good_enough": False,
            }
        if "correctness" in data or "aesthetics" in data:
            return {
                "correctness": _safe_float(data.get("correctness", 0.0)),
                "aesthetics": _safe_float(data.get("aesthetics", 0.0)),
                "correctness_scores": {},
                "aesthetics_scores": {},
                "findings": str(data.get("findings") or ""),
                "good_enough": False,
            }
    return None


def _good_enough(parsed: dict[str, Any]) -> bool:
    thr_raw = os.getenv("AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD") or os.getenv("AGENTIC_REFLECT_GOOD_ENOUGH") or "0.80"
    try:
        thr = float(thr_raw)
    except ValueError:
        thr = 0.80
    if "good_enough" in parsed and parsed.get("correctness_scores") and parsed.get("aesthetics_scores"):
        # Prefer shared normalize path's YES when available.
        try:
            from verl_omni.utils.agentic_image_judge_parse import normalize_judge_payload

            norm = normalize_judge_payload(
                {
                    "correctness_scores": parsed.get("correctness_scores") or {},
                    "aesthetics_scores": parsed.get("aesthetics_scores") or {},
                    "findings": parsed.get("findings") or "",
                }
            )
            if norm is not None:
                return bool(norm["good_enough"])
        except Exception:  # noqa: BLE001
            pass
    return float(parsed.get("correctness", 0.0)) >= thr and float(parsed.get("aesthetics", 0.0)) >= thr


def _fmt_facets(scores: dict[str, float], *, limit: int = 5) -> str:
    if not scores:
        return "-"
    items = sorted(scores.items(), key=lambda kv: kv[1])[:limit]
    return " ".join(f"{k}={v:.2f}" for k, v in items)


def format_judge_log_line(
    *,
    parsed: dict[str, Any] | None,
    user_snip: str = "",
    latency_hint: str = "",
) -> str:
    """One-line monitor summary for sidecar stdout."""
    if parsed is None:
        base = "[judge_image] parse_ok=0"
        if user_snip:
            base += f" user={user_snip!r}"
        return base
    yes = bool(parsed.get("good_enough")) if "rubber_stamp" in parsed else _good_enough(parsed)
    c = float(parsed.get("correctness", 0.0))
    a = float(parsed.get("aesthetics", 0.0))
    stamp = 1 if parsed.get("rubber_stamp") else 0
    findings = re.sub(r"\s+", " ", str(parsed.get("findings") or "")).strip()[:220]
    parts = [
        "[judge_image]",
        "parse_ok=1",
        f"C={c:.2f}",
        f"A={a:.2f}",
        f"rubber_stamp={stamp}",
        f"good_enough={'YES' if yes else 'NO'}",
    ]
    if latency_hint:
        parts.append(latency_hint)
    if user_snip:
        parts.append(f"user={user_snip!r}")
    c_low = _fmt_facets(dict(parsed.get("correctness_scores") or {}))
    a_low = _fmt_facets(dict(parsed.get("aesthetics_scores") or {}))
    parts.append(f"C_low[{c_low}]")
    parts.append(f"A_low[{a_low}]")
    if findings:
        parts.append(f"findings={findings!r}")
    return " ".join(parts)


def format_judge_parse_fail_line(*, user_snip: str = "", raw: str = "") -> str:
    """parse_ok=0 line plus a short raw head (helps spot thinking/truncation)."""
    base = format_judge_log_line(parsed=None, user_snip=user_snip)
    snip = re.sub(r"\s+", " ", (raw or "").strip())[:160]
    if snip:
        base += f" raw={snip!r}"
    return base


def _user_snip_from_request_body(body: bytes) -> str:
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return ""
    messages = payload.get("messages") or []
    for message in messages:
        content = message.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text") or ""))
        blob = "\n".join(texts)
        match = _USER_REQUEST_RE.search(blob)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()[:100]
        if blob.strip():
            return re.sub(r"\s+", " ", blob).strip()[:100]
    return ""


def _assistant_text(response_body: bytes) -> str:
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError:
        return ""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _judge_enable_thinking_from_env() -> bool:
    return os.getenv("AGENTIC_JUDGE_ENABLE_THINKING", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rewrite_chat_body_disable_thinking(body: bytes) -> bytes:
    """Force ``chat_template_kwargs.enable_thinking`` so old trainer clients still work.

    Qwen3.5 burns ``max_tokens`` on CoT and never emits JSON → parse_ok=0. Client
    may omit the flag until restarted; the sidecar enforces the default here.
    """
    if not body:
        return body
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if not isinstance(payload, dict):
        return body
    enable = _judge_enable_thinking_from_env()
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    if kwargs.get("enable_thinking") is enable:
        return body
    kwargs = dict(kwargs)
    kwargs["enable_thinking"] = enable
    payload["chat_template_kwargs"] = kwargs
    return json.dumps(payload).encode()


async def judge_score_log_middleware(request, call_next):
    """Starlette/FastAPI HTTP middleware entrypoint for ``vllm serve --middleware``."""
    from starlette.requests import Request

    path = str(request.url.path).rstrip("/")
    is_chat = path.endswith("/chat/completions")
    req_body = b""
    if is_chat and str(request.method).upper() == "POST":
        req_body = await request.body()
        req_body = _rewrite_chat_body_disable_thinking(req_body)

        async def receive():
            return {"type": "http.request", "body": req_body, "more_body": False}

        request = Request(request.scope, receive)

    response = await call_next(request)
    if not is_chat or int(response.status_code) != 200:
        return response

    resp_body = b""
    async for chunk in response.body_iterator:
        resp_body += chunk

    try:
        user_snip = _user_snip_from_request_body(req_body)
        raw = _assistant_text(resp_body)
        parsed = _parse_scores(raw) if raw else None
        if parsed is None:
            line = format_judge_parse_fail_line(user_snip=user_snip, raw=raw)
        else:
            line = format_judge_log_line(parsed=parsed, user_snip=user_snip)
        # print + logger: uvicorn may filter logger names; print hits the tmux pane.
        print(line, flush=True)
        logger.info("%s", line)
    except Exception as exc:  # noqa: BLE001
        print(f"[judge_image] log_failed err={exc}", flush=True)

    headers = {
        key.decode() if isinstance(key, bytes) else str(key): (
            value.decode() if isinstance(value, bytes) else str(value)
        )
        for key, value in response.raw_headers
        if key.lower() not in {b"content-length", b"transfer-encoding"}
    }
    return Response(
        content=resp_body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
        background=response.background,
    )
