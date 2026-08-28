#!/usr/bin/env bash
# Visualize verl torch Chrome traces from profile_agenticrpco_grpo.sh.
#
# Traces land at:
#   outputs/profile/e2e/prof_rank-{0,1}_*.json.gz
# These are actor-rank GPU/CPU Chrome traces (update_actor / old_log_prob).
# They do NOT contain sidecar generate_image / judge_image timelines —
# those live in trainer timing_s + rl-insight Tempo, not here.
#
# Modes:
#   summarize  Write a compact HTML bubble report (cached next to the .gz).
#   serve      Load the trace into Perfetto via local trace_processor.
#   both       summarize, then serve (default).
#
# Usage:
#   bash examples/agenticllmgrpo_trainer/agent_llm/vis_profile_trace.sh
#   bash .../vis_profile_trace.sh outputs/profile/e2e
#   bash .../vis_profile_trace.sh outputs/profile/e2e/prof_rank-0_....json.gz
#   RANK=1 MODE=summarize bash .../vis_profile_trace.sh
#   RANK=all MODE=summarize bash .../vis_profile_trace.sh   # both ranks
#
# Then open the printed HTML path, and/or follow the Perfetto steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_DIR="${REPO_ROOT}/outputs/profile/e2e"

MODE="${MODE:-both}"          # summarize | serve | both
RANK="${RANK:-0}"             # 0 | 1 | all
PORT="${PORT:-9001}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
FORCE_RESUMMARIZE="${FORCE_RESUMMARIZE:-0}"
TRACE_PROCESSOR="${TRACE_PROCESSOR:-${REPO_ROOT}/.tools/trace_processor}"
TRACE_PROCESSOR_URL="${TRACE_PROCESSOR_URL:-https://get.perfetto.dev/trace_processor}"
SSH_TARGET="${SSH_TARGET:-}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [TRACE_DIR | TRACE_FILE]

Serve / summarize verl torch Chrome traces under outputs/profile.

  TRACE_DIR   Directory of prof_rank-*.json.gz  (default: ${DEFAULT_DIR})
  TRACE_FILE  One specific .json.gz

Env:
  MODE=summarize|serve|both   default both
  RANK=0|1|all                which rank(s); default 0 (serve uses first)
  PORT=9001                   Perfetto trace_processor port
  FORCE_RESUMMARIZE=1         rebuild HTML even if cache is fresh
  SSH_TARGET=<host>           printed in the tunnel hint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INPUT="${1:-$DEFAULT_DIR}"
if [[ ! -e "$INPUT" ]]; then
  echo "ERROR: path not found: $INPUT" >&2
  exit 1
fi

# Resolve selected .gz files into SELECTED_TRACES (newline-separated abs paths).
mapfile -t SELECTED_TRACES < <("$PYTHON" - "$INPUT" "$RANK" <<'PY'
import pathlib
import sys

inp = pathlib.Path(sys.argv[1]).resolve()
rank = sys.argv[2].strip().lower()

def is_trace(p: pathlib.Path) -> bool:
    name = p.name
    return name.endswith(".json.gz") or name.endswith(".json")

def collect(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root] if is_trace(root) else []
    files = sorted(root.glob("prof_rank-*.json.gz"))
    if not files:
        files = sorted(root.glob("*.json.gz"))
    return [p for p in files if p.is_file()]

cands = collect(inp)
if not cands:
    print("ERROR: no Chrome traces (*.json.gz) under", inp, file=sys.stderr)
    sys.exit(2)

def rank_of(p: pathlib.Path):
    # prof_rank-0_....json.gz
    for part in p.stem.replace(".json", "").split("_"):
        if part.startswith("rank-") and part[5:].isdigit():
            return int(part[5:])
        if part.startswith("rank") and part[4:].isdigit():
            return int(part[4:])
    return None

if rank == "all":
    chosen = cands
elif rank.isdigit():
    want = int(rank)
    matched = [p for p in cands if rank_of(p) == want]
    chosen = matched or [max(cands, key=lambda p: p.stat().st_mtime)]
else:
    print(f"ERROR: RANK must be 0|1|all (got {rank!r})", file=sys.stderr)
    sys.exit(2)

for p in chosen:
    print(p)
PY
)

if [[ ${#SELECTED_TRACES[@]} -eq 0 ]]; then
  echo "ERROR: no traces selected under $INPUT" >&2
  exit 1
fi

echo "Profile traces"
echo "  input   : $INPUT"
echo "  mode    : $MODE"
echo "  rank    : $RANK"
echo "  selected:"
for t in "${SELECTED_TRACES[@]}"; do
  sz_mb="$("$PYTHON" -c "import pathlib,sys; print(f'{pathlib.Path(sys.argv[1]).stat().st_size/1e6:.0f}')" "$t")"
  echo "    - $(basename "$t")  (${sz_mb} MB)"
done
echo

summarize_one() {
  local gz="$1"
  local html="${gz%.json.gz}.bubble.html"
  if [[ "$gz" == *.json ]]; then
    html="${gz%.json}.bubble.html"
  fi
  if [[ "$FORCE_RESUMMARIZE" != "1" && -f "$html" && "$html" -nt "$gz" ]]; then
    echo "  cached  : $html" >&2
    echo "$html"
    return 0
  fi
  echo "  parsing : $(basename "$gz")  (large traces take a few minutes) ..." >&2
  "$PYTHON" - "$gz" "$html" <<'PY'
from __future__ import annotations

import array
import collections
import gzip
import html as html_mod
import json
import pathlib
import sys
import time


def classify_kernel(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("nccl", "allreduce", "all_gather", "allgather", "reduce_scatter", "broadcast")):
        return "nccl_comm"
    if any(x in n for x in ("memcpy", "copy_kernel", "copy_device", "memset")):
        return "memcpy"
    if any(x in n for x in ("gemm", "cutlass", "cublas", "mma", "wmma", "tensorop", "sgemm", "hgemm", "nvjet")):
        return "gemm"
    if any(x in n for x in ("flash", "fmha", "attention", "sdpa", "fused_attn", "paged_attn")):
        return "attention"
    if any(x in n for x in ("softmax", "layernorm", "rms", "gelu", "silu", "elementwise", "pointwise", "vectorized")):
        return "elementwise_norm"
    return "other_kernel"


def extract_field(raw: str, key: str) -> str | None:
    needle = f'"{key}":'
    i = raw.find(needle)
    if i < 0:
        return None
    j = i + len(needle)
    while j < len(raw) and raw[j] in " \n\t":
        j += 1
    if j >= len(raw):
        return None
    if raw[j] == '"':
        k = j + 1
        while k < len(raw):
            if raw[k] == "\\" and k + 1 < len(raw):
                k += 2
                continue
            if raw[k] == '"':
                return raw[j + 1 : k]
            k += 1
        return None
    k = j
    while k < len(raw) and raw[k] not in ",}\n":
        k += 1
    return raw[j:k].strip()


def occupancy(tss: array.array, durs: array.array):
    iv = sorted(zip(tss, durs), key=lambda x: x[0])
    if not iv:
        return None
    cs = iv[0][0]
    ce = iv[0][0] + iv[0][1]
    idle = 0.0
    gaps = []
    buckets = {"<1ms": 0, "1-10ms": 0, "10-100ms": 0, "100ms-1s": 0, "1-10s": 0, ">10s": 0}
    bsum = {k: 0.0 for k in buckets}
    busy = 0.0
    t0 = cs
    for ts, dur in iv[1:]:
        e = ts + dur
        if ts <= ce:
            if e > ce:
                ce = e
        else:
            g = ts - ce
            idle += g
            busy += ce - cs
            gs = g / 1e6
            if gs < 0.001:
                key = "<1ms"
            elif gs < 0.01:
                key = "1-10ms"
            elif gs < 0.1:
                key = "10-100ms"
            elif gs < 1:
                key = "100ms-1s"
            elif gs < 10:
                key = "1-10s"
            else:
                key = ">10s"
            buckets[key] += 1
            bsum[key] += gs
            gaps.append(g)
            gaps.sort(reverse=True)
            gaps = gaps[:12]
            cs, ce = ts, e
    busy += ce - cs
    span = ce - t0
    return {
        "n_kernels": len(iv),
        "span_s": span / 1e6,
        "busy_s": busy / 1e6,
        "idle_s": idle / 1e6,
        "util_pct": (100.0 * busy / span) if span else 0.0,
        "top_gaps_s": [round(g / 1e6, 3) for g in gaps],
        "gap_hist": {k: {"count": buckets[k], "sum_s": round(bsum[k], 2)} for k in buckets},
    }


def parse(path: pathlib.Path) -> dict:
    t0 = time.time()
    proc_label: dict = {}
    cat_dur: collections.Counter = collections.Counter()
    cat_n: collections.Counter = collections.Counter()
    gpu_class_dur: collections.Counter = collections.Counter()
    gpu_class_n: collections.Counter = collections.Counter()
    kernel_name_dur: collections.Counter = collections.Counter()
    user_ann_dur: collections.Counter = collections.Counter()
    gpu_ts: dict[int, array.array] = {}
    gpu_dur: dict[int, array.array] = {}
    n_obj = n_x = 0

    def handle_raw(raw: str) -> None:
        nonlocal n_x
        ph = extract_field(raw, "ph")
        name = extract_field(raw, "name") or ""
        if ph == "M":
            if name == "process_labels":
                try:
                    obj = json.loads(raw)
                    proc_label[obj.get("pid")] = obj.get("args", {}).get("labels")
                except Exception:
                    pass
            return
        if ph != "X":
            return
        n_x += 1
        cat = extract_field(raw, "cat") or ""
        try:
            dur = float(extract_field(raw, "dur") or "0")
            ts = float(extract_field(raw, "ts") or "0")
            pid = int(extract_field(raw, "pid") or "-1")
        except ValueError:
            return
        cat_dur[cat] += dur
        cat_n[cat] += 1
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            k = classify_kernel(name) if cat == "kernel" else ("memcpy" if "copy" in cat else "memset")
            gpu_class_dur[k] += dur
            gpu_class_n[k] += 1
            kernel_name_dur[name[:140]] += dur
            if pid not in gpu_ts:
                gpu_ts[pid] = array.array("d")
                gpu_dur[pid] = array.array("d")
            gpu_ts[pid].append(ts)
            gpu_dur[pid].append(dur)
        elif cat in ("user_annotation", "gpu_user_annotation"):
            user_ann_dur[name[:140]] += dur

    leftover = ""
    in_events = False
    depth = 0
    start = -1
    in_str = False
    esc = False
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            if not in_events:
                leftover += chunk
                marker = leftover.find('"traceEvents"')
                if marker < 0:
                    leftover = leftover[-64:]
                    continue
                br = leftover.find("[", marker)
                if br < 0:
                    leftover = leftover[marker:]
                    continue
                in_events = True
                chunk = leftover[br + 1 :]
                leftover = ""
            s = leftover + chunk
            i = len(leftover) if depth > 0 else 0
            leftover = ""
            n = len(s)
            while i < n:
                ch = s[i]
                if depth == 0:
                    if ch == "{":
                        depth = 1
                        start = i
                        in_str = False
                        esc = False
                    elif ch == "]":
                        n = i
                        break
                    i += 1
                    continue
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            n_obj += 1
                            handle_raw(s[start : i + 1])
                            start = -1
                i += 1
            if depth > 0 and start >= 0:
                leftover = s[start:]
                start = 0
            else:
                leftover = ""
                depth = 0
                in_str = False
                esc = False
                start = -1

    # Prefer GPU 0 occupancy (primary device on this rank).
    occ = None
    for pid in sorted(gpu_ts, key=lambda p: (-len(gpu_ts[p]), p)):
        lab = str(proc_label.get(pid, ""))
        if "GPU 0" in lab or pid == 0:
            occ = occupancy(gpu_ts[pid], gpu_dur[pid])
            occ["pid"] = pid
            occ["label"] = lab or f"pid={pid}"
            break
    if occ is None and gpu_ts:
        pid = max(gpu_ts, key=lambda p: len(gpu_ts[p]))
        occ = occupancy(gpu_ts[pid], gpu_dur[pid])
        occ["pid"] = pid
        occ["label"] = str(proc_label.get(pid, f"pid={pid}"))

    return {
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "parse_s": round(time.time() - t0, 1),
        "n_events": n_x,
        "cat_dur_s": {k: round(v / 1e6, 3) for k, v in cat_dur.most_common(12)},
        "gpu_class_s": {k: round(v / 1e6, 3) for k, v in gpu_class_dur.most_common()},
        "gpu_class_n": dict(gpu_class_n),
        "top_kernels": [(round(v / 1e6, 3), k) for k, v in kernel_name_dur.most_common(15)],
        "top_annotations": [(round(v / 1e6, 3), k) for k, v in user_ann_dur.most_common(15)],
        "occupancy": occ,
    }


def bar_row(label: str, value: float, total: float, color: str) -> str:
    pct = (100.0 * value / total) if total > 0 else 0.0
    width = max(0.5, pct)
    return (
        f'<div class="row"><div class="lab">{html_mod.escape(label)}</div>'
        f'<div class="track"><div class="fill" style="width:{width:.2f}%;background:{color}"></div></div>'
        f'<div class="val">{value:.2f}s · {pct:.1f}%</div></div>'
    )


def render(data: dict) -> str:
    occ = data.get("occupancy") or {}
    classes = data.get("gpu_class_s") or {}
    class_total = sum(classes.values()) or 1.0
    colors = {
        "nccl_comm": "#c44e52",
        "elementwise_norm": "#4c72b0",
        "other_kernel": "#8172b3",
        "gemm": "#55a868",
        "attention": "#64b5cd",
        "memcpy": "#ccb974",
        "memset": "#8c8c8c",
    }
    class_bars = "\n".join(
        bar_row(k, v, class_total, colors.get(k, "#888")) for k, v in classes.items()
    )
    kern_rows = "".join(
        f"<tr><td>{s:.3f}s</td><td><code>{html_mod.escape(n)}</code></td></tr>" for s, n in data["top_kernels"]
    )
    ann_rows = "".join(
        f"<tr><td>{s:.3f}s</td><td><code>{html_mod.escape(n)}</code></td></tr>" for s, n in data["top_annotations"]
    )
    gap_rows = ""
    if occ.get("gap_hist"):
        for k, info in occ["gap_hist"].items():
            gap_rows += f"<tr><td>{k}</td><td>{info['count']}</td><td>{info['sum_s']:.2f}s</td></tr>"
    util = occ.get("util_pct", 0.0)
    nccl = classes.get("nccl_comm", 0.0)
    nccl_pct = 100.0 * nccl / class_total if class_total else 0.0
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Bubble report · {html_mod.escape(pathlib.Path(data['file']).name)}</title>
<style>
  :root {{ --bg:#0f1115; --panel:#171a21; --text:#e8eaed; --muted:#9aa0a6; --line:#2a2f3a; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:14px/1.45 ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--text); }}
  main {{ max-width:980px; margin:0 auto; padding:28px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 6px; font-weight:650; }}
  h2 {{ font-size:16px; margin:28px 0 10px; font-weight:600; }}
  .muted {{ color:var(--muted); }}
  .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
  .stat .v {{ font-size:22px; font-weight:650; }}
  .stat .l {{ color:var(--muted); margin-top:4px; font-size:12px; }}
  .callout {{ border:1px solid var(--line); border-left:3px solid #f0ad4e; background:var(--panel); padding:12px 14px; border-radius:8px; margin:16px 0; }}
  .row {{ display:grid; grid-template-columns:160px 1fr 120px; gap:10px; align-items:center; margin:6px 0; }}
  .lab {{ color:var(--muted); font-size:12px; }}
  .track {{ height:10px; background:#222733; border-radius:999px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:999px; }}
  .val {{ text-align:right; font-variant-numeric:tabular-nums; font-size:12px; color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; }}
  th,td {{ text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ color:var(--muted); font-weight:500; font-size:12px; }}
  code {{ font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; color:#d7dae0; word-break:break-all; }}
  .note {{ font-size:12px; color:var(--muted); margin-top:24px; }}
</style></head><body><main>
  <h1>Actor Chrome-trace bubble report</h1>
  <p class="muted">{html_mod.escape(pathlib.Path(data['file']).name)} · {data['size_mb']} MB · {data['n_events']:,} events · parsed in {data['parse_s']}s</p>

  <div class="callout">
    <strong>Scope:</strong> this file is the FSDP actor rank window
    (<code>old_log_prob</code> + <code>update_actor</code>). Sidecar
    <code>generate_image</code> / <code>judge_image</code> waits are <em>not</em>
    in this trace — they dominate step wall-clock via trainer
    <code>timing_s/gen</code> / <code>timing_s/agent_loop/tool_calls</code>.
  </div>

  <div class="grid">
    <div class="stat"><div class="v">{occ.get('span_s', 0):.1f}s</div><div class="l">Kernel span (first→last)</div></div>
    <div class="stat"><div class="v">{util:.1f}%</div><div class="l">GPU busy / span</div></div>
    <div class="stat"><div class="v">{nccl_pct:.0f}%</div><div class="l">Busy time that is NCCL</div></div>
    <div class="stat"><div class="v">{occ.get('idle_s', 0):.1f}s</div><div class="l">Idle inside actor window</div></div>
  </div>

  <h2>Kernel class mix (busy time)</h2>
  {class_bars or '<p class="muted">No kernel events found.</p>'}

  <h2>Idle-gap histogram (inside actor window)</h2>
  <table><thead><tr><th>Gap size</th><th>Count</th><th>Idle sum</th></tr></thead>
  <tbody>{gap_rows or '<tr><td colspan="3" class="muted">n/a</td></tr>'}</tbody></table>
  <p class="muted">Top gaps: {', '.join(str(x)+'s' for x in (occ.get('top_gaps_s') or [])) or 'n/a'}</p>

  <h2>Top kernels</h2>
  <table><thead><tr><th>Time</th><th>Kernel</th></tr></thead><tbody>{kern_rows}</tbody></table>

  <h2>Top user annotations (CPU overlapping; not wall exclusive)</h2>
  <table><thead><tr><th>Time</th><th>Annotation</th></tr></thead><tbody>{ann_rows}</tbody></table>

  <p class="note">
    Deep dive: <code>MODE=serve bash vis_profile_trace.sh {html_mod.escape(str(pathlib.Path(data['file'])))}</code>
    then open <a href="https://ui.perfetto.dev/" style="color:#8ab4f8">ui.perfetto.dev</a>.
  </p>
</main></body></html>
"""


gz = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
data = parse(gz)
out.write_text(render(data), encoding="utf-8")
# Also dump machine-readable sidecar for reuse.
out.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")
print(out)
PY
}

SUMMARY_HTMLS=()
if [[ "$MODE" == "summarize" || "$MODE" == "both" ]]; then
  echo "Summarize"
  for t in "${SELECTED_TRACES[@]}"; do
    SUMMARY_HTMLS+=("$(summarize_one "$t")")
  done
  echo
  echo "Open bubble report(s) in a browser:"
  for h in "${SUMMARY_HTMLS[@]}"; do
    echo "  file://$h"
  done
  echo
fi

ensure_trace_processor() {
  if [[ -x "$TRACE_PROCESSOR" ]]; then
    return 0
  fi
  if command -v trace_processor >/dev/null 2>&1; then
    TRACE_PROCESSOR="$(command -v trace_processor)"
    return 0
  fi
  echo "Downloading Perfetto trace_processor → $TRACE_PROCESSOR"
  mkdir -p "$(dirname "$TRACE_PROCESSOR")"
  curl -fsSL -o "$TRACE_PROCESSOR" "$TRACE_PROCESSOR_URL"
  chmod +x "$TRACE_PROCESSOR"
}

if [[ "$MODE" != "serve" && "$MODE" != "both" ]]; then
  exit 0
fi

SERVE_TRACE="${SELECTED_TRACES[0]}"
echo "Serve (Perfetto)"
echo "  trace : $SERVE_TRACE"
echo "  note  : large .gz loads can take several minutes the first time"
echo

ensure_trace_processor

# Pick a free port starting at PORT.
PORT="$("$PYTHON" - "$PORT" "$BIND_HOST" <<'PY'
import socket, sys
requested = int(sys.argv[1]); host = sys.argv[2]
for port in range(requested, requested + 30):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit(f"no free port near {requested}")
PY
)"

echo "Keep this process running. On your laptop:"
if [[ -n "$SSH_TARGET" ]]; then
  echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${SSH_TARGET}"
else
  echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} <user>@<this-host>"
fi
echo
echo "Then open:"
if [[ "$PORT" == "9001" ]]; then
  echo "  https://ui.perfetto.dev/"
  echo "  → Perfetto auto-probes :9001; click YES, use loaded trace"
else
  echo "  https://ui.perfetto.dev/#!/?rpc_port=${PORT}"
  echo "  (one-time: enable https://ui.perfetto.dev/#!/flags/cspAllowAnyWebsocketPort )"
fi
echo
echo "What to look at in Perfetto:"
echo "  1. GPU tracks  → kernel occupancy / NCCL bubbles in the actor window"
echo "  2. Search user annotations: update_actor, compute_log_prob, FullyShardedDataParallel"
echo "  3. Ignore the long quiet region if any — gen/sidecar waits are not traced here"
echo
echo "Starting trace_processor on ${BIND_HOST}:${PORT} (Ctrl-C to stop) ..."
exec "$TRACE_PROCESSOR" \
  --httpd \
  --http-port "$PORT" \
  --http-ip-address "$BIND_HOST" \
  "$SERVE_TRACE"
