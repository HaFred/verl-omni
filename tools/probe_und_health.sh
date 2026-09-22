#!/usr/bin/env bash
# Poll the live UND AR replica's health and print a timestamped verdict.
#
# The UND replica is a plain AR model: for "Tell me about bananas" it must answer sensibly.
# When it is corrupted (after a sleep/wake/weight-sync) it instead repeats the last token of
# the prompt -- which is exactly the `assistant\n` loop the rollouts show, because the UND
# prompt's last content token is the role word "assistant" from `<|im_start|>assistant\n`.
#
# Correlating these verdicts with the trainer log's sleep/wake/update_weights events pinpoints
# which lifecycle step corrupts the replica.
set -uo pipefail
OUT="${1:-/tmp/und_health.log}"

find_ar_port() {
  for p in $(ss -ltnp 2>/dev/null | grep -oE "10\.248\.[0-9.]+:[0-9]+" | cut -d: -f2 | sort -un); do
    resp=$(curl -s -m 25 "http://10.248.12.145:$p/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"bagel","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' 2>/dev/null)
    case "$resp" in
      *'"choices"'*) echo "$p"; return 0 ;;
    esac
  done
  return 1
}

PORT=$(find_ar_port) || { echo "$(date +%H:%M:%S) no AR port yet"; exit 0; }
resp=$(curl -s -m 90 "http://10.248.12.145:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"bagel","messages":[{"role":"user","content":"Tell me about bananas"}],"max_tokens":16,"temperature":0.0}')
verdict=$(printf '%s' "$resp" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    c=d["choices"][0]["message"]["content"]
except Exception as e:
    print(f"ERR {e}"); raise SystemExit
first=(c.strip().split() or [""])[0].lower().strip(",.:")
words=c.strip().split()
rep = len(words)>3 and sum(w.lower().rstrip(",.")==first for w in words)/max(len(words),1) > 0.5
print(("LOOP " if rep else "OK   ")+repr(c[:90]))
')
echo "$(date +%H:%M:%S) port=$PORT $verdict" >> "$OUT"
