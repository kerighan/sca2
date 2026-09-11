#!/usr/bin/env bash
# Curves + gap table for the d=1024 run, at any point during it.
#   ./curves.sh                 -> runs/long5h_d1024.jsonl
#   ./curves.sh runs/other.jsonl [extra plot_lm.py args...]
# Picks d1024_gdn as the reference once it exists, else the first arm in the log, so
# it works from the first eval of arm 1 rather than only once all three have run.
set -u
LOG=${1:-runs/long5h_d1024.jsonl}; shift 2>/dev/null || true
[ -s "$LOG" ] || { echo "no evals yet in $LOG"; exit 0; }
REF=$(python - "$LOG" <<'PY'
import json, sys
seen = []
for l in open(sys.argv[1]):
    m = json.loads(l)["model"]
    if m not in seen: seen.append(m)
print("d1024_gdn" if "d1024_gdn" in seen else seen[0])
PY
)
OUT=plot/$(basename "${LOG%.jsonl}").png
python plot_lm.py "$LOG" --ref "$REF" --out "$OUT" "$@"
echo
echo "live: $(grep -c . "$LOG") evals; arms running now:"
for p in $(pgrep -x python); do c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); case "$c" in
  *pretrain.py*) echo "  $(echo $c | grep -o -- '--label [^ ]*')";; esac; done
tail -2 "${LOG%.jsonl}.log" 2>/dev/null | grep -E "^  d1024" || true
