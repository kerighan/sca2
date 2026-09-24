#!/usr/bin/env bash
# Curves for the gate scope ablation on wikitext.
#   ./curves_gate.sh
set -u
LOG=runs/wiki_gate.jsonl
[ -s "$LOG" ] || { echo "no evals yet in $LOG"; exit 0; }

REF=gate_long

python plot_lm.py "$LOG" --ref "$REF" --out plot/wiki_gate.png 2>&1 | grep -vE "UserWarning|ax2\.axhline"
echo
python plot_lm.py "$LOG" --ref "$REF" --out plot/wiki_gate_wallclock.png --wallclock 2>&1 | grep -vE "UserWarning|ax2\.axhline"

echo
python -c '
import json, sys
rows = {}
for l in open(sys.argv[1]):
    r = json.loads(l); rows.setdefault(r["model"], []).append(r)
budget = float(sys.argv[2])
print("  %-16s %5s %8s %7s %8s %8s %7s" % ("arm","evals","tokens","val","tok/s","elapsed","left"))
for k, v in rows.items():
    e = v[-1]
    left = max(0.0, budget - e["train_s"])
    print("  %-16s %5d %7.0fM %7.4f %8d %7.0fm %6.0fm"
          % (k, len(v), e["tokens"]/1e6, e["val"], e["tok_s"], e["train_s"]/60, left/60))
' "$LOG" 1800

for p in $(pgrep -x python); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *pretrain.py*) echo "  running:$(echo "$c" | grep -o -- ' --label [^ ]*')";; esac
done
