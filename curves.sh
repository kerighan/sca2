#!/usr/bin/env bash
# Curves + gap table + live progress for the d=1024 run, at any point during it.
#   ./curves.sh                 -> runs/long5h_d1024.jsonl
#   ./curves.sh runs/other.jsonl [extra plot_lm.py args...]
# Picks d1024_gdn as the reference once it exists, else the first arm in the log, so
# it works from the first eval of arm 1 rather than only once all three have run.
#
# Note on reading it early: plot_lm.py drops the first eval of each arm, and its gap
# panel needs BOTH arms past 60M tokens, so with one arm running the lower panel is
# empty by construction and the upper one is a short line. The progress table below
# is the useful view until the second arm is well underway.
set -u
# POSIX-clean, so `sh curves.sh` works as well as `./curves.sh`. In particular: guard
# the shift. `shift` is a SPECIAL BUILTIN, so under dash (which is /bin/sh here) an
# error in it kills the shell outright and `|| true` does not catch it -- `sh curves.sh`
# with no arguments exited 2 before printing anything, while ./curves.sh was fine.
LOG=${1:-runs/long5h_d1024.jsonl}
[ $# -gt 0 ] && shift
[ -s "$LOG" ] || { echo "no evals yet in $LOG (cwd: $(pwd))"; exit 0; }

REF=$(python -c '
import json, sys
seen = []
for l in open(sys.argv[1]):
    m = json.loads(l)["model"]
    if m not in seen: seen.append(m)
print("d1024_gdn" if "d1024_gdn" in seen else seen[0])
' "$LOG")

BASE=$(basename "${LOG%.jsonl}")
OUT=plot/${BASE}.png
OUT_WC=plot/${BASE}_wallclock.png

# Show only the arms that matter: the reference, the starting point, key milestones,
# and anything currently running. The full data stays in the JSONL.
SHOW="d1024_gdn,d1024_lapa,d1024_lsfree,d1024_gdngate,d1024_gdngate_L128,d1024_gatemix"

python plot_lm.py "$LOG" --ref "$REF" --out "$OUT" --only "$SHOW" "$@" 2>&1 | grep -vE "UserWarning|ax2\.axhline"
echo
python plot_lm.py "$LOG" --ref "$REF" --out "$OUT_WC" --wallclock --only "$SHOW" "$@" 2>&1 | grep -vE "UserWarning|ax2\.axhline"

echo
python -c '
import json, sys
rows = {}
for l in open(sys.argv[1]):
    r = json.loads(l); rows.setdefault(r["model"], []).append(r)
budget = float(sys.argv[2])
print("  %-12s %5s %8s %7s %8s %8s %7s" % ("arm","evals","tokens","val","tok/s","elapsed","left"))
for k, v in rows.items():
    e = v[-1]
    left = max(0.0, budget - e["train_s"])
    print("  %-12s %5d %7.0fM %7.4f %8d %7.0fm %6.0fm"
          % (k, len(v), e["tokens"]/1e6, e["val"], e["tok_s"], e["train_s"]/60, left/60))
' "$LOG" "${SECONDS_BUDGET:-18000}"

for p in $(pgrep -x python); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *pretrain.py*) echo "  running:$(echo "$c" | grep -o -- ' --label [^ ]*')";; esac
done
