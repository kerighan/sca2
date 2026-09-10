"""
Read runs/hlong.jsonl. The number that matters is not the val loss, it is the
SLOPE of the loss across positions in the window.

A model that only uses a short window has a profile that flattens: predicting
token 1900 is no easier than predicting token 300. One that uses distance keeps
improving. Reporting it as (late - early) makes it a within-model difference, so
the per-eval noise that makes the aggregate loss useless here (sd = 0.039 nats)
largely cancels.
"""
import json
import statistics as st
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/hlong.jsonl"
rows = [json.loads(x) for x in open(path)]
seen, arms = {}, []
for r in rows:
    if r["step"] == 1:
        seen[r["model"]] = []
        if r["model"] not in arms:
            arms.append(r["model"])
    seen.setdefault(r["model"], []).append(r)

w = max(len(a) for a in arms) + 1
T = 2048

print("final per-position val loss (8 buckets of ~256 positions)")
hdr = " ".join(f"{i*T//8:>5d}+" for i in range(8))
print(f"{'arm':{w}s} {'tok/s':>7s} {'Mtok':>6s} {'val':>6s}  {hdr}")
for a in arms:
    r = seen[a][-1]
    p = r.get("pos")
    cells = " ".join(f"{v:6.3f}" for v in p) if p else "no --pos-buckets"
    print(f"{a:{w}s} {r['tok_s']:7d} {r['tokens']/1e6:6.1f} {r['val']:6.3f}  {cells}")

print("\nuse of distance: mean(last 2 buckets) - mean(first 2 buckets)")
print("more negative = the model keeps gaining from more context")
print(f"{'arm':{w}s} {'slope':>7s} {'sd over last 3 evals':>22s}")
for a in arms:
    rs = [r for r in seen[a] if r.get("pos")]
    if not rs:
        continue
    def slope(r):
        p = r["pos"]
        return (p[-1] + p[-2]) / 2 - (p[0] + p[1]) / 2
    s = [slope(r) for r in rs[-3:]]
    sd = f"{st.stdev(s):.4f}" if len(s) > 2 else "-"
    print(f"{a:{w}s} {st.mean(s):+7.3f} {sd:>22s}")
