"""
Read runs/h4.jsonl two ways, because at equal wall clock the arms do not see the
same number of tokens -- which is the whole point of the speed arms, and also the
easiest way to fool yourself.

  equal TIME   who wins with 30 minutes of this GPU (the deployment question)
  equal TOKENS who wins per unit of data (the architecture question)
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/h4.jsonl"
rows = [json.loads(l) for l in open(path)]

arms, seen = [], {}
for r in rows:
    if r["step"] == 1:
        seen[r["model"]] = []          # a restart resets that arm
        if r["model"] not in arms:
            arms.append(r["model"])
    seen.setdefault(r["model"], []).append(r)


def at_tokens(rs, tok):
    c = [r for r in rs if r["tokens"] >= tok]
    return c[0]["val"] if c else None


def at_time(rs, s):
    c = [r for r in rs if r["train_s"] >= s]
    return c[0]["val"] if c else None


grid_t = [300, 600, 1200, 1800]
grid_k = [50e6, 100e6, 150e6]
w = max(len(a) for a in arms) + 1

print("equal WALL CLOCK (val loss)")
print(f"{'arm':{w}s} {'tok/s':>7s} {'Mtok':>7s} " + " ".join(f"{s//60:>6d}min" for s in grid_t))
for a in arms:
    rs = seen[a]
    last = rs[-1]
    cells = " ".join(f"{v:9.3f}" if (v := at_time(rs, s)) else "        -" for s in grid_t)
    print(f"{a:{w}s} {last['tok_s']:7d} {last['tokens']/1e6:7.1f} {cells}")

print("\nequal TOKENS (val loss)")
print(f"{'arm':{w}s} " + " ".join(f"{k/1e6:>7.0f}M" for k in grid_k))
for a in arms:
    rs = seen[a]
    cells = " ".join(f"{v:8.3f}" if (v := at_tokens(rs, k)) else "       -" for k in grid_k)
    print(f"{a:{w}s} {cells}")
