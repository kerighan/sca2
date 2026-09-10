"""Read runs/confirm_pycode.jsonl and answer the two questions it was run for.

Unlike dump of the sweep, this reports ENDPOINTS, not fits: every arm is meant to
have completed exactly one epoch over the same 177.4M-token corpus, so the final
val loss is directly comparable and no extrapolation is needed. The sweep showed
that fitted slopes over an 800s window carry ~0.2 nats of projection noise, which
is why nothing here is projected.

Two columns carry the conclusions:
  * val at the common token count (min over arms, in case an arm was capped by
    the 3000s limit instead of by corpus exhaustion) -- the quality answer.
  * pos slope, last bucket minus first -- whether distance is used at all.

The 2-layer GDN reference from runs/pycode.jsonl is printed alongside, but the
4-layer arms are only comparable to gdn4, not to it: they hold 2x layer params.

    python dump_confirm.py
"""
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def load(path, keep=None):
    d = defaultdict(list)
    if not path.exists():
        return d
    for line in path.open():
        r = json.loads(line)
        if keep is None or r["model"] in keep:
            d[r["model"]].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["step"])
    return d


def at_tokens(rows, n):
    """Val loss at the last eval with tokens <= n."""
    ok = [r for r in rows if r["tokens"] <= n]
    return ok[-1] if ok else rows[0]


def main():
    arms = load(RUNS / "confirm_pycode.jsonl")
    ref = load(RUNS / "pycode.jsonl")
    if not arms:
        print("no runs/confirm_pycode.jsonl yet")
        return

    common = min(v[-1]["tokens"] for v in arms.values())
    print(f"{len(arms)} arms, common token count {common/1e6:.1f}M "
          f"(corpus = 177.4M; an arm below that was capped by the clock)\n")
    print(f"{'arm':14s} {'tok/s':>7} {'epoch':>7} {'val@common':>11} "
          f"{'val final':>10} {'pos slope':>10}")
    print("-" * 66)

    rank = []
    for k, v in arms.items():
        last, cut = v[-1], at_tokens(v, common)
        p = last.get("pos")
        slope = (p[-1] - p[0]) if p else float("nan")
        rank.append((cut["val"], k, last, slope))
    for val, k, last, slope in sorted(rank):
        print(f"{k:14s} {last['tok_s']:>7} {last['tokens']/177.4e6:>6.2f}x "
              f"{val:>11.4f} {last['val']:>10.4f} {slope:>+10.3f}")

    print("\nreference, 2 layers, one epoch (runs/pycode.jsonl):")
    for k, v in ref.items():
        p = v[-1].get("pos")
        s = f"{p[-1]-p[0]:+.3f}" if p else "n/a"
        print(f"  {k:12s} val {v[-1]['val']:.4f}  tokens {v[-1]['tokens']/1e6:.1f}M"
              f"  pos slope {s}")


if __name__ == "__main__":
    main()
