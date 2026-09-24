"""Merge the arms' eval logs and compare them at equal, host-normalised wall clock.

    python -m vast.curves                      # table
    python -m vast.curves --plot plot/zyda.png

Arms run on different rented hosts cannot be compared on raw elapsed time: a
host 10% slower gives its arm 10% fewer tokens, which at the fitted slope is
worth ~0.025 nats -- more than the effect. `vast.calibrate` times one identical
workload everywhere and records tok/s per host; this converts each arm's
elapsed seconds into REFERENCE-HOST seconds before comparing.

With every arm on one host the factors are all 1 and this is just the usual
wall-clock comparison.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import statistics

from .common import ROOT, instances

OUT = ROOT / "runs"


def load(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    """{arm: [(train_s, val, tokens)]} sorted by time."""
    per: dict[str, list] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val" not in r or "train_s" not in r:
                continue
            per.setdefault(r["model"], []).append(
                (float(r["train_s"]), float(r["val"]), float(r.get("tokens", 0))))
    return {k: sorted(v) for k, v in per.items()}


def at(series, x):
    xs = [a for a, _, _ in series]
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    i = bisect.bisect_left(xs, x)
    if xs[i] == x:
        return series[i][1]
    (x0, y0, _), (x1, y1, _) = series[i - 1], series[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def host_factors(arm_host: dict[str, int]) -> dict[str, float]:
    """arm -> multiply its elapsed seconds by this to get reference-host seconds."""
    pool = {p["slot"]: p for p in instances()}
    rates = {s: p.get("calib_tok_s") for s, p in pool.items()}
    known = [r for r in rates.values() if r]
    if not known:
        return {a: 1.0 for a in arm_host}
    ref = known[0]
    return {a: (rates.get(s) or ref) / ref for a, s in arm_host.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="zyda")
    p.add_argument("--ref", default=None, help="baseline arm; default the one named *gdn*")
    p.add_argument("--map", default="", help="arm=slot,arm=slot when arms ran on "
                                             "different hosts")
    p.add_argument("--plot", default=None)
    a = p.parse_args()

    series = load(OUT / f"{a.log}.jsonl")
    if not series:
        raise SystemExit(f"no evals in runs/{a.log}.jsonl")
    arm_host = {}
    for item in filter(None, a.map.split(",")):
        k, v = item.split("=")
        arm_host[k] = int(v)
    for arm in series:
        arm_host.setdefault(arm, 0)
    factors = host_factors(arm_host)

    ref = a.ref or next((k for k in series if "gdn" in k), sorted(series)[0])
    if ref not in series:
        raise SystemExit(f"reference {ref!r} not among {sorted(series)}")

    print(f"{'arm':>18} {'evals':>6} {'tokens':>9} {'val':>8} {'host x':>7}")
    for arm, s in sorted(series.items()):
        print(f"{arm:>18} {len(s):6d} {s[-1][2]/1e9:8.2f}B {s[-1][1]:8.4f} "
              f"{factors[arm]:7.4f}")

    # Comparable horizon: the largest normalised time every arm reached.
    horizon = min(s[-1][0] * factors[k] for k, s in series.items())
    grid = [horizon * f for f in (0.25, 0.5, 0.75, 1.0)]
    print(f"\nval at equal host-normalised training time (to {horizon/3600:.1f} h):")
    head = " ".join(f"{g/3600:>10.1f}h" for g in grid)
    print(f"{'arm':>18} {head}")
    for arm, s in sorted(series.items()):
        row = [at(s, g / factors[arm]) for g in grid]
        print(f"{arm:>18} " + " ".join(
            f"{v:11.4f}" if v is not None else f"{'—':>11}" for v in row))

    print(f"\ngap to {ref}:")
    for arm, s in sorted(series.items()):
        if arm == ref:
            continue
        d = [at(s, g / factors[arm]) - at(series[ref], g / factors[ref])
             for g in grid
             if at(s, g / factors[arm]) is not None
             and at(series[ref], g / factors[ref]) is not None]
        if d:
            print(f"{arm:>18} " + " ".join(f"{x:+11.4f}" for x in d)
                  + f"   median {statistics.median(d):+.4f}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2]})
        for arm, s in sorted(series.items()):
            x = [t * factors[arm] / 3600 for t, _, _ in s][1:]
            y = [v for _, v, _ in s][1:]
            ax1.plot(x, y, lw=2.4 if arm == ref else 1.4,
                     color="#2c3e50" if arm == ref else None, label=arm)
        ax1.set_ylabel("val loss (nats)")
        ax1.grid(alpha=.3)
        ax1.legend(fontsize=8)
        ax1.set_title(f"runs/{a.log}.jsonl — host-normalised wall clock")
        for arm, s in sorted(series.items()):
            if arm == ref:
                continue
            xs = [horizon * k / 40 for k in range(4, 41)]
            g = [(x / 3600, at(s, x / factors[arm]) - at(series[ref], x / factors[ref]))
                 for x in xs
                 if at(s, x / factors[arm]) is not None
                 and at(series[ref], x / factors[ref]) is not None]
            if g:
                ax2.plot([p_[0] for p_ in g], [p_[1] for p_ in g], lw=1.6, label=arm)
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel("host-normalised training time (h)")
        ax2.set_ylabel(f"gap to {ref} (nats)")
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)
        Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(a.plot, dpi=130)
        print(f"\nsaved {a.plot}")


if __name__ == "__main__":
    main()
