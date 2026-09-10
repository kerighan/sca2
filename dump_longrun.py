"""
Read longrun.sh: is the gap closing, flat, or widening -- and does it cross?

The deliverable is the SLOPE of the gap against log-tokens, not the endpoint. A
seed changes a run's level far more than its shape, so the slope survives n=2
where the endpoint does not. The endpoint is printed anyway, with the warning
attached, because it is the number that will be misquoted otherwise.

Three outcomes, and the run is designed so they look different:
  slope > 0   the gap CLOSES -- generation 3's win is transient. If the fitted
              crossing lands inside the measured range, the headline is retracted.
  slope ~ 0   a constant offset in nats: generation 3 is worth a fixed multiple of
              GDN's tokens, and the advantage is durable but does not compound.
  slope < 0   the gap WIDENS, i.e. a genuine scaling-exponent difference. This is
              the strongest possible result and the least likely a priori.

    python dump_longrun.py
"""
import json
import math
from collections import defaultdict

import numpy as np

LOG = "runs/longrun.jsonl"
ARMS = {"gen3": ["long_gen3_s0", "long_gen3_s1"],
        "gdn": ["long_gdn_s0", "long_gdn_s1"]}
# The historical window, for continuity with WINNERS.md's -0.0854 at 170M.
PRIOR = (169.8e6, -0.0854, +0.222)


def load():
    runs = defaultdict(list)
    for line in open(LOG):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        runs[r["model"]].append((r["tokens"], r["val"], r.get("pos")))
    return {k: sorted(v) for k, v in runs.items()}


def curve(runs, label):
    d = runs[label]
    return (np.array([x[0] for x in d], float),
            np.array([x[1] for x in d], float))


def main():
    runs = load()
    have = {n: [lb for lb in ls if lb in runs] for n, ls in ARMS.items()}
    for n, ls in have.items():
        print(f"{n:5} {len(ls)} run(s): " + ", ".join(
            f"{lb} to {max(runs[lb])[0]/1e6:.0f}M" for lb in ls))
    if not all(have.values()):
        print("\nincomplete -- at least one arm has no run yet")
        return

    # Fit only where BOTH arms have data, and drop the steep early descent: the
    # power-law regime is what the extrapolation is about, and interpolating
    # across mid-descent evals manufactures whatever trend you look for.
    hi = min(max(curve(runs, lb)[0]) for ls in have.values() for lb in ls)
    lo = max(40e6, 0.08 * hi)
    grid = np.linspace(lo, hi, 12)
    print(f"\nfit window {lo/1e6:.0f}M..{hi/1e6:.0f}M  ({hi/lo:.1f}x lever arm)")

    M = {n: np.array([np.interp(grid, *curve(runs, lb)) for lb in ls])
         for n, ls in have.items()}

    print(f"\n{'tokens':>9} {'gen3':>8} {'gdn':>8} {'gap':>9}")
    for i, g in enumerate(grid):
        a, b = M["gen3"][:, i], M["gdn"][:, i]
        print(f"{g/1e6:8.1f}M {a.mean():8.4f} {b.mean():8.4f} "
              f"{a.mean()-b.mean():+9.4f}")

    x = np.log(grid)
    gaps = M["gen3"].mean(0) - M["gdn"].mean(0)
    slope, icept = np.polyfit(x, gaps, 1)

    # Bootstrap over seeds: with n=2 this is coarse, so it is reported as a range
    # of what the available seed combinations give, not as a confidence interval.
    combos = [np.polyfit(x, M["gen3"][[i]].mean(0) - M["gdn"][[j]].mean(0), 1)[0]
              for i in range(len(M["gen3"])) for j in range(len(M["gdn"]))]
    print(f"\nd(gap)/d(ln tokens) = {slope:+.4f}   "
          f"per seed pair: [{min(combos):+.4f}, {max(combos):+.4f}]")
    print(f"  prior estimate on 40M..170M was {PRIOR[2]:+.4f} -- "
          f"{'REPRODUCED' if slope * PRIOR[2] > 0 else 'SIGN FLIPPED'}")
    for n in ("gen3", "gdn"):
        print(f"  {n:5} dval/dln(tokens) = "
              f"{np.polyfit(x, M[n].mean(0), 1)[0]:+.4f}")

    end = gaps[-1]
    print(f"\nendpoint gap at {hi/1e6:.0f}M: {end:+.4f}")
    print(f"  (was {PRIOR[1]:+.4f} at {PRIOR[0]/1e6:.0f}M, n=3 vs n=6)")
    print("  DO NOT quote this as a headline: n=2 cannot resolve 0.08 nats.")

    if slope > 0:
        cross = math.exp(-icept / slope)
        inside = lo <= cross <= hi
        print(f"\nThe gap is CLOSING. Fitted crossing at {cross/1e6:.0f}M tokens"
              f" -- {'INSIDE' if inside else 'outside'} the measured range.")
        if inside:
            print("  Observed, not extrapolated: generation 3 loses its advantage")
            print("  within the budget. The headline must be retracted or scoped")
            print("  to a token budget.")
        else:
            print("  Still an EXTRAPOLATION, and this project has been burned by")
            print("  those before (confirm_pycode.sh: fitted slopes differing by")
            print("  0.074 became 0.20 nats when extended). Do not quote it.")
    elif slope < 0:
        print("\nThe gap is WIDENING: a scaling-exponent difference, the strongest")
        print("  available result. Verify at n>=3 before it goes anywhere.")
    else:
        print("\nThe gap is FLAT: a durable constant-nat advantage.")

    # Position profile at the long budget: the crossover at 0-127 is the sharpest
    # structural claim in WINNERS.md, so check it did not evaporate.
    prof = {}
    for n, ls in have.items():
        ps = [runs[lb][-1][2] for lb in ls if runs[lb][-1][2]]
        if ps:
            prof[n] = np.array(ps).mean(0)
    if len(prof) == 2 and len(prof["gen3"]) == len(prof["gdn"]):
        nb = len(prof["gen3"])
        w = 1024 // nb
        print(f"\nposition profile at the endpoint ({nb} buckets):")
        for i in range(nb):
            d = prof["gen3"][i] - prof["gdn"][i]
            print(f"  {i*w:4d}-{(i+1)*w-1:4d}  gen3 {prof['gen3'][i]:7.4f}  "
                  f"gdn {prof['gdn'][i]:7.4f}  {d:+.4f}"
                  + ("   <- gen3 behind" if d > 0 else ""))
        print("  WINNERS.md says gen3 is behind on 0-127 and ~-0.10 from 400 on.")


if __name__ == "__main__":
    main()
