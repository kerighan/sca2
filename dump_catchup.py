"""Read runs/catchup.jsonl: loss BY TOKEN CLASS along training, arm by arm.

    python dump_catchup.py [--log runs/catchup.jsonl] [--ref catch_gdn_s3]

For every arm: class losses at the end. For every arm against --ref: the gap per
class on a common token grid, and the LATE SLOPE of that gap against log-tokens
(fit over the second half of the shared window). The slope is the deliverable --
a seed moves a run's level far more than its shape (longrun.sh) -- and the
conjecture-1 prediction is specifically:

    gap slope on word_rep  ~ 0       the torus keeps exact retrieval
    gap slope on word_new  > 0       GDN catches up where similarity, not
                                     retrieval, is what helps
    bp/bp2 vs gen3: word_new slope less positive, word_rep not worse

Anything read before ~90% of the epoch is mid-descent and is printed as such.
"""
import argparse
import collections
import json
import math

import numpy as np

from sca2.tokclass import NAMES


def load(path):
    runs = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if "cls" in r:
            runs[r["model"]].append((r["tokens"], r["val"], r["cls"]))
    return {k: sorted(v) for k, v in runs.items()}


def interp(c, x, key):
    if x < c[0][0] or x > c[-1][0]:
        return None
    for i in range(1, len(c)):
        if c[i][0] >= x:
            (x0, _, a), (x1, _, b) = c[i - 1], c[i]
            w = (x - x0) / max(x1 - x0, 1)
            va = a[key] if key != "val" else c[i - 1][1]
            vb = b[key] if key != "val" else c[i][1]
            return va + (vb - va) * w


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="runs/catchup.jsonl")
    p.add_argument("--ref", default="catch_gdn_s3")
    a = p.parse_args(argv)
    runs = load(a.log)
    if not runs:
        print("no records with a class breakdown in", a.log); return 1
    keys = ["val"] + NAMES
    print(f"{'arm':>16} {'tokens':>7} " + " ".join(f"{k:>8}" for k in keys))
    for name, c in runs.items():
        t, v, cls = c[-1]
        row = [v] + [cls.get(k, float("nan")) for k in NAMES]
        print(f"{name:>16} {t/1e6:6.1f}M " + " ".join(f"{x:8.4f}" for x in row))

    ref = runs.get(a.ref)
    if ref is None:
        print(f"\n(no --ref arm {a.ref!r} yet; gaps skipped)"); return 0
    for name, c in runs.items():
        if name == a.ref:
            continue
        lo = max(c[0][0], ref[0][0]); hi = min(c[-1][0], ref[-1][0])
        if hi <= lo * 1.5:
            print(f"\n{name}: shared window too short for a slope"); continue
        grid = np.exp(np.linspace(math.log(lo), math.log(hi), 12))
        full = c[-1][0] >= 0.9 * 177e6
        print(f"\n=== {name} - {a.ref}   (gap < 0: {name} better)"
              + ("" if full else "   [MID-DESCENT: not a verdict]"))
        print(f"{'tokens':>8} " + " ".join(f"{k:>8}" for k in keys))
        G = {k: [] for k in keys}
        for x in grid:
            row = []
            for k in keys:
                g = interp(c, x, k) - interp(ref, x, k)
                G[k].append(g); row.append(g)
            print(f"{x/1e6:7.1f}M " + " ".join(f"{g:+8.4f}" for g in row))
        L = np.log(grid); h = len(grid) // 2
        print(f"{'slope/ln':>8} " + " ".join(
            f"{np.polyfit(L[h:], np.array(G[k])[h:], 1)[0]:+8.3f}" for k in keys)
              + "   <- late half; + = gap closing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
