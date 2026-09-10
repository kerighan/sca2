"""Read runs/theta.jsonl: does the C head's content phase close the gap to GDN?

Three arms at ~743.5k layer params, 4 layers, one epoch, n=3 seeds each:

    theta 0     md4_dv32, from runs/md_axis.jsonl + runs/seeds.jsonl (NOT rerun)
    theta 0.02  runs/theta.jsonl
    theta 0.1   runs/theta.jsonl

against gdn4, n=3 (mean 2.7882, sd 0.0726).

Two things to read, and they can disagree:
  * val, which is the claim.
  * the position slope. The content phase is supposed to buy CONTENT recall, so
    it could improve val while leaving the positional profile alone -- that would
    still be the hypothesis confirmed, and is why both are printed.

All runs are interpolated to a common token count: each does exactly one epoch,
but the last eval lands wherever the 120s clock put it (167-176M), and val is
still falling there.

    python dump_theta.py
"""
import itertools
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
FILES = ("theta.jsonl", "seeds.jsonl", "md_axis.jsonl", "confirm_pycode.jsonl")


def arm_of(label):
    """Map a log label onto (arm, seed)."""
    if label.startswith("th002_s"):
        return "theta 0.02", int(label[-1])
    if label.startswith("th010_s"):
        return "theta 0.1", int(label[-1])
    if label.startswith("md4_dv32_s"):
        return "theta 0", int(label[-1])
    if label == "md4_dv32":
        return "theta 0", 0
    if label.startswith("gdn4_s"):
        return "gdn4", int(label[-1])
    if label == "gdn4":
        return "gdn4", 0
    return None, None


def load():
    runs = defaultdict(list)
    for f in FILES:
        p = RUNS / f
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            arm, seed = arm_of(r["model"])
            if arm:
                runs[(arm, seed)].append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["step"])
    return runs


def welch(a, b):
    na, nb = len(a), len(b)
    va, vb = st.variance(a) / na, st.variance(b) / nb
    t = (st.mean(a) - st.mean(b)) / (va + vb) ** .5
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, dof, (va + vb) ** .5


def main():
    runs = load()
    if not runs:
        print("no logs yet")
        return
    common = min(v[-1]["tokens"] for v in runs.values())
    print(f"n={len(runs)} runs, interpolated to {common/1e6:.1f}M tokens\n")

    val, slope = defaultdict(list), defaultdict(list)
    for (arm, seed), v in sorted(runs.items()):
        val[arm].append(float(np.interp(common, [r["tokens"] for r in v],
                                        [r["val"] for r in v])))
        p = v[-1].get("pos")
        if p:
            slope[arm].append(p[-1] - p[0])

    print(f"{'arm':12s} {'n':>2} {'val mean':>9} {'sd':>7} {'slope':>8} {'sd':>7}")
    print("-" * 50)
    for arm in ("gdn4", "theta 0", "theta 0.02", "theta 0.1"):
        if arm not in val:
            continue
        vs, ss = val[arm], slope.get(arm, [float("nan")])
        sdv = st.stdev(vs) if len(vs) > 1 else float("nan")
        sds = st.stdev(ss) if len(ss) > 1 else float("nan")
        print(f"{arm:12s} {len(vs):>2} {st.mean(vs):>9.4f} {sdv:>7.4f} "
              f"{st.mean(ss):>+8.3f} {sds:>7.3f}")

    print("\npairwise on val (Welch, two-sided):")
    arms = [a for a in ("theta 0", "theta 0.02", "theta 0.1", "gdn4")
            if len(val.get(a, [])) > 1]
    for a, b in itertools.combinations(arms, 2):
        t, dof, se = welch(val[a], val[b])
        d = st.mean(val[a]) - st.mean(val[b])
        try:
            from scipy.special import betainc
            p = float(betainc(dof / 2, 0.5, dof / (dof + t * t)))
            ps = f"p={p:.3f}"
        except Exception:
            ps = "p=n/a"
        flag = "" if abs(t) > 2.5 else "   NOT RESOLVED"
        print(f"  {a:11s} - {b:11s} = {d:+.4f} +/- {se:.4f}  "
              f"t={t:+.2f} dof={dof:.1f} {ps}{flag}")


if __name__ == "__main__":
    main()
