"""Read runs/wg_confirm.jsonl: is WG=2 a NET win, at n=3 and a full epoch?

Same convention as dump_seeds.py, deliberately: every run does one epoch but its
last eval lands wherever the 120s clock put it (167-176M tokens) and val is still
falling there, so all runs are interpolated to a common token count before being
compared. Control arms are not rerun -- md4_dv32 and gdn4 at n=3 already live in
runs/md_axis.jsonl, runs/seeds.jsonl and runs/confirm_pycode.jsonl.

Two statistics, and they answer different questions:

  val    absolute loss. This is what the 900s probes could NOT resolve: mid
         descent, wg2 minus wg1 swung +0.055, -0.181, -0.183, -0.085, -0.087,
         +0.024 across six evals. At a full epoch the curve has flattened, which
         is the only regime where this number means anything.

  slope  pos[7] - pos[0], a WITHIN-model measure, so each run's own trajectory
         noise largely cancels (see pretrain.py::evaluate). This is where WG=2
         showed a stable -0.08 to -0.13 against the control at every probe eval.

A null on val with a real effect on slope would mean WG=2 redistributes capacity
toward long positions without paying for itself overall -- which is a result, not
a failure. With n=3 only an effect a few times the seed sd is detectable, so a
null is "not resolved", never "equal".

    python dump_wg_confirm.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
LOGS = ("wg_confirm.jsonl", "seeds.jsonl", "md_axis.jsonl", "confirm_pycode.jsonl")
# label in the log -> (arm, seed).  Seed 0 of the controls predates the _s0 naming.
SEED0 = {"md4_dv32": ("md4_dv32", 0), "gdn4": ("gdn4", 0)}


def load():
    runs = defaultdict(list)
    for f in LOGS:
        p = RUNS / f
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            lab = r["model"]
            if lab.endswith(("_s0", "_s1", "_s2")):
                arm, seed = lab[:-3], int(lab[-1])
            elif lab in SEED0:
                arm, seed = SEED0[lab]
            else:
                continue
            runs[(arm, seed)].append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["step"])
    return {k: v for k, v in runs.items() if k[0] in ("wg2", "md4_dv32", "gdn4")}


def welch(a, b):
    """Two-sided Welch t and dof; scipy is not a dependency of this repo."""
    na, nb = len(a), len(b)
    va, vb = st.variance(a) / na, st.variance(b) / nb
    t = (st.mean(a) - st.mean(b)) / (va + vb) ** .5
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, dof


def main():
    runs = load()
    if not runs:
        print("no logs yet")
        return
    common = min(v[-1]["tokens"] for v in runs.values())
    print(f"all runs interpolated to {common/1e6:.1f}M tokens "
          f"(one full epoch is 177.4M)\n")
    print(f"{'arm':10} {'seed':>4} {'val':>8} {'slope':>8} {'end tok':>8} "
          f"{'tok/s':>7}")
    print("-" * 50)

    val, slope = defaultdict(list), defaultdict(list)
    for (arm, seed), v in sorted(runs.items()):
        tk = [r["tokens"] for r in v]
        vc = float(np.interp(common, tk, [r["val"] for r in v]))
        s7 = float(np.interp(common, tk, [r["pos"][7] for r in v]))
        s0 = float(np.interp(common, tk, [r["pos"][0] for r in v]))
        val[arm].append(vc)
        slope[arm].append(s7 - s0)
        print(f"{arm:10} {seed:>4} {vc:>8.4f} {s7-s0:>+8.3f} "
              f"{v[-1]['tokens']/1e6:>7.1f}M {v[-1]['tok_s']:>7}")

    print()
    for name, d in (("val", val), ("slope", slope)):
        print(f"{name}:")
        for arm, xs in sorted(d.items()):
            if len(xs) > 1:
                print(f"  {arm:10} n={len(xs)} mean={st.mean(xs):+.4f} "
                      f"sd={st.stdev(xs):.4f}  {[round(x, 4) for x in xs]}")
            else:
                print(f"  {arm:10} n=1 {xs[0]:+.4f}")
        for a, b in (("wg2", "md4_dv32"), ("wg2", "gdn4")):
            if len(d.get(a, [])) > 1 and len(d.get(b, [])) > 1:
                t, dof = welch(d[a], d[b])
                print(f"  {a} - {b} = {st.mean(d[a])-st.mean(d[b]):+.4f}, "
                      f"Welch t={t:+.2f} on {dof:.1f} dof"
                      f"{'' if abs(t) > 2.5 else '   NOT RESOLVED'}")
        print()

    print("Position profile at the common token count, wg2 minus md4_dv32:")
    for arm in ("wg2", "md4_dv32"):
        if not any(k[0] == arm for k in runs):
            return
    prof = {}
    for arm in ("wg2", "md4_dv32"):
        vs = [v for k, v in runs.items() if k[0] == arm]
        prof[arm] = [st.mean([float(np.interp(common, [r["tokens"] for r in v],
                                              [r["pos"][i] for r in v]))
                              for v in vs]) for i in range(8)]
    for i in range(8):
        d = prof["wg2"][i] - prof["md4_dv32"][i]
        print(f"  {i*128:>5}-{i*128+127:<4} {d:>+8.3f}")


if __name__ == "__main__":
    main()
