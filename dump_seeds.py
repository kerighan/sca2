"""Read runs/seeds.jsonl and settle the two open questions.

Q1  Mc or C-state? md4_dv16 has more Mc than md4_dv32 (504 vs 378) but the same
    C state as deep4 (16128 vs 16384). Near 2.919 => state is the operative
    variable; below 2.876 => Mc is.

Q2  Is SCA2's remaining deficit real? n=3 seeds for md4_dv32 and gdn4, reported
    as mean +/- sd of the final val, with a two-sided Welch t on 3v3. With n=3
    this can only detect an effect a few times the seed sd, so a null result
    here means "not resolved", not "equal".

Seed 0 of each arm lives in the earlier logs, which are pulled in by label.

    python dump_seeds.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
# label in the log -> (arm it belongs to, seed)
SEED0 = {"md4_dv32": ("md4_dv32", 0), "gdn4": ("gdn4", 0),
         "deep4": ("deep4", 0), "md16_dv32": ("md16_dv32", 0),
         "md4_dv16": ("md4_dv16", 0)}
CSTATE = {"md4_dv32": 24192, "md4_dv16": 16128, "deep4": 16384,
          "md16_dv32": 10624, "gdn4": 10800}
MC = {"md4_dv32": 378, "md4_dv16": 504, "deep4": 128, "md16_dv32": 166}


def load():
    runs = defaultdict(list)
    for f in ("seeds.jsonl", "md_axis.jsonl", "confirm_pycode.jsonl"):
        p = RUNS / f
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            lab = r["model"]
            if lab.endswith(("_s1", "_s2")):
                arm, seed = lab[:-3], int(lab[-1])
            elif lab in SEED0:
                arm, seed = SEED0[lab]
            else:
                continue
            runs[(arm, seed)].append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["step"])
    return runs


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

    # Every run does exactly one epoch, but its LAST eval lands wherever the
    # 120s eval clock put it -- 167M to 176M tokens here. val is still falling
    # at that point, so a final-eval comparison rewards whichever run happened
    # to be evaluated latest. Interpolate all runs to a common token count.
    common = min(v[-1]["tokens"] for v in runs.values())
    print(f"all runs interpolated to {common/1e6:.1f}M tokens "
          f"(each did one full epoch of 177.4M)\n")
    print(f"{'arm':12s} {'seed':>4} {'val@common':>11} {'val end':>8} "
          f"{'end tok':>8} {'pos slope':>10} {'Mc':>5} {'C state':>8}")
    print("-" * 76)
    by_arm = defaultdict(list)
    for (arm, seed), v in sorted(runs.items()):
        last = v[-1]
        p = last.get("pos")
        slope = (p[-1] - p[0]) if p else float("nan")
        vc = float(np.interp(common, [r["tokens"] for r in v],
                             [r["val"] for r in v]))
        by_arm[arm].append(vc)
        print(f"{arm:12s} {seed:>4} {vc:>11.4f} {last['val']:>8.4f} "
              f"{last['tokens']/1e6:>7.1f}M {slope:>+10.3f} "
              f"{MC.get(arm, 0):>5} {CSTATE.get(arm, 0):>8}")

    print("\nQ1  Mc or C-state?")
    for k in ("md4_dv32", "md4_dv16", "deep4"):
        if k in by_arm:
            print(f"  {k:10s} Mc={MC[k]:4d} state={CSTATE[k]:6d} "
                  f"val={st.mean(by_arm[k]):.4f} (n={len(by_arm[k])})")

    print("\nQ2  seed spread")
    for arm, vals in sorted(by_arm.items()):
        if len(vals) > 1:
            print(f"  {arm:12s} n={len(vals)} mean={st.mean(vals):.4f} "
                  f"sd={st.stdev(vals):.4f}  {[round(v, 4) for v in vals]}")
        else:
            print(f"  {arm:12s} n=1 val={vals[0]:.4f}")
    a, b = by_arm.get("md4_dv32", []), by_arm.get("gdn4", [])
    if len(a) > 1 and len(b) > 1:
        t, dof = welch(a, b)
        print(f"\n  md4_dv32 - gdn4 = {st.mean(a)-st.mean(b):+.4f} nats, "
              f"Welch t={t:+.2f} on {dof:.1f} dof")
        print("  |t| < 2.5 at this dof means NOT RESOLVED, not equal.")


if __name__ == "__main__":
    main()
