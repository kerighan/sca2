"""Largest string length a stack can copy PERFECTLY, as a function of depth.

    python bench_lmax.py --depths 1,2,3,4 --arm v3polarflat_cc

For each depth this walks UP a ladder of lengths and stops at the first length
that is not solved, so the reported L_max is the last solved rung. Cost is
proportional to the answer rather than to the ladder.

Why one length per run instead of a mix
---------------------------------------
A mixed-length curriculum cannot measure this. Training on a mix that contains
an out-of-capacity length collapses the lengths that ARE in capacity -- same
layer, same padding, same budget, L=128 exact-match drops 0.97 -> 0.33 once
L=256 joins the mix (plot/fig5_contamination.png). Since "out of capacity"
means something different at each depth, a fixed mix would penalise the shallow
arms twice and the measurement would be circular. So every cell here trains on
a single length, with Tmax = 2L+2 and no padding.

L_max is a LOWER BOUND at the given step budget: a rung counts as failed if it
is not solved within --steps, and some rungs are still improving at the cut.
Raising --steps can only move L_max up.
"""
import argparse
import json
import time
from pathlib import Path

import bench_copy

LADDER = [16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512]


def cell(arm, depth, L, a):
    """Train one (depth, length) cell. Returns (solved_at_step or None, exact)."""
    spec = f"{arm}/layers={depth}" + (f":{a.extra}" if a.extra else "")
    argv = ["--arms", spec,
            "--lengths", str(L),
            "--steps", str(a.steps),
            "--eval-every", str(a.eval_every),
            "--eval-batches", str(a.eval_batches),
            "--target", str(a.target),
            "--Mc", str(a.Mc),
            "--d", str(a.d),
            "--batch", str(a.batch),
            "--log", a.log]
    res = bench_copy.main(argv)[0]
    return res["solved"], res["exact"][L], res["state"], res["s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="v3polarflat_cc")
    p.add_argument("--extra", default="",
                   help="extra ':'-separated bench_copy overrides for every "
                        "cell, e.g. gdn_head_k=71 to state-match GDN")
    p.add_argument("--depths", default="1,2,3,4")
    p.add_argument("--ladder", default=",".join(str(v) for v in LADDER))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--target", type=float, default=0.98,
                   help="exact-match that counts as PERFECTLY solved")
    p.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    p.add_argument("--eval-batches", type=int, default=4, dest="eval_batches")
    p.add_argument("--Mc", type=int, default=128)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--log", default="runs/lmax_cells.jsonl")
    p.add_argument("--out", default="runs/lmax.json")
    a = p.parse_args()

    depths = [int(v) for v in a.depths.split(",")]
    ladder = [int(v) for v in a.ladder.split(",")]
    out = {"arm": a.arm, "extra": a.extra, "Mc": a.Mc, "d": a.d, "steps": a.steps,
           "target": a.target, "ladder": ladder, "depths": {}}
    t0 = time.time()
    for depth in depths:
        rungs, lmax = [], None
        for L in ladder:
            solved, exact, state, secs = cell(a.arm, depth, L, a)
            rungs.append({"L": L, "solved": solved, "exact": exact,
                          "state": state, "s": secs})
            print(f"\n>>> depth {depth}  L={L}  "
                  f"{'SOLVED at ' + str(solved) if solved else 'FAILED'}  "
                  f"exact={exact:.2f}  state={state}  {secs:.0f}s", flush=True)
            if solved:
                lmax = L
            else:
                break
        out["depths"][str(depth)] = {"L_max": lmax, "rungs": rungs}
        Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"\n{'='*54}\nL_max per depth ({a.arm}, Mc={a.Mc}, d={a.d}, "
          f"target {a.target}, <={a.steps} steps)\n{'='*54}")
    print(f"{'depth':>6s} {'L_max':>7s} {'steps@L_max':>12s}")
    for depth in depths:
        d = out["depths"][str(depth)]
        at = next((r["solved"] for r in d["rungs"] if r["L"] == d["L_max"]), None)
        print(f"{depth:6d} {str(d['L_max']):>7s} {str(at):>12s}")
    print(f"\ntotal {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
