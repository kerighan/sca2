"""
Generation 3's throughput, and the chunk sizes that move it, under a BLOCKED
design because the sequential one does not work on this machine.

THE MEASUREMENT PROBLEM, measured. Timing configs one after another gives numbers
that cannot be compared. Re-running the SAME config three times across one such
session gave 71.6k / 65.1k / 76.7k tok/s -- 15.2% apart -- while three consecutive
windows of a single config agree to 2.2%. So the between-config variation this
machine imposes is 7x the within-config noise, and it swamps everything worth
looking for here (a chunk effect of ~5%, the GDN gap of ~8%).

Note the shape of it: 71.6 -> 65.1 -> 76.7 is NOT monotone, so it is not thermal
decay, and "run it on a cold GPU" would not have fixed it. Something exogenous
moves the clock on a minute scale. That also retro-invalidates the tok/s in
runs/*.jsonl -- including the 77405 / 76325 / 70702 spread on dv=56 that was read
as the machine heating up. It was never evidence of that.

THE FIX is to block, not to average harder. Every round times every config
back to back, so whatever the clock is doing in that minute applies to all of
them, and only the RATIO within a round is kept. Drift becomes a nuisance shared
inside each block instead of a bias between configs. Rounds are shuffled so that
position-in-round cannot alias with the effect either.

This is the same pairing idea that FAILED for val loss (see WINNERS.md -- the
paired estimator is retired) and it works here for the reason it failed there:
the nuisance is genuinely shared and genuinely simultaneous. Two arms trained on
different seeds share no such term; two configs timed 3 seconds apart share the
GPU's clock state. Pairing is not a trick to reduce variance, it is a claim about
what the nuisance is, and the claim has to be true.

WHY CHUNK IS A REAL LEVER. Per chunk of C the C head forms a (B,C,2M)x(B,2C,2M)^T
score matrix, so 4.B.C^2.M, and there are T/C chunks: total 4.B.T.C.M, LINEAR in
C. Halving the chunk halves the FLOPs of the dominant term. It does not halve the
state carry -- the einsums against sr/si cost B.T.M.dv per sequence whatever C is
-- and it doubles the sequential loop iterations and their launch overhead. Hence
an interior optimum, and it moves with Mc. Which is the point: Mc just fell
378 -> 190, halving the C-dependent term while launch overhead stayed put, so the
optimum should have moved UP from the 128 the whole campaign has been running.

NOT A QUALITY EXPERIMENT. Chunking is provably exact -- 2.7e-15 across boundaries,
checked by `python -m sca2.arch_cdelta` -- so C cannot change val, only time.

    python sweep_chunk.py                # ~6 min
    python sweep_chunk.py --rounds 4     # smoke test
"""
import argparse
import itertools
import random
import statistics as st
import time

import torch
import torch.nn.functional as F

from bench_tinypython import SCA2
from sca2.arch_cdelta import CHeadDelta
from sca2.fast_dhead import DHeadSepQPolarFlat
from sca2.ref import LayerCfg

# Generation 3's shape and pretrain.py's real training geometry, because the
# optimum is occupancy-dependent and a toy B/T would move it.
SHAPE = dict(d=128, Mc=190, Md=4, G=8, ff=364, dv=56)
B, T, V, LAYERS = 8, 1024, 16384, 4
REF = ("cdelta_cc", 128, 16)     # what the campaign has been running


def build(variant, ctx, dch):
    # The env vars are read at class-definition time, so os.environ would silently
    # do nothing here; the class attributes are the only live knob.
    CHeadDelta.CTX = ctx
    DHeadSepQPolarFlat.CHUNK = dch
    # GDN gets ff=260, not 364. It spends its parameters differently and this is
    # the matching the campaign ran (gdn_seeds.sh): at a common ff=364 GDN carries
    # 212442 params/layer against our 185959, 14% MORE, and a speed comparison at
    # unmatched params flatters us. Do not simplify this away.
    ff = 260 if variant.startswith("gdn") else SHAPE["ff"]
    cfg = LayerCfg(SHAPE["d"], SHAPE["Mc"], SHAPE["Md"], SHAPE["G"], ff,
                   freq="rope", max_len=T, dv=SHAPE["dv"], theta_scale=0.02)
    torch.manual_seed(0)
    return SCA2(V, cfg, variant, "cuda", LAYERS).to("cuda")


class Cell:
    """One config, kept resident with its optimizer so rounds cost only compute."""

    def __init__(self, variant, ctx, dch):
        self.key = (variant, ctx, dch)
        self.name = "gdn" if variant.startswith("gdn") else "cdelta"
        self.ctx, self.dch = ctx, dch
        self.m = build(variant, ctx, dch)
        self.par = self.m.core_params() // LAYERS
        self.opt = torch.optim.AdamW(self.m.parameters(), lr=1e-3)
        self.tps = []

    def step(self, x, y):
        # The chunk knobs are read per forward off the class, so they must be
        # re-set every time: another cell moved them since this one last ran.
        CHeadDelta.CTX = self.ctx
        DHeadSepQPolarFlat.CHUNK = self.dch
        self.opt.zero_grad(set_to_none=True)
        F.cross_entropy(self.m(x).flatten(0, 1), y.flatten()).backward()
        self.opt.step()

    def time(self, x, y, iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            self.step(x, y)
        torch.cuda.synchronize()
        v = B * T * iters / (time.perf_counter() - t0)
        self.tps.append(v)
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warm", type=int, default=12)
    a = ap.parse_args()

    # The LOW end matters most: cost is linear in C, so theory points down, not up.
    # A first blocked pass covered 128..384 x 16,32 and found 128/16 already at the
    # optimum with everything larger monotonically worse -- but ctx=64 and dch=8 had
    # been dropped from that grid on the strength of the CONTAMINATED sequential
    # run, which is not evidence. They are back in.
    grid = [("cdelta_cc", c, d)
            for c, d in itertools.product([48, 64, 96, 128, 192], [8, 16])]
    grid.append(("gdn_cc", 128, 16))

    x = torch.randint(0, V, (B, T), device="cuda")
    y = torch.randint(0, V, (B, T), device="cuda")

    print(f"B={B} T={T} layers={LAYERS} Mc={SHAPE['Mc']} dv={SHAPE['dv']} "
          f"Md={SHAPE['Md']}  rounds={a.rounds} iters={a.iters}")
    cells = []
    for key in grid:
        c = Cell(*key)
        for _ in range(a.warm):                      # absorbs compilation
            c.step(x, y)
        cells.append(c)
        print(f"  built {c.name:6} ctx={c.ctx:<4} dch={c.dch:<3} "
              f"{c.par} par/layer", flush=True)
    torch.cuda.synchronize()

    rng = random.Random(0)
    for r in range(a.rounds):
        order = cells[:]
        rng.shuffle(order)                           # order must not alias
        for c in order:
            c.time(x, y, a.iters)
        print(f"  round {r+1}/{a.rounds} done", flush=True)

    # ------------------------------------------------------------ per-round ratio
    ref = next(c for c in cells if c.key == REF)
    print(f"\n{'variant':8} {'ctx':>5} {'dch':>4}  {'tok/s median':>12} "
          f"{'ratio to 128/16':>16} {'sd of ratio':>11}")
    out = []
    for c in cells:
        ratios = [v / rv for v, rv in zip(c.tps, ref.tps)]
        med, sd = st.median(ratios), st.stdev(ratios)
        out.append((c, med, sd))
        print(f"{c.name:8} {c.ctx:5d} {c.dch:4d}  {st.median(c.tps)/1e3:11.1f}k "
              f"{med:16.3f} {sd:11.3f}")

    # The blocking is only worth something if it actually removed the drift, so
    # show the reference's own raw spread next to the residual ratio noise.
    raw = (max(ref.tps) - min(ref.tps)) / max(ref.tps)
    resid = st.median([sd for _, _, sd in out if sd > 0])
    print(f"\nreference config raw spread across rounds: {raw*100:.1f}%")
    print(f"median residual sd of the within-round ratios: {resid*100:.1f}%")
    print("  The second is what the comparisons below actually rest on. If it is")
    print("  not far below the first, blocking did not help and nothing resolves.")

    cd = [(c, m, s) for c, m, s in out if c.name == "cdelta"]
    best, bm, bs = max(cd, key=lambda t: t[1])
    print(f"\nBEST CHUNK: ctx={best.ctx} dch={best.dch}, x{bm:.3f} on 128/16 "
          f"(+-{bs:.3f})")
    if bm - 1 < 2 * bs:
        print("  Not distinguishable from the campaign's current setting.")
    else:
        print(f"  Real: worth {(bm-1)*100:.1f}% of training time. Set "
              f"SCA2_CTX_CHUNK={best.ctx} SCA2_D_CHUNK={best.dch}.")
    print("  It is the best of 8 cells, so confirm before quoting -- the winner of")
    print("  a search is biased upward even when the search is cheap.")

    g, gm, gs = next(t for t in out if t[0].name == "gdn")
    print(f"\nvs GDN at matched params ({g.par} vs {best.par} per layer):")
    print(f"  best cdelta / gdn = x{bm/gm:.3f}  (+-{(bs+gs):.3f}, added crudely)")
    print(f"  campaign cdelta / gdn = x{1/gm:.3f}")
    print("This is the only throughput claim WINNERS.md may carry, and it is a")
    print("single-session measurement -- rerun it before relying on it.")


if __name__ == "__main__":
    main()
