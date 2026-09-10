"""Read runs/cdelta.jsonl: does a complex error-correcting write beat the
additive one? And -- decisively for interpreting a null -- did the gate move?

Same convention as dump_seeds.py / dump_wg_confirm.py: every arm runs one epoch
but its last eval lands wherever the 120s clock put it, and val is still falling
there, so all runs are interpolated to a common token count. Controls are read
from the existing n=3 logs rather than rerun.

Read val ONLY, and only at the end. Twice now a mid-descent reading has pointed
the wrong way: wg2 was 0.065 BETTER than the control at 15.3M tokens and ended
0.065 WORSE, and its position slope was stably better throughout while its
profile at the end was uniformly worse. This script prints the mid-descent
column too, precisely so that disagreement stays visible instead of being
quoted as a result.

THE GATE COLUMN IS NOT OPTIONAL. beta = sigmoid(bproj(z)) starts at 0.12 with a
zero weight, and beta = 0 is the additive baseline bit-for-bit. So:

    gate still at ~0.12, flat   -> the arm IS the baseline; a null is
                                   UNINFORMATIVE about erasure, not evidence
                                   against it. Look for an optimisation reason
                                   (is the gradient reaching bproj at all?).
    gate moved up               -> erasure was selected for, and val then means
                                   what it says.
    gate driven to ~0           -> the model actively rejected erasure. That is
                                   a real (negative) result about the mechanism.

    python dump_cdelta.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
LOGS = ("cdelta.jsonl", "seeds.jsonl", "md_axis.jsonl", "confirm_pycode.jsonl")
ARMS = ("cdelta_t0", "cdelta_t02", "cdeltaw_t02", "md4_t02", "md4_dv32",
        "gdn4",
        # shape retrade at matched params (sweep_shape.sh): Mc traded for dv.
        "shape_dv24", "shape_dv40", "shape_dv48", "shape_dv64", "shape_dv56")
SEED0 = {"md4_dv32": ("md4_dv32", 0), "gdn4": ("gdn4", 0),
         "cdelta_t0": ("cdelta_t0", 0), "cdelta_t02": ("cdelta_t02", 0)}
MID = 15.3e6      # where wg2 looked good; kept as an explicit anti-anchor
EPOCH = 177.4e6
RIPE = 0.9 * EPOCH   # below this, no verdict is printed at all


def load():
    runs = defaultdict(list)
    for f in LOGS:
        p = RUNS / f
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            lab = r["model"]
            if lab.endswith(("_s0", "_s1", "_s2", "_s3", "_s4", "_s5")):
                arm, seed = lab[:-3], int(lab[-1])
            elif lab in SEED0:
                arm, seed = SEED0[lab]
            else:
                continue
            if arm in ARMS:
                runs[(arm, seed)].append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["step"])
    return runs


def gates():
    """sigmoid(bias) and |weight| of each layer's beta projection."""
    out = {}
    for arm in ("cdelta_t0", "cdelta_t02", "cdeltaw_t02_s0"):
        # pretrain.py writes f"{--save}.{label.lower()}.pt", so the suffix is
        # the arm name, not "sca2".
        hits = sorted(RUNS.glob(f"ck_{arm}.*.pt"))
        if not hits:
            continue
        sd = torch.load(hits[-1], map_location="cpu")["model"]
        # The state_dict exposes layer 0 a second time under an alias
        # ("layer.layer.c"); keep only the indexed path so the mean is not
        # double-counting it.
        b = [(k, v) for k, v in sd.items()
             if "bproj.bias" in k and k.startswith("layers.")]
        w = {k.replace(".weight", ""): v for k, v in sd.items()
             if "bproj.weight" in k}
        out[arm] = [(k.replace(".bproj.bias", ""),
                     torch.sigmoid(v).item(),
                     w[k.replace(".bias", "")].norm().item() if
                     k.replace(".bias", "") in w else float("nan"))
                    for k, v in b]
    return out


def paired(runs, a, b, span=30e6, k=6):
    """Per-seed mean difference over the last `span` tokens, on a `k`-point grid.

    Why not just interpolate to one common token count, as the table above does:
    the per-eval noise here is +-0.05 (60 eval batches), which is LARGER than the
    0.04 effects this control is measuring. Reading one point inherits all of it,
    and it showed: the seed-0 difference for md4_t02 swung from +0.056 at 155.9M
    to -0.038 at 168.0M, changing sign, purely from where it was read.

    Averaging k points cuts that by ~sqrt(k), and pairing by seed removes
    seed-level variance, which is the dominant term here (gdn4's seed sd is
    0.073-0.149 depending on readout).

    WHAT THIS ESTIMATOR IS NOT: unbiased for the END-OF-EPOCH gap. Using one grid
    for both arms cancels the descent they have in COMMON, but not a difference in
    descent RATE, and these arms do differ there. So this measures the mean gap
    over the window, not the gap at the end, and the two genuinely differ -- for
    cdelta_t02 - md4_dv32 it reads -0.1865 against the endpoint's -0.1577. Neither
    is wrong; they answer different questions. Quote both, and treat a conclusion
    that depends on which one is used as not established.

    NOT PAIRABLE BEYOND THE SHARED SEEDS. gdn4 runs 6 seeds and the SCA2 arms 3,
    so seeds 3-5 have no partner here and are silently dropped -- they raise n only
    for the UNPAIRED table above. And pairing gdn4 against an SCA2 arm by seed
    number is weak to begin with: the two have different parameter shapes, so the
    same --seed does not give them a comparable init, and there is no shared
    nuisance term for the pairing to cancel. The residuals show it
    (cdelta_t02-gdn4: [-0.175, +0.018, -0.093], no consistent sign). Pairing is
    sound for SCA2-vs-SCA2 arms, which are nested; read the GDN rows as unpaired.
    """
    out = {}
    for s in sorted({s for arm, s in runs if arm in (a, b)}):
        va, vb = runs.get((a, s)), runs.get((b, s))
        if not va or not vb:
            continue
        hi = min(va[-1]["tokens"], vb[-1]["tokens"])
        grid = np.linspace(hi - span, hi, k)
        fa = np.interp(grid, [r["tokens"] for r in va], [r["val"] for r in va])
        fb = np.interp(grid, [r["tokens"] for r in vb], [r["val"] for r in vb])
        out[s] = float(np.mean(fa - fb))
    return out


# Two-sided 95% critical values of Student's t. Needed because this campaign runs
# n=3, where dof is 2-4 and the critical value is 2.8-4.3 -- NOT the ~2 that a
# large-sample habit suggests. An earlier version of this script thresholded at
# |t|>2.5 and consequently printed two differences as resolved that are not:
# md4_dv32-gdn4 (t=2.53, dof=2.9, p=0.088) and cdelta_t02-cdelta_t0 (t=2.80,
# dof=3.7, p=0.053). Both were quoted in the docs before this was caught.
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def crit(dof):
    d = max(1.0, min(10.0, dof))
    lo, hi = int(d), min(10, int(d) + 1)
    return TCRIT[lo] + (TCRIT[hi] - TCRIT[lo]) * (d - lo)


def welch(a, b):
    na, nb = len(a), len(b)
    va, vb = st.variance(a) / na, st.variance(b) / nb
    t = (st.mean(a) - st.mean(b)) / (va + vb) ** .5
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, dof


def main():
    all_runs = load()
    if not all_runs:
        print("no evals yet")
        return
    # An arm still in flight used to drag `common` down and mark the WHOLE table
    # mid-descent, which made every settled result unreadable while anything else
    # was running. Exclude it instead: that keeps the same protection (no
    # mid-descent number is ever printed) without hiding finished arms.
    runs = {k: v for k, v in all_runs.items() if v[-1]["tokens"] >= RIPE}
    flying = sorted(k for k in all_runs if k not in runs)
    if flying:
        print("EXCLUDED, still under 90% of an epoch: "
              + ", ".join(f"{a}_s{s} ({all_runs[(a, s)][-1]['tokens']/1e6:.0f}M)"
                          for a, s in flying))
    if not runs:
        print("nothing has reached 90% of an epoch yet")
        return
    common = min(v[-1]["tokens"] for v in runs.values())
    ripe = True
    print(f"interpolated to {common/1e6:.1f}M tokens (a full epoch is 177.4M)")
    print(f"'mid' is val at {MID/1e6:.1f}M, shown only to expose disagreement "
          f"with the end -- it is not a result\n")
    print(f"{'arm':11} {'seed':>4} {'val':>8} {'mid':>8} {'slope':>7} "
          f"{'end tok':>8} {'tok/s':>7}")
    print("-" * 56)

    val, slope = defaultdict(list), defaultdict(list)
    for (arm, seed), v in sorted(runs.items()):
        tk = [r["tokens"] for r in v]
        def at(key, x=common, i=None):
            ys = [(r[key][i] if i is not None else r[key]) for r in v]
            return float(np.interp(x, tk, ys))
        vc, mid = at("val"), at("val", MID)
        s = at("pos", i=7) - at("pos", i=0)
        val[arm].append(vc)
        slope[arm].append(s)
        print(f"{arm:11} {seed:>4} {vc:>8.4f} {mid:>8.4f} {s:>+7.3f} "
              f"{v[-1]['tokens']/1e6:>7.1f}M {v[-1]['tok_s']:>7}")

    print()
    for arm in ARMS:
        if arm in val:
            xs = val[arm]
            sd = f"sd={st.stdev(xs):.4f}" if len(xs) > 1 else "sd=n/a"
            print(f"  {arm:11} n={len(xs)} mean={st.mean(xs):+.4f} {sd}  "
                  f"{[round(x, 4) for x in xs]}")

    # Both reference arms matter now and they say different things: md4_dv32 is
    # the same layer with the additive write (so the pair isolates the delta
    # rule), gdn4 is the target. cdelta_t02 vs cdelta_t0 asks whether the theta
    # INIT matters at all, given both arms drift to large theta anyway.
    pairs = (("cdelta_t02", "md4_dv32"), ("cdelta_t0", "md4_dv32"),
             ("cdelta_t02", "gdn4"), ("cdelta_t0", "gdn4"),
             ("cdelta_t02", "cdelta_t0"), ("md4_dv32", "gdn4"),
             # the speed arm: vs cdelta_t02 is "what did freezing the write
             # phase cost", vs md4_dv32 is "is any of the gain left".
             ("cdeltaw_t02", "cdelta_t02"), ("cdeltaw_t02", "md4_dv32"),
             ("cdeltaw_t02", "gdn4"),
             # the attribution control. md4_dv32 ran at --theta-scale's default
             # of 0.0, so cdelta_t02 - md4_dv32 spans TWO changes. These two
             # pairs split it: the init's effect WITHOUT the delta rule, and the
             # delta rule's effect at matched init 0.02.
             ("md4_t02", "md4_dv32"), ("cdelta_t02", "md4_t02"),
             # the shape retrade, each against the champion shape it replaces.
             ("shape_dv24", "cdelta_t02"), ("shape_dv40", "cdelta_t02"),
             ("shape_dv48", "cdelta_t02"), ("shape_dv64", "cdelta_t02"),
             # dv56 brackets the optimum from above, and does it at LOWER state
             # (21280) than either dv48 (24288) or the champion (24192).
             ("shape_dv56", "cdelta_t02"), ("shape_dv56", "shape_dv48"),
             # THE comparison now: the champion is shape_dv48, not cdelta_t02, so
             # the GDN rows above are about a superseded shape.
             ("shape_dv48", "gdn4"), ("shape_dv56", "gdn4"))
    print()
    for a, b in pairs:
        if a not in val or b not in val:
            continue
        d = st.mean(val[a]) - st.mean(val[b])
        if not ripe:
            print(f"  {a} - {b} = {d:+.4f}   (mid-descent, NOT a verdict)")
            continue
        if len(val[a]) > 1 and len(val[b]) > 1:
            t, dof = welch(val[a], val[b])
            c = crit(dof)
            tag = "" if abs(t) > c else "   NOT RESOLVED"
            print(f"  {a:11} - {b:9} = {d:+.4f}  Welch t={t:+.2f} "
                  f"dof={dof:.1f} crit={c:.2f}{tag}")
        else:
            # The multiplier below is in units of the CONTROL's seed sd, because
            # at n=1 the candidate has none. That understates the noise whenever
            # the candidate is the noisier arm, and it did: shape_dv48 screened at
            # -0.067 = 4.6x cdelta_t02's sd of 0.0147, but its OWN endpoint sd
            # turned out to be 0.0430, so -0.067 was 1.6 of its own sd -- no
            # evidence at all. Selecting the best of 4 such screens then inflates
            # the winner further (its true paired effect is -0.021). Read these
            # lines as ranking only, never as effect sizes.
            sd = st.stdev(val[b])
            print(f"  {a:11} - {b:9} = {d:+.4f}  ({abs(d)/sd:.1f}x the {b} seed "
                  f"sd of {sd:.3f}; n=1, SCREENS only)")

    print("\nseed-PAIRED, averaged over the last 30M tokens (6 points) -- the\n"
          "estimator the table above is too noisy for; see paired.__doc__:")
    for a, b in pairs:
        d = paired(runs, a, b)
        if len(d) < 2:
            continue
        xs = list(d.values())
        m, sd = st.mean(xs), st.stdev(xs)
        t = m / (sd / len(xs) ** .5)
        c = crit(len(xs) - 1)
        tag = "" if abs(t) > c else "   NOT RESOLVED"
        per = " ".join(f"s{s}{v:+.3f}" for s, v in sorted(d.items()))
        print(f"  {a:11} - {b:9} = {m:+.4f}  paired t={t:+.2f} n={len(xs)}"
              f"{tag}   [{per}]")

    print("\nbeta gate after training (init 0.120, |w|=0; beta=0 is the "
          "additive baseline):")
    g = gates()
    if not g:
        print("  no checkpoint yet")
    for arm, rows in g.items():
        for name, b, wn in rows:
            print(f"  {arm:11} {name:22} beta={b:.3f}  |w|={wn:.3f}")
        bs = [b for _, b, _ in rows]
        print(f"  {arm:11} mean beta {st.mean(bs):.3f}"
              + ("   GATE DID NOT MOVE -> a null here is uninformative"
                 if max(abs(b - 0.12) for b in bs) < 0.02 else ""))

    print("\nposition profile at the common token count, arm minus md4_dv32:")
    prof = {}
    for arm in ARMS:
        vs = [v for k, v in runs.items() if k[0] == arm]
        if not vs:
            continue
        prof[arm] = [st.mean([float(np.interp(common, [r["tokens"] for r in v],
                                              [r["pos"][i] for r in v]))
                              for v in vs]) for i in range(8)]
    if "md4_dv32" in prof:
        hdr = [a for a in ("cdelta_t0", "cdelta_t02", "gdn4") if a in prof]
        print("  " + " " * 12 + "".join(f"{a:>12}" for a in hdr))
        for i in range(8):
            row = "".join(f"{prof[a][i]-prof['md4_dv32'][i]:>+12.3f}"
                          for a in hdr)
            print(f"  {i*128:>5}-{i*128+127:<5}{row}")


if __name__ == "__main__":
    main()
