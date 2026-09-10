"""Figures for the confirmation run (runs/confirm_pycode.jsonl).

The sweep proposed two levers, depth and c_decay, and projected that depth would
carry SCA2 past GDN. It did not: the projections carried ~0.2 nats of noise, as
the base/gc_off control had already warned. This run replaces them with endpoints
from a full epoch -- all four arms exhausted the same 177.4M-token corpus at the
same step 21641, so equal-tokens is exact rather than interpolated.

Three panels:
  * val vs tokens: the equal-token verdict, with the 2-layer references.
  * val vs seconds: the other verdict, which goes the other way. SCA2 is 1.43x
    faster, so deep4 finishes the epoch in 1681s where gdn4 needs 2401s.
  * position profile: depth, not decay, is what makes SCA2 use distance -- the
    finding that overturns the sweep's reading.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent / "runs"

C = {"gdn4": "#d62728", "deep4": "#1f77b4", "decay_deep4": "#9467bd",
     "decay2": "#2ca02c", "GDNpy": "#d62728", "SCA2py": "#1f77b4"}
LAB = {"gdn4": "GDN, 4 layers", "deep4": "SCA2, 4 layers",
       "decay_deep4": "SCA2, 4 layers + c_decay",
       "decay2": "SCA2, 2 layers + c_decay",
       "GDNpy": "GDN, 2 layers (ref)", "SCA2py": "SCA2, 2 layers (ref)"}


def load(path):
    d = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        d[r["model"]].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["step"])
    return d


def main():
    d = load(RUNS / "confirm_pycode.jsonl")
    ref = load(RUNS / "pycode.jsonl")
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 3.9))

    for k, v in d.items():
        tok = np.array([r["tokens"] for r in v], float)
        val = np.array([r["val"] for r in v], float)
        sec = np.array([r["train_s"] for r in v], float)
        ax[0].plot(tok, val, "-", color=C[k], lw=1.6, label=f"{LAB[k]}  {val[-1]:.3f}")
        ax[1].plot(sec, val, "-", color=C[k], lw=1.6, label=LAB[k])
    # the 2-layer references, dotted: same corpus, same protocol, earlier run
    for k, v in ref.items():
        tok = [r["tokens"] for r in v]
        val = [r["val"] for r in v]
        ax[0].plot(tok, val, ":", color=C[k], lw=1.2, alpha=.8,
                   label=f"{LAB[k]}  {val[-1]:.3f}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("tokens seen (one epoch = 177.4M)", fontsize=9)
    ax[0].set_ylabel("val loss (nats/token)", fontsize=9)
    ax[0].set_title("equal tokens: GDN still ahead by 0.081", fontsize=9)

    ax[1].axvline(1681, color="grey", ls=":", lw=1)
    ax[1].annotate("deep4 epoch done\n1681s", (1681, 3.25), fontsize=7.5,
                   xytext=(-46, 10), textcoords="offset points", color="grey")
    ax[1].set_xlabel("training seconds (eval excluded)", fontsize=9)
    ax[1].set_ylabel("val loss (nats/token)", fontsize=9)
    ax[1].set_ylim(2.75, 3.6)
    ax[1].set_title("equal wall clock: SCA2 ahead (1.43x faster)", fontsize=9)

    for src in (d, ref):
        for k, v in src.items():
            p = v[-1].get("pos")
            if not p:
                continue
            xs = np.arange(len(p)) * 1024 // len(p)
            ax[2].plot(xs, p, "-o" if src is d else ":s", color=C[k], ms=3.5,
                       lw=1.6 if src is d else 1.2,
                       label=f"{LAB[k]}  ({p[-1]-p[0]:+.3f})")
    ax[2].set_xlabel("position in the 1024-token window", fontsize=9)
    ax[2].set_ylabel("val loss at that position", fontsize=9)
    ax[2].set_title("depth, not decay, buys distance", fontsize=9)

    for a_ in ax:
        a_.grid(alpha=.3, lw=.5); a_.tick_params(labelsize=8)
        a_.legend(fontsize=7)
    fig.suptitle("codeparrot-clean, whole-file windows T=1024, one epoch, "
                 "~743k layer params in every 4-layer arm", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_confirm_pycode.png", dpi=160)
    print("wrote fig9_confirm_pycode.png")


if __name__ == "__main__":
    main()
