"""Figures for the Mc/Md trade at equal parameters (runs/md_axis.jsonl).

The hypothesis under test: Mc buys copy capacity, Md buys induction, and SCA2's
LM deficit comes from having run every pycode arm at Md=4. It is refuted, and the
sign is the informative part -- moving parameters from Mc into Md costs 0.098
nats AND flattens the position profile, while moving them the other way (halving
dv to fund Mc=378) gains 0.043 and takes the profile to GDN's own slope.

Four arms, all 4 layers, ~743.6k layer params, one epoch over the same corpus:

    gdn4       2.8394   slope -0.283
    md4_dv32   2.8760   slope -0.277   dv 32, Mc 378, Md  4
    deep4      2.9188   slope -0.204   dv 64, Mc 128, Md  4
    md16_dv32  3.0193   slope -0.162   dv 32, Mc 166, Md 16

Panel 3 puts val against the C-head state size, which is the axis all three SCA2
arms actually differ along once Md is seen to be the wrong lever.
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

C = {"gdn4": "#d62728", "md4_dv32": "#1f77b4", "deep4": "#7fbde8",
     "md16_dv32": "#ff7f0e"}
LAB = {"gdn4": "GDN (state 10800)",
       "md4_dv32": "SCA2 dv32 Mc378 Md4",
       "deep4": "SCA2 dv64 Mc128 Md4",
       "md16_dv32": "SCA2 dv32 Mc166 Md16"}
CSTATE = {"md4_dv32": 24192, "deep4": 16384, "md16_dv32": 10624}


def load(*paths):
    d = defaultdict(list)
    for p in paths:
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r["model"] in C:
                d[r["model"]].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["step"])
    return d


def main():
    d = load(RUNS / "md_axis.jsonl", RUNS / "confirm_pycode.jsonl")
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 3.9))
    order = ["gdn4", "md4_dv32", "deep4", "md16_dv32"]

    for k in order:
        v = d[k]
        tok = np.array([r["tokens"] for r in v], float)
        val = np.array([r["val"] for r in v], float)
        ax[0].plot(tok, val, "-", color=C[k], lw=1.6,
                   label=f"{LAB[k]}  {val[-1]:.4f}")
        p = v[-1].get("pos")
        if p:
            xs = np.arange(len(p)) * 1024 // len(p)
            ax[1].plot(xs, p, "-o", color=C[k], ms=3.5, lw=1.6,
                       label=f"{LAB[k]}  ({p[-1]-p[0]:+.3f})")
    ax[0].set_xlim(2e7, 2e8); ax[0].set_ylim(2.8, 3.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("tokens seen (one epoch = 177.4M)", fontsize=9)
    ax[0].set_ylabel("val loss (nats/token)", fontsize=9)
    ax[0].set_title("equal tokens, equal params: Md hurts", fontsize=9)

    ax[1].set_xlabel("position in the 1024-token window", fontsize=9)
    ax[1].set_ylabel("val loss at that position", fontsize=9)
    ax[1].set_title("more Md flattens the profile, more Mc steepens it", fontsize=9)

    xs = [CSTATE[k] for k in CSTATE]
    ys = [d[k][-1]["val"] for k in CSTATE]
    ax[2].plot(xs, ys, "o", ms=9, color="#1f77b4")
    for k in CSTATE:
        ax[2].annotate(f" Md={4 if k != 'md16_dv32' else 16}",
                       (CSTATE[k], d[k][-1]["val"]), fontsize=8, va="center")
    ax[2].axhline(d["gdn4"][-1]["val"], color="#d62728", ls="--", lw=1.2,
                  label=f"GDN {d['gdn4'][-1]['val']:.4f}")
    ax[2].set_xlabel("C-head state floats per layer", fontsize=9)
    ax[2].set_ylabel("val loss at one epoch", fontsize=9)
    ax[2].set_title("the axis that actually moves the loss", fontsize=9)

    for a_ in ax:
        a_.grid(alpha=.3, lw=.5)
        a_.tick_params(labelsize=8)
        a_.legend(fontsize=7)
    fig.suptitle("codeparrot-clean, whole-file windows T=1024, 4 layers, "
                 "~743.6k layer params, one epoch", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_md_axis.png", dpi=160)
    print("wrote fig10_md_axis.png")


if __name__ == "__main__":
    main()
