"""Figures for the long-context Python-code run (runs/pycode.jsonl).

    python plot/pycode_figs.py

Three panels, because this comparison has been misread twice already:

  * val vs TOKENS, log-log. Both arms did exactly one epoch over the same
    177M-token corpus, so this is a like-for-like curve. The dashed lines are
    power-law fits over the last quarter, extended to 2x, which is the only
    honest way to answer "what if we trained twice as long".
  * val vs SECONDS. SCA2 is 1.13x faster, so it leads early; the crossover is
    marked. Reading only the left part of this panel is what makes the run look
    favourable to SCA2.
  * loss by POSITION in the 1024-token window -- the long-context instrument.
    A within-model relative measure, so each arm's own noise largely cancels.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SRC = OUT.parent / "runs" / "pycode.jsonl"
C = {"SCA2py": "#1f77b4", "GDNpy": "#d62728"}
LAB = {"SCA2py": "SCA2 (v3polarflat)", "GDNpy": "Gated DeltaNet"}


def load():
    d = defaultdict(list)
    for line in SRC.open():
        r = json.loads(line)
        d[r["model"]].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["step"])
    return d


def main():
    d = load()
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7))

    for k, v in d.items():
        tok = np.array([r["tokens"] for r in v], float)
        val = np.array([r["val"] for r in v], float)
        sec = np.array([r["train_s"] for r in v], float)
        ax[0].plot(tok, val, "-", color=C[k], lw=1.5, label=LAB[k])
        ax[1].plot(sec, val, "-", color=C[k], lw=1.5, label=LAB[k])
        # power-law fit on the last quarter, extended to twice the corpus
        m = tok >= .75 * tok[-1]
        a, b = np.polyfit(np.log(tok[m]), np.log(val[m]), 1)
        xs = np.array([tok[m][0], 2 * tok[-1]])
        ax[0].plot(xs, np.exp(a * np.log(xs) + b), "--", color=C[k], lw=1,
                   label=f"pente {a:+.3f}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("tokens seen (1 epoch = 177M)", fontsize=9)
    ax[0].set_ylabel("val loss (nats/token)", fontsize=9)
    ax[0].set_title("equal tokens (log-log) + fit extended to 2x", fontsize=9)

    # crossover in the wall-clock view
    s = {k: np.array([r["train_s"] for r in v]) for k, v in d.items()}
    g = np.linspace(60, min(s["SCA2py"][-1], s["GDNpy"][-1]), 400)
    i = {k: np.interp(g, s[k], [r["val"] for r in d[k]]) for k in d}
    diff = i["SCA2py"] - i["GDNpy"]
    sign = np.where(np.diff(np.sign(diff)))[0]
    if len(sign):
        x = g[sign[0]]
        ax[1].axvline(x, color="grey", ls=":", lw=1)
        ax[1].annotate(f"crossover {x:.0f}s", (x, i['SCA2py'][sign[0]]),
                       fontsize=8, xytext=(6, 18), textcoords="offset points")
    ax[1].set_xlabel("training seconds (eval excluded)", fontsize=9)
    ax[1].set_title("equal wall clock", fontsize=9)

    for k, v in d.items():
        p = v[-1].get("pos")
        if p:
            xs = np.arange(len(p)) * 1024 // len(p)
            ax[2].plot(xs, p, "-o", color=C[k], ms=4,
                       label=f"{LAB[k]}  ({p[-1]-p[0]:+.3f})")
    ax[2].set_xlabel("position in the 1024-token window", fontsize=9)
    ax[2].set_ylabel("val loss at that position", fontsize=9)
    ax[2].set_title("does distance get used? (end of run)", fontsize=9)

    for a_ in ax:
        a_.grid(alpha=.3, lw=.5); a_.tick_params(labelsize=8); a_.legend(fontsize=8)
    fig.suptitle("codeparrot-clean, whole-file windows T=1024, 2 layers, "
                 "d=128, parameter-matched, one epoch", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_pycode_long.png", dpi=160)
    print("wrote fig8_pycode_long.png")


if __name__ == "__main__":
    main()
