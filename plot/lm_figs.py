"""Figures for the TinyPython language-modelling comparison.

    python plot/lm_figs.py runs/pub3.jsonl

Two panels, because the two architectures differ on both axes and only showing
one of them would flatter whichever arm you prefer:

  * val loss vs TOKENS SEEN  -- equal data, the modelling comparison
  * val loss vs TRAINING SECONDS -- equal compute, what you actually pay

An earlier version of this run trained 132 epochs on a 20k-example corpus. Val
loss bottomed out around step 2.5k and then climbed for the remaining 24k
steps, so the arms were being ranked on memorisation. The run plotted here is
capped at one epoch (`--epochs 1`), and the dashed marker shows each arm's best
val loss so a reader can see whether it is still improving at the cut.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
COLOR = {"V3POLARFLAT_CC": "#1f77b4", "GDN_CC": "#d62728",
         "SCA2": "#1f77b4", "TRANSFORMER": "#2ca02c"}
LABEL = {"V3POLARFLAT_CC": "SCA2 (v3polarflat)", "GDN_CC": "Gated DeltaNet"}


def load(path):
    runs, meta = defaultdict(list), {}
    for line in Path(path).open():
        r = json.loads(line)
        if r.get("event") == "start":
            meta[r["model"]] = r
        elif r.get("event") == "eval":
            runs[r["model"]].append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["step"])
    return runs, meta


def main(path="runs/pub3.jsonl"):
    runs, meta = load(path)
    if not runs:
        sys.exit(f"no eval rows in {path}")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for name, rows in runs.items():
        c = COLOR.get(name, "#888888")
        lab = LABEL.get(name, name)
        npar = meta.get(name, {}).get("params")
        if npar:
            lab += f"  ({npar/1e6:.2f}M par)"
        for ax, xk, xlab in [(axes[0], "tokens", "tokens seen"),
                             (axes[1], "train_s", "training seconds (eval excluded)")]:
            ax.plot([r[xk] for r in rows], [r["val"] for r in rows],
                    "-", color=c, lw=1.4, label=lab)
            best = min(rows, key=lambda r: r["val"])
            ax.plot([best[xk]], [best["val"]], "o", color=c, ms=5)
            ax.set_xlabel(xlab, fontsize=9)
            ax.grid(alpha=.3, lw=.5)
            ax.tick_params(labelsize=8)
    axes[0].set_ylabel("val loss (nats/token)", fontsize=9)
    axes[0].legend(fontsize=8)
    ep = max((r["epochs"] for v in runs.values() for r in v), default=0)
    fig.suptitle(f"TinyPython causal LM, 2 layers, d=128, compact vocab "
                 f"({ep:.2f} epoch)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_lm_tinypython.png", dpi=160)
    print("wrote fig6_lm_tinypython.png")
    for name, rows in runs.items():
        best = min(rows, key=lambda r: r["val"])
        print(f"  {name:18s} best val {best['val']:.4f} at step {best['step']}"
              f"  ({best['epochs']:.2f} ep, {best['train_s']:.0f}s, "
              f"{rows[-1]['tok_s']} tok/s)")


if __name__ == "__main__":
    main(*sys.argv[1:])
