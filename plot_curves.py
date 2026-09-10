"""
What the loss curves actually look like: generation 3 against GDN.

Three panels, because one is misleading. The full curve is dominated by the first
10M tokens, where every arm falls 6 nats and the two are indistinguishable at the
scale of the ink; a reader shown only that concludes the arms are identical. The
zoom is where the comparison lives, and the gap panel is where the finding lives.

The individual seeds are drawn as thin lines under every mean. This is deliberate:
GDN's seed spread is 4x generation 3's, which is the single most important fact
about every comparison in this project, and a plot of two mean curves hides it
completely.

The trend line in the bottom panel is fitted only on the shaded-out region's
complement -- the power-law stretch -- and its continuation past the data is drawn
dashed because this project has been burned by exactly that extrapolation before
(confirm_pycode.sh: two fits differing by 0.074 became 0.20 nats when extended).

    python plot_curves.py                     # the campaign's 177M runs
    python plot_curves.py --log runs/longrun.jsonl   # tomorrow morning
    python plot_curves.py --show
"""
import argparse
import glob
import json
import math
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter   # noqa: E402


def _mfmt(scale=1.0):
    """Plain '120M' ticks on a log axis; 4x10^1 is unreadable for token counts."""
    def f(v, _):
        t = v * scale
        if t <= 0:
            return ""
        if t >= 1e9:
            return f"{t/1e9:g}B"
        if t >= 1e6:
            return f"{t/1e6:g}M"
        if t >= 1e3:
            return f"{t/1e3:g}k"
        return f"{t:g}"
    return FuncFormatter(f)

CAMPAIGN = ["runs/cdelta.jsonl", "runs/seeds.jsonl", "runs/confirm_pycode.jsonl",
            "runs/md_axis.jsonl"]

# label -> (display, colour, z-order weight). Generations 1 and 2 are context.
ARMS_CAMPAIGN = {
    "gen 3  Mc=190 dv=56": (["shape_dv56_s0", "shape_dv56_s1", "shape_dv56_s2"],
                            "#1a6fdf"),
    "GDN": (["gdn4", "gdn4_s1", "gdn4_s2", "gdn4_s3", "gdn4_s4", "gdn4_s5"],
            "#d1495b"),
    "gen 2  Mc=378 dv=32": (["cdelta_t02", "cdelta_t02_s1", "cdelta_t02_s2"],
                            "#7a9e9f"),
    "gen 1  additive": (["md4_dv32", "md4_dv32_s1", "md4_dv32_s2"], "#b9a44c"),
}
ARMS_LONGRUN = {
    "gen 3  Mc=190 dv=56": (["long_gen3_s0", "long_gen3_s1"], "#1a6fdf"),
    "GDN": (["long_gdn_s0", "long_gdn_s1"], "#d1495b"),
}
MAIN = ("gen 3  Mc=190 dv=56", "GDN")


def load(paths):
    runs = defaultdict(list)
    for p in paths:
        for f in glob.glob(p):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "tokens" in r and "val" in r and "model" in r:
                    runs[r["model"]].append((r["tokens"], r["val"]))
    return {k: np.array(sorted(set(v)), float).T for k, v in runs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="single log (e.g. the longrun)")
    ap.add_argument("--out", default="curves.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--zoom-from", type=float, default=40e6, dest="zoom_from")
    a = ap.parse_args()

    if a.log:
        runs, ARMS = load([a.log]), ARMS_LONGRUN
    else:
        runs, ARMS = load(CAMPAIGN), ARMS_CAMPAIGN
    ARMS = {k: v for k, v in ARMS.items() if any(s in runs for s in v[0])}
    if not all(m in ARMS for m in MAIN):
        print("missing one of the two main arms -- nothing to compare")
        return

    print(f"{'arm':22} {'n':>2}  {'endpoint':>9}  {'sd':>7}")
    curves = {}
    for name, (labels, col) in ARMS.items():
        cs = [runs[s] for s in labels if s in runs]
        curves[name] = (cs, col)
        ends = [c[1][-1] for c in cs]
        print(f"{name:22} {len(cs):2d}  {np.mean(ends):9.4f}  "
              f"{(np.std(ends, ddof=1) if len(cs) > 1 else float('nan')):7.4f}")

    hi = min(max(c[0][-1] for c in cs) for cs, _ in curves.values())
    lo = a.zoom_from if a.zoom_from < 0.5 * hi else 0.2 * hi
    grid = np.linspace(lo, hi, 40)

    def interp(cs):
        return np.array([np.interp(grid, c[0], c[1]) for c in cs])

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(9.5, 11.5),
        gridspec_kw=dict(height_ratios=[1.05, 1.25, 1.0], hspace=0.30))

    # ---- panel 1: the whole descent, which is the misleading one -------------
    for name, (cs, col) in curves.items():
        main = name in MAIN
        for c in cs:
            ax0.plot(c[0], c[1], color=col, lw=0.7,
                     alpha=0.55 if main else 0.25)
        ax0.plot([], [], color=col, lw=2 if main else 1.2, label=name,
                 alpha=1 if main else 0.5)
    ax0.axvspan(grid[0], hi, color="0.85", alpha=0.5, lw=0)
    ax0.set_xscale("log")
    ax0.set_ylabel("val loss (nats)")
    ax0.set_title("Full descent — the arms are indistinguishable at this scale.\n"
                  "Grey band is what panel 2 zooms into.", fontsize=10, loc="left")
    ax0.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax0.grid(alpha=0.25)

    # ---- panel 2: the zoom, where the comparison is legible ------------------
    for name, (cs, col) in curves.items():
        main = name in MAIN
        ys = interp(cs)
        for y in ys:
            ax1.plot(grid / 1e6, y, color=col, lw=0.7,
                     alpha=0.5 if main else 0.22)
        ax1.plot(grid / 1e6, ys.mean(0), color=col, lw=2.4 if main else 1.3,
                 alpha=1 if main else 0.55,
                 label=f"{name}  (n={len(cs)})")
        if main and len(cs) > 1:
            sd = ys.std(0, ddof=1)
            ax1.fill_between(grid / 1e6, ys.mean(0) - sd, ys.mean(0) + sd,
                             color=col, alpha=0.14, lw=0)
    ax1.set_xscale("log")
    ax1.set_ylabel("val loss (nats)")
    ax1.set_title("Zoom on the power-law stretch. Thin lines are individual "
                  "seeds, band is ±1 sd.\nNote GDN's spread is ~4x generation 3's"
                  " — that, not our variance, is what limits every p-value.",
                  fontsize=10, loc="left")
    ax1.legend(fontsize=8.5, framealpha=0.9)
    ax1.grid(alpha=0.25)

    # ---- panel 3: the gap, where the finding is -----------------------------
    g3, c3 = curves[MAIN[0]][0], curves[MAIN[0]][1]
    gd = curves[MAIN[1]][0]
    A, Bm = interp(g3), interp(gd)
    gap = A.mean(0) - Bm.mean(0)
    sem = np.sqrt(A.var(0, ddof=1) / len(A) + Bm.var(0, ddof=1) / len(Bm)) \
        if min(len(A), len(Bm)) > 1 else np.zeros_like(gap)

    x = np.log(grid)
    slope, icept = np.polyfit(x, gap, 1)
    ax2.axhline(0, color="0.35", lw=1.1, ls="-")
    ax2.fill_between(grid / 1e6, gap - sem, gap + sem, color=c3, alpha=0.18, lw=0)
    ax2.plot(grid / 1e6, gap, color=c3, lw=2.4, label="gen 3 − GDN  (±1 sem)")
    ax2.plot(grid / 1e6, slope * x + icept, color="0.2", lw=1.3, ls="--",
             label=f"fit: {slope:+.3f} nats per e-fold")

    # continuation, dashed and labelled as extrapolation
    if slope > 0:
        cross = math.exp(-icept / slope)
        if hi < cross < 40 * hi:
            xe = np.linspace(hi, min(cross * 1.25, 8 * hi), 40)
            ax2.plot(xe / 1e6, slope * np.log(xe) + icept, color="0.55",
                     lw=1.1, ls=":")
            ax2.axvline(cross / 1e6, color="0.55", lw=1, ls=":")
            ax2.annotate(f"fitted crossing\n{cross/1e6:.0f}M — EXTRAPOLATION,\n"
                         "do not quote",
                         xy=(cross / 1e6, 0), xytext=(cross / 1e6, gap.min() * 0.55),
                         fontsize=8, color="0.35", ha="center")
    ax2.annotate(f"{gap[-1]:+.4f}\nat {hi/1e6:.0f}M",
                 xy=(hi / 1e6, gap[-1]), xytext=(-14, 34),
                 textcoords="offset points", fontsize=9, ha="right",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    ax2.set_xscale("log")
    ax2.set_xlabel("training tokens (M, log scale)")
    ax2.set_ylabel("gap (nats)  —  negative = gen 3 ahead")
    ax2.set_title("The gap is CLOSING, monotonically. Below zero means generation 3"
                  " is ahead;\nthe trend says its advantage is shrinking as tokens"
                  " grow.", fontsize=10, loc="left")
    ax2.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
    ax2.grid(alpha=0.25)

    ax0.xaxis.set_major_formatter(_mfmt(1.0))
    ax0.xaxis.set_minor_formatter(NullFormatter())
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(_mfmt(1e6))
        ax.xaxis.set_minor_formatter(_mfmt(1e6))
        ax.tick_params(axis="x", which="minor", labelsize=7.5)

    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"\nfit window {lo/1e6:.0f}M..{hi/1e6:.0f}M   "
          f"d(gap)/d(ln tokens) = {slope:+.4f}")
    print(f"endpoint gap {gap[-1]:+.4f}  ->  wrote {a.out}")
    if a.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()
