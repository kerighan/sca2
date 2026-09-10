"""
Shift test: is gen3 just a horizontally-shifted copy of GDN?

If two curves have the same shape but one is shifted left in log-token space,
the trailing curve always looks like it's catching up -- it's on a steeper
part of the same descent.  But when it reaches the loss level where the leader
sits now, it will slow down too, and the gap stabilises instead of closing.

The test: compute d(loss)/d(ln tokens) as a function of *loss level* for each
model.  If the slopes match at the same loss, the curves are the same shape
and the "closing gap" is a shift artifact.  If GDN is systematically steeper
at the same loss, it is genuinely learning faster and will cross.

    python shift_test.py
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

LOG = "runs/cdelta.jsonl"
GEN3 = ["shape_dv56_s0", "shape_dv56_s1", "shape_dv56_s2"]
GDN = ["gdn4_s3", "gdn4_s4", "gdn4_s5"]


def load_labels(labels):
    runs = defaultdict(list)
    for line in open(LOG):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["model"] in labels:
            runs[r["model"]].append((r["tokens"], r["val"]))
    return {k: sorted(v) for k, v in runs.items()}


def slope_vs_loss(runs_dict, labels):
    """For each run, compute local d(loss)/d(ln tokens) and pair with loss level."""
    all_loss = []
    all_slope = []
    for lb in labels:
        if lb not in runs_dict:
            continue
        d = runs_dict[lb]
        toks = np.array([x[0] for x in d], float)
        vals = np.array([x[1] for x in d], float)
        ln_t = np.log(toks)
        # local slope via central differences on the sorted eval points
        for i in range(1, len(d) - 1):
            dl = (vals[i + 1] - vals[i - 1]) / (ln_t[i + 1] - ln_t[i - 1])
            all_loss.append(vals[i])
            all_slope.append(dl)
    return np.array(all_loss), np.array(all_slope)


def smooth_slope(loss_arr, slope_arr, grid, bandwidth=0.15):
    """Local polynomial (degree 1) regression of slope on loss."""
    out = np.full_like(grid, np.nan)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((loss_arr - g) / bandwidth) ** 2)
        if w.sum() < 1e-6:
            continue
        fit = np.polyfit(loss_arr - g, slope_arr, 1, w=w)
        out[i] = fit[1]  # intercept = slope at g
    return out


def main():
    runs = load_labels(GEN3 + GDN)

    gl, gs = slope_vs_loss(runs, GEN3)
    dl, ds = slope_vs_loss(runs, GDN)

    # Shared loss range where both have data
    lo = max(gl.min(), dl.min())
    hi = min(gl.max(), dl.max())
    grid = np.linspace(lo, hi, 80)

    g_smooth = smooth_slope(gl, gs, grid)
    d_smooth = smooth_slope(dl, ds, grid)

    # Print table
    print(f"{'loss':>7} {'gen3 dL/dlnT':>14} {'gdn dL/dlnT':>14} {'ratio':>7}")
    for i in range(0, len(grid), 4):
        g, d = g_smooth[i], d_smooth[i]
        r = d / g if g and abs(g) > 1e-6 else float("nan")
        print(f"{grid[i]:7.3f} {g:14.4f} {d:14.4f} {r:7.2f}")

    # Summary: is GDN steeper at the same loss?
    mask = np.isfinite(g_smooth) & np.isfinite(d_smooth)
    ratios = d_smooth[mask] / g_smooth[mask]
    print("\nGDN/gen3 slope ratio at matched loss:")
    print(f"  median {np.median(ratios):.3f}  mean {np.mean(ratios):.3f}")
    print(f"  range [{np.min(ratios):.3f}, {np.max(ratios):.3f}]")
    if np.median(ratios) > 1.05:
        print("  -> GDN is STEEPER at the same loss: genuine catch-up, not a shift.")
    elif np.median(ratios) < 0.95:
        print("  -> GDN is SHALLOWER at the same loss: gen3 pulling away.")
    else:
        print("  -> Slopes MATCH at the same loss: the gap is a shift artifact.")
        print("     The closing is just GDN traversing the same curve later.")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: slope vs loss level
    ax1.scatter(gl, gs, c="tab:blue", alpha=0.4, s=20, label="gen3 evals")
    ax1.scatter(dl, ds, c="tab:orange", alpha=0.4, s=20, label="gdn evals")
    ax1.plot(grid, g_smooth, "b-", lw=2, label="gen3 smoothed")
    ax1.plot(grid, d_smooth, "r-", lw=2, label="gdn smoothed")
    ax1.set_xlabel("validation loss (nats)")
    ax1.set_ylabel("d(loss) / d(ln tokens)")
    ax1.set_title("Slope at matched loss level")
    ax1.legend()
    ax1.invert_xaxis()  # high loss on left = early training
    ax1.axhline(0, color="gray", lw=0.5)

    # Right: classic loss vs tokens (for reference)
    for lb in GEN3:
        if lb in runs:
            d = runs[lb]
            ax2.plot([x[0] / 1e6 for x in d], [x[1] for x in d],
                     "b-", alpha=0.3, lw=0.8)
    for lb in GDN:
        if lb in runs:
            d = runs[lb]
            ax2.plot([x[0] / 1e6 for x in d], [x[1] for x in d],
                     "r-", alpha=0.3, lw=0.8)
    # Means
    for labels, color, name in [(GEN3, "blue", "gen3"), (GDN, "red", "gdn")]:
        all_d = [runs[lb] for lb in labels if lb in runs]
        if not all_d:
            continue
        min_len = min(len(d) for d in all_d)
        toks = np.array([x[0] for x in all_d[0][:min_len]], float)
        mean_vals = np.mean([np.array([x[1] for x in d[:min_len]], float)
                             for d in all_d], axis=0)
        ax2.plot(toks / 1e6, mean_vals, color=color, lw=2.5, label=f"{name} mean")
    ax2.set_xlabel("tokens (M)")
    ax2.set_ylabel("validation loss (nats)")
    ax2.set_title("Loss curves (reference)")
    ax2.legend()
    ax2.set_xscale("log")

    fig.tight_layout()
    fig.savefig("shift_test.png", dpi=150)
    print("\nsaved shift_test.png")


if __name__ == "__main__":
    main()
