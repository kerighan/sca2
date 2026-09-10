"""
Shift test across ALL sca2 variants vs GDN, plus a mega-plot of every curve.

Question: did we pick the champion (shape_dv56) because it converges fast
(high-loss regime) rather than because it is genuinely steeper at low loss?
If another variant has a better slope-at-matched-loss profile, it may be
the better long-run candidate.

    python shift_all.py
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

LOG = "runs/cdelta.jsonl"

GDN = ["gdn4_s3", "gdn4_s4", "gdn4_s5"]

# All SCA2 variant families, ordered by dv where applicable
FAMILIES = {
    "dv24":  ["shape_dv24_s0"],
    "dv40":  ["shape_dv40_s0"],
    "dv48":  ["shape_dv48_s0", "shape_dv48_s1", "shape_dv48_s2"],
    "dv56":  ["shape_dv56_s0", "shape_dv56_s1", "shape_dv56_s2"],
    "dv64":  ["shape_dv64_s0"],
    "cdelta_t0":   ["cdelta_t0", "cdelta_t0_s1", "cdelta_t0_s2"],
    "cdelta_t02":  ["cdelta_t02", "cdelta_t02_s1", "cdelta_t02_s2"],
    "md4_t02":     ["md4_t02_s0", "md4_t02_s1", "md4_t02_s2"],
}

COLORS = {
    "dv24": "tab:gray",
    "dv40": "tab:olive",
    "dv48": "tab:green",
    "dv56": "tab:blue",
    "dv64": "tab:purple",
    "cdelta_t0":  "tab:cyan",
    "cdelta_t02": "tab:pink",
    "md4_t02":    "tab:brown",
}


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
    all_loss = []
    all_slope = []
    for lb in labels:
        if lb not in runs_dict:
            continue
        d = runs_dict[lb]
        toks = np.array([x[0] for x in d], float)
        vals = np.array([x[1] for x in d], float)
        ln_t = np.log(toks)
        for i in range(1, len(d) - 1):
            dl = (vals[i + 1] - vals[i - 1]) / (ln_t[i + 1] - ln_t[i - 1])
            all_loss.append(vals[i])
            all_slope.append(dl)
    return np.array(all_loss), np.array(all_slope)


def smooth_slope(loss_arr, slope_arr, grid, bandwidth=0.20):
    out = np.full_like(grid, np.nan)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((loss_arr - g) / bandwidth) ** 2)
        if w.sum() < 1e-6:
            continue
        fit = np.polyfit(loss_arr - g, slope_arr, 1, w=w)
        out[i] = fit[1]
    return out


def main():
    all_labels = GDN + [lb for ls in FAMILIES.values() for lb in ls]
    runs = load_labels(all_labels)

    # GDN reference
    dl, ds = slope_vs_loss(runs, GDN)

    # Shared loss range
    lo = max(dl.min(), max(
        slope_vs_loss(runs, ls)[0].min() for ls in FAMILIES.values()
        if any(lb in runs for lb in ls)))
    hi = min(dl.max(), min(
        slope_vs_loss(runs, ls)[0].max() for ls in FAMILIES.values()
        if any(lb in runs for lb in ls)))
    grid = np.linspace(lo, hi, 80)
    d_smooth = smooth_slope(dl, ds, grid)

    # Per-family shift test
    print(f"{'family':>12} {'ratio@hi_loss':>13} {'ratio@lo_loss':>13} "
          f"{'median_ratio':>13} {'endpoint':>9}")
    print("-" * 70)

    results = {}
    for fam, labels in FAMILIES.items():
        if not any(lb in runs for lb in labels):
            continue
        gl, gs = slope_vs_loss(runs, labels)
        g_smooth = smooth_slope(gl, gs, grid)
        mask = np.isfinite(g_smooth) & np.isfinite(d_smooth)
        if mask.sum() < 3:
            continue
        ratios = d_smooth[mask] / g_smooth[mask]
        # ratio at high loss (start) and low loss (end)
        hi_mask = mask & (grid > np.median(grid))
        lo_mask = mask & (grid < np.median(grid))
        r_hi = np.median(ratios[hi_mask]) if hi_mask.sum() > 0 else np.nan
        r_lo = np.median(ratios[lo_mask]) if lo_mask.sum() > 0 else np.nan
        # endpoint loss (mean of seeds at max tokens)
        endpoints = [runs[lb][-1][1] for lb in labels if lb in runs]
        ep = np.mean(endpoints)
        results[fam] = {
            "g_smooth": g_smooth, "ratios": ratios,
            "r_hi": r_hi, "r_lo": r_lo, "endpoint": ep,
        }
        print(f"{fam:>12} {r_hi:13.3f} {r_lo:13.3f} "
              f"{np.median(ratios):13.3f} {ep:9.3f}")

    gdn_ep = np.mean([runs[lb][-1][1] for lb in GDN if lb in runs])
    print(f"{'GDN':>12} {'(ref)':>13} {'(ref)':>13} {'(ref)':>13} {gdn_ep:9.3f}")

    print("\nInterpretation:")
    print("  ratio > 1  => GDN steeper at same loss (variant will be caught)")
    print("  ratio < 1  => variant steeper (variant pulling ahead)")
    print("  r_lo > 1   => GDN catches up at the END (bad for variant)")
    print("  r_lo < 1   => variant still ahead at the END (good)")

    # Best variant at low loss
    best_lo = min(results.items(), key=lambda x: x[1]["r_lo"])
    print(f"\nBest at low loss: {best_lo[0]} (ratio {best_lo[1]['r_lo']:.3f})")
    best_ep = min(results.items(), key=lambda x: x[1]["endpoint"])
    print(f"Best endpoint:   {best_ep[0]} (loss {best_ep[1]['endpoint']:.3f})")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: slope vs loss, all families + GDN
    ax1.scatter(dl, ds, c="tab:red", alpha=0.3, s=15, zorder=5)
    ax1.plot(grid, d_smooth, "r-", lw=2.5, label="GDN", zorder=6)
    for fam, res in results.items():
        gl, gs = slope_vs_loss(runs, FAMILIES[fam])
        ax1.scatter(gl, gs, c=COLORS.get(fam, "k"), alpha=0.15, s=10)
        ax1.plot(grid, res["g_smooth"], color=COLORS.get(fam, "k"),
                 lw=1.8, alpha=0.85, label=fam)
    ax1.set_xlabel("validation loss (nats)")
    ax1.set_ylabel("d(loss) / d(ln tokens)")
    ax1.set_title("Slope at matched loss — all variants")
    ax1.legend(fontsize=8, ncol=2)
    ax1.invert_xaxis()
    ax1.axhline(0, color="gray", lw=0.5)

    # Right: loss vs tokens, all curves
    for lb in GDN:
        if lb in runs:
            d = runs[lb]
            ax2.plot([x[0] / 1e6 for x in d], [x[1] for x in d],
                     "r-", alpha=0.2, lw=0.8)
    # GDN mean
    gdn_d = [runs[lb] for lb in GDN if lb in runs]
    min_len = min(len(d) for d in gdn_d)
    toks = np.array([x[0] for x in gdn_d[0][:min_len]], float)
    mean_vals = np.mean([np.array([x[1] for x in d[:min_len]], float)
                         for d in gdn_d], axis=0)
    ax2.plot(toks / 1e6, mean_vals, "r-", lw=3, label="GDN mean", zorder=10)

    for fam, labels in FAMILIES.items():
        fam_d = [runs[lb] for lb in labels if lb in runs]
        if not fam_d:
            continue
        # individual seeds
        for d in fam_d:
            ax2.plot([x[0] / 1e6 for x in d], [x[1] for x in d],
                     color=COLORS.get(fam, "k"), alpha=0.12, lw=0.7)
        # mean
        min_len = min(len(d) for d in fam_d)
        toks = np.array([x[0] for x in fam_d[0][:min_len]], float)
        mean_vals = np.mean([np.array([x[1] for x in d[:min_len]], float)
                             for d in fam_d], axis=0)
        ax2.plot(toks / 1e6, mean_vals,
                 color=COLORS.get(fam, "k"), lw=2, alpha=0.9, label=fam)

    ax2.set_xlabel("tokens (M)")
    ax2.set_ylabel("validation loss (nats)")
    ax2.set_title("All loss curves vs GDN")
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_xscale("log")
    ax2.set_ylim(2.5, 4.5)

    fig.tight_layout()
    fig.savefig("shift_all.png", dpi=150)
    print("\nsaved shift_all.png")


if __name__ == "__main__":
    main()
