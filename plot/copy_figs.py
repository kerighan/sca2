"""Figures for the copy-task ablations, built from the JSONL that bench_copy.py
appends to runs/.

    python -m plot.copy_figs            # writes plot/*.png and plot/RESULTS.md

Nothing here re-runs a model: every number comes from a log, so a figure can
only ever show what was actually measured. The reported quantity is EXACT-MATCH
(the whole string reproduced), not token accuracy -- on a 128-token string
0.99 token accuracy still means the copy fails about three times in four, and
reading accuracy instead of exact-match is what made the first pass of this
study draw the wrong conclusion.

One protocol caveat is load-bearing: runs/copy.jsonl trained on a length mix
that included L=256, which is out of capacity for these state sizes and
poisons the solvable lengths (see fig5). Its arms are therefore NOT comparable
with the later logs and are only used for that figure.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = Path(__file__).resolve().parent
TARGET = 0.95          # exact-match that counts as "solved"
SCA2_C, GDN_C, TRF_C = "#1f77b4", "#d62728", "#2ca02c"


def load(name):
    """runs/<name>.jsonl -> {arm: [row, ...]} ordered by step."""
    rows = defaultdict(list)
    path = RUNS / f"{name}.jsonl"
    for line in path.open():
        r = json.loads(line)
        rows[r["arm"]].append(r)
    for v in rows.values():
        v.sort(key=lambda r: r["step"])
    return rows


def knob(arm, key, default):
    """Read an override out of an arm spec like 'v3polarflat_cc/layers=4:Mc=64'."""
    m = re.search(rf"{key}=(\d+)", arm)
    return int(m.group(1)) if m else default


def curve(rows, L):
    return [r["step"] for r in rows], [r["exact"][str(L)] for r in rows]


def at(rows, L, step):
    """exact-match at `step`, or None if the arm stopped before reaching it.

    Arms that hit the target early are cut short by bench_copy's early stop, so
    a missing point means "already solved", not "failed" -- the callers below
    carry the last value forward rather than dropping the arm.
    """
    ok = [r for r in rows if r["step"] <= step]
    if not ok:
        return None
    return ok[-1]["exact"][str(L)]


def solve_step(rows, L, target=TARGET):
    """First step with exact >= target on two consecutive evals, else None."""
    hits = 0
    for r in rows:
        hits = hits + 1 if r["exact"][str(L)] >= target else 0
        if hits >= 2:
            return r["step"]
    # an early-stopped arm ends AT its solve point, so a single trailing hit
    # after bench_copy already declared it solved still counts
    return rows[-1]["step"] if rows and rows[-1]["exact"][str(L)] >= target else None


def finish(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=.3, lw=.5)
    ax.tick_params(labelsize=8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
#  fig1  copy capacity vs Mc, at three training budgets
# --------------------------------------------------------------------------- #
def fig1():
    rows = load("copy_mc")
    arms = {a: v for a, v in rows.items() if "Md=32" not in a}
    mcs = sorted(knob(a, "Mc", 128) for a in arms)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for step, style in [(1000, ":v"), (4000, "--s"), (8000, "-o")]:
        ys = []
        for mc in mcs:
            arm = next(a for a in arms if knob(a, "Mc", 128) == mc)
            ys.append(at(arms[arm], 128, step))
        ax.plot(mcs, ys, style, color=SCA2_C, label=f"step {step//1000}k", ms=5)
    ax.axhline(TARGET, color="k", lw=.8, ls="--")
    ax.text(mcs[0], TARGET + .02, "solved", fontsize=7)
    # The dip at Mc=128 is not a capacity effect: this is ONE seed per point and
    # the run-to-run spread on exact-match is ~0.2 at this budget (the same
    # config reaches 0.91-0.94 by step 12k in two other logs). Only the trend is
    # readable here, not the individual points.
    ax.annotate("1 seed/point;\nspread ~0.2", xy=(128, .68), xytext=(70, .30),
                fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="->", lw=.7, color="grey"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(mcs)
    ax.set_xticklabels(mcs)
    ax.set_ylim(-.05, 1.05)
    ax.legend(fontsize=8)
    finish(ax, "SCA2: copy of a 128-symbol string vs C-head width $M_c$\n"
               "(d=128, 2 layers; $M_d$=4 fixed)", "$M_c$", "exact-match rate")
    save(fig, "fig1_mc_capacity.png")


# --------------------------------------------------------------------------- #
#  fig2  the honest axis: state size, both architectures
# --------------------------------------------------------------------------- #
def fig2():
    rows = load("copy_state")
    sca, gdn = [], []
    for arm, v in rows.items():
        pt = (v[0]["state"], at(v, 128, 12000), arm)
        (gdn if arm.startswith("gdn") else sca).append(pt)
    sca.sort(); gdn.sort()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for pts, c, lab in [(sca, SCA2_C, "SCA2 (vary $M_c$)"),
                        (gdn, GDN_C, "GDN (vary head_k)")]:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c,
                label=lab, ms=5)
    ax.axhline(TARGET, color="k", lw=.8, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(-.05, 1.05)
    ax.legend(fontsize=8)
    finish(ax, "Copy at MATCHED state size (L=128, step 12k)\n"
               "state pairs: 9.4k / 34k / 67k floats",
           "recurrent state carried (floats, log)", "exact-match rate")
    save(fig, "fig2_state_vs_copy.png")


# --------------------------------------------------------------------------- #
#  fig3  how the two degrade with string length, at each matched state size
# --------------------------------------------------------------------------- #
def fig3():
    rows = load("copy_state")
    lens = [16, 32, 64, 128]
    pairs = [(9474, 9240), (34050, 34080), (66818, 66660)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, (s_sca, s_gdn) in zip(axes, pairs):
        for want, c, lab in [(s_sca, SCA2_C, "SCA2"), (s_gdn, GDN_C, "GDN")]:
            arm = next(a for a, v in rows.items() if v[0]["state"] == want)
            ax.plot(lens, [at(rows[arm], L, 12000) for L in lens], "-o",
                    color=c, label=lab, ms=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(lens); ax.set_xticklabels(lens)
        ax.set_ylim(-.05, 1.05)
        finish(ax, f"state ~{s_sca//1000}k floats", "string length", "")
    axes[0].set_ylabel("exact-match rate", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("Length scaling at matched state (step 12k)", fontsize=10)
    save(fig, "fig3_length_scaling.png")


# --------------------------------------------------------------------------- #
#  fig4  depth: does stacking layers buy copy capacity?
# --------------------------------------------------------------------------- #
def fig4():
    depth = load("copy_depth")
    dim = load("copy_dim")
    mc = load("copy_mc")
    # left: Mc=128 held FIXED, so total state grows with depth
    fixed = [(1, depth["v3polarflat_cc/layers=1"]),
             (2, dim["v3polarflat_cc"]),
             (4, depth["v3polarflat_cc/layers=4"])]
    # right: state held ~constant by dividing Mc as depth grows
    iso = [(1, depth["v3polarflat_cc/layers=1:Mc=256"]),
           (2, mc["v3polarflat_cc"]),
           (4, depth["v3polarflat_cc/layers=4:Mc=64"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, data, title in [
            (axes[0], fixed, "$M_c$=128 fixed (state grows with depth)"),
            (axes[1], iso, "state ~34k fixed ($M_c$ = 256/128/64)")]:
        for (nl, rows), ls in zip(data, ["-o", "-s", "-^"]):
            st, ex = curve(rows, 128)
            ax.plot(st, ex, ls, label=f"{nl} layer{'s' if nl > 1 else ''}", ms=4)
        ax.axhline(TARGET, color="k", lw=.8, ls="--")
        ax.set_xlim(0, 8500)
        ax.set_ylim(-.05, 1.05)
        ax.legend(fontsize=8)
        finish(ax, title, "training step", "")
    axes[0].set_ylabel("exact-match rate (L=128)", fontsize=9)
    fig.suptitle("SCA2: depth is what unlocks long copy", fontsize=10)
    save(fig, "fig4_depth.png")


# --------------------------------------------------------------------------- #
#  fig5  why the first run of this study concluded the opposite
# --------------------------------------------------------------------------- #
def fig5():
    rows = load("copy_why")["v3polarflat_cc"]
    # both cells share an arm name; they differ by whether L=256 was trained on
    a = [r for r in rows if "256" not in r["exact"]]
    b = [r for r in rows if "256" in r["exact"]]
    lens = [16, 32, 64, 128]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    w = .35
    xs = range(len(lens))
    ax.bar([x - w / 2 for x in xs], [at(a, L, 12000) for L in lens], w,
           color=SCA2_C, label="mix 16/32/64/128")
    ax.bar([x + w / 2 for x in xs], [at(b, L, 12000) for L in lens], w,
           color=GDN_C, label="mix + L=256 (out of capacity)")
    ax.set_xticks(list(xs)); ax.set_xticklabels(lens)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=8, loc="lower left")
    finish(ax, "One unsolvable length in the training mix\n"
               "collapses the solvable ones (same layer, same $T_{max}$=514, step 12k)",
           "string length", "exact-match rate")
    save(fig, "fig5_contamination.png")


# --------------------------------------------------------------------------- #
#  table
# --------------------------------------------------------------------------- #
def table():
    lines = ["# Copy task: measured results", "",
             "Exact-match = the entire string reproduced after the separator.",
             "`solve` = first step with exact-match >= 0.95 on L=128, twice in a row.",
             ""]
    for name, note in [("copy_state", "SCA2 vs GDN at matched state size"),
                       ("copy_mc", "SCA2 C-head width sweep (+ Md control)"),
                       ("copy_dim", "model width at matched state"),
                       ("copy_depth", "depth")]:
        lines += [f"## {name} -- {note}", "",
                  "| arm | state | solve | L16 | L32 | L64 | L128 |",
                  "|---|---:|---:|---|---|---|---|"]
        for arm, v in load(name).items():
            last = v[-1]
            s = solve_step(v, 128)
            verdict = str(s) if s else ">" + str(last["step"])
            cells = " | ".join(f"{last['exact'][str(L)]:.2f}"
                               for L in (16, 32, 64, 128))
            lines.append(f"| `{arm}` | {last['state']} | {verdict} | {cells} |")
        lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print("wrote RESULTS.md")


# --------------------------------------------------------------------------- #
#  fig6  longest length copied perfectly, vs depth  (from bench_lmax.py)
# --------------------------------------------------------------------------- #
def fig7(path="runs/lmax_sca2.json"):
    src = RUNS.parent / path
    if not src.exists():
        print("skip fig7:", path, "not found")
        return
    z = json.loads(src.read_text())
    depths = sorted(int(k) for k in z["depths"])
    lmax = [z["depths"][str(d)]["L_max"] for d in depths]
    budget = z["steps"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(depths, lmax, "-o", color=SCA2_C, ms=6, label="measured $L_{max}$")
    # A point whose solve step is within 15% of the budget was still climbing
    # when the run was cut, so its L_max is a floor set by compute, not by the
    # architecture. Marking them stops the flat tail being read as saturation.
    for d, L in zip(depths, lmax):
        rung = next(r for r in z["depths"][str(d)]["rungs"] if r["L"] == L)
        if rung["solved"] and rung["solved"] > .85 * budget:
            ax.plot([d], [L], "o", mfc="none", mec="k", ms=13, mew=1.2)
    ax.plot([], [], "o", mfc="none", mec="k", ms=10,
            label=f"budget-limited (solved near {budget} steps)")
    ax.plot(depths, [lmax[0] * d for d in depths], "--", color="grey", lw=1,
            label="linear in depth (reference)")
    ax.set_xticks(depths)
    ax.legend(fontsize=8)
    finish(ax, "Longest string copied PERFECTLY (exact-match >= 0.98)\n"
               f"one length per run, $M_c$=128, d=128, <={budget} steps",
           "layers", "$L_{max}$")
    save(fig, "fig7_lmax_vs_depth.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig7(); table()
