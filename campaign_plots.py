"""Every figure of the Zyda-2 campaign, regenerated from runs/zyda.jsonl.

    python campaign_plots.py

`vast.curves --plot` draws the campaign at its own scale, where the 20 h arms
are a smudge in the first inch. The three figures here are the ones that scale
does not show: the early hours against their own control, the router occupancy
that motivated mix8, and the frequency grid mixanch is measuring for us.

Host factors come from vast.common's recorded table, not from the live pool:
an arm on a 5.6% slower card gets 5.6% fewer tokens per hour, which at the
fitted slope is worth more than most of the effects measured here, and the
hosts two of these arms ran on no longer exist to be asked.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vast.common import arm_factors

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "runs" / "zyda.jsonl"
OUT = ROOT / "plot"          # where every figure in this repo has always gone

# arm -> slot it ran on, so elapsed seconds can be put on one clock
SLOT = {"z_dv256": 0, "z_mix8wide": 0, "z_dirichlet": 0, "z_dirichletg": 0,
        "z_gdn": 2, "z_mix8m32": 2, "z_lambdaonly": 2,
        "z_mixanch": 3, "z_dirichlet4": 3,
        "z_mix4": 1, "z_mixsal": 1, "z_mix8": 1, "z_dv384": 1, "z_dsoft": 1}
COL = {"z_gdn": "k", "z_dv256": "tab:purple", "z_mix4": "tab:orange", "z_mix8wide": "tab:brown", "z_mix8m32": "tab:cyan",
       "z_mix8": "tab:blue", "z_mixanch": "tab:green", "z_mixsal": "tab:red",
       "z_lambdaonly": "tab:olive", "z_dirichletg": "tab:pink",
       "z_dirichlet4": "tab:cyan", "z_dirichlet": "0.6"}
R_OF = {"z_mix4": 4, "z_mix8": 8, "z_mix8wide": 8, "z_mix8m32": 8, "z_mixanch": 4,
        "z_mixsal": 4, "z_dv256": 1, "z_lambdaonly": 8}


def host_factors() -> dict[str, float]:
    """arm -> multiply its elapsed seconds by this to get reference-host seconds.

    The table lives in vast.common because vast.curves needs the same one:
    slot numbers shift on teardown and destroyed hosts cannot be queried.
    """
    return arm_factors(SLOT)



def load() -> dict[str, list[dict]]:
    h = collections.defaultdict(list)
    for line in LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("model") in SLOT and r.get("val") is not None:
            h[r["model"]].append(r)
    for m in h:
        h[m].sort(key=lambda r: r["train_s"])
    return h


def fig_zoom(h, fac, arms, ref="z_mix4", hours=8.0, name="zyda_zoom.png"):
    """Early hours against the control the experimental arms were built from."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    # the step-1 eval is the untrained model: it compresses the y axis by two
    # nats and says nothing about any arm
    # reshape(-1, 2): an arm whose only eval is the untrained step-1 one filters
    # down to an EMPTY list, and np.array([]) is 1-D, so the column index below
    # raises instead of drawing nothing.
    xy = {m: np.array([(r["train_s"] * fac[m] / 3600, r["val"])
                       for r in h[m] if r["train_s"] > 300],
                      dtype=float).reshape(-1, 2) for m in arms}
    rf = xy[ref]
    for m in arms:
        a = xy[m][xy[m][:, 0] <= hours]
        if not len(a):
            continue
        lw = 2.4 if m in (ref, "z_gdn") else 1.8
        a1.plot(a[:, 0], a[:, 1], label=m[2:], lw=lw, marker="o", ms=3, color=COL[m])
        if m != ref:
            a2.plot(a[:, 0], a[:, 1] - np.interp(a[:, 0], rf[:, 0], rf[:, 1]),
                    label=m[2:], lw=lw, marker="o", ms=3, color=COL[m])
    a2.axhline(0, color=COL[ref], lw=2)
    # the noise floor, measured on a pair whose true separation is known
    a2.axhspan(-0.008, 0.008, color="0.85", zorder=0,
               label="plancher de bruit +/-0.008")
    a1.set_ylabel("val loss (nats)")
    a2.set_ylabel(f"ecart a {ref[2:]} (nats)")
    a2.set_xlabel("wall clock normalise hote (h)")
    a1.set_title(f"Zyda-2 — {hours:.0f} premieres heures, controle = {ref[2:]} "
                 "(negatif = mieux)")
    for ax in (a1, a2):
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    print(f"saved {OUT.name}/{name}")


def fig_router(h, arms=("z_mix4", "z_mix8", "z_mix8wide", "z_mixanch")):
    """Effective kernels exp(aH): does a router given 8 keep more than one given 4?

    Plotted against tokens, not hours: the routers are compared at equal data
    seen, and the arms differ in throughput by more than the effect.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in arms:
        if m not in h:
            continue
        R = R_OF[m]
        t = np.array([r["tokens"] / 1e9 for r in h[m]])
        A = np.array([r["alpha_H"] for r in h[m]], float)
        k = np.exp(A.mean(1))
        a1.plot(t, k, label=f"{m[2:]} (R={R})", lw=2, marker="o", ms=3.5, color=COL[m])
        a1.fill_between(t, np.exp(A.min(1)), np.exp(A.max(1)), alpha=.15, color=COL[m])
        a1.axhline(R, color=COL[m], ls=":", lw=1)
        a2.plot(t, A.mean(1) / math.log(R) * 100, label=f"{m[2:]} (R={R})",
                lw=2, marker="o", ms=3.5, color=COL[m])
    a1.set_xlabel("tokens vus (B)"); a1.set_ylabel("noyaux effectifs exp(aH)")
    a1.set_title("occupation du routeur (bande = min/max sur les 8 couches)\n"
                 "pointilles = le plafond R")
    a2.set_xlabel("tokens vus (B)"); a2.set_ylabel("saturation aH / ln R (%)")
    a2.set_title("saturation relative : 100% = routeur uniforme")
    for ax in (a1, a2):
        ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "zyda_router.png", dpi=110)
    print(f"saved {OUT.name}/zyda_router.png")


def fig_omega(h, m="z_mixanch"):
    """What the learned frequency grid is asking for, per layer.

    omega_stats logs [drift, log-span, n_eff] per layer. n_eff is how flat the
    grid is (higher = more frequencies carry weight), log-span how wide.
    """
    if m not in h or not h[m][0].get("omega"):
        print(f"no omega stats on {m}")
        return
    t = np.array([r["tokens"] / 1e9 for r in h[m]])
    O = np.array([r["omega"] for r in h[m]], float)        # (evals, layers, 3)
    L = O.shape[1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    cm = plt.cm.viridis(np.linspace(0, .92, L))
    for i in range(L):
        ax[0].plot(t, O[:, i, 2], color=cm[i], lw=1.8, label=f"L{i}")
        ax[1].plot(t, O[:, i, 1], color=cm[i], lw=1.8)
        ax[2].plot(t, O[:, i, 0], color=cm[i], lw=1.8)
    ax[0].axhline(O[0, :, 2].mean(), color="r", ls="--", lw=1.4, label="init")
    ax[1].axhline(O[0, :, 1].mean(), color="r", ls="--", lw=1.4)
    ax[0].set_ylabel("n_eff (frequences portantes)")
    ax[1].set_ylabel("log-span de la grille")
    ax[2].set_ylabel("drift cumule depuis l'init")
    ax[0].set_title("plus haut = grille plus plate que l'init")
    ax[1].set_title("plus bas = intervalle plus etroit")
    ax[2].set_title("pas de plateau = omega n'a pas converge")
    for a in ax:
        a.set_xlabel("tokens vus (B)"); a.grid(alpha=.3)
    ax[0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"{m[2:]} — ce que la grille de frequences apprise reclame, couche par couche")
    fig.tight_layout(); fig.savefig(OUT / "zyda_omega.png", dpi=110)
    print(f"saved {OUT.name}/zyda_omega.png")

    o = O[-1]
    print(f"\n{m[2:]} a {t[-1]:.2f}B, par couche")
    print("  n_eff " + " ".join(f"L{i}:{x:.0f}" for i, x in enumerate(o[:, 2])))
    print("  span  " + " ".join(f"L{i}:{x:.1f}" for i, x in enumerate(o[:, 1])))
    print(f"  corr(n_eff, span) = {np.corrcoef(o[:, 1], o[:, 2])[0, 1]:+.3f}")


def fig_mdose(h, fac, ref="z_mix8wide", arm="z_mix8m32", base="z_mix8"):
    """The M dose-response, which the mix4-controlled zoom cannot show.

    mix8wide and mix8m32 differ in M alone: same R=8, same post-norm, same
    dv=512. Drawn twice because the two readings disagree and only one of them
    is the decision. At equal TOKENS a smaller M can only lose, since it is
    strictly less machinery per token. At equal WALL CLOCK it also runs 5.7%
    faster and sees more tokens, and wall clock is what a card costs.
    """
    if arm not in h or ref not in h:
        print("M dose-response: both arms not present yet")
        return
    def xy(m, col):
        v = [(r["tokens"] / 1e9 if col == 0 else r["train_s"] * fac[m] / 3600, r["val"])
             for r in h[m] if r["train_s"] > 300]
        return np.array(v, dtype=float).reshape(-1, 2)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
    for col, xlab in ((0, "tokens vus (B)"), (1, "wall clock normalise hote (h)")):
        a, b = xy(ref, col), xy(arm, col)
        for m, d in ((base, xy(base, col)), (ref, a), (arm, b)):
            if len(d):
                ax[0][col].plot(d[:, 0], d[:, 1], label=m[2:], lw=2,
                                marker="o", ms=3, color=COL[m])
        lo, hi = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
        g = np.linspace(lo, hi, 60)
        ax[1][col].plot(g, np.interp(g, b[:, 0], b[:, 1]) - np.interp(g, a[:, 0], a[:, 1]),
                        lw=2, color=COL[arm])
        ax[1][col].axhline(0, color=COL[ref], lw=2)
        ax[1][col].axhspan(-0.008, 0.008, color="0.85", zorder=0,
                           label="plancher de bruit +/-0.008")
        ax[0][col].set_ylabel("val loss (nats)")
        ax[1][col].set_ylabel("M=32 moins M=128 (nats)")
        ax[1][col].set_xlabel(xlab)
        ax[0][col].set_title(f"a {'tokens appaires' if col == 0 else 'WALL CLOCK egal'}")
        for a_ in (ax[0][col], ax[1][col]):
            a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.suptitle("dose-reponse sur M a dv=512 — negatif = moins de modes est MIEUX")
    fig.tight_layout(); fig.savefig(OUT / "zyda_mdose.png", dpi=110)
    print(f"saved {OUT.name}/zyda_mdose.png")

    for col, lab in ((0, "tokens"), (1, "wall clock")):
        a, b = xy(ref, col), xy(arm, col)
        g = np.linspace(max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0]), 6)
        d = np.interp(g, b[:, 0], b[:, 1]) - np.interp(g, a[:, 0], a[:, 1])
        print(f"  a {lab:11s} " + "  ".join(f"{x:.2f}:{y:+.4f}" for x, y in zip(g, d)))


def fig_heads(h, fac, ref="z_mix8",
              arms=("z_lambdaonly", "z_dirichletg", "z_dirichlet4", "z_dirichlet")):
    """Which half of the layer carries the model.

    Three panels because the loss alone does not explain itself. The third is
    the one that does: b0 - b15 of the per-position loss is what the model gains
    from 4000 tokens of history rather than 250, and the Dirichlet head's
    receptive field is ~layers*(L-1) = 1016 of a 4096 block.
    """
    live = [m for m in arms if m in h and len(h[m]) > 1]
    if ref not in h or not live:
        print("head ablation: not enough data yet")
        return
    def xy(m, col):
        v = [(r["tokens"] / 1e9 if col == 0 else r["train_s"] * fac[m] / 3600,
              r["val"], r["pos"][0] - r["pos"][-1])
             for r in h[m] if r["train_s"] > 300 and r.get("pos")]
        return np.array(v, dtype=float).reshape(-1, 3)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for m in [ref] + live:
        a = xy(m, 0)
        if not len(a):
            continue
        lw = 2.4 if m == ref else 1.8
        lab = m[2:] + (" (non gate)" if m == "z_dirichlet" else "")
        ax[0].plot(a[:, 0], a[:, 1], label=lab, lw=lw, marker="o", ms=3, color=COL[m])
        ax[2].plot(a[:, 0], a[:, 2], label=lab, lw=lw, marker="o", ms=3, color=COL[m])
    r1 = xy(ref, 1)
    for m in live:
        b = xy(m, 1)
        if not len(b):
            continue
        ax[1].plot(b[:, 0], b[:, 1] - np.interp(b[:, 0], r1[:, 0], r1[:, 1]),
                   label=m[2:], lw=1.8, marker="o", ms=3, color=COL[m])
    ax[1].axhline(0, color=COL[ref], lw=2, label=ref[2:])
    ax[1].axhspan(-0.008, 0.008, color="0.85", zorder=0, label="bruit +/-0.008")
    ax[0].set_xlabel("tokens vus (B)"); ax[0].set_ylabel("val loss (nats)")
    ax[0].set_title("val a tokens appaires")
    ax[1].set_xlabel("wall clock normalise hote (h)")
    ax[1].set_ylabel(f"ecart a {ref[2:]} (nats)")
    ax[1].set_title("a WALL CLOCK egal — negatif = mieux que les deux tetes")
    ax[2].set_xlabel("tokens vus (B)"); ax[2].set_ylabel("gain de contexte b0 - b15")
    ax[2].set_title("ce que le modele tire de 4000 tokens plutot que 250")
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.suptitle("Zyda-2 — quelle moitie du layer porte le modele")
    fig.tight_layout(); fig.savefig(OUT / "zyda_heads.png", dpi=110)
    print(f"saved {OUT.name}/zyda_heads.png")

    a0 = xy(ref, 0)
    tmax = min(xy(m, 0)[-1, 0] for m in live if len(xy(m, 0)))
    print(f"\n  a {tmax:.2f}B tokens, ecart a {ref[2:]} et gain de contexte")
    print(f"    {ref[2:]:12s}  ---       {np.interp(tmax, a0[:,0], a0[:,2]):+.4f}")
    for m in live:
        b = xy(m, 0)
        print(f"    {m[2:]:12s} {np.interp(tmax, b[:,0], b[:,1]) - np.interp(tmax, a0[:,0], a0[:,1]):+.4f}"
              f"   {np.interp(tmax, b[:,0], b[:,2]):+.4f}")


def main() -> None:
    h, fac = load(), host_factors()
    print("arms: " + ", ".join(f"{m[2:]}({len(h[m])})" for m in h))
    fig_zoom(h, fac, ["z_gdn", "z_mix4", "z_mix8", "z_mix8wide", "z_mix8m32",
                      "z_mixanch", "z_mixsal", "z_dv256"], hours=10.0)
    fig_router(h)
    fig_omega(h)
    fig_mdose(h, fac)
    fig_heads(h, fac)


if __name__ == "__main__":
    main()
