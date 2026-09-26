"""Every figure of the Zyda-2 campaign, regenerated from runs/zyda.jsonl.

    python campaign_plots.py

`vast.curves --plot` draws the campaign at its own scale, where the 20 h arms
are a smudge in the first inch. The three figures here are the ones that scale
does not show: the early hours against their own control, the router occupancy
that motivated mix8, and the frequency grid mixanch is measuring for us.

Host factors come from vast/runtime/instances.json (calib_tok_s per slot) via
the same normalisation vast.curves applies: an arm on a 5.6% slower card gets
5.6% fewer tokens per hour, which at the fitted slope is worth more than most
of the effects being measured.
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

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "runs" / "zyda.jsonl"
OUT = ROOT / "plot"          # where every figure in this repo has always gone

# arm -> slot it ran on, so elapsed seconds can be put on one clock
SLOT = {"z_dv256": 0, "z_gdn": 2, "z_mixanch": 3,
        "z_mix4": 1, "z_mixsal": 1, "z_mix8": 1, "z_dv384": 1, "z_dsoft": 1}
COL = {"z_gdn": "k", "z_dv256": "tab:purple", "z_mix4": "tab:orange",
       "z_mix8": "tab:blue", "z_mixanch": "tab:green", "z_mixsal": "tab:red"}
R_OF = {"z_mix4": 4, "z_mix8": 8, "z_mixanch": 4, "z_mixsal": 4, "z_dv256": 1}


def host_factors() -> dict[str, float]:
    pool = {p["slot"]: p for p in json.loads((ROOT / "vast/runtime/instances.json").read_text())}
    rates = {s: p.get("calib_tok_s") for s, p in pool.items()}
    ref = rates.get(0) or next(v for v in rates.values() if v)
    return {a: (rates.get(s) or ref) / ref for a, s in SLOT.items()}


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
    xy = {m: np.array([(r["train_s"] * fac[m] / 3600, r["val"])
                       for r in h[m] if r["train_s"] > 300]) for m in arms}
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


def fig_router(h, arms=("z_mix4", "z_mix8", "z_mixanch")):
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


def main() -> None:
    h, fac = load(), host_factors()
    print("arms: " + ", ".join(f"{m[2:]}({len(h[m])})" for m in h))
    fig_zoom(h, fac, ["z_gdn", "z_mix4", "z_mix8", "z_mixanch", "z_mixsal", "z_dv256"],
             hours=8.0)
    fig_router(h)
    fig_omega(h)


if __name__ == "__main__":
    main()
