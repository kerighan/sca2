"""Val-loss curves and gap-to-reference for pretrain.py JSONL logs. Generic: every label
in the log is drawn unless hidden; the reference arm is the one the gaps are taken to.

    python plot_lm.py runs/long5h.jsonl --ref l5_gdn_s0 --out plot/long5h.png
    python plot_lm.py runs/long5h.jsonl --ref l5_gdn_s0 --hide l5_B_s0 --bold l5_Aropew64_s0
    python plot_lm.py runs/catchup.jsonl --ref catch_gdn_s0 --only catch_gen3_s0,catch_shortdamp_s0

Also prints, per arm, the median gap to the reference over the last 25% of the shared
token range and the gap's slope vs ln(tokens) beyond 200M (the crossover diagnostic).
"""
import argparse, collections, json, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path, wallclock=False):
    C = collections.OrderedDict()
    tps = {}  # model -> list of tok_s for median throughput
    for l in open(path):
        r = json.loads(l)
        m = r["model"]
        if wallclock:
            # x-axis = equivalent wall-clock seconds, normalised by each arm's
            # own throughput so arms that run faster get MORE tokens per second
            # of wall time.  train_s is the actual elapsed training time.
            if "train_s" in r:
                C.setdefault(m, []).append((r["train_s"], r["val"]))
        else:
            C.setdefault(m, []).append((r["tokens"], r["val"]))
    return {k: sorted(v) for k, v in C.items()}


def ip(c, x):
    if x < c[0][0] or x > c[-1][0]:
        return None
    for i in range(1, len(c)):
        if c[i][0] >= x:
            (x0, y0), (x1, y1) = c[i - 1], c[i]
            return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("log"); p.add_argument("--ref", required=True)
    p.add_argument("--out", default=None); p.add_argument("--hide", default=""); p.add_argument("--only", default="")
    p.add_argument("--bold", default="", help="labels drawn thick (default: ref and the newest arm)")
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--wallclock", action="store_true",
                   help="x-axis = training wall-clock seconds instead of tokens")
    a = p.parse_args(argv)
    C = load(a.log, wallclock=a.wallclock)
    if a.ref not in C:
        sys.exit(f"ref {a.ref!r} not in log; labels: {list(C)}")
    hide = [h for h in a.hide.split(",") if h]; only = [o for o in a.only.split(",") if o]
    labels = [k for k in C if (not only or k in only or k == a.ref) and not any(h in k for h in hide)]
    bold = set(b for b in a.bold.split(",") if b) or {a.ref, labels[-1]}
    G = C[a.ref]
    out = a.out or f"plot/{a.log.split('/')[-1].split('.')[0]}.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    cmap = plt.get_cmap("tab10")
    # perplexity = exp(val loss), from the MEDIAN of the last 3 evals (single evals swing by ±0.02-0.05 nats)
    print(f"{'arm':>20} {'tokens':>7} {'val':>7} | {'gap to '+a.ref:>18} {'slope/ln(tok) >200M':>20} | {'ppl (last 3)':>12}")
    for j, k in enumerate(labels):
        c = C[k]
        xdiv = 60.0 if a.wallclock else 1e6  # seconds -> minutes, or tokens -> M
        x = np.array([t for t, _ in c]) / xdiv; y = np.array([v for _, v in c])
        lw = 2.4 if k in bold else 1.0; col = "#2c3e50" if k == a.ref else cmap(j % 10)
        done = c[-1][0] >= 0.95 * G[-1][0]
        ax1.plot(x[1:], y[1:], color=col, lw=lw, alpha=1 if lw > 1.5 else 0.75,
                 label=k + ("" if done else f"  (running, {c[-1][0]/1e6:.0f}M)"))
        ppl = math.exp(float(np.median([v for _, v in c[-3:]])))
        if k == a.ref:
            print(f"{k:>20} {c[-1][0]/1e6:6.0f}M {c[-1][1]:7.4f} | {'(reference)':>18} {'':>20} | {ppl:12.2f}"); continue
        hi = min(c[-1][0], G[-1][0])
        lo_cut = 600.0 if a.wallclock else 60e6     # skip the first 10min / 60M
        step   = 60.0  if a.wallclock else 10e6     # grid: 1min / 10M
        lo_gap = 300.0 if a.wallclock else 40e6     # gap starts here
        hi_200 = 3600.0 if a.wallclock else 200e6   # "reliable zone" for slope
        if hi < lo_cut: continue
        xs = np.arange(lo_gap, hi + 1, step); g = np.array([ip(c, v) - ip(G, v) for v in xs])
        ax2.plot(xs / xdiv, g, color=col, lw=0.7, alpha=0.3)
        if len(g) >= 5:
            ax2.plot(xs[2:-2] / xdiv, np.convolve(g, np.ones(5) / 5, "valid"), color=col, lw=lw, label=k + " (5-pt smooth)")
        last = xs >= hi - 0.25 * (hi - lo_gap)
        m = xs >= hi_200
        slope = np.polyfit(np.log(xs[m]), g[m], 1)[0] if m.sum() >= 4 else float("nan")
        print(f"{k:>20} {c[-1][0]/1e6:6.0f}M {c[-1][1]:7.4f} | median {np.median(g[last]):+.4f} (last 25%)  {slope:+.3f} | {ppl:12.2f}")
    ax1.set_ylabel("val loss (nats)"); ax1.grid(alpha=.3); ax1.legend(fontsize=8)
    if a.ymax: ax1.set_ylim(top=a.ymax)
    floors = [min(v for _, v in C[k][3:]) for k in labels if len(C[k]) > 3]
    if floors:
        ax1.set_ylim(bottom=min(floors) - 0.05)
    xlabel = "wall-clock (min)" if a.wallclock else "tokens (M)"
    mode = "wall-clock" if a.wallclock else "matched tokens"
    ax1.set_title(f"{a.log}  --  val loss ({mode}); gaps to {a.ref}")
    ax2.axhline(0, color="k", lw=0.8); ax2.set_xlabel(xlabel); ax2.set_ylabel(f"gap to {a.ref} (nats)"); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=130); print("saved", out)


if __name__ == "__main__":
    main()
