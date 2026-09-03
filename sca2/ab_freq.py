"""
A/B the two candidate semantic changes on real loss.

Both candidates are principled, so theory cannot pick between them (see
freq_grid's docstring: `dft` is an exact delta at init with one alias spike of
height 1.0 at lag M; `len` is a delta with 1.6% Dirichlet ripple and no alias in
the window). This measures instead.

Two things make the measurement mean something:

  * COMPACT VOCAB. The cached TinyPython tokens use 2278 distinct ids out of
    cl100k's 100277, so in the original setup 98% of parameters and most of the
    loss is vocabulary bookkeeping, which swamps any layer-level effect. Ids are
    remapped to a dense range here. This is measurement hygiene, not a change to
    the benchmark.
  * IDENTICAL RNG CONSUMPTION. Every model is constructed the same way with
    theta_scale=0, then `omega` and `theta` are patched afterwards from a
    dedicated generator. Otherwise `randn(M)` for a non-zero theta shifts every
    later parameter init and the arms differ by more than the knob.

Run:  python -m sca2.ab_freq --steps 1500
"""
import argparse, math, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import LayerCfg, freq_grid
from .registry import build

CACHE = "tinypython_cl100k_20000.pt"


def load_compact(path=CACHE):
    z = torch.load(path)
    tr, va = z["train"], z["val"]
    used = torch.unique(torch.cat([tr, va]))
    lut = torch.full((int(used.max()) + 1,), -1, dtype=torch.long)
    lut[used] = torch.arange(used.numel())
    return lut[tr], lut[va], used.numel()


class TinyLM(nn.Module):
    def __init__(self, V, cfg, variant, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.e = nn.Embedding(V, cfg.d)
        self.on = nn.LayerNorm(cfg.d)
        self.o = nn.Linear(cfg.d, V)
        # build() reseeds, so the layer is bit-identical across arms
        self.layer = build(variant, cfg, seed=seed)

    def forward(self, t):
        y, _ = self.layer.prefill(self.e(t))
        return self.o(self.on(y))


def patch_head(model, freq, theta_scale, M, max_len, seed=1234):
    """Apply the knob AFTER construction so nothing else moves."""
    c = getattr(model.layer, "layer", model.layer).c
    with torch.no_grad():
        c.omega.copy_(freq_grid(freq, M, max_len).to(c.omega))
        if theta_scale == 0.0:
            c.theta.zero_()
        else:
            g = torch.Generator().manual_seed(seed)
            c.theta.copy_(theta_scale * torch.randn(M, generator=g).to(c.theta))


def get_batch(data, B, T, g, device):
    ix = torch.randint(len(data) - T - 1, (B,), generator=g)
    x = torch.stack([data[i:i + T] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + T + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(m, va, B, T, device, nb=25):
    m.eval(); g = torch.Generator().manual_seed(999)
    ls = [F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
          for x, y in (get_batch(va, B, T, g, device) for _ in range(nb))]
    m.train(); return sum(ls) / len(ls)


def run(arm, tr, va, V, a, device, seed=0, Mc=None):
    freq, ts = arm
    Mc = Mc if Mc is not None else a.Mc
    cfg = LayerCfg(a.d, Mc, a.Md, a.G, a.ff, freq="dft", theta_scale=0.0,
                   max_len=a.block)
    m = TinyLM(V, cfg, a.variant, seed=seed).to(device)
    patch_head(m, freq, ts, Mc, a.block, seed=1234 + seed)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(123 + seed)
    hist = {}
    t0 = time.perf_counter()
    for st in range(1, a.steps + 1):
        x, y = get_batch(tr, a.batch, a.block, g, device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        loss.backward(); opt.step()
        if st % a.every == 0 or st == a.steps:
            hist[st] = evaluate(m, va, a.batch, a.block, device)
    del m, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return hist, time.perf_counter() - t0


ALL_ARMS = {
    "dft":       ("dft", 0.0),
    "len":       ("len", 0.0),
    "rope":      ("rope", 0.0),
    "dft_theta": ("dft", 0.02),
    "len_theta": ("len", 0.02),
}
LABEL = {"dft": "original", "len": "omega -> 2pi.k/L", "rope": "rope",
         "dft_theta": "theta=0.02", "len_theta": "len + theta"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--block", type=int, default=128)
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--Md", type=int, default=16); p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--lrs", default="1e-3,3e-4")
    p.add_argument("--variant", default="v1")
    p.add_argument("--arms", default="dft,len,rope,dft_theta,len_theta")
    p.add_argument("--seeds", default="0", help="comma list; >1 reports mean and spread")
    p.add_argument("--mcs", default=None, help="comma list of Mc to sweep")
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [k.strip() for k in a.arms.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    mcs = [int(s) for s in a.mcs.split(",")] if a.mcs else [a.Mc]

    tr, va, V = load_compact()
    print(f"compact vocab {V} (was 100277)  train {len(tr)}  val {len(va)}  device {device}")
    print(f"variant={a.variant} steps={a.steps} B={a.batch} T={a.block} "
          f"seeds={seeds} Mc={mcs}\n")

    for lr in [float(s) for s in a.lrs.split(",")]:
        a.lr = lr
        for Mc in mcs:
            print(f"--- lr={lr:g}  Mc={Mc}  final val loss over {len(seeds)} seed(s) ---")
            hdr = f"{'arm':<18s} " + " ".join(f"{'seed '+str(s):>9s}" for s in seeds)
            print(hdr + f" {'mean':>9s} {'spread':>8s} {'vs first':>9s}")
            base = None
            for name in arms:
                vals = []
                for sd in seeds:
                    hist, _ = run(ALL_ARMS[name], tr, va, V, a, device, seed=sd, Mc=Mc)
                    vals.append(hist[max(hist)])
                mean = sum(vals) / len(vals)
                spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
                if base is None:
                    base, delta = mean, ""
                else:
                    delta = f"{mean - base:+.4f}"
                print(f"{LABEL[name]:<18s} " + " ".join(f"{v:9.4f}" for v in vals)
                      + f" {mean:9.4f} {spread:8.4f} {delta:>9s}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
