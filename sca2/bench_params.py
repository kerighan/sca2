"""
SCA2 vs Transformer at MATCHED non-embedding parameter budgets.

The original benchmark compared a 419k-parameter SCA2 layer against a 133k
attention block and hid both behind 25.7M of embedding, so it could not answer
"is this architecture better". This does.

Method:
  * One knob per architecture, so nothing is confounded: SCA2 moves `Md` (its
    q projections are 262k of the 419k), the Transformer moves `ff`. Four
    matched budgets, agreeing to within 0.2%.
  * Compact vocabulary (2318 of cl100k's 100277 ids actually occur), so the
    loss reflects the layer rather than the softmax size.
  * lr swept per architecture and the BEST reported -- a single shared lr biases
    the comparison toward whichever architecture happens to like it.
  * Identical data order and seeds across every arm.

Run:  python -m sca2.bench_params --steps 2000
"""
import argparse, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import LayerCfg
from .registry import build
from .ab_freq import load_compact, get_batch, evaluate

# (label, sca2 Md, transformer ff) -- budgets verified in the table this prints
PAIRS = [(2, 367), (4, 512), (8, 798), (16, 1372)]


class SCA2LM(nn.Module):
    def __init__(self, V, cfg, variant, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.e = nn.Embedding(V, cfg.d); self.on = nn.LayerNorm(cfg.d)
        self.o = nn.Linear(cfg.d, V)
        self.layer = build(variant, cfg, seed=seed)

    def core_params(self):
        return sum(p.numel() for p in getattr(self.layer, "layer", self.layer).parameters())

    def forward(self, t):
        return self.o(self.on(self.layer.prefill(self.e(t))[0]))


class TrfLM(nn.Module):
    def __init__(self, V, d, heads, ff, maxlen, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.e = nn.Embedding(V, d); self.p = nn.Embedding(maxlen, d)
        self.b = nn.TransformerEncoderLayer(d, heads, ff, dropout=0, batch_first=True,
                                            norm_first=True, activation="gelu")
        self.n = nn.LayerNorm(d); self.o = nn.Linear(d, V)

    def core_params(self):
        return sum(p.numel() for p in self.b.parameters())

    def forward(self, t):
        T = t.size(1)
        x = self.e(t) + self.p(torch.arange(T, device=t.device))[None]
        mask = torch.triu(torch.ones(T, T, device=t.device, dtype=torch.bool), 1)
        return self.o(self.n(self.b(x, src_mask=mask)))


def train_one(m, tr, va, a, device, lr):
    m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(123)
    t0 = time.perf_counter()
    for st in range(1, a.steps + 1):
        x, y = get_batch(tr, a.batch, a.block, g, device)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward()
        opt.step()
    return evaluate(m, va, a.batch, a.block, device), time.perf_counter() - t0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--block", type=int, default=128)
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--G", type=int, default=8); p.add_argument("--ff", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lrs", default="1e-3,3e-4")
    p.add_argument("--variant", default="v1")
    p.add_argument("--freq", default="dft"); p.add_argument("--theta-scale", type=float, default=0.0)
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lrs = [float(s) for s in a.lrs.split(",")]

    tr, va, V = load_compact()
    print(f"compact vocab {V}  train {len(tr)}  device {device}  steps {a.steps}  lrs {lrs}")
    print(f"sca2 variant={a.variant} freq={a.freq} theta_scale={a.theta_scale}\n")
    print(f"{'budget':>8s}  {'SCA2 cfg':>10s} {'params':>8s} {'val':>8s}   "
          f"{'Trf cfg':>10s} {'params':>8s} {'val':>8s}   {'winner':>10s}")

    for Md, ff in PAIRS:
        cfg = LayerCfg(a.d, a.Mc, Md, a.G, a.ff, freq=a.freq,
                       theta_scale=a.theta_scale, max_len=a.block)
        s_best, s_par = float("inf"), None
        for lr in lrs:
            m = SCA2LM(V, cfg, a.variant)
            s_par = m.core_params()
            v, _ = train_one(m, tr, va, a, device, lr)
            s_best = min(s_best, v)
            del m; torch.cuda.empty_cache() if device == "cuda" else None
        t_best, t_par = float("inf"), None
        for lr in lrs:
            m = TrfLM(V, a.d, a.heads, ff, a.block)
            t_par = m.core_params()
            v, _ = train_one(m, tr, va, a, device, lr)
            t_best = min(t_best, v)
            del m; torch.cuda.empty_cache() if device == "cuda" else None
        gap = t_best - s_best
        win = f"SCA2 {gap:+.3f}" if gap > 0 else f"Trf {gap:+.3f}"
        print(f"{max(s_par,t_par):>8d}  {'Md='+str(Md):>10s} {s_par:>8d} {s_best:>8.4f}   "
              f"{'ff='+str(ff):>10s} {t_par:>8d} {t_best:>8.4f}   {win:>10s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
