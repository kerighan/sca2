"""How far the SCA2 layer sits from the attention block it is competing with.

Same d/ff/B/T, so the ratio is the honest "cost of the architecture" number that
the token-level benchmark's tok/s was trying (and failing) to report.
"""
import argparse, time, sys, torch, torch.nn as nn
from .ref import LayerCfg
from .registry import build


def _time(fn, iters=10, warmup=4, trials=3):
    for _ in range(warmup): fn()
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="*", default=["ref"])
    p.add_argument("-B", type=int, default=8); p.add_argument("-T", type=int, default=128)
    p.add_argument("--d", type=int, default=128); p.add_argument("--ff", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--Mc", type=int, default=64); p.add_argument("--Md", type=int, default=16)
    p.add_argument("--G", type=int, default=8)
    a = p.parse_args(argv)
    dev = "cuda"
    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff)
    x = torch.randn(a.B, a.T, a.d, device=dev)

    enc = nn.TransformerEncoderLayer(a.d, a.heads, a.ff, dropout=0, batch_first=True,
                                     norm_first=True, activation="gelu").to(dev)
    mask = torch.triu(torch.ones(a.T, a.T, device=dev, dtype=torch.bool), 1)
    xg = x.clone().requires_grad_(True)
    def trf():
        enc.zero_grad(set_to_none=True)
        enc(xg, src_mask=mask).square().mean().backward()
    with torch.no_grad():
        t_fwd = _time(lambda: enc(x, src_mask=mask))
    t_tr = _time(trf)
    nptrf = sum(q.numel() for q in enc.parameters())
    print(f"B={a.B} T={a.T} d={a.d} ff={a.ff}")
    print(f"{'layer':<16s} {'params':>8s} {'fwd ms':>8s} {'fwd+bwd ms':>11s} {'vs attn':>9s}")
    print(f"{'attention':<16s} {nptrf:>8d} {t_fwd:>8.2f} {t_tr:>11.2f} {'1.00x':>9s}")
    for n in a.variants:
        m = build(n, cfg, device=dev)
        inner = getattr(m, "layer", m)
        npar = sum(q.numel() for q in inner.parameters())
        xv = x.clone().requires_grad_(True)
        def sca():
            m.zero_grad(set_to_none=True)
            m.prefill(xv)[0].square().mean().backward()
        with torch.no_grad():
            f = _time(lambda: m.prefill(x))
        t = _time(sca)
        print(f"{n:<16s} {npar:>8d} {f:>8.2f} {t:>11.2f} {t/t_tr:>8.2f}x")
        del m; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
