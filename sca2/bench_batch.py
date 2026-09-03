"""
Cost per TOKEN vs batch size -- occupancy, not architecture.

Everything else in this repo was measured at B=8 on a 36-SM RTX 2070, where the
layer's tensors (Md=16, dv=64) are far too small to fill the GPU: the profile
shows gemms at 12-16 us, nowhere near peak. If SCA2's per-token cost falls
faster with B than attention's does, the 3.8x per-step gap shrinks for free at
any realistic training batch size -- no code change, just a config.

Reports us per token so the arms are directly comparable across B.
"""
import argparse, sys, time
import torch
import torch.nn as nn

from .ref import LayerCfg
from .registry import build
from .measure import interleaved


def _time(fn, iters=5, warmup=3, trials=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3


def sweep_interleaved(names, Bs, cfg, T, rounds):
    """One model per (arm, B), all timed round-robin so drift is shared."""
    dev = "cuda"
    enc = nn.TransformerEncoderLayer(cfg.d, 4, cfg.ff, dropout=0, batch_first=True,
                                     norm_first=True, activation="gelu").to(dev)
    print(f"T={T} interleaved, median over {rounds} rounds, us per token")
    for B in Bs:
        x = torch.randn(B, T, cfg.d, device=dev, requires_grad=True)
        mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
        arms, keep = {}, []
        def mk(mod, attn=False):
            def f():
                mod.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad = None
                y = mod(x, src_mask=mask) if attn else mod.prefill(x)[0]
                y.square().mean().backward()
            return f
        arms["attention"] = mk(enc, attn=True)
        for n in names:
            try:
                m = build(n, cfg, device=dev); keep.append(m)
                arms[n] = mk(m)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
        try:
            out = interleaved(arms, rounds=rounds)
        except torch.OutOfMemoryError:
            print(f"B={B}: OOM"); keep.clear(); torch.cuda.empty_cache(); continue
        base = out["attention"][0]
        print(f"B={B}")
        for k, (med, lo, hi) in out.items():
            per = med / (B * T) * 1e6
            print(f"  {k:<12s} {per:8.2f} us/tok  x attn {med/base:6.2f}  "
                  f"[spread {100*(hi-lo)/med:.0f}%]")
        keep.clear(); del x
        torch.cuda.empty_cache()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="v1_cc,sepq_cc")
    p.add_argument("--batches", default="4,8,16,32,64,128")
    p.add_argument("-T", type=int, default=128)
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--Md", type=int, default=16); p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=256); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--interleaved", action="store_true")
    p.add_argument("--rounds", type=int, default=5)
    a = p.parse_args(argv)
    dev = "cuda"
    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff)
    names = a.variants.split(",")
    Bs = [int(s) for s in a.batches.split(",")]
    if a.interleaved:
        sweep_interleaved(names, Bs, cfg, a.T, a.rounds)
        return 0

    enc = nn.TransformerEncoderLayer(a.d, a.heads, a.ff, dropout=0, batch_first=True,
                                     norm_first=True, activation="gelu").to(dev)
    mask = torch.triu(torch.ones(a.T, a.T, device=dev, dtype=torch.bool), 1)

    print(f"T={a.T} d={a.d} fwd+bwd, us per token")
    hdr = f"{'B':>5s} {'attention':>10s} " + " ".join(f"{n:>12s}" for n in names) \
        + "   " + " ".join(f"{'x attn':>8s}" for _ in names)
    print(hdr)
    for B in Bs:
        row, ratios = [], []
        x = torch.randn(B, a.T, a.d, device=dev, requires_grad=True)
        try:
            def trf():
                enc.zero_grad(set_to_none=True)
                if x.grad is not None: x.grad = None
                enc(x, src_mask=mask).square().mean().backward()
            t_at = _time(trf) / (B * a.T) * 1e3
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"{B:>5d} {'OOM':>10s}"); continue
        for n in names:
            try:
                m = build(n, cfg, device=dev)
                def sca():
                    m.zero_grad(set_to_none=True)
                    if x.grad is not None: x.grad = None
                    m.prefill(x)[0].square().mean().backward()
                t = _time(sca) / (B * a.T) * 1e3
                row.append(f"{t:12.3f}"); ratios.append(f"{t/t_at:7.2f}x")
                del m
            except torch.OutOfMemoryError:
                row.append(f"{'OOM':>12s}"); ratios.append(f"{'-':>8s}")
            torch.cuda.empty_cache()
        print(f"{B:>5d} {t_at:10.3f} " + " ".join(row) + "   " + " ".join(ratios))
        del x; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
