"""Two short measurements: the compiled-vs-eager batch cliff, and the dv sweep."""
import sys, torch
from .ref import LayerCfg
from .registry import build
from .diag_batch_cliff import step_time

D, T = 128, 128


def cliff():
    print("=== compiled vs eager across the batch cliff (sepq, T=128) ===")
    print(f"{'B':>5s} {'eager':>10s} {'compiled':>10s} {'ratio':>8s}   us/token")
    for B in (8, 16, 32, 64):
        r = {}
        for n in ("sepq", "sepq_cc"):
            try:
                m = build(n, LayerCfg(D, 64, 16, 8, 256), device="cuda")
                r[n] = step_time(m, B, T, D)[0]
                del m
            except torch.OutOfMemoryError:
                r[n] = float("nan")
            torch.cuda.empty_cache()
        print(f"{B:>5d} {r['sepq']:>10.2f} {r['sepq_cc']:>10.2f} "
              f"{r['sepq_cc'] / r['sepq']:>7.2f}x")


def dv_sweep():
    print("\n=== dv sweep (sepq, eager, B=16) ===")
    print(f"{'dv':>4s} {'params':>8s} {'us/tok':>9s} {'peak MB':>8s} {'vs dv=64':>9s}")
    base = None
    for dv in (64, 32, 16):
        m = build("sepq", LayerCfg(D, 64, 16, 8, 256, dv=dv), device="cuda")
        n = sum(p.numel() for p in m.parameters())
        us, mem = step_time(m, 16, T, D)
        base = base or us
        print(f"{dv:>4d} {n:>8d} {us:>9.2f} {mem:>8.0f} {us / base:>8.2f}x")
        del m
        torch.cuda.empty_cache()


def polar():
    """sepq cartesian vs log-polar gate: speed (interleaved) then loss."""
    import torch.nn as nn
    from .measure import interleaved
    from .bench_params import SCA2LM, train_one
    from .ab_freq import load_compact
    import argparse

    cfg = LayerCfg(D, 64, 16, 8, 256, freq="rope", theta_scale=0.0, max_len=T)
    dev, B = "cuda", 16
    print("=== speed, interleaved, B=16 ===")
    x = torch.randn(B, T, D, device=dev, requires_grad=True)
    mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
    enc = nn.TransformerEncoderLayer(D, 4, 256, dropout=0, batch_first=True,
                                     norm_first=True, activation="gelu").to(dev)
    keep, arms = [], {}

    def mk(mod, attn=False):
        def f():
            mod.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            y = mod(x, src_mask=mask) if attn else mod.prefill(x)[0]
            y.square().mean().backward()
        return f

    arms["attention"] = mk(enc, attn=True)
    for n in ("sepq_cc", "polar_cc"):
        m = build(n, cfg, device=dev); keep.append(m); arms[n] = mk(m)
    out = interleaved(arms, rounds=5)
    base = out["attention"][0]
    for k, (med, lo, hi) in out.items():
        print(f"  {k:<12s} {med/(B*T)*1e6:8.2f} us/tok  x attn {med/base:6.2f}  "
              f"[spread {100*(hi-lo)/med:.0f}%]")
    keep.clear(); del x
    torch.cuda.empty_cache()

    print("\n=== loss, 1500 steps, rope, compact vocab ===")
    tr, va, V = load_compact()
    a = argparse.Namespace(steps=1500, batch=8, block=T, d=D)
    print(f"  {'arm':<10s} {'params':>8s} {'val':>8s}")
    for n in ("sepq", "polar"):
        best, npar = float("inf"), None
        for lr in (1e-3, 3e-4):
            m = SCA2LM(V, cfg, n)
            npar = m.core_params()
            v, _ = train_one(m, tr, va, a, dev, lr)
            best = min(best, v)
            del m
            torch.cuda.empty_cache()
        print(f"  {n:<10s} {npar:>8d} {best:>8.4f}")


def dv_loss():
    """dv is the value width of both heads' state. Speed said -18% time and
    -18% params at dv=32; this is the quality side of that trade."""
    import argparse
    from .bench_params import SCA2LM, train_one
    from .ab_freq import load_compact
    tr, va, V = load_compact()
    a = argparse.Namespace(steps=1500, batch=8, block=T, d=D)
    print("=== dv loss (polar, rope, 1500 steps, compact vocab) ===")
    print(f"  {'dv':>4s} {'params':>8s} {'val':>8s} {'vs dv=64':>9s}")
    base = None
    for dv in (64, 32, 16):
        cfg = LayerCfg(D, 64, 16, 8, 256, freq="rope", theta_scale=0.0,
                       max_len=T, dv=dv)
        best, npar = float("inf"), None
        for lr in (1e-3, 3e-4):
            m = SCA2LM(V, cfg, "polar")
            npar = m.core_params()
            v, _ = train_one(m, tr, va, a, "cuda", lr)
            best = min(best, v)
            del m
            torch.cuda.empty_cache()
        d = "" if base is None else f"{best - base:+.4f}"
        base = base if base is not None else best
        print(f"  {dv:>4d} {npar:>8d} {best:>8.4f} {d:>9s}")


def triton_bench():
    """Fused D-head kernel vs the PyTorch path, interleaved, B=16."""
    from .measure import interleaved
    cfg = LayerCfg(D, 64, 16, 8, 256, freq="rope", theta_scale=0.0, max_len=T)
    dev, B = "cuda", 16
    x = torch.randn(B, T, D, device=dev)
    xg = x.clone().requires_grad_(True)
    names = ["polar", "tri", "polar_cc", "tri_cc"]
    mods = {n: build(n, cfg, device=dev) for n in names}

    print("=== forward only (no_grad) ===")
    with torch.no_grad():
        out = interleaved({n: (lambda m=m: m.prefill(x)) for n, m in mods.items()},
                          rounds=5)
    base = out["polar_cc"][0]
    for k, (med, lo, hi) in out.items():
        print(f"  {k:<10s} {med/(B*T)*1e6:8.2f} us/tok   vs polar_cc {base/med:5.2f}x"
              f"   [spread {100*(hi-lo)/med:.0f}%]")

    print("\n=== full training step (fwd+bwd) ===")
    def mk(m):
        def f():
            m.zero_grad(set_to_none=True)
            if xg.grad is not None:
                xg.grad = None
            m.prefill(xg)[0].square().mean().backward()
        return f
    out = interleaved({n: mk(m) for n, m in mods.items()}, rounds=5)
    base = out["polar_cc"][0]
    for k, (med, lo, hi) in out.items():
        print(f"  {k:<10s} {med/(B*T)*1e6:8.2f} us/tok   vs polar_cc {base/med:5.2f}x"
              f"   [spread {100*(hi-lo)/med:.0f}%]")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "cliff"):
        cliff()
    if which in ("all", "dv"):
        dv_sweep()
    if which in ("all", "polar"):
        polar()
    if which in ("all", "dvloss"):
        dv_loss()
    if which in ("all", "triton"):
        triton_bench()
