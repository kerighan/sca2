"""Blocked-design speed benchmark for the layer.

Sequential timing is not a measurement on this hardware: re-running one config three
times can land 15% apart, non-monotone. Every arm is therefore timed inside every
round and only WITHIN-round ratios are kept; the reported figure is the median of
those ratios, and the median wall time is printed alongside for scale.

    python -m lapa.benchmarks.speed                      # d=1024 reference shape
    python -m lapa.benchmarks.speed --d 2048 --M 512 --dv 512 --ff 8192 --batch 4
    python -m lapa.benchmarks.speed --sections           # long / short / FFN split
    python -m lapa.benchmarks.speed --chunks 64,128,256,512 --paths batched,chunk

The default arm list answers "what does compiling and bf16 buy on this machine"; the
sweeps answer "which chunk and which long_path for this GPU" (the equivalent of
`python -m sca2.autotune`, at scale-model shapes).
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from ..layer import LaplaceAttention, LaplaceConfig


def blocked(arms, rounds=5, iters=5, warmup=3):
    """arms: {name: callable()}. Returns {name: (median ms, median within-round ratio)}."""
    names = list(arms)
    for n in names:
        for _ in range(warmup):
            arms[n]()
    torch.cuda.synchronize()
    per = {n: [] for n in names}
    for _ in range(rounds):
        for n in names:                       # every arm inside every round
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                arms[n]()
            torch.cuda.synchronize()
            per[n].append((time.perf_counter() - t) / iters * 1e3)
    base = names[0]
    return {n: (statistics.median(per[n]),
                statistics.median([per[n][r] / per[base][r] for r in range(rounds)]))
            for n in names}


def report(res, title=""):
    base = list(res)[0]
    w = max(len(n) for n in res)
    if title:
        print(f"\n{title}")
    print(f"  {'arm':<{w}}   ms      vs {base}   tok/s")
    for n, (ms, ratio) in res.items():
        print(f"  {n:<{w}}  {ms:6.2f}   {ratio:5.3f}x", end="")
        print(f"   {report.tokens / (ms / 1e3):>9,.0f}" if report.tokens else "")


report.tokens = 0


def build(a, **over):
    cfg = dict(d=a.d, M=a.M, dv=a.dv, L=a.L, ff=a.ff, chunk=a.chunk,
               rope_base=a.rope_base, slow_frac=a.slow_frac, max_len=a.T)
    cfg.update(over)
    torch.manual_seed(0)
    return LaplaceAttention(LaplaceConfig(**cfg)).cuda()


def runner(m, x, compile_=True, amp=True):
    f = torch.compile(m) if compile_ else m

    def one():
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = f(x)
        else:
            y = f(x)
        y.float().square().mean().backward()
    return one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--dv", type=int, default=256)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--ff", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--rope-base", type=float, default=1000.0)
    p.add_argument("--slow-frac", type=float, default=0.25)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--chunks", default="", help="comma list: sweep cfg.chunk")
    p.add_argument("--paths", default="", help="comma list: sweep cfg.long_path")
    p.add_argument("--sections", action="store_true", help="long / short / FFN split")
    a = p.parse_args()

    torch._dynamo.config.cache_size_limit = 256   # see SPARK.md: 8 silently falls back to eager
    x = torch.randn(a.batch, a.T, a.d, device="cuda", requires_grad=True)
    report.tokens = a.batch * a.T
    print(f"{torch.cuda.get_device_name(0)}  B={a.batch} T={a.T} d={a.d} "
          f"M={a.M} dv={a.dv} L={a.L} ff={a.ff} chunk={a.chunk}  (fwd+bwd)")

    if a.sections:
        m = build(a)
        zp = torch.zeros(a.batch, a.d, device="cuda")
        z = m.n(x).detach().requires_grad_()
        cm, cl, cs, cf = (torch.compile(g) for g in
                          (m, m.long.prefill, m.short.prefill, lambda t: m.ff(m.fn(t))))

        def wrap(f):
            def one():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = f()
                (y[0] if isinstance(y, tuple) else y).float().square().mean().backward()
            return one
        report(blocked({"full": wrap(lambda: cm(x)),
                        "long head": wrap(lambda: cl(z, zp, None)[0]),
                        "short head": wrap(lambda: cs(z, zp, None)[0]),
                        "FFN": wrap(lambda: cf(x))},
                       a.rounds, a.iters), "sections")
        return

    if a.chunks or a.paths:
        chunks = [int(c) for c in (a.chunks or str(a.chunk)).split(",")]
        paths = (a.paths or "batched").split(",")
        arms = {f"chunk {c:<4} {pa}": runner(build(a, chunk=c, long_path=pa), x)
                for pa in paths for c in chunks}
        report(blocked(arms, a.rounds, a.iters), "chunk x path")
        return

    m = build(a)
    report(blocked({"compiled bf16": runner(m, x),
                    "compiled fp32": runner(m, x, amp=False),
                    "eager bf16": runner(m, x, compile_=False),
                    "eager fp32": runner(m, x, compile_=False, amp=False)},
                   a.rounds, a.iters), "paths")


if __name__ == "__main__":
    main()
