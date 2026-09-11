"""LapA vs Gated DeltaNet at scale-model width: speed, parameters and decode state.

SPARK.md, question (c): there is no shape that matches LapA and GDN on all three of
total parameters, mixer parameters and decode state -- they are different resources
for these two architectures -- so this prints all three next to the timing rather
than picking the axis that flatters either side.

SPARK.md, question (b): every speed number in this repo before the Spark compared our
inductor path against fla's NAIVE PyTorch reference, because their Triton kernels do
not build on sm_75. They do here, so GDN is timed with its real kernels by default;
`--gdn-kernel naive` reproduces the old, unfair comparison for reference.

Both arms are compiled and run under the same bf16 autocast.

    python -m lapa.benchmarks.vs_gdn
    python -m lapa.benchmarks.vs_gdn --d 2048 --M 512 --dv 512 --ff 8192 --batch 4
"""
from __future__ import annotations

import argparse

import torch

from ..layer import LaplaceAttention, LaplaceConfig
from .baselines import GDNLayer
from .speed import blocked


def counts(layer, mixer_attr):
    total = sum(p.numel() for p in layer.parameters())
    mixer = sum(p.numel() for p in getattr(layer, mixer_attr).parameters()) \
        if isinstance(mixer_attr, str) else \
        sum(p.numel() for a in mixer_attr for p in getattr(layer, a).parameters())
    return total, mixer, layer.state_floats()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--dv", type=int, default=256)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--ff", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--gdn-heads", type=int, default=8)
    p.add_argument("--gdn-head-k", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--rope-base", type=float, default=1000.0)
    p.add_argument("--slow-frac", type=float, default=0.25)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--gdn-kernel", default="auto", choices=("auto", "triton", "naive"))
    a = p.parse_args()

    torch._dynamo.config.cache_size_limit = 256
    x = torch.randn(a.batch, a.T, a.d, device="cuda", requires_grad=True)

    torch.manual_seed(0)
    lapa = LaplaceAttention(LaplaceConfig(
        d=a.d, M=a.M, dv=a.dv, L=a.L, ff=a.ff, chunk=a.chunk,
        rope_base=a.rope_base, slow_frac=a.slow_frac, max_len=a.T)).cuda()
    torch.manual_seed(0)
    gdn = GDNLayer(a.d, a.gdn_heads, a.gdn_head_k, ff=a.ff, kernel=a.gdn_kernel).cuda()
    torch.manual_seed(0)
    gdn_naive = GDNLayer(a.d, a.gdn_heads, a.gdn_head_k, ff=a.ff, kernel="naive").cuda()

    def run(m):
        f = torch.compile(m)

        def one():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = f(x)
            y.float().square().mean().backward()
        return one

    kern = "triton" if gdn.mix._tri is not None else "naive"
    print(f"{torch.cuda.get_device_name(0)}  B={a.batch} T={a.T} d={a.d}  fwd+bwd, "
          f"both compiled, bf16 autocast\n"
          f"LapA M={a.M} dv={a.dv} L={a.L} ff={a.ff} | "
          f"GDN {a.gdn_heads}x{a.gdn_head_k} ff={a.ff} ({kern} kernel)")

    res = blocked({f"GDN ({kern})": run(gdn),
                   "LapA": run(lapa),
                   "GDN (naive ref)": run(gdn_naive)}, a.rounds, a.iters)

    print(f"\n  {'arm':<17}   ms    vs GDN     tok/s     params   mixer    state")
    info = {f"GDN ({kern})": counts(gdn, "mix"),
            "LapA": counts(lapa, ("long", "short", "mix")),
            "GDN (naive ref)": counts(gdn_naive, "mix")}
    for n, (ms, ratio) in res.items():
        tot, mix, st = info[n]
        print(f"  {n:<17} {ms:6.2f}  {ratio:5.3f}x {a.batch*a.T/(ms/1e3):>9,.0f}"
              f"  {tot/1e6:7.2f}M {mix/1e6:6.2f}M {st/1e3:7.0f}k")


if __name__ == "__main__":
    main()
