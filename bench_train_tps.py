"""
Steady-state training throughput, whole model, pretrain.py's exact step.

Why not just read pretrain.py's tok/s: that number is a CUMULATIVE average from
step 1, so a short run is dominated by the clock ramp out of idle (measured:
675 MHz idle -> 1815 MHz under load on this 2070). Here warmup is long enough to
reach steady clocks, and each reported figure is a fresh timed window, repeated,
so the spread is visible instead of averaged away.
"""
import argparse, time
import torch
import torch.nn.functional as F

from sca2.ref import LayerCfg
from bench_tinypython import SCA2, Transformer


def tps(m, B, T, V, device, warm=60, iters=60, reps=3):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, V, (B, T), device=device)
    y = torch.randint(0, V, (B, T), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward()
        opt.step()

    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out.append(B * T * iters / dt)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--Mc", type=int, default=128)
    p.add_argument("--Md", type=int, default=16)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--trf-ff", type=int, default=464, dest="trf_ff")
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--freq", default="rope")
    p.add_argument("--arms", default="v3polar_cc,gdn_cc,transformer")
    a = p.parse_args()
    device = "cuda"
    V = a.vocab
    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff, freq=a.freq, max_len=a.block)

    print(f"B={a.batch} T={a.block} d={a.d} layers={a.layers} Mc={a.Mc} Md={a.Md}")
    print(f"{'arm':16s} {'params/layer':>12s} {'tok/s (3 windows)':>34s} {'ms/step':>9s}")
    for arm in a.arms.split(","):
        torch.manual_seed(0)
        if arm == "transformer":
            m = Transformer(V, a.d, 4, a.trf_ff, a.block, a.layers).to(device)
        else:
            m = SCA2(V, cfg, arm, device, a.layers).to(device)
        r = tps(m, a.batch, a.block, V, device)
        core = m.core_params() // a.layers
        best = max(r)
        print(f"{arm:16s} {core:12d}   " + " ".join(f"{v/1e3:8.1f}k" for v in r)
              + f" {a.batch*a.block/best*1e3:8.2f}")
        del m, r
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
