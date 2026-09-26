"""Marginal wall-clock cost of read_mix R, everything else held at the campaign's flags.

Blocked design, not sequential. Timing the arms one after the other on this
machine gave R=4 as 14% FASTER than R=1 once, purely because the first arm paid
the clock ramp out of idle (675 MHz -> 1815 MHz); alternating the arms inside
one loop puts that drift in both columns instead of only the first.

    python bench_readmix.py --R 4,8 --batch 6 --block 4096
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from sca2.ref import LayerCfg
from bench_tinypython import SCA2

# Exactly the flags every mix* arm of the Zyda campaign runs with, minus read_mix.
CAMPAIGN = dict(
    freq="rope", theta_scale=0.02, dv=256, conv=4, Ls=128, rope_base=2048,
    slow_frac=0.25, layer_scale=True, lam_free=True, damp_mem=(4.0, 20000.0),
    gdn_gate=True, v_silu=True, init_v2=True, post_norm=True,
)


def build(R: int, V: int, d: int, ff: int, layers: int, block: int, device: str):
    cfg = LayerCfg(d, 256, 4, 8, ff, max_len=block, read_mix=R, **CAMPAIGN)
    torch.manual_seed(0)
    # `device` places the mixer layers only; the embedding and head are built on
    # CPU and the first forward dies on a device mismatch without this .to().
    return SCA2(V, cfg, "lapa_cc", device, layers, tie_embed=True).to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--R", default="4,8")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--ff", type=int, default=4096)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--warm", type=int, default=12)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--blocks", type=int, default=5)
    a = p.parse_args()

    device = "cuda"
    Rs = [int(v) for v in a.R.split(",")]
    B, T = a.batch, a.block

    models, opts = {}, {}
    for R in Rs:
        m = build(R, a.vocab, a.d, a.ff, a.layers, a.block, device)
        models[R] = m
        opts[R] = torch.optim.AdamW(m.parameters(), lr=1e-4)
        n = sum(q.numel() for q in m.parameters())
        print(f"R={R}: {n:,} params")
    base = sum(q.numel() for q in models[Rs[0]].parameters())
    for R in Rs[1:]:
        n = sum(q.numel() for q in models[R].parameters())
        print(f"R={Rs[0]}->{R}: {n - base:+,} params ({(n - base) / base * 100:+.3f}%)")

    x = torch.randint(0, a.vocab, (B, T), device=device)
    y = torch.randint(0, a.vocab, (B, T), device=device)

    def step(R):
        opts[R].zero_grad(set_to_none=True)
        F.cross_entropy(models[R](x).flatten(0, 1), y.flatten()).backward()
        opts[R].step()

    for R in Rs:                      # warm every arm before any arm is timed
        for _ in range(a.warm):
            step(R)
    torch.cuda.synchronize()

    tps = {R: [] for R in Rs}
    for b in range(a.blocks):
        order = Rs if b % 2 == 0 else Rs[::-1]   # ABBA: drift lands on both
        for R in order:
            t0 = time.perf_counter()
            for _ in range(a.iters):
                step(R)
            torch.cuda.synchronize()
            tps[R].append(B * T * a.iters / (time.perf_counter() - t0))
        print(f"  block {b + 1}/{a.blocks}  " +
              "  ".join(f"R={R}:{tps[R][-1]:8,.0f}" for R in Rs))

    print()
    med = {R: statistics.median(v) for R, v in tps.items()}
    for R in Rs:
        s = statistics.pstdev(tps[R]) / med[R] * 100
        print(f"R={R}: {med[R]:9,.0f} tok/s  (spread {s:.2f}%)")
    for R in Rs[1:]:
        print(f"R={Rs[0]}->{R}: {(med[R] / med[Rs[0]] - 1) * 100:+.2f}% throughput")
    print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB (both models resident)")


if __name__ == "__main__":
    main()
