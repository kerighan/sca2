"""Training throughput of each arm at the Zyda-2 vocabulary.

Every throughput number in the campaign was measured at V=16384 on
codeparrot. Zyda-2 uses a 32k BPE, which doubles the embedding AND the output
head -- two d x V matmuls per step that the mixer does not touch. That cost is
the same for every arm in absolute terms, so it COMPRESSES the relative speed
advantage: the faster the mixer, the larger a fraction of the step the vocab
becomes. Sizing a 64 h experiment on the V=16384 ratios would over-estimate how
many tokens the small arms see.

Blocked design: every arm is timed inside every round and only within-round
ratios are kept, because sequential timing on this machine drifts by 15%.

    python bench_vocab.py
    python bench_vocab.py --vocabs 16384,32000 --rounds 7
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from bench_tinypython import SCA2
from sca2.ref import LayerCfg

LAPA = dict(freq="rope", theta_scale=0.02, rope_base=2048, slow_frac=0.25,
            Ls=128, conv=4, gdn_gate=True, lam_free=True, damp_mem=(4.0, 20000.0),
            layer_scale=True, init_v2=True, v_silu=True)

ARMS = {
    "dv128":  dict(Mc=128, dv=128, ff=4096, variant="lapa_cc", **LAPA),
    "dv256":  dict(Mc=256, dv=256, ff=4096, variant="lapa_cc", **LAPA),
    "dv384":  dict(Mc=384, dv=384, ff=4096, variant="lapa_cc", **LAPA),
    "gdn":    dict(Mc=256, dv=256, ff=4096, variant="gdn_cc", freq="rope",
                   theta_scale=0.02, Ls=128, rope_base=2048, slow_frac=0.25,
                   gdn_heads=8, gdn_head_k=128, gdn_expand_v=1.0),
}


def build(name: str, V: int, d: int, layers: int, T: int, tie: bool = True):
    kw = dict(ARMS[name])
    variant = kw.pop("variant")
    cfg = LayerCfg(d=d, Md=4, G=8, max_len=T, **kw)
    torch.manual_seed(0)
    m = SCA2(V, cfg, variant, "cuda", layers, tie_embed=tie).cuda()
    n = sum(p.numel() for p in m.parameters())
    return torch.compile(m, dynamic=False), n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocabs", default="16384,32000")
    p.add_argument("--arms", default="dv128,dv256,dv384,gdn")
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--blocks", default="2048")
    p.add_argument("--untied", action="store_true")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iters", type=int, default=3)
    a = p.parse_args()

    torch._dynamo.config.cache_size_limit = 256
    vocabs = [int(v) for v in a.vocabs.split(",")]
    blocks = [int(b) for b in a.blocks.split(",")]
    names = a.arms.split(",")
    B = a.batch

    for V in vocabs:
      for T in blocks:
          models, params = {}, {}
          for n in names:
              models[n], params[n] = build(n, V, a.d, a.layers, T, not a.untied)
          x = torch.randint(0, V, (B, T), device="cuda")

          def step(m):
              with torch.autocast("cuda", dtype=torch.bfloat16):
                  y = m(x)
              y.float().square().mean().backward()

          for n in names:                                   # warmup + compile
              for _ in range(3):
                  step(models[n])
          torch.cuda.synchronize()

          per = {n: [] for n in names}
          for _ in range(a.rounds):
              for n in names:                               # every arm inside every round
                  torch.cuda.synchronize()
                  t0 = time.perf_counter()
                  for _ in range(a.iters):
                      step(models[n])
                  torch.cuda.synchronize()
                  per[n].append((time.perf_counter() - t0) / a.iters)

          base = names[-1]                                  # gdn is the reference
          print(f"\nV={V}  d={a.d} L={a.layers} B={B} T={T}  "
                f"{'untied' if a.untied else 'TIED'}  (fwd+bwd, compiled)")
          print(f"  {'arm':8s} {'params':>12} {'ms':>8} {'tok/s':>10} {'vs '+base:>9} "
                f"{'tokens in 64 h':>15}")
          for n in names:
              ms = statistics.median(per[n]) * 1e3
              tps = B * T / (statistics.median(per[n]))
              ratio = statistics.median([per[base][r] / per[n][r] for r in range(a.rounds)])
              print(f"  {n:8s} {params[n]:12,} {ms:8.1f} {tps:10,.0f} {ratio:8.3f}x "
                    f"{tps*64*3600/1e9:14.1f}B")
          for n in names:
              del models[n]
          torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
