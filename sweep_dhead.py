"""
Where the training step's time actually goes, and the two knobs that move it.

bench_heads.py at the real config (B=16 T=256 Mc=128 Md=16) puts the C head at
~5 ms fwd+bwd and the D head at ~82: the D head IS the training step. Its cost
is the pair of einsums in arch_sepq.prefill,

    ire = einsum("bmgtr,bgrj->bmgtj", mag * dp.cos(), vc)

i.e. B.M.G.c.c.gs -- only ~67 MMAC at T=256, c=8, so this is occupancy-bound
(M.G = 128 separate 8x8x8 matmuls), not FLOP-bound. Hence two knobs:

  CHUNK  larger c = fewer, bigger matmuls (FLOPs grow as c, launches fall as 1/c)
  Md     `vc` does not depend on m, so the cost is linear in Md while
         bench_params says Md=2 -> 16 buys only 0.043 nats with rope

Run on an IDLE GPU: a concurrent training run halves these numbers.
"""
import torch

from sca2.arch_sepq import DHeadSepQPolar
from sca2.ref import LayerCfg
from bench_tinypython import SCA2
from bench_train_tps import tps

B, T, V = 16, 256, 16384
base = None
print("Md CHUNK  par/layer            tok/s (3 windows)   vs base")
for Md in [16, 8, 4]:
    for ch in [8, 16]:
        DHeadSepQPolar.CHUNK = ch
        cfg = LayerCfg(128, 128, Md, 8, 256, freq="rope", max_len=T)
        torch.manual_seed(0)
        m = SCA2(V, cfg, "v3polar_cc", "cuda", 2).to("cuda")
        if base is None:
            inner = getattr(m.layer, "layer", m.layer)
            print("D head:", type(inner.dh).__name__)
        r = tps(m, B, T, V, "cuda")
        best = max(r)
        base = base or best
        print(f"{Md:2d} {ch:5d} {m.core_params()//2:10d}   "
              + " ".join(f"{v/1e3:7.1f}k" for v in r)
              + f"   x{best/base:.2f}", flush=True)
        del m
        torch.cuda.empty_cache()
