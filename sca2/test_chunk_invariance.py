"""
Chunk-size invariance.

`iso.py` sweeps SHAPES (B, T, dtype) but never CHUNK SIZES, and several heads
carry state across chunk boundaries. A carry bug there is invisible whenever
T <= chunk size, because the first chunk's carry is empty -- which is exactly how
an off-by-one in `DHeadCumsumRel` (the carry advances one step from a-1 to a)
survived a clean iso run: its chunk size is derived from the learned slopes, so
the default was 150 and every tested T was smaller.

Every chunked head computes a chunk-free quantity. Vary the chunking, the answer
must not move.

    python -m sca2.test_chunk_invariance
"""
import sys
import torch
import torch.nn.functional as F

from .ref import LayerCfg
from .registry import build

TOL = 1e-10          # float64


def _decode_ref(head, z):
    """Token-by-token, which has no chunking by construction."""
    B, T, _ = z.shape
    st = head.init_state(B, z.device, z.dtype)
    out = []
    for t in range(T):
        y, st = head.step(z[:, t], z[:, t - 1] if t else torch.zeros_like(z[:, 0]), st)
        out.append(y)
    return torch.stack(out, 1)


def check(name, attr, values, T=96, d=64, M=8, G=4):
    torch.manual_seed(0)
    layer = build(name, LayerCfg(d, 16, M, G, 128, freq="rope", max_len=T),
                  dtype=torch.float64)
    head = layer.dh
    z = torch.randn(2, T, d, dtype=torch.float64)
    h = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], 1)
    ref = _decode_ref(head, z)
    worst = 0.0
    for v in values:
        setattr(type(head), attr, v)
        got = head.prefill(z, h)[0]
        e = (got - ref).abs().max().item()
        worst = max(worst, e)
        print(f"    {attr}={v:<6} n_chunks~{max(1, T // max(1, head._chunk(T) if hasattr(head,'_chunk') else v)):<3d} "
              f"max|prefill - decode| = {e:.3e}")
    return worst


def main():
    fails = 0
    print("=== chunk-size invariance ===")
    print("  v1 D head (fixed CHUNK)")
    w = check("v1", "CHUNK", [96, 32, 16, 8, 4])
    fails += w > TOL
    print("  length-free cumsum D head (chunk derived from the slopes)")
    torch.manual_seed(0)
    lay = build("csr", LayerCfg(64, 16, 8, 4, 128, freq="rope", max_len=96),
                dtype=torch.float64)
    smax = float(F.softplus(lay.dh.slopes.detach()).max())
    w2 = check("csr", "LOG_HEADROOM", [smax * c for c in (96, 32, 16, 8, 4)])
    fails += w2 > TOL
    print(f"\n{'CHUNK INVARIANCE OK' if not fails else 'FAILED'}  "
          f"(worst {max(w, w2):.2e}, tol {TOL:.0e})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
