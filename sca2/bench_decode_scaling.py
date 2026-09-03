"""
O(1) decode vs O(T) attention decode -- cost per generated token as the context
grows.

This is the structural claim the SCA2 layer makes: its state is (B, M, dv), a
fixed size, so decoding token N costs the same as decoding token 1. Attention
with a KV cache must re-read N keys and values, so its per-token cost grows
linearly and its cache grows linearly in memory.

Fairness notes:
  * The attention baseline loads its weights from `nn.TransformerEncoderLayer`,
    so the parameter set is provably the same block the benchmark trains, not a
    lookalike.
  * Both are measured EAGER, so the comparison is apples to apples. SCA2's
    CUDA-graph number is reported as an extra column, since its static-shaped
    state can be captured while a growing KV slice cannot.
  * SCA2's decode cost is independent of the state's CONTENTS, so the state is
    fabricated at the target position instead of prefilled -- otherwise the
    quadratic-prefill C head would OOM long before the interesting lengths.

Run:  python -m sca2.bench_decode_scaling -B 8
"""
import argparse, math, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import LayerCfg
from .registry import build


class CachedAttnBlock(nn.Module):
    """nn.TransformerEncoderLayer (norm_first, gelu) with a static KV cache."""

    def __init__(self, d, heads, ff, max_len, device, dtype=torch.float32):
        super().__init__()
        src = nn.TransformerEncoderLayer(d, heads, ff, dropout=0, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.self_attn = src.self_attn
        self.linear1, self.linear2 = src.linear1, src.linear2
        self.norm1, self.norm2 = src.norm1, src.norm2
        self.d, self.h, self.dh = d, heads, d // heads
        self.max_len = max_len
        self.to(device=device, dtype=dtype)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def new_cache(self, B, L, device, dtype):
        return (torch.zeros(B, L, self.d, device=device, dtype=dtype),
                torch.zeros(B, L, self.d, device=device, dtype=dtype))

    def step(self, x_t, cache, pos):
        """x_t: (B,d). Attends over cache[:, :pos+1]. O(pos) per token."""
        B = x_t.size(0)
        ck, cv = cache
        h = self.norm1(x_t)
        qkv = F.linear(h, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.split(self.d, -1)
        ck[:, pos] = k
        cv[:, pos] = v
        K = ck[:, :pos + 1].view(B, pos + 1, self.h, self.dh).transpose(1, 2)
        V = cv[:, :pos + 1].view(B, pos + 1, self.h, self.dh).transpose(1, 2)
        qh = q.view(B, self.h, 1, self.dh)
        att = torch.softmax((qh @ K.transpose(-1, -2)) / math.sqrt(self.dh), -1)
        o = (att @ V).view(B, self.d)
        x = x_t + self.self_attn.out_proj(o)
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


def _time(fn, iters, warmup=5, trials=3):
    for _ in range(warmup): fn()
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e6      # microseconds


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-B", type=int, default=8)
    p.add_argument("--lengths", default="128,512,2048,8192,32768")
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--Md", type=int, default=16); p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=256); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--variant", default="v1")
    a = p.parse_args(argv)
    dev = "cuda"; dt = torch.float32
    Ls = [int(s) for s in a.lengths.split(",")]
    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff)

    sca = build(a.variant, cfg, device=dev, dtype=dt)
    inner = getattr(sca, "layer", sca)
    n_sca = sum(q.numel() for q in inner.parameters())
    attn = CachedAttnBlock(a.d, a.heads, a.ff, max(Ls), dev, dt)
    xt = torch.randn(a.B, a.d, device=dev, dtype=dt)

    # SCA2 state size is constant; attention cache grows with L
    st_bytes = sum(t.numel() * t.element_size()
                   for k in ("c", "d") for t in inner.init_state(a.B, dev, dt)[k].values()
                   if torch.is_tensor(t))

    print(f"B={a.B} d={a.d} eager, fp32, RTX 2070")
    print(f"SCA2 layer {n_sca} params, state {st_bytes/1e6:.2f} MB (constant)")
    print(f"attention  {attn.n_params()} params, KV cache grows 2.B.L.d\n")
    print(f"{'context L':>10s} {'SCA2 us/tok':>12s} {'graph us/tok':>13s} "
          f"{'attn us/tok':>12s} {'KV MB':>8s} {'attn/SCA2':>10s}")

    from .decode import GraphDecoder, _check_capturable
    for L in Ls:
        st = inner.init_state(a.B, dev, dt)
        if torch.is_tensor(st["c"]["pos"]):
            st["c"]["pos"] = st["c"]["pos"] + float(L)      # cost is content-independent
        with torch.no_grad():
            t_sca = _time(lambda st=st: sca.step(xt, st), 20)
            t_g = float("nan")
            try:
                _check_capturable(st)
                g = GraphDecoder(sca, inner.init_state(a.B, dev, dt))
                t_g = _time(lambda: g.step(xt), 50)
                del g
            except TypeError:
                pass
            ck, cv = attn.new_cache(a.B, L, dev, dt)
            kv_mb = (ck.numel() + cv.numel()) * ck.element_size() / 1e6
            t_at = _time(lambda: attn.step(xt, (ck, cv), L - 1), 20)
            del ck, cv
        torch.cuda.empty_cache()
        print(f"{L:>10d} {t_sca:>12.1f} {t_g:>13.1f} {t_at:>12.1f} {kv_mb:>8.1f} "
              f"{t_at/t_sca:>9.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
