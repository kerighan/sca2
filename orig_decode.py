"""Decode latency: the ORIGINAL SeqCond block vs attention with a KV cache.

Training throughput and decode latency are different regimes. The original is
built for the second -- fixed-size state, and a Triton kernel that exists only
for `step`. This measures that regime, which the training benchmark says nothing
about.
"""
import torch
from ref_nautile.modeling_seqcond import SeqCondBlock
from sca2.bench_decode_scaling import CachedAttnBlock, _time

B, d = 8, 128
blk = SeqCondBlock(d_model=d, num_heads=8, num_query_heads=8, num_anchor_heads=2,
                   num_thetas=1, maxlen=131072, out_expand_factor=3).cuda().eval()
attn = CachedAttnBlock(d, 4, 367, 32768, "cuda")
xt = torch.randn(B, d, device="cuda")

print("B=%d d=%d  us per generated token" % (B, d))
print("%10s %16s %10s %17s %11s %9s" % ("context L", "orig eager", "orig+triton", "attention", "attn/orig", "KV MB"))
with torch.no_grad():
    warm = blk(torch.randn(B, 8, d, device="cuda"), return_state=True)[1]
    for L in (128, 512, 2048, 8192, 32768):
        st = list(warm)
        st[3] = torch.full((B,), float(L), device="cuda")   # position counter
        st = tuple(st)
        t_o = _time(lambda: blk.step(xt, st), 20)
        try:
            t_k = _time(lambda: blk.step(xt, st, use_triton=True), 20)
        except Exception as e:
            t_k = float("nan")
        ck, cv = attn.new_cache(B, L, "cuda", torch.float32)
        kv = (ck.numel() + cv.numel()) * 4 / 1e6
        t_a = _time(lambda: attn.step(xt, (ck, cv), L - 1), 20)
        del ck, cv
        torch.cuda.empty_cache()
        print("%10d %16.1f %10.1f %17.1f %10.2fx %9.1f" % (L, t_o, t_k, t_a, t_a / t_o, kv))
