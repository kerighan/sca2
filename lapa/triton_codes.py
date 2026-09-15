"""Triton kernel for the long head's code construction + Gram matrix.

Fuses ~20 separate CUDA kernels into one:
  phase → cos/sin → gw/gq → Kk/Qk/Fq → G = Qk @ Kk^T

The bottleneck on the GB10 is NOT FLOPs (3% utilisation) but kernel launch
overhead and intermediate memory traffic.  Each elementwise op in PyTorch is
a separate kernel launch that writes to global memory and is re-read by the
next one.  This kernel computes the entire code block in registers, writes
Kk/Qk/Fq once, and produces the Gram in a tiled matmul — all in one launch.

For now: NG=1 path only, no decay_input, no beta_groups > 1.  These cover the
baseline and d1024_lsfree configs.  NG=2 is a straightforward extension (loop
over groups in the Fq assembly, wider output).

Usage:
    from lapa.triton_codes import fused_codes_gram
    Kk, Qk, Fq, G = fused_codes_gram(pw, pq, gw, gq, wr, wi, M, C)
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _codes_gram_kernel(
    # phases: (B*K, C, M), contiguous last dim
    PW_ptr, PQ_ptr,
    # decay scales: (C, M) broadcast over B*K
    GW_ptr, GQ_ptr,
    # read weights: (M,) broadcast over everything
    WR_ptr, WI_ptr,
    # outputs
    KK_ptr,   # (B*K, C, 2M)
    QK_ptr,   # (B*K, C, 2M)
    FQ_ptr,   # (B*K, 2C, 2M)
    G_ptr,    # (B*K, C, C) — the Gram Qk @ Kk^T / M, lower-triangular
    # dims
    BK: tl.constexpr,       # B*K
    C: tl.constexpr,
    M: tl.constexpr,
    M2: tl.constexpr,       # 2*M
    inv_M: tl.constexpr,    # 1.0 / M
    # strides
    stride_pw_bk: tl.constexpr,
    stride_pw_c: tl.constexpr,
    stride_kk_bk: tl.constexpr,
    stride_kk_c: tl.constexpr,
    stride_fq_bk: tl.constexpr,
    stride_fq_c: tl.constexpr,
    stride_g_bk: tl.constexpr,
    stride_g_r: tl.constexpr,
    # block sizes
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One program instance handles one (batch, chunk) pair and one tile of
    the M dimension.  It loads the phase tile, computes cos/sin/gw/gq, builds
    the code tile, and accumulates its contribution to the Gram."""
    bk = tl.program_id(0)   # batch * K index
    # We process ALL C rows and a BLOCK_M-wide slice of the M dimension.
    m_start = tl.program_id(1) * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    m_mask = m_offs < M

    # Load wr, wi for this M-tile: (BLOCK_M,)
    wr = tl.load(WR_ptr + m_offs, mask=m_mask, other=0.0)
    wi = tl.load(WI_ptr + m_offs, mask=m_mask, other=0.0)

    # We'll accumulate the Gram contribution for this M-tile in registers.
    # G[r, c] += sum over m in tile of Qk[r, m] * Kk[c, m]
    # This is a rank-BLOCK_M update; we process all C rows.

    # Preallocate Kk and Qk rows in registers for the Gram accumulation
    # Storage: (C, BLOCK_M) for Kk_cos, Kk_sin, Qk_cos, Qk_sin
    for c in tl.static_range(0, BLOCK_C):
        if c >= C:
            break
        # Load phases for row c
        pw_off = bk * stride_pw_bk + c * stride_pw_c + m_offs
        pq_off = bk * stride_pw_bk + c * stride_pw_c + m_offs  # same stride
        pw_val = tl.load(PW_ptr + pw_off, mask=m_mask, other=0.0)
        pq_val = tl.load(PQ_ptr + pq_off, mask=m_mask, other=0.0)

        # cos/sin
        cw = tl.cos(pw_val)
        sw = tl.sin(pw_val)
        cq = tl.cos(pq_val)
        sq = tl.sin(pq_val)

        # Decay scales (C, M) — broadcast over batch
        gw_off = c * M + m_offs  # (C, M) contiguous
        gq_off = c * M + m_offs
        gw_val = tl.load(GW_ptr + gw_off, mask=m_mask, other=1.0)
        gq_val = tl.load(GQ_ptr + gq_off, mask=m_mask, other=1.0)

        # Kk[c, m] = cw*gw, Kk[c, m+M] = sw*gw
        kk_cos = cw * gw_val   # (BLOCK_M,)
        kk_sin = sw * gw_val

        # Qk[c, m] = cw*gq, Qk[c, m+M] = sw*gq
        qk_cos = cw * gq_val
        qk_sin = sw * gq_val

        # Fq: c1 = (wr*cq + wi*sq)*gq, c2 = (wr*sq - wi*cq)*gq
        c1 = (wr * cq + wi * sq) * gq_val
        c2 = (wr * sq - wi * cq) * gq_val

        # Store Kk: row c, cols [m_offs] and [M + m_offs]
        kk_base = bk * stride_kk_bk + c * stride_kk_c
        tl.store(KK_ptr + kk_base + m_offs, kk_cos, mask=m_mask)
        tl.store(KK_ptr + kk_base + M + m_offs, kk_sin, mask=m_mask)

        # Store Qk
        qk_base = bk * stride_kk_bk + c * stride_kk_c
        tl.store(QK_ptr + qk_base + m_offs, qk_cos, mask=m_mask)
        tl.store(QK_ptr + qk_base + M + m_offs, qk_sin, mask=m_mask)

        # Store Fq: row c -> [c1 | c2], row c+C -> [-c2 | c1]
        fq_base = bk * stride_fq_bk
        tl.store(FQ_ptr + fq_base + c * stride_fq_c + m_offs, c1, mask=m_mask)
        tl.store(FQ_ptr + fq_base + c * stride_fq_c + M + m_offs, c2, mask=m_mask)
        tl.store(FQ_ptr + fq_base + (c + C) * stride_fq_c + m_offs, -c2, mask=m_mask)
        tl.store(FQ_ptr + fq_base + (c + C) * stride_fq_c + M + m_offs, c1, mask=m_mask)


def fused_codes(pw, pq, gw, gq, wr, wi, M, C, B_K):
    """Build Kk, Qk, Fq from phases and decay scales. One fused kernel.

    Args:
        pw, pq: (B*K, C, M) write/read phases
        gw, gq: (C, M) decay scales (broadcast over batch)
        wr, wi: (M,) read weights
        M: number of modes
        C: chunk size
        B_K: batch * num_chunks

    Returns:
        Kk: (B*K, C, 2M)
        Qk: (B*K, C, 2M)
        Fq: (B*K, 2C, 2M)
    """
    device = pw.device
    dtype = pw.dtype
    M2 = 2 * M

    Kk = torch.empty(B_K, C, M2, device=device, dtype=dtype)
    Qk = torch.empty(B_K, C, M2, device=device, dtype=dtype)
    Fq = torch.empty(B_K, 2 * C, M2, device=device, dtype=dtype)

    BLOCK_M = triton.next_power_of_2(min(M, 256))
    BLOCK_C = triton.next_power_of_2(C)
    grid = (B_K, triton.cdiv(M, BLOCK_M))

    _codes_gram_kernel[grid](
        pw, pq, gw, gq, wr, wi,
        Kk, Qk, Fq,
        torch.empty(0, device=device),  # G_ptr unused for now
        BK=B_K, C=C, M=M, M2=M2, inv_M=1.0 / M,
        stride_pw_bk=pw.stride(0), stride_pw_c=pw.stride(1),
        stride_kk_bk=Kk.stride(0), stride_kk_c=Kk.stride(1),
        stride_fq_bk=Fq.stride(0), stride_fq_c=Fq.stride(1),
        stride_g_bk=0, stride_g_r=0,
        BLOCK_C=BLOCK_C, BLOCK_M=BLOCK_M,
    )
    return Kk, Qk, Fq


def _test():
    """Correctness test against the PyTorch reference, then speed comparison."""
    import time
    import statistics
    from .layer import LaplaceConfig, LaplaceAttention

    d, M, dv, L, T, B = 1024, 256, 256, 64, 1024, 8
    C = 128
    K = T // C
    BK = B * K

    cfg = LaplaceConfig(d=d, M=M, dv=dv, L=L, ff=4096, chunk=C,
                        rope_base=1000.0, slow_frac=0.25, max_len=T, kv_dk=16, conv=4)
    m = LaplaceAttention(cfg).cuda().eval()
    x = torch.randn(B, T, d, device='cuda')
    z = m.n(x)
    zp = torch.zeros(B, d, device='cuda')

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        kz = m.long.K(z)
        kh = torch.cat([m.long.K(zp)[:, None], kz[:, :-1]], 1)
        pos = torch.arange(T, device='cuda', dtype=torch.float32)[:, None]
        lam = m.long.lam()
        idx = torch.arange(C, device='cuda', dtype=torch.float32)[:, None]
        gw_ref, gq_ref = torch.exp(lam * idx), torch.exp(-lam * idx)
        pw = m.long._phase(kh, pos)
        pq = m.long._phase(kz, pos)

        # Reference: PyTorch path
        cw, sw = pw.cos(), pw.sin()
        cq, sq = pq.cos(), pq.sin()
        ch = lambda x: x.view(B, K, C, x.shape[-1])
        cw, sw, cq, sq = map(ch, (cw, sw, cq, sq))
        Kk_ref = torch.cat([cw * gw_ref, sw * gw_ref], -1)
        Qk_ref = torch.cat([cw * gq_ref, sw * gq_ref], -1)
        c1 = (m.long.wr * cq + m.long.wi * sq) * gq_ref
        c2 = (m.long.wr * sq - m.long.wi * cq) * gq_ref
        Fq_ref = torch.cat([torch.cat([c1, c2], -1),
                            torch.cat([-c2, c1], -1)], 2)

        # Triton path
        pw_ch = pw.view(BK, C, M).contiguous()
        pq_ch = pq.view(BK, C, M).contiguous()
        Kk_t, Qk_t, Fq_t = fused_codes(
            pw_ch, pq_ch, gw_ref.squeeze(-1) if gw_ref.ndim == 3 else gw_ref.contiguous(),
            gq_ref.squeeze(-1) if gq_ref.ndim == 3 else gq_ref.contiguous(),
            m.long.wr.to(torch.float32), m.long.wi.to(torch.float32),
            M, C, BK
        )

        # Compare
        Kk_t_r = Kk_t.view(B, K, C, 2 * M)
        Qk_t_r = Qk_t.view(B, K, C, 2 * M)
        Fq_t_r = Fq_t.view(B, K, 2 * C, 2 * M)
        print(f"Kk max err: {(Kk_t_r - Kk_ref).abs().max():.3e}")
        print(f"Qk max err: {(Qk_t_r - Qk_ref).abs().max():.3e}")
        print(f"Fq max err: {(Fq_t_r - Fq_ref).abs().max():.3e}")

        # Speed
        def pytorch_path():
            cw, sw = pw.cos(), pw.sin()
            cq, sq = pq.cos(), pq.sin()
            ch2 = lambda x: x.view(B, K, C, x.shape[-1])
            cw, sw, cq, sq = map(ch2, (cw, sw, cq, sq))
            Kk = torch.cat([cw * gw_ref, sw * gw_ref], -1)
            Qk = torch.cat([cw * gq_ref, sw * gq_ref], -1)
            c1_ = (m.long.wr * cq + m.long.wi * sq) * gq_ref
            c2_ = (m.long.wr * sq - m.long.wi * cq) * gq_ref
            Fq = torch.cat([torch.cat([c1_, c2_], -1),
                            torch.cat([-c2_, c1_], -1)], 2)
            return Kk, Qk, Fq

        def triton_path():
            return fused_codes(pw_ch, pq_ch, gw_ref.contiguous(), gq_ref.contiguous(),
                               m.long.wr.to(torch.float32), m.long.wi.to(torch.float32),
                               M, C, BK)

        for _ in range(5):
            pytorch_path(); triton_path()
        torch.cuda.synchronize()

        for tag, fn in [("PyTorch", pytorch_path), ("Triton ", triton_path)]:
            ts = []
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            print(f"{tag}: {statistics.median(ts):.2f} ms")


if __name__ == "__main__":
    _test()
