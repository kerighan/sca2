"""Whole-sequence long-head scan with shape-specific Triton autotuning.

The chunk loop is sequential only in the chunk index: given the codes, the
triangular inverse and the values, every column of the value width is
independent. One program therefore owns (batch, group, column block) and walks
the entire chunk loop. The reverse scan uses two launches per chunk to expose
mode parallelism. Live state stays in a two-slot buffer that fits in L2.
"""

import torch
import triton
from .triton_scan_tune import FTUNE, GTUNE, KTUNE, TUNE


PTUNE = [32, 64, 32, 4]

def _prod(left, right, grad, out, gram, mode, acc=False):
    """One launch of the shared causal code product."""
    from .triton_product import _product

    b, k, g, u, m2 = left.shape
    c = right.shape[2]
    rows, cols = (u, c) if mode == 0 else ((u, m2) if mode == 1 else (c, m2))
    tm, tn, tk, warps = PTUNE
    tm = min(tm, max(16, triton.next_power_of_2(rows)))
    tn = min(tn, max(16, triton.next_power_of_2(cols)))
    tk = min(tk, max(16, triton.next_power_of_2(m2 if mode == 0 else
                                                (c if mode == 1 else u))))
    _product[(b * k * (g if mode != 2 else 1), triton.cdiv(rows, tm),
              triton.cdiv(cols, tn))](
        left, right, grad, out, K=k, C=c, U=u, M2=m2, G=g, GRAM=gram, MODE=mode,
        TM=tm, TN=tn, TK=tk, ACC=acc, num_warps=warps, enable_fp_fusion=False,
    )


def _k2run(fc, kk, dk2, out, mode):
    from .triton_scan_kernel import _k2, _k2_small, _k2_fp32

    b, k, g, c, r = fc.shape
    h = r // 2
    kernel = _k2_small if c < 32 or r < 64 else (_k2_fp32 if fc.dtype == torch.float32 else _k2)
    grid = lambda meta: (b * k * (g if mode != 2 else 1), triton.cdiv(c, meta["TM"]),
                         triton.cdiv(c if mode == 0 else h, meta["TN"]))
    kernel[grid](fc, kk, dk2, out, B=b, K=k, C=c, R=r, G=g, MODE=mode, DEV=fc.device.index,
              enable_fp_fusion=False)


def chunk_forward(kk, qk, fq, v, beta, s, d1, dc, gt, save):
    """Gram, triangular inverse, causal kernel and the scan, in one graph node."""
    from .triton_scan_kernel import _fwd, _fwd_small, _fwd_fp32
    from .triton_solve_kernel import inverse_forward

    b, k, c, r = kk.shape
    d, g = v.shape[-1], fq.shape[2]
    qv = qk.unsqueeze(2)
    gram = torch.empty((b, k, 1, c, c), device=kk.device, dtype=torch.float32)
    _prod(qv, kk, qv, gram, True, 0)
    gram = gram.squeeze(2)
    w = inverse_forward(gram, beta)
    k2 = torch.empty((b, k, g, 2 * c, c), device=kk.device, dtype=kk.dtype)
    _k2run(fq, kk, kk, k2, 0)
    bc = max(16, triton.next_power_of_2(c))
    states = torch.empty((2, b, r, d), device=s.device, dtype=torch.float32)
    states[0] = s
    out = torch.empty((b, k * c, 2 * d), device=s.device, dtype=torch.float32)
    # Distinct buffers even when unused: aliasing them onto `out` would make
    # Inductor's user-kernel functionalization clone the wrong one.
    mk = lambda sh, dt: torch.empty(sh if save else (1,), device=s.device, dtype=dt)
    eb, rb, hb = (mk((b, k, c, d), t) for t in (kk.dtype, torch.float32, torch.float32))
    s0 = mk((k, b, r, d), kk.dtype)
    kernel = _fwd_small if c < 32 or r < 64 else (_fwd_fp32 if kk.dtype == torch.float32 else _fwd)
    kernel[lambda meta: (b, g, triton.cdiv(d // g, meta["BN"]))](
        qk, kk, fq, k2, w, v, beta, states, d1, dc, gt, out, eb, rb, hb, s0,
        B=b, K=k, C=c, R=r, D=d, G=g, BC=bc, SAVE=save, DEV=kk.device.index, enable_fp_fusion=False)
    return out, states[k % 2], eb, rb, hb, gram, w, k2, s0


def chunk_backward(kk, qk, fq, k2, w, gram, beta, d1, dc, gt, eb, rb, hb,
                   s0, do, dsf):
    from .triton_solve_kernel import inverse_backward

    # Rounding the output gradient once is exactly what every consumer did.
    from .triton_scan_kernel import _scaled

    dob = torch.empty(do.shape, device=do.device, dtype=kk.dtype)
    _scaled[(triton.cdiv(do.numel(), 1024),)](do, dob, do.numel(),
                                              2.0 / fq.shape[-1], BLK=1024,
                                              num_warps=4)
    do = dob
    dk, dq, dfq, dk2, dw, dv, dbeta, dsr, dd1, ddc, dgt = _scan_backward(
        kk, qk, fq, k2, w, beta, d1, dc, gt, eb, rb, hb, s0, do, dsf
    )
    dgram, dbi = inverse_backward(gram, w, dw, beta)
    # Every gradient that the codes feed lands in one buffer: the causal
    # products accumulate onto what the scan already wrote, so no add pass runs.
    _k2run(fq, kk, dk2, dfq, 1)
    _k2run(fq, kk, dk2, dk, 2)
    qv, dqv = qk.unsqueeze(2), dq.unsqueeze(2)
    _prod(qv, kk, dgram, dqv, True, 1, acc=True)
    _prod(qv, kk, dgram, dk, True, 2, acc=True)
    return dk, dq, dfq, dv, dbeta + dbi, dsr, dd1, ddc, dgt


def _scan_backward(kk, qk, fq, k2, w, beta, d1, dc, gt, eb, rb, hb,
                   s0, do, dsf):
    from .triton_scan_kernel import (_bwd, _bwd2, _dfc, _grad, _grad0,
                                     _bwd_small, _bwd2_small, _dfc_small, _grad_small,
                                     _bwd_fp32, _bwd2_fp32, _dfc_fp32, _grad_fp32)

    b, k, c, r = kk.shape
    d, g = hb.shape[-1], fq.shape[2]
    bc = max(16, triton.next_power_of_2(c))
    # Kernels write/zero every scratch slot regardless of the selected tile.
    nc = triton.cdiv(d // g, 16)
    dsr = dsf.contiguous().clone()
    de = torch.empty_like(eb)
    drm = torch.empty_like(eb)
    dsg = torch.empty((b, k, r, d), device=kk.device, dtype=kk.dtype)
    dv = torch.empty((b, k, c, d), device=kk.device, dtype=torch.float32)
    pb = torch.empty((g * nc, b, k, c), device=kk.device, dtype=torch.float32)
    pc = torch.empty((b * g * nc * k, r), device=kk.device, dtype=torch.float32)
    pd = torch.empty_like(pc)
    small = c < 32 or r < 64
    fp32 = kk.dtype == torch.float32
    backward = _bwd_small if small else (_bwd_fp32 if fp32 else _bwd)
    backward2 = _bwd2_small if small else (_bwd2_fp32 if fp32 else _bwd2)
    dfc = _dfc_small if small else (_dfc_fp32 if fp32 else _dfc)
    for n in range(k - 1, -1, -1):
        backward[lambda meta: (b, g, triton.cdiv(d // g, meta["BN"]))](
            kk, k2, w, beta, gt, rb, do, dsr, de, drm, dsg,
            dv, pb, n, B=b, K=k, C=c, R=r, D=d, G=g, BC=bc,
            NC=nc, DEV=kk.device.index, enable_fp_fusion=False)
        backward2[lambda meta: (b * g, triton.cdiv(d // g, meta["BN"]),
                            triton.cdiv(r // 2, meta["TK"]))](
            qk, fq, do, drm, s0, dsr, d1, dc, pd, pc, n,
            B=b, K=k, C=c, R=r, D=d, G=g, BC=bc, NC=nc, DEV=kk.device.index, enable_fp_fusion=False)
    pg = torch.empty((b * k * triton.cdiv(c, 16), r), device=kk.device,
                     dtype=torch.float32)
    dw = torch.empty_like(w)
    dq = torch.empty_like(qk)
    dk = torch.empty_like(kk)
    df = torch.empty_like(fq)
    dk2 = torch.empty_like(k2)
    dfc[lambda meta: (b * k * g, triton.cdiv(c, meta["TM"]),
                       triton.cdiv(r // 2, meta["TN"]))](
        do, s0, d1, df, B=b, K=k, C=c, R=r, D=d, G=g, DEV=kk.device.index, enable_fp_fusion=False)
    for kind, (a, bb, out, rows, cols, wid) in ((0, (de, hb, dw, c, c, d)),
                                                (1, (drm, None, dq, c, r, d)),
                                                (2, (eb, dsg, dk, c, r, d)),
                                                (4, (do, eb, dk2, 2 * c, c, d // g))):
        grad = _grad_small if small else (_grad0 if kind == 0 else (_grad_fp32 if fp32 else _grad))
        grad[lambda meta: (b * k * (g if kind >= 3 else 1),
                            triton.cdiv(rows, meta["TM"]), triton.cdiv(cols, meta["TN"]))](
            a, bb if bb is not None else a, s0, d1, out, kk, gt, pg,
            B=b, K=k, C=c, R=r, D=d, G=g, KIND=kind, DEV=kk.device.index, enable_fp_fusion=False)
    return (dk, dq, df, dk2, dw, dv, pb.sum(0).unsqueeze(-1), dsr,
            pd.sum(0).unsqueeze(-1), pc.sum(0).unsqueeze(-1),
            pg.sum(0).unsqueeze(-1))


class _LongChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kk, qk, fq, v, beta, s, d1, dc, gt, save):
        out, final, eb, rb, hb, gram, w, k2, s0 = chunk_forward(
            kk, qk, fq, v, beta, s, d1, dc, gt, save
        )
        ctx.save_for_backward(kk, qk, fq, k2, w, gram, beta, d1, dc, gt,
                              eb, rb, hb, s0)
        return out, final

    @staticmethod
    def backward(ctx, do, dsf):
        return (*chunk_backward(*ctx.saved_tensors, do.contiguous(), dsf), None)


def long_chunk(kk, qk, fq, v, beta, s, d1, dc, gt):
    """Gram, inverse, causal kernel and chunk scan as a single autograd node.

    Returns the packed (B, K*C, 2*D) output -- [Re | Im] channels, already
    divided by M -- and the final state. The incoming state is not mutated.
    """
    return _LongChunk.apply(
        kk.contiguous(), qk.contiguous(), fq.contiguous(), v.contiguous(), beta,
        s.contiguous(), d1, dc, gt, torch.is_grad_enabled(),
    )
