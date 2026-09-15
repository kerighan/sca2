"""Whole-sequence long-head scan: one launch forward, one reverse, five gradients.

The chunk loop is sequential only in the chunk index: given the codes, the
triangular inverse and the values, every column of the value width is
independent. One program therefore owns (batch, group, column block) and walks
the entire chunk loop, so the scan costs one launch per direction instead of a
handful per chunk, and the state never leaves L2.
"""

import torch
import triton
from triton.runtime.errors import OutOfResources


def _launch(kernel, grid, args, stages=3, **meta):
    """Run with the largest pipeline depth the SM's shared memory allows.

    Under torch.compile these launches are traced into the Inductor graph and
    the depth is baked in, so the configured depth has to fit on its own; the
    retry is the eager safety net for shapes the defaults were not tuned for.
    """
    for depth in range(stages, 0, -1):
        try:
            return kernel[grid](*args, num_stages=depth, **meta)
        except OutOfResources:
            if depth == 1:
                raise


PTUNE = [32, 64, 32, 4]
TUNE = {"fwd": (64, 32, 8, 3), "bwd": (16, 128, 8, 2),
        "bwd2": (64, 32, 4, 3)}


def _tiles(c, d, g, which):
    """(BC, BN, TK, warps, stages) for the scan kernels."""
    bn, tk, warps, stages = TUNE[which]
    bc = max(16, triton.next_power_of_2(c))
    bn = min(bn, max(16, triton.next_power_of_2(d // g)))
    return bc, bn, tk, warps, stages


# Per kind: the fp32 dW operands need small tiles, the bf16 code
# gradients want one row tile so their large operand is read once.
GTUNE = {0: [32, 128, 64, 8, 2], 1: [128, 128, 64, 8, 2],
         2: [128, 128, 64, 8, 2], 4: [128, 128, 64, 8, 2]}
KTUNE = [64, 128, 32, 8, 3]
FTUNE = [128, 128, 32, 8, 3]


def _grid(rows, cols, wid, kind=0):
    tm, tn, tw, warps, stages = GTUNE[kind]
    tm = min(tm, max(16, triton.next_power_of_2(rows)))
    tn = min(tn, max(16, triton.next_power_of_2(cols)))
    tw = min(tw, max(16, triton.next_power_of_2(wid)))
    return tm, tn, tw, warps, stages


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
    from .triton_scan_kernel import _k2

    b, k, g, c, r = fc.shape
    h = r // 2
    tm, tn, tk, warps, stages = KTUNE
    tm = min(tm, max(16, triton.next_power_of_2(c)))
    tn = min(tn, max(16, triton.next_power_of_2(c if mode == 0 else h)))
    tk = min(tk, max(16, triton.next_power_of_2(h if mode == 0 else c)))
    grid = (b * k * (g if mode != 2 else 1), triton.cdiv(c, tm),
            triton.cdiv(c if mode == 0 else h, tn))
    _launch(_k2, grid, (fc, kk, dk2, out), stages=stages,
            K=k, C=c, R=r, G=g, MODE=mode, TM=tm, TN=tn, TK=tk,
            num_warps=warps, enable_fp_fusion=False)


def chunk_forward(kk, qk, fq, v, beta, s, d1, dc, gt, save):
    """Gram, triangular inverse, causal kernel and the scan, in one graph node."""
    from .triton_scan_kernel import _fwd
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
    bc, bn, tk, warps, stages = _tiles(c, d, g, "fwd")
    tk = min(tk, max(16, triton.next_power_of_2(r)))
    states = torch.empty((2, b, r, d), device=s.device, dtype=torch.float32)
    states[0] = s
    out = torch.empty((b, k * c, 2 * d), device=s.device, dtype=torch.float32)
    # Distinct buffers even when unused: aliasing them onto `out` would make
    # Inductor's user-kernel functionalization clone the wrong one.
    mk = lambda sh, dt: torch.empty(sh if save else (1,), device=s.device, dtype=dt)
    eb, rb, hb = (mk((b, k, c, d), t) for t in (kk.dtype, torch.float32, torch.float32))
    s0 = mk((k, b, r, d), kk.dtype)
    _launch(_fwd, (b, g, triton.cdiv(d // g, bn)),
            (qk, kk, fq, k2, w, v, beta, states, d1, dc, gt, out, eb, rb, hb, s0),
            stages=stages, B=b, K=k, C=c, R=r, D=d, G=g, BC=bc, BN=bn, TK=tk,
            SAVE=save, num_warps=warps, enable_fp_fusion=False)
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
    from .triton_scan_kernel import _bwd, _bwd2, _dfc, _grad

    b, k, c, r = kk.shape
    d, g = hb.shape[-1], fq.shape[2]
    bc, bn, tk, warps, stages = _tiles(c, d, g, "bwd")
    tk = min(tk, max(16, triton.next_power_of_2(r)))
    nc = triton.cdiv(d // g, bn)
    bn2, tk2, warps2, stages2 = TUNE["bwd2"]
    bn2 = min(bn2, max(16, triton.next_power_of_2(d // g)))
    tk2 = min(tk2, max(16, triton.next_power_of_2(r // 2)))
    nc2, mt = triton.cdiv(d // g, bn2), triton.cdiv(r // 2, tk2)
    dsr = dsf.contiguous().clone()
    de = torch.empty_like(eb)
    drm = torch.empty_like(eb)
    dsg = torch.empty((b, k, r, d), device=kk.device, dtype=kk.dtype)
    dv = torch.empty((b, k, c, d), device=kk.device, dtype=torch.float32)
    pb = torch.empty((g * nc, b, k, c), device=kk.device, dtype=torch.float32)
    pc = torch.empty((b * g * nc * k, r), device=kk.device, dtype=torch.float32)
    pd = torch.empty((b * g * nc2 * k, r), device=kk.device, dtype=torch.float32)
    for n in range(k - 1, -1, -1):
        _launch(_bwd, (b, g, nc),
                (kk, k2, w, beta, s0, d1, gt, eb, rb, do, dsr, de, drm, dsg,
                 dv, pb, pc, n),
                stages=stages, B=b, K=k, C=c, R=r, D=d, G=g, BC=bc, BN=bn,
                TK=tk, NC=nc, num_warps=warps, enable_fp_fusion=False)
        _launch(_bwd2, (b * g, nc2, mt),
                (qk, fq, do, drm, s0, dsr, d1, dc, pd, n),
                stages=stages2, B=b, K=k, C=c, R=r, D=d, G=g, BC=bc, BN=bn2,
                TK=tk2, NC=nc2, num_warps=warps2, enable_fp_fusion=False)
    tm0 = min(GTUNE[2][0], max(16, triton.next_power_of_2(c)))
    pg = torch.empty((b * k * triton.cdiv(c, tm0), r), device=kk.device,
                     dtype=torch.float32)
    dw = torch.empty_like(w)
    dq = torch.empty_like(qk)
    dk = torch.empty_like(kk)
    df = torch.empty_like(fq)
    dk2 = torch.empty_like(k2)
    ftm, ftn, ftw, fw, fs = FTUNE
    ftm = min(ftm, max(16, triton.next_power_of_2(c)))
    ftn = min(ftn, max(16, triton.next_power_of_2(r // 2)))
    ftw = min(ftw, max(16, triton.next_power_of_2(d // g)))
    _launch(_dfc, (b * k * g, triton.cdiv(c, ftm), triton.cdiv(r // 2, ftn)),
            (do, s0, d1, df), stages=fs, B=b, K=k, C=c, R=r, D=d, G=g,
            TM=ftm, TN=ftn, TW=ftw, num_warps=fw, enable_fp_fusion=False)
    for kind, (a, bb, out, rows, cols, wid) in ((0, (de, hb, dw, c, c, d)),
                                                (1, (drm, None, dq, c, r, d)),
                                                (2, (eb, dsg, dk, c, r, d)),
                                                (4, (do, eb, dk2, 2 * c, c, d // g))):
        tm, tn, tw, gw, gs = _grid(rows, cols, wid, kind)
        _launch(_grad, (b * k * (g if kind >= 3 else 1), triton.cdiv(rows, tm),
                        triton.cdiv(cols, tn)),
                (a, bb if bb is not None else a, s0, d1, out, kk, gt, pg),
                stages=gs,
                B=b, K=k, C=c, R=r, D=d, G=g, KIND=kind, TM=tm, TN=tn, TW=tw,
                num_warps=gw, enable_fp_fusion=False)
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
