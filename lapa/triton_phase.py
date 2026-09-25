"""Fused phase/decay/code construction and analytical code gradients.

Inputs are projected keys K(z), where z already includes conv_silu when set.
The activation is fused before the projections in triton_conv.py: moving it
here would compute silu(K(z)) instead of K(silu(z)), changing both heads.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _codes(
    KZ,
    KH,
    TH,
    OM,
    LAM,
    WR,
    WI,
    POS,
    KK,
    QK,
    FQ,
    CT,
    K: tl.constexpr,
    C: tl.constexpr,
    M: tl.constexpr,
    G: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    COMPACT: tl.constexpr,
    HAS_CT: tl.constexpr,
):
    bk = tl.program_id(0) // tl.cdiv(C, BT)
    tile = tl.program_id(0) % tl.cdiv(C, BT)
    t = tile * BT + tl.arange(0, BT)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    mask = (t[:, None] < C) & (m[None, :] < M)
    off = (bk * C + t[:, None]) * M + m[None, :]
    kz = tl.load(KZ + off, mask, 0).to(tl.float32)
    kh = tl.load(KH + off, mask, 0).to(tl.float32)
    theta = tl.load(TH + m, m < M, 0)
    omega = tl.load(OM + m, m < M, 0)
    pos = tl.load(POS) + (bk % K) * C + t
    pw = kh * theta[None, :] + pos[:, None] * omega[None, :]
    pq = kz * theta[None, :] + pos[:, None] * omega[None, :]
    cw = libdevice.cos(pw)
    sw = libdevice.sin(pw)
    cq = libdevice.cos(pq)
    sq = libdevice.sin(pq)
    if HAS_CT:
        # Data-dependent decay: the ramp is a per-token cumulative sum computed
        # by the caller, not idx * lam. Without this branch `decay_input` fell
        # off this kernel entirely (layer.py's use_codes had `and not di`) and
        # the whole code path reverted to PyTorch -- measured at 27% of the
        # arm's throughput on a 5090, which is more than the mechanism itself
        # was ever going to be worth.
        ct = tl.load(CT + off, mask, 0).to(tl.float32)
        gw = tl.exp(ct)
        gq = tl.exp(-ct)
    else:
        lam = tl.load(LAM + m, m < M, 0)
        idx = tl.minimum(t, C - 1)  # Padding must not overflow before a masked reduction.
        gw = tl.exp(idx[:, None] * lam[None, :])
        gq = tl.exp(-idx[:, None] * lam[None, :])
    dest = (bk * C + t[:, None]) * (2 * M) + m[None, :]
    tl.store(KK + dest, cw * gw, mask)
    tl.store(KK + dest + M, sw * gw, mask)
    tl.store(QK + dest, cw * gq, mask)
    tl.store(QK + dest + M, sw * gq, mask)
    ROWS: tl.constexpr = C if COMPACT else 2 * C
    for g in tl.static_range(G):
        wr = tl.load(WR + m * G + g, m < M, 0)
        wi = tl.load(WI + m * G + g, m < M, 0)
        a = (wr[None, :] * cq + wi[None, :] * sq) * gq
        z = (wr[None, :] * sq - wi[None, :] * cq) * gq
        dest = ((bk * G + g) * ROWS + t[:, None]) * (2 * M) + m[None, :]
        tl.store(FQ + dest, a, mask)
        tl.store(FQ + dest + M, z, mask)
        if not COMPACT:
            # The second row block is the same pair rotated; the compact layout
            # stores [c1 | c2] once and lets the consumers apply the rotation.
            tl.store(FQ + dest + C * 2 * M, -z, mask)
            tl.store(FQ + dest + C * 2 * M + M, a, mask)


@triton.jit
def _codes_backward(
    KZ,
    KH,
    TH,
    OM,
    LAM,
    WR,
    WI,
    POS,
    DK,
    DQ,
    DF,
    DZ,
    DH,
    PART,
    CT,
    DCT,
    K: tl.constexpr,
    C: tl.constexpr,
    M: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    COMPACT: tl.constexpr,
    HAS_CT: tl.constexpr,
):
    pid = tl.program_id(0)
    bk = pid // tl.cdiv(C, BT)
    tile = pid % tl.cdiv(C, BT)
    t = tile * BT + tl.arange(0, BT)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    mask = (t[:, None] < C) & (m[None, :] < M)
    off = (bk * C + t[:, None]) * M + m[None, :]
    kz = tl.load(KZ + off, mask, 0).to(tl.float32)
    kh = tl.load(KH + off, mask, 0).to(tl.float32)
    theta = tl.load(TH + m, m < M, 0)
    omega = tl.load(OM + m, m < M, 0)
    pos = tl.load(POS) + (bk % K) * C + t
    pw = kh * theta[None, :] + pos[:, None] * omega[None, :]
    pq = kz * theta[None, :] + pos[:, None] * omega[None, :]
    cw = libdevice.cos(pw)
    sw = libdevice.sin(pw)
    cq = libdevice.cos(pq)
    sq = libdevice.sin(pq)
    if HAS_CT:
        ct = tl.load(CT + off, mask, 0).to(tl.float32)
        gw = tl.exp(ct)
        gq = tl.exp(-ct)
    else:
        lam = tl.load(LAM + m, m < M, 0)
        idx = tl.minimum(t, C - 1)
        gw = tl.exp(idx[:, None] * lam[None, :])
        gq = tl.exp(-idx[:, None] * lam[None, :])
    ptr = (bk * C + t[:, None]) * (2 * M) + m[None, :]
    dkr = tl.load(DK + ptr, mask, 0).to(tl.float32)
    dki = tl.load(DK + ptr + M, mask, 0).to(tl.float32)
    dqr = tl.load(DQ + ptr, mask, 0).to(tl.float32)
    dqi = tl.load(DQ + ptr + M, mask, 0).to(tl.float32)
    dcw = dkr * gw + dqr * gq
    dsw = dki * gw + dqi * gq
    dpw = dsw * cw - dcw * sw
    dgw = dkr * cw + dki * sw
    dgq = dqr * cw + dqi * sw
    dcq = tl.full((BT, BM), 0, tl.float32)
    dsq = tl.full((BT, BM), 0, tl.float32)
    ROWS: tl.constexpr = C if COMPACT else 2 * C
    for g in tl.static_range(G):
        wr = tl.load(WR + m * G + g, m < M, 0)
        wi = tl.load(WI + m * G + g, m < M, 0)
        dest = ((bk * G + g) * ROWS + t[:, None]) * (2 * M) + m[None, :]
        da = tl.load(DF + dest, mask, 0).to(tl.float32)
        dz = tl.load(DF + dest + M, mask, 0).to(tl.float32)
        if not COMPACT:
            da += tl.load(DF + dest + C * 2 * M + M, mask, 0).to(tl.float32)
            dz -= tl.load(DF + dest + C * 2 * M, mask, 0).to(tl.float32)
        dgq += da * (wr[None, :] * cq + wi[None, :] * sq) + dz * (
            wr[None, :] * sq - wi[None, :] * cq
        )
        da = da * gq
        dz = dz * gq
        dcq += da * wr[None, :] - dz * wi[None, :]
        dsq += da * wi[None, :] + dz * wr[None, :]
        tl.store(
            PART + ((4 + g) * P + pid) * M + m, tl.sum(da * cq + dz * sq, 0), m < M
        )
        tl.store(
            PART + ((4 + G + g) * P + pid) * M + m, tl.sum(da * sq - dz * cq, 0), m < M
        )
    dpq = dsq * cq - dcq * sq
    tl.store(DZ + off, dpq * theta[None, :], mask)
    tl.store(DH + off, dpw * theta[None, :], mask)
    tl.store(PART + pid * M + m, tl.sum(dpw * kh + dpq * kz, 0), m < M)
    if HAS_CT:
        # gw = exp(Ct) and gq = exp(-Ct), so the gradient is elementwise and
        # there is no reduction over t and no factor t. lam itself is unused on
        # this branch and gets no gradient.
        tl.store(DCT + off, dgw * gw - dgq * gq, mask)
        tl.store(PART + (P + pid) * M + m, tl.zeros((BM,), tl.float32), m < M)
    else:
        tl.store(
            PART + (P + pid) * M + m,
            tl.sum(t[:, None] * (dgw * gw - dgq * gq), 0), m < M
        )
    tl.store(PART + (2 * P + pid) * M + m, tl.sum((dpw + dpq) * pos[:, None], 0), m < M)
    tl.store(
        PART + (3 * P + pid) * M + m, tl.sum((dpw + dpq) * omega[None, :], 0), m < M
    )


class _Codes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kz, kh, theta, omega, lam, wr, wi, pos, c, dtype, compact, ct):
        b, t, m = kz.shape
        k = t // c
        g = 1 if wr.ndim == 1 else wr.shape[1]
        kk = torch.empty((b, k, c, 2 * m), device=kz.device, dtype=dtype)
        qk = torch.empty_like(kk)
        rows = c if compact else 2 * c
        fq = torch.empty((b, k, g, rows, 2 * m), device=kz.device, dtype=dtype)
        _codes[(b * k * triton.cdiv(c, 16), triton.cdiv(m, 64))](
            kz,
            kh,
            theta,
            omega,
            lam,
            wr,
            wi,
            pos,
            kk,
            qk,
            fq,
            ct,
            K=k,
            C=c,
            M=m,
            G=g,
            BT=16,
            BM=64,
            COMPACT=compact,
            HAS_CT=ct is not None,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(kz, kh, theta, omega, lam, wr, wi, pos)
        ctx.ct = ct
        ctx.c = c
        ctx.compact = compact
        return kk, qk, fq

    @staticmethod
    def backward(ctx, dk, dq, df):
        kz, kh, theta, omega, lam, wr, wi, pos = ctx.saved_tensors
        b, t, m = kz.shape
        c = ctx.c
        k = t // c
        g = 1 if wr.ndim == 1 else wr.shape[1]
        p = b * k * triton.cdiv(c, 16)
        part = torch.empty((4 + 2 * g, p, m), device=kz.device, dtype=torch.float32)
        dz = torch.empty_like(kz)
        dh = torch.empty_like(kh)
        ct = ctx.ct
        dct = torch.empty_like(ct) if ct is not None else None
        _codes_backward[(p, triton.cdiv(m, 64))](
            kz,
            kh,
            theta,
            omega,
            lam,
            wr,
            wi,
            pos,
            dk.contiguous(),
            dq.contiguous(),
            df.contiguous(),
            dz,
            dh,
            part,
            ct,
            dct,
            K=k,
            C=c,
            M=m,
            G=g,
            P=p,
            BT=16,
            BM=64,
            COMPACT=ctx.compact,
            HAS_CT=ct is not None,
            enable_fp_fusion=False,
        )
        grad = part.sum(1)
        return (
            dz,
            dh,
            grad[0],
            grad[2],
            None if ct is not None else grad[1],
            grad[4 : 4 + g].T.reshape_as(wr),
            grad[4 + g :].T.reshape_as(wi),
            grad[3].sum().reshape_as(pos),
            None,
            None,
            None,
            dct,
        )


def phase_codes(kz, kh, theta, omega, lam, wr, wi, pos, c, dtype, compact=False,
                ct=None):
    """Kk, Qk and the read codes. `compact` emits [c1 | c2] without the rotation.

    `ct` (B, K, C, M) replaces the idx * lam ramp with a per-token cumulative
    decay, which is what --decay-input needs. Passing it makes lam unused and
    ungradiented; the gradient comes back on ct instead.
    """
    return _Codes.apply(
        kz.contiguous(), kh.contiguous(), theta, omega, lam, wr, wi, pos, c, dtype,
        compact, None if ct is None else ct.contiguous(),
    )
