"""CUDA implementation, lazily imported by the experimental fp32 solve path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse(
    G,
    BETA,
    OUT,
    C: tl.constexpr,
    R: tl.constexpr,
    COLS: tl.constexpr,
    SHARED: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, R)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    w = tl.cast(rows[:, None] == cols[None, :], tl.float32)
    for i in range(C):
        a = tl.load(G + batch * C * C + i * C + rows, mask=rows < i, other=0.0)
        if SHARED:
            a = a * tl.load(BETA + batch * C + i)
        row = tl.cast(cols == i, tl.float32) - tl.sum(a[:, None] * w, axis=0)
        w = tl.where(rows[:, None] == i, row[None, :], w)
    tl.store(
        OUT + batch * C * C + rows[:, None] * C + cols[None, :],
        w,
        mask=(rows[:, None] < C) & (cols[None, :] < C),
    )


def inverse_forward(gram, beta):
    c = gram.shape[-1]
    out = torch.empty_like(gram)
    _inverse[(gram.numel() // (c * c), triton.cdiv(c, 16))](
        gram,
        gram if beta is None else beta,
        out,
        C=c,
        R=triton.next_power_of_2(c),
        COLS=16,
        SHARED=beta is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _inverse_bwd_left(W, DW, TMP, C: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(0)
    i = tl.program_id(1) * 32 + tl.arange(0, 32)
    j = tl.program_id(2) * 32 + tl.arange(0, 32)
    rr = tl.arange(0, 32)
    acc = tl.full((32, 32), 0, tl.float32)
    for base in range(tl.cdiv(C, 32)):
        r = base * 32 + rr
        a = tl.load(
            W + (b * C + r[None, :]) * C + i[:, None],
            (r[None, :] < C) & (i[:, None] < C),
            0,
        )
        z = tl.load(
            DW + (b * C + r[:, None]) * C + j[None, :],
            (r[:, None] < C) & (j[None, :] < C),
            0,
        )
        acc += tl.dot(a, z, input_precision="tf32x3")
    tl.store(
        TMP + (b * C + i[:, None]) * C + j[None, :],
        acc,
        (i[:, None] < C) & (j[None, :] < C),
    )


@triton.jit
def _inverse_bwd_right(
    W, TMP, G, BETA, DG, DB, C: tl.constexpr, BC: tl.constexpr, SHARED: tl.constexpr
):
    b = tl.program_id(0)
    i = tl.program_id(1) * 32 + tl.arange(0, 32)
    j = tl.arange(0, BC)
    rr = tl.arange(0, 32)
    acc = tl.full((32, BC), 0, tl.float32)
    for base in range(tl.cdiv(C, 32)):
        r = base * 32 + rr
        a = tl.load(
            TMP + (b * C + i[:, None]) * C + r[None, :],
            (i[:, None] < C) & (r[None, :] < C),
            0,
        )
        w = tl.load(
            W + (b * C + j[None, :]) * C + r[:, None],
            (j[None, :] < C) & (r[:, None] < C),
            0,
        )
        acc += tl.dot(a, w, input_precision="tf32x3")
    da = tl.where(i[:, None] > j[None, :], -acc, 0)
    mask = (i[:, None] < C) & (j[None, :] < C)
    if SHARED:
        beta = tl.load(BETA + b * C + i, i < C, 0)
        g = tl.load(
            G + (b * C + i[:, None]) * C + j[None, :],
            mask & (i[:, None] > j[None, :]),
            0,
        )
        tl.store(DB + b * C + i, tl.sum(da * g, 1), i < C)
        da = da * beta[:, None]
    tl.store(DG + (b * C + i[:, None]) * C + j[None, :], da, mask)


def inverse_backward(gram, inverse, grad, beta):
    c = gram.shape[-1]
    batches = gram.numel() // (c * c)
    tmp = torch.empty_like(gram)
    dg = torch.empty_like(gram)
    db = (
        torch.empty_like(beta)
        if beta is not None
        else torch.empty(0, device=gram.device)
    )
    bc = max(16, triton.next_power_of_2(c))
    _inverse_bwd_left[(batches, triton.cdiv(c, 32), triton.cdiv(c, 32))](
        inverse, grad.contiguous(), tmp, C=c, BC=bc, num_warps=4, enable_fp_fusion=False
    )
    _inverse_bwd_right[(batches, triton.cdiv(c, 32))](
        inverse,
        tmp,
        gram,
        gram if beta is None else beta,
        dg,
        db,
        C=c,
        BC=bc,
        SHARED=beta is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dg, db if beta is not None else None
