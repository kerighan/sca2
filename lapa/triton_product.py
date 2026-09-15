"""Causal code products without broadcast copies, with fused masked gradients."""

import torch
import triton
import triton.language as tl


@triton.jit
def _product(
    L,
    R,
    DY,
    OUT,
    K: tl.constexpr,
    C: tl.constexpr,
    U: tl.constexpr,
    M2: tl.constexpr,
    G: tl.constexpr,
    GRAM: tl.constexpr,
    MODE: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    ACC: tl.constexpr,
):
    pid = tl.program_id(0)
    if MODE == 2:
        bk = pid
        group = 0
    else:
        bk = pid // G
        group = pid % G
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    j = tl.program_id(2) * TN + tl.arange(0, TN)
    rr = tl.arange(0, TK)
    total = tl.full((TM, TN), 0, tl.float32)
    RED: tl.constexpr = M2 if MODE == 0 else (C if MODE == 1 else U)
    for gidx in tl.static_range(G if MODE == 2 else 1):
        g = gidx if MODE == 2 else group
        acc = tl.full((TM, TN), 0, tl.float32)
        for base in range(tl.cdiv(RED, TK)):
            r = base * TK + rr
            if MODE == 0:
                a = tl.load(
                    L + ((bk * G + g) * U + i[:, None]) * M2 + r[None, :],
                    (i[:, None] < U) & (r[None, :] < M2),
                    0,
                )
                b = tl.load(
                    R + (bk * C + j[:, None]) * M2 + r[None, :],
                    (j[:, None] < C) & (r[None, :] < M2),
                    0,
                )
            elif MODE == 1:
                keep = (
                    (i[:, None] % C > r[None, :])
                    if GRAM
                    else (i[:, None] % C >= r[None, :])
                )
                a = tl.load(
                    DY + ((bk * G + g) * U + i[:, None]) * C + r[None, :],
                    (i[:, None] < U) & (r[None, :] < C) & keep,
                    0,
                ).to(tl.float32)
                if GRAM:
                    a = a / (M2 // 2)
                a = a.to(L.dtype.element_ty)
                b = tl.load(
                    R + (bk * C + r[None, :]) * M2 + j[:, None],
                    (r[None, :] < C) & (j[:, None] < M2),
                    0,
                )
            else:
                keep = (
                    (r[None, :] % C > i[:, None])
                    if GRAM
                    else (r[None, :] % C >= i[:, None])
                )
                a = tl.load(
                    DY + ((bk * G + g) * U + r[None, :]) * C + i[:, None],
                    (r[None, :] < U) & (i[:, None] < C) & keep,
                    0,
                ).to(tl.float32)
                if GRAM:
                    a = a / (M2 // 2)
                a = a.to(L.dtype.element_ty)
                b = tl.load(
                    L + ((bk * G + g) * U + r[None, :]) * M2 + j[:, None],
                    (r[None, :] < U) & (j[:, None] < M2),
                    0,
                )
            acc += tl.dot(a, tl.trans(b), input_precision="tf32x3")
        total += acc.to(L.dtype.element_ty).to(tl.float32)
    if MODE == 0:
        keep = (i[:, None] % C > j[None, :]) if GRAM else (i[:, None] % C >= j[None, :])
        if GRAM:
            total = total / (M2 // 2)
        tl.store(
            OUT + (pid * U + i[:, None]) * C + j[None, :],
            tl.where(keep, total, 0),
            (i[:, None] < U) & (j[None, :] < C),
        )
    else:
        ROWS: tl.constexpr = U if MODE == 1 else C
        ptr = OUT + (pid * ROWS + i[:, None]) * M2 + j[None, :]
        keep = (i[:, None] < ROWS) & (j[None, :] < M2)
        if ACC:
            total += tl.load(ptr, keep, 0.0).to(tl.float32)
        tl.store(ptr, total, keep)


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, gram):
        b, k, g, u, m2 = left.shape
        c = right.shape[2]
        out = torch.empty(
            (b, k, g, u, c),
            device=left.device,
            dtype=torch.float32 if gram else left.dtype,
        )
        _product[(b * k * g, triton.cdiv(u, 32), triton.cdiv(c, 64))](
            left,
            right,
            left,  # MODE 0 ignores DY; never pass one tensor in two output slots
            out,
            K=k,
            C=c,
            U=u,
            M2=m2,
            G=g,
            GRAM=gram,
            MODE=0,
            TM=32,
            TN=64,
            TK=32,
            ACC=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(left, right)
        ctx.gram = gram
        return out

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        b, k, g, u, m2 = left.shape
        c = right.shape[2]
        dl = torch.empty_like(left)
        dr = torch.empty_like(right)
        grad = grad.contiguous()
        kw = dict(
            K=k,
            C=c,
            U=u,
            M2=m2,
            G=g,
            GRAM=ctx.gram,
            TM=32,
            TN=64,
            TK=32,
            ACC=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _product[(b * k * g, triton.cdiv(u, 32), triton.cdiv(m2, 64))](
            left, right, grad, dl, MODE=1, **kw
        )
        _product[(b * k, triton.cdiv(c, 32), triton.cdiv(m2, 64))](
            left, right, grad, dr, MODE=2, **kw
        )
        return dl, dr, None


def code_product(left, right, gram=False):
    return _Product.apply(left.contiguous(), right.contiguous(), gram)
