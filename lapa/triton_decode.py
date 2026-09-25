"""One fused Triton kernel for the long head's decode step.

Decoding one token touches the (2M, dv) state four times: decay it, reduce it
against the write code to get vhat, rank-1 update it, then reduce it again
against the read code. In PyTorch those are four kernels over the same 0.5 MB,
and torch.compile cannot merge them -- it sees one graph with no breaks and 116
ops, and still emits separate launches, because they are reductions of
different shapes chained by a dependency. Measured on a GB10 at batch 1, decode
is neither bandwidth- nor CPU-bound: it is bound by the NUMBER of kernels, and
long.step alone moves ~6 MB in 0.250 ms, which is 24 GB/s on a card that does
an order of magnitude more.

This does the whole step in one launch. Each program owns a tile of dv columns
and walks the 2M rows twice: once to accumulate vhat, once to write the updated
state and accumulate the two read rows. The second walk hits L2, since the
entire state is 0.5 MB.

SCOPE. The fast path covers the configuration all three campaign arms use --
NG=1, one beta group, no decay_input, no kv_dk, no beta_write. Anything else
falls back to the PyTorch step, which stays the reference: `check()` asserts
the two agree, and `lapa/test_decode_kernel.py` runs it over the real
checkpoints rather than over random weights, because a kernel that is correct
on N(0,1) and wrong on a trained decay is exactly the failure this would hide.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                           # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _long_step(
        S, DAMP, KT, QT, VV, BETA, SOUT, U,
        M2: tl.constexpr, DV: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr,
        INV_M,
    ):
        b = tl.program_id(0)
        t = tl.program_id(1)
        cols = t * BLOCK_DV + tl.arange(0, BLOCK_DV)
        cmask = cols < DV

        # ---- pass 1: vhat = (kt . (s * damp)) / M -------------------------- #
        vhat = tl.zeros([BLOCK_DV], dtype=tl.float32)
        for r0 in range(0, M2, BLOCK_M):
            rows = r0 + tl.arange(0, BLOCK_M)
            rmask = rows < M2
            s = tl.load(S + b * M2 * DV + rows[:, None] * DV + cols[None, :],
                        mask=rmask[:, None] & cmask[None, :], other=0.0)
            d = tl.load(DAMP + rows, mask=rmask, other=0.0)
            k = tl.load(KT + b * M2 + rows, mask=rmask, other=0.0)
            vhat += tl.sum(s * d[:, None] * k[:, None], 0)
        vhat = vhat * INV_M

        beta = tl.load(BETA + b)
        v = tl.load(VV + b * DV + cols, mask=cmask, other=0.0)
        e = v - beta * vhat                                   # (BLOCK_DV,)

        # ---- pass 2: s <- s*damp + e (x) kt, and u = (qt . s) / M ---------- #
        u0 = tl.zeros([BLOCK_DV], dtype=tl.float32)
        u1 = tl.zeros([BLOCK_DV], dtype=tl.float32)
        for r0 in range(0, M2, BLOCK_M):
            rows = r0 + tl.arange(0, BLOCK_M)
            rmask = rows < M2
            off = b * M2 * DV + rows[:, None] * DV + cols[None, :]
            full = rmask[:, None] & cmask[None, :]
            s = tl.load(S + off, mask=full, other=0.0)
            d = tl.load(DAMP + rows, mask=rmask, other=0.0)
            k = tl.load(KT + b * M2 + rows, mask=rmask, other=0.0)
            sn = s * d[:, None] + e[None, :] * k[:, None]
            tl.store(SOUT + off, sn, mask=full)
            q0 = tl.load(QT + b * 2 * M2 + rows, mask=rmask, other=0.0)
            q1 = tl.load(QT + b * 2 * M2 + M2 + rows, mask=rmask, other=0.0)
            u0 += tl.sum(sn * q0[:, None], 0)
            u1 += tl.sum(sn * q1[:, None], 0)

        tl.store(U + b * 2 * DV + cols, u0 * INV_M, mask=cmask)
        tl.store(U + b * 2 * DV + DV + cols, u1 * INV_M, mask=cmask)


def supported(head) -> bool:
    """True when the fast path implements exactly this head's function."""
    cfg = head.cfg
    return (HAVE_TRITON
            and head.NG == 1 and head.bg == 1 and head.dk == 0
            and not cfg.decay_input and not cfg.beta_write
            and head.wd == torch.float32)


def long_step(s, damp, kt, qt, v, beta, block_dv: int = 32, block_m: int = 64):
    """s (B,2M,dv) -> (s_new, u) with u (B, 2*dv), matching LongHead.step.

    `damp` is (2M,1) or (2M,), `kt` (B,2M), `qt` (B,2,2M), `v` (B,dv),
    `beta` (B,1) or (B,). Everything float32 and contiguous.
    """
    B, M2, DV = s.shape
    s = s.contiguous()
    out = torch.empty_like(s)
    u = torch.empty((B, 2, DV), device=s.device, dtype=torch.float32)
    grid = (B, triton.cdiv(DV, block_dv))
    _long_step[grid](
        s, damp.reshape(-1).contiguous(), kt.contiguous(), qt.contiguous(),
        v.contiguous(), beta.reshape(-1).contiguous(), out, u,
        M2=M2, DV=DV, BLOCK_M=block_m, BLOCK_DV=block_dv,
        INV_M=2.0 / M2,                       # M = M2 / 2
    )
    return out, u.reshape(B, 2 * DV)
