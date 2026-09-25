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
                        mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
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
            s = tl.load(S + off, mask=full, other=0.0).to(tl.float32)
            d = tl.load(DAMP + rows, mask=rmask, other=0.0)
            k = tl.load(KT + b * M2 + rows, mask=rmask, other=0.0)
            sn = s * d[:, None] + e[None, :] * k[:, None]
            tl.store(SOUT + off, sn.to(SOUT.dtype.element_ty), mask=full)
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
    out = torch.empty_like(s)   # same dtype as the incoming state
    u = torch.empty((B, 2, DV), device=s.device, dtype=torch.float32)
    grid = (B, triton.cdiv(DV, block_dv))
    _long_step[grid](
        s, damp.reshape(-1).contiguous(), kt.contiguous(), qt.contiguous(),
        v.contiguous(), beta.reshape(-1).contiguous(), out, u,
        M2=M2, DV=DV, BLOCK_M=block_m, BLOCK_DV=block_dv,
        INV_M=2.0 / M2,                       # M = M2 / 2
    )
    return out, u.reshape(B, 2 * DV)


# =============================================================================
#  SHORT HEAD: the whole decode step in one launch
# =============================================================================
if HAVE_TRITON:

    @triton.jit
    def _short_step(
        KH, KZ, VZ, THETA, OMEGA, WR, WI, POS, PTR,
        CW, SW, EW, OUT,
        L: tl.constexpr, DV: tl.constexpr,
        BLOCK_DV: tl.constexpr, EPS,
    ):
        b = tl.program_id(0)
        l = tl.arange(0, L)

        # ---- phases, and the write code for this token --------------------- #
        pos = tl.load(POS)
        base = (pos % L).to(tl.float32) * tl.load(OMEGA + l)
        theta = tl.load(THETA + l)
        phi = tl.load(KH + b * L + l) * theta + base
        psi = tl.load(KZ + b * L + l) * theta + base

        # ---- the current token's row, kept in REGISTERS ---------------------- #
        # NOT stored-then-read-back. Writing the ring and loading it again inside
        # one kernel is a read-after-write with no fence: nothing orders the
        # store before the load, and it silently returned the STALE row. It cost
        # 3.44 on an output of magnitude 3.44 -- the kernel read zeros -- and
        # only from a cold state, because at every later step the stale row
        # happened to be close enough to hide it. tl.debug_barrier() fixes it
        # too; keeping the row in registers removes the hazard instead of
        # ordering it, and skips a round trip.
        ptr = tl.load(PTR)
        w = tl.arange(0, L)
        is_cur = w == ptr

        # ---- read codes ----------------------------------------------------- #
        cq, sq = tl.cos(psi), tl.sin(psi)
        wr, wi = tl.load(WR + l), tl.load(WI + l)
        c1 = wr * cq + wi * sq
        c2 = wr * sq - wi * cq

        # ---- kappa over the window: one (W,L) reduction, W = L -------------- #
        off_r = b * L * L + w[:, None] * L + l[None, :]
        cwv = tl.where(is_cur[:, None], tl.cos(phi)[None, :],
                       tl.load(CW + off_r).to(tl.float32))
        swv = tl.where(is_cur[:, None], tl.sin(phi)[None, :],
                       tl.load(SW + off_r).to(tl.float32))
        k_re = (tl.sum(cwv * c1[None, :], 1) + tl.sum(swv * c2[None, :], 1)) / L
        k_im = (tl.sum(swv * c1[None, :], 1) - tl.sum(cwv * c2[None, :], 1)) / L

        # the ring itself is written once, at the end, for the NEXT call
        tl.store(CW + b * L * L + ptr * L + l, tl.cos(phi).to(CW.dtype.element_ty))
        tl.store(SW + b * L * L + ptr * L + l, tl.sin(phi).to(SW.dtype.element_ty))
        dvj = tl.arange(0, BLOCK_DV)

        # ---- contract with the value window, and the RMS over 2*DV ---------- #
        acc = tl.zeros([], dtype=tl.float32)
        for j0 in range(0, DV, BLOCK_DV):
            cols = j0 + dvj
            cmask = cols < DV
            vzc = tl.load(VZ + b * DV + cols, mask=cmask, other=0.0)
            e = tl.where(is_cur[:, None], vzc[None, :],
                         tl.load(EW + b * L * DV + w[:, None] * DV + cols[None, :],
                                 mask=cmask[None, :], other=0.0).to(tl.float32))
            tl.store(EW + b * L * DV + ptr * DV + cols,
                     vzc.to(EW.dtype.element_ty), mask=cmask)
            u0 = tl.sum(e * k_re[:, None], 0)
            u1 = tl.sum(e * k_im[:, None], 0)
            tl.store(OUT + b * 2 * DV + cols, u0, mask=cmask)
            tl.store(OUT + b * 2 * DV + DV + cols, u1, mask=cmask)
            acc += tl.sum(tl.where(cmask, u0 * u0 + u1 * u1, 0.0))

        scale = 1.0 / tl.sqrt(acc / (2 * DV) + EPS)
        for j0 in range(0, DV, BLOCK_DV):
            cols = j0 + dvj
            cmask = cols < DV
            o0 = tl.load(OUT + b * 2 * DV + cols, mask=cmask, other=0.0)
            o1 = tl.load(OUT + b * 2 * DV + DV + cols, mask=cmask, other=0.0)
            tl.store(OUT + b * 2 * DV + cols, o0 * scale, mask=cmask)
            tl.store(OUT + b * 2 * DV + DV + cols, o1 * scale, mask=cmask)


def short_supported(head) -> bool:
    return (HAVE_TRITON and head.G == 1 and head.wd == torch.float32
            and not (head.cfg.gdn_gate and head.cfg.gdn_gate_scope == "both"))


def short_step(kh, kz, vz, theta, omega, wr, wi, pos, ptr, cw, sw, ew,
               eps: float = 1e-6, block_dv: int = 64):
    """One launch for phase + ring write + kappa + read + RMS.

    cw/sw (B,L,L) and ew (B,L,dv) are mutated in place at `ptr`, exactly as the
    PyTorch ring does. Returns (B, 2*dv), already RMS-normalised.
    """
    B, L, _ = cw.shape
    DV = ew.shape[2]
    out = torch.empty((B, 2 * DV), device=cw.device, dtype=torch.float32)
    _short_step[(B,)](
        kh.contiguous(), kz.contiguous(), vz.contiguous(),
        theta.contiguous(), omega.contiguous(), wr.contiguous(), wi.contiguous(),
        pos, ptr, cw, sw, ew, out,
        L=L, DV=DV, BLOCK_DV=block_dv, EPS=eps,
    )
    return out
