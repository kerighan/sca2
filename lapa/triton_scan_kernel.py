"""Whole-sequence forward scan and two-part reverse chunks.

The chunk loop is sequential in the chunk index only. Given the codes, the
inverse and the values, every column of the value width is independent -- the
state read, the delta solve, the output and the state write all act column by
column -- so one program owns (batch, group, column block) and walks the whole
chunk loop by itself, keeping the state slice in L2 rather than in launches.

Reductions over the chunk length C are single `tl.dot` calls (the operands fit);
reductions over the packed mode axis 2M are tiled and stream the state.
"""

import torch
import triton
import triton.language as tl

from .triton_scan_tune import matrix_tuner, scan_tuner


@triton.jit
def _scale(x, dt):
    """Round an accumulator the way a torch matmul in `dt` would."""
    return x.to(dt).to(tl.float32)


@scan_tuner("fwd")
@triton.jit
def _fwd(
    QK, KK, FQ, K2, W, V, BE, ST, D1, DC, GT, OUT, EB, RB, HB, S0,
    B: tl.constexpr, K: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
    D: tl.constexpr, G: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr,
    TK: tl.constexpr, SAVE: tl.constexpr, DEV: tl.constexpr,
):
    b, g, jb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    WID: tl.constexpr = D // G
    dt = KK.dtype.element_ty
    jj = jb * BN + tl.arange(0, BN)
    jm = jj < WID
    j = g * WID + jj
    ci = tl.arange(0, BC)
    cm = ci < C
    tkr = tl.arange(0, TK)
    cj = cm[:, None] & jm[None, :]
    H: tl.constexpr = R // 2
    for n in range(K):
        # The chunk loop carries its dependency through the state buffer; the
        # barrier keeps the pipeliner from prefetching a chunk's state before
        # the previous chunk has written it.
        tl.debug_barrier()
        bk = b * K + n
        # The live state ping-pongs between two slots: only the rounded copy
        # the backward needs is kept per chunk, so nothing streams to DRAM here.
        rp = ST + (((n % 2) * B + b) * R + tkr[:, None]) * D + j[None, :]
        wp = ST + ((((n + 1) % 2) * B + b) * R + tkr[:, None]) * D + j[None, :]
        # ---- state read: r = (QK @ s0) / M, s0 = round(s * d1) --------------
        acc = tl.zeros((BC, BN), tl.float32)
        for m0 in range(0, R, TK):
            mm = m0 + tkr
            mk = mm < R
            q = tl.load(QK + (bk * C + ci[:, None]) * R + mm[None, :],
                        cm[:, None] & mk[None, :], 0.0)
            sm = mk[:, None] & jm[None, :]
            s = tl.load(rp + m0 * D, sm, 0.0)
            s0 = (s * tl.load(D1 + mm, mk, 0.0)[:, None]).to(dt)
            if SAVE:
                # The code gradients read this back instead of the fp32 state.
                tl.store(S0 + ((n * B + b) * R + mm[:, None]) * D + j[None, :], s0, sm)
            acc = tl.dot(q, s0, acc, input_precision="tf32x3")
        r = _scale(acc, dt) / (R // 2)
        # ---- delta rule: e = W (v - beta r) --------------------------------
        v = tl.load(V + (bk * C + ci[:, None]) * D + j[None, :], cj, 0.0)
        h = v - tl.load(BE + bk * C + ci, cm, 0.0)[:, None] * r
        w = tl.load(W + (bk * C + ci[:, None]) * C + ci[None, :],
                    cm[:, None] & cm[None, :], 0.0)
        e = tl.dot(w, h, input_precision="tf32x3")
        eb = e.to(dt)
        # Save the solve intermediates now: keeping r/h alive through the
        # output/state loop caused hundreds of register spills at D=384.
        if SAVE:
            tl.store(EB + (bk * C + ci[:, None]) * D + j[None, :], eb, cj)
            tl.store(RB + (bk * C + ci[:, None]) * D + j[None, :], r, cj)
            tl.store(HB + (bk * C + ci[:, None]) * D + j[None, :], h, cj)
        # ---- output/state update -----------------------------------------
        # Delay K2 @ e until after the mode loop, so its two accumulators do
        # not compete for registers with the live state and z accumulators.
        z_re = tl.zeros((BC, BN), tl.float32)
        z_im = tl.zeros((BC, BN), tl.float32)
        fp = FQ + ((bk * G + g) * C + ci[:, None]) * R
        # Compact read codes: FQ holds [c1 | c2] once. The imaginary rows are
        # [-c2 | c1], so both halves of the output come from the same two tiles.
        # The same pass does the state write: one load of s serves all three.
        for m0 in range(0, H, TK):
            mm = m0 + tkr
            mk = mm < H
            sm = mk[:, None] & jm[None, :]
            msk = cm[:, None] & mk[None, :]
            slo = tl.load(rp + m0 * D, sm, 0.0)
            shi = tl.load(rp + (m0 + H) * D, sm, 0.0)
            s0lo = (slo * tl.load(D1 + mm, mk, 0.0)[:, None]).to(dt)
            s0hi = (shi * tl.load(D1 + H + mm, mk, 0.0)[:, None]).to(dt)
            c1 = tl.load(fp + mm[None, :], msk, 0.0)
            c2 = tl.load(fp + H + mm[None, :], msk, 0.0)
            z_re = tl.dot(c1, s0lo, z_re, input_precision="tf32x3")
            z_re = tl.dot(c2, s0hi, z_re, input_precision="tf32x3")
            z_im = tl.dot(c1, s0hi, z_im, input_precision="tf32x3")
            z_im = tl.dot(-c2, s0lo, z_im, input_precision="tf32x3")
            kkp = KK + (bk * C + ci[:, None]) * R
            wlo = tl.dot(tl.trans(tl.load(kkp + mm[None, :], msk, 0.0)), eb,
                         input_precision="tf32x3")
            whi = tl.dot(tl.trans(tl.load(kkp + H + mm[None, :], msk, 0.0)), eb,
                         input_precision="tf32x3")
            tl.store(wp + m0 * D,
                     slo * tl.load(DC + mm, mk, 0.0)[:, None]
                     + _scale(wlo, dt) * tl.load(GT + mm, mk, 0.0)[:, None], sm)
            tl.store(wp + (m0 + H) * D,
                     shi * tl.load(DC + H + mm, mk, 0.0)[:, None]
                     + _scale(whi, dt) * tl.load(GT + H + mm, mk, 0.0)[:, None], sm)
        kp = K2 + ((bk * G + g) * 2 * C + ci[:, None]) * C + ci[None, :]
        cc = cm[:, None] & cm[None, :]
        op = OUT + (bk * C + ci[:, None]) * 2 * D + j[None, :]
        o_re = _scale(tl.dot(tl.load(kp, cc, 0.0), eb, input_precision="tf32x3"), dt)
        tl.store(op, _scale(o_re + _scale(z_re, dt), dt) / (R // 2), cj)
        o_im = _scale(tl.dot(tl.load(kp + C * C, cc, 0.0), eb, input_precision="tf32x3"), dt)
        tl.store(op + D, _scale(o_im + _scale(z_im, dt), dt) / (R // 2), cj)


@scan_tuner("bwd")
@triton.jit(do_not_specialize=["n"])
def _bwd(
    KK, K2, W, BE, GT, RB, DO, DSR, DE, DRM, DSG, DV, PB, n,
    B: tl.constexpr, K: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
    D: tl.constexpr, G: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr,
    TK: tl.constexpr, NC: tl.constexpr, DEV: tl.constexpr,
):
    """One reverse chunk: the state-write adjoint, the delta solve and dv/dbeta.

    Everything here needs the whole mode axis reduced, so the program owns a
    column block and walks all 2M modes. The mode-parallel half of the chunk
    lives in _bwd2.
    """
    b, g, jb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    WID: tl.constexpr = D // G
    H: tl.constexpr = R // 2
    dt = KK.dtype.element_ty
    jj = jb * BN + tl.arange(0, BN)
    jm = jj < WID
    j = g * WID + jj
    ci = tl.arange(0, BC)
    cm = ci < C
    tkr = tl.arange(0, TK)
    cj = cm[:, None] & jm[None, :]
    cc = cm[:, None] & cm[None, :]
    # Each tile owns BN/16 scratch slots, zeroing its unused slots. This
    # keeps scratch layouts independent of autotuning without reset passes.
    slots = tl.arange(0, BN // 16)
    part_col = jb * (BN // 16) + slots
    valid_part = part_col < NC
    pc = g * NC + part_col
    bk = b * K + n
    dp = DSR + (b * R + tkr[:, None]) * D + j[None, :]
    op = DO + (b * K * C + n * C + ci[:, None]) * 2 * D + j[None, :]
    dor = tl.load(op, cj, 0.0)
    doi = tl.load(op + D, cj, 0.0)
    dek = tl.zeros((BC, BN), tl.float32)
    for m0 in range(0, R, TK):
        mm = m0 + tkr
        mk = mm < R
        sm = mk[:, None] & jm[None, :]
        msk = cm[:, None] & mk[None, :]
        ds = tl.load(dp + m0 * D, sm, 0.0)
        # The gT gradient is sum_j ds . (Kk^T e), which reassociates into
        # sum_c Kk . (e ds^T): the code-gradient kernel already forms that
        # product, so the write is never recomputed here.
        tl.store(DSG + (bk * R + mm[:, None]) * D + j[None, :], ds.to(dt), sm)
        dsg = (ds * tl.load(GT + mm, mk, 0.0)[:, None]).to(dt)
        kk = tl.load(KK + (bk * C + ci[:, None]) * R + mm[None, :], msk, 0.0)
        dek = tl.dot(kk, dsg, dek, input_precision="tf32x3")
    kp = K2 + ((bk * G + g) * 2 * C + ci[:, None]) * C + ci[None, :]
    da = tl.dot(tl.trans(tl.load(kp, cc, 0.0)), dor, input_precision="tf32x3")
    da = tl.dot(tl.trans(tl.load(kp + C * C, cc, 0.0)), doi, da,
                input_precision="tf32x3")
    de = _scale(_scale(da, dt) + _scale(dek, dt), dt)
    w = tl.load(W + (bk * C + ci[:, None]) * C + ci[None, :], cc, 0.0)
    dh = tl.dot(tl.trans(w), de, input_precision="tf32x3")
    r = tl.load(RB + (bk * C + ci[:, None]) * D + j[None, :], cj, 0.0)
    be = tl.load(BE + bk * C + ci, cm, 0.0)
    tl.store(PB + pc[:, None] * B * K * C + bk * C + ci[None, :],
             tl.where(slots[:, None] == 0, -tl.sum(dh * r, 1)[None, :], 0.0),
             valid_part[:, None] & cm[None, :])
    tl.store(DV + (bk * C + ci[:, None]) * D + j[None, :], dh, cj)
    tl.store(DE + (bk * C + ci[:, None]) * D + j[None, :], de.to(dt), cj)
    tl.store(DRM + (bk * C + ci[:, None]) * D + j[None, :],
             (-be[:, None] * dh / H).to(dt), cj)


@scan_tuner("bwd2")
@triton.jit(do_not_specialize=["n"])
def _bwd2(QK, FQ, DO, DRM, ST, DSR, D1, DC, PR, PC, n,
          B: tl.constexpr, K: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr,
          TK: tl.constexpr, NC: tl.constexpr, DEV: tl.constexpr):
    """The mode-parallel half: the read adjoint and the state-gradient step.

    ds0 = Fq^T do + Qk^T dr is a per-mode outer product and the state gradient
    recursion is elementwise, so this splits over the mode axis as well and
    runs at a far larger grid than the reduction half.
    """
    WID: tl.constexpr = D // G
    H: tl.constexpr = R // 2
    dt = QK.dtype.element_ty
    b = tl.program_id(0) // G
    g = tl.program_id(0) % G
    jj = tl.program_id(1) * BN + tl.arange(0, BN)
    jm = jj < WID
    j = g * WID + jj
    mm = tl.program_id(2) * TK + tl.arange(0, TK)
    mk = mm < H
    ci = tl.arange(0, BC)
    cm = ci < C
    cj = cm[:, None] & jm[None, :]
    msk = cm[:, None] & mk[None, :]
    sm = mk[:, None] & jm[None, :]
    bk = b * K + n
    op = DO + (b * K * C + n * C + ci[:, None]) * 2 * D + j[None, :]
    dor = tl.load(op, cj, 0.0)
    doi = tl.load(op + D, cj, 0.0)
    drm = tl.load(DRM + (bk * C + ci[:, None]) * D + j[None, :], cj, 0.0)
    fp = FQ + ((bk * G + g) * C + ci[:, None]) * R
    c1 = tl.load(fp + mm[None, :], msk, 0.0)
    c2 = tl.load(fp + H + mm[None, :], msk, 0.0)
    flo = tl.dot(tl.trans(c1), dor, input_precision="tf32x3")
    flo = tl.dot(tl.trans(-c2), doi, flo, input_precision="tf32x3")
    fhi = tl.dot(tl.trans(c2), dor, input_precision="tf32x3")
    fhi = tl.dot(tl.trans(c1), doi, fhi, input_precision="tf32x3")
    qp = QK + (bk * C + ci[:, None]) * R
    qlo = tl.dot(tl.trans(tl.load(qp + mm[None, :], msk, 0.0)), drm,
                 input_precision="tf32x3")
    qhi = tl.dot(tl.trans(tl.load(qp + H + mm[None, :], msk, 0.0)), drm,
                 input_precision="tf32x3")
    lo = _scale(_scale(flo, dt) + _scale(qlo, dt), dt)
    hi = _scale(_scale(fhi, dt) + _scale(qhi, dt), dt)
    sp = ST + ((n * B + b) * R + mm[:, None]) * D + j[None, :]
    dp = DSR + (b * R + mm[:, None]) * D + j[None, :]
    dlo = tl.load(dp, sm, 0.0)
    dhi = tl.load(dp + H * D, sm, 0.0)
    dlo1 = tl.load(D1 + mm, mk, 1.0)[:, None]
    dhi1 = tl.load(D1 + H + mm, mk, 1.0)[:, None]
    slo = tl.load(sp, sm, 0.0).to(tl.float32) / dlo1
    shi = tl.load(sp + H * D, sm, 0.0).to(tl.float32) / dhi1
    slots = tl.arange(0, BN // 16)
    part_col = tl.program_id(1) * (BN // 16) + slots
    pd = PR + (((b * G + g) * NC + part_col[:, None]) * K + n) * R + mm[None, :]
    pm = (part_col[:, None] < NC) & mk[None, :]
    tl.store(pd, tl.where(slots[:, None] == 0, tl.sum(lo * slo, 1)[None, :], 0.0), pm)
    tl.store(pd + H, tl.where(slots[:, None] == 0, tl.sum(hi * shi, 1)[None, :], 0.0), pm)
    # dDC uses the same undecayed state and incoming adjoint as dD1.
    # Compute it here instead of rereading the entire saved state in _bwd.
    pc = PC + (((b * G + g) * NC + part_col[:, None]) * K + n) * R + mm[None, :]
    tl.store(pc, tl.where(slots[:, None] == 0, tl.sum(dlo * slo, 1)[None, :], 0.0), pm)
    tl.store(pc + H, tl.where(slots[:, None] == 0, tl.sum(dhi * shi, 1)[None, :], 0.0), pm)
    tl.store(dp, dlo * tl.load(DC + mm, mk, 0.0)[:, None] + lo * dlo1, sm)
    tl.store(dp + H * D, dhi * tl.load(DC + H + mm, mk, 0.0)[:, None] + hi * dhi1, sm)


@matrix_tuner("grad")
@triton.jit
def _grad(A, BB, ST, D1, OUT, KK, GT, PG,
          B: tl.constexpr, K: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, KIND: tl.constexpr,
          TM: tl.constexpr, TN: tl.constexpr, TW: tl.constexpr, DEV: tl.constexpr):
    """Matrix gradients contracted over the value width: OUT = A B^T.

    KIND 0 dW = de h^T, 1 dQk = dr s0^T, 2 dKk = gT . (e ds^T) with the gT
    gradient reduced alongside, 4 dK2 = do e^T per group, reading the packed
    [Re | Im] output gradient in place.
    """
    ROWS: tl.constexpr = 2 * C if KIND >= 3 else C
    COLS: tl.constexpr = C if (KIND == 0 or KIND == 4) else R
    WID: tl.constexpr = (D // G) if KIND >= 3 else D
    dt = OUT.dtype.element_ty
    pid = tl.program_id(0)
    g = pid % G if KIND >= 3 else 0
    bk = pid // G if KIND >= 3 else pid
    b = bk // K
    n = bk % K
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    jn = tl.program_id(2) * TN + tl.arange(0, TN)
    im = i < ROWS
    jm = jn < COLS
    hi = i >= C
    ir = tl.where(hi, i - C, i)
    acc = tl.zeros((TM, TN), tl.float32)
    for w0 in range(0, WID, TW):
        ww = w0 + tl.arange(0, TW)
        wm = ww < WID
        col = g * WID + ww
        am = im[:, None] & wm[None, :]
        bm = jm[:, None] & wm[None, :]
        if KIND >= 3:
            a = tl.load(A + (b * K * C + n * C + ir[:, None]) * 2 * D
                        + col[None, :] + tl.where(hi, D, 0)[:, None], am, 0.0)
        elif KIND == 0:
            a = tl.load(A + (bk * C + i[:, None]) * D + col[None, :], am, 0.0)
            a = a.to(tl.float32)
        else:
            a = tl.load(A + (bk * C + i[:, None]) * D + col[None, :], am, 0.0)
        if KIND == 1:
            bv = tl.load(ST + ((n * B + b) * R + jn[:, None]) * D + col[None, :], bm, 0.0)
        elif KIND == 2:
            bv = tl.load(BB + (bk * R + jn[:, None]) * D + col[None, :], bm, 0.0)
        else:
            bv = tl.load(BB + (bk * C + jn[:, None]) * D + col[None, :], bm, 0.0)
        acc = tl.dot(a, tl.trans(bv), acc, input_precision="tf32x3")
    out = acc if KIND == 0 else _scale(acc, dt)
    if KIND == 2:
        # dKk = gT . (e ds^T); the same product reduced along C gives d gT.
        km = tl.load(KK + (bk * C + i[:, None]) * R + jn[None, :],
                     im[:, None] & jm[None, :], 0.0)
        slots = tl.arange(0, TM // 16)
        row = tl.program_id(1) * (TM // 16) + slots
        tl.store(PG + (pid * tl.cdiv(C, 16) + row[:, None]) * R + jn[None, :],
                 tl.where(slots[:, None] == 0, tl.sum(out * km.to(tl.float32), 0)[None, :], 0.0),
                 (row[:, None] < tl.cdiv(C, 16)) & jm[None, :])
        out = _scale(out * tl.load(GT + jn, jm, 0.0)[None, :], dt)
    if KIND >= 3:
        op = OUT + ((bk * G + g) * 2 * C + i[:, None]) * COLS + jn[None, :]
    else:
        op = OUT + (bk * C + i[:, None]) * COLS + jn[None, :]
    tl.store(op, out, im[:, None] & jm[None, :])

@matrix_tuner("dfc")
@triton.jit
def _dfc(DO, ST, D1, OUT,
         B: tl.constexpr, K: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
         D: tl.constexpr, G: tl.constexpr,
         TM: tl.constexpr, TN: tl.constexpr, TW: tl.constexpr, DEV: tl.constexpr):
    """Read-code gradient in the compact layout: dc1 and dc2 from one pass."""
    H: tl.constexpr = R // 2
    WID: tl.constexpr = D // G
    dt = OUT.dtype.element_ty
    pid = tl.program_id(0)
    g = pid % G
    bk = pid // G
    b = bk // K
    n = bk % K
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    jn = tl.program_id(2) * TN + tl.arange(0, TN)
    im = i < C
    jm = jn < H
    a1 = tl.zeros((TM, TN), tl.float32)
    a2 = tl.zeros((TM, TN), tl.float32)
    for w0 in range(0, WID, TW):
        ww = w0 + tl.arange(0, TW)
        wm = ww < WID
        col = g * WID + ww
        op = DO + (b * K * C + n * C + i[:, None]) * 2 * D + col[None, :]
        am = im[:, None] & wm[None, :]
        bm = jm[:, None] & wm[None, :]
        dor = tl.load(op, am, 0.0)
        doi = tl.load(op + D, am, 0.0)
        sp = ST + ((n * B + b) * R + jn[:, None]) * D + col[None, :]
        slo = tl.load(sp, bm, 0.0)
        shi = tl.load(sp + H * D, bm, 0.0)
        a1 = tl.dot(dor, tl.trans(slo), a1, input_precision="tf32x3")
        a1 = tl.dot(doi, tl.trans(shi), a1, input_precision="tf32x3")
        a2 = tl.dot(dor, tl.trans(shi), a2, input_precision="tf32x3")
        a2 = tl.dot(-doi, tl.trans(slo), a2, input_precision="tf32x3")
    keep = im[:, None] & jm[None, :]
    ptr = OUT + (pid * C + i[:, None]) * R + jn[None, :]
    tl.store(ptr, _scale(a1, dt), keep)
    tl.store(ptr + H, _scale(a2, dt), keep)


@matrix_tuner("k2")
@triton.jit
def _k2(FC, KK, DK2, OUT,
        K: tl.constexpr, C: tl.constexpr, R: tl.constexpr, G: tl.constexpr,
        MODE: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
        B: tl.constexpr, DEV: tl.constexpr):
    """Causal intra-chunk kernel K2 = Fq Kk^T and its two adjoints.

    MODE 0 builds K2 from the compact codes into OUT, 1 accumulates dFc and 2
    dKk, both into OUT. The causal mask bounds the reduction, so the masked
    half of the triangle costs nothing. Exactly one buffer is written, and no
    tensor is passed in two slots: Inductor clones what it believes a user
    kernel mutates, and an aliased pair makes it clone the wrong one.
    """
    H: tl.constexpr = R // 2
    dt = FC.dtype.element_ty
    pid = tl.program_id(0)
    g = pid % G if MODE != 2 else 0
    bk = pid // G if MODE != 2 else pid
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    jn = tl.program_id(2) * TN + tl.arange(0, TN)
    im = i < C
    a1 = tl.zeros((TM, TN), tl.float32)
    a2 = tl.zeros((TM, TN), tl.float32)
    if MODE == 0:
        jm = jn < C
        keep = im[:, None] & jm[None, :]
        op = OUT + (pid * 2 * C + i[:, None]) * C + jn[None, :]
        if tl.program_id(1) * TM + TM - 1 < tl.program_id(2) * TN:
            tl.store(op, tl.zeros((TM, TN), dt), keep)
            tl.store(op + C * C, tl.zeros((TM, TN), dt), keep)
            return
        fp = FC + (pid * C + i[:, None]) * R
        kp = KK + (bk * C + jn[:, None]) * R
        for m0 in range(0, H, TK):
            mm = m0 + tl.arange(0, TK)
            mk = mm < H
            c1 = tl.load(fp + mm[None, :], im[:, None] & mk[None, :], 0.0)
            c2 = tl.load(fp + H + mm[None, :], im[:, None] & mk[None, :], 0.0)
            ka = tl.trans(tl.load(kp + mm[None, :], jm[:, None] & mk[None, :], 0.0))
            kb = tl.trans(tl.load(kp + H + mm[None, :], jm[:, None] & mk[None, :], 0.0))
            a1 = tl.dot(c1, ka, a1, input_precision="tf32x3")
            a1 = tl.dot(c2, kb, a1, input_precision="tf32x3")
            a2 = tl.dot(c1, kb, a2, input_precision="tf32x3")
            a2 = tl.dot(-c2, ka, a2, input_precision="tf32x3")
        low = i[:, None] >= jn[None, :]
        tl.store(op, tl.where(low, _scale(a1, dt), 0.0).to(dt), keep)
        tl.store(op + C * C, tl.where(low, _scale(a2, dt), 0.0).to(dt), keep)
        return
    jm = jn < H
    keep = im[:, None] & jm[None, :]
    if MODE == 1:
        # dc1 = dK2r A + dK2i B, dc2 = dK2r B - dK2i A, over the causal columns.
        dp = DK2 + (pid * 2 * C + i[:, None]) * C
        kp = KK + bk * C * R
        for r0 in range(0, tl.program_id(1) * TM + TM, TK):
            rr = r0 + tl.arange(0, TK)
            rm = (rr[None, :] < C) & (i[:, None] >= rr[None, :]) & im[:, None]
            d1 = tl.load(dp + rr[None, :], rm, 0.0)
            d2 = tl.load(dp + C * C + rr[None, :], rm, 0.0)
            km = (rr[:, None] < C) & jm[None, :]
            ka = tl.load(kp + rr[:, None] * R + jn[None, :], km, 0.0)
            kb = tl.load(kp + rr[:, None] * R + H + jn[None, :], km, 0.0)
            a1 = tl.dot(d1, ka, a1, input_precision="tf32x3")
            a1 = tl.dot(d2, kb, a1, input_precision="tf32x3")
            a2 = tl.dot(d1, kb, a2, input_precision="tf32x3")
            a2 = tl.dot(-d2, ka, a2, input_precision="tf32x3")
        ptr = OUT + (pid * C + i[:, None]) * R + jn[None, :]
        tl.store(ptr, _scale(a1, dt) + tl.load(ptr, keep, 0.0).to(tl.float32), keep)
        tl.store(ptr + H, _scale(a2, dt) + tl.load(ptr + H, keep, 0.0).to(tl.float32),
                 keep)
        return
    # MODE 2: dA = dK2r^T c1 - dK2i^T c2, dB = dK2r^T c2 + dK2i^T c1, summed
    # over the read groups; rows index the write position, so i <= r.
    for gg in tl.static_range(G):
        dp = DK2 + ((bk * G + gg) * 2 * C) * C + i[None, :]
        fp = FC + ((bk * G + gg) * C) * R + jn[None, :]
        for r0 in range(tl.program_id(1) * TM, C, TK):
            rr = r0 + tl.arange(0, TK)
            rm = (rr[:, None] < C) & (rr[:, None] >= i[None, :]) & im[None, :]
            d1 = tl.trans(tl.load(dp + rr[:, None] * C, rm, 0.0))
            d2 = tl.trans(tl.load(dp + C * C + rr[:, None] * C, rm, 0.0))
            fm = (rr[:, None] < C) & jm[None, :]
            c1 = tl.load(fp + rr[:, None] * R, fm, 0.0)
            c2 = tl.load(fp + H + rr[:, None] * R, fm, 0.0)
            a1 = tl.dot(d1, c1, a1, input_precision="tf32x3")
            a1 = tl.dot(-d2, c2, a1, input_precision="tf32x3")
            a2 = tl.dot(d1, c2, a2, input_precision="tf32x3")
            a2 = tl.dot(d2, c1, a2, input_precision="tf32x3")
    ptr = OUT + (bk * C + i[:, None]) * R + jn[None, :]
    tl.store(ptr, _scale(a1, dt) + tl.load(ptr, keep, 0.0).to(tl.float32), keep)
    tl.store(ptr + H, _scale(a2, dt) + tl.load(ptr + H, keep, 0.0).to(tl.float32), keep)


@triton.jit
def _scaled(X, OUT, N, SCALE, BLK: tl.constexpr):
    """out = (x * scale) in OUT's dtype.

    Inductor realises a pointwise result once per user-defined Triton kernel
    that reads it, so the output gradient -- read by every chunk of the reverse
    scan -- is written to a real buffer here instead of being fused in.
    """
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < N
    tl.store(OUT + i, (tl.load(X + i, m, 0.0) * SCALE).to(OUT.dtype.element_ty), m)


# Separate single-config entry points for tiny shapes avoid expensive tuning.
# Keep restore_value intact under Inductor; see triton_scan_tune.py.
_fwd_small = scan_tuner("fwd", small=True)(_fwd.fn)
_bwd_small = scan_tuner("bwd", small=True)(_bwd.fn)
_bwd2_small = scan_tuner("bwd2", small=True)(_bwd2.fn)
_grad_small = matrix_tuner("grad", small=True)(_grad.fn)
_grad0 = matrix_tuner("grad0", fp32=True)(_grad.fn)
_dfc_small = matrix_tuner("dfc", small=True)(_dfc.fn)
_k2_small = matrix_tuner("k2", small=True)(_k2.fn)

_fwd_fp32 = scan_tuner("fwd", fp32=True)(_fwd.fn)
_bwd_fp32 = scan_tuner("bwd", fp32=True)(_bwd.fn)
_bwd2_fp32 = scan_tuner("bwd2", fp32=True)(_bwd2.fn)
_grad_fp32 = matrix_tuner("grad", fp32=True)(_grad.fn)
_dfc_fp32 = matrix_tuner("dfc", fp32=True)(_dfc.fn)
_k2_fp32 = matrix_tuner("k2", fp32=True)(_k2.fn)
