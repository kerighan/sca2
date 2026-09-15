"""Tiled GEMMs with fused state-loop epilogues and an explicit reverse scan."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _read(
    Q,
    S,
    STATES,
    D1,
    V,
    BETA,
    H,
    READ,
    S0,
    K: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    SAVE: tl.constexpr,
    TM: tl.constexpr = 32,
    TN: tl.constexpr = 32,
    TK: tl.constexpr = 32,
):
    b = tl.program_id(0)
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    j = tl.program_id(2) * TN + tl.arange(0, TN)
    rr = tl.arange(0, TK)
    acc = tl.full((TM, TN), 0, tl.float32)
    for base in range(tl.cdiv(R, TK)):
        r = base * TK + rr
        q = tl.load(
            Q + ((b * K + N) * C + i[:, None]) * R + r[None, :],
            (i[:, None] < C) & (r[None, :] < R),
            0,
        )
        if N == 0:
            sp = S + (b * R + r[:, None]) * D + j[None, :]
        else:
            sp = STATES + ((b * K + N - 1) * R + r[:, None]) * D + j[None, :]
        s = tl.load(sp, (r[:, None] < R) & (j[None, :] < D), 0)
        d = tl.load(D1 + r, r < R, 0)
        s = (s * d[:, None]).to(q.dtype)
        if tl.program_id(1) == 0:
            tl.store(
                S0 + ((b * K + N) * R + r[:, None]) * D + j[None, :],
                s,
                (r[:, None] < R) & (j[None, :] < D),
            )
        acc += tl.dot(q, s, input_precision="tf32x3")
    read = acc.to(Q.dtype.element_ty).to(tl.float32) / (R // 2)
    off = ((b * K + N) * C + i[:, None]) * D + j[None, :]
    v = tl.load(V + off, (i[:, None] < C) & (j[None, :] < D), 0)
    beta = tl.load(BETA + (b * K + N) * C + i, i < C, 0)
    tl.store(H + off, v - beta[:, None] * read, (i[:, None] < C) & (j[None, :] < D))
    if SAVE:
        tl.store(READ + off, read, (i[:, None] < C) & (j[None, :] < D))


@tr.jit
def _delta(
    W,
    H,
    E,
    K: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    REVERSE: tl.constexpr = False,
    TM: tl.constexpr = 32,
    TN: tl.constexpr = 32,
    TK: tl.constexpr = 32,
):
    b = tl.program_id(0)
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    j = tl.program_id(2) * TN + tl.arange(0, TN)
    rr = tl.arange(0, TK)
    acc = tl.full((TM, TN), 0, tl.float32)
    for base in range(tl.cdiv(C, TK)):
        r = base * TK + rr
        if REVERSE:
            off = ((b * K + N) * C + r[None, :]) * C + i[:, None]
        else:
            off = ((b * K + N) * C + i[:, None]) * C + r[None, :]
        w = tl.load(W + off, (i[:, None] < C) & (r[None, :] < C), 0)
        h = tl.load(
            H + ((b * K + N) * C + r[:, None]) * D + j[None, :],
            (r[:, None] < C) & (j[None, :] < D),
            0,
        )
        acc += tl.dot(w, h, input_precision="tf32x3")
    tl.store(
        E + ((b * K + N) * C + i[:, None]) * D + j[None, :],
        acc,
        (i[:, None] < C) & (j[None, :] < D),
    )


@tr.jit
def _output(
    F,
    K2,
    E,
    S0,
    O,
    F0: tl.constexpr,
    F1: tl.constexpr,
    F2: tl.constexpr,
    F3: tl.constexpr,
    F4: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    TM: tl.constexpr = 32,
    TN: tl.constexpr = 32,
    TK: tl.constexpr = 32,
):
    bg = tl.program_id(0)
    b = bg // G
    g = bg % G
    width = D // G
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    jj = tl.program_id(2) * TN + tl.arange(0, TN)
    j = g * width + jj
    rr = tl.arange(0, TK)
    a = tl.full((TM, TN), 0, tl.float32)
    z = tl.full((TM, TN), 0, tl.float32)
    for base in range(tl.cdiv(C, TK)):
        r = base * TK + rr
        k = tl.load(
            K2 + (((b * K + N) * G + g) * 2 * C + i[:, None]) * C + r[None, :],
            (i[:, None] < 2 * C) & (r[None, :] < C),
            0,
        )
        e = tl.load(
            E + ((b * K + N) * C + r[:, None]) * D + j[None, :],
            (r[:, None] < C) & (jj[None, :] < width),
            0,
        )
        a += tl.dot(k, e, input_precision="tf32x3")
    for base in range(tl.cdiv(R, TK)):
        r = base * TK + rr
        f = tl.load(
            F + b * F0 + N * F1 + g * F2 + i[:, None] * F3 + r[None, :] * F4,
            (i[:, None] < 2 * C) & (r[None, :] < R),
            0,
        )
        sp = S0 + ((b * K + N) * R + r[:, None]) * D + j[None, :]
        s = tl.load(sp, (r[:, None] < R) & (jj[None, :] < width), 0)
        z += tl.dot(f, s, input_precision="tf32x3")
    out = (
        a.to(E.dtype.element_ty).to(tl.float32)
        + z.to(E.dtype.element_ty).to(tl.float32)
    ).to(E.dtype.element_ty)
    tl.store(
        O + ((b * K + N) * 2 * C + i[:, None]) * D + j[None, :],
        out,
        (i[:, None] < 2 * C) & (jj[None, :] < width),
    )


@tr.jit
def _write(
    KK,
    E,
    S,
    STATES,
    DC,
    GT,
    WRITE,
    K: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    SAVE: tl.constexpr,
    TM: tl.constexpr = 32,
    TN: tl.constexpr = 32,
    TK: tl.constexpr = 32,
):
    b = tl.program_id(0)
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    j = tl.program_id(2) * TN + tl.arange(0, TN)
    rr = tl.arange(0, TK)
    acc = tl.full((TM, TN), 0, tl.float32)
    for base in range(tl.cdiv(C, TK)):
        r = base * TK + rr
        k = tl.load(
            KK + ((b * K + N) * C + r[:, None]) * R + i[None, :],
            (r[:, None] < C) & (i[None, :] < R),
            0,
        )
        e = tl.load(
            E + ((b * K + N) * C + r[:, None]) * D + j[None, :],
            (r[:, None] < C) & (j[None, :] < D),
            0,
        )
        acc += tl.dot(tl.trans(k), e, input_precision="tf32x3")
    if N == 0:
        sp = S + (b * R + i[:, None]) * D + j[None, :]
    else:
        sp = STATES + ((b * K + N - 1) * R + i[:, None]) * D + j[None, :]
    s = tl.load(sp, (i[:, None] < R) & (j[None, :] < D), 0)
    dc = tl.load(DC + i, i < R, 0)
    gt = tl.load(GT + i, i < R, 0)
    write = acc.to(E.dtype.element_ty).to(tl.float32)
    off = ((b * K + N) * R + i[:, None]) * D + j[None, :]
    tl.store(
        STATES + off,
        s * dc[:, None] + write * gt[:, None],
        (i[:, None] < R) & (j[None, :] < D),
    )
    if SAVE:
        tl.store(WRITE + off, write, (i[:, None] < R) & (j[None, :] < D))


@tr.jit
def _de(
    KK,
    K2,
    DO,
    DS,
    DSFINAL,
    GT,
    DE,
    K: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    TM: tl.constexpr = 32,
    TN: tl.constexpr = 32,
    TK: tl.constexpr = 32,
):
    bg = tl.program_id(0)
    b = bg // G
    g = bg % G
    width = D // G
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    jj = tl.program_id(2) * TN + tl.arange(0, TN)
    j = g * width + jj
    rr = tl.arange(0, TK)
    a = tl.full((TM, TN), 0, tl.float32)
    z = tl.full((TM, TN), 0, tl.float32)
    for base in range(tl.cdiv(2 * C, TK)):
        r = base * TK + rr
        k = tl.load(
            K2 + (((b * K + N) * G + g) * 2 * C + r[:, None]) * C + i[None, :],
            (r[:, None] < 2 * C) & (i[None, :] < C),
            0,
        )
        do = tl.load(
            DO + ((b * K + N) * 2 * C + r[:, None]) * D + j[None, :],
            (r[:, None] < 2 * C) & (jj[None, :] < width),
            0,
        ).to(k.dtype)
        a += tl.dot(tl.trans(k), do, input_precision="tf32x3")
    for base in range(tl.cdiv(R, TK)):
        r = base * TK + rr
        k = tl.load(
            KK + ((b * K + N) * C + i[:, None]) * R + r[None, :],
            (i[:, None] < C) & (r[None, :] < R),
            0,
        )
        if N == K - 1:
            dp = DSFINAL + (b * R + r[:, None]) * D + j[None, :]
        else:
            dp = DS + ((b * K + N + 1) * R + r[:, None]) * D + j[None, :]
        ds = tl.load(dp, (r[:, None] < R) & (jj[None, :] < width), 0)
        gt = tl.load(GT + r, r < R, 0)
        z += tl.dot(k, (ds * gt[:, None]).to(k.dtype), input_precision="tf32x3")
    out = (
        (
            a.to(KK.dtype.element_ty).to(tl.float32)
            + z.to(KK.dtype.element_ty).to(tl.float32)
        )
        .to(KK.dtype.element_ty)
        .to(tl.float32)
    )
    tl.store(
        DE + ((b * K + N) * C + i[:, None]) * D + j[None, :],
        out,
        (i[:, None] < C) & (jj[None, :] < width),
    )


@tr.jit
def _ds(
    F,
    Q,
    DO,
    DH,
    BETA,
    DS,
    DSFINAL,
    D1,
    DC,
    S,
    STATES,
    WRITE,
    PART,
    P: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    F2: tl.constexpr,
    F3: tl.constexpr,
    F4: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    TM: tl.constexpr = 32,
    TN: tl.constexpr = 32,
    TK: tl.constexpr = 32,
):
    bg = tl.program_id(0)
    b = bg // G
    g = bg % G
    width = D // G
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    jj = tl.program_id(2) * TN + tl.arange(0, TN)
    j = g * width + jj
    rr = tl.arange(0, TK)
    a = tl.full((TM, TN), 0, tl.float32)
    z = tl.full((TM, TN), 0, tl.float32)
    for base in range(tl.cdiv(2 * C, TK)):
        r = base * TK + rr
        f = tl.load(
            F + b * F0 + N * F1 + g * F2 + r[:, None] * F3 + i[None, :] * F4,
            (r[:, None] < 2 * C) & (i[None, :] < R),
            0,
        )
        do = tl.load(
            DO + ((b * K + N) * 2 * C + r[:, None]) * D + j[None, :],
            (r[:, None] < 2 * C) & (jj[None, :] < width),
            0,
        ).to(f.dtype)
        a += tl.dot(tl.trans(f), do, input_precision="tf32x3")
    for base in range(tl.cdiv(C, TK)):
        r = base * TK + rr
        q = tl.load(
            Q + ((b * K + N) * C + r[:, None]) * R + i[None, :],
            (r[:, None] < C) & (i[None, :] < R),
            0,
        )
        dh = tl.load(
            DH + ((b * K + N) * C + r[:, None]) * D + j[None, :],
            (r[:, None] < C) & (jj[None, :] < width),
            0,
        )
        beta = tl.load(BETA + (b * K + N) * C + r, r < C, 0)
        dr = (-dh * beta[:, None] / (R // 2)).to(q.dtype)
        z += tl.dot(tl.trans(q), dr, input_precision="tf32x3")
    ds0 = (
        (
            a.to(Q.dtype.element_ty).to(tl.float32)
            + z.to(Q.dtype.element_ty).to(tl.float32)
        )
        .to(Q.dtype.element_ty)
        .to(tl.float32)
    )
    if N == K - 1:
        dp = DSFINAL + (b * R + i[:, None]) * D + j[None, :]
    else:
        dp = DS + ((b * K + N + 1) * R + i[:, None]) * D + j[None, :]
    ds = tl.load(dp, (i[:, None] < R) & (jj[None, :] < width), 0)
    d1 = tl.load(D1 + i, i < R, 0)
    dc = tl.load(DC + i, i < R, 0)
    off = ((b * K + N) * R + i[:, None]) * D + j[None, :]
    tl.store(
        DS + off,
        ds * dc[:, None] + ds0 * d1[:, None],
        (i[:, None] < R) & (jj[None, :] < width),
    )
    if N == 0:
        sp = S + (b * R + i[:, None]) * D + j[None, :]
    else:
        sp = STATES + ((b * K + N - 1) * R + i[:, None]) * D + j[None, :]
    s = tl.load(sp, (i[:, None] < R) & (jj[None, :] < width), 0)
    write = tl.load(WRITE + off, (i[:, None] < R) & (jj[None, :] < width), 0).to(
        tl.float32
    )
    p = ((b * K + N) * G + g) * tl.cdiv(D // G, TN) + tl.program_id(2)
    tl.store(PART + p * R + i, tl.sum(ds0 * s, 1), i < R)
    tl.store(PART + (P + p) * R + i, tl.sum(ds * s, 1), i < R)
    tl.store(PART + (2 * P + p) * R + i, tl.sum(ds * write, 1), i < R)


@tr.jit
def _reduce_ramps(PART, D1, DC, GT, P: tl.constexpr, R: tl.constexpr, BP: tl.constexpr):
    kind = tl.program_id(0)
    p = tl.arange(0, BP)
    r = tl.program_id(1) * 16 + tl.arange(0, 16)
    x = tl.load(
        PART + (kind * P + p[:, None]) * R + r[None, :],
        (p[:, None] < P) & (r[None, :] < R),
        0,
    )
    value = tl.sum(x, 0)
    if kind == 0:
        tl.store(D1 + r, value, r < R)
    elif kind == 1:
        tl.store(DC + r, value, r < R)
    else:
        tl.store(GT + r, value, r < R)


@tr.jit
def _param_grad(
    A,
    B,
    S,
    STATES,
    DS,
    DSFINAL,
    BETA,
    D1,
    GT,
    OUT,
    K: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    KIND: tl.constexpr,
    TM: tl.constexpr = 32,
    TN: tl.constexpr = 32,
    TK: tl.constexpr = 32,
):
    # KIND: 0 dW=de h^T, 1 dQ=dr s0^T, 2 dK=e dwrite^T,
    #       3 dF=do s0^T (grouped), 4 dK2=do e^T (grouped).
    pid = tl.program_id(0)
    width: tl.constexpr = D // G if KIND >= 3 else D
    if KIND >= 3:
        g = pid % G
        bk = pid // G
    else:
        g = 0
        bk = pid
    b = bk // K
    n = bk % K
    i = tl.program_id(1) * TM + tl.arange(0, TM)
    j = tl.program_id(2) * TN + tl.arange(0, TN)
    rr = tl.arange(0, TK)
    acc = tl.full((TM, TN), 0, tl.float32)
    ROWS: tl.constexpr = 2 * C if KIND >= 3 else C
    COLS: tl.constexpr = C if KIND == 0 or KIND == 4 else R
    for base in range(tl.cdiv(width, TK)):
        v = base * TK + rr
        dv = g * width + v
        a = tl.load(
            A + (bk * ROWS + i[:, None]) * D + dv[None, :],
            (i[:, None] < ROWS) & (v[None, :] < width),
            0,
        )
        if KIND == 1:
            beta = tl.load(BETA + bk * C + i, i < C, 0)
            a = (-a * beta[:, None] / (R // 2)).to(OUT.dtype.element_ty)
        elif KIND >= 3:
            a = a.to(OUT.dtype.element_ty)
        if KIND == 0 or KIND == 4:
            bb = tl.load(
                B + (bk * C + j[:, None]) * D + dv[None, :],
                (j[:, None] < C) & (v[None, :] < width),
                0,
            )
        elif KIND == 1 or KIND == 3:
            # S holds the rounded, decayed state cached by the forward read.
            sp = S + (bk * R + j[:, None]) * D + dv[None, :]
            bb = tl.load(sp, (j[:, None] < R) & (v[None, :] < width), 0)
        else:
            if n == K - 1:
                sp = DSFINAL + (b * R + j[:, None]) * D + dv[None, :]
            else:
                sp = DS + ((bk + 1) * R + j[:, None]) * D + dv[None, :]
            bb = tl.load(sp, (j[:, None] < R) & (v[None, :] < width), 0)
            gt = tl.load(GT + j, j < R, 0)
            bb = (bb * gt[:, None]).to(OUT.dtype.element_ty)
        acc += tl.dot(a, tl.trans(bb), input_precision="tf32x3")
    tl.store(
        OUT + (pid * ROWS + i[:, None]) * COLS + j[None, :],
        acc,
        (i[:, None] < ROWS) & (j[None, :] < COLS),
    )


@tr.jit
def _beta_grad(DH, READ, DB, NUM: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BD)
    a = tl.load(DH + row * D + j, j < D, 0)
    r = tl.load(READ + row * D + j, j < D, 0)
    tl.store(DB + row, -tl.sum(a * r, 0))


def forward(kk, qk, fq, k2, w, v, beta, s, d1, dc, gt, save=True):
    b, k, c, r = kk.shape
    d = v.shape[-1]
    g = fq.shape[2]
    states = torch.empty((b, k, r, d), device=s.device, dtype=torch.float32)
    s0 = torch.empty_like(states, dtype=kk.dtype)
    h = torch.empty_like(v)
    e = torch.empty_like(v, dtype=kk.dtype)
    read = torch.empty_like(v) if save else torch.empty(0, device=s.device)
    write = (
        torch.empty_like(states, dtype=kk.dtype)
        if save
        else torch.empty(0, device=s.device)
    )
    o = torch.empty((b, k, 2 * c, d), device=s.device, dtype=torch.float32)
    kw = dict(K=k, C=c, R=r, D=d, TK=32, num_warps=4, enable_fp_fusion=False)
    for n in range(k):
        _read[(b, tr.cdiv(c, 64), tr.cdiv(d, 64))](
            qk, s, states, d1, v, beta, h, read, s0, N=n, SAVE=save, TM=64, TN=64, **kw
        )
        _delta[(b, tr.cdiv(c, 64), tr.cdiv(d, 32))](
            w,
            h,
            e,
            K=k,
            C=c,
            D=d,
            N=n,
            REVERSE=False,
            TM=64,
            TN=32,
            TK=32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _output[(b * g, tr.cdiv(2 * c, 64), tr.cdiv(d // g, 64))](
            fq, k2, e, s0, o, *fq.stride(), G=g, N=n, TM=64, TN=64, **kw
        )
        _write[(b, tr.cdiv(r, 32), tr.cdiv(d, 64))](
            kk, e, s, states, dc, gt, write, N=n, SAVE=save, TM=32, TN=64, **kw
        )
    return o, states, h, read, write, e, s0


def backward(
    kk, qk, fq, k2, w, beta, s, d1, dc, gt, states, h, read, write, e, s0, do, dsfinal
):
    b, k, c, r = kk.shape
    d = e.shape[-1]
    g = fq.shape[2]
    ds = torch.empty_like(states)
    p = b * k * g * tr.cdiv(d // g, 64)
    part = torch.empty((3, p, r), device=s.device, dtype=torch.float32)
    de = torch.empty_like(h)
    dh = torch.empty_like(h)
    kw = dict(K=k, C=c, R=r, D=d, TK=32, enable_fp_fusion=False)
    for n in range(k - 1, -1, -1):
        _de[(b * g, tr.cdiv(c, 64), tr.cdiv(d // g, 32))](
            kk, k2, do, ds, dsfinal, gt, de, G=g, N=n, TM=64, TN=32, num_warps=4, **kw
        )
        _delta[(b, tr.cdiv(c, 64), tr.cdiv(d, 32))](
            w,
            de,
            dh,
            K=k,
            C=c,
            D=d,
            N=n,
            REVERSE=True,
            TM=64,
            TN=32,
            TK=32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _ds[(b * g, tr.cdiv(r, 64), tr.cdiv(d // g, 64))](
            fq,
            qk,
            do,
            dh,
            beta,
            ds,
            dsfinal,
            d1,
            dc,
            s,
            states,
            write,
            part,
            p,
            *fq.stride(),
            G=g,
            N=n,
            TM=64,
            TN=64,
            num_warps=8,
            **kw
        )
    dw = torch.empty_like(w)
    dq = torch.empty_like(qk)
    dk = torch.empty_like(kk)
    df = torch.empty(fq.shape, device=fq.device, dtype=fq.dtype)
    dk2 = torch.empty_like(k2)
    for kind, a, bb, out in (
        (0, de, h, dw),
        (1, dh, h, dq),
        (2, e, h, dk),
        (3, do, h, df),
        (4, do, e, dk2),
    ):
        rows = 2 * c if kind >= 3 else c
        cols = c if kind in (0, 4) else r
        tm, tn = (64, 64) if kind == 0 else ((128, 64) if kind <= 2 else (64, 128))
        _param_grad[
            (b * k * (g if kind >= 3 else 1), tr.cdiv(rows, tm), tr.cdiv(cols, tn))
        ](
            a,
            bb,
            s0,
            states,
            ds,
            dsfinal,
            beta,
            d1,
            gt,
            out,
            G=g,
            KIND=kind,
            TM=tm,
            TN=tn,
            num_warps=4,
            **kw
        )
    db = torch.empty_like(beta)
    _beta_grad[(b * k * c,)](dh, read, db, NUM=b * k * c, D=d, BD=tr.next_power_of_2(d))
    dd1 = torch.empty_like(d1)
    ddc = torch.empty_like(dc)
    dgt = torch.empty_like(gt)
    _reduce_ramps[(3, tr.cdiv(r, 16))](
        part, dd1, ddc, dgt, P=p, R=r, BP=tr.next_power_of_2(p), num_warps=4
    )
    return dk, dq, df, dk2, dw, dh, db, ds[:, 0], dd1, ddc, dgt
