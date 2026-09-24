"""Causal depthwise convolution with optional SiLU in its epilogue and adjoint.

SiLU belongs before the K/V projections. Fusing it in phase_codes would instead
compute silu(K(z)) and change the model. Match autocast's rounding at the conv
output and SiLU output, while returning the layer norm's input dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _preact(X, W, b, t, d, valid, T: tl.constexpr, D: tl.constexpr,
             H: tl.constexpr, DT: tl.constexpr):
    a = tl.full(t.shape, 0, tl.float32)
    for h in tl.static_range(H):
        x = tl.load(X + (b * (T + H - 1) + t + h) * D + d, valid, 0).to(DT).to(tl.float32)
        w = tl.load(W + d * H + h, valid, 0).to(DT).to(tl.float32)
        a = a + x * w
    return a.to(DT).to(tl.float32)


@triton.jit
def _activation_grad(X, W, DY, b, t, d, valid, T: tl.constexpr, D: tl.constexpr,
               H: tl.constexpr, DT: tl.constexpr, SILU: tl.constexpr):
    dy = tl.load(DY + (b * T + t) * D + d, valid, 0).to(DT).to(tl.float32)
    if SILU:
        a = _preact(X, W, b, t, d, valid, T, D, H, DT)
        sig = 1.0 / (1.0 + tl.exp(-a))
        dy = (dy * (sig * (1.0 + a * (1.0 - sig)))).to(DT).to(tl.float32)
    return dy


@triton.jit
def _conv_fwd(X, W, Y, B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
              H: tl.constexpr, DT: tl.constexpr, BLOCK: tl.constexpr, SILU: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d, t, b = i % D, (i // D) % T, i // (T * D)
    valid = i < B * T * D
    a = _preact(X, W, b, t, d, valid, T, D, H, DT)
    if SILU:
        a = a / (1.0 + tl.exp(-a))
    tl.store(Y + i, a.to(DT), valid)


@triton.jit
def _conv_dx(X, W, DY, DX, B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
             H: tl.constexpr, DT: tl.constexpr, BLOCK: tl.constexpr, SILU: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d, tx, b = i % D, (i // D) % (T + H - 1), i // ((T + H - 1) * D)
    valid = i < B * (T + H - 1) * D
    dx = tl.full((BLOCK,), 0, tl.float32)
    for h in tl.static_range(H):
        t = tx - h
        mask = valid & (t >= 0) & (t < T)
        da = _activation_grad(X, W, DY, b, t, d, mask, T, D, H, DT, SILU)
        w = tl.load(W + d * H + h, valid, 0).to(DT).to(tl.float32)
        dx = dx + da * w
    tl.store(DX + i, dx.to(DT), valid)


@triton.jit
def _conv_dw(X, W, DY, PART, B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
             H: tl.constexpr, DT: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, SILU: tl.constexpr):
    bt = tl.program_id(0) * BT + tl.arange(0, BT)
    ds = tl.program_id(1) * BD + tl.arange(0, BD)
    b = tl.broadcast_to((bt // T)[:, None], (BT, BD))
    t = tl.broadcast_to((bt % T)[:, None], (BT, BD))
    d = tl.broadcast_to(ds[None, :], (BT, BD))
    valid = (bt[:, None] < B * T) & (d < D)
    da = _activation_grad(X, W, DY, b, t, d, valid, T, D, H, DT, SILU)
    for h in tl.static_range(H):
        x = tl.load(X + (b * (T + H - 1) + t + h) * D + d, valid, 0).to(DT).to(tl.float32)
        tl.store(PART + (tl.program_id(0) * D + ds) * H + h,
                 tl.sum(da * x, 0), ds < D)


class _Conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, dtype, silu):
        b, tx, d = x.shape
        h = w.shape[-1]
        t = tx - h + 1
        dt = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32
        y = torch.empty((b, t, d), device=x.device, dtype=x.dtype)
        _conv_fwd[(triton.cdiv(y.numel(), 256),)](
            x, w, y, B=b, T=t, D=d, H=h, DT=dt, BLOCK=256, SILU=silu,
            enable_fp_fusion=False)
        ctx.save_for_backward(x, w)
        ctx.dtype = dtype
        ctx.silu = silu
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        b, tx, d = x.shape
        h = w.shape[-1]
        t = tx - h + 1
        dt = tl.bfloat16 if ctx.dtype == torch.bfloat16 else tl.float32
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        part = torch.empty((triton.cdiv(b * t, 32), d, h), device=x.device, dtype=torch.float32)
        _conv_dx[(triton.cdiv(x.numel(), 256),)](
            x, w, dy, dx, B=b, T=t, D=d, H=h, DT=dt, BLOCK=256, SILU=ctx.silu,
            enable_fp_fusion=False)
        _conv_dw[(triton.cdiv(b * t, 32), triton.cdiv(d, 32))](
            x, w, dy, part, B=b, T=t, D=d, H=h, DT=dt, BT=32, BD=32, SILU=ctx.silu,
            enable_fp_fusion=False)
        dw = part.sum(0).to(ctx.dtype).to(w.dtype).reshape_as(w)
        return dx, dw, None, None


def causal_conv(x, w, silu=False):
    """Valid depthwise conv on [history | tokens], optionally followed by SiLU."""
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    return _Conv.apply(x.contiguous(), w.contiguous(), dtype, silu)


def conv_silu(x, w):
    return causal_conv(x, w, silu=True)
