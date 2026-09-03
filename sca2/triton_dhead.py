"""
Fused Triton kernel for the D-head chunked scan.

What it removes. Per chunk, the PyTorch path materializes dl, dp, mag, D_re,
D_im, ire, iim, cr, ci and the permuted state -- about eleven B.M.G.C.C tensors
written to and read from DRAM, ~11.4M element-transfers per prefill at
B=8,T=128,C=8. The decay matrix is pure intermediate: it is built from two
cumsums and consumed immediately by a matmul. A fused kernel keeps it in
registers and writes only the state, ~2.1M transfers -- and collapses ~20 kernel
launches per chunk into one.

Layout (one program per (b, m), one launch per chunk):

    cla, cph : (B, M, G, T)   log-magnitude and phase cumsums, per chunk
    v        : (B, T, dv)     value stream, shared across m
    carry    : (B, M, dv)     complex state entering the chunk
    out      : (B, T, M, dv)  complex state, this chunk's C rows

Each program builds D[t,r,j] = exp(cla[t,g]-cla[r,g]) . e^{i(cph[t,g]-cph[r,g])}
for r <= t entirely in registers -- shape (C, C, dv), 4096 elements -- contracts
it against v over r, adds the rotated carry, and writes the C new state rows plus
the outgoing carry.

The gate is expected in LOG-POLAR form (see MATH.md 4b), so `cla` is already
<= 0 by construction and no `tiny` floor or anti-NaN clamp is baked in here.

MEASURED OUTCOME -- the kernel is correct but does NOT win.

    scan in isolation, B=16 T=128       time      vs kernel
      this kernel                      5.68 ms      1.00x
      PyTorch eager                   10.15 ms      0.56x
      PyTorch + torch.compile          4.91 ms      1.16x

It beats the eager formulation by 1.79x and loses to inductor by 16%. In the
full layer the gap is larger still (3.18 vs 1.66 us/token) because an
autograd.Function is opaque to inductor: inserting it also forfeits the fusion
of the scan with the cumsums before it and the read after it. Two costs, both
real.

To beat the compiler the kernel would have to subsume the WHOLE head -- cumsums,
scan, and the read-out with its contraction over m -- so that it captures the
cross-op fusion inductor is already doing on top of the register residency
inductor cannot do. That needs a different tiling, because the m-contraction
spans programs while the carry chain forces chunks to be sequential.

Kept, iso-gated (`python -m sca2.iso tri --against polar`), as the honest
starting point for that larger kernel and as the measurement that says the
smaller one is not worth shipping.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_fwd(
    CLA, CPH, V, CR_IN, CI_IN, OUT_RE, OUT_IM, CR_OUT, CI_OUT,
    s_cla_b, s_cla_m, s_cla_g, s_cla_t,
    s_v_b, s_v_t,
    s_c_b, s_c_m,
    s_o_b, s_o_t, s_o_m,
    n_off,
    M: tl.constexpr, C: tl.constexpr, DV: tl.constexpr, GS: tl.constexpr,
):
    """One program per (b, m); one launch per chunk.

    The decay row for a query position is built, used and discarded inside the
    loop, so only (C, DV) is ever live. Materializing the whole (C, C, DV) block
    instead costs ~6 live tiles of 4096 floats, which is 96 KB against Turing's
    64 KB register file per SM -- it spills to local memory, i.e. back to DRAM,
    which is the exact traffic this kernel exists to remove. Measured: 1.6x
    slower than the PyTorch path in that form.
    """
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M

    r = tl.arange(0, C)
    j = tl.arange(0, DV)
    g = j // GS

    base = CLA + b * s_cla_b + m * s_cla_m + g * s_cla_g
    basep = CPH + b * s_cla_b + m * s_cla_m + g * s_cla_g
    # decay source positions (the "r" axis), loaded once
    cla_r = tl.load(base + (n_off + r[:, None]) * s_cla_t)          # (C, DV)
    cph_r = tl.load(basep + (n_off + r[:, None]) * s_cla_t)
    v = tl.load(V + b * s_v_b + (n_off + r[:, None]) * s_v_t + j[None, :])

    c_re = tl.load(CR_IN + b * s_c_b + m * s_c_m + j)               # (DV,)
    c_im = tl.load(CI_IN + b * s_c_b + m * s_c_m + j)

    for ti in tl.static_range(C):
        cla_i = tl.load(base + (n_off + ti) * s_cla_t)              # (DV,)
        cph_i = tl.load(basep + (n_off + ti) * s_cla_t)
        dl = cla_i[None, :] - cla_r                                 # (C, DV)
        dp = cph_i[None, :] - cph_r
        keep = (ti >= r)[:, None]
        mag = tl.where(keep, tl.exp(tl.minimum(dl, 0.0)), 0.0)
        s_re = tl.sum(mag * tl.cos(dp) * v, axis=0)                 # (DV,)
        s_im = tl.sum(mag * tl.sin(dp) * v, axis=0)

        am = tl.exp(cla_i)
        a_re = am * tl.cos(cph_i)
        a_im = am * tl.sin(cph_i)
        o_re = s_re + a_re * c_re - a_im * c_im
        o_im = s_im + a_re * c_im + a_im * c_re

        o_off = b * s_o_b + (n_off + ti) * s_o_t + m * s_o_m + j
        tl.store(OUT_RE + o_off, o_re)
        tl.store(OUT_IM + o_off, o_im)
        if ti == C - 1:
            tl.store(CR_OUT + b * s_c_b + m * s_c_m + j, o_re)
            tl.store(CI_OUT + b * s_c_b + m * s_c_m + j, o_im)


def scan_forward(cla, cph, v, sr0, si0, C):
    """cla, cph: (B,M,G,T); v: (B,T,dv); sr0, si0: (B,M,dv).

    Returns (S_re, S_im) of shape (B,T,M,dv) and the closing carry.
    """
    B, M, G, T = cla.shape
    dv = v.shape[-1]
    gs = dv // G
    assert T % C == 0, "caller pads to a whole number of chunks"
    out_re = torch.empty(B, T, M, dv, device=v.device, dtype=v.dtype)
    out_im = torch.empty_like(out_re)
    # CLONE, never alias: `.contiguous()` returns the argument itself when it
    # already is, and the ping-pong below would then write the caller's incoming
    # state on the second chunk -- silently corrupting a decode-after-prefill.
    cr, ci = sr0.contiguous().clone(), si0.contiguous().clone()
    cr_n, ci_n = torch.empty_like(cr), torch.empty_like(ci)
    cla, cph, v = cla.contiguous(), cph.contiguous(), v.contiguous()

    for n in range(0, T, C):
        _chunk_fwd[(B * M,)](
            cla, cph, v, cr, ci, out_re, out_im, cr_n, ci_n,
            cla.stride(0), cla.stride(1), cla.stride(2), cla.stride(3),
            v.stride(0), v.stride(1),
            cr.stride(0), cr.stride(1),
            out_re.stride(0), out_re.stride(1), out_re.stride(2),
            n, M=M, C=C, DV=dv, GS=gs, num_warps=4,
        )
        cr, ci, cr_n, ci_n = cr_n, ci_n, cr, ci      # ping-pong, no allocation
    return out_re, out_im, cr, ci


def scan_forward_torch(cla, cph, v, sr0, si0, C):
    """Reference for `scan_forward`, same contract, plain PyTorch.

    Used as the backward of the fused op (see `FusedScan`) and as the fp64 path,
    since Triton's float64 support is thin.
    """
    B, M, G, T = cla.shape
    dv = v.shape[-1]
    gs = dv // G
    sr, si = sr0, si0
    outs_r, outs_i = [], []
    for n in range(0, T, C):
        cl = cla[..., n:n + C]                                  # (B,M,G,C)
        cp = cph[..., n:n + C]
        dl = cl.unsqueeze(-1) - cl.unsqueeze(-2)                # (B,M,G,t,r)
        dp = cp.unsqueeze(-1) - cp.unsqueeze(-2)
        keep = torch.tril(torch.ones(C, C, device=v.device, dtype=v.dtype))
        mag = dl.clamp(max=0).exp() * keep
        vg = v[:, n:n + C].view(B, C, G, gs).permute(0, 2, 1, 3)  # (B,G,C,gs)
        ire = torch.einsum("bmgtr,bgrj->bmgtj", mag * dp.cos(), vg)
        iim = torch.einsum("bmgtr,bgrj->bmgtj", mag * dp.sin(), vg)
        am = cl.exp()
        a_re = (am * cp.cos()).unsqueeze(-1)                     # (B,M,G,C,1)
        a_im = (am * cp.sin()).unsqueeze(-1)
        s_in_r = sr.view(B, M, G, 1, gs)
        s_in_i = si.view(B, M, G, 1, gs)
        cr = ire + a_re * s_in_r - a_im * s_in_i                 # (B,M,G,C,gs)
        ci = iim + a_re * s_in_i + a_im * s_in_r
        o_r = cr.permute(0, 3, 1, 2, 4).reshape(B, C, M, dv)
        o_i = ci.permute(0, 3, 1, 2, 4).reshape(B, C, M, dv)
        outs_r.append(o_r); outs_i.append(o_i)
        sr, si = o_r[:, -1], o_i[:, -1]
    return torch.cat(outs_r, 1), torch.cat(outs_i, 1), sr, si


class FusedScan(torch.autograd.Function):
    """Triton forward, exact backward.

    The backward recomputes the scan in PyTorch under `enable_grad` and lets
    autograd differentiate it, rather than hand-deriving dcla/dcph/dv. That is
    exact by construction -- the gradient of the reference computation, checked
    against the unfused head by `sca2.iso` -- and it costs a recompute, so the
    BACKWARD IS NOT YET ACCELERATED. Only the forward is. A hand-written
    backward kernel is the follow-up; this staging keeps the forward win
    measurable without betting it on hand-derived gradients.
    """

    @staticmethod
    def forward(ctx, cla, cph, v, sr0, si0, C):
        ctx.save_for_backward(cla, cph, v, sr0, si0)
        ctx.C = C
        with torch.no_grad():
            return scan_forward(cla, cph, v, sr0, si0, C)

    @staticmethod
    def backward(ctx, d_re, d_im, d_cr, d_ci):
        cla, cph, v, sr0, si0 = ctx.saved_tensors
        with torch.enable_grad():
            ins = [t.detach().requires_grad_(True) for t in (cla, cph, v, sr0, si0)]
            out = scan_forward_torch(*ins, ctx.C)
            grads = torch.autograd.grad(
                out, ins, (d_re.contiguous(), d_im.contiguous(),
                           d_cr.contiguous(), d_ci.contiguous()),
                allow_unused=True)
        return (*grads, None)


def _pow2(n):
    return n >= 2 and (n & (n - 1)) == 0


def fused_scan(cla, cph, v, sr0, si0, C):
    """Fused scan where it applies, reference otherwise.

    Falls back for: non-CUDA, float64 (Triton's support is thin, and iso's
    tightest check runs there), and any tile size that is not a power of two >= 2
    -- `tl.arange` requires it, and C=1 happens whenever T=1.
    """
    if (v.is_cuda and v.dtype == torch.float32
            and _pow2(C) and _pow2(v.shape[-1]) and cla.shape[-1] % C == 0):
        return FusedScan.apply(cla, cph, v, sr0, si0, C)
    return scan_forward_torch(cla, cph, v, sr0, si0, C)


# --------------------------------------------------------------------------- #
#  head wiring
# --------------------------------------------------------------------------- #
import torch.nn.functional as F                                    # noqa: E402
from .arch_sepq import DHeadSepQPolar                              # noqa: E402
from .ref import _rms                                              # noqa: E402
from .registry import register                                     # noqa: E402
from .versions.v1_quad_scan import CHeadQuad                       # noqa: E402
from .compiled import wrap as _cw                                  # noqa: E402


def _chunk_cumsum(x, C):
    B, M, G, T = x.shape
    return x.view(B, M, G, T // C, C).cumsum(-1).view(B, M, G, T)


class DHeadTriton(DHeadSepQPolar):
    """Same function as DHeadSepQPolar, with the chunk loop replaced by the
    fused kernel. Parameters and semantics are identical, so it is iso-checkable
    against `polar` directly (`python -m sca2.iso tri --against polar`)."""

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        C = min(self.CHUNK, T)
        pad = (-T) % C
        lap, php = self._log_polar(h, B, T)              # (B,M,G,T)
        v = self.V(z)                                    # (B,T,dv)
        if pad:
            # identity gate and zero value on the padding: it follows every real
            # position, so real rows and the closing state are untouched
            lap = F.pad(lap, (0, pad)); php = F.pad(php, (0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        S_re, S_im, _, _ = fused_scan(_chunk_cumsum(lap, C), _chunk_cumsum(php, C),
                                      v, st["sr"], st["si"], C)
        S_re, S_im = S_re[:, :T], S_im[:, :T]
        ar, ai, br, bi = self._alpha_beta(z)
        u = self._read(S_re, S_im, ar, ai, br, bi, 2)
        return _rms(u), {"sr": S_re[:, T - 1], "si": S_im[:, T - 1]}


register("tri", CHeadQuad, DHeadTriton, arch=True,
         note="polar + fused Triton D-head scan")
register("tri_cc", CHeadQuad, DHeadTriton, arch=True, wrap=_cw,
         note="polar + fused Triton D-head scan + compile")
