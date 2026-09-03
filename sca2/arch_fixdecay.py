"""
D-head with an INPUT-INDEPENDENT decay -- the "convolution" formulation.

Why. The measured breakdown (B=16, T=128, compiled, forward) is

    C head 0.529 ms | D head 1.878 ms | mix 0.156 | FFN 0.189 | layer 3.378
    attention block 0.899 ms

The D head alone costs 2.09x the whole attention block and is 55.6% of the
layer, while doing only 16.8M MACs -- about 0.1% of peak, against the C head's
1.5%. It is 15x less efficient per FLOP than its neighbour. The cost is not
arithmetic, it is the shape of the scan.

What changes. The gate stops depending on the input: `a[m,g]` becomes a learned
complex decay. Then

    D[t,r,m,g] = a[m,g]^(t-r)

depends only on the LAG, so it is a fixed convolution kernel -- the same object
S4/LRU/RetNet exploit. Three consequences, each removing a different cost:

  * the decay is shared across batch AND chunk: (C,C,M,G) = 8k elements computed
    once per forward, instead of (B,N,M,G,C,C) = 2.1M rebuilt every call;
  * the intra-chunk contraction becomes one batched GEMM with the decay as a
    shared operand, instead of B.N tiny per-chunk matmuls;
  * the carry recurrence carry[n+1] = a^C . carry[n] + s_end[n] also has a
    lag-only kernel, so the chunk loop disappears entirely -- no python loop, no
    sequential dependency.

What it costs. Input-dependent gating is presumably what carries the copy
behaviour, so this is an architecture change with a real quality risk, not a
free win. This module exists to measure the SPEED CEILING of the convolution
route first: if the layer does not drop well below 2x attention here, the route
does not reach 1.x either and no quality experiment is needed.

A lower-risk middle ground, if the ceiling looks good: keep `a` input-dependent
but constant WITHIN a chunk. Intra-chunk the decay is again lag-only (so the
fast form survives) and only the chunk boundaries stay data-dependent.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import DHeadBase, _rms
from .registry import register
from .versions.v1_quad_scan import CHeadQuad
from .compiled import wrap as _cw
from .arch_sepq import DHeadSepQ


class DHeadFixedDecay(DHeadSepQ):
    CHUNK = 8

    def __init__(self, d, M=16, G=8, dv=None, max_len=None, delta_rule=False):
        super().__init__(d, M, G, dv, max_len, delta_rule)
        del self.gr, self.gi                      # no input-dependent gate
        # log|a| = -softplus(dec_mag) <= 0 by construction; phase is free
        self.dec_mag = nn.Parameter(torch.randn(M, G))
        self.dec_ph = nn.Parameter(torch.rand(M, G) * 2 * math.pi)

    # ---- fixed lag kernels ------------------------------------------------ #
    def _kernels(self, C, N, device, dtype):
        la = -F.softplus(self.dec_mag.to(dtype))          # (M,G) <= 0
        ph = self.dec_ph.to(dtype)
        t = torch.arange(C, device=device, dtype=dtype)
        lag = t[:, None] - t[None, :]                     # (C,C)
        keep = lag >= 0
        e = lag.clamp(min=0)[..., None, None] * la        # (C,C,M,G)
        p = lag.clamp(min=0)[..., None, None] * ph
        mag = torch.where(keep[..., None, None], e.exp(), torch.zeros((), dtype=dtype, device=device))
        d_re, d_im = mag * p.cos(), mag * p.sin()

        # A_pos[t] = a^(t+1): product of gates from the chunk start through t
        e1 = (t + 1)[:, None, None] * la                  # (C,M,G)
        p1 = (t + 1)[:, None, None] * ph
        a_re, a_im = e1.exp() * p1.cos(), e1.exp() * p1.sin()

        # level 2: the chunk-to-chunk decay is a^C, also lag-only
        n = torch.arange(N, device=device, dtype=dtype)
        lag2 = n[:, None] - n[None, :] - 1                # carry[n] <- s_end[p<n]
        keep2 = lag2 >= 0
        e2 = lag2.clamp(min=0)[..., None, None] * (C * la)
        p2 = lag2.clamp(min=0)[..., None, None] * (C * ph)
        m2 = torch.where(keep2[..., None, None], e2.exp(), torch.zeros((), dtype=dtype, device=device))
        d2_re, d2_im = m2 * p2.cos(), m2 * p2.sin()
        # A_chunk^n for the incoming state
        e3 = n[:, None, None] * (C * la)
        p3 = n[:, None, None] * (C * ph)
        a2_re, a2_im = e3.exp() * p3.cos(), e3.exp() * p3.sin()
        return d_re, d_im, a_re, a_im, d2_re, d2_im, a2_re, a2_im

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        C = min(self.CHUNK, T)
        pad = (-T) % C
        v = self.V(z)
        if pad:
            v = F.pad(v, (0, 0, 0, pad))
        Tp = T + pad
        N = Tp // C
        d_re, d_im, a_re, a_im, d2_re, d2_im, a2_re, a2_im = \
            self._kernels(C, N, z.device, z.dtype)

        vg = v.view(B, N, C, G, gs).permute(0, 1, 3, 2, 4)        # (B,N,G,C,gs)
        ire = torch.einsum("trmg,bngrj->bnmgtj", d_re, vg)
        iim = torch.einsum("trmg,bngrj->bnmgtj", d_im, vg)

        # carry[n] = sum_{p<n} (a^C)^(n-1-p) . s_end[p]  +  (a^C)^n . s_in
        er, ei = ire[..., -1, :], iim[..., -1, :]                 # (B,N,M,G,gs)
        cr = (torch.einsum("npmg,bpmgj->bnmgj", d2_re, er)
              - torch.einsum("npmg,bpmgj->bnmgj", d2_im, ei))
        ci = (torch.einsum("npmg,bpmgj->bnmgj", d2_re, ei)
              + torch.einsum("npmg,bpmgj->bnmgj", d2_im, er))
        if not st.get("empty", False):
            sr = st["sr"].view(B, 1, M, G, gs)
            si = st["si"].view(B, 1, M, G, gs)
            A2r, A2i = a2_re[None, :, :, :, None], a2_im[None, :, :, :, None]
            cr = cr + A2r * sr - A2i * si
            ci = ci + A2r * si + A2i * sr

        Ar = a_re.permute(1, 2, 0)[None, None, :, :, :, None]     # (1,1,M,G,C,1)
        Ai = a_im.permute(1, 2, 0)[None, None, :, :, :, None]
        c4r, c4i = cr[..., None, :], ci[..., None, :]             # (B,N,M,G,1,gs)
        s_re = ire + Ar * c4r - Ai * c4i
        s_im = iim + Ar * c4i + Ai * c4r
        s_re = s_re.permute(0, 1, 4, 2, 3, 5).reshape(B, Tp, M, dv)[:, :T]
        s_im = s_im.permute(0, 1, 4, 2, 3, 5).reshape(B, Tp, M, dv)[:, :T]

        ar, ai, br, bi = self._alpha_beta(z)
        return _rms(self._read(s_re, s_im, ar, ai, br, bi, 2)), \
            {"sr": s_re[:, -1], "si": s_im[:, -1], "empty": False}

    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        la = -F.softplus(self.dec_mag.to(z_t.dtype))
        ph = self.dec_ph.to(z_t.dtype)
        gr = (la.exp() * ph.cos())[None, :, :, None]
        gi = (la.exp() * ph.sin())[None, :, :, None]
        rg = state["sr"].view(B, M, G, gs)
        ig = state["si"].view(B, M, G, gs)
        sr = (gr * rg - gi * ig).reshape(B, M, dv) + self.V(z_t)[:, None, :]
        si = (gr * ig + gi * rg).reshape(B, M, dv)
        ar, ai, br, bi = self._alpha_beta(z_t)
        return _rms(self._read(sr, si, ar, ai, br, bi, 1)), \
            {"sr": sr, "si": si, "empty": False}

    def init_state(self, B, device, dtype):
        st = super().init_state(B, device, dtype)
        st["empty"] = True
        return st


register("fix", CHeadQuad, DHeadFixedDecay, arch=True,
         note="input-independent decay (convolution form)")
register("fix_cc", CHeadQuad, DHeadFixedDecay, arch=True, wrap=_cw,
         note="input-independent decay + compile")


# --------------------------------------------------------------------------- #
#  long-context combination: v3's chunked C head + the polar/separable D head
# --------------------------------------------------------------------------- #
from .versions.v3_longctx import CHead as CHeadChunked   # noqa: E402
from .arch_sepq import DHeadSepQPolar as _DPolar         # noqa: E402

register("v3polar", CHeadChunked, _DPolar, arch=True,
         note="chunked C head + polar separable D head")
register("v3polar_cc", CHeadChunked, _DPolar, arch=True, wrap=_cw,
         note="chunked C head + polar separable D head + compile")
