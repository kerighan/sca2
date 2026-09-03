"""
Version v2 -- "merged projection".

v1 with ONE change: the projections. Its D-head scan is v1's, reused via
`prefill_core` rather than forked.

  * The once-per-sequence projections (c.K, c.V, dh.V, dh.gr, dh.gi) act on the
    same B.T rows, so they are one d -> 448 gemm instead of five small-N ones.
  * `h` is `z` shifted by one and every projection is linear, so K(h), gr(h),
    gi(h) are SHIFTS of the z-projections -- three gemms that never needed to
    exist. At decode the shift is a value cached in the state, so a step needs
    one gemm instead of eight.
  * `dh.qr`/`dh.qi` stay INSIDE the chunk loop, and so are not in the merged
    gemm. v1's HOIST_Q experiment measured that retaining 2.B.T.M.dv of
    normalized q activations costs 5x on the backward; per-chunk recomputation
    is cheaper than retention here.

Bit-exact with v0/v1 up to gemm reduction order: no parameter, shape or semantic
change, so it passes `python -m sca2.iso` unchanged and is a drop-in swap.

WHAT THIS MODULE ALSO RECORDS: a measured negative result.

`DHeadLoopFree` below makes the chunk axis a real tensor axis -- a two-level
log-space scan, intra-chunk batched over all chunks at once, inter-chunk carry
in closed form -- so the D-head prefill has no python loop at all. The reasoning
was that v1 is kernel-count bound: its compiled prefill runs 375 kernels for
2.08 ms, and its two biggest line items are 32 tiny gemms of ~16 us and 32 of
~12 us that exist only because the chunk loop is a python loop.

Measured, it is 2.6x SLOWER on prefill (6.31 vs 2.43 ms compiled) at 89 vs
54 MB. Two reasons, both of which invalidate that reasoning:

  * The python loop was acting as CACHE BLOCKING. One chunk at a time keeps the
    working set in L2; batching every chunk turns each intermediate into a DRAM
    round trip, and adds 6-D strided permutes over 1M-element tensors.
  * The "78% elementwise at ~20 us" profile that motivated it was measured on
    v1 EAGER. Under torch.compile, inductor had already fused those chains, so
    the kernel-count headroom was mostly spent before v2 started. Carrying an
    eager profile's conclusion into the compiled regime was the mistake.

It is kept, registered as `x_loopfree`, so the result stays measurable rather
than becoming folklore.

NOT done here: splitting `mix` to avoid its concat is a wash (one gemm plus a
262k copy against two gemms). The separable D-head query, the log-polar gate and
the chunked C head all change parameters or semantics, so they belong behind
their own A/B, not in a version.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ref import CHeadBase, DHeadBase, SCA2Layer, LayerCfg, _rms
from .v1_quad_scan import causal_mask, DHeadScan as V1DHeadScan


# ==========================================================================  #
#  C head -- v1's scalar-kernel form, taking projections from the caller
# ==========================================================================  #
class CHead(CHeadBase):
    def init_state(self, B, device, dtype):
        st = super().init_state(B, device, dtype)
        st["pos"] = torch.zeros((), device=device, dtype=dtype)
        st["empty"] = True
        return st

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    # -- standalone entry points (own projections), so the head stays testable
    def prefill(self, z, h, state=None):
        return self.prefill_from(self.K(z), self.K(h), self.V(z), z.shape[:2],
                                 z.device, z.dtype, state)

    def step(self, z_t, h_t, state):
        return self.step_from(self.K(z_t), self.K(h_t), self.V(z_t), state)

    # -- fused entry points -------------------------------------------------
    def prefill_from(self, Kz, Kh, v, shape, device, dtype, state=None):
        B, T = shape
        st = state if state is not None else self.init_state(B, device, dtype)
        p = (torch.arange(T, device=device, dtype=dtype) + st["pos"])[:, None] * self.omega
        pw, pq = Kh * self.theta + p, Kz * self.theta + p

        cw, sw = pw.cos(), pw.sin()
        cq, sq = pq.cos(), pq.sin()
        A = self.wr * cw - self.wi * sw
        Bm = self.wr * sw + self.wi * cw

        Fq = torch.cat([cq, sq], -1)                       # (B,T,2M)
        Fk = torch.cat([torch.cat([A, Bm], -1),
                        torch.cat([Bm, -A], -1)], 1)       # (B,2T,2M)
        K2 = (Fq @ Fk.transpose(1, 2)).view(B, T, 2, T)
        K2 = K2.masked_fill(causal_mask(T, device)[None, :, None, :], 0)
        o = (K2.reshape(B, T * 2, T) @ v).view(B, T, 2 * self.dv) / self.M

        if not st.get("empty", False):
            c1 = self.wr * cq + self.wi * sq
            c2 = self.wr * sq - self.wi * cq
            r1 = torch.einsum("btm,bmj->btj", c1, st["sr"])
            r2 = torch.einsum("btm,bmj->btj", c2, st["si"])
            i1 = torch.einsum("btm,bmj->btj", c2, st["sr"])
            i2 = torch.einsum("btm,bmj->btj", c1, st["si"])
            o = o + torch.cat([r1 + r2, i2 - i1], -1) / self.M

        sr = torch.einsum("btm,btj->bmj", cw, v) + st["sr"]
        si = torch.einsum("btm,btj->bmj", sw, v) + st["si"]
        return _rms(o), {"sr": sr, "si": si, "pos": st["pos"] + T, "empty": False}

    def step_from(self, Kz, Kh, v, state):
        p = state["pos"]
        pw = Kh * self.theta + p * self.omega
        pq = Kz * self.theta + p * self.omega
        sr = torch.addcmul(state["sr"], v[:, None, :], pw.cos()[:, :, None])
        si = torch.addcmul(state["si"], v[:, None, :], pw.sin()[:, :, None])
        cq, sq = pq.cos(), pq.sin()
        c1 = self.wr * cq + self.wi * sq
        c2 = self.wr * sq - self.wi * cq
        cc = torch.stack([c1, c2], 1)
        ss = torch.stack([sr, si], 1)
        m = torch.einsum("bam,bcmj->bacj", cc, ss) / self.M
        u = torch.cat([m[:, 0, 0] + m[:, 1, 1], m[:, 0, 1] - m[:, 1, 0]], -1)
        return _rms(u), {"sr": sr, "si": si, "pos": state["pos"] + 1, "empty": False}


# ==========================================================================  #
#  D head -- v1's scan, taking projections from the caller
# ==========================================================================  #
class DHead(V1DHeadScan):
    """v1's chunked scan, reused. Only the entry points are new."""
    CHUNK = 8

    def prefill_from(self, z, v, grh, gih, state=None):
        """grh/gih are PRE-activation (B,T,M,G)."""
        s2 = 1.0 / math.sqrt(2)
        return self.prefill_core(z, v, torch.tanh(grh) * s2,
                                 torch.tanh(gih) * s2, state)

    def step_from(self, z_t, v, grh, gih, state):
        B = z_t.size(0)
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        s2 = 1.0 / math.sqrt(2)
        ar = (torch.tanh(grh) * s2)[:, :, :, None]
        ai = (torch.tanh(gih) * s2)[:, :, :, None]
        rg = state["sr"].view(B, M, G, gs)
        ig = state["si"].view(B, M, G, gs)
        sr = (ar * rg - ai * ig).reshape(B, M, dv) + v[:, None, :]
        si = (ar * ig + ai * rg).reshape(B, M, dv)
        qr, qi = self._q(z_t[:, None], B, 1)
        qr, qi = qr[:, 0], qi[:, 0]
        u = torch.cat([(sr * qr + si * qi).mean(1), (si * qr - sr * qi).mean(1)], -1)
        return _rms(u), {"sr": sr, "si": si}


# ==========================================================================  #
#  D head, loop-free -- MEASURED SLOWER. Kept as x_loopfree.
# ==========================================================================  #
class DHeadLoopFree(DHeadBase):
    r"""Level 1 unrolls a chunk in closed form, for every chunk at once:

        s[n,t] = A[n,t] . carry[n] + sum_{r<=t} D[n,t,r] . v[n,r]

    Level 2 is the same recurrence one level up, over chunk index, so it too has
    a closed form instead of a loop. With `alast[n]` the product of all gates in
    chunk n, `send[n]` the intra-chunk state at that chunk's last position, and
    `f` the inclusive cumsum of `log|alast|` over chunks:

        carry[n] = exp(f[n-1]) . s_in  +  sum_{p<n} exp(f[n-1] - f[p]) . send[p]

    Both levels use cumsum differences of log-magnitude, which are always <= 0
    (|a| <= 1), so nothing overflows and nothing is divided by a decayed prefix.
    """
    CHUNK = 8

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        grh = self.gr(h).view(B, T, self.M, self.G)
        gih = self.gi(h).view(B, T, self.M, self.G)
        return self.prefill_from(self.V(z), grh, gih,
                                 self.qr(z).view(B, T, self.M, self.dv),
                                 self.qi(z).view(B, T, self.M, self.dv), state)

    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        return self.step_from(self.V(z_t),
                              self.gr(h_t).view(B, self.M, self.G),
                              self.gi(h_t).view(B, self.M, self.G),
                              self.qr(z_t).view(B, self.M, self.dv),
                              self.qi(z_t).view(B, self.M, self.dv), state)

    def prefill_from(self, v, grh, gih, qr, qi, state=None):
        """grh/gih are PRE-activation (B,T,M,G); qr/qi un-normalized (B,T,M,dv)."""
        B, T, dv = v.shape
        M, G, gs = self.M, self.G, self.gs
        st = state if state is not None else self.init_state(B, v.device, v.dtype)
        C = min(self.CHUNK, T)
        pad = (-T) % C
        if pad:                     # padded positions follow every real one, so
            v = F.pad(v, (0, 0, 0, pad))          # real outputs and the closing
            grh = F.pad(grh, (0, 0, 0, 0, 0, pad))  # state are unaffected
            gih = F.pad(gih, (0, 0, 0, 0, 0, pad))
            qr = F.pad(qr, (0, 0, 0, 0, 0, pad))
            qi = F.pad(qi, (0, 0, 0, 0, 0, pad))
        Tp, N = T + pad, (T + pad) // C
        tiny = torch.finfo(v.dtype).tiny
        s2 = 1.0 / math.sqrt(2)

        gr = torch.tanh(grh) * s2
        gi = torch.tanh(gih) * s2
        # permute while still B.Tp.M.G -- C times smaller than the decay matrix
        la = (0.5 * torch.log(gr.square() + gi.square() + tiny)) \
            .view(B, N, C, M, G).permute(0, 1, 3, 4, 2)          # (B,N,M,G,C)
        ph = torch.atan2(gi, gr).view(B, N, C, M, G).permute(0, 1, 3, 4, 2)
        cla, cph = la.cumsum(-1), ph.cumsum(-1)

        # ---- level 1: every chunk at once ---------------------------------
        dl = cla.unsqueeze(-1) - cla.unsqueeze(-2)               # (B,N,M,G,t,r)
        dp = cph.unsqueeze(-1) - cph.unsqueeze(-2)
        keep = torch.tril(torch.ones(C, C, device=v.device, dtype=v.dtype))
        # clamp BEFORE exp. On kept entries (r <= t) the exponent is already <= 0
        # because cla is non-increasing, so the clamp is exact there; on masked
        # entries it prevents exp() overflowing to inf, which would then be
        # multiplied by a zero mask and yield NaN. Padding makes this reachable:
        # a zero-padded gate gives |a|^2 = tiny, hence la = -354 in fp64, hence a
        # masked exponent of +2620.
        mag = dl.clamp(max=0).exp() * keep
        vg = v.view(B, N, C, G, gs).permute(0, 1, 3, 2, 4)       # (B,N,G,C,gs)
        ire = torch.einsum("bnmgtr,bngrj->bnmgtj", mag * dp.cos(), vg)
        iim = torch.einsum("bnmgtr,bngrj->bnmgtj", mag * dp.sin(), vg)

        # ---- level 2: carry across chunks, closed form ---------------------
        f = cla[..., -1].cumsum(1)                               # (B,N,M,G)
        fp = cph[..., -1].cumsum(1)
        z0 = torch.zeros_like(f[:, :1])
        fs = torch.cat([z0, f[:, :-1]], 1)                       # fs[n] = f[n-1]
        fsp = torch.cat([z0, fp[:, :-1]], 1)
        fs_, f_ = fs.permute(0, 2, 3, 1), f.permute(0, 2, 3, 1)  # (B,M,G,N)
        fsp_, fp_ = fsp.permute(0, 2, 3, 1), fp.permute(0, 2, 3, 1)
        low = torch.tril(torch.ones(N, N, device=v.device, dtype=v.dtype), -1)
        m2 = (fs_.unsqueeze(-1) - f_.unsqueeze(-2)).clamp(max=0).exp() * low
        d2p = fsp_.unsqueeze(-1) - fp_.unsqueeze(-2)
        D2re, D2im = m2 * d2p.cos(), m2 * d2p.sin()
        sr_end = ire[..., -1, :].permute(0, 2, 3, 1, 4)           # (B,M,G,N,gs)
        si_end = iim[..., -1, :].permute(0, 2, 3, 1, 4)
        cr = (torch.einsum("bmgnp,bmgpj->bmgnj", D2re, sr_end)
              - torch.einsum("bmgnp,bmgpj->bmgnj", D2im, si_end))
        ci = (torch.einsum("bmgnp,bmgpj->bmgnj", D2re, si_end)
              + torch.einsum("bmgnp,bmgpj->bmgnj", D2im, sr_end))
        if not st.get("empty", False):
            A2r = (fs_.exp() * fsp_.cos()).unsqueeze(-1)          # (B,M,G,N,1)
            A2i = (fs_.exp() * fsp_.sin()).unsqueeze(-1)
            sin_r = st["sr"].view(B, M, G, 1, gs)
            sin_i = st["si"].view(B, M, G, 1, gs)
            cr = cr + A2r * sin_r - A2i * sin_i
            ci = ci + A2r * sin_i + A2i * sin_r
        carry_r = cr.permute(0, 3, 1, 2, 4).unsqueeze(-2)         # (B,N,M,G,1,gs)
        carry_i = ci.permute(0, 3, 1, 2, 4).unsqueeze(-2)

        # ---- combine and read ---------------------------------------------
        am = cla.exp()
        Ar = (am * cph.cos()).unsqueeze(-1)                       # (B,N,M,G,C,1)
        Ai = (am * cph.sin()).unsqueeze(-1)
        s_re = ire + Ar * carry_r - Ai * carry_i
        s_im = iim + Ar * carry_i + Ai * carry_r
        s_re = s_re.permute(0, 1, 4, 2, 3, 5).reshape(B, Tp, M, dv)
        s_im = s_im.permute(0, 1, 4, 2, 3, 5).reshape(B, Tp, M, dv)

        qn = torch.rsqrt(qr.square() + qi.square() + 1e-6)
        qr, qi = qr * qn, qi * qn
        u = torch.cat([(s_re * qr + s_im * qi).mean(2),
                       (s_im * qr - s_re * qi).mean(2)], -1)[:, :T]
        return _rms(u), {"sr": s_re[:, T - 1], "si": s_im[:, T - 1], "empty": False}

    def step_from(self, v, grh, gih, qr, qi, state):
        B = v.size(0)
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        s2 = 1.0 / math.sqrt(2)
        ar = (torch.tanh(grh) * s2)[:, :, :, None]
        ai = (torch.tanh(gih) * s2)[:, :, :, None]
        rg = state["sr"].view(B, M, G, gs)
        ig = state["si"].view(B, M, G, gs)
        sr = (ar * rg - ai * ig).reshape(B, M, dv) + v[:, None, :]
        si = (ar * ig + ai * rg).reshape(B, M, dv)
        qn = torch.rsqrt(qr.square() + qi.square() + 1e-6)
        qr, qi = qr * qn, qi * qn
        u = torch.cat([(sr * qr + si * qi).mean(1), (si * qr - sr * qi).mean(1)], -1)
        return _rms(u), {"sr": sr, "si": si, "empty": False}


# ==========================================================================  #
#  Layer -- one merged projection GEMM, h-projections by shift
# ==========================================================================  #
class Layer(SCA2Layer):
    def __init__(self, cfg: LayerCfg, c_cls=CHead, d_cls=DHead):
        super().__init__(cfg, c_cls, d_cls)
        c, dh = self.c, self.dh
        # slice widths of the merged projection, in order
        # q is absent on purpose -- it stays inside the D head's chunk loop
        self._w = (c.M, c.dv, dh.dv, dh.M * dh.G, dh.M * dh.G)
        self._nh = c.M + 2 * dh.M * dh.G      # the K/gr/gi prefix, needed on h

    # -- merged weights ------------------------------------------------------
    def _W(self):
        c, dh = self.c, self.dh
        W = torch.cat([c.K.weight, c.V.weight, dh.V.weight,
                       dh.gr.weight, dh.gi.weight], 0)
        b = torch.cat([W.new_zeros(c.M + c.dv + dh.dv),
                       dh.gr.bias, dh.gi.bias], 0)
        return W, b

    def _split(self, P):
        return P.split(self._w, -1)

    def init_state(self, B, device, dtype=torch.float32):
        c, dh = self.c, self.dh
        # projection of z_prev = 0: zero for the bias-free K, the raw bias for gr/gi
        p_prev = torch.cat([
            torch.zeros(B, c.M, device=device, dtype=dtype),
            dh.gr.bias.detach().to(device=device, dtype=dtype).expand(B, -1),
            dh.gi.bias.detach().to(device=device, dtype=dtype).expand(B, -1)], -1)
        return {"c": self.c.init_state(B, device, dtype),
                "d": self.dh.init_state(B, device, dtype),
                "p_prev": p_prev.contiguous()}

    # -- prefill -------------------------------------------------------------
    def prefill(self, x, state=None):
        B, T, _ = x.shape
        st = state if state is not None else self.init_state(B, x.device, x.dtype)
        z = self.n(x)
        W, b = self._W()
        P = F.linear(z, W, b)                                   # ONE gemm
        Kz, Vc, Vd, GRz, GIz = self._split(P)
        # h is z shifted, and every projection is linear -> shift, do not re-project
        pk, pg, pi = st["p_prev"].split((self.c.M, GRz.size(-1), GIz.size(-1)), -1)
        Kh = torch.cat([pk[:, None], Kz[:, :-1]], 1)
        GRh = torch.cat([pg[:, None], GRz[:, :-1]], 1)
        GIh = torch.cat([pi[:, None], GIz[:, :-1]], 1)

        M, G = self.dh.M, self.dh.G
        uc, cs = self.c.prefill_from(Kz, Kh, Vc, (B, T), x.device, x.dtype, st["c"])
        ud, ds = self.dh.prefill_from(z, Vd, GRh.view(B, T, M, G),
                                      GIh.view(B, T, M, G), st["d"])
        y = x + self.mix(torch.cat([uc, ud], -1))
        y = y + self.ff(self.fn(y))
        p_prev = torch.cat([Kz[:, -1], GRz[:, -1], GIz[:, -1]], -1)
        return y, {"c": cs, "d": ds, "p_prev": p_prev}

    # -- decode --------------------------------------------------------------
    def step(self, x_t, state):
        B = x_t.size(0)
        z = self.n(x_t)
        W, b = self._W()
        P = F.linear(z, W, b)                                   # ONE gemm
        Kz, Vc, Vd, GRz, GIz = self._split(P)
        pk, pg, pi = state["p_prev"].split((self.c.M, GRz.size(-1), GIz.size(-1)), -1)
        M, G = self.dh.M, self.dh.G
        uc, cs = self.c.step_from(Kz, pk, Vc, state["c"])
        ud, ds = self.dh.step_from(z, Vd, pg.view(B, M, G), pi.view(B, M, G), state["d"])
        y = x_t + self.mix(torch.cat([uc, ud], -1))
        y = y + self.ff(self.fn(y))
        return y, {"c": cs, "d": ds,
                   "p_prev": torch.cat([Kz, GRz, GIz], -1)}


class DHead4(DHead):
    CHUNK = 4
class DHead8(DHead):
    CHUNK = 8
class DHead16(DHead):
    CHUNK = 16


STATUS = "regression"
NOTE = "one merged projection gemm, h-projections by shift"
