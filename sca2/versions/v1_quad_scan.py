"""
Version v1 -- "quad + scan".

  C head: the reference's cumulative-sum form rewritten as causal attention with
          a structured complex kernel (DERIVATION.md section 1). 55x forward.
  D head: the reference's T-step python loop rewritten as a chunked log-space
          scan (section 2). 4x forward.

Measured on RTX 2070, B=8 T=128 fp32, whole layer vs v0: prefill 21.5x,
training step 21.1x, graph decode 10.9x.

Exports the version interface: CHead, DHead, Layer, plus DHead variants for the
chunk-size sweep and the two measured-and-rejected experiments at the bottom.
"""
import math
import torch
import torch.nn as nn

from ..ref import _gated_out, CHeadBase, DHeadBase, CHeadRef, DHeadRef, SCA2Layer, _rms

_MASK = {}


def causal_mask(T, device):
    """Cached strictly-upper-triangular mask (True = must be zeroed)."""
    k = (T, str(device))
    m = _MASK.get(k)
    if m is None:
        m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
        _MASK[k] = m
    return m

# ==========================================================================  #
#  C head, quadratic scalar-kernel form
# ==========================================================================  #
class CHeadQuad(CHeadBase):
    r"""The reference C head is *exactly* causal attention with a structured
    complex kernel, because the read-out contracts over M with no dv dependence:

        (rr + i.ii)[b,t,m,j] = sum_{s<=t} v[s,j] . e^{i (pw[s,m] - pq[t,m])}
        u_re + i.u_im        = mean_m (rr + i.ii) . (wr + i.wi)
                             = sum_{s<=t} v[s,j] . kappa[b,t,s]

    so kappa[b,t,s] = (1/M) sum_m (wr+i.wi)_m e^{i(pw[s,m]-pq[t,m])} is a SCALAR
    score, independent of the value dimension. Expanding cos/sin of the phase
    difference turns both its real and imaginary parts into a single real matmul
    over a 2M-dim feature map:

        A  = wr.cos(pw) - wi.sin(pw)       Bm = wr.sin(pw) + wi.cos(pw)
        Kre = [cos(pq), sin(pq)] @ [A , Bm]^T / M
        Kim = [cos(pq), sin(pq)] @ [Bm, -A ]^T / M
        u   = concat(Kre @ v, Kim @ v)

    Cost B.T^2.(2M+dv) instead of B.T.M.dv, but the B.T.M.dv *activations* (six
    of them, all retained for backward) collapse to two B.T.T score matrices.
    """

    def init_state(self, B, device, dtype):
        st = super().init_state(B, device, dtype)
        # 0-dim tensor, not a python int: an int makes torch.compile specialize on
        # the position and recompile at every decode step.
        st["pos"] = torch.zeros((), device=device, dtype=dtype)
        st["empty"] = True
        return st

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def _phases(self, z, h, p0):
        T = z.size(1)
        p = (torch.arange(T, device=z.device, dtype=z.dtype) + p0)[:, None] * self.omega
        return self.K(h) * self.theta + p, self.K(z) * self.theta + p

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        pw, pq = self._phases(z, h, st["pos"])

        cw, sw = pw.cos(), pw.sin()
        cq, sq = pq.cos(), pq.sin()
        A = self.wr * cw - self.wi * sw
        Bm = self.wr * sw + self.wi * cw

        Fq = torch.cat([cq, sq], -1)                      # (B,T,2M)
        Fk = torch.cat([torch.cat([A, Bm], -1),
                        torch.cat([Bm, -A], -1)], 1)      # (B,2T,2M)
        K2 = (Fq @ Fk.transpose(1, 2)).view(B, T, 2, T)   # (B,T,2,T)
        K2 = K2.masked_fill(causal_mask(T, z.device)[None, :, None, :], 0)

        v = self.V(z)                                     # (B,T,dv)
        o = (K2.reshape(B, T * 2, T) @ v).view(B, T, 2 * self.dv) / self.M

        if not st.get("empty", False):
            sr0, si0 = st["sr"], st["si"]
            c1 = self.wr * cq + self.wi * sq              # (B,T,M)
            c2 = self.wr * sq - self.wi * cq
            r1 = torch.einsum("btm,bmj->btj", c1, sr0)
            r2 = torch.einsum("btm,bmj->btj", c2, si0)
            i1 = torch.einsum("btm,bmj->btj", c2, sr0)
            i2 = torch.einsum("btm,bmj->btj", c1, si0)
            o = o + torch.cat([r1 + r2, i2 - i1], -1) / self.M

        # closing state: sum_s v[s] e^{i pw[s]}  (+ incoming)
        sr = torch.einsum("btm,btj->bmj", cw, v) + st["sr"]
        si = torch.einsum("btm,btj->bmj", sw, v) + st["si"]
        return _gated_out(self, o, z), {"sr": sr, "si": si, "pos": st["pos"] + T, "empty": False}

    def step(self, z_t, h_t, state):
        p = state["pos"]
        pw = self.K(h_t) * self.theta + p * self.omega
        pq = self.K(z_t) * self.theta + p * self.omega
        v = self.V(z_t)
        sr = torch.addcmul(state["sr"], v[:, None, :], pw.cos()[:, :, None])
        si = torch.addcmul(state["si"], v[:, None, :], pw.sin()[:, :, None])
        cq, sq = pq.cos(), pq.sin()
        c1 = self.wr * cq + self.wi * sq                  # (B,M)
        c2 = self.wr * sq - self.wi * cq
        # u_re = mean_m(sr.c1 + si.c2), u_im = mean_m(si.c1 - sr.c2)
        cc = torch.stack([c1, c2], 1)                     # (B,2,M)
        ss = torch.stack([sr, si], 1)                     # (B,2,M,dv)
        m = torch.einsum("bam,bcmj->bacj", cc, ss) / self.M
        u = torch.cat([m[:, 0, 0] + m[:, 1, 1], m[:, 0, 1] - m[:, 1, 0]], -1)
        return _gated_out(self, u, z_t), {"sr": sr, "si": si, "pos": state["pos"] + 1, "empty": False}


# ==========================================================================  #
#  D head, chunked parallel scan
# ==========================================================================  #
class DHeadChunk(DHeadBase):
    r"""The reference D head runs a T-step Python loop (~15 kernel launches per
    token). Its recurrence is a diagonal complex linear scan

        s[t] = a[t] . s[t-1] + v[t]        a = gr + i.gi,  |a| <= 1,  v real

    which unrolls in closed form over a chunk of length C:

        s[t] = A[t] . s_in + sum_{r<=t} D[t,r] . v[r]
        A[t] = prod_{u<=t} a[u]            D[t,r] = prod_{r<u<=t} a[u]

    D is built in log-magnitude / phase space, so the products become cumsum
    differences: exp(cla[t]-cla[r]) with cla non-increasing, hence always <= 1 --
    no overflow, and no division by a decayed prefix (which is what makes the
    naive A[t]/A[r] form blow up). Gradients stay bounded because d(la)/d(gr) =
    gr/|a|^2 is always multiplied by a D factor that itself carries |a[u]|.

    The T-step loop becomes ceil(T/C) matmul-shaped iterations. The last chunk is
    ragged rather than padded, so the closing state comes from the last real
    position.
    """
    CHUNK = 32

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def _gates(self, h, B, T):
        gr = torch.tanh(self.gr(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        gi = torch.tanh(self.gi(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        return gr, gi

    def _q(self, z_chunk, B, C):
        qr = self.qr(z_chunk).view(B, C, self.M, self.dv)
        qi = self.qi(z_chunk).view(B, C, self.M, self.dv)
        qn = torch.sqrt(qr.square() + qi.square() + 1e-6)
        return qr / qn, qi / qn

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        tiny = torch.finfo(z.dtype).tiny

        v = self.V(z)
        gr, gi = self._gates(h, B, T)
        la = 0.5 * torch.log(gr.square() + gi.square() + tiny)   # (B,T,M,G)
        ph = torch.atan2(gi, gr)

        sr, si = st["sr"], st["si"]
        C = min(self.CHUNK, T)
        out = []
        for s0 in range(0, T, C):
            s1 = min(s0 + C, T)
            c = s1 - s0
            cla = la[:, s0:s1].cumsum(1)                          # (B,c,M,G)
            cph = ph[:, s0:s1].cumsum(1)

            # D[t,r] = prod_{r<u<=t} a[u] = exp(cla[t]-cla[r]) e^{i(cph[t]-cph[r])}
            dl = cla.unsqueeze(1) - cla.unsqueeze(2)              # (B,c_r,c_t,M,G) -> see perm
            dp = cph.unsqueeze(1) - cph.unsqueeze(2)
            dl = dl.permute(0, 3, 4, 2, 1)                        # (B,M,G,t,r)
            dp = dp.permute(0, 3, 4, 2, 1)
            keep = torch.tril(torch.ones(c, c, device=z.device, dtype=torch.bool))
            mag = torch.where(keep, dl.exp(), torch.zeros((), dtype=z.dtype, device=z.device))
            Dre, Dim = mag * dp.cos(), mag * dp.sin()

            vc = v[:, s0:s1].view(B, c, G, gs).permute(0, 2, 1, 3)   # (B,G,c,gs)
            ire = torch.einsum("bmgtr,bgrj->bmgtj", Dre, vc)
            iim = torch.einsum("bmgtr,bgrj->bmgtj", Dim, vc)

            # A[t] . s_in
            Amag = cla.exp()                                      # (B,c,M,G)
            Are = (Amag * cph.cos()).permute(0, 2, 3, 1)[..., None]   # (B,M,G,c,1)
            Aim = (Amag * cph.sin()).permute(0, 2, 3, 1)[..., None]
            sr_in = sr.view(B, M, G, 1, gs)
            si_in = si.view(B, M, G, 1, gs)
            cr = Are * sr_in - Aim * si_in + ire                  # (B,M,G,c,gs)
            ci = Are * si_in + Aim * sr_in + iim

            s_re = cr.permute(0, 3, 1, 2, 4).reshape(B, c, M, dv)   # (B,c,M,dv)
            s_im = ci.permute(0, 3, 1, 2, 4).reshape(B, c, M, dv)
            qr, qi = self._q(z[:, s0:s1], B, c)
            out.append(torch.cat([(s_re * qr + s_im * qi).mean(2),
                                  (s_im * qr - s_re * qi).mean(2)], -1))
            sr = s_re[:, -1]
            si = s_im[:, -1]
        return _gated_out(self, torch.cat(out, 1), z), {"sr": sr, "si": si}

    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        gs, M, dv = self.gs, self.M, self.dv
        gr, gi = self._gates(h_t[:, None], B, 1)
        ar, ai = gr[:, 0, :, :, None], gi[:, 0, :, :, None]
        rg = state["sr"].view(B, M, self.G, gs)
        ig = state["si"].view(B, M, self.G, gs)
        sr = (ar * rg - ai * ig).reshape(B, M, dv) + self.V(z_t)[:, None, :]
        si = (ar * ig + ai * rg).reshape(B, M, dv)
        qr, qi = self._q(z_t[:, None], B, 1)
        qr, qi = qr[:, 0], qi[:, 0]
        u = torch.cat([(sr * qr + si * qi).mean(1), (si * qr - sr * qi).mean(1)], -1)
        return _gated_out(self, u, z_t), {"sr": sr, "si": si}




class DHeadScan(DHeadBase):
    r"""DHeadChunk, with the layout fixed.

    The chunked form's cost is dominated not by its matmuls (B.M.G.C^2.gs MACs is
    nothing) but by memory traffic on the (B,M,G,C,C) decay matrices. Two things
    matter, in this order:

      * permute BEFORE the outer difference, never after. Building dl as
        cla.permute(...) then subtracting produces a strided read of a
        B.M.G.C.C tensor; slicing/cumsum on an already-(B,M,G,C) layout and
        then broadcasting produces the same values contiguously, from a
        tensor M.G/C times smaller.
      * total decay traffic is B.M.G.C.T, i.e. LINEAR in the chunk size, while
        launch count is T/C. C therefore trades bandwidth against launches and
        has a real optimum (swept in bench_scan.py).
    """
    CHUNK = 16
    HOIST_Q = False   # one big q GEMM for the whole sequence vs one per chunk

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        gr = torch.tanh(self.gr(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        gi = torch.tanh(self.gi(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        return self.prefill_core(z, self.V(z), gr, gi, state)

    def prefill_core(self, z, v, gr, gi, state=None):
        """The scan itself. `gr`/`gi` are POST-activation (B,T,M,G).

        Split out so a caller that already computed the projections (v2's merged
        gemm) can reuse this body instead of forking it. `z` is still needed
        because q is projected PER CHUNK on purpose -- see v1's rejected
        HOIST_Q experiment and v2's docstring.
        """
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        tiny = torch.finfo(z.dtype).tiny

        vg = v.view(B, T, G, gs).permute(0, 2, 1, 3).contiguous()      # (B,G,T,gs)
        # permuted while still B.T.M.G -- M.G/C times smaller than the decay matrix
        lap = (0.5 * torch.log(gr.square() + gi.square() + tiny)).permute(0, 2, 3, 1)
        php = torch.atan2(gi, gr).permute(0, 2, 3, 1)

        # The q projection is per-token and its normalization is elementwise, so
        # it can be one (B.T x d) @ (d x M.dv) GEMM instead of T/C small ones.
        # Costs 2.B.T.M.dv of activation; at chunk 8 that buys back 30 tiny GEMMs.
        QR = QI = None
        if self.HOIST_Q:
            QR = self.qr(z).view(B, T, M, dv)
            QI = self.qi(z).view(B, T, M, dv)
            qn = torch.rsqrt(QR.square() + QI.square() + 1e-6)
            QR, QI = QR * qn, QI * qn

        sr, si = st["sr"], st["si"]
        C = min(self.CHUNK, T)
        out = []
        for s0 in range(0, T, C):
            s1 = min(s0 + C, T)
            c = s1 - s0
            cla = lap[..., s0:s1].cumsum(-1)                            # (B,M,G,c)
            cph = php[..., s0:s1].cumsum(-1)
            dl = cla.unsqueeze(-1) - cla.unsqueeze(-2)                  # (B,M,G,t,r)
            dp = cph.unsqueeze(-1) - cph.unsqueeze(-2)
            keep = torch.tril(torch.ones(c, c, device=z.device, dtype=z.dtype))
            # clamp before exp: exact on kept entries (exponent <= 0 there), and
            # stops a masked +inf from becoming inf*0 = NaN when a gate
            # saturates to |a| = 0 and la drops to log(tiny). See v2_fused.
            mag = dl.clamp(max=0).exp() * keep
            vc = vg[:, :, s0:s1]                                        # (B,G,c,gs)
            ire = torch.einsum("bmgtr,bgrj->bmgtj", mag * dp.cos(), vc)
            iim = torch.einsum("bmgtr,bgrj->bmgtj", mag * dp.sin(), vc)

            am = cla.exp()
            Are = (am * cph.cos()).unsqueeze(-1)                        # (B,M,G,c,1)
            Aim = (am * cph.sin()).unsqueeze(-1)
            sr_in = sr.view(B, M, G, 1, gs)
            si_in = si.view(B, M, G, 1, gs)
            cr = torch.addcmul(ire, Are, sr_in) - Aim * si_in           # (B,M,G,c,gs)
            ci = torch.addcmul(iim, Are, si_in) + Aim * sr_in

            s_re = cr.permute(0, 3, 1, 2, 4).reshape(B, c, M, dv)
            s_im = ci.permute(0, 3, 1, 2, 4).reshape(B, c, M, dv)
            if QR is not None:
                qr, qi = QR[:, s0:s1], QI[:, s0:s1]
            else:
                zc = z[:, s0:s1]
                qr = self.qr(zc).view(B, c, M, dv)
                qi = self.qi(zc).view(B, c, M, dv)
                qn = torch.rsqrt(qr.square() + qi.square() + 1e-6)
                qr, qi = qr * qn, qi * qn
            out.append(torch.cat([(s_re * qr + s_im * qi).mean(2),
                                  (s_im * qr - s_re * qi).mean(2)], -1))
            sr, si = s_re[:, -1], s_im[:, -1]
        return _gated_out(self, torch.cat(out, 1), z), {"sr": sr, "si": si}

    _gates = DHeadChunk._gates
    _q = DHeadChunk._q
    step = DHeadChunk.step


class DHeadScan8(DHeadScan):
    CHUNK = 8
class DHeadScan16(DHeadScan):
    CHUNK = 16
class DHeadScan32(DHeadScan):
    CHUNK = 32
class DHeadScan64(DHeadScan):
    CHUNK = 64
class DHeadScan128(DHeadScan):
    CHUNK = 128


class DHeadScan4(DHeadScan):
    CHUNK = 4

# --------------------------------------------------------------------------- #
#  Measured and REJECTED. Kept so the regression stays visible rather than
#  becoming folklore.
# --------------------------------------------------------------------------- #
#  Hoisting q out of the chunk loop replaces T/C small GEMMs with two big ones.
#  It wins ~10% on prefill (2.06 vs 2.36 ms) and loses 5x on the backward
#  (train 54.8 vs 10.6 ms): 16 narrow python slices of a retained
#  2.B.T.M.dv tensor each scatter-add in the backward. v2 gets the same GEMM
#  merge for free by making the chunk axis a real tensor axis, so there is no
#  python slicing to pay for.
class DHeadScanQ4(DHeadScan):
    CHUNK, HOIST_Q = 4, True


class DHeadScanQ8(DHeadScan):
    CHUNK, HOIST_Q = 8, True


class DHeadScanQ16(DHeadScan):
    CHUNK, HOIST_Q = 16, True


# --------------------------------------------------------------------------- #
#  version interface
# --------------------------------------------------------------------------- #
CHead = CHeadQuad
DHead = DHeadScan8          # chunk 8 measured best under torch.compile
Layer = SCA2Layer
STATUS = "recommended"
NOTE = "quad C head + chunked log-space D scan"
