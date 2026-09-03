"""
D head as a WEIGHTED CUMSUM -- the original SeqCond formulation.

Taken from trickstr-ai/nautile-370m (`modeling_seqcond.py`), whose temporal
machinery is fundamentally cheaper than a gated recurrence, and the reason is one
line of algebra:

    gated recurrence   s[t] = a[t] . s[t-1] + v[t]
    weighted cumsum    S[t] = sum_{s<=t} w[s] . v[s]  /  sum_{s<=t} w[s]

A multiplicative gate on the STATE does not commute with the prefix sum, so it
forces a scan -- chunked decay matrices, a sequential carry, ~320 kernels, 0.1%
of peak. A multiplicative weight on the VALUE does commute, so the whole temporal
structure collapses into one `torch.cumsum` over a contiguous tensor. Data
dependence survives: `w` depends on the input. It just enters where it does not
break the parallel form.

The original keeps a recency prior too, and notably NOT as a relative decay:

    log_tw[t] = -softplus(slope) * (max_len - 1 - t)        "decay" heads
    log_tw[t] = -softplus(slope) * t                        "anchor" heads

These are ABSOLUTE position weights, identical for every query position, so no
gamma^(-s) ever appears and nothing explodes -- the factorization trap that makes
relative geometric decay awkward is avoided rather than fought. Decay heads
weight the recent prefix, anchor heads its beginning; together they span both
ends.

Two details from the original that our own A/Bs independently support: theta is
initialized geomspace(0.001, 3.0) -- a GEOMETRIC frequency spread, like the
`rope` grid that beat `dft` by 0.113 over three seeds -- and the content phase is
bounded, phi = (k / (1 + |k|)) . theta, rather than the raw K(h).theta of the
benchmark's SCA2.

The read-out keeps our separable query, which is orthogonal to all of this and
worth -21% time / -58% parameters on its own.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import DHeadBase, _rms
from .registry import register
from .versions.v1_quad_scan import CHeadQuad
from .compiled import wrap as _cw


class DHeadCumsum(DHeadBase):
    def __init__(self, d, M=16, G=8, dv=None, max_len=None, delta_rule=False,
                 anchor_frac=0.5):
        nn.Module.__init__(self)
        self.d, self.M, self.G = d, M, G
        self.dv = d // 2 if dv is None else dv
        self.gs = self.dv // G
        # MUST match the real context. The recency weight is
        # exp(-slope . (max_len - 1 - pos)); if max_len overshoots, every
        # position saturates the 1e-4 clamp and the steep heads collapse into
        # uniform averages -- the prior is silently destroyed, not just scaled.
        self.max_len = max_len or 512
        self.n_anchor = max(1, int(M * anchor_frac))

        self.V = nn.Linear(d, self.dv, False)
        self.score = nn.Linear(d, M)                  # content weight, one per m
        self.K = nn.Linear(d, M, False)               # content phase
        th = np.geomspace(0.001, 3.0, M).astype(np.float32)
        self.theta = nn.Parameter(torch.from_numpy(th))
        self.slopes = nn.Parameter(torch.from_numpy(
            np.log(np.exp(np.geomspace(0.001, 0.1, M)) - 1).astype(np.float32)))
        self.qa_r = nn.Linear(d, M, False); self.qa_i = nn.Linear(d, M, False)
        self.qb_r = nn.Linear(d, self.dv, False); self.qb_i = nn.Linear(d, self.dv, False)

    def _log_time_weight(self, T, pos0, device, dtype):
        p = torch.arange(T, device=device, dtype=dtype)[:, None] + pos0
        s = F.softplus(self.slopes.to(dtype))[None, :]
        w = -s * (self.max_len - 1 - p).clamp(min=0.0)
        if self.n_anchor:
            w = torch.cat([-s[:, :self.n_anchor] * p, w[:, self.n_anchor:]], -1)
        return w

    def _weights(self, z, pos0):
        sc = F.softplus(self.score(z))
        ltw = self._log_time_weight(z.size(1), pos0, z.device, z.dtype)[None]
        return (sc * ltw.exp()).clamp(1e-4, 5000.0)

    def _phase(self, z):
        k = self.K(z)
        return (k / (1.0 + k.abs())) * self.theta.to(z.dtype)

    def _alpha_beta(self, z):
        ar, ai = self.qa_r(z), self.qa_i(z)
        br, bi = self.qb_r(z), self.qb_i(z)
        an = torch.rsqrt(ar.square() + ai.square() + 1e-6)
        bn = torch.rsqrt(br.square() + bi.square() + 1e-6)
        return ar * an, ai * an, br * bn, bi * bn

    def _read(self, s_re, s_im, ar, ai, br, bi, mdim):
        P = (s_re * ar.unsqueeze(-1) + s_im * ai.unsqueeze(-1)).mean(mdim)
        Q = (s_im * ar.unsqueeze(-1) - s_re * ai.unsqueeze(-1)).mean(mdim)
        return torch.cat([br * P + bi * Q, br * Q - bi * P], -1)

    def init_state(self, B, device, dtype):
        return {"sr": torch.zeros(B, self.M, self.dv, device=device, dtype=dtype),
                "si": torch.zeros(B, self.M, self.dv, device=device, dtype=dtype),
                "den": torch.zeros(B, self.M, device=device, dtype=dtype),
                "pos": torch.zeros((), device=device, dtype=dtype)}

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        w = self._weights(z, st["pos"])
        phi = self._phase(z)
        v = self.V(z)
        wr = (w * phi.cos()).unsqueeze(-1) * v.unsqueeze(-2)
        wi = (w * phi.sin()).unsqueeze(-1) * v.unsqueeze(-2)
        s_re = wr.cumsum(1) + st["sr"].unsqueeze(1)
        s_im = wi.cumsum(1) + st["si"].unsqueeze(1)
        den = w.cumsum(1) + st["den"].unsqueeze(1)
        inv = den.clamp(min=1e-4).reciprocal().unsqueeze(-1)
        ar, ai, br, bi = self._alpha_beta(z)
        u = self._read(s_re * inv, s_im * inv, ar, ai, br, bi, 2)
        return _rms(u), {"sr": s_re[:, -1], "si": s_im[:, -1], "den": den[:, -1],
                         "pos": st["pos"] + T}

    def step(self, z_t, h_t, state):
        w = self._weights(z_t[:, None], state["pos"])[:, 0]
        phi = self._phase(z_t[:, None])[:, 0]
        v = self.V(z_t)
        sr = state["sr"] + (w * phi.cos()).unsqueeze(-1) * v.unsqueeze(-2)
        si = state["si"] + (w * phi.sin()).unsqueeze(-1) * v.unsqueeze(-2)
        den = state["den"] + w
        inv = den.clamp(min=1e-4).reciprocal().unsqueeze(-1)
        ar, ai, br, bi = self._alpha_beta(z_t)
        u = self._read(sr * inv, si * inv, ar, ai, br, bi, 1)
        return _rms(u), {"sr": sr, "si": si, "den": den, "pos": state["pos"] + 1}


register("cs", CHeadQuad, DHeadCumsum, arch=True,
         note="weighted-cumsum D head (original SeqCond temporal form)")
register("cs_cc", CHeadQuad, DHeadCumsum, arch=True, wrap=_cw,
         note="weighted-cumsum D head + compile")


class DHeadCumsumRel(DHeadCumsum):
    r"""Weighted cumsum with NO declared maximum length.

    The original writes the recency prior as an absolute position weight,
    `exp(-slope . (max_len - 1 - t))`, so the numbers stay <= 1. Because the
    read-out divides by the accumulated denominator, a per-head constant cancels
    exactly, and

        exp(-s (L-1-t)) = exp(-s (L-1)) . exp(s t)

    is that constant times something L-free. So the FUNCTION never depended on
    the context length -- only the arithmetic did, and specifically the
    `clamp(1e-4, 5000)`, which is not scale-invariant. Grow L and the clamp
    starts biting different positions, the prior silently changes, and the model
    has to be retrained.

    Here the reference is CHUNK-LOCAL instead of global. For a decay head over a
    chunk [a, b), with c = softplus(score):

        N[t] = exp(-s(t-a)) . ( N_carry + cumsum( exp(s(u-a)) c[u] v[u] )[t] )

    The growing factor is `exp(s (u-a))`, bounded by `exp(s C)` -- bounded by the
    CHUNK SIZE, never by the sequence length. C is chosen from the steepest slope
    (s.C <= 15), so it is a property of the learned decay rates, not of the
    context. The carry is the true state value, so no rescaling is needed either.

    Anchor heads are already length-free: `exp(-s u)` is absolute from position
    0, one plain cumsum, and the underflow of distant terms is exactly right --
    an anchor is not supposed to see recent tokens.

    Consequence: the context length can be changed at any time, before or after
    training, without touching a single learned weight.
    """
    LOG_HEADROOM = 15.0

    def _chunk(self, T):
        s_max = float(F.softplus(self.slopes.detach()).max())
        c = int(self.LOG_HEADROOM / max(s_max, 1e-6))
        return max(1, min(T, c))

    def _parts(self, z, pos0):
        """(content weight c, phase) -- both length-independent."""
        return F.softplus(self.score(z)), self._phase(z)

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        na = self.n_anchor
        c, phi = self._parts(z, st["pos"])
        v = self.V(z)
        s = F.softplus(self.slopes.to(z.dtype))                      # (M,)
        p = torch.arange(T, device=z.device, dtype=z.dtype) + st["pos"]

        # anchor heads: absolute weight from position 0, one cumsum
        w = torch.empty_like(c)
        if na:
            w[..., :na] = c[..., :na] * torch.exp(-s[:na] * p[:, None])

        # decay heads: chunk-local reference, so nothing scales with T
        C = self._chunk(T)
        sd = s[na:]
        if sd.numel():
            wd = c[..., na:]
            outs = []
            for a in range(0, T, C):
                loc = (p[a:a + C] - p[a])[:, None]                   # 0..C-1
                outs.append(wd[:, a:a + C] * torch.exp(sd * loc))
            w[..., na:] = torch.cat(outs, 1)

        wr = (w * phi.cos()).unsqueeze(-1) * v.unsqueeze(-2)
        wi = (w * phi.sin()).unsqueeze(-1) * v.unsqueeze(-2)
        sr, si, den = st["sr"], st["si"], st["den"]
        SR, SI, DEN = [], [], []
        for a in range(0, T, C):
            b = min(a + C, T)
            loc = (p[a:b] - p[a])[:, None]
            dec = torch.ones(b - a, self.M, device=z.device, dtype=z.dtype)
            if sd.numel():
                dec[:, na:] = torch.exp(-sd * loc)                   # (c,M)
            d3 = dec.unsqueeze(-1)[None]                             # (1,c,M,1)
            # N[t] = dec[t-a] . ( one . N[a-1] + cumsum_{u=a..t} exp(s(u-a)) c v )
            # `one` is exp(-s) for decay heads -- from a-1 to a is ONE step, and
            # dropping it is an off-by-one that vanishes on the first chunk
            # (empty carry) and only shows up once T exceeds the chunk size.
            one = torch.ones(self.M, device=z.device, dtype=z.dtype)
            if sd.numel():
                one[na:] = torch.exp(-sd)
            csr = wr[:, a:b].cumsum(1) + (sr * one.unsqueeze(-1)).unsqueeze(1)
            csi = wi[:, a:b].cumsum(1) + (si * one.unsqueeze(-1)).unsqueeze(1)
            cde = w[:, a:b].cumsum(1) + (den * one).unsqueeze(1)
            SR.append(csr * d3); SI.append(csi * d3); DEN.append(cde * dec[None])
            # The carry is the TRUE state N[b-1]: the next chunk re-references
            # to its own start, so it needs no rescaling. (An earlier version
            # divided by the last decay factor here -- wrong, and invisible at
            # T <= chunk size, which is why the shape sweep alone did not catch
            # it. See test_chunks below.)
            sr, si, den = SR[-1][:, -1], SI[-1][:, -1], DEN[-1][:, -1]
        s_re, s_im, dn = torch.cat(SR, 1), torch.cat(SI, 1), torch.cat(DEN, 1)
        inv = dn.clamp(min=1e-12).reciprocal().unsqueeze(-1)
        ar, ai, br, bi = self._alpha_beta(z)
        u = self._read(s_re * inv, s_im * inv, ar, ai, br, bi, 2)
        return _rms(u), {"sr": sr, "si": si, "den": den, "pos": st["pos"] + T}

    def step(self, z_t, h_t, state):
        na = self.n_anchor
        s = F.softplus(self.slopes.to(z_t.dtype))
        c = F.softplus(self.score(z_t))
        phi = self._phase(z_t[:, None])[:, 0]
        v = self.V(z_t)
        p = state["pos"]
        w = c.clone()
        if na:
            w[..., :na] = c[..., :na] * torch.exp(-s[:na] * p)
        dec = torch.ones(self.M, device=z_t.device, dtype=z_t.dtype)
        dec[na:] = torch.exp(-s[na:])                    # one step of relative decay
        sr = state["sr"] * dec.unsqueeze(-1) + (w * phi.cos()).unsqueeze(-1) * v.unsqueeze(-2)
        si = state["si"] * dec.unsqueeze(-1) + (w * phi.sin()).unsqueeze(-1) * v.unsqueeze(-2)
        den = state["den"] * dec + w
        inv = den.clamp(min=1e-12).reciprocal().unsqueeze(-1)
        ar, ai, br, bi = self._alpha_beta(z_t)
        u = self._read(sr * inv, si * inv, ar, ai, br, bi, 1)
        return _rms(u), {"sr": sr, "si": si, "den": den, "pos": state["pos"] + 1}


register("csr", CHeadQuad, DHeadCumsumRel, arch=True,
         note="weighted cumsum, chunk-local reference: NO max_len dependence")
register("csr_cc", CHeadQuad, DHeadCumsumRel, arch=True, wrap=_cw,
         note="length-free weighted cumsum + compile")
