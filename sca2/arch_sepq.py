"""
Architecture candidate: separable D-head query.

NOT a version -- this changes parameters, so it is not drop-in and cannot be
iso-checked against the reference. It gets its own A/B (`python -m sca2.ab_sepq`)
and its own self-consistency check (`python -m sca2.iso <name> --self`).

Why. Three independent measurements point at the same term:
  * dh.qr/qi are 53.9% of layer MACs and 62.6% of its parameters.
  * bench_params: Md=2 -> Md=16 costs 230k parameters and buys 0.064 nats
    (0.043 with rope). The projections are heavily over-provisioned.
  * They carry M.dv = 1024 degrees of freedom per token -- exactly the size of
    the state they read -- while the state's m-dependence comes only from the
    gate phase, so those dof cannot be used independently.

What. Factor the complex query as an outer product, q[m,j] = alpha[m].beta[j]:

    qr[m,j] = ar[m].br[j] - ai[m].bi[j]
    qi[m,j] = ar[m].bi[j] + ai[m].br[j]

The per-element normalization factors exactly, because |alpha.beta| =
|alpha|.|beta|: normalize alpha over m and beta over j separately and the
product is already unit-modulus. The read-out then collapses:

    A[j] = mean_m(sr.ar + si.ai)      B[j] = mean_m(si.ar - sr.ai)
    u_re = br.A + bi.B                u_im = br.B - bi.A

so `m` is contracted BEFORE the query is applied, and no (B,T,M,dv) query
tensor is ever materialized -- which is what made every previous q experiment
(x_hoistq, x_loopfree, v2) regress on the backward.

Cost at d=128, Md=16: 20k parameters instead of 262k and 21M MACs instead of
268M, both 12.8x down.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import _gated_out, DHeadBase, SCA2Layer, _rms
from .registry import register
from .versions.v1_quad_scan import CHeadQuad


class DHeadSepQ(DHeadBase):
    CHUNK = 8

    def __init__(self, d, M=16, G=8, dv=None, max_len=None, delta_rule=False,
                 gated_read=False):
        nn.Module.__init__(self)
        self.d, self.M, self.G, self.dv = d, M, G, (d // 2 if dv is None else dv)
        self.max_len = max_len
        self.gated_read = gated_read
        if gated_read:
            self.rgate = nn.Linear(d, 2 * self.dv, bias=False)
        # False | "exact" (per-token, sequential) | "chunk" (lagged, full speed)
        self.delta_rule = "exact" if delta_rule is True else delta_rule
        if self.delta_rule:
            # write strength, as in the delta rule's `beta`
            self.wbeta = nn.Linear(d, 1)
        assert self.dv % G == 0
        self.gs = self.dv // G
        self.V = nn.Linear(d, self.dv, False)
        self.gr = nn.Linear(d, M * G)
        self.gi = nn.Linear(d, M * G)
        # the query, factored: alpha over M, beta over dv
        self.qa_r = nn.Linear(d, M, False)
        self.qa_i = nn.Linear(d, M, False)
        self.qb_r = nn.Linear(d, self.dv, False)
        self.qb_i = nn.Linear(d, self.dv, False)

    # ---- gate ------------------------------------------------------------- #
    def _log_polar(self, h, B, T):
        """(log|a|, arg a), each (B,M,G,T). Cartesian tanh parameterization."""
        tiny = torch.finfo(h.dtype).tiny
        gr = torch.tanh(self.gr(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        gi = torch.tanh(self.gi(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        la = 0.5 * torch.log(gr.square() + gi.square() + tiny)
        return la.permute(0, 2, 3, 1), torch.atan2(gi, gr).permute(0, 2, 3, 1)

    def _gate_cartesian(self, h, B):
        """(Re a, Im a), each (B,M,G), for one decode step."""
        s2 = 1.0 / math.sqrt(2)
        return (torch.tanh(self.gr(h)).view(B, self.M, self.G) * s2,
                torch.tanh(self.gi(h)).view(B, self.M, self.G) * s2)

    # ---- read-out --------------------------------------------------------- #
    def _alpha_beta(self, z):
        ar, ai = self.qa_r(z), self.qa_i(z)
        br, bi = self.qb_r(z), self.qb_i(z)
        an = torch.rsqrt(ar.square() + ai.square() + 1e-6)
        bn = torch.rsqrt(br.square() + bi.square() + 1e-6)
        return ar * an, ai * an, br * bn, bi * bn

    def _read(self, s_re, s_im, ar, ai, br, bi, mdim):
        """s_*: (..., M, dv); ar/ai: (..., M); br/bi: (..., dv)."""
        A = (s_re * ar.unsqueeze(-1) + s_im * ai.unsqueeze(-1)).mean(mdim)
        Bv = (s_im * ar.unsqueeze(-1) - s_re * ai.unsqueeze(-1)).mean(mdim)
        return torch.cat([br * A + bi * Bv, br * Bv - bi * A], -1)

    # ---- prefill / step --------------------------------------------------- #
    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        if self.delta_rule == "exact":
            return self._prefill_delta(z, h, state)
        B, T, _ = z.shape
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        v = self.V(z)
        vg = v.view(B, T, G, gs).permute(0, 2, 1, 3).contiguous()
        lap, php = self._log_polar(h, B, T)
        AR, AI, BR, BI = self._alpha_beta(z)

        sr, si = st["sr"], st["si"]
        C = min(self.CHUNK, T)
        out = []
        for s0 in range(0, T, C):
            s1 = min(s0 + C, T); c = s1 - s0
            cla = lap[..., s0:s1].cumsum(-1)
            cph = php[..., s0:s1].cumsum(-1)
            dl = cla.unsqueeze(-1) - cla.unsqueeze(-2)
            dp = cph.unsqueeze(-1) - cph.unsqueeze(-2)
            keep = torch.tril(torch.ones(c, c, device=z.device, dtype=z.dtype))
            mag = dl.clamp(max=0).exp() * keep
            if self.delta_rule == "chunk":
                vcorr = self._delta_lag(v[:, s0:s1], sr, si, AR[:, s0:s1], AI[:, s0:s1],
                                        BR[:, s0:s1], BI[:, s0:s1], z[:, s0:s1])
                vc = vcorr.view(B, c, G, gs).permute(0, 2, 1, 3)
            else:
                vc = vg[:, :, s0:s1]
            ire = torch.einsum("bmgtr,bgrj->bmgtj", mag * dp.cos(), vc)
            iim = torch.einsum("bmgtr,bgrj->bmgtj", mag * dp.sin(), vc)
            am = cla.exp()
            Are = (am * cph.cos()).unsqueeze(-1)
            Aim = (am * cph.sin()).unsqueeze(-1)
            cr = torch.addcmul(ire, Are, sr.view(B, M, G, 1, gs)) - Aim * si.view(B, M, G, 1, gs)
            ci = torch.addcmul(iim, Are, si.view(B, M, G, 1, gs)) + Aim * sr.view(B, M, G, 1, gs)
            s_re = cr.permute(0, 3, 1, 2, 4).reshape(B, c, M, dv)
            s_im = ci.permute(0, 3, 1, 2, 4).reshape(B, c, M, dv)
            out.append(self._read(s_re, s_im, AR[:, s0:s1], AI[:, s0:s1],
                                  BR[:, s0:s1], BI[:, s0:s1], 2))
            sr, si = s_re[:, -1], s_im[:, -1]
        return _gated_out(self, torch.cat(out, 1), z), {"sr": sr, "si": si}

    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        _ar, _ai = self._gate_cartesian(h_t, B)
        gr, gi = _ar[:, :, :, None], _ai[:, :, :, None]
        rg = state["sr"].view(B, M, G, gs)
        ig = state["si"].view(B, M, G, gs)
        sr = (gr * rg - gi * ig).reshape(B, M, dv)
        si = (gr * ig + gi * rg).reshape(B, M, dv)
        ar, ai, br, bi = self._alpha_beta(z_t)
        vt = self.V(z_t)
        if self.delta_rule:
            vt = self._delta_v(vt, sr, si, ar, ai, br, bi, z_t)
        sr = sr + vt[:, None, :]
        return _gated_out(self, self._read(sr, si, ar, ai, br, bi, 1), z_t), {"sr": sr, "si": si}


register("sepq", CHeadQuad, DHeadSepQ, arch=True,
         note="separable D-head query (ARCH: different function)")

from .compiled import wrap as _cw  # noqa: E402
register("sepq_cc", CHeadQuad, DHeadSepQ, arch=True, wrap=_cw,
         note="separable D-head query + compile (ARCH: different function)")

from .compiled import wrap_dynamic as _cwd  # noqa: E402
register("sepq_dyn", CHeadQuad, DHeadSepQ, arch=True, wrap=_cwd,
         note="separable D-head query + compile(dynamic=True)")


class DHeadSepQPolar(DHeadSepQ):
    r"""sepq with the gate emitted directly in LOG-POLAR form.

    The cartesian parameterization computes a complex gate with two tanh, then
    immediately takes log|a| and arg(a) to run the scan -- reconstructing a polar
    form the network could have emitted in the first place:

        cartesian:   a = (tanh(gr) + i.tanh(gi))/sqrt(2)
                     log|a| = 0.5 log(gr^2 + gi^2 + tiny),  arg = atan2(gi, gr)
        polar:       log|a| = -softplus(w_mag),             arg = w_phase

    Same parameter shapes and count (two Linear(d, M.G) with bias), so this is a
    reinterpretation rather than a resize -- but a different function, hence an
    A/B rather than an iso check.

    What it removes: two tanh, a log and an atan2; the `tiny` floor; and with it
    the whole NaN class that forced `clamp(max=0)` before `exp` (|a| <= 1 is now
    exact by construction, not the result of tanh/sqrt(2) arithmetic). The
    gradient becomes d(log|a|)/dw = -sigmoid(w), bounded, instead of gr/|a|^2,
    which diverges as |a| -> 0.

    Expected speed gain is small: these tensors are B.T.Md.G = 131k elements
    against 1M for the decay matrices. The point is numerical, and it matters
    before freezing anything into a hand-written kernel.
    """

    # ---- error-correcting write (delta rule) ------------------------------ #
    def _delta_v(self, v_t, sr, si, ar, ai, br, bi, z_t):
        """`v - beta . read(state)`: subtract what the memory already returns for
        this query before writing, instead of only ever adding.

        The state is decayed FIRST and the correction reads the decayed state,
        matching the delta rule's ordering (h *= exp(g); v_new = beta(v - h^T k)).
        """
        P = (sr * ar.unsqueeze(-1) + si * ai.unsqueeze(-1)).mean(1)
        Q = (si * ar.unsqueeze(-1) - sr * ai.unsqueeze(-1)).mean(1)
        beta = torch.sigmoid(self.wbeta(z_t))
        return v_t - beta * (br * P + bi * Q)

    def _delta_lag(self, v_c, sr, si, ar_c, ai_c, br_c, bi_c, z_c):
        """Correction against the state at the CHUNK START, for a whole chunk.

        `v_eff` then no longer depends on the running state, so the transition
        stays diagonal and the chunked closed form applies unchanged -- the exact
        per-token version makes it (a - beta.P), non-diagonal, which is what
        forces a sequential scan. The reference is at most CHUNK tokens stale.
        """
        M = self.M
        P = (torch.einsum("btm,bmj->btj", ar_c, sr)
             + torch.einsum("btm,bmj->btj", ai_c, si)) / M
        Q = (torch.einsum("btm,bmj->btj", ar_c, si)
             - torch.einsum("btm,bmj->btj", ai_c, sr)) / M
        beta = torch.sigmoid(self.wbeta(z_c))
        return v_c - beta * (br_c * P + bi_c * Q)

    def _prefill_delta(self, z, h, state=None):
        """Sequential, and exact.

        The correction makes the transition (a - beta.P) NON-DIAGONAL, so the
        chunked closed form of `prefill` does not apply -- this is exactly why
        fla's reference carries a 64-step loop for the UT/WY transform. Until
        that representation is written here, this path trades speed for
        correctness so the IDEA can be A/B'd on quality now.
        """
        B, T, _ = z.shape
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        v = self.V(z)
        ar, ai, br, bi = self._alpha_beta(z)
        s2 = 1.0 / math.sqrt(2)
        gr = torch.tanh(self.gr(h)).view(B, T, M, G) * s2
        gi = torch.tanh(self.gi(h)).view(B, T, M, G) * s2
        if isinstance(self, DHeadSepQPolar):
            mag = torch.exp(-F.softplus(self.gr(h)).view(B, T, M, G))
            ph = self.gi(h).view(B, T, M, G)
            gr, gi = mag * ph.cos(), mag * ph.sin()
        sr, si, out = st["sr"], st["si"], []
        for t in range(T):
            a_r = gr[:, t].unsqueeze(-1)                    # (B,M,G,1)
            a_i = gi[:, t].unsqueeze(-1)
            rg, ig = sr.view(B, M, G, gs), si.view(B, M, G, gs)
            sr = (a_r * rg - a_i * ig).reshape(B, M, dv)    # decay first
            si = (a_r * ig + a_i * rg).reshape(B, M, dv)
            ve = self._delta_v(v[:, t], sr, si, ar[:, t], ai[:, t],
                               br[:, t], bi[:, t], z[:, t])
            sr = sr + ve[:, None, :]                        # then write
            out.append(self._read(sr, si, ar[:, t], ai[:, t], br[:, t], bi[:, t], 1))
        return _gated_out(self, torch.stack(out, 1), z), {"sr": sr, "si": si, "empty": False}

    def _log_polar(self, h, B, T):
        la = -F.softplus(self.gr(h)).view(B, T, self.M, self.G)
        ph = self.gi(h).view(B, T, self.M, self.G)
        return la.permute(0, 2, 3, 1), ph.permute(0, 2, 3, 1)

    def _gate_cartesian(self, h, B):
        mag = torch.exp(-F.softplus(self.gr(h)).view(B, self.M, self.G))
        ph = self.gi(h).view(B, self.M, self.G)
        return mag * torch.cos(ph), mag * torch.sin(ph)


register("polar", CHeadQuad, DHeadSepQPolar, arch=True,
         note="sepq + log-polar gate (ARCH: different function)")
register("polar_cc", CHeadQuad, DHeadSepQPolar, arch=True, wrap=_cw,
         note="sepq + log-polar gate + compile")


# --- delta-rule variants ---------------------------------------------------- #
from dataclasses import replace as _replace            # noqa: E402
from .ref import SCA2Layer as _SCA2Layer               # noqa: E402


class DeltaLayer(_SCA2Layer):
    """Forces `delta_rule=True` in the config.

    Registering the plain classes and relying on the CALLER to pass
    `LayerCfg(delta_rule=True)` silently produced a run with the delta rule OFF
    -- the variant name promised a behaviour the config did not deliver. The
    name now carries it.
    """

    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__(_replace(cfg, delta_rule=True), c_cls, d_cls)
        assert self.dh.delta_rule, "delta rule failed to activate"


register("polar_delta", CHeadQuad, DHeadSepQPolar, arch=True, layer_cls=DeltaLayer,
         note="polar + DELTA-RULE error-correcting write")
register("polar_delta_cc", CHeadQuad, DHeadSepQPolar, arch=True, wrap=_cw,
         layer_cls=DeltaLayer, note="polar + delta-rule write + compile")
