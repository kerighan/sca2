r"""C head with one spectral mixture PER VALUE GROUP -- lifting the rank-2 read.

Why. The C head's score is a SCALAR, independent of the value coordinate j:

    kappa[t,s] = (1/M) sum_m w_m e^{i(pw[s,m] - pq[t,m])},   w_m = wr_m + i.wi_m

which is exactly what CHeadQuad exploits to compute the head as attention with a
2M-dim feature map. The cost of that scalarity is a bound. At theta = 0 the
kernel depends only on the lag, f_l = (1/M) sum_m w_m e^{i l omega_m}, and the
pre-RMS linear path through an output projection [P_R, P_I] is

    H_l = Re(f_l) P_R W_V + Im(f_l) P_I W_V     =>     dim span{H_l}_l <= 2.

So Mc = 378 modes and a 24,192-real state buy a SHARPER single filter, not more
filters: the linear branch has only two shared temporal profiles. That is a
candidate explanation for our measured Mc plateau (raising Mc steepened the
position slope but stopped paying on loss).

What. Give each of WG contiguous slices of the dv value coordinates its own
spectral mixture, w_{m,g}, leaving the phases, the state and the write path
untouched:

    u[t, j] = sum_{s<=t} kappa^{g(j)}[t,s] . v[s, j]

The state stays sum_s v_s e^{i pw_s} -- identical, shared by all groups -- so
this costs NO state and 2.M.(WG-1) parameters per layer. It buys WG independent
temporal profiles instead of 2.

Cost. Scalarity is what made one score matrix enough, so the quadratic form now
needs one per group: the (2M)-contraction is WG times more expensive, while each
group's value matmul is 1/WG as wide. Use the smallest WG that shows an effect.

WG=1 is the current function exactly, and `python -m sca2.arch_wgroup` checks it
against v3polarflat.
"""
import os

import torch

from .compiled import wrap as _cw
from .fast_dhead import DHeadSepQPolarFlat
from .ref import _gated_out
from .registry import register
from .versions.v1_quad_scan import CHeadQuad, causal_mask


class CHeadWGroup(CHeadQuad):
    """CHeadQuad with wr/wi of shape (WG, M) instead of (M,)."""

    WG = 1
    CTX = int(os.environ.get("SCA2_CTX_CHUNK", 256))

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        g = self.WG
        assert self.dv % g == 0, f"dv={self.dv} not divisible by WG={g}"
        self.gw = self.dv // g
        self.wr = torch.nn.Parameter(self.wr.detach()[None].repeat(g, 1))
        self.wi = torch.nn.Parameter(self.wi.detach()[None].repeat(g, 1))

    # ---- prefill ---------------------------------------------------------- #
    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        C = self.CTX
        if T <= C:
            return self._one(z, h, state)
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        outs = []
        for s0 in range(0, T, C):
            o, st = self._one(z[:, s0:s0 + C], h[:, s0:s0 + C], st)
            outs.append(o)
        return torch.cat(outs, 1), st

    def _one(self, z, h, state=None):
        B, T, _ = z.shape
        G, gw, M = self.WG, self.gw, self.M
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        pw, pq = self._phases(z, h, st["pos"])

        cw, sw = pw.cos(), pw.sin()                            # (B,T,M)
        cq, sq = pq.cos(), pq.sin()
        wr, wi = self.wr[:, None, None, :], self.wi[:, None, None, :]   # (G,1,1,M)
        A = wr * cw - wi * sw                                  # (G,B,T,M)
        Bm = wr * sw + wi * cw

        Fq = torch.cat([cq, sq], -1)                           # (B,T,2M)
        Fk = torch.cat([torch.cat([A, Bm], -1),
                        torch.cat([Bm, -A], -1)], 2)           # (G,B,2T,2M)
        K2 = (Fq @ Fk.transpose(2, 3)).view(G, B, T, 2, T)
        K2 = K2.masked_fill(causal_mask(T, z.device)[None, None, :, None, :], 0)

        v = self.V(z).view(B, T, G, gw)                         # (B,T,G,gw)
        vg = v.permute(2, 0, 1, 3)                              # (G,B,T,gw)
        o = (K2.reshape(G, B, T * 2, T) @ vg).view(G, B, T, 2, gw) / M

        if not st.get("empty", False):
            sr0 = st["sr"].view(B, M, G, gw).permute(2, 0, 1, 3)   # (G,B,M,gw)
            si0 = st["si"].view(B, M, G, gw).permute(2, 0, 1, 3)
            c1 = wr * cq + wi * sq                              # (G,B,T,M)
            c2 = wr * sq - wi * cq
            r1 = torch.einsum("gbtm,gbmj->gbtj", c1, sr0)
            r2 = torch.einsum("gbtm,gbmj->gbtj", c2, si0)
            i1 = torch.einsum("gbtm,gbmj->gbtj", c2, sr0)
            i2 = torch.einsum("gbtm,gbmj->gbtj", c1, si0)
            o = o + torch.stack([r1 + r2, i2 - i1], 3) / M

        # (G,B,T,2,gw) -> (B,T,2*dv), re block then im block, groups in order
        u = o.permute(1, 2, 3, 0, 4).reshape(B, T, 2, self.dv).reshape(B, T, 2 * self.dv)

        vf = self.V(z)
        sr = torch.einsum("btm,btj->bmj", cw, vf) + st["sr"]
        si = torch.einsum("btm,btj->bmj", sw, vf) + st["si"]
        return _gated_out(self, u, z), {"sr": sr, "si": si,
                                        "pos": st["pos"] + T, "empty": False}

    # ---- decode ----------------------------------------------------------- #
    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        G, gw, M = self.WG, self.gw, self.M
        p = state["pos"]
        pw = self.K(h_t) * self.theta + p * self.omega
        pq = self.K(z_t) * self.theta + p * self.omega
        v = self.V(z_t)
        sr = torch.addcmul(state["sr"], v[:, None, :], pw.cos()[:, :, None])
        si = torch.addcmul(state["si"], v[:, None, :], pw.sin()[:, :, None])
        cq, sq = pq.cos(), pq.sin()                             # (B,M)
        c1 = self.wr[:, None] * cq + self.wi[:, None] * sq      # (G,B,M)
        c2 = self.wr[:, None] * sq - self.wi[:, None] * cq
        srg = sr.view(B, M, G, gw).permute(2, 0, 1, 3)          # (G,B,M,gw)
        sig = si.view(B, M, G, gw).permute(2, 0, 1, 3)
        re = (torch.einsum("gbm,gbmj->gbj", c1, srg)
              + torch.einsum("gbm,gbmj->gbj", c2, sig)) / M
        im = (torch.einsum("gbm,gbmj->gbj", c1, sig)
              - torch.einsum("gbm,gbmj->gbj", c2, srg)) / M
        u = torch.cat([re.permute(1, 0, 2).reshape(B, self.dv),
                       im.permute(1, 0, 2).reshape(B, self.dv)], -1)
        return _gated_out(self, u, z_t), {"sr": sr, "si": si,
                                          "pos": p + 1, "empty": False}


def _mk(g):
    return type(f"CHeadWG{g}", (CHeadWGroup,), {"WG": g})


for _g in (1, 2, 4, 8):
    register(f"wg{_g}", _mk(_g), DHeadSepQPolarFlat, arch=True,
             note=f"C head with {_g} spectral mixtures (one per value group)")
    register(f"wg{_g}_cc", _mk(_g), DHeadSepQPolarFlat, arch=True, wrap=_cw,
             note=f"C head with {_g} spectral mixtures + compile")


def _check():
    """WG=1 must be v3polarflat exactly; WG>1 must be prefill/decode consistent."""
    from .ref import LayerCfg
    from .registry import build

    cfg = LayerCfg(d=128, Mc=378, Md=4, G=8, ff=364, freq="rope",
                   theta_scale=0.02, max_len=1024, dv=32)
    ref = build("v3polarflat", cfg, dtype=torch.float64)
    x = torch.randn(2, 96, 128, dtype=torch.float64)

    # T must exceed CTX, or prefill runs as ONE chunk with an empty incoming
    # state and the carry term -- where the group axis meets the batch axis --
    # is never executed. That is exactly the bug this check first missed.
    CHeadWGroup.CTX = 32
    assert x.size(1) > CHeadWGroup.CTX

    one = build("wg1", cfg, dtype=torch.float64)
    sd = dict(ref.state_dict())
    sd["c.wr"] = sd["c.wr"][None]
    sd["c.wi"] = sd["c.wi"][None]
    one.load_state_dict(sd)
    with torch.no_grad():
        d = (ref(x) - one(x)).abs().max().item()
    print(f"wg1 vs v3polarflat: max|diff| = {d:.3e}")
    assert d < 1e-9, "WG=1 is not the reference function"

    for g in (2, 4, 8):
        m = build(f"wg{g}", cfg, dtype=torch.float64)
        with torch.no_grad():
            full, _ = m.prefill(x)
            st, outs = None, []
            for t in range(x.size(1)):
                y, st = m.step(x[:, t], st) if st is not None else \
                    m.step(x[:, t], m.init_state(x.size(0), x.device, x.dtype))
                outs.append(y)
            seq = torch.stack(outs, 1)
        n = sum(p.numel() for p in m.parameters())
        print(f"wg{g}: params {n}  prefill-vs-decode max|diff| = "
              f"{(full - seq).abs().max().item():.3e}")
        assert torch.allclose(full, seq, atol=1e-9), f"wg{g} prefill != decode"
    print("OK")


if __name__ == "__main__":
    _check()
