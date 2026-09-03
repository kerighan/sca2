"""
Key-addressed complex delta rule.

The previous `delta_rule` option corrected in the direction of the QUERY, which
is rank-1 and cannot overwrite: the state has no address, so "what is already
stored here" is undefined. Measured effect: none (mean -0.003 over 8 matched
token counts, sd 0.021).

This gives the D head an address. State is a complex matrix H (dk x dv) per
head -- a map from key space to value space, exactly like Gated DeltaNet -- but
with OUR complex gate instead of their real scalar decay, so the phase
interference survives:

    H     <- diag(a[t]) . H                      |a| <= 1, complex, per key row
    p[t]  = Re( sum_k conj(k_k) H[k,:] )         what is stored AT THIS KEY
    v_new = beta[t] . ( v[t] - p[t] )            error-correcting write
    H     <- H + k[t] (x) v_new
    o[t]  = Re( sum_k conj(q_k) H[k,:] )

Overwrite property: keys are L2-normalized, so sum_k conj(k_k) k_k = 1 and an
immediate re-read at the same key returns (1-beta).p + beta.v -- at beta = 1 the
old value is exactly replaced. That is the thing the keyless version could not
do.

Prefill is sequential and exact. The chunked closed form needs a UT/WY
representation as in fla; with torch.compile the sequential loop runs at ~14k
tok/s end to end, which is enough to answer the quality question first.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import DHeadBase, _rms, _gated_out
from .registry import register
from .versions.v1_quad_scan import CHeadQuad
from .compiled import wrap as _cw


class DHeadKeyed(DHeadBase):
    def __init__(self, d, M=4, G=4, dv=None, max_len=None, delta_rule=True,
                 gated_read=False, dk=16):
        nn.Module.__init__(self)
        self.d = d
        self.H = M                       # heads
        self.G = G                       # gate groups over the key rows
        self.dk = dk
        self.dv = (d // 2 if dv is None else dv)
        self.dvh = self.dv // self.H
        assert self.dv % self.H == 0 and dk % G == 0
        self.gks = dk // G
        self.delta_rule = delta_rule
        self.gated_read = gated_read
        kd = self.H * dk
        self.kr = nn.Linear(d, kd, False); self.ki = nn.Linear(d, kd, False)
        self.qr = nn.Linear(d, kd, False); self.qi = nn.Linear(d, kd, False)
        self.V = nn.Linear(d, self.dv, False)
        self.gr = nn.Linear(d, self.H * G)        # log|a|  (via -softplus)
        self.gp = nn.Linear(d, self.H * G)        # phase
        self.wbeta = nn.Linear(d, self.H)
        if gated_read:
            self.rgate = nn.Linear(d, 2 * self.dv, bias=False)

    # ---- pieces ----------------------------------------------------------- #
    def _kq(self, z, n):
        """L2-normalized complex key and query, (B,n,H,dk) each."""
        sh = (z.shape[0], n, self.H, self.dk)
        kr, ki = self.kr(z).view(sh), self.ki(z).view(sh)
        qr, qi = self.qr(z).view(sh), self.qi(z).view(sh)
        kn = torch.rsqrt(kr.square().sum(-1, keepdim=True)
                         + ki.square().sum(-1, keepdim=True) + 1e-6)
        qn = torch.rsqrt(qr.square().sum(-1, keepdim=True)
                         + qi.square().sum(-1, keepdim=True) + 1e-6)
        return kr * kn, ki * kn, qr * qn, qi * qn

    def _gate(self, h, n):
        """Complex gate per (head, group), |a| <= 1 by construction."""
        sh = (h.shape[0], n, self.H, self.G)
        mag = torch.exp(-F.softplus(self.gr(h)).view(sh))
        ph = self.gp(h).view(sh)
        return mag * ph.cos(), mag * ph.sin()

    @staticmethod
    def _addr(xr, xi, Hr, Hi):
        """Re( sum_k conj(x_k) H[k,:] ) -- what is stored at address x.

        Only the real part is needed for the correction: the write puts a REAL
        v_new at key k, and since ||k|| = 1 that term contributes exactly v_new
        to the real part and 0 to the imaginary one.
        """
        return (torch.einsum("bhk,bhkj->bhj", xr, Hr)
                + torch.einsum("bhk,bhkj->bhj", xi, Hi))

    @staticmethod
    def _addr2(xr, xi, Hr, Hi):
        """Full complex read: conj(x)^T H, real and imaginary parts."""
        re = (torch.einsum("bhk,bhkj->bhj", xr, Hr)
              + torch.einsum("bhk,bhkj->bhj", xi, Hi))
        im = (torch.einsum("bhk,bhkj->bhj", xr, Hi)
              - torch.einsum("bhk,bhkj->bhj", xi, Hr))
        return re, im

    def init_state(self, B, device, dtype):
        z = torch.zeros(B, self.H, self.dk, self.dvh, device=device, dtype=dtype)
        return {"Hr": z, "Hi": z.clone()}

    # ---- one token -------------------------------------------------------- #
    def _one(self, Hr, Hi, kr, ki, qr, qi, v, ar, ai, beta):
        # decay / rotate each key row (grouped)
        B, H, dk, dvh = Hr.shape
        a_r = ar.repeat_interleave(self.gks, -1)[..., None]      # (B,H,dk,1)
        a_i = ai.repeat_interleave(self.gks, -1)[..., None]
        Hr, Hi = a_r * Hr - a_i * Hi, a_r * Hi + a_i * Hr
        p = self._addr(kr, ki, Hr, Hi)                           # (B,H,dvh)
        v_new = beta[..., None] * (v - p)
        Hr = Hr + kr[..., None] * v_new[:, :, None, :]
        Hi = Hi + ki[..., None] * v_new[:, :, None, :]
        o_re, o_im = self._addr2(qr, qi, Hr, Hi)                 # (B,H,dvh)
        return Hr, Hi, o_re, o_im

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        kr, ki, qr, qi = self._kq(z, T)
        ar, ai = self._gate(h, T)
        v = self.V(z).view(B, T, self.H, self.dvh)
        beta = torch.sigmoid(self.wbeta(z))                      # (B,T,H)
        Hr, Hi, out = st["Hr"], st["Hi"], []
        for t in range(T):
            Hr, Hi, o_re, o_im = self._one(Hr, Hi, kr[:, t], ki[:, t], qr[:, t],
                                           qi[:, t], v[:, t], ar[:, t], ai[:, t],
                                           beta[:, t])
            out.append(torch.cat([o_re.reshape(B, self.dv),
                                  o_im.reshape(B, self.dv)], -1))
        return _gated_out(self, torch.stack(out, 1), z), {"Hr": Hr, "Hi": Hi}

    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        kr, ki, qr, qi = self._kq(z_t[:, None], 1)
        ar, ai = self._gate(h_t[:, None], 1)
        v = self.V(z_t).view(B, self.H, self.dvh)
        beta = torch.sigmoid(self.wbeta(z_t))
        Hr, Hi, o_re, o_im = self._one(state["Hr"], state["Hi"], kr[:, 0], ki[:, 0],
                                       qr[:, 0], qi[:, 0], v, ar[:, 0], ai[:, 0], beta)
        u = torch.cat([o_re.reshape(B, self.dv), o_im.reshape(B, self.dv)], -1)
        return _gated_out(self, u, z_t), {"Hr": Hr, "Hi": Hi}


class DHeadKeyedMatched(DHeadKeyed):
    """dk fixed so the head lands on our D-head budget (61,988 vs 61,696)."""

    def __init__(self, d, M=4, G=4, dv=None, max_len=None, delta_rule=True,
                 gated_read=False):
        super().__init__(d, M=4, G=4, dv=dv, max_len=max_len,
                         delta_rule=delta_rule, gated_read=gated_read, dk=24)


register("keyed", CHeadQuad, DHeadKeyedMatched, arch=True,
         note="key-addressed complex delta rule (ARCH)")
register("keyed_cc", CHeadQuad, DHeadKeyedMatched, arch=True, wrap=_cw,
         note="key-addressed complex delta rule + compile")
