"""GDN with a shift-invariant key kernel, built from random Fourier features.

This is the other frequency axis. The campaign's equalizer arms -- kda and
gdnrope -- both act on the TIME conjugate: how the kernel decays or oscillates
with the lag. LapA's actual spectral idea is on the KEY conjugate:

    kappa(t,u) = sum_m w_m e^{-lam_m D} e^{i theta_m (k_t - k_u)} e^{i omega_m D}
                                        ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
                                        conjugate to the KEY       to TIME

s_m = sum_u e^{i theta_m k_u} beta_u v_u is an empirical CHARACTERISTIC
FUNCTION of the key distribution, weighted by the values. By Bochner's theorem a
shift-invariant kernel is the Fourier transform of a positive measure, so w is
that kernel's spectral density and the theta_m are random Fourier features
(Rahimi & Recht). Attenuating a band does not change a volume: it changes the
kernel's WIDTH, i.e. how finely the model still distinguishes nearby keys.
Checked in chead_numpy: cutting the high half of |theta| takes a distant pair
from 0.072 to 0.646 of similarity while a close pair does not move.

What this arm changes, and ONLY this:

    GDN       kappa(t,u) = (q_t . k_u)             * decay      inner product
    here      kappa(t,u) = sum_m cos(th_m (p_t - p_u)) * decay  key DIFFERENCE

The gate stays GDN's single scalar per head and the delta rule is untouched, so
a result here is attributable to the content term alone -- which is the axis the
Zyda campaign never isolated, and one of the two suspects left once every
temporal knob had been shown to span 0.022 nat against a 0.086 gap.

Construction. One projection P: d -> M per head, SHARED between query and key --
the difference p_t - p_u is only meaningful in a common space, and it is what
LapA does. Then

    phi(x)_m = [cos(th_m x_m), sin(th_m x_m)] / sqrt(M)

gives 2M = dk features whose inner product is sum_m cos(th_m (x_m - y_m)) / M.
Note ||phi(x)|| = 1 by construction, so these features need no L2 normalisation:
the sphere GDN reaches by normalising, this reaches by identity.

The state, (dk, dv), is unchanged and indexed BY FREQUENCY, which is what makes
the per-frequency gate a one-line follow-up rather than a rewrite: fla's kda
kernel already takes a gate of shape [B,T,H,K]. That arm is deliberately not
this one.

NOT a superset of GDN. kda and gdnrope both contain the baseline at a known
point in parameter space; this does not -- at theta = 0 every feature collapses
to [1, 0] and the kernel becomes constant. It is a different function, and the
comparison is a comparison, not an ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_gdn import ShortConv, _ref, _triton, _triton_recurrent


class RFFDeltaNet(nn.Module):
    """GatedDeltaNet whose content term is a shift-invariant kernel on the keys."""

    def __init__(self, d, heads=4, head_k=32, expand_v=2.0, conv_k=4,
                 theta_scale=1.0, learn_theta=True):
        super().__init__()
        assert head_k % 2 == 0, f"head_k must be even for cos/sin pairs, got {head_k}"
        self.d, self.H = d, heads
        self.dk = head_k
        self.M = head_k // 2                        # frequencies per head
        self.dv = int(head_k * expand_v)
        self.key_dim, self.value_dim = heads * self.dk, heads * self.dv
        # ONE projection, shared by query and key: p_t - p_u must live in a
        # single space. This is also where the parameter count drops against
        # GDN, which pays for two d -> H*dk projections.
        self.p = nn.Linear(d, heads * self.M, False)
        self.v = nn.Linear(d, self.value_dim, False)
        self.a = nn.Linear(d, heads, False)
        self.b = nn.Linear(d, heads, False)
        self.gp = nn.Linear(d, self.value_dim, False)
        self.o = nn.Linear(self.value_dim, d, False)
        self.cp, self.cv = ShortConv(heads * self.M, conv_k), ShortConv(self.value_dim, conv_k)
        # theta: the spectral measure of the kernel. Gaussian-kernel RFF draws
        # theta ~ N(0, 1/sigma^2); the scale sets the kernel's WIDTH, and it is
        # the one hyper-parameter this arm really has.
        th = theta_scale * torch.randn(heads, self.M)
        self.theta = nn.Parameter(th) if learn_theta else None
        if not learn_theta:
            self.register_buffer("theta_fixed", th)
        A = torch.empty(heads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm = nn.LayerNorm(self.dv)
        self.conv_k = conv_k

    def _theta(self):
        return self.theta if self.theta is not None else self.theta_fixed

    def _phi(self, p):
        """p (B,T,H,M) -> (B,T,H,2M), unit norm, interleaved so that dims 2m and
        2m+1 are the cos/sin of band m -- which is what lets a later per-band
        gate address a frequency rather than half of one."""
        a = p * self._theta()
        return torch.stack([a.cos(), a.sin()], -1).flatten(-2) / math.sqrt(self.M)

    def _gates(self, x):
        g = -torch.exp(self.A_log.float()) * F.softplus(self.a(x).float() + self.dt_bias)
        return g, self.b(x).float().sigmoid()

    def _read(self, o, x, B, T):
        o = self.o_norm(o) * F.silu(self.gp(x)).view(B, T, self.H, self.dv)
        return self.o(o.reshape(B, T, self.value_dim))

    def init_state(self, B, device, dtype):
        z = lambda n: torch.zeros(B, self.conv_k - 1, n, device=device, dtype=dtype)
        return {"h": torch.zeros(B, self.H, self.dk, self.dv, device=device, dtype=dtype),
                "cp": z(self.H * self.M), "cv": z(self.value_dim)}

    def forward(self, x, state=None):
        return self.prefill(x, state)[0]

    def prefill(self, x, state=None):
        B, T, _ = x.shape
        st = state if state is not None else self.init_state(B, x.device, x.dtype)
        p = self.cp(self.p(x)).view(B, T, self.H, self.M)
        v = self.cv(self.v(x)).view(B, T, self.H, self.dv)
        qk = self._phi(p)                       # query and key are the SAME map
        g, beta = self._gates(x)
        h0 = st["h"] if state is not None else None
        if _triton is not None and x.is_cuda:
            dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.bfloat16
            o, h = _triton(qk.to(dt), qk.to(dt), v.to(dt), g.float(), beta.to(dt),
                           initial_state=None if h0 is None else h0.float(),
                           output_final_state=True)
        else:
            o, h = _ref.naive_chunk_gated_delta_rule(
                qk, qk, v, g, beta, chunk_size=64,
                initial_state=h0, output_final_state=True)
        y = self._read(o.to(x.dtype), x, B, T)
        tail = lambda z_: z_[:, -(self.conv_k - 1):] if T >= self.conv_k - 1 else \
            F.pad(z_, (0, 0, self.conv_k - 1 - T, 0))
        return y, {"h": h.to(x.dtype), "cp": tail(self.p(x)), "cv": tail(self.v(x))}

    def step(self, x_t, state):
        B = x_t.size(0)
        pr, cp = self.cp.step(self.p(x_t), state["cp"])
        vr, cv = self.cv.step(self.v(x_t), state["cv"])
        qk = self._phi(pr.view(B, 1, self.H, self.M))
        v = vr.view(B, 1, self.H, self.dv)
        g, beta = self._gates(x_t[:, None])
        if _triton_recurrent is not None and x_t.is_cuda:
            o, h = _triton_recurrent(qk, qk, v, g=g, beta=beta,
                                     initial_state=state["h"].float(),
                                     output_final_state=True)
        else:
            o, h = _ref.naive_recurrent_gated_delta_rule(
                qk, qk, v, beta, g, initial_state=state["h"].float(),
                output_final_state=True)
        y = self._read(o.to(x_t.dtype), x_t[:, None], B, 1)[:, 0]
        return y, {"h": h.to(x_t.dtype), "cp": cp, "cv": cv}


class RFFLayer(nn.Module):
    """Same wrapper as GDNLayer: norm -> mixer -> residual -> norm -> FFN."""

    def __init__(self, cfg, heads=4, head_k=32, expand_v=2.0, theta_scale=1.0):
        super().__init__()
        self.cfg = cfg
        self.n = nn.LayerNorm(cfg.d)
        self.mix = RFFDeltaNet(cfg.d, heads, head_k, expand_v, theta_scale=theta_scale)
        self.fn = nn.LayerNorm(cfg.d)
        self.ff = nn.Sequential(nn.Linear(cfg.d, cfg.ff), nn.GELU(),
                                nn.Linear(cfg.ff, cfg.d))

    def init_state(self, B, device, dtype=torch.float32):
        return self.mix.init_state(B, device, dtype)

    def forward(self, x):
        return self.prefill(x)[0]

    def prefill(self, x, state=None):
        y, st = self.mix.prefill(self.n(x), state)
        x = x + y
        return x + self.ff(self.fn(x)), st

    def step(self, x_t, state):
        y, st = self.mix.step(self.n(x_t), state)
        x_t = x_t + y
        return x_t + self.ff(self.fn(x_t)), st


class RFFLayerMatched(RFFLayer):
    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__(cfg, heads=cfg.gdn_heads, head_k=cfg.gdn_head_k,
                         expand_v=cfg.gdn_expand_v, theta_scale=cfg.rff_theta)


from .registry import register                                   # noqa: E402
from .compiled import wrap as _cw                                # noqa: E402
from .versions.v1_quad_scan import CHeadQuad                     # noqa: E402

register("rff", CHeadQuad, None, arch=True, layer_cls=RFFLayerMatched,
         note="GDN with a shift-invariant key kernel from random Fourier features (ARCH)")
register("rff_cc", CHeadQuad, None, arch=True, layer_cls=RFFLayerMatched, wrap=_cw,
         note="GDN with an RFF key kernel + torch.compile")
