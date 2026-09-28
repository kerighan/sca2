"""GDN with an equalizer: one decay per key dimension instead of one per head.

The Zyda campaign tuned LapA toward GDN and lost by 0.086 nat, 74% of it
architectural. The whole spread of everything tried was 0.022. You cannot
local-search back to a baseline from five times that distance, so this goes the
other way: start from GDN's exact code and change ONE thing.

GDN's kernel, with the delta rule's correction off:

    kappa_GDN(t,u) = (q_t . k_u) * exp(sum_{s=u+1..t} g_s)

Content times ONE real decay per head: the same gain applied to every term of
the inner product. Here the gate becomes per key dimension:

    kappa(t,u) = sum_j q_{t,j} k_{u,j} * exp(r_j * G_{u,t})

so each of the dk coordinates carries its own memory length. See chead_numpy.py,
which checks both closed forms and that r = 1 reproduces GDN to 2.2e-16 with the
delta rule included.

The recurrence is NOT reimplemented: fla ships it as `kda` (Kimi Delta
Attention), whose reference loop is GDN's word for word with g of shape
[B,T,H,K] instead of [B,T,H]. So this is a known architecture, not a new one,
and the Triton kernels come with it -- which is what makes the wall-clock
comparison against GDN honest rather than a slow-scan handicap.

What differs from KDA as published: the gate is FACTORISED. KDA projects
d -> H*K, which on this shape is 1.05M parameters a layer. Here a(x) stays
d -> H and only A_log becomes (H, K), so

    g_{t,h,j} = -exp(A_log[h,j]) * softplus(a(x)[h] + dt_bias[h])

is a fixed learned SPECTRUM modulated by one shared data-dependent gate:
1024 parameters a layer, 8192 in total. That is the minimal step from GDN and
the one the equalizer analogy is actually about -- a bank of time constants, not
a per-dimension gate network.

A_log is initialised identically across j, so the model STARTS at GDN exactly.
Whether the r_j spread apart is then itself the measurement, independent of the
loss: LapA had a spectrum from the start and its lambdas COLLAPSED to 7-14 token
memories on half its layers, so a spectrum has never been exercised in a model
that works.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_gdn import ShortConv

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except Exception:                                          # pragma: no cover
    chunk_kda = fused_recurrent_kda = None
from fla.ops.kda.naive import naive_recurrent_kda


class EqualizedDeltaNet(nn.Module):
    """GatedDeltaNet with a per-key-dimension decay spectrum."""

    def __init__(self, d, heads=4, head_k=32, expand_v=2.0, conv_k=4,
                 gate_full=False):
        super().__init__()
        self.d, self.H = d, heads
        self.dk = head_k
        self.dv = int(head_k * expand_v)
        self.key_dim, self.value_dim = heads * self.dk, heads * self.dv
        self.q = nn.Linear(d, self.key_dim, False)
        self.k = nn.Linear(d, self.key_dim, False)
        self.v = nn.Linear(d, self.value_dim, False)
        # gate_full: KDA as published, a(x) -> H*K, 1.05M params a layer.
        # Default is the factorised form: one shared gate, a learned spectrum.
        self.gate_full = gate_full
        self.a = nn.Linear(d, self.key_dim if gate_full else heads, False)
        self.b = nn.Linear(d, heads, False)
        self.gp = nn.Linear(d, self.value_dim, False)
        self.o = nn.Linear(self.value_dim, d, False)
        self.cq, self.ck, self.cv = (ShortConv(self.key_dim, conv_k),
                                     ShortConv(self.key_dim, conv_k),
                                     ShortConv(self.value_dim, conv_k))
        # GDN's init, then broadcast across the key axis: every band starts with
        # the same time constant, so the model starts AT GDN and has to move.
        A = torch.empty(heads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A)[:, None].repeat(1, self.dk).contiguous())
        dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm = nn.LayerNorm(self.dv)
        self.conv_k = conv_k

    # ---- pieces ----------------------------------------------------------- #
    def _gates(self, x):
        """g: (B,T,H,K) in log space, beta: (B,T,H)."""
        B, T, _ = x.shape
        sp = F.softplus(self.a(x).float()
                        + (0.0 if self.gate_full else self.dt_bias))
        if self.gate_full:
            sp = sp.view(B, T, self.H, self.dk)
        else:
            sp = sp[..., None]                                   # (B,T,H,1)
        g = -torch.exp(self.A_log.float()) * sp                  # broadcast over K
        return g, self.b(x).float().sigmoid()

    def _read(self, o, x, B, T):
        o = self.o_norm(o) * F.silu(self.gp(x)).view(B, T, self.H, self.dv)
        return self.o(o.reshape(B, T, self.value_dim))

    def init_state(self, B, device, dtype):
        z = lambda n: torch.zeros(B, self.conv_k - 1, n, device=device, dtype=dtype)
        return {"h": torch.zeros(B, self.H, self.dk, self.dv, device=device, dtype=dtype),
                "cq": z(self.key_dim), "ck": z(self.key_dim), "cv": z(self.value_dim)}

    def decay_spread(self):
        """Per layer: how far the bands have moved apart. The measurement.

        Returns [log-span, n_eff] of exp(A_log) over the key axis, averaged over
        heads. At init every band shares a time constant, so the span is 0 and
        n_eff is dk. A spectrum forming means the span grows.
        """
        with torch.no_grad():
            a = torch.exp(self.A_log.float())                     # (H, K)
            q = torch.quantile(a, torch.tensor([0.05, 0.95], device=a.device), dim=-1)
            span = torch.log(q[1].clamp(min=1e-9) / q[0].clamp(min=1e-9)).mean()
            p = a / a.sum(-1, keepdim=True).clamp(min=1e-9)
            neff = torch.exp(-(p * (p + 1e-12).log()).sum(-1)).mean()
        return [round(span.item(), 4), round(neff.item(), 1)]

    # ---- prefill ---------------------------------------------------------- #
    def forward(self, x, state=None):
        return self.prefill(x, state)[0]

    def prefill(self, x, state=None):
        B, T, _ = x.shape
        st = state if state is not None else self.init_state(B, x.device, x.dtype)
        q = self.cq(self.q(x)).view(B, T, self.H, self.dk)
        k = self.ck(self.k(x)).view(B, T, self.H, self.dk)
        v = self.cv(self.v(x)).view(B, T, self.H, self.dv)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        g, beta = self._gates(x)
        h0 = st["h"] if state is not None else None
        if chunk_kda is not None and x.is_cuda:
            dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.bfloat16
            o, h = chunk_kda(q.to(dt), k.to(dt), v.to(dt), g.float(), beta.to(dt),
                             initial_state=None if h0 is None else h0.float(),
                             output_final_state=True)
        else:
            o, h = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0,
                                       output_final_state=True)
        y = self._read(o.to(x.dtype), x, B, T)
        tail = lambda z_: z_[:, -(self.conv_k - 1):] if T >= self.conv_k - 1 else \
            F.pad(z_, (0, 0, self.conv_k - 1 - T, 0))
        return y, {"h": h.to(x.dtype), "cq": tail(self.q(x)),
                   "ck": tail(self.k(x)), "cv": tail(self.v(x))}

    # ---- decode ----------------------------------------------------------- #
    def step(self, x_t, state):
        B = x_t.size(0)
        qr, cq = self.cq.step(self.q(x_t), state["cq"])
        kr, ck = self.ck.step(self.k(x_t), state["ck"])
        vr, cv = self.cv.step(self.v(x_t), state["cv"])
        q = F.normalize(qr.view(B, 1, self.H, self.dk), dim=-1)
        k = F.normalize(kr.view(B, 1, self.H, self.dk), dim=-1)
        v = vr.view(B, 1, self.H, self.dv)
        g, beta = self._gates(x_t[:, None])
        if fused_recurrent_kda is not None and x_t.is_cuda:
            o, h = fused_recurrent_kda(q, k, v, g, beta,
                                       initial_state=state["h"].float(),
                                       output_final_state=True)
        else:
            o, h = naive_recurrent_kda(q, k, v, g, beta,
                                       initial_state=state["h"].float(),
                                       output_final_state=True)
        y = self._read(o.to(x_t.dtype), x_t[:, None], B, 1)[:, 0]
        return y, {"h": h.to(x_t.dtype), "cq": cq, "ck": ck, "cv": cv}


class KDALayer(nn.Module):
    """Same wrapper as GDNLayer: norm -> mixer -> residual -> norm -> FFN."""

    def __init__(self, cfg, heads=4, head_k=32, expand_v=2.0, gate_full=False):
        super().__init__()
        d = cfg.d
        self.cfg = cfg
        self.n = nn.LayerNorm(d)
        self.mix = EqualizedDeltaNet(d, heads, head_k, expand_v, gate_full=gate_full)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff), nn.GELU(), nn.Linear(cfg.ff, d))

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


class KDALayerMatched(KDALayer):
    """Shaped from LayerCfg, exactly as GDNLayerMatched is."""

    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__(cfg, heads=cfg.gdn_heads, head_k=cfg.gdn_head_k,
                         expand_v=cfg.gdn_expand_v, gate_full=cfg.kda_gate_full)


from .registry import register                                   # noqa: E402
from .compiled import wrap as _cw                                # noqa: E402
from .versions.v1_quad_scan import CHeadQuad                     # noqa: E402

register("kda", CHeadQuad, None, arch=True, layer_cls=KDALayerMatched,
         note="GDN with a per-key-dimension decay spectrum (ARCH)")
register("kda_cc", CHeadQuad, None, arch=True, layer_cls=KDALayerMatched, wrap=_cw,
         note="GDN with a per-key-dimension decay spectrum + torch.compile")
