"""
Gated multi-head C head -- forgetting, heads, and an asymmetric key.

Why. Per token SCA2 and Gated DeltaNet are indistinguishable (runs/overnight.jsonl:
4.959 vs 4.957 at 93M tokens, 4.792 vs 4.788 at 194M), and SCA2's 1.4x speed
does not convert into loss. The lever is quality per token, and the C head is
the only "attention-like" head in the comparison set WITHOUT forgetting: its
state is a pure cumsum. Three of our own measurements point at that:

  * `rope` beats `dft` by 0.11 nats although the theory rules it out -- its
    low-frequency channels are a de facto smooth recency kernel, i.e. the network
    buys a recency prior with phase channels it would rather spend on content;
  * the score at a match is O(1) while the background from T unmatched positions
    is O(sqrt(T)/Mc) -- without decay the effective T is the whole context;
  * the D head HAS a gate, and it is the head that costs 55% of the time for
    0.04 nats (Md=2 -> 16). Selectivity sits in the wrong head.

What. Three orthogonal changes, each a cfg knob so it can be ablated, and with
every knob off this is CHeadQuad bit for bit (`iso gc_plain --against v3polar`):

  heads   Mc and dv are split into H heads; each head has its own scalar kernel
          kappa_h and reads its own value slice. Same FLOPs as one head
          (H matmuls of T x T x 2Mc/H), H attention patterns instead of one.
  decay   per head a scalar log-decay g_t <= 0 (GDN parameterization,
          -exp(A_log).softplus(a(z)+dt_bias)). In the quadratic form it is an
          elementwise mask exp(cg_t - cg_s) on the T x T score, exactly as in
          GDN's chunked form: cg_t - cg_s <= 0 on kept entries, so nothing
          overflows; masked entries are clamped before exp. The state carry is
          scaled by exp(cg_t). The scalar-kernel trick is untouched because the
          decay is shared across the m of a head. Decode: S <- e^g S + v e^{i pw}.
  sepq    the read phase uses its own projection Kq(z) instead of K(z). With a
          shared K the kernel is exp(i.theta.(K h_s - K z_t)): a symmetric
          difference kernel that can only express "z_t resembles z_{s-1}", never
          "when I see A, look up what followed B".

Plus, at layer level, `conv`: a causal depthwise conv of width k on z before
both heads (identity-initialized, no activation), generalizing the hand-made
h = shift(z). Decode keeps a (k-1)-token buffer in the state.

None of this touches the diagonal recurrence, so unlike the delta-rule variants
(`keyed`: 3x slower, no gain) it costs no matmul, no loop, and d.H + d.Mc + d.k
parameters.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace as _replace

from .ref import CHeadBase, SCA2Layer, _gated_out, LayerCfg
from .registry import register
from .versions.v1_quad_scan import causal_mask
from .arch_sepq import DHeadSepQPolar
from .compiled import wrap as _cw


class CHeadGated(CHeadBase):
    CTX = int(os.environ.get("SCA2_CTX_CHUNK", 256))

    def __init__(self, d, M, freq="dft", theta_scale=0.0, max_len=128, dv=None,
                 gated_read=False, heads=1, decay=False, sepq=False, decay_init="gdn"):
        super().__init__(d, M, freq, theta_scale, max_len, dv, gated_read)
        assert M % heads == 0 and self.dv % heads == 0
        self.H, self.Mh, self.dvh = heads, M // heads, self.dv // heads
        self.decay, self.sepq = decay, sepq
        if sepq:
            self.Kq = nn.Linear(d, M, False)
            with torch.no_grad():
                self.Kq.weight.copy_(self.K.weight)
        if decay:
            self.a = nn.Linear(d, heads, False)
            if decay_init == "soft":
                # start near "never forget": |a| = e^-0.01 -> horizon ~100 tokens
                A = torch.ones(heads)
                dt = torch.full((heads,), 0.01)
            else:
                A = torch.empty(heads).uniform_(1, 16)
                dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001))
                               + math.log(0.001)).clamp(min=1e-4)
            self.A_log = nn.Parameter(torch.log(A))
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    # ---- state ------------------------------------------------------------ #
    def init_state(self, B, device, dtype):
        z = torch.zeros(B, self.H, self.Mh, self.dvh, device=device, dtype=dtype)
        return {"sr": z, "si": z.clone(),
                "pos": torch.zeros((), device=device, dtype=dtype), "empty": True}

    # ---- pieces ----------------------------------------------------------- #
    def _phases(self, z, h, p0):
        T = z.size(1)
        p = (torch.arange(T, device=z.device, dtype=z.dtype) + p0)[:, None] * self.omega
        kq = self.Kq(z) if self.sepq else self.K(z)
        return self.K(h) * self.theta + p, kq * self.theta + p

    def _logdecay(self, z):
        """g <= 0, shape (..., H)."""
        return -torch.exp(self.A_log) * F.softplus(self.a(z) + self.dt_bias)

    def _split(self, x):
        """(B,T,M) -> (B,H,T,Mh)"""
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.Mh).transpose(1, 2)

    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        C = self.CTX
        if T <= C:
            return self._chunk(z, h, st)
        outs = []
        for s0 in range(0, T, C):
            o, st = self._chunk(z[:, s0:s0 + C], h[:, s0:s0 + C], st)
            outs.append(o)
        return torch.cat(outs, 1), st

    def _chunk(self, z, h, st):
        B, T, _ = z.shape
        H, Mh, dvh = self.H, self.Mh, self.dvh
        pw, pq = self._phases(z, h, st["pos"])
        cw, sw = pw.cos(), pw.sin()
        cq, sq = pq.cos(), pq.sin()
        A = self.wr * cw - self.wi * sw
        Bm = self.wr * sw + self.wi * cw

        Fq = torch.cat([self._split(cq), self._split(sq)], -1)            # (B,H,T,2Mh)
        Fk = torch.cat([torch.cat([self._split(A), self._split(Bm)], -1),
                        torch.cat([self._split(Bm), -self._split(A)], -1)], 2)  # (B,H,2T,2Mh)
        K2 = (Fq @ Fk.transpose(2, 3)).view(B, H, T, 2, T)
        K2 = K2.masked_fill(causal_mask(T, z.device)[None, None, :, None, :], 0)

        if self.decay:
            cg = self._logdecay(z).cumsum(1).transpose(1, 2)               # (B,H,T)
            dm = (cg[:, :, :, None] - cg[:, :, None, :]).clamp(max=0).exp()  # (B,H,T,T)
            K2 = K2 * dm[:, :, :, None, :]

        v = self.V(z).view(B, T, H, dvh).transpose(1, 2)                  # (B,H,T,dvh)
        o = (K2.reshape(B, H, 2 * T, T) @ v).view(B, H, T, 2, dvh) / Mh

        if not st.get("empty", False):
            c1 = self._split(self.wr * cq + self.wi * sq)                 # (B,H,T,Mh)
            c2 = self._split(self.wr * sq - self.wi * cq)
            sr0, si0 = st["sr"], st["si"]                                 # (B,H,Mh,dvh)
            re = torch.einsum("bhtm,bhmj->bhtj", c1, sr0) + torch.einsum("bhtm,bhmj->bhtj", c2, si0)
            im = torch.einsum("bhtm,bhmj->bhtj", c1, si0) - torch.einsum("bhtm,bhmj->bhtj", c2, sr0)
            carry = torch.stack([re, im], 3) / Mh                         # (B,H,T,2,dvh)
            if self.decay:
                carry = carry * cg.exp()[:, :, :, None, None]
            o = o + carry

        u = torch.cat([o[:, :, :, 0].transpose(1, 2).reshape(B, T, self.dv),
                       o[:, :, :, 1].transpose(1, 2).reshape(B, T, self.dv)], -1)

        # closing state: e^{cg_T} S0 + sum_s e^{cg_T - cg_s} v_s e^{i pw_s}
        if self.decay:
            wdec = (cg[:, :, -1:] - cg).exp()                             # (B,H,T)
            vw = v * wdec[..., None]
            gT = cg[:, :, -1].exp()[:, :, None, None]
            sr = torch.einsum("bhtm,bhtj->bhmj", self._split(cw), vw) + gT * st["sr"]
            si = torch.einsum("bhtm,bhtj->bhmj", self._split(sw), vw) + gT * st["si"]
        else:
            sr = torch.einsum("bhtm,bhtj->bhmj", self._split(cw), v) + st["sr"]
            si = torch.einsum("bhtm,bhtj->bhmj", self._split(sw), v) + st["si"]
        return _gated_out(self, u, z), {"sr": sr, "si": si, "pos": st["pos"] + T,
                                        "empty": False}

    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        H, Mh, dvh = self.H, self.Mh, self.dvh
        p = state["pos"]
        pw = self.K(h_t) * self.theta + p * self.omega
        kq = self.Kq(z_t) if self.sepq else self.K(z_t)
        pq = kq * self.theta + p * self.omega
        v = self.V(z_t).view(B, H, 1, dvh)
        cw = pw.cos().view(B, H, Mh, 1)
        sw = pw.sin().view(B, H, Mh, 1)
        sr, si = state["sr"], state["si"]
        if self.decay:
            g = self._logdecay(z_t).exp()[:, :, None, None]
            sr, si = g * sr, g * si
        sr = torch.addcmul(sr, v, cw)
        si = torch.addcmul(si, v, sw)
        cq, sq = pq.cos(), pq.sin()
        c1 = (self.wr * cq + self.wi * sq).view(B, H, Mh)
        c2 = (self.wr * sq - self.wi * cq).view(B, H, Mh)
        re = torch.einsum("bhm,bhmj->bhj", c1, sr) + torch.einsum("bhm,bhmj->bhj", c2, si)
        im = torch.einsum("bhm,bhmj->bhj", c1, si) - torch.einsum("bhm,bhmj->bhj", c2, sr)
        u = torch.cat([re.reshape(B, self.dv), im.reshape(B, self.dv)], -1) / Mh
        return _gated_out(self, u, z_t), {"sr": sr, "si": si, "pos": p + 1, "empty": False}


# --------------------------------------------------------------------------- #
#  Layer: cfg-driven C head knobs + optional causal depthwise conv on z
# --------------------------------------------------------------------------- #
class GatedLayer(SCA2Layer):
    def __init__(self, cfg: LayerCfg, c_cls=None, d_cls=None):
        d_cls = d_cls or DHeadSepQPolar

        def c_cls(d, M, **kw):
            return CHeadGated(d, M, heads=cfg.c_heads, decay=cfg.c_decay,
                              sepq=cfg.c_sepq, decay_init=cfg.c_decay_init, **kw)
        super().__init__(cfg, c_cls, d_cls)
        self.k = cfg.conv
        if self.k:
            w = torch.zeros(cfg.d, 1, self.k)
            w[:, 0, -1] = 1.0                       # identity at init
            self.cw = nn.Parameter(w)

    def init_state(self, B, device, dtype=torch.float32):
        st = super().init_state(B, device, dtype)
        if self.k:
            st["cbuf"] = torch.zeros(B, self.k - 1, self.cfg.d, device=device, dtype=dtype)
        return st

    def _conv(self, z, buf):
        """Causal depthwise conv; `buf` holds the k-1 tokens preceding `z`."""
        zz = torch.cat([buf, z], 1)                                 # (B,k-1+T,d)
        zc = F.conv1d(zz.transpose(1, 2), self.cw, groups=self.cfg.d).transpose(1, 2)
        return zc, zz[:, -(self.k - 1):]

    def prefill(self, x, state=None):
        B = x.size(0)
        if state is None:
            state = self.init_state(B, x.device, x.dtype)
        z = self.n(x)
        new = {}
        if self.k:
            z, new["cbuf"] = self._conv(z, state["cbuf"])
        h = torch.empty_like(z)
        h[:, 0] = state["z_prev"]
        h[:, 1:] = z[:, :-1]
        uc, cs = self.c.prefill(z, h, state["c"])
        ud, ds = self.dh.prefill(z, h, state["d"])
        x = x + self.mix(torch.cat([uc, ud], -1))
        x = x + self.ff(self.fn(x))
        new.update({"c": cs, "d": ds, "z_prev": z[:, -1]})
        return x, new

    def step(self, x_t, state):
        z = self.n(x_t)
        new = {}
        if self.k:
            win = torch.cat([state["cbuf"], z[:, None]], 1)         # (B,k,d)
            z = (win * self.cw[:, 0, :].t()[None]).sum(1)
            new["cbuf"] = win[:, 1:]
        h = state["z_prev"]
        uc, cs = self.c.step(z, h, state["c"])
        ud, ds = self.dh.step(z, h, state["d"])
        y = x_t + self.mix(torch.cat([uc, ud], -1))
        y = y + self.ff(self.fn(y))
        new.update({"c": cs, "d": ds, "z_prev": z})
        return y, new


def _forced(**kw):
    class L(GatedLayer):
        def __init__(self, cfg, c_cls=None, d_cls=None):
            super().__init__(_replace(cfg, **kw), c_cls, d_cls)
    return L


# cfg-driven: knobs come from LayerCfg / pretrain.py flags
register("gc", CHeadGated, DHeadSepQPolar, arch=True, layer_cls=GatedLayer,
         note="gated multi-head C head, knobs from cfg (ARCH)")
register("gc_cc", CHeadGated, DHeadSepQPolar, arch=True, layer_cls=GatedLayer, wrap=_cw,
         note="gated multi-head C head, knobs from cfg + compile")
# all knobs off: must be iso with v3polar (`python -m sca2.iso gc_plain --against v3polar`)
register("gc_plain", CHeadGated, DHeadSepQPolar, arch=True,
         layer_cls=_forced(c_heads=1, c_decay=False, c_sepq=False, conv=0),
         note="gated C head with every knob off == v3polar")
# everything on
register("gcfull", CHeadGated, DHeadSepQPolar, arch=True,
         layer_cls=_forced(c_heads=4, c_decay=True, c_sepq=True, conv=4, gated_read=True),
         note="4 heads + decay + Kq + conv4 + gated read (ARCH)")
register("gcfull_cc", CHeadGated, DHeadSepQPolar, arch=True, wrap=_cw,
         layer_cls=_forced(c_heads=4, c_decay=True, c_sepq=True, conv=4, gated_read=True),
         note="4 heads + decay + Kq + conv4 + gated read + compile")
