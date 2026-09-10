"""Baseline layers speaking LapA's protocol, so `lapa.LM` can stack them.

  GDNLayer        Gated DeltaNet on flash-linear-attention's own reference
                  implementation (loaded from the installed `fla` package), with
                  the same norm -> mixer -> FFN wrapper as LaplaceAttention.
  AttentionLayer  a pre-norm causal transformer block (ceiling for copy tasks:
                  it carries the whole prefix, O(T) memory).

`build(name, d, **kw)` constructs by name: "lapa", "gdn", "attn".
"""

from __future__ import annotations

import importlib.util
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer import LaplaceAttention, LaplaceConfig


# --------------------------------------------------------------------------- #
#  Gated DeltaNet (fla reference)
# --------------------------------------------------------------------------- #
def _load_fla_naive():
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "flash-linear-attention (`fla`) is not installed; GDN baseline unavailable"
        )
    path = spec.submodule_search_locations[0] + "/ops/gated_delta_rule/naive.py"
    s = importlib.util.spec_from_file_location("_gdn_naive", path)
    m = importlib.util.module_from_spec(s)
    sys.modules["_gdn_naive"] = m  # Dynamo needs the module registered
    s.loader.exec_module(m)
    return m


class _ShortConv(nn.Module):
    def __init__(self, dim, k=4):
        super().__init__()
        self.dim, self.k = dim, k
        self.w = nn.Parameter(torch.randn(dim, 1, k) * (1.0 / math.sqrt(k)))

    def forward(self, x):
        y = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return F.silu(F.conv1d(y, self.w, groups=self.dim).transpose(1, 2))

    def step(self, x_t, buf):
        win = torch.cat([buf, x_t[:, None]], 1)
        return F.silu((win.transpose(1, 2) * self.w.squeeze(1)).sum(-1)), win[:, 1:]


class GatedDeltaNet(nn.Module):
    def __init__(self, d, heads=4, head_k=32, expand_v=1.0, conv_k=4):
        super().__init__()
        self._ref = _load_fla_naive()
        self.d, self.H, self.dk = d, heads, head_k
        self.dv = int(head_k * expand_v)
        self.key_dim, self.value_dim = heads * self.dk, heads * self.dv
        self.q = nn.Linear(d, self.key_dim, False)
        self.k = nn.Linear(d, self.key_dim, False)
        self.v = nn.Linear(d, self.value_dim, False)
        self.a = nn.Linear(d, heads, False)
        self.b = nn.Linear(d, heads, False)
        self.gp = nn.Linear(d, self.value_dim, False)
        self.o = nn.Linear(self.value_dim, d, False)
        self.cq, self.ck, self.cv = (
            _ShortConv(self.key_dim, conv_k),
            _ShortConv(self.key_dim, conv_k),
            _ShortConv(self.value_dim, conv_k),
        )
        A = torch.empty(heads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(
            torch.rand(heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm = nn.LayerNorm(self.dv)
        self.conv_k = conv_k

    def _gates(self, x):
        g = -torch.exp(self.A_log.float()) * F.softplus(
            self.a(x).float() + self.dt_bias
        )
        return g, self.b(x).float().sigmoid()

    def _read(self, o, x, B, T):
        o = self.o_norm(o) * F.silu(self.gp(x)).view(B, T, self.H, self.dv)
        return self.o(o.reshape(B, T, self.value_dim))

    def init_state(self, B, device):
        z = lambda n: torch.zeros(B, self.conv_k - 1, n, device=device)
        return {
            "h": torch.zeros(B, self.H, self.dk, self.dv, device=device),
            "cq": z(self.key_dim),
            "ck": z(self.key_dim),
            "cv": z(self.value_dim),
        }

    def prefill(self, x, state=None):
        B, T, _ = x.shape
        st = state if state is not None else self.init_state(B, x.device)
        q = F.normalize(self.cq(self.q(x)).view(B, T, self.H, self.dk), dim=-1)
        k = F.normalize(self.ck(self.k(x)).view(B, T, self.H, self.dk), dim=-1)
        v = self.cv(self.v(x)).view(B, T, self.H, self.dv)
        g, beta = self._gates(x)
        o, h = self._ref.naive_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=64,
            initial_state=st["h"] if state is not None else None,
            output_final_state=True,
        )
        tail = (
            lambda z_: z_[:, -(self.conv_k - 1) :]
            if T >= self.conv_k - 1
            else F.pad(z_, (0, 0, self.conv_k - 1 - T, 0))
        )
        return self._read(o.to(x.dtype), x, B, T), {
            "h": h.float(),
            "cq": tail(self.q(x)),
            "ck": tail(self.k(x)),
            "cv": tail(self.v(x)),
        }

    def step(self, x_t, state):
        B = x_t.size(0)
        qr, cq = self.cq.step(self.q(x_t), state["cq"])
        kr, ck = self.ck.step(self.k(x_t), state["ck"])
        vr, cv = self.cv.step(self.v(x_t), state["cv"])
        q = F.normalize(qr.view(B, 1, self.H, self.dk), dim=-1)
        k = F.normalize(kr.view(B, 1, self.H, self.dk), dim=-1)
        v = vr.view(B, 1, self.H, self.dv)
        g, beta = self._gates(x_t[:, None])
        o, h = self._ref.naive_recurrent_gated_delta_rule(
            q, k, v, beta, g, initial_state=state["h"].float(), output_final_state=True
        )
        return self._read(o.to(x_t.dtype), x_t[:, None], B, 1)[:, 0], {
            "h": h.float(),
            "cq": cq,
            "ck": ck,
            "cv": cv,
        }


class GDNLayer(nn.Module):
    """norm -> GatedDeltaNet -> residual -> norm -> FFN -> residual (LapA's wrapper)."""

    def __init__(self, d, heads=4, head_k=32, expand_v=1.0, ff=256):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.mix = GatedDeltaNet(d, heads, head_k, expand_v)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def init_state(self, B, device):
        return self.mix.init_state(B, device)

    def forward(self, x):
        return self.prefill(x)[0]

    def prefill(self, x, state=None):
        y, st = self.mix.prefill(self.n(x), state)
        x = x + y
        return x + self.ff(self.fn(x)), st

    def step(self, x_t, state):
        y, st = self.mix.step(self.n(x_t), state)
        x = x_t + y
        return x + self.ff(self.fn(x)), st

    def state_floats(self):
        m = self.mix
        return m.H * m.dk * m.dv + (m.conv_k - 1) * (2 * m.key_dim + m.value_dim)


# --------------------------------------------------------------------------- #
#  Causal attention block (ceiling)
# --------------------------------------------------------------------------- #
class AttentionLayer(nn.Module):
    """Pre-norm causal self-attention + FFN. State = the KV cache (grows with T)."""

    def __init__(self, d, heads=4, ff=256, max_len=4096):
        super().__init__()
        self.d, self.H, self.hd = d, heads, d // heads
        self.n = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, False)
        self.o = nn.Linear(d, d, False)
        self.pos = nn.Embedding(max_len, d)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def init_state(self, B, device):
        return {
            "k": torch.zeros(B, self.H, 0, self.hd, device=device),
            "v": torch.zeros(B, self.H, 0, self.hd, device=device),
            "pos": torch.zeros((), device=device, dtype=torch.long),
        }

    def forward(self, x):
        return self.prefill(x)[0]

    def _attend(self, x, st, causal):
        B, T, _ = x.shape
        p = torch.arange(T, device=x.device) + st["pos"]
        z = self.n(x + self.pos(p)[None])
        q, k, v = (
            self.qkv(z).view(B, T, 3, self.H, self.hd).transpose(1, 3).unbind(2)
        )  # (B,H,T,hd)
        k = torch.cat([st["k"], k], 2)
        v = torch.cat([st["v"], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, self.d))
        return x + self.ff(self.fn(x)), {"k": k, "v": v, "pos": st["pos"] + T}

    def prefill(self, x, state=None):
        st = state if state is not None else self.init_state(x.size(0), x.device)
        assert st["k"].size(2) == 0, (
            "AttentionLayer.prefill with a non-empty cache is not supported"
        )
        return self._attend(x, st, causal=True)

    def step(self, x_t, state):
        y, st = self._attend(x_t[:, None], state, causal=False)
        return y[:, 0], st


# --------------------------------------------------------------------------- #
def build(name: str, d: int, **kw) -> nn.Module:
    """'lapa' (LaplaceConfig kwargs), 'gdn' (heads, head_k, expand_v, ff), 'attn' (heads, ff, max_len)."""
    if name == "lapa":
        return LaplaceAttention(LaplaceConfig(d=d, **kw))
    if name == "gdn":
        return GDNLayer(d, **kw)
    if name == "attn":
        return AttentionLayer(d, **kw)
    raise ValueError(f"unknown layer {name!r}")
