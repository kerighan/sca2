"""
Gated DeltaNet, as a same-shaped baseline.

The core is NOT reimplemented: it is `flash-linear-attention`'s own reference
(`fla/ops/gated_delta_rule/naive.py`), loaded by file path so the package's
Triton-dependent `__init__` never runs. Their chunked and recurrent forms agree
to ~1e-6 on CPU, which is the property this baseline needs -- it means the
parallel training path and the O(1) decode path are the same function, checked
by the authors, not by me. No straw man.

Around it, the canonical layer structure from `fla/layers/gated_deltanet.py`:
short causal conv on q/k/v, L2-normalized q/k, a Mamba-style gate
`g = -exp(A_log) . softplus(a_proj(x) + dt_bias)`, `beta = sigmoid(b_proj(x))`,
a gated RMSNorm read-out and an output projection.

How it differs from our D head, which is why it is worth comparing against:

  * the state is an associative MATRIX (K, V) built from outer products, not a
    per-channel vector;
  * the write is ERROR-CORRECTING -- `v - h^T k` removes whatever is already
    stored at that key before writing, where ours only ever adds;
  * the gate is a real scalar per head, where ours is complex per (m, g). Its
    selectivity comes from the delta rule; ours from phase interference.

Two opposite solutions to the same problem, which makes it a useful judge.
"""
import importlib.util
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ref import _rms
from .registry import register
from .versions.v1_quad_scan import CHeadQuad
from .compiled import wrap as _cw

def _fla_naive_path():
    """Locate fla's own naive reference without importing fla/__init__ (which pulls Triton).

    $SCA2_FLA_NAIVE overrides; otherwise the installed package is located by spec so the
    path follows the environment instead of one machine's site-packages.
    """
    env = os.environ.get("SCA2_FLA_NAIVE")
    if env:
        return env
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.submodule_search_locations:
        return ""
    return os.path.join(list(spec.submodule_search_locations)[0],
                        "ops", "gated_delta_rule", "naive.py")


_FLA = _fla_naive_path()


def _load_ref():
    if not _FLA or not os.path.exists(_FLA):
        raise ImportError(
            "fla reference not found (pip install flash-linear-attention, "
            "or set SCA2_FLA_NAIVE to .../fla/ops/gated_delta_rule/naive.py); "
            f"looked at {_FLA!r}")
    import sys
    spec = importlib.util.spec_from_file_location("_gdn_naive", _FLA)
    m = importlib.util.module_from_spec(spec)
    # register before executing: torch.compile resolves the defining module of
    # every traced function, and a spec-loaded module that is not in sys.modules
    # makes Dynamo raise ModuleNotFoundError.
    sys.modules["_gdn_naive"] = m
    spec.loader.exec_module(m)          # bypasses fla/__init__ and Triton
    return m


_ref = _load_ref()


def _load_triton():
    """fla's own fused Triton kernels, or None where they cannot run.

    Every GDN number in this repo before the Spark came from the naive PyTorch
    reference above, because fla's Triton kernels do not build on sm_75. They do on
    Blackwell, and they are worth 1.66x at d=1024 (SPARK.md §9) -- training or timing
    LapA against the reference is racing a crippled baseline. $SCA2_GDN_KERNEL takes
    "triton" (default when importable), or "naive" to force the reference back."""
    if os.environ.get("SCA2_GDN_KERNEL", "auto") == "naive":
        return None
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        return chunk_gated_delta_rule
    except Exception:
        return None


_triton = _load_triton()


class ShortConv(nn.Module):
    """Depthwise causal conv, kernel 4, with a cache for decode."""

    def __init__(self, dim, k=4):
        super().__init__()
        self.dim, self.k = dim, k
        self.w = nn.Parameter(torch.randn(dim, 1, k) * (1.0 / math.sqrt(k)))

    def forward(self, x):                                   # (B,T,D)
        y = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return F.silu(F.conv1d(y, self.w, groups=self.dim).transpose(1, 2))

    def step(self, x_t, buf):
        """buf: (B, k-1, D) of previous inputs."""
        win = torch.cat([buf, x_t[:, None]], 1)             # (B,k,D)
        y = F.silu((win.transpose(1, 2) * self.w.squeeze(1)).sum(-1))
        return y, win[:, 1:]


class GatedDeltaNet(nn.Module):
    def __init__(self, d, heads=4, head_k=32, expand_v=2.0, conv_k=4):
        super().__init__()
        self.d, self.H = d, heads
        self.dk = head_k
        self.dv = int(head_k * expand_v)
        self.key_dim, self.value_dim = heads * self.dk, heads * self.dv
        self.q = nn.Linear(d, self.key_dim, False)
        self.k = nn.Linear(d, self.key_dim, False)
        self.v = nn.Linear(d, self.value_dim, False)
        self.a = nn.Linear(d, heads, False)                 # gate
        self.b = nn.Linear(d, heads, False)                 # beta
        self.gp = nn.Linear(d, self.value_dim, False)       # output gate
        self.o = nn.Linear(self.value_dim, d, False)
        self.cq, self.ck, self.cv = (ShortConv(self.key_dim, conv_k),
                                     ShortConv(self.key_dim, conv_k),
                                     ShortConv(self.value_dim, conv_k))
        A = torch.empty(heads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm = nn.LayerNorm(self.dv)
        self.conv_k = conv_k

    # ---- pieces ----------------------------------------------------------- #
    def _gates(self, x):
        g = -torch.exp(self.A_log.float()) * F.softplus(self.a(x).float() + self.dt_bias)
        return g, self.b(x).float().sigmoid()

    def _read(self, o, x, B, T):
        o = self.o_norm(o) * F.silu(self.gp(x)).view(B, T, self.H, self.dv)
        return self.o(o.reshape(B, T, self.value_dim))

    def init_state(self, B, device, dtype):
        z = lambda n: torch.zeros(B, self.conv_k - 1, n, device=device, dtype=dtype)
        return {"h": torch.zeros(B, self.H, self.dk, self.dv, device=device, dtype=dtype),
                "cq": z(self.key_dim), "ck": z(self.key_dim), "cv": z(self.value_dim)}

    # ---- prefill (fla chunked reference) ---------------------------------- #
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
        if _triton is not None and x.is_cuda:
            # bf16 activations, fp32 gates and fp32 state -- the same split the layer
            # itself uses, so neither arm is handicapped by dtype.
            dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.bfloat16
            o, h = _triton(q.to(dt), k.to(dt), v.to(dt), g.float(), beta.to(dt),
                           initial_state=None if h0 is None else h0.float(),
                           output_final_state=True)
        else:
            o, h = _ref.naive_chunk_gated_delta_rule(
                q, k, v, g, beta, chunk_size=64,
                initial_state=h0, output_final_state=True)
        y = self._read(o.to(x.dtype), x, B, T)
        tail = lambda z_, n: z_[:, -(self.conv_k - 1):] if T >= self.conv_k - 1 else \
            F.pad(z_, (0, 0, self.conv_k - 1 - T, 0))
        return y, {"h": h.to(x.dtype), "cq": tail(self.q(x), 0),
                   "ck": tail(self.k(x), 0), "cv": tail(self.v(x), 0)}

    # ---- decode (fla recurrent reference, one token) ----------------------- #
    def step(self, x_t, state):
        B = x_t.size(0)
        qr, cq = self.cq.step(self.q(x_t), state["cq"])
        kr, ck = self.ck.step(self.k(x_t), state["ck"])
        vr, cv = self.cv.step(self.v(x_t), state["cv"])
        q = F.normalize(qr.view(B, 1, self.H, self.dk), dim=-1)
        k = F.normalize(kr.view(B, 1, self.H, self.dk), dim=-1)
        v = vr.view(B, 1, self.H, self.dv)
        g, beta = self._gates(x_t[:, None])
        o, h = _ref.naive_recurrent_gated_delta_rule(
            q, k, v, beta, g, initial_state=state["h"].float(), output_final_state=True)
        y = self._read(o.to(x_t.dtype), x_t[:, None], B, 1)[:, 0]
        return y, {"h": h.to(x_t.dtype), "cq": cq, "ck": ck, "cv": cv}


class GDNLayer(nn.Module):
    """Same wrapper as SCA2Layer: norm -> mixer -> residual -> norm -> FFN."""

    def __init__(self, cfg, heads=4, head_k=32, expand_v=2.0):
        super().__init__()
        d = cfg.d
        self.cfg = cfg
        self.n = nn.LayerNorm(d)
        self.mix = GatedDeltaNet(d, heads, head_k, expand_v)
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
        x = x_t + y
        return x + self.ff(self.fn(x)), st


class GDNLayerMatched(GDNLayer):
    """Registry adapter: (cfg, c_cls, d_cls) signature. The head shape now comes
    from cfg, whose DEFAULTS are the values this class used to hardcode -- the
    configuration that matches our layer budget (184,686 vs 185,984, -0.70%).
    Overriding them breaks that parameter match, which is the point when the
    comparison being made is against state size rather than parameter count."""

    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__(cfg, heads=cfg.gdn_heads, head_k=cfg.gdn_head_k,
                         expand_v=cfg.gdn_expand_v)


register("gdn", CHeadQuad, None, arch=True, layer_cls=GDNLayerMatched,
         note="Gated DeltaNet on fla's own reference (ARCH: different function)")
register("gdn_cc", CHeadQuad, None, arch=True, layer_cls=GDNLayerMatched, wrap=_cw,
         note="Gated DeltaNet + torch.compile")
