"""
SCA2 layer — frozen reference semantics.

Extracted verbatim from bench_tinypython.py (SCA2.forward body, minus embedding
and output head). The maths here is the CONTRACT: every optimized variant must
reproduce it bit-for-bit-ish (see iso.py for tolerances).

Known-but-frozen quirks of the reference (do NOT "fix" in a variant — a fix is a
semantic change and must be a separate, explicit commit):
  * CHead omega = 2*pi*k/M makes the positional phase periodic with period M,
    so positions p and p+M are positionally indistinguishable.
  * theta is zero-init, so CHead.K receives exactly zero gradient at step 0.
  * DHead's imaginary state receives no input injection (only sr gets +v).

Every module exposes three entry points with identical semantics:
    forward(z, h)                  -> u                 (prefill, no state out)
    prefill(z, h, state=None)      -> (u, state)        (prefill, state out)
    step(z_t, h_t, state)          -> (u_t, state)      (one token)

`state` is a plain dict of tensors so variants can add fields freely.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def _rms(u, eps=1e-6):
    return u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)


def _gated_out(head, u, z):
    """RMS read-out, optionally gated by a projection of the input.

    Both reference architectures (SeqCond and Gated DeltaNet) end their mixer
    with a GATED norm; SCA2 had a bare one. The gate lets the network mute the
    mixer's contribution token by token -- useful exactly when the state holds
    nothing relevant for the current position.
    """
    if getattr(head, "rms_read", True):
        u = _rms(u)
    else:
        # RAW read: keep the magnitude. The RMS above erases the one signal that
        # says whether anything matched -- a hash read on a key new to the window
        # is a small random mixture, an exact repeat is a full-size value -- and
        # the layer downstream is linear (mix), so nothing can recover it. A
        # learned per-feature scale replaces the normalisation (arch_cdelta.CHeadDeltaRaw).
        u = u * head.rscale
    return u * F.silu(head.rgate(z)) if getattr(head, "gated_read", False) else u


# --------------------------------------------------------------------------- #
#  C head positional frequency grids
# --------------------------------------------------------------------------- #
def freq_grid(name, M, max_len=128, base=10000.0, slow_frac=0.0):
    """Positional angular frequencies for the C head.

    Not a free choice: this grid decides what the C head computes AT INIT.
    From the scalar-kernel form (DERIVATION.md section 1), with theta=0, wr=1,
    wi=0 the score collapses to a function of the lag alone,

        kappa[n] = (1/M) sum_m cos(n . omega_m),   n = s - t

    which is a Dirichlet kernel. So the grid is really a choice of what delta
    the head starts as, and the sidelobe budget is conserved (total off-peak
    mass is 1.0 for any equispaced grid) -- only its SHAPE changes.

      dft   2*pi*k/M -- ORIGINAL. The geometric sum is exact:
            sum_m e^{i 2 pi m n / M} = M . delta(n mod M). A perfectly clean
            delta -- and therefore the whole sidelobe budget lands in one place,
            an alias spike of height 1.0 at lag M. Measured at M=64: kappa[0]=1,
            kappa[n]=0 for 0<n<64, kappa[64]=1. The head attends to the current
            token AND, just as strongly, to the token 64 back.
      len   2*pi*k/L for a declared context L -- a real DFT basis over [0, L)
            instead of over [0, M). Same conserved sidelobe mass, but spread as
            Dirichlet ripple instead of concentrated: at M=64, L=128 the largest
            off-peak term is 0.016 and there is no alias inside the window.
            (For L = 2M this is identical to pi*k/M.)
      rope  pi * base^(-k/(M-1)) -- geometric, RoPE-style. Never aliases
            exactly (period 2*base), but the spacing is so uneven that it is not
            a delta at all: kappa[1] = 0.808 and the off-peak mass is 50x. The
            basis is also numerically near-degenerate (rank 49/128 at T=128).

    MEASURED (sca2/ab_freq.py, 1500 steps, compact vocab, val loss vs `dft`):

        dft   0.8368   baseline
        len   0.7670   -0.070
        rope  0.7080   -0.129     <- best, at both lr 1e-3 and 3e-4

    The delta-at-init argument above is sound about what the grid computes at
    step 0, and it does NOT predict trained quality: `rope`, which the theory
    "rules out" for having 50x the off-peak mass, wins by roughly double the
    margin of `len`. In hindsight a delta means the head starts as near-identity
    and must learn every lag from scratch, whereas a broad kernel has immediate
    access to a range of lags -- the off-peak mass is coverage, not noise. The
    aliasing does cost something too (`len` beats `dft` consistently), it is just
    not the dominant term.

    `rope` is also the only grid whose period (2e4) exceeds any context this
    layer would plausibly decode, which matters because decode cost is O(1) in
    the context length (bench_decode_scaling.py) while `dft`/`len` alias after
    64/128 positions.

    Caveat: one seed per arm. The ranking is stable across two learning rates,
    not yet across seeds.
    """
    k = torch.arange(M, dtype=torch.float32)
    if name == "dft":
        return 2 * math.pi * k / M
    if name == "len":
        return 2 * math.pi * k / max_len
    if name == "rope":
        n_slow = int(round(slow_frac * M))
        n_fast = M - n_slow
        kf = torch.arange(n_fast, dtype=torch.float32)
        fast = math.pi * (base ** (-kf / max(n_fast - 1, 1)))
        if n_slow == 0:
            return fast
        # slow integrators: periods 2*max_len .. 20*max_len (what a base-1e4 grid gave at T=1024:
        # 47/190 modes carrying 55-85% of the trained LM's state energy)
        p = torch.logspace(math.log10(2 * max_len), math.log10(20 * max_len), n_slow)
        return torch.cat([fast, 2 * math.pi / p])
    raise ValueError(f"unknown freq grid {name!r}")


# --------------------------------------------------------------------------- #
#  C head
# --------------------------------------------------------------------------- #
class CHeadBase(nn.Module):
    """Parameter container. Shared by every variant so state_dicts are portable."""

    def __init__(self, d, M, freq="dft", theta_scale=0.0, max_len=128, dv=None,
                 gated_read=False, rope_base=10000.0, slow_frac=0.0):
        super().__init__()
        self.d, self.M, self.dv = d, M, (d // 2 if dv is None else dv)
        self.freq, self.theta_scale, self.max_len = freq, theta_scale, max_len
        self.K = nn.Linear(d, M, False)
        self.V = nn.Linear(d, self.dv, False)
        # theta scales the CONTENT-dependent part of the phase. At theta == 0 the
        # gradient w.r.t. K is exactly zero (dL/dK = dL/dpw * theta * h), so K is
        # dead at step 0 and only starts learning once theta has drifted. A small
        # non-zero init keeps the near-positional warm start and unblocks K.
        self.theta = nn.Parameter(
            torch.zeros(M) if theta_scale == 0.0 else theta_scale * torch.randn(M))
        self.register_buffer("omega", freq_grid(freq, M, max_len, rope_base, slow_frac))
        self.wr = nn.Parameter(torch.ones(M))
        self.wi = nn.Parameter(torch.zeros(M))
        self.gated_read = gated_read
        if gated_read:
            self.rgate = nn.Linear(d, 2 * self.dv, bias=False)

    def max_unaliased_len(self):
        """Longest context over which positional phases stay distinct."""
        w = self.omega[self.omega > 0]
        return float("inf") if w.numel() == 0 else float(2 * math.pi / w.min())

    def init_state(self, B, device, dtype):
        return {
            "sr": torch.zeros(B, self.M, self.dv, device=device, dtype=dtype),
            "si": torch.zeros(B, self.M, self.dv, device=device, dtype=dtype),
            "pos": 0,
        }


class CHeadRef(CHeadBase):
    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        if state is None:
            state = self.init_state(B, z.device, z.dtype)
        p0 = state["pos"]
        p = torch.arange(p0, p0 + T, device=z.device, dtype=z.dtype)

        pw = self.K(h) * self.theta + p[:, None] * self.omega
        pq = self.K(z) * self.theta + p[:, None] * self.omega
        v = self.V(z)
        zr = v[:, :, None] * pw.cos()[:, :, :, None]
        zi = v[:, :, None] * pw.sin()[:, :, :, None]
        sr = zr.cumsum(1) + state["sr"][:, None]
        si = zi.cumsum(1) + state["si"][:, None]
        qr, qi = pq.cos()[:, :, :, None], -pq.sin()[:, :, :, None]
        rr, ii = sr * qr - si * qi, sr * qi + si * qr
        u = torch.cat(
            [(rr * self.wr[None, None, :, None] - ii * self.wi[None, None, :, None]).mean(2),
             (rr * self.wi[None, None, :, None] + ii * self.wr[None, None, :, None]).mean(2)], -1)
        new = {"sr": sr[:, -1], "si": si[:, -1], "pos": p0 + T}
        return _rms(u), new

    def step(self, z_t, h_t, state):
        p = float(state["pos"])
        pw = self.K(h_t) * self.theta + p * self.omega          # (B,M)
        pq = self.K(z_t) * self.theta + p * self.omega
        v = self.V(z_t)                                          # (B,dv)
        sr = state["sr"] + v[:, None, :] * pw.cos()[:, :, None]
        si = state["si"] + v[:, None, :] * pw.sin()[:, :, None]
        qr, qi = pq.cos()[:, :, None], -pq.sin()[:, :, None]     # (B,M,1)
        rr, ii = sr * qr - si * qi, sr * qi + si * qr
        wr, wi = self.wr[None, :, None], self.wi[None, :, None]
        u = torch.cat([(rr * wr - ii * wi).mean(1), (rr * wi + ii * wr).mean(1)], -1)
        return _rms(u), {"sr": sr, "si": si, "pos": state["pos"] + 1}


# --------------------------------------------------------------------------- #
#  D head
# --------------------------------------------------------------------------- #
class DHeadBase(nn.Module):
    def __init__(self, d, M=16, G=8, dv=None, max_len=None, delta_rule=False,
                 gated_read=False):
        super().__init__()
        self.d, self.M, self.G, self.dv = d, M, G, (d // 2 if dv is None else dv)
        self.gated_read = gated_read
        if gated_read:
            self.rgate = nn.Linear(d, 2 * self.dv, bias=False)
        # only the heads whose temporal weighting is absolute in position use it
        self.max_len = max_len
        self.delta_rule = delta_rule
        assert self.dv % G == 0
        self.gs = self.dv // G
        self.V = nn.Linear(d, self.dv, False)
        self.gr = nn.Linear(d, M * G)
        self.gi = nn.Linear(d, M * G)
        self.qr = nn.Linear(d, M * self.dv, False)
        self.qi = nn.Linear(d, M * self.dv, False)

    def init_state(self, B, device, dtype):
        return {
            "sr": torch.zeros(B, self.M, self.dv, device=device, dtype=dtype),
            "si": torch.zeros(B, self.M, self.dv, device=device, dtype=dtype),
        }


class DHeadRef(DHeadBase):
    def forward(self, z, h):
        return self.prefill(z, h)[0]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        if state is None:
            state = self.init_state(B, z.device, z.dtype)
        gs, v = self.gs, self.V(z)
        gr = torch.tanh(self.gr(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        gi = torch.tanh(self.gi(h)).view(B, T, self.M, self.G) / math.sqrt(2)
        sr, si, out = state["sr"], state["si"], []
        for t in range(T):
            rg, ig = sr.view(B, self.M, self.G, gs), si.view(B, self.M, self.G, gs)
            ar, ai = gr[:, t, :, :, None], gi[:, t, :, :, None]
            sr = (ar * rg - ai * ig).reshape(B, self.M, self.dv) + v[:, t, None, :]
            si = (ar * ig + ai * rg).reshape(B, self.M, self.dv)
            qr = self.qr(z[:, t]).view(B, self.M, self.dv)
            qi = self.qi(z[:, t]).view(B, self.M, self.dv)
            qn = torch.sqrt(qr.square() + qi.square() + 1e-6)
            qr, qi = qr / qn, qi / qn
            out.append(torch.cat([(sr * qr + si * qi).mean(1), (si * qr - sr * qi).mean(1)], -1))
        return _rms(torch.stack(out, 1)), {"sr": sr, "si": si}

    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        gs = self.gs
        v = self.V(z_t)
        gr = torch.tanh(self.gr(h_t)).view(B, self.M, self.G) / math.sqrt(2)
        gi = torch.tanh(self.gi(h_t)).view(B, self.M, self.G) / math.sqrt(2)
        rg = state["sr"].view(B, self.M, self.G, gs)
        ig = state["si"].view(B, self.M, self.G, gs)
        ar, ai = gr[:, :, :, None], gi[:, :, :, None]
        sr = (ar * rg - ai * ig).reshape(B, self.M, self.dv) + v[:, None, :]
        si = (ar * ig + ai * rg).reshape(B, self.M, self.dv)
        qr = self.qr(z_t).view(B, self.M, self.dv)
        qi = self.qi(z_t).view(B, self.M, self.dv)
        qn = torch.sqrt(qr.square() + qi.square() + 1e-6)
        qr, qi = qr / qn, qi / qn
        u = torch.cat([(sr * qr + si * qi).mean(1), (si * qr - sr * qi).mean(1)], -1)
        return _rms(u), {"sr": sr, "si": si}


# --------------------------------------------------------------------------- #
#  Layer
# --------------------------------------------------------------------------- #
@dataclass
class LayerCfg:
    d: int = 128
    Mc: int = 64
    Md: int = 16
    G: int = 8
    ff: int = 256
    # Defaults ARE the original semantics: ref.py stays the frozen contract until
    # a measurement says otherwise. Candidate changes go through LayerCfg.cand().
    freq: str = "dft"
    theta_scale: float = 0.0
    max_len: int = 128
    # error-correcting write in the D head (see arch_sepq.DHeadSepQPolar)
    delta_rule: bool = False
    # gated read-out on each head (both reference architectures have one)
    gated_read: bool = False
    # value width of BOTH heads' state. None -> d//2, the original. The state is
    # M.dv and the read is M.dv, so this is the one shape lever that touches the
    # state size, the decay einsum's inner width and the read at once -- and
    # unlike Mc it shrinks the VALUE dimension, not the addressing capacity.
    dv: int = None
    # --- gated C head knobs (arch_gatedc.py); all off == CHeadQuad exactly ---
    c_heads: int = 1        # split Mc (and dv) into this many heads
    c_decay: bool = False   # data-dependent scalar forget gate per head
    c_decay_init: str = "gdn"   # "gdn" (A~U(1,16), dt~logU) | "soft" (A=1, dt=0.01)
    c_sepq: bool = False    # separate read-key projection Kq (else K is shared)
    conv: int = 0           # causal depthwise conv width on z before the heads (0 = none)
    # --- GDN baseline head shape (arch_gdn.GDNLayerMatched) ------------------
    # Defaults are exactly what that class used to hardcode, so nothing moves.
    # Exposed because the state GDN carries is gdn_heads.head_k^2.expand_v, and
    # comparing it against SCA2 only means something at a MATCHED state size --
    # raising Mc grows SCA2's state fast, so GDN needs the same lever.
    Ls: int = 16            # window of the short dft C head (arch_short.py), in tokens
    rope_base: float = 10000.0   # long-head grid omega_m = pi * base^(-m/(M-1)); unaliased range 2*base. Copy bench: base ~ T wins
    layer_scale: bool = False    # learned gain on each residual branch (lapa path)
    ls_mix_init: float = 1.0
    ls_ff_init: float = 1.0
    ls_mix_per_channel: bool = False
    w_antipodal: float = 0.0
    gdn_gate: bool = False       # GDN-style readout: LayerNorm * silu(Linear(x)), per channel
    decay_input: bool = False    # data-dependent forgetting on the lapa path: lam becomes a
    #   function of the token instead of a constant per mode. See lapa.layer.
    beta_init: float = -2.0      # erase-gate bias: sigmoid(-2) = 0.12 at init. GDN's b has
    #   NO bias, so its erase gate starts at sigmoid(0) = 0.5 -- 4x bolder than ours, and its
    #   delta rule ablates 3x heavier (+4.51 vs our +1.42).
    conv_silu: bool = False      # SiLU after the causal conv (lapa path), as GDN does
    long_groups: int = 1         # same, on the LONG head (lapa path). This is wg2's
    #   experiment, which LOST at d=128 (+0.065); see lapa.layer.LaplaceConfig.long_groups.
    short_groups: int = 1        # spectral read weights of the SHORT head, per group of value
    #   channels (lapa path). 1 = one shared L-tap filter for all dv channels. See
    #   lapa.layer.LaplaceConfig.short_groups -- wg2 refuted this on the LONG head, the short
    #   head is where the loss lives and has never been tried.
    beta_groups: int = 1         # erase-gate granularity on the lapa path; 3 = per spectral
    #   band (slow integrators / persistent-fast / damped). See lapa.layer.LaplaceConfig.
    kv_gate_pc: bool = False     # per-channel key-verification gate (ga, gb as 2*dv vectors)
    kv_dk: int = 0               # key-verification width for the lapa path (0 = off); sca2's
    #   cshort_damphkv hardcodes 16. See lapa.layer.LaplaceConfig.kv_dk.
    persist: float = 0.5         # fraction of long-head modes starting persistent (lambda = 0)
    learn_persist: bool = False  # no hard pin: lambda = lam_max*sigmoid(a), the gradient decides
    #   the persistent/damped split and `persist` only sets where it STARTS. Also removes the
    #   hard clamp, which has zero gradient above its bound -- see SPARK.md §9.
    rope_min_period: float = None  # shortest period in the fast rope grid; None = 2 (historical).
    #   2*Ls starts the long head where the short head's exact window ends instead of overlapping
    #   it. At M=256/Ls=64 the overlap wastes 115 of 256 modes; see lapa.layer.rope_grid.
    slow_frac: float = 0.0       # fraction of modes kept as slow integrators (periods 2..20 x max_len); LM used ~1/4 of a base-1e4 grid that way
    damp_mem: tuple = None       # init memories of the damped modes; None -> (Ls, 32*Ls) (window-aligned)
    lam_max: float = None        # decay cap; None -> 1/Ls (a damped mode never forgets faster than the window remembers)
    lam_free: bool = False       # FREE MODES: lambda = exp(a), no pin, no cap but the fp32 safety
    #   ceiling. The trained d=1024 checkpoint has 41-100% of each layer's free modes sitting
    #   exactly at the softplus clamp (zero gradient, never escapes) and the other half pinned at
    #   0, so the realised spectrum is two points and the 8 layers' 16 temporal profiles span an
    #   effective rank of 2.82. See lapa.layer.LaplaceConfig.lam_free and SPARK.md §9.
    lam_ceil: float = None       # fp32 safety ceiling for lam_free; None -> 55/chunk
    gdn_heads: int = 3
    gdn_head_k: int = 60
    gdn_expand_v: float = 1.0

    @classmethod
    def legacy(cls, **kw):
        """Explicit alias for the original semantics (same as the defaults)."""
        return cls(freq="dft", theta_scale=0.0, **kw)

    @classmethod
    def cand(cls, freq="len", theta_scale=0.02, **kw):
        """A candidate semantic change, pending an A/B on the loss."""
        return cls(freq=freq, theta_scale=theta_scale, **kw)


class SCA2Layer(nn.Module):
    """norm -> (C head || D head) -> mix -> residual -> FFN -> residual.

    Identical parameter names/shapes for every (c_cls, d_cls) pair, so any
    variant can `load_state_dict` a reference layer's weights.
    """

    def __init__(self, cfg: LayerCfg, c_cls=CHeadRef, d_cls=DHeadRef):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        dv = cfg.dv if cfg.dv is not None else d // 2
        self.n = nn.LayerNorm(d)
        self.c = c_cls(d, cfg.Mc, freq=cfg.freq, theta_scale=cfg.theta_scale,
                       max_len=cfg.max_len, dv=dv, gated_read=cfg.gated_read,
                       rope_base=cfg.rope_base, slow_frac=cfg.slow_frac)
        self.dh = d_cls(d, cfg.Md, cfg.G, dv=dv, max_len=cfg.max_len,
                        delta_rule=cfg.delta_rule, gated_read=cfg.gated_read)
        self.mix = nn.Linear(4 * dv, d)      # each head emits 2*dv (re || im)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff), nn.GELU(), nn.Linear(cfg.ff, d))

    # ---- state ------------------------------------------------------------ #
    def init_state(self, B, device, dtype=torch.float32):
        return {
            "c": self.c.init_state(B, device, dtype),
            "d": self.dh.init_state(B, device, dtype),
            "z_prev": torch.zeros(B, self.cfg.d, device=device, dtype=dtype),
        }

    # ---- prefill ---------------------------------------------------------- #
    def forward(self, x):
        return self.prefill(x)[0]

    def prefill(self, x, state=None):
        B = x.size(0)
        if state is None:
            state = self.init_state(B, x.device, x.dtype)
        z = self.n(x)
        h = torch.empty_like(z)
        h[:, 0] = state["z_prev"]
        h[:, 1:] = z[:, :-1]
        uc, cs = self.c.prefill(z, h, state["c"])
        ud, ds = self.dh.prefill(z, h, state["d"])
        x = x + self.mix(torch.cat([uc, ud], -1))
        x = x + self.ff(self.fn(x))
        return x, {"c": cs, "d": ds, "z_prev": z[:, -1]}

    # ---- decode ----------------------------------------------------------- #
    def step(self, x_t, state):
        """x_t: (B,d) -> y_t: (B,d)"""
        z = self.n(x_t)
        h = state["z_prev"]
        uc, cs = self.c.step(z, h, state["c"])
        ud, ds = self.dh.step(z, h, state["d"])
        y = x_t + self.mix(torch.cat([uc, ud], -1))
        y = y + self.ff(self.fn(y))
        return y, {"c": cs, "d": ds, "z_prev": z}


def make_layer(cfg=None, c_cls=CHeadRef, d_cls=DHeadRef, seed=0, device="cpu", dtype=torch.float32):
    cfg = cfg or LayerCfg()
    torch.manual_seed(seed)
    return SCA2Layer(cfg, c_cls, d_cls).to(device=device, dtype=dtype)
