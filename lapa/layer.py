r"""LAPLACE ATTENTION -- self-contained, production-ready layer.  Generation 4 of the
lineage in WINNERS.md (generation 3 = best_layer_cdelta.py).  No dependency on the
sca2 package; `python laplace_attention.py` verifies this file against the repo's
fast path (variant `cshort_damph`) in float64, then checks decode == prefill in
float32 and reports the bf16 deviation.

================================================================================
WHAT THE LAYER COMPUTES
================================================================================
    z_t = LN(x_t),  h_t = z_{t-1}                       (write key = PREVIOUS token)
    y_t = x_t + mix([ long(z, h) ; short(z, h) ])        each head emits 2*dv (Re || Im)
    out = y_t + FFN(LN(y_t))

Both heads are the same object -- a windowed, damped Fourier sum of value writes
addressed by phase codes -- with different window, grid and damping:

    write code   c_s = exp(i phi_s),   phi_{s,m} = theta_m K(h_s)_m + s . omega_m     |c_{s,m}| = 1
    read  code   q_t = exp(-i psi_t),  psi_{t,m} = theta_m K(z_t)_m + t . omega_m
    state        S_t = diag(e^{-lambda}) S_{t-1} + c_t (x) e_t                       (M, dv) complex
    read         o_t = (1/M) sum_m w_m q_{t,m} S_{t,m}                              (dv,) complex
    kernel       o_t = sum_{s in window(t)} kappa(t,s) e_s,
                 kappa(t,s) = (1/M) sum_m w_m e^{-lambda_m (t-s)} e^{i(phi_s - psi_t)}

LONG head: window = everything so far, grid = rope (omega_m = pi . base^{-m/(M-1)},
no aliasing below 2.base positions), state = the accumulator S, error-correcting
write (delta rule) e_t = v_t - beta_t . Re(c_t^H S~_{t-1})/M -- exact read-back
because ||c_t||^2 = M identically -- and a learned per-mode decay lambda_m >= 0
with a fraction of the lowest-frequency modes pinned at lambda = 0 (persistent
memory).  Trained, the damped modes forget in ~5-8 tokens: half the spectrum is a
long memory, half a fast one.  That is a Laplace transform of the token stream
(poles at lambda_m + i omega_m), hence the name.

SHORT head: window = the last L tokens, grid = DFT (omega_m = 2 pi m / L), whose
Dirichlet comb (1/L) sum_m e^{i n omega_m} = delta(n mod L) is an EXACT tap at every
lag but periodic -- so the state is a ring buffer of the L-1 previous writes, the
write is additive, and with theta = 0 the head IS a learned L-tap causal filter
(w_m = sum_n taps_n e^{i n omega_m}); theta != 0 makes the taps content-dependent.

Chunked closed form (long head): e^{-lambda (t-s)} factorises into write codes scaled
by e^{+lambda (s - t0)} and read codes scaled by e^{-lambda (t - t0)}, chunk-relative,
so the delta rule is one triangular solve per chunk, E = (I + diag(beta) tril(G,-1))^{-1}
(V - beta .* R), with the Gram G damped the same way.  lambda . chunk must stay below
~60 for float32 (e^{lambda . chunk} is formed): lam_max = 1/8 at chunk 128 is safe.

================================================================================
PRECISION POLICY (the part that makes this trainable in bf16)
================================================================================
The rule is: the STATE and the PHASES are float32, everything else follows autocast.

* The recurrent STATE is always float32, whatever the model dtype, and so is the
  delta-rule solve that writes into it (`e = W (v - beta r)`) and the Gram's
  triangular factor.  Those are the two places where a reduced mantissa could
  compound along the sequence rather than just perturb one output.
* PHASES are always float32: `p . omega` reaches thousands of radians over a
  context and the codes are cos/sin of it, so the sum can never be narrowed.  The
  K/V/bproj projections that feed it are ordinary GEMMs and follow autocast; only
  `K(x) . theta + p . omega` is fp32.
* Everything else -- the cos/sin codes once formed, the Gram GEMM, the intra-chunk
  kernel, the state reads and writes, and the short head -- runs in `gemm_dtype`:
  None (default) follows autocast if active, else float32.  Accumulation stays fp32
  inside the tensor-core GEMM, which is what protects the short head's Dirichlet
  comb: rounding its codes to bf16 leaves the exact tap at cosine similarity
  0.999998.  Set `gemm_dtype=torch.float32` to force the old all-fp32 behaviour.
Measured deviation bf16-autocast vs float32 is printed by the self-test (~3e-3).

The long head keeps its state as ONE (B, 2M, dv) block and its codes as (., 2M)
blocks, so every read and write of the state is a single GEMM rather than a pair and
the sequential chunk loop contains no concatenation at all.

Decode is O(1) per token in the context length: state = 2.M.dv + (L-1).(2L + dv)
floats per layer (LapA at d=128: 21.3k + 1.4k), one small GEMV per head.

Reference results (CATCHUP.md, seed 0, 4 layers d=128, ~186k params/layer, pycode):
generation 3 was overtaken by Gated DeltaNet at ~300M tokens; this layer holds
-0.03..-0.05 nats below GDN from 300M to 1.4B tokens at 79k vs 74k tok/s train and
~2.7x GDN's decode speed.  Learned decays: median memory 5-8 tokens on the damped
half, the pinned half infinite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

State = Dict[str, torch.Tensor]


@dataclass
class LaplaceConfig:
    d: int = 128  # model width
    M: int = 190  # long-head modes (frequency resolution; NOT tied to d)
    dv: int = 56  # value width; each head emits 2*dv (Re || Im)
    L: int = 16  # short-head window (tokens)
    ff: int = 448  # FFN hidden width
    theta_scale: float = (
        0.02  # init scale of the content phase (0 -> content path is dead)
    )
    persist: float = (
        0.5  # fraction of long-head modes pinned at lambda = 0 (lowest freqs)
    )
    learn_persist: bool = False  # no hard pin: lambda_m = lam_max*sigmoid(a_m); the `persist`
    #                              fraction merely starts persistent (a=-8), the gradient decides
    mem_range: Optional[Tuple[float, float]] = None  # init memories 1/lambda of the damped modes;
    #   None -> (L, 32*L): the damped half starts just beyond the exact window and takes over from it
    lam_max: Optional[float] = None  # decay cap; None -> 1/L (a damped mode never forgets faster than
    #   the window remembers, so the two heads overlap instead of meeting at a hard edge); lam_max*chunk <~ 60
    chunk: int = 128  # prefill chunk (also decides lam_max's safety)
    rope_base: float = 10000.0
    rope_min_period: Optional[float] = None  # shortest period in the fast rope grid. None = 2
    #   (historical). Set to 2*L to start the long head where the short head's exact window ends
    #   instead of overlapping it -- see rope_grid(). Free: no parameters, no state.
    slow_frac: float = 0.0  # fraction of long-head modes reserved as SLOW integrators (periods
    #                          2T..20T at max_len T, i.e. rope base 10*max_len over that slice);
    #                          the rest is the geometric rope grid of `rope_base`. The LM keeps
    #                          55-85% of its state energy in such modes (document memory); copy wants 0.
    max_len: int = 1024     # context the slow slice is sized for
    long_path: str = (
        "batched"  # "batched" (intra-chunk work for all chunks at once) | "chunk"
    )
    gemm_dtype: Optional[torch.dtype] = (
        None  # dtype of the long-head GEMM OPERANDS (codes, and the state as it is read
        #       and written); None = autocast if active, else fp32. The state itself, the
        #       phases, the Gram's solve and the short head stay fp32 regardless.
    )
    beta_init: float = -2.0  # erase gate bias: sigmoid(-2) = 0.12 at init
    conv: int = 0  # width of a causal depthwise conv applied to z BEFORE both heads (0 = none).
    #   Every competitive linear mixer has one -- Mamba, GDN (kernel 4 on q/k/v), LFM2 -- and
    #   this layer did not. Initialised to the identity, so at init it is exactly a no-op.


def _rms(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)


def _wd(p: torch.Tensor) -> torch.dtype:
    """Working dtype of the fp32 sections: float64 if the module is float64 (tests), else float32."""
    return torch.float64 if p.dtype == torch.float64 else torch.float32


def rope_grid(M: int, base: float, slow_frac: float = 0.0, max_len: int = 1024,
              min_period: Optional[float] = None) -> torch.Tensor:
    """omega_m for the long head. slow_frac = 0: pi * base^(-m/(M-1)) (geometric, unaliased over
    2*base). slow_frac > 0: the last n_slow = round(slow_frac*M) frequencies are replaced by a
    geometric slice over periods [2*max_len, 20*max_len] -- integrators, near-constant over a
    sequence -- and the other M - n_slow keep the base grid. Sorted decreasing, as before.

    min_period moves the FAST end of the grid. None (default, and the only setting anything has
    been measured at) keeps the historical grid, which starts at period 2 whatever the short
    head's window is. The observation behind the knob: every mode with period < 2L addresses a
    lag the short head already taps EXACTLY, and that overlap grows with L -- 57 of 190 modes at
    the d=128 campaign's Ls=16 (30%), 115 of 256 at v1's Ls=64 (45%), so raising M from 190 to
    256 moved the modes reaching BEYOND the short window only from 133 to 141.

    UNTESTED, and not obviously right. The rope grid is a positional ENCODING, not a bank of
    independent period detectors: the high frequencies are what separate NEARBY positions, and
    dropping them does not hand those modes to long lags for free, it coarsens resolution. At
    min_period = 2L nothing but the single fastest mode distinguishes positions 64-128 apart.
    Measure before believing. Note also that the damped modes and the grid's SLOW end are
    already tied to L and to the context (lam_max = 1/L, mem_range = (L, 32L), 2*base ~ context);
    this knob is a third tie, not a replacement for those."""
    n_slow = int(round(slow_frac * M))
    n_fast = M - n_slow
    k = torch.arange(n_fast, dtype=torch.float32)
    w_hi = math.pi if min_period is None else 2 * math.pi / min_period   # fastest mode kept
    span = (math.pi / base) / w_hi                                       # down to period 2*base
    fast = w_hi * span ** (k / max(n_fast - 1, 1))
    if n_slow == 0:
        return fast
    p = torch.logspace(math.log10(2 * max_len), math.log10(20 * max_len), n_slow)
    slow = 2 * math.pi / p
    return torch.cat([fast, slow])


def _causal_mask(T: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)


def _no_autocast(device):
    """Context: disable autocast for the given device type (fp32 section).

    Returns `torch.autocast` itself rather than a wrapper class. Dynamo traces
    torch.autocast natively but not a custom context manager, and the wrapper
    that used to live here cost one graph break per `with` in the layer -- 11 of
    them, which is most of the elementwise fusion in the long head.
    """
    return torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                          enabled=False)


# =============================================================================
#  LONG HEAD: rope grid, damped accumulator, delta-rule write
# =============================================================================
def _damp_params(cfg: "LaplaceConfig"):
    """(mem_lo, mem_hi, lam_max) with the window-aligned defaults resolved."""
    lam_max = cfg.lam_max if cfg.lam_max is not None else 1.0 / cfg.L
    lo, hi = cfg.mem_range if cfg.mem_range is not None else (float(cfg.L), 32.0 * cfg.L)
    return lo, hi, lam_max


class LongHead(nn.Module):
    def __init__(self, cfg: LaplaceConfig):
        super().__init__()
        d, M, dv = cfg.d, cfg.M, cfg.dv
        self.d, self.M, self.dv, self.cfg = d, M, dv, cfg
        self.K = nn.Linear(d, M, False)
        self.V = nn.Linear(d, dv, False)
        self.theta = nn.Parameter(
            torch.zeros(M)
            if cfg.theta_scale == 0.0
            else cfg.theta_scale * torch.randn(M)
        )
        self.wr = nn.Parameter(torch.ones(M))  # spectral read weights, w = wr + i wi
        self.wi = nn.Parameter(torch.zeros(M))
        self.register_buffer("omega", rope_grid(M, cfg.rope_base, cfg.slow_frac, cfg.max_len,
                                                cfg.rope_min_period))
        self.bproj = nn.Linear(d, 1, True)  # erase gate beta = sigmoid(bproj(z))
        nn.init.zeros_(self.bproj.weight)
        nn.init.constant_(self.bproj.bias, cfg.beta_init)
        lo, hi, self.lam_max = _damp_params(cfg)
        mem = torch.exp(torch.empty(M).uniform_(math.log(lo), math.log(hi)))
        n_pin = int(round(cfg.persist * M))
        low = self.omega.abs().argsort()[:n_pin]  # lowest frequencies persist
        if cfg.learn_persist:
            # lambda_m = lam_max * sigmoid(a_m): reaches ~0 (a=-8 -> memory > 20k tokens) or the
            # cap within a few hundred steps either way; nothing pinned, the task decides the split.
            a = torch.logit((1.0 / mem / self.lam_max).clamp(1e-4, 1 - 1e-4))
            a[low] = -8.0
            self.lam_raw = nn.Parameter(a)
            self.register_buffer("lam_mask", torch.ones(M))
        else:
            self.lam_raw = nn.Parameter(
                torch.log(torch.expm1(1.0 / mem))
            )  # softplus^{-1}(lambda)
            mask = torch.ones(M)
            if n_pin:
                mask[low] = 0.0
            self.register_buffer("lam_mask", mask)

    # ---- pieces ----------------------------------------------------------- #
    @property
    def wd(self) -> torch.dtype:
        return _wd(self.wr)

    def lam(self) -> torch.Tensor:
        if self.cfg.learn_persist:
            return self.lam_max * torch.sigmoid(self.lam_raw.to(self.wd))
        return (
            F.softplus(self.lam_raw.to(self.wd)).clamp(max=self.lam_max)
            * self.lam_mask
        )

    def _phase(self, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """k (...,M) = K(x), ALREADY projected -> phase (...,M) in float32; p (...,1) positions.

        The projection is taken outside the fp32 section by every caller so that it follows
        autocast like any other linear layer: it is a d x M GEMM, and fp32 costs 3.4x bf16
        here. Only the sum is fp32 -- p.omega reaches thousands of radians and the codes
        are cos/sin of it, so that part can never be reduced."""
        return k.to(self.wd) * self.theta.to(self.wd) + p * self.omega

    def _gemm_dtype(self, ref: torch.Tensor) -> torch.dtype:
        if self.cfg.gemm_dtype is not None:
            return self.cfg.gemm_dtype
        if torch.is_autocast_enabled() and ref.is_cuda:
            return torch.get_autocast_gpu_dtype()
        return self.wd

    def init_state(self, B: int, device) -> State:
        # One (B, 2M, dv) block, rows [Re ; Im]. Keeping the two halves in ONE tensor is
        # what lets every read and write of the state be a single GEMM against a (., 2M)
        # code block instead of a pair -- see _batched.
        return {
            "s": torch.zeros(B, 2 * self.M, self.dv, device=device, dtype=self.wd),
            "pos": torch.zeros((), device=device, dtype=self.wd),
        }

    def _damp(self, lam, n):
        """(2M,1) decay factors for the packed state."""
        return torch.exp(-lam * n)[:, None].repeat(2, 1)

    # ---- prefill: one chunk against an incoming state ---------------------- #
    def _chunk(self, kz, kh, vz, bz, st: State):
        B, T, _ = kz.shape
        M = self.M
        dev = kz.device
        gd = self._gemm_dtype(kz)
        with _no_autocast(dev):
            lam = self.lam()
            idx = torch.arange(T, device=dev, dtype=self.wd)[:, None]
            gw, gq = torch.exp(lam * idx), torch.exp(-lam * idx)  # (T,M)
            gT_2 = torch.exp(-lam * (T - 1))[:, None].repeat(2, 1)
            p = idx + st["pos"]
            pw, pq = self._phase(kh, p), self._phase(kz, p)  # (B,T,M) fp32
            cw, sw, cq, sq = pw.cos(), pw.sin(), pq.cos(), pq.sin()
            Kk = torch.cat([cw * gw, sw * gw], -1).to(gd)  # (B,T,2M) keys
            Qk = torch.cat([cw * gq, sw * gq], -1).to(gd)  # (B,T,2M) Gram lhs / read-back
            c1 = (self.wr * cq + self.wi * sq) * gq
            c2 = (self.wr * sq - self.wi * cq) * gq
            Fq = torch.cat([torch.cat([c1, c2], -1),
                            torch.cat([-c2, c1], -1)], 1).to(gd)  # (B,2T,2M) queries
            s0 = (st["s"] * self._damp(lam, 1)).to(gd)  # damped incoming state
            beta = torch.sigmoid(bz.to(self.wd))  # (B,T,1)
            G = (Qk @ Kk.transpose(1, 2)).to(self.wd) / M
            A = torch.eye(T, device=dev, dtype=self.wd) + beta * G.tril(-1)
            r = (Qk @ s0).to(self.wd) / M
            e = torch.linalg.solve_triangular(
                A, vz.to(self.wd) - beta * r, upper=False, unitriangular=True
            )
            ec = e.to(gd)
            K2 = (Fq @ Kk.transpose(1, 2)).masked_fill(
                _causal_mask(T, dev).repeat(2, 1)[None], 0)  # (B,2T,T) [Re;Im]
            o = (K2 @ ec + Fq @ s0).to(self.wd)  # (B,2T,dv) kernel + state read
            o = torch.cat([o[:, :T], o[:, T:]], -1) / M  # (B,T,2dv) = Re || Im
            sn = st["s"] * self._damp(lam, T) + (
                Kk.transpose(1, 2) @ ec).to(self.wd) * gT_2
        return o, {"s": sn, "pos": st["pos"] + T}

    # ---- prefill: all full chunks batched, state loop only ----------------- #
    def _batched(self, kz, kh, vz, bz, st: State, K: int):
        B, T, _ = kz.shape
        M, dv, C = self.M, self.dv, self.cfg.chunk
        dev = kz.device
        gd = self._gemm_dtype(kz)
        with _no_autocast(dev):
            lam = self.lam()
            idx = torch.arange(C, device=dev, dtype=self.wd)[:, None]
            gw, gq = torch.exp(lam * idx), torch.exp(-lam * idx)
            dC, gT, d1 = (
                torch.exp(-lam * C)[:, None],
                torch.exp(-lam * (C - 1))[:, None],
                torch.exp(-lam)[:, None],
            )
            p = torch.arange(T, device=dev, dtype=self.wd)[:, None] + st["pos"]
            pw, pq = self._phase(kh, p), self._phase(kz, p)
            v = vz.to(self.wd)
            beta = torch.sigmoid(bz.to(self.wd))
            ch = lambda x: x.view(B, K, C, x.shape[-1])
            cw, sw, cq, sq, v, beta = map(
                ch, (pw.cos(), pw.sin(), pq.cos(), pq.sin(), v, beta)
            )  # (B,K,C,.)
            # Three code blocks, each (., 2M) wide, and a state packed as (B,2M,dv). The
            # spectral weight w is folded into the QUERY side (it is a per-mode diagonal
            # in the contraction, so it may sit on either operand), which lets ONE key
            # block serve the Gram, the intra-chunk kernel and the state write, and lets
            # every state read and write be a single GEMM instead of a pair. Nothing in
            # the sequential loop below is a cat any more.
            Kk = torch.cat([cw * gw, sw * gw], -1).to(gd)  # (B,K,C,2M) keys
            Qk = torch.cat([cw * gq, sw * gq], -1).to(gd)  # (B,K,C,2M) Gram lhs / read-back
            c1 = (self.wr * cq + self.wi * sq) * gq
            c2 = (self.wr * sq - self.wi * cq) * gq
            Fq = torch.cat([torch.cat([c1, c2], -1),
                            torch.cat([-c2, c1], -1)], 2).to(gd)  # (B,K,2C,2M) queries
            G = (Qk @ Kk.transpose(-1, -2)).to(self.wd) / M
            eye = torch.eye(C, device=dev, dtype=self.wd)
            W = torch.linalg.solve_triangular(
                eye + beta * G.tril(-1),
                eye.expand(B, K, C, C),
                upper=False,
                unitriangular=True,
            )  # (B,K,C,C)
            # K2 = Fq Kk^T in one GEMM: [c1|c2] Kk^T is the real part and [-c2|c1] Kk^T
            # the imaginary one, so the (2C,C) block comes out already stacked [Re ; Im]
            # -- the same layout the state read below produces, so the two just add.
            K2 = (Fq @ Kk.transpose(-1, -2)).masked_fill(
                _causal_mask(C, dev).repeat(2, 1)[None, None], 0)  # (B,K,2C,C)
            d1_2, dC_2, gT_2 = (x.repeat(2, 1) for x in (d1, dC, gT))
        Fq, W, v, beta, K2, Kk, Qk = (
            x.unbind(1) for x in (Fq, W, v, beta, K2, Kk, Qk)
        )
        s = st["s"]
        outs = []
        for k in range(K):
            with _no_autocast(dev):
                s0 = (s * d1_2).to(gd)  # (B,2M,dv) damped incoming state
                r = (Qk[k] @ s0).to(self.wd) / M
                e = W[k] @ (v[k] - beta[k] * r)  # (B,C,dv) fp32: the delta-rule solve
                ec = e.to(gd)
                outs.append((K2[k] @ ec + Fq[k] @ s0).to(self.wd))  # (B,2C,dv) [Re;Im]
                s = s * dC_2 + (Kk[k].transpose(-1, -2) @ ec).to(self.wd) * gT_2
        o = torch.stack(outs, 1)  # (B,K,2C,dv)
        o = torch.cat([o[:, :, :C], o[:, :, C:]], -1).reshape(B, K * C, 2 * dv) / M
        return o, {"s": s, "pos": st["pos"] + T}

    def prefill(self, z, z_prev, state: Optional[State] = None):
        """z (B,T,d); z_prev (B,d) is the token before z[:,0] -- the write key at t is z_{t-1}."""
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device)
        # K, V and bproj are linear and follow autocast; doing them once here rather than
        # per chunk also means the write keys K(h) are K(z) shifted by one row, so the
        # second d x M projection and the (B,T,d) shifted copy of z both disappear.
        kz, vz, bz = self.K(z), self.V(z), self.bproj(z)
        kh = torch.cat([self.K(z_prev)[:, None], kz[:, :-1]], 1)
        C = self.cfg.chunk
        K = T // C
        cut = lambda a, b: (kz[:, a:b], kh[:, a:b], vz[:, a:b], bz[:, a:b])
        if K >= 2 and self.cfg.long_path == "batched":
            o, st = self._batched(*cut(0, K * C), st, K)
            if K * C < T:
                o2, st = self._chunk(*cut(K * C, T), st)
                o = torch.cat([o, o2], 1)
        else:
            outs = []
            for s0 in range(0, T, C):
                o, st = self._chunk(*cut(s0, s0 + C), st)
                outs.append(o)
            o = torch.cat(outs, 1)
        return _rms(o).to(z.dtype), st

    # ---- decode: one token, all float32 ------------------------------------ #
    def step(self, z_t, h_t, state: State):
        kh, kz, vz, bz = self.K(h_t), self.K(z_t), self.V(z_t), self.bproj(z_t)
        with _no_autocast(z_t.device):
            M = self.M
            s0 = state["s"] * self._damp(self.lam(), 1)  # (B,2M,dv)
            p = state["pos"]
            pw, pq = self._phase(kh, p), self._phase(kz, p)  # (B,M)
            kt = torch.cat([pw.cos(), pw.sin()], -1)  # (B,2M) write code
            beta = torch.sigmoid(bz.to(self.wd))
            vhat = torch.einsum("bm,bmj->bj", kt, s0) / M
            e = vz.to(self.wd) - beta * vhat
            s = torch.addcmul(s0, e[:, None, :], kt[:, :, None])
            cq, sq = pq.cos(), pq.sin()
            c1, c2 = self.wr * cq + self.wi * sq, self.wr * sq - self.wi * cq
            qt = torch.stack([torch.cat([c1, c2], -1),
                              torch.cat([-c2, c1], -1)], 1)  # (B,2,2M)
            u = torch.einsum("bam,bmj->baj", qt, s).reshape(z_t.size(0), 2 * self.dv) / M
        return _rms(u).to(z_t.dtype), {"s": s, "pos": state["pos"] + 1}


# =============================================================================
#  SHORT HEAD: DFT grid, ring buffer of the last L-1 writes (an exact L-tap window)
# =============================================================================
class ShortHead(nn.Module):
    def __init__(self, cfg: LaplaceConfig):
        super().__init__()
        d, L, dv = cfg.d, cfg.L, cfg.dv
        self.d, self.L, self.dv, self.cfg = d, L, dv, cfg
        self.K = nn.Linear(d, L, False)
        self.V = nn.Linear(d, dv, False)
        self.theta = nn.Parameter((cfg.theta_scale or 0.02) * torch.randn(L))
        self.wr = nn.Parameter(
            torch.ones(L)
        )  # w = 1: delta at lag 0 at init (o_t = V(z_t))
        self.wi = nn.Parameter(torch.zeros(L))
        # built in float64: 2*pi/L rounded in float32 breaks the comb's exact cancellation
        self.register_buffer(
            "omega", (torch.arange(L, dtype=torch.float64) * (2 * math.pi / L)).float()
        )

    @property
    def wd(self) -> torch.dtype:
        return _wd(self.wr)

    def _gemm_dtype(self, ref: torch.Tensor) -> torch.dtype:
        if self.cfg.gemm_dtype is not None:
            return self.cfg.gemm_dtype
        if torch.is_autocast_enabled() and ref.is_cuda:
            return torch.get_autocast_gpu_dtype()
        return self.wd

    def init_state(self, B: int, device) -> State:
        n, L, wd = self.L - 1, self.L, self.wd
        return {
            "c": torch.ones(B, n, L, device=device, dtype=wd),
            "s": torch.zeros(B, n, L, device=device, dtype=wd),
            "e": torch.zeros(B, n, self.dv, device=device, dtype=wd),
            "pos": torch.zeros((), device=device, dtype=torch.long),
        }

    def _phase(self, k, p):
        """k (...,L) = K(x), already projected (see LongHead._phase for why)."""
        return (
            k.to(self.wd) * self.theta.to(self.wd)
            + (p % self.L).to(self.wd)[..., None] * self.omega
        )

    def _read(self, cq, sq, cw, sw, e):
        """Re/Im kappa(t,s) over the window via two folded read vectors, then contract with e."""
        c1, c2 = self.wr * cq + self.wi * sq, self.wr * sq - self.wi * cq
        k_re = (
            torch.einsum("btwl,btl->btw", cw, c1)
            + torch.einsum("btwl,btl->btw", sw, c2)
        ) / self.L
        k_im = (
            torch.einsum("btwl,btl->btw", sw, c1)
            - torch.einsum("btwl,btl->btw", cw, c2)
        ) / self.L
        return torch.cat(
            [
                torch.einsum("btw,btwj->btj", k_re, e),
                torch.einsum("btw,btwj->btj", k_im, e),
            ],
            -1,
        )

    def prefill(self, z, z_prev, state: Optional[State] = None):
        """z (B,T,d); z_prev (B,d) is the token before z[:,0] (see LongHead.prefill)."""
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device)
        L, n = self.L, self.L - 1
        gd = self._gemm_dtype(z)  # read before autocast is disabled below
        kz, vz = self.K(z), self.V(z)  # follow autocast
        kh = torch.cat([self.K(z_prev)[:, None], kz[:, :-1]], 1)
        with _no_autocast(z.device):
            p = torch.arange(T, device=z.device) + st["pos"]
            phi = self._phase(kh, p[None].expand(B, T))
            cw = torch.cat([st["c"], phi.cos()], 1)
            sw = torch.cat([st["s"], phi.sin()], 1)
            e = torch.cat([st["e"], vz.to(self.wd)], 1)
            psi = self._phase(kz, p[None].expand(B, T))
            u = self._banded(psi.cos(), psi.sin(), cw, sw, e, T, gd)
        new = {"c": cw[:, -n:], "s": sw[:, -n:], "e": e[:, -n:], "pos": st["pos"] + T}
        return _rms(u).to(z.dtype), new

    def _banded(self, cq, sq, cw, sw, e, T, gd=None):
        """The window read as BANDED GEMMs, all chunks at once (no sequential dependency).

        kappa(t,s) = Fq_t . Fk_s / L  with  Fk_s = [cw_s ; sw_s]  (2L)  and, for the real /
        imaginary parts,  Fq1_t = [c1 ; c2],  Fq2_t = [-c2 ; c1],  c1 = wr cq + wi sq,
        c2 = wr sq - wi cq.  Queries are cut into chunks of C = L; the chunk with queries
        [t0, t0+C) reads extended keys [t0, t0+C+L-1) (the L-1 buffered writes come first
        in the extended arrays, so query t reads extended indices t .. t+L-1 = lags L-1 .. 0).
        S = Fq (B,K,2C,2L) @ Fk_ext (B,K,2L,C+L-1), band mask 0 <= j - i <= L-1, o = S @ e_ext.
        Same numbers as the unfolded contraction (checked against the repo path in __main__),
        dense GEMMs instead of an O(T.L.L) einsum on strided views.

        `gd` is the dtype of the GEMM OPERANDS. The comb's cancellation is carried by the
        fp32 accumulator inside the tensor-core GEMM, not by the operands: rounding the
        codes to bf16 leaves the exact tap at cosine similarity 0.999998 and the whole
        head 2.8e-3 from float64, the same order as the long head's bf16 deviation."""
        B = cq.size(0)
        L = self.L
        gd = gd if gd is not None else self.wd
        C = L
        K = -(-T // C)
        pad = K * C - T
        if pad:                                                   # ragged tail: pad queries and keys
            cq, sq = F.pad(cq, (0, 0, 0, pad)), F.pad(sq, (0, 0, 0, pad))
            cw, sw, e = F.pad(cw, (0, 0, 0, pad)), F.pad(sw, (0, 0, 0, pad)), F.pad(e, (0, 0, 0, pad))
        c1 = (self.wr * cq + self.wi * sq).to(gd)                 # (B,KC,L)
        c2 = (self.wr * sq - self.wi * cq).to(gd)
        Fq = torch.cat([torch.cat([c1, c2], -1).view(B, K, C, 2 * L),
                        torch.cat([-c2, c1], -1).view(B, K, C, 2 * L)], 2)          # (B,K,2C,2L)
        Fk = torch.cat([cw, sw], -1).to(gd)                                         # (B,KC+L-1,2L)
        N = C + L - 1
        Fk = Fk.unfold(1, N, C).movedim(-1, 2)                                      # (B,K,N,2L)
        ek = e.to(gd).unfold(1, N, C).movedim(-1, 2)                                # (B,K,N,dv)
        S = Fq @ Fk.transpose(-1, -2) / L                                           # (B,K,2C,N)
        i = torch.arange(C, device=cq.device)[:, None]
        j = torch.arange(N, device=cq.device)[None]
        band = ((j - i) >= 0) & ((j - i) <= L - 1)                                  # (C,N)
        S = S.masked_fill(~torch.cat([band, band], 0)[None, None], 0)
        o = (S @ ek).to(self.wd)                                                    # (B,K,2C,dv)
        o = torch.cat([o[:, :, :C], o[:, :, C:]], -1).reshape(B, K * C, 2 * self.dv)
        return o[:, :T]

    def step(self, z_t, h_t, state: State):
        kh, kz, vz = self.K(h_t), self.K(z_t), self.V(z_t)
        with _no_autocast(z_t.device):
            B = z_t.size(0)
            p = state["pos"].expand(B)
            phi = self._phase(kh, p)
            cw = torch.cat([state["c"], phi.cos()[:, None]], 1)
            sw = torch.cat([state["s"], phi.sin()[:, None]], 1)
            e = torch.cat([state["e"], vz.to(self.wd)[:, None]], 1)
            psi = self._phase(kz, p)
            u = self._read(
                psi.cos()[:, None],
                psi.sin()[:, None],
                cw[:, None],
                sw[:, None],
                e[:, None],
            )[:, 0]
        return _rms(u).to(z_t.dtype), {
            "c": cw[:, 1:],
            "s": sw[:, 1:],
            "e": e[:, 1:],
            "pos": state["pos"] + 1,
        }


# =============================================================================
#  THE LAYER
# =============================================================================
class LaplaceAttention(nn.Module):
    """norm -> (long head || short head) -> mix -> residual -> norm -> FFN -> residual.

    prefill(x, state) processes a (B,T,d) block; step(x_t, state) one (B,d) token;
    both return (output, new_state) and agree to float rounding (the self-test
    checks it).  forward(x) = prefill(x)[0].
    """

    def __init__(self, cfg: LaplaceConfig = LaplaceConfig()):
        super().__init__()
        self.cfg = cfg
        d, dv = cfg.d, cfg.dv
        self.n = nn.LayerNorm(d)
        self.long = LongHead(cfg)
        self.short = ShortHead(cfg)
        self.mix = nn.Linear(4 * dv, d)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff), nn.GELU(), nn.Linear(cfg.ff, d))
        self.ck = cfg.conv
        if self.ck:
            w = torch.zeros(d, 1, self.ck)
            w[:, 0, -1] = 1.0  # identity at init: the conv starts as a no-op
            self.cw = nn.Parameter(w)

    def init_state(self, B: int, device) -> State:
        dt = self.n.weight.dtype
        st = {
            "long": self.long.init_state(B, device),
            "short": self.short.init_state(B, device),
            "z_prev": torch.zeros(B, self.cfg.d, device=device, dtype=dt),
        }
        if self.ck:
            st["cbuf"] = torch.zeros(B, self.ck - 1, self.cfg.d, device=device, dtype=dt)
        return st

    def _conv(self, z, buf):
        """Causal depthwise conv on (B,T,d); `buf` holds the ck-1 tokens preceding z."""
        zz = torch.cat([buf.to(z.dtype), z], 1)
        zc = F.conv1d(zz.transpose(1, 2), self.cw.to(z.dtype),
                      groups=self.cfg.d).transpose(1, 2)
        return zc, zz[:, -(self.ck - 1):]

    def _conv_step(self, z_t, buf):
        """Same filter, one token. out = sum_j w_j . window_j, window = [buf ; z_t]."""
        win = torch.cat([buf.to(z_t.dtype), z_t[:, None]], 1)          # (B,ck,d)
        return (win.transpose(1, 2) * self.cw.squeeze(1).to(z_t.dtype)).sum(-1), win[:, 1:]

    def forward(self, x):
        return self.prefill(x)[0]

    def prefill(self, x, state: Optional[State] = None):
        B = x.size(0)
        st = state if state is not None else self.init_state(B, x.device)
        z = self.n(x)
        new = {}
        if self.ck:
            z, new["cbuf"] = self._conv(z, st["cbuf"])
        zp = st["z_prev"].to(z.dtype)
        ul, sl = self.long.prefill(z, zp, st["long"])
        us, ss = self.short.prefill(z, zp, st["short"])
        x = x + self.mix(torch.cat([ul, us], -1))
        x = x + self.ff(self.fn(x))
        return x, {**new, "long": sl, "short": ss, "z_prev": z[:, -1]}

    def step(self, x_t, state: State):
        z = self.n(x_t)
        new = {}
        if self.ck:
            z, new["cbuf"] = self._conv_step(z, state["cbuf"])
        h = state["z_prev"].to(z.dtype)
        ul, sl = self.long.step(z, h, state["long"])
        us, ss = self.short.step(z, h, state["short"])
        y = x_t + self.mix(torch.cat([ul, us], -1))
        y = y + self.ff(self.fn(y))
        return y, {**new, "long": sl, "short": ss, "z_prev": z}

    def state_floats(self) -> int:
        cfg = self.cfg
        return 2 * cfg.M * cfg.dv + (cfg.L - 1) * (2 * cfg.L + cfg.dv) + cfg.d


# =============================================================================
#  SELF-TEST
# =============================================================================
def _verify_against_repo():
    """This file == the repo's `cshort_damph` variant, in float64, both prefill paths and decode."""
    try:
        from sca2.ref import LayerCfg
        from sca2.registry import build
    except ImportError:
        print("repo not importable: skipping equivalence check")
        return
    torch.manual_seed(0)
    cfg = LaplaceConfig(d=32, M=24, dv=8, L=8, ff=64, chunk=16)
    ref = build(
        "cshort_damph",
        LayerCfg(
            32, 24, 4, 8, 64, freq="rope", theta_scale=0.02, dv=8, Ls=8, max_len=128
        ),
    ).double()
    mine = LaplaceAttention(cfg).double()
    sd = {
        k.replace("c.", "long.", 1)
        if k.startswith("c.")
        else k.replace("dh.", "short.", 1)
        if k.startswith("dh.")
        else k: v
        for k, v in ref.state_dict().items()
    }
    mine.load_state_dict(sd, strict=True)
    assert torch.allclose(mine.long.omega, ref.c.omega.double()) and torch.allclose(
        mine.short.omega, ref.dh.omega.double()
    )
    B, T = 2, 45
    x = torch.randn(B, T, 32, dtype=torch.float64)
    ref.c.CTX = 16
    ref.dh.CTX = 16
    with torch.no_grad():
        y_ref, _ = ref.prefill(x)
        for path in ("batched", "chunk"):
            mine.cfg.long_path = path
            y, _ = mine.prefill(x)
            print(
                f"  float64 vs repo, long_path={path:8s}: {(y - y_ref).abs().max().item():.2e}"
            )
            assert (y - y_ref).abs().max() < 1e-11
        st = mine.init_state(B, x.device)
        ys = []
        for t in range(T):
            o, st = mine.step(x[:, t], st)
            ys.append(o)
        err = (torch.stack(ys, 1) - y_ref).abs().max().item()
        print(f"  float64 decode vs repo prefill:          {err:.2e}")
        assert err < 1e-11


def _verify_iso(device):
    """prefill == step, and a split prefill == one prefill, in float32."""
    torch.manual_seed(1)
    m = LaplaceAttention(LaplaceConfig(chunk=64)).to(device)
    B, T = 2, 200
    x = torch.randn(B, T, 128, device=device)
    with torch.no_grad():
        y, _ = m.prefill(x)
        st = m.init_state(B, device)
        ys = []
        for t in range(T):
            o, st = m.step(x[:, t], st)
            ys.append(o)
        e1 = ((torch.stack(ys, 1) - y).abs().max() / y.abs().max()).item()
        y1, s1 = m.prefill(x[:, :77])
        y2, _ = m.prefill(x[:, 77:], s1)
        e2 = ((torch.cat([y1, y2], 1) - y).abs().max() / y.abs().max()).item()
    print(
        f"  float32 on {device}: decode vs prefill rel {e1:.1e}, split 77|123 rel {e2:.1e}"
    )
    assert e1 < 3e-4 and e2 < 3e-4
    return m, x, y


def _report_bf16(m, x, y_fp32):
    if not x.is_cuda:
        print("  bf16 report needs cuda: skipped")
        return
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y, _ = m.prefill(x)
    rel = ((y.float() - y_fp32).abs().max() / y_fp32.abs().max()).item()
    print(
        f"  bf16 autocast vs float32 (state + phases fp32, rest bf16): max rel dev {rel:.1e}"
    )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("== equivalence with the repo fast path (cshort_damph) ==")
    _verify_against_repo()
    print("== prefill / decode / split consistency ==")
    m, x, y = _verify_iso(dev)
    print("== precision ==")
    _report_bf16(m, x, y)
    m128 = LaplaceAttention(LaplaceConfig())
    print(
        f"\nLaplaceAttention(d=128, M=190, dv=56, L=16, ff=448): "
        f"{sum(p.numel() for p in m128.parameters())} params, {m128.state_floats()} state floats per sequence.  ALL OK"
    )
