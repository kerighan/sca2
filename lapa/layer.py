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
* The recurrent STATE is always float32, whatever the model dtype.
* Phases, cos/sin codes, decay scales, the Gram matrix and its triangular solve are
  computed in float32 (autocast disabled around them): they are cheap and they are
  where bf16 would hurt (unit-modulus codes summed over 2M terms, a solve).
* The two large GEMMs of the long head -- the intra-chunk kernel K2 = Fq Fk^T and
  its application K2 e -- run in `gemm_dtype`: None (default) follows autocast if it
  is active, else float32.  Projections, mix and FFN follow autocast as usual.
* The short head runs in float32 (12% of the layer; its Dirichlet comb relies on
  exact cancellation).
Measured deviation bf16-autocast vs float32 is printed by the self-test.

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
    mem_range: Tuple[float, float] = (
        64.0,
        4096.0,
    )  # init memories 1/lambda of the damped modes
    lam_max: float = 0.125  # decay cap; keep lam_max * chunk <~ 60 for float32
    chunk: int = 128  # prefill chunk (also decides lam_max's safety)
    rope_base: float = 10000.0
    long_path: str = (
        "batched"  # "batched" (intra-chunk work for all chunks at once) | "chunk"
    )
    gemm_dtype: Optional[torch.dtype] = (
        None  # dtype of the two big long-head GEMMs; None = autocast/fp32
    )
    beta_init: float = -2.0  # erase gate bias: sigmoid(-2) = 0.12 at init


def _rms(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)


def _wd(p: torch.Tensor) -> torch.dtype:
    """Working dtype of the fp32 sections: float64 if the module is float64 (tests), else float32."""
    return torch.float64 if p.dtype == torch.float64 else torch.float32


def _causal_mask(T: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)


class _NoAutocast:
    """Context: disable autocast for the given device type (fp32 section)."""

    def __init__(self, device):
        self.dev = "cuda" if device.type == "cuda" else "cpu"

    def __enter__(self):
        self.ctx = torch.autocast(device_type=self.dev, enabled=False)
        self.ctx.__enter__()

    def __exit__(self, *a):
        self.ctx.__exit__(*a)


# =============================================================================
#  LONG HEAD: rope grid, damped accumulator, delta-rule write
# =============================================================================
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
        k = torch.arange(M, dtype=torch.float32)
        self.register_buffer("omega", math.pi * cfg.rope_base ** (-k / max(M - 1, 1)))
        self.bproj = nn.Linear(d, 1, True)  # erase gate beta = sigmoid(bproj(z))
        nn.init.zeros_(self.bproj.weight)
        nn.init.constant_(self.bproj.bias, cfg.beta_init)
        lo, hi = cfg.mem_range
        mem = torch.exp(torch.empty(M).uniform_(math.log(lo), math.log(hi)))
        n_pin = int(round(cfg.persist * M))
        low = self.omega.abs().argsort()[:n_pin]  # lowest frequencies persist
        if cfg.learn_persist:
            # lambda_m = lam_max * sigmoid(a_m): reaches ~0 (a=-8 -> memory > 20k tokens) or the
            # cap within a few hundred steps either way; nothing pinned, the task decides the split.
            a = torch.logit((1.0 / mem / cfg.lam_max).clamp(1e-4, 1 - 1e-4))
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
            return self.cfg.lam_max * torch.sigmoid(self.lam_raw.to(self.wd))
        return (
            F.softplus(self.lam_raw.to(self.wd)).clamp(max=self.cfg.lam_max)
            * self.lam_mask
        )

    def _phase(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """x (...,d) -> phase (...,M) in float32; p (...,1) positions."""
        return self.K(x).to(self.wd) * self.theta.to(self.wd) + p * self.omega

    def _gemm_dtype(self, ref: torch.Tensor) -> torch.dtype:
        if self.cfg.gemm_dtype is not None:
            return self.cfg.gemm_dtype
        if torch.is_autocast_enabled() and ref.is_cuda:
            return torch.get_autocast_gpu_dtype()
        return self.wd

    def init_state(self, B: int, device) -> State:
        z = lambda: torch.zeros(B, self.M, self.dv, device=device, dtype=self.wd)
        return {
            "sr": z(),
            "si": z(),
            "pos": torch.zeros((), device=device, dtype=self.wd),
        }

    # ---- prefill: one chunk against an incoming state ---------------------- #
    def _chunk(self, z, h, st: State):
        B, T, _ = z.shape
        M, dv = self.M, self.dv
        dev = z.device
        with _NoAutocast(dev):
            lam = self.lam()
            idx = torch.arange(T, device=dev, dtype=self.wd)[:, None]
            gw, gq = torch.exp(lam * idx), torch.exp(-lam * idx)  # (T,M)
            dT, gT, d1 = (
                torch.exp(-lam * T)[:, None],
                torch.exp(-lam * (T - 1))[:, None],
                torch.exp(-lam)[:, None],
            )
            p = idx + st["pos"]
            pw, pq = self._phase(h, p), self._phase(z, p)  # (B,T,M) fp32
            cw, sw, cq, sq = pw.cos(), pw.sin(), pq.cos(), pq.sin()
            cw_w, sw_w, cw_q, sw_q, cq_q, sq_q = (
                cw * gw,
                sw * gw,
                cw * gq,
                sw * gq,
                cq * gq,
                sq * gq,
            )
            sr0, si0 = st["sr"] * d1, st["si"] * d1  # damped incoming state
            beta = torch.sigmoid(self.bproj(z).to(self.wd))  # (B,T,1)
            G = (cw_q @ cw_w.transpose(1, 2) + sw_q @ sw_w.transpose(1, 2)) / M
            A = torch.eye(T, device=dev, dtype=self.wd) + beta * G.tril(-1)
            r = (
                torch.einsum("btm,bmj->btj", cw_q, sr0)
                + torch.einsum("btm,bmj->btj", sw_q, si0)
            ) / M
            e = torch.linalg.solve_triangular(
                A, self.V(z).to(self.wd) - beta * r, upper=False, unitriangular=True
            )
            Am, Bm = self.wr * cw_w - self.wi * sw_w, self.wr * sw_w + self.wi * cw_w
            Fq = torch.cat([cq_q, sq_q], -1)  # (B,T,2M)
            Fk = torch.cat(
                [torch.cat([Am, Bm], -1), torch.cat([Bm, -Am], -1)], 1
            )  # (B,2T,2M)
        gd = self._gemm_dtype(z)
        K2 = (Fq.to(gd) @ Fk.to(gd).transpose(1, 2)).view(B, T, 2, T)
        K2 = K2.masked_fill(_causal_mask(T, dev)[None, :, None, :], 0)
        o = (K2.reshape(B, 2 * T, T) @ e.to(gd)).to(self.wd).view(B, T, 2 * dv) / M
        with _NoAutocast(dev):
            c1, c2 = self.wr * cq_q + self.wi * sq_q, self.wr * sq_q - self.wi * cq_q
            S4 = torch.cat(
                [torch.cat([sr0, si0], -1), torch.cat([si0, -sr0], -1)], 1
            )  # (B,2M,2dv)
            o = o + (torch.cat([c1, c2], -1) @ S4) / M
            upd = (
                torch.cat([cw_w, sw_w], -1).transpose(1, 2) @ e * torch.cat([gT, gT], 0)
            )
            sr, si = st["sr"] * dT + upd[:, :M], st["si"] * dT + upd[:, M:]
        return o, {"sr": sr, "si": si, "pos": st["pos"] + T}

    # ---- prefill: all full chunks batched, state loop only ----------------- #
    def _batched(self, z, h, st: State, K: int):
        B, T, _ = z.shape
        M, dv, C = self.M, self.dv, self.cfg.chunk
        dev = z.device
        with _NoAutocast(dev):
            lam = self.lam()
            idx = torch.arange(C, device=dev, dtype=self.wd)[:, None]
            gw, gq = torch.exp(lam * idx), torch.exp(-lam * idx)
            dC, gT, d1 = (
                torch.exp(-lam * C)[:, None],
                torch.exp(-lam * (C - 1))[:, None],
                torch.exp(-lam)[:, None],
            )
            p = torch.arange(T, device=dev, dtype=self.wd)[:, None] + st["pos"]
            pw, pq = self._phase(h, p), self._phase(z, p)
            v = self.V(z).to(self.wd)
            beta = torch.sigmoid(self.bproj(z).to(self.wd))
            ch = lambda x: x.view(B, K, C, x.shape[-1])
            cw, sw, cq, sq, v, beta = map(
                ch, (pw.cos(), pw.sin(), pq.cos(), pq.sin(), v, beta)
            )
            cw_w, sw_w, cw_q, sw_q, cq_q, sq_q = (
                cw * gw,
                sw * gw,
                cw * gq,
                sw * gq,
                cq * gq,
                sq * gq,
            )
            G = (cw_q @ cw_w.transpose(-1, -2) + sw_q @ sw_w.transpose(-1, -2)) / M
            eye = torch.eye(C, device=dev, dtype=self.wd)
            W = torch.linalg.solve_triangular(
                eye + beta * G.tril(-1),
                eye.expand(B, K, C, C),
                upper=False,
                unitriangular=True,
            )  # (B,K,C,C)
            Am, Bm = self.wr * cw_w - self.wi * sw_w, self.wr * sw_w + self.wi * cw_w
            Fq = torch.cat([cq_q, sq_q], -1)
            Fk = torch.cat([torch.cat([Am, Bm], -1), torch.cat([Bm, -Am], -1)], 2)
            Rq = torch.cat([cw_q, sw_q], -1)
            Cq = torch.cat(
                [self.wr * cq_q + self.wi * sq_q, self.wr * sq_q - self.wi * cq_q], -1
            )
            Pw = torch.cat([cw_w, sw_w], -1).transpose(-1, -2)
            gT2 = torch.cat([gT, gT], 0)
        gd = self._gemm_dtype(z)
        K2 = (Fq.to(gd) @ Fk.to(gd).transpose(-1, -2)).view(B, K, C, 2, C)
        K2 = K2.masked_fill(_causal_mask(C, dev)[None, None, :, None, :], 0).reshape(
            B, K, 2 * C, C
        )
        Rq, W, v, beta, K2, Cq, Pw = (x.unbind(1) for x in (Rq, W, v, beta, K2, Cq, Pw))
        sr, si = st["sr"], st["si"]
        outs = []
        for k in range(K):
            with _NoAutocast(dev):
                sr0, si0 = sr * d1, si * d1
                r = (Rq[k] @ torch.cat([sr0, si0], 1)) / M
                e = W[k] @ (v[k] - beta[k] * r)  # (B,C,dv)
            o = (K2[k] @ e.to(gd)).to(self.wd).view(B, C, 2 * dv) / M
            with _NoAutocast(dev):
                S4 = torch.cat(
                    [torch.cat([sr0, si0], -1), torch.cat([si0, -sr0], -1)], 1
                )
                o = o + (Cq[k] @ S4) / M
                upd = (Pw[k] @ e) * gT2
                sr, si = sr * dC + upd[:, :M], si * dC + upd[:, M:]
            outs.append(o)
        return torch.cat(outs, 1), {"sr": sr, "si": si, "pos": st["pos"] + T}

    def prefill(self, z, h, state: Optional[State] = None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device)
        C = self.cfg.chunk
        K = T // C
        if K >= 2 and self.cfg.long_path == "batched":
            o, st = self._batched(z[:, : K * C], h[:, : K * C], st, K)
            if K * C < T:
                o2, st = self._chunk(z[:, K * C :], h[:, K * C :], st)
                o = torch.cat([o, o2], 1)
        else:
            outs = []
            for s0 in range(0, T, C):
                o, st = self._chunk(z[:, s0 : s0 + C], h[:, s0 : s0 + C], st)
                outs.append(o)
            o = torch.cat(outs, 1)
        return _rms(o).to(z.dtype), st

    # ---- decode: one token, all float32 ------------------------------------ #
    def step(self, z_t, h_t, state: State):
        with _NoAutocast(z_t.device):
            M = self.M
            d1 = torch.exp(-self.lam())[:, None]
            sr0, si0 = state["sr"] * d1, state["si"] * d1
            p = state["pos"]
            pw, pq = self._phase(h_t, p), self._phase(z_t, p)  # (B,M)
            cwt, swt = pw.cos(), pw.sin()
            beta = torch.sigmoid(self.bproj(z_t).to(self.wd))
            vhat = (
                torch.einsum("bm,bmj->bj", cwt, sr0)
                + torch.einsum("bm,bmj->bj", swt, si0)
            ) / M
            e = self.V(z_t).to(self.wd) - beta * vhat
            sr = torch.addcmul(sr0, e[:, None, :], cwt[:, :, None])
            si = torch.addcmul(si0, e[:, None, :], swt[:, :, None])
            cq, sq = pq.cos(), pq.sin()
            c1, c2 = self.wr * cq + self.wi * sq, self.wr * sq - self.wi * cq
            m = (
                torch.einsum(
                    "bam,bcmj->bacj", torch.stack([c1, c2], 1), torch.stack([sr, si], 1)
                )
                / M
            )
            u = torch.cat([m[:, 0, 0] + m[:, 1, 1], m[:, 0, 1] - m[:, 1, 0]], -1)
        return _rms(u).to(z_t.dtype), {"sr": sr, "si": si, "pos": state["pos"] + 1}


# =============================================================================
#  SHORT HEAD: DFT grid, ring buffer of the last L-1 writes (an exact L-tap window)
# =============================================================================
class ShortHead(nn.Module):
    def __init__(self, cfg: LaplaceConfig):
        super().__init__()
        d, L, dv = cfg.d, cfg.L, cfg.dv
        self.d, self.L, self.dv = d, L, dv
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

    def init_state(self, B: int, device) -> State:
        n, L, wd = self.L - 1, self.L, self.wd
        return {
            "c": torch.ones(B, n, L, device=device, dtype=wd),
            "s": torch.zeros(B, n, L, device=device, dtype=wd),
            "e": torch.zeros(B, n, self.dv, device=device, dtype=wd),
            "pos": torch.zeros((), device=device, dtype=torch.long),
        }

    def _phase(self, x, p):
        return (
            self.K(x).to(self.wd) * self.theta.to(self.wd)
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

    def prefill(self, z, h, state: Optional[State] = None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device)
        L, n = self.L, self.L - 1
        with _NoAutocast(z.device):
            p = torch.arange(T, device=z.device) + st["pos"]
            phi = self._phase(h, p[None].expand(B, T))
            cw = torch.cat([st["c"], phi.cos()], 1)
            sw = torch.cat([st["s"], phi.sin()], 1)
            e = torch.cat([st["e"], self.V(z).to(self.wd)], 1)
            psi = self._phase(z, p[None].expand(B, T))
            u = self._banded(psi.cos(), psi.sin(), cw, sw, e, T)
        new = {"c": cw[:, -n:], "s": sw[:, -n:], "e": e[:, -n:], "pos": st["pos"] + T}
        return _rms(u).to(z.dtype), new

    def _banded(self, cq, sq, cw, sw, e, T):
        """The window read as BANDED GEMMs, all chunks at once (no sequential dependency).

        kappa(t,s) = Fq_t . Fk_s / L  with  Fk_s = [cw_s ; sw_s]  (2L)  and, for the real /
        imaginary parts,  Fq1_t = [c1 ; c2],  Fq2_t = [-c2 ; c1],  c1 = wr cq + wi sq,
        c2 = wr sq - wi cq.  Queries are cut into chunks of C = L; the chunk with queries
        [t0, t0+C) reads extended keys [t0, t0+C+L-1) (the L-1 buffered writes come first
        in the extended arrays, so query t reads extended indices t .. t+L-1 = lags L-1 .. 0).
        S = Fq (B,K,2C,2L) @ Fk_ext (B,K,2L,C+L-1), band mask 0 <= j - i <= L-1, o = S @ e_ext.
        Same numbers as the unfolded contraction (checked against the repo path in __main__),
        dense GEMMs instead of an O(T.L.L) einsum on strided views."""
        B = cq.size(0)
        L = self.L
        C = L
        K = -(-T // C)
        pad = K * C - T
        if pad:                                                   # ragged tail: pad queries and keys
            cq, sq = F.pad(cq, (0, 0, 0, pad)), F.pad(sq, (0, 0, 0, pad))
            cw, sw, e = F.pad(cw, (0, 0, 0, pad)), F.pad(sw, (0, 0, 0, pad)), F.pad(e, (0, 0, 0, pad))
        c1 = self.wr * cq + self.wi * sq                          # (B,KC,L)
        c2 = self.wr * sq - self.wi * cq
        Fq = torch.cat([torch.cat([c1, c2], -1).view(B, K, C, 2 * L),
                        torch.cat([-c2, c1], -1).view(B, K, C, 2 * L)], 2)          # (B,K,2C,2L)
        Fk = torch.cat([cw, sw], -1)                                                # (B,KC+L-1,2L)
        N = C + L - 1
        Fk = Fk.unfold(1, N, C).movedim(-1, 2)                                      # (B,K,N,2L)
        ek = e.unfold(1, N, C).movedim(-1, 2)                                       # (B,K,N,dv)
        S = Fq @ Fk.transpose(-1, -2) / L                                           # (B,K,2C,N)
        i = torch.arange(C, device=cq.device)[:, None]
        j = torch.arange(N, device=cq.device)[None]
        band = ((j - i) >= 0) & ((j - i) <= L - 1)                                  # (C,N)
        S = S.masked_fill(~torch.cat([band, band], 0)[None, None], 0)
        o = S @ ek                                                                  # (B,K,2C,dv)
        o = torch.cat([o[:, :, :C], o[:, :, C:]], -1).reshape(B, K * C, 2 * self.dv)
        return o[:, :T]

    def step(self, z_t, h_t, state: State):
        with _NoAutocast(z_t.device):
            B = z_t.size(0)
            p = state["pos"].expand(B)
            phi = self._phase(h_t, p)
            cw = torch.cat([state["c"], phi.cos()[:, None]], 1)
            sw = torch.cat([state["s"], phi.sin()[:, None]], 1)
            e = torch.cat([state["e"], self.V(z_t).to(self.wd)[:, None]], 1)
            psi = self._phase(z_t, p)
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

    def init_state(self, B: int, device) -> State:
        return {
            "long": self.long.init_state(B, device),
            "short": self.short.init_state(B, device),
            "z_prev": torch.zeros(
                B, self.cfg.d, device=device, dtype=self.n.weight.dtype
            ),
        }

    def forward(self, x):
        return self.prefill(x)[0]

    def prefill(self, x, state: Optional[State] = None):
        B = x.size(0)
        st = state if state is not None else self.init_state(B, x.device)
        z = self.n(x)
        h = torch.cat([st["z_prev"].to(z.dtype)[:, None], z[:, :-1]], 1)
        ul, sl = self.long.prefill(z, h, st["long"])
        us, ss = self.short.prefill(z, h, st["short"])
        x = x + self.mix(torch.cat([ul, us], -1))
        x = x + self.ff(self.fn(x))
        return x, {"long": sl, "short": ss, "z_prev": z[:, -1]}

    def step(self, x_t, state: State):
        z = self.n(x_t)
        h = state["z_prev"].to(z.dtype)
        ul, sl = self.long.step(z, h, state["long"])
        us, ss = self.short.step(z, h, state["short"])
        y = x_t + self.mix(torch.cat([ul, us], -1))
        y = y + self.ff(self.fn(y))
        return y, {"long": sl, "short": ss, "z_prev": z}

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
        f"  bf16 autocast vs float32 (big GEMMs in bf16, rest fp32): max rel dev {rel:.1e}"
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
