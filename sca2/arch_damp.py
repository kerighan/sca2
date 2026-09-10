r"""DAMPED long C head: cdelta with a learned PER-MODE DECAY on the accumulator.
A Laplace transform instead of a Fourier one.  Semantics: chead_numpy.py, third head.

Why (CATCHUP.md). The long head's read is a superposition of every write so far:
the addressing is right (the dominant write's key is the current token 97% of the
time in the induction layers) but it carries only ~4% of the read, the other 96%
being 100-200 unrelated writes -- an interference floor in sqrt(N/M) that no
key geometry escapes, and that the delta rule does not touch (it cleans what was
stored at a key BEFORE its write, not the cross-talk that arrives after).  GDN's
state is just as small; it stays clean because it FORGETS.  Switched on at
inference on the trained gen3 checkpoint, a scalar decay already cut the
effective number of writes read from 79 to 19 and improved words new to the
window by 0.08 nats, at the price of the long retrieval the model had been
trained to rely on.  Trained with it, the model can keep retrieval in slow modes
and clean with fast ones.

    S_t = diag(e^{-lambda}) S_{t-1} + c_t (x) e_t,      |c_{t,m}| = 1 still
    o_t = sum_{s<=t} kappa_lam(t,s) e_s,   kappa_lam = (1/M) sum_m w_m e^{-lambda_m (t-s)} e^{i(phi_s - psi_t)}

lambda_m = softplus(a_m), init with memories 1/lambda log-uniform in [64, 4096]
tokens: mild, gen3 nearly nested.  Codes stay unit-modulus, so ||c||^2 = M and the
delta rule keeps its constant read-back normalisation; the decay lives on the
STATE.  In the chunked closed form e^{-lambda (t-s)} factorises into write codes
scaled by e^{+lambda (s-t0)} and read codes scaled by e^{-lambda (t-t0)}, relative
to the chunk start so nothing overflows (lambda <= 1/8, chunks <= 256).

Self-test (float64, against a token loop): python -m sca2.arch_damp
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_cdelta import CHeadDelta, CHeadDeltaKV
from .compiled import wrap as _cw
from .fast_dhead import DHeadSepQPolarFlat
from .registry import register
from .versions.v1_quad_scan import causal_mask


class CHeadDeltaDamp(CHeadDelta):
    """PERSIST: fraction of modes pinned at lambda = 0 exactly -- a guaranteed infinite
    memory that the optimiser cannot erode (softplus never reaches 0 and the gradient
    may push every mode toward forgetting).  Which modes: on the rope grid omega_m
    decreases with m, so the LAST modes are the low frequencies that carry
    long-range addressing -- those are the ones pinned; the high-frequency modes
    (short-range precision, fast aliasing) are the ones that forget.  Horizon
    follows frequency."""
    LAM_MAX = 0.125
    PERSIST = 0.0

    def __init__(self, *a, mem_range=(64.0, 4096.0), **kw):
        super().__init__(*a, **kw)
        lo, hi = mem_range
        mem = torch.exp(torch.empty(self.M).uniform_(math.log(lo), math.log(hi)))
        lam = 1.0 / mem                                            # (M,)
        self.lam_raw = nn.Parameter(torch.log(torch.expm1(lam)))  # softplus^{-1}
        n_pin = int(round(self.PERSIST * self.M))
        mask = torch.ones(self.M)
        if n_pin:
            order = self.omega.abs().argsort()                     # ascending frequency
            mask[order[:n_pin]] = 0.0                              # lowest frequencies persist
        self.register_buffer("lam_mask", mask)

    def lam(self):
        return F.softplus(self.lam_raw).clamp(max=self.LAM_MAX) * self.lam_mask


    # ---- prefill: one chunk, decay folded into scaled codes ------------------ #
    def _prefill_chunk(self, z, h, state):
        B, T, _ = z.shape
        st = state
        M, dv = self.M, self.dv
        lam = self.lam().to(z.dtype)                                  # (M,)
        idx = torch.arange(T, device=z.device, dtype=z.dtype)[:, None]
        gw = torch.exp(lam * idx)                                     # (T,M)  write scale  e^{+lam i}
        gq = torch.exp(-lam * idx)                                    # (T,M)  read scale   e^{-lam i}
        dT = torch.exp(-lam * T)                                      # (M,)   whole-chunk decay
        p = idx + st["pos"]
        pw, pq = self._pw(h, p), self._pq(z, p)
        cw, sw = pw.cos(), pw.sin()
        cq, sq = pq.cos(), pq.sin()
        cw_w, sw_w = cw * gw, sw * gw                                 # scaled write codes
        cw_q, sw_q = cw * gq, sw * gq                                 # write codes as READ at their own time
        cq_q, sq_q = cq * gq, sq * gq                                 # scaled read codes

        empty = st.get("empty", False)
        if not empty:                                                 # incoming state, damped once: D S_in
            sr0 = st["sr"] * torch.exp(-lam)[:, None]
            si0 = st["si"] * torch.exp(-lam)[:, None]

        # --- delta rule: G_lam[i,j] = sum_m e^{-lam(i-j)} cos(phi_i - phi_j)/M  (j < i used)
        beta = torch.sigmoid(self.bproj(z))                           # (B,T,1)
        G = (cw_q @ cw_w.transpose(1, 2) + sw_q @ sw_w.transpose(1, 2)) / M
        A = torch.eye(T, device=z.device, dtype=z.dtype) + beta * G.tril(-1)
        rhs = self._value(z, h)
        if not empty:                                                 # r_i = Re(c_i^H D^{i+1} S_in)/M
            r = (torch.einsum("btm,bmj->btj", cw_q, sr0)
                 + torch.einsum("btm,bmj->btj", sw_q, si0)) / M
            rhs = rhs - beta * r
        e = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)

        # --- read within the chunk: CHeadQuad's kernel on scaled codes
        Am = self.wr * cw_w - self.wi * sw_w
        Bm = self.wr * sw_w + self.wi * cw_w
        Fq = torch.cat([cq_q, sq_q], -1)                              # (B,T,2M)
        Fk = torch.cat([torch.cat([Am, Bm], -1),
                        torch.cat([Bm, -Am], -1)], 1)                 # (B,2T,2M)
        K2 = (Fq @ Fk.transpose(1, 2)).view(B, T, 2, T)
        K2 = K2.masked_fill(causal_mask(T, z.device)[None, :, None, :], 0)
        o = (K2.reshape(B, T * 2, T) @ e).view(B, T, 2 * dv) / M

        if not empty:                                                 # + (w q_i D^{i+1}) S_in / M
            c1 = self.wr * cq_q + self.wi * sq_q
            c2 = self.wr * sq_q - self.wi * cq_q
            r1 = torch.einsum("btm,bmj->btj", c1, sr0)
            r2 = torch.einsum("btm,bmj->btj", c2, si0)
            i1 = torch.einsum("btm,bmj->btj", c2, sr0)
            i2 = torch.einsum("btm,bmj->btj", c1, si0)
            o = o + torch.cat([r1 + r2, i2 - i1], -1) / M

        # --- state out: D^T S_in + D^{T-1} sum_j D^{-j} c_j e_j
        gT = torch.exp(-lam * (T - 1))[:, None]                       # (M,1)
        sr = torch.einsum("btm,btj->bmj", cw_w, e) * gT
        si = torch.einsum("btm,btj->bmj", sw_w, e) * gT
        if not empty:
            sr = sr + st["sr"] * dT[:, None]
            si = si + st["si"] * dT[:, None]
        return (o if getattr(self, "_raw_out", False) else self._out(o, z),
                {"sr": sr, "si": si, "pos": st["pos"] + T, "empty": False})

    # ---- prefill, batched over chunks -------------------------------------- #
    # Everything that does not depend on the incoming state -- scaled codes, Gram,
    # its triangular inverse W = A^{-1}, the intra-chunk kernel K2 -- is computed
    # for ALL full chunks in one batched call (fla's chunked structure). The
    # sequential loop then only carries the state: 5 small GEMMs per chunk instead
    # of the ~12 (Gram, trsm, K2, ...) the per-chunk path launches. Same function
    # bit for bit up to float rounding: checked against the token loop in __main__.
    def _prefill_batched(self, z, h, st, K):
        B, T, _ = z.shape
        M, dv, C = self.M, self.dv, self.CTX
        lam = self.lam().to(z.dtype)                                  # (M,)
        idx = torch.arange(C, device=z.device, dtype=z.dtype)[:, None]
        gw, gq = torch.exp(lam * idx), torch.exp(-lam * idx)         # (C,M) chunk-relative scales
        dC = torch.exp(-lam * C)[:, None]                             # (M,1)  whole-chunk decay
        gT = torch.exp(-lam * (C - 1))[:, None]                       # (M,1)
        d1 = torch.exp(-lam)[:, None]                                 # (M,1)  one-step decay of S_in
        p = torch.arange(T, device=z.device, dtype=z.dtype)[:, None] + st["pos"]
        pw, pq = self._pw(h, p), self._pq(z, p)                       # (B,T,M)
        v = self._value(z, h)                                         # (B,T,dv)
        beta = torch.sigmoid(self.bproj(z))                           # (B,T,1)
        ch = lambda x: x.view(B, K, C, x.shape[-1])                   # (B,K,C,.)
        cw, sw, cq, sq, v, beta = map(ch, (pw.cos(), pw.sin(), pq.cos(), pq.sin(), v, beta))
        cw_w, sw_w = cw * gw, sw * gw                                 # write codes, e^{+lam i}
        cw_q, sw_q = cw * gq, sw * gq                                 # write codes as read, e^{-lam i}
        cq_q, sq_q = cq * gq, sq * gq                                 # read codes, e^{-lam i}
        # Gram and its inverse, all chunks at once
        G = (cw_q @ cw_w.transpose(-1, -2) + sw_q @ sw_w.transpose(-1, -2)) / M        # (B,K,C,C)
        eye = torch.eye(C, device=z.device, dtype=z.dtype)
        A = eye + beta * G.tril(-1)
        W = torch.linalg.solve_triangular(A, eye.expand(B, K, C, C), upper=False,
                                          unitriangular=True)                              # (B,K,C,C)
        # intra-chunk read kernel, all chunks at once
        Am = self.wr * cw_w - self.wi * sw_w
        Bm = self.wr * sw_w + self.wi * cw_w
        Fq = torch.cat([cq_q, sq_q], -1)                                                   # (B,K,C,2M)
        Fk = torch.cat([torch.cat([Am, Bm], -1), torch.cat([Bm, -Am], -1)], 2)             # (B,K,2C,2M)
        K2 = (Fq @ Fk.transpose(-1, -2)).view(B, K, C, 2, C)
        K2 = K2.masked_fill(causal_mask(C, z.device)[None, None, :, None, :], 0)
        K2 = K2.reshape(B, K, 2 * C, C) / M
        # state-side operands, stacked so each chunk needs ONE GEMM per role
        Rq = torch.cat([cw_q, sw_q], -1)                                                   # (B,K,C,2M)  r = Rq @ [sr;si] / M
        c1 = self.wr * cq_q + self.wi * sq_q
        c2 = self.wr * sq_q - self.wi * cq_q
        Cq = torch.cat([c1, c2], -1)                                                       # (B,K,C,2M)  o_state = Cq @ [[sr,si],[si,-sr]] / M
        Pw = torch.cat([cw_w, sw_w], -1).transpose(-1, -2)                                 # (B,K,2M,C)  S += Pw @ e
        sr, si = st["sr"], st["si"]                                                        # (B,M,dv)
        # unbind ONCE: the backward of X[:, k] inside the loop is a full-size zero
        # fill + copy per slice per chunk (measured: 2300 launches/iteration, 60% of
        # the head's time); unbind's backward is a single stack.
        Rq, W, v, beta, K2, Cq, Pw = (x.unbind(1) for x in (Rq, W, v, beta, K2, Cq, Pw))
        gT2 = torch.cat([gT, gT], 0)
        outs = []
        for k in range(K):
            sr0, si0 = sr * d1, si * d1                                                    # damp S_in once
            S2 = torch.cat([sr0, si0], 1)                                                  # (B,2M,dv)
            r = (Rq[k] @ S2) / M                                                           # (B,C,dv)
            e = W[k] @ (v[k] - beta[k] * r)                                                # (B,C,dv)
            o = (K2[k] @ e).view(B, C, 2 * dv)                                             # intra-chunk
            S4 = torch.cat([torch.cat([sr0, si0], -1), torch.cat([si0, -sr0], -1)], 1)     # (B,2M,2dv)
            o = o + (Cq[k] @ S4) / M                                                       # + read of S_in
            outs.append(o)
            upd = (Pw[k] @ e) * gT2                                                        # (B,2M,dv)
            sr = sr * dC + upd[:, :M]
            si = si * dC + upd[:, M:]
        return torch.cat(outs, 1), {"sr": sr, "si": si, "pos": st["pos"] + T, "empty": False}

    # Which prefill implementation: "batched" (intra-chunk work for all chunks in one
    # call, state loop only) or "chunk" (the per-chunk path, one chunk at a time).
    # Same function to float rounding; which is faster depends on the GPU and on
    # (B, T, M) -- `python -m sca2.autotune` measures both in a blocked design and
    # prints the setting. Env SCA2_LONG_PATH overrides the class default.
    LONG_PATH = os.environ.get("SCA2_LONG_PATH", "batched")

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        C = self.CTX
        K = T // C
        if K < 2 or self.LONG_PATH != "batched":                      # short input, or forced
            return super().prefill(z, h, st)
        o, st = self._prefill_batched(z[:, :K * C], h[:, :K * C], st, K)
        if K * C < T:                                                 # ragged tail through the loop
            self._raw_out = True
            try:
                o2, st = self._prefill_chunk(z[:, K * C:], h[:, K * C:], st)
            finally:
                self._raw_out = False
            o = torch.cat([o, o2], 1)
        return self._out(o, z), st

    # ---- decode: damp, read back, write, read ------------------------------- #
    def step(self, z_t, h_t, state):
        d = torch.exp(-self.lam().to(z_t.dtype))[:, None]             # (M,1)
        sr0, si0 = state["sr"] * d, state["si"] * d
        p = state["pos"]
        pw, pq = self._pw(h_t, p), self._pq(z_t, p)
        cwt, swt = pw.cos(), pw.sin()                                 # (B,M)
        beta = torch.sigmoid(self.bproj(z_t))
        vhat = (torch.einsum("bm,bmj->bj", cwt, sr0)
                + torch.einsum("bm,bmj->bj", swt, si0)) / self.M
        e = self._value(z_t, h_t) - beta * vhat
        sr = torch.addcmul(sr0, e[:, None, :], cwt[:, :, None])
        si = torch.addcmul(si0, e[:, None, :], swt[:, :, None])
        cq, sq = pq.cos(), pq.sin()
        c1 = self.wr * cq + self.wi * sq
        c2 = self.wr * sq - self.wi * cq
        cc = torch.stack([c1, c2], 1)
        ss = torch.stack([sr, si], 1)
        m = torch.einsum("bam,bcmj->bacj", cc, ss) / self.M
        u = torch.cat([m[:, 0, 0] + m[:, 1, 1], m[:, 0, 1] - m[:, 1, 0]], -1)
        return (self._out(u, z_t),
                {"sr": sr, "si": si, "pos": state["pos"] + 1, "empty": False})


class CHeadDeltaDampHalf(CHeadDeltaDamp):
    PERSIST = 0.5

class CHeadDeltaDampHalfFast(CHeadDeltaDampHalf):
    """Same, with the decay cap lifted: in catch_shortdamp_s0 the damped half of the
    modes ran INTO the cap (1/lambda = 8 tokens: 71/95 modes in layers 0-1, 53/95 in
    2-3; median memory 8 tokens, p90 20-26, from an init median of ~512). The
    optimiser wanted the damped half to forget faster than 1/8 allowed. The cap is
    bounded by the folded closed form: write codes carry e^{+lambda (s - t0)} with
    s - t0 < CTX, and float32 holds e^{88}, so lambda * CTX must stay under ~60.
    LAM_MAX = 60 / CTX: 0.47 at CTX=128, i.e. memories down to ~2 tokens."""
    LAM_MAX = min(2.0, 60.0 / CHeadDelta.CTX)


class CHeadDeltaDampHalfKV(CHeadDeltaKV, CHeadDeltaDampHalf):
    """key verification on top of the half-persistent damped head: KV supplies
    _value/_out (key copy stored beside the value, cosine gate on the read), Damp
    supplies _prefill_chunk/step (decay on the state). MRO: KV -> DampHalf -> Damp -> Delta."""


register("cdelta_damp", CHeadDeltaDamp, DHeadSepQPolarFlat, arch=True,
         note="cdelta with learned per-mode decay on the accumulator (Laplace)")
register("cdelta_damp_cc", CHeadDeltaDamp, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="cdelta_damp + compile")
register("cdelta_damphf", CHeadDeltaDampHalfFast, DHeadSepQPolarFlat, arch=True,
         note="cdelta_damph with the decay cap lifted to lambda <= 2")
register("cdelta_damph", CHeadDeltaDampHalf, DHeadSepQPolarFlat, arch=True,
         note="cdelta_damp with the lowest-frequency half of the modes pinned at lambda=0")
register("cdelta_damph_cc", CHeadDeltaDampHalf, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="cdelta_damph + compile")


if __name__ == "__main__":
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    d, M, dv, B, T = 8, 12, 4, 2, 45
    hd = CHeadDeltaDamp(d, M, freq="rope", theta_scale=0.3, dv=dv, max_len=T, mem_range=(4.0, 60.0)).double()
    # (.double() converts the float32 omega buffer; registry.build does the same via .to(dtype).
    #  Without it a 0-dim position tensor times float32 omega yields a float32 phase: 1e-8 noise.)
    hd.CTX = 16                                                       # several chunks, ragged tail
    z = torch.randn(B, T, d); h = torch.roll(z, 1, 1); h[:, 0] = 0
    with torch.no_grad():
        # independent reference: the numpy semantics, token by token, complex state
        lam = hd.lam(); S = torch.zeros(B, M, dv, dtype=torch.complex128); ref = []
        for t in range(T):
            S = S * torch.exp(-lam)[None, :, None]
            phi = hd._pw(h[:, t], torch.tensor(float(t))); c = torch.exp(1j * phi)           # (B,M)
            vhat = torch.einsum("bm,bmj->bj", c.conj(), S).real / M
            beta = torch.sigmoid(hd.bproj(z[:, t]))
            e = hd.V(z[:, t]) - beta * vhat
            S = S + c[:, :, None] * e[:, None, :]
            psi = hd._pq(z[:, t], torch.tensor(float(t))); q = torch.exp(-1j * psi)
            o = torch.einsum("bm,bmj->bj", (hd.wr + 1j * hd.wi) * q, S) / M
            ref.append(torch.cat([o.real, o.imag], -1))
        ref = torch.stack(ref, 1)
        hd.rms_read = False; hd.rscale = torch.ones(2 * dv)          # raw output for the comparison
        u, _ = hd.prefill(z, h)                                       # chunked (16|16|13)
        print(f"chunked prefill vs token-loop reference: {(u - ref).abs().max().item():.2e}")
        st = hd.init_state(B, z.device, z.dtype); outs = []
        for t in range(T):
            o, st = hd.step(z[:, t], h[:, t], st); outs.append(o)
        print(f"step vs reference:                        {(torch.stack(outs, 1) - ref).abs().max().item():.2e}")
        hd.CTX = 1024; u1, _ = hd.prefill(z, h)
        print(f"single-chunk prefill vs reference:        {(u1 - ref).abs().max().item():.2e}")
        hd.CTX = 16; ub, stb = hd.prefill(z, h)                       # batched: K=2 chunks + tail 13
        print(f"BATCHED prefill (2 chunks + tail) vs ref: {(ub - ref).abs().max().item():.2e}")
        hd.CTX = 15; ub2, stb2 = hd.prefill(z, h)                     # K=3 chunks, tail 0
        print(f"BATCHED prefill (3 chunks, no tail) vs ref:{(ub2 - ref).abs().max().item():.2e}")
        st_ref = S
        print(f"BATCHED final state vs reference:         {max((stb2['sr'] - st_ref.real).abs().max().item(), (stb2['si'] - st_ref.imag).abs().max().item()):.2e}")
        assert (ub - ref).abs().max() < 1e-10 and (ub2 - ref).abs().max() < 1e-10
        # lambda -> 0 must give cdelta exactly
        hd.lam_raw.fill_(-60.0); hd.CTX = 16; u0, _ = hd.prefill(z, h)
        # compare against the parent's own chunk implementation on the same weights
        parent = CHeadDelta(d, M, freq="rope", theta_scale=0.3, dv=dv, max_len=T).double(); parent.load_state_dict(
            {k: v for k, v in hd.state_dict().items() if k not in ("lam_raw", "lam_mask")}); parent.CTX = 16
        parent.rms_read = False; parent.rscale = torch.ones(2 * dv)
        up, _ = parent.prefill(z, h)
        print(f"lambda=0 vs cdelta:                       {(u0 - up).abs().max().item():.2e}")
    assert (u - ref).abs().max() < 1e-10 and (u1 - ref).abs().max() < 1e-10 and (u0 - up).abs().max() < 1e-10
    print("ALL OK")
