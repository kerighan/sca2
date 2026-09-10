r"""C head with an error-correcting (delta) write in the complex domain.

WHY. Every other lever tried on the C head moved capacity around and none of it
closed the 0.08 nats against GDN: Mc/Md/dv reshaping, depth, gated read-out,
c_decay, conv, theta, and finally per-value-group spectral weights (arch_wgroup,
a measured null -- so the rank-2 limit on the temporal profile was NOT the
binding constraint). The one structural difference left is the WRITE:

    SCA2   S <- S + phi_t v_t^H          additive, |phi_{t,m}| = 1 for all m
    GDN    S <- S + k_t (v_t - S^T k_t)^H  error-correcting

Additive means a write can never be taken back. GDN's write removes whatever was
already stored at its own key before storing the new value, so it can overwrite
an association; SCA2 can only pile on, and with constant-amplitude codes every
write touches all M slots with equal magnitude. That is the "no selective
erasure" property, and it is what this module removes.

THE DERIVATION. It is short, and its conclusion is that the delta rule costs one
triangular solve and nothing else -- no change to the read, no change to the
state carry.

State S in C^{M x dv}, write code phi_t = e^{i.pw_t} in C^M, and ||phi_t||^2 = M
exactly (constant amplitude), so the value stored at key phi is recovered by

    vhat^H = Re(phi^H S) / M          (exact: S = phi v^H  =>  vhat = v)

Write, with a data-dependent gate beta_t in (0,1) on the erase term only:

    S_t = S_{t-1} + phi_t (v_t - beta_t.vhat_t)^H,   vhat_t^H = Re(phi_t^H S_{t-1})/M

beta_t = 0 is the additive baseline EXACTLY, beta_t = 1 is the full delta rule,
so the baseline is nested and the gate is free to stay at zero if erasure does
not pay. (Putting beta on the whole write instead, as DeltaNet does, would not
nest the baseline -- beta = 0 would write nothing at all.)

Let e_t be the written value, e_t = v_t - beta_t.vhat_t. Since S_{t-1} =
S_0 + sum_{s<t} phi_s e_s^H and G[t,s] = Re(phi_t^H phi_s)/M,

    e_t + beta_t . sum_{s<t} G[t,s] e_s = v_t - beta_t . r_t,   r_t^H = Re(phi_t^H S_0)/M

which in matrix form over a chunk is UNIT lower triangular, hence one solve:

    (I + diag(beta) . tril(G,-1)) E = V - beta .* R

Then S_T = S_0 + sum_t phi_t e_t^H and the read at time t sees S_t, so BOTH the
intra-chunk quadratic term and the closing state are the baseline's, with v
replaced by E. The whole delta rule is: solve for E, then run CHeadQuad.

WHAT SELECTIVITY THE ERASURE ACTUALLY HAS -- the caveat that decides this.
The erase at t removes the component along phi_t, and it hits a previous write
at phi_s in proportion to G[t,s] = (1/M) sum_m cos(pw[t,m] - pw[s,m]). With
theta = 0 the phase is purely positional, so G[t,s] = kappa(t-s) is a function of
LAG alone and the erasure is position-selective, not content-selective. Two
consequences, and they are different experiments:

  theta = 0, freq=dft.  kappa is a clean delta, so G = I, tril(G,-1) = 0 and
        E = V: the delta rule is an exact NO-OP. Nothing to test.
  theta = 0, freq=rope. kappa is broad (kappa[1] = 0.808) and the positional
        basis is near-degenerate (rank 49/128 at T=128, see ref.freq_grid).
        Here G is the GRAM MATRIX of a non-orthogonal dictionary and
        E = (I + tril(G,-1))^{-1} V is its Gram-Schmidt whitening. So on the
        best-measured grid the delta rule attacks a documented defect --
        redundancy of the code book -- rather than erasure per se.
  theta > 0.  The code depends on content, G mixes lag and content, and this is
        the actual GDN mechanism: overwrite the association at a key.

rope is the grid that measured best and it is also the one where erasure is
LEAST content-selective. That tension is not resolvable on paper; both arms are
cheap, so both get run.

Chunk size via SCA2_CTX_CHUNK (default 256), as in versions/v3_longctx.py. Added
cost per chunk is one (B,C,C) Gram matmul and one (B,C,C)x(B,C,dv) triangular
solve, next to a quadratic kernel of the same shape that is already being formed.

`python -m sca2.arch_cdelta` checks the closed form against the sequential
recurrence, and checks that beta = 0 reproduces CHeadQuad bit-for-bit.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiled import wrap as _cw
from .fast_dhead import DHeadSepQPolarFlat
from .ref import _gated_out
from .registry import register
from .versions.v1_quad_scan import CHeadQuad, causal_mask


class CHeadDelta(CHeadQuad):
    CTX = int(os.environ.get("SCA2_CTX_CHUNK", 256))

    def __init__(self, *a, cdelta_init=-2.0, **kw):
        super().__init__(*a, **kw)
        # Zero weight + negative bias: beta starts data-independent and small
        # (sigmoid(-2) = 0.12), so training departs from the known-good additive
        # baseline gradually. The gradient is NOT dead there -- dE/dbeta != 0 at
        # beta = 0 -- unlike theta, which really was blocked at its zero init.
        self.bproj = nn.Linear(self.d, 1, True)
        nn.init.zeros_(self.bproj.weight)
        nn.init.constant_(self.bproj.bias, cdelta_init)

    # hooks: what is stored at a code, and what leaves the head (CHeadDeltaKV)
    def _value(self, z, h):
        return self.V(z)

    def _out(self, u, z):
        return _gated_out(self, u, z)

    def _pw(self, h, p):
        """Write phase. The Gram matrix G, and hence the whole cost of the delta
        rule, depends on THIS and not on the read phase -- see CHeadDeltaWPos."""
        return self.K(h) * self.theta + p * self.omega

    def _pq(self, z, p):
        return self.K(z) * self.theta + p * self.omega

    def _gram(self, cw, sw):
        """G[t,s] = Re(phi_t^H phi_s)/M, the pairwise overlap of write codes."""
        return (cw @ cw.transpose(1, 2) + sw @ sw.transpose(1, 2)) / self.M

    def _solve(self, cw, sw, v, beta, st):
        """E = (I + diag(beta).tril(G,-1))^{-1} (V - beta.*R)."""
        B, T, _ = v.shape
        G = self._gram(cw, sw)
        A = torch.eye(T, device=v.device, dtype=v.dtype) + beta * G.tril(-1)
        rhs = v
        if not st.get("empty", False):
            r = (torch.einsum("btm,bmj->btj", cw, st["sr"])
                 + torch.einsum("btm,bmj->btj", sw, st["si"])) / self.M
            rhs = v - beta * r
        return torch.linalg.solve_triangular(A, rhs, upper=False,
                                             unitriangular=True)

    def _prefill_chunk(self, z, h, state):
        B, T, _ = z.shape
        st = state
        p = (torch.arange(T, device=z.device, dtype=z.dtype)
             + st["pos"])[:, None]
        pw, pq = self._pw(h, p), self._pq(z, p)
        cw, sw = pw.cos(), pw.sin()
        cq, sq = pq.cos(), pq.sin()

        beta = torch.sigmoid(self.bproj(z))                   # (B,T,1)
        e = self._solve(cw, sw, self._value(z, h), beta, st)  # (B,T,dv)

        # --- from here this is CHeadQuad.prefill verbatim, with v -> e ---
        A = self.wr * cw - self.wi * sw
        Bm = self.wr * sw + self.wi * cw
        Fq = torch.cat([cq, sq], -1)                          # (B,T,2M)
        Fk = torch.cat([torch.cat([A, Bm], -1),
                        torch.cat([Bm, -A], -1)], 1)          # (B,2T,2M)
        K2 = (Fq @ Fk.transpose(1, 2)).view(B, T, 2, T)
        K2 = K2.masked_fill(causal_mask(T, z.device)[None, :, None, :], 0)
        o = (K2.reshape(B, T * 2, T) @ e).view(B, T, 2 * self.dv) / self.M

        if not st.get("empty", False):
            sr0, si0 = st["sr"], st["si"]
            c1 = self.wr * cq + self.wi * sq
            c2 = self.wr * sq - self.wi * cq
            r1 = torch.einsum("btm,bmj->btj", c1, sr0)
            r2 = torch.einsum("btm,bmj->btj", c2, si0)
            i1 = torch.einsum("btm,bmj->btj", c2, sr0)
            i2 = torch.einsum("btm,bmj->btj", c1, si0)
            o = o + torch.cat([r1 + r2, i2 - i1], -1) / self.M

        sr = torch.einsum("btm,btj->bmj", cw, e) + st["sr"]
        si = torch.einsum("btm,btj->bmj", sw, e) + st["si"]
        return (self._out(o, z),
                {"sr": sr, "si": si, "pos": st["pos"] + T, "empty": False})

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        C = self.CTX
        if T <= C:
            return self._prefill_chunk(z, h, st)
        outs = []
        for s0 in range(0, T, C):
            o, st = self._prefill_chunk(z[:, s0:s0 + C], h[:, s0:s0 + C], st)
            outs.append(o)
        return torch.cat(outs, 1), st

    def step(self, z_t, h_t, state):
        """One token. vhat is read from the state with the WRITE code, then the
        corrected value is written -- the parent's step with v -> e."""
        p = state["pos"]
        pw, pq = self._pw(h_t, p), self._pq(z_t, p)
        cwt, swt = pw.cos(), pw.sin()                          # (B,M)
        beta = torch.sigmoid(self.bproj(z_t))                  # (B,1)
        vhat = (torch.einsum("bm,bmj->bj", cwt, state["sr"])
                + torch.einsum("bm,bmj->bj", swt, state["si"])) / self.M
        e = self._value(z_t, h_t) - beta * vhat
        sr = torch.addcmul(state["sr"], e[:, None, :], cwt[:, :, None])
        si = torch.addcmul(state["si"], e[:, None, :], swt[:, :, None])
        cq, sq = pq.cos(), pq.sin()
        c1 = self.wr * cq + self.wi * sq
        c2 = self.wr * sq - self.wi * cq
        cc = torch.stack([c1, c2], 1)
        ss = torch.stack([sr, si], 1)
        m = torch.einsum("bam,bcmj->bacj", cc, ss) / self.M
        u = torch.cat([m[:, 0, 0] + m[:, 1, 1], m[:, 0, 1] - m[:, 1, 0]], -1)
        return (self._out(u, z_t),
                {"sr": sr, "si": si, "pos": state["pos"] + 1, "empty": False})


class CHeadDeltaWPos(CHeadDelta):
    r"""REJECTED. Delta rule with a POSITIONAL write phase: fast, and much worse.

    Kept registered so the result stays measurable instead of becoming folklore,
    and because the reasoning error below is worth not repeating.

    It did buy the speed back -- 77,900 tok/s against the additive baseline's
    81,400 and cdelta's 68,400, so 1.04x instead of 1.19x. And it cost about 0.9
    nats: at 140M tokens val 3.705 against cdelta's 2.789, having separated by
    27M and plateaued near 3.75 from 100M while cdelta kept descending.

    WHY, AND THE ERROR THIS DOCSTRING ORIGINALLY MADE. It claimed "the read stays
    content-addressed, so this is not a return to a fixed kernel". That is wrong.
    The read kernel depends on the DIFFERENCE of the two phases,

        kappa(t,s) ~ sum_m w_m e^{i(pq_{t,m} - pw_{s,m})}

    With both phases content-dependent this is
    (xi(z_t) - zeta(h_s)) + (t-s).omega, a content MATCH -- token t retrieves
    from the positions whose stored content phase agrees with its query. Freeze
    the write phase and it becomes xi(z_t) + (t-s).omega, which depends on s only
    through the lag: the query's content term now merely shifts WHICH lag is read,
    uniformly, and cannot select on the content of what was WRITTEN. Associative
    retrieval is gone from the whole head, not just from the erase.

    So the read's content phase is only meaningful RELATIVE to the write's. The
    two cannot be decoupled, which also explains why theta drifts to |theta| =
    1.8..11.4 rather than staying near zero.

    CONSEQUENCE FOR THE SPEED PROBLEM: this route is closed, not just suboptimal.
    A constant Gram requires a content-free write phase, and a content-free write
    phase costs the mechanism that makes the layer work. The Gram must be paid.
    The remaining levers are the chunk size C (cost is linear in it, and chunking
    is exact) and shrinking M itself, which the delta rule may now permit.

    The original rationale, kept because the FLOP accounting in it is still right:

    Delta rule with a POSITIONAL write phase, which makes it nearly free.

    Measured cost of `cdelta` is 1.19x the additive baseline, and the FLOP
    accounting says where it goes -- per chunk of C, with M=378 and dv=32:

        K2 kernel (baseline)   4.C^2.M
        Gram G     (added)     2.C^2.M      <- 50% of the kernel
        triangular solve       C^2.dv       <- 4% of the Gram

    So the solve is not the cost, the Gram is. And G depends only on the WRITE
    phase. Freeze the content part of that phase and

        G[t,s] = (1/M) sum_m cos((t-s).omega_m)

    is a function of the lag alone: one constant Toeplitz matrix, identical for
    every batch element, every chunk and every training step. It is computed
    once and cached, and the 2.C^2.M matmul disappears. The C head also stops
    reading h altogether, which removes a second B.T.d.M matmul per layer.

    WHY FREEZE THE WRITE AND NOT ALL OF theta. Freezing all of it zeroes the
    read phase too, so K becomes entirely unused -- 193,536 dead parameters out
    of 743,528, 26% of the layer budget. That arm would be 26% smaller in
    effect and the comparison would be confounded. Freezing only the write keeps
    K fully used, on the read, and leaves the parameter count IDENTICAL to
    cdelta (theta has the same M entries; it is simply no longer applied to h).

    WHAT IT COSTS FUNCTIONALLY, stated plainly: the erase is now addressed by
    LAG, not by content. It removes what was written at nearby positions, per
    the Dirichlet kernel of the grid, rather than the association at a matching
    key. That is a weaker mechanism than GDN's, and on the `rope` grid it is
    exactly the Gram-Schmidt whitening of a dictionary documented as
    near-degenerate (rank 49/128 at T=128). Since theta cannot drift here, this
    is also the ONLY arm that actually tests that whitening hypothesis --
    cdelta_t0 did not, because its theta ran from 0 to |theta| = 1.8..11.4.

    -- end of the original rationale. Its error was accounting for the Gram while
    forgetting that the same phase also carries the read's addressing.
    """
    _TOEPLITZ = {}

    def _pw(self, h, p):
        # h is ignored: the write phase is positional. p is (T,1) in prefill and
        # a 0-dim tensor in step; expand to a batch so the downstream matmuls and
        # einsums keep their shapes.
        pw = p * self.omega
        B = h.shape[0]
        return pw.expand(B, *pw.shape) if pw.dim() == 2 else pw.expand(B, -1)

    def _gram(self, cw, sw):
        C = cw.shape[-2]
        key = (C, cw.device, cw.dtype)
        G = self._TOEPLITZ.get(key)
        if G is None:
            lag = (torch.arange(C, device=cw.device, dtype=cw.dtype)[:, None]
                   - torch.arange(C, device=cw.device, dtype=cw.dtype)[None, :])
            G = (lag[:, :, None] * self.omega.to(cw.dtype)).cos().mean(-1)
            self._TOEPLITZ[key] = G = G[None]          # (1,C,C), broadcasts
        return G



class CHeadDeltaBP(CHeadDelta):
    """cdelta with a BOUNDED content phase: hash keys -> metric keys.

    Conjecture 1 of CATCHUP.md. In cdelta the content phase K(h)*theta is
    unbounded and theta drifts to |theta| = 1.8..11.4, so two inputs h, h' that
    differ by a little in K(h) land at phases that differ by several turns: the
    code cos(Delta) is then a HASH of the content -- excellent for retrieving an
    exact repeat, no notion of "similar". GDN's L2-normalised dot-product keys
    are the opposite: nearby contents share their memory.

    Here the content phase is squashed to (-BOUND, BOUND) per mode:

        c = BOUND . tanh(K(h) theta / BOUND),      pw = c + p . omega

    so Delta_c lies in (-2.BOUND, 2.BOUND). cos(Delta_c) is monotone in |Delta_c|
    only on [0, pi], hence BOUND = pi/2 is the largest bound for which the key
    overlap is a genuine similarity (a metric key); BOUND = pi still wraps once
    and is kept as the intermediate arm. The positional term is untouched: the
    lag kernel and the Dirichlet init are exactly those of cdelta, and at the
    theta_scale = 0.02 init the tanh is in its linear regime, so the two arms
    start from (numerically) the same function and diverge only where theta
    grows.

    Nothing else changes: same Gram, same solve, same read, so prefill and
    token-by-token decode stay iso by construction (checked by iso.py --self).
    Prediction that would confirm the conjecture: the LATE slope of the loss on
    word_new tokens (sca2/tokclass.py) improves, word_rep does not worsen.
    """
    BOUND = float(os.environ.get("SCA2_PHASE_BOUND", math.pi / 2))

    def _content(self, x):
        b = self.BOUND
        return b * torch.tanh(self.K(x) * self.theta / b)

    def _pw(self, h, p):
        return self._content(h) + p * self.omega

    def _pq(self, z, p):
        return self._content(z) + p * self.omega


class CHeadDeltaBP2(CHeadDeltaBP):
    BOUND = math.pi


register("cdelta_bp", CHeadDeltaBP, DHeadSepQPolarFlat, arch=True,
         note="cdelta, content phase bounded to +-pi/2 (metric keys)")
register("cdelta_bp_cc", CHeadDeltaBP, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="cdelta_bp + compile")
register("cdelta_bp2", CHeadDeltaBP2, DHeadSepQPolarFlat, arch=True,
         note="cdelta, content phase bounded to +-pi")
register("cdelta_bp2_cc", CHeadDeltaBP2, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="cdelta_bp2 + compile")


class CHeadDeltaRaw(CHeadDelta):
    """cdelta whose READ keeps its magnitude (no RMS normalisation).

    CATCHUP.md: with the C head crippled (Mc=2) the word_new deficit vanishes
    (+0.28 -> ~0), so the C head actively hurts tokens for which it holds nothing.
    Mechanism proposed: _rms() rescales every read to unit RMS, so a read that
    matched nothing (small, random) leaves the head looking exactly like one that
    retrieved an exact repeat; the mix is linear and cannot tell them apart; the
    input-driven gate (gated_read) could not either, because the evidence is in
    the read, not in z. Keeping the magnitude hands that evidence downstream.
    rscale (2*dv, init 1) is the only new parameter: +112 per layer.
    """
    rms_read = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rscale = nn.Parameter(torch.ones(2 * self.dv))


register("cdelta_raw", CHeadDeltaRaw, DHeadSepQPolarFlat, arch=True,
         note="cdelta, C read keeps its magnitude (no RMS)")
register("cdelta_raw_cc", CHeadDeltaRaw, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="cdelta_raw + compile")


class CHeadDeltaKV(CHeadDelta):
    """cdelta with KEY VERIFICATION: the write stores a copy of its own key next to
    the value, the read checks it, and the check gates the output.

    CATCHUP.md: the C head costs ~0.28 nats on words new to the window, uniformly,
    and neither the query z (gated_read) nor the read's magnitude (cdelta_raw) tells
    "found" from "not found" -- measured: read norms on new and repeated words are
    the same distribution. So the gate needs evidence the read itself carries.

        stored value   e_s = [ V(z_s) ; Kv(h_s) ]          (dv + dk)
        read           o_t = [ value part ; key part ]     complex, as usual
        evidence       m_t = cos( Re key part , Kv(z_t) )  in [-1, 1]
        gate           g_t = sigmoid(a . m_t + b)
        output         RMS(value part) . g_t               (2*dv, as before)

    A genuine match wrote its key Kv(h_s) with h_s ~ z_t, so the key read back
    agrees with Kv(z_t); a random mixture does not. Everything else -- codes, Gram,
    delta rule, decode -- is CHeadDelta's with dv -> dv + dk, so prefill and step stay
    iso by construction. +2*d*dk + 2 params (dk=16: 4098 per layer)."""
    DK = 16

    def __init__(self, d, M, dv=None, **kw):
        dv = dv if dv is not None else d // 2
        super().__init__(d, M, dv=dv + self.DK, **kw)   # internal width carries the key copy
        self.dv_out = dv
        self.V = nn.Linear(d, dv, False)                # value projection back to dv (no dead rows)
        self.Kv = nn.Linear(d, self.DK, False)
        self.ga = nn.Parameter(torch.tensor(4.0))       # gate slope on the cosine
        self.gb = nn.Parameter(torch.tensor(0.0))       # gate bias: g = 0.5 at zero evidence

    def _value(self, z, h):
        return torch.cat([self.V(z), self.Kv(h)], -1)

    def _out(self, u, z):
        dv, dk = self.dv_out, self.DK
        re, im = u[..., :dv + dk], u[..., dv + dk:]
        val = torch.cat([re[..., :dv], im[..., :dv]], -1)                      # (..., 2*dv)
        m = F.cosine_similarity(re[..., dv:], self.Kv(z), dim=-1, eps=1e-6)    # (...)
        g = torch.sigmoid(self.ga * m + self.gb)[..., None]
        return _gated_out(self, val, z) * g


register("cdelta_kv", CHeadDeltaKV, DHeadSepQPolarFlat, arch=True,
         note="cdelta + key verification gate on the read")
register("cdelta_kv_cc", CHeadDeltaKV, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="cdelta_kv + compile")

register("cdelta", CHeadDelta, DHeadSepQPolarFlat, arch=True,
         note="complex error-correcting C write (delta rule)")
register("cdeltaw", CHeadDeltaWPos, DHeadSepQPolarFlat, arch=True,
         note="delta rule with positional write phase (Toeplitz Gram)")
register("cdeltaw_cc", CHeadDeltaWPos, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="delta rule, positional write phase + compile")
register("cdelta_cc", CHeadDelta, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="complex error-correcting C write + compile")


def _selfcheck():
    """Closed form vs the sequential recurrence it claims to implement.

    Two things are checked, and the second is the one that catches sign errors:
    beta = 0 must reproduce CHeadQuad, so the delta rule is a strict extension.
    T is forced above CTX so the multi-chunk carry runs -- a single chunk skips
    the r_t term entirely and would pass even if that term were wrong.
    """
    torch.manual_seed(0)
    d, M, dv, B, T = 32, 16, 8, 2, 40
    CHeadDelta.CTX = 16
    hd = CHeadDelta(d, M, freq="rope", theta_scale=0.1, dv=dv).double()
    z = torch.randn(B, T, d, dtype=torch.float64)
    h = torch.roll(z, 1, 1)
    h[:, 0] = 0

    o_fast, st_fast = hd.prefill(z, h)

    # sequential: the recurrence, one token at a time, exactly as derived
    st = hd.init_state(B, z.device, z.dtype)
    outs = []
    for t in range(T):
        o, st = hd.step(z[:, t], h[:, t], st)
        outs.append(o)
    o_seq = torch.stack(outs, 1)

    e1 = (o_fast - o_seq).abs().max().item()
    e2 = max((st_fast[k] - st[k]).abs().max().item() for k in ("sr", "si"))
    print(f"closed form vs sequential:  out {e1:.2e}  state {e2:.2e}")

    # beta -> 0 must be CHeadQuad exactly
    nn.init.constant_(hd.bproj.bias, -60.0)
    ref = CHeadQuad(d, M, freq="rope", theta_scale=0.1, dv=dv).double()
    ref.load_state_dict({k: v for k, v in hd.state_dict().items()
                         if not k.startswith("bproj")}, strict=False)
    o_ref, st_ref = ref.prefill(z, h)
    o_b0, st_b0 = hd.prefill(z, h)
    e3 = (o_ref - o_b0).abs().max().item()
    print(f"beta=0 vs CHeadQuad:        out {e3:.2e}")
    assert e1 < 1e-9 and e2 < 1e-9 and e3 < 1e-9, (e1, e2, e3)

    # --- CHeadDeltaWPos ---------------------------------------------------- #
    # The cached Toeplitz IS the speed win, so it is checked against the explicit
    # Gram it replaces. A wrong G would not raise: it would silently train a
    # different model, which is the worst possible failure here.
    wp = CHeadDeltaWPos(d, M, freq="rope", theta_scale=0.1, dv=dv).double()
    p = torch.arange(40, dtype=torch.float64)[:, None]
    pw = wp._pw(z, p)
    cw, sw = pw.cos(), pw.sin()
    G_explicit = (cw @ cw.transpose(1, 2) + sw @ sw.transpose(1, 2)) / M
    G_cached = wp._gram(cw, sw).expand_as(G_explicit)
    e4 = (G_explicit - G_cached).abs().max().item()
    print(f"Toeplitz G vs explicit:     {e4:.2e}")

    o_fast, st_fast = wp.prefill(z, h)
    st = wp.init_state(B, z.device, z.dtype)
    outs = []
    for t in range(T):
        o, st = wp.step(z[:, t], h[:, t], st)
        outs.append(o)
    e5 = (o_fast - torch.stack(outs, 1)).abs().max().item()
    e6 = max((st_fast[k] - st[k]).abs().max().item() for k in ("sr", "si"))
    print(f"wpos closed vs sequential:  out {e5:.2e}  state {e6:.2e}")

    # The write phase must not depend on h at all -- that is what makes G constant.
    o_other, _ = wp.prefill(z, torch.randn_like(h))
    e7 = (o_fast - o_other).abs().max().item()
    print(f"wpos ignores h:             {e7:.2e}")
    assert e4 < 1e-12 and e5 < 1e-9 and e6 < 1e-9 and e7 == 0.0, (e4, e5, e6, e7)
    print("OK")


if __name__ == "__main__":
    _selfcheck()
