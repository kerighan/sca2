r"""SCA2's best measured configuration, as one self-contained reference layer.

Extracted for external mathematical analysis. This is the exact function of the
`v3polarflat` variant at the shape that measured best (see FACTS below); the
repo's production path computes the same thing with a chunked closed form and a
loop-free D head. `python best_layer.py` checks this file against that path.

STATUS: THE GAP THIS DOCUMENT WAS WRITTEN TO EXPLAIN IS CLOSED. Question 1
below -- does the additive write cost capacity against an error-correcting one
-- was the right question, and the answer is yes: adding a delta rule to the C
head's write, and changing nothing else, is worth -0.111 to -0.118 nats at n=3
depending on which theta init it is matched at, resolved either way and the
strongest effect in the campaign. Against Gated DeltaNet nothing resolves at n=3,
before or after -- see the correction in FACTS. Question 2, the rank-2 limit on
the shared temporal profile, was measured and is NOT the binding constraint.
Details in FACTS and in the questions themselves.

The layer described below is still the ADDITIVE one, unchanged, because it is
the control the delta rule is measured against. sca2/arch_cdelta.py holds the
fix and derives it in full.

================================================================================
NOTATION
================================================================================
d = 128 residual width, T sequence length, B batch.
z_t = LayerNorm(x_t) is the layer input at position t; h_t = z_{t-1} (a hard
shift by one, not learned). Both heads carry a state of shape (M, dv), read at
every position, and are summed through one output projection.

--- C head (Mc = 378, dv = 32, no forgetting) -----------------------------------
Per-channel phases, m = 1..Mc, with omega_m = pi * 10000^{-(m-1)/(Mc-1)}:

    pw_{t,m} = theta_m <K_m, h_t> + t . omega_m          (write phase)
    pq_{t,m} = theta_m <K_m, z_t> + t . omega_m          (read phase)

    S_{t,m} = sum_{s<=t} V(z_s) e^{i pw_{s,m}}   in C^{dv}      (pure cumsum)
    u^C_t   = RMS( mean_m [ w_m S_{t,m} e^{-i pq_{t,m}} ] )     in R^{2 dv}

where w_m = wr_m + i wi_m and the R^{2dv} embedding is (Re, Im). Equivalently,
u^C_t = RMS( sum_{s<=t} kappa_{t,s} V(z_s) ) with the SCALAR kernel

    kappa_{t,s} = (1/Mc) sum_m w_m exp( i [ theta_m <K_m, h_s - z_t> + (s-t) omega_m ] )

so the C head is linear attention with a kernel that is (a) shift-invariant in
the content difference h_s - z_t and (b) modulated by the lag s-t.

Caveat on that reading: w_m = wr_m + i.wi_m is a FREE complex weight, so the
spectral measure is signed and kappa need not be positive definite. This is a
learned Fourier expansion, NOT a Random Fourier Features estimator of a PSD
kernel, and Bochner's guarantees do not transfer.

At theta = 0 the content term vanishes and kappa becomes a learned causal
convolution in the lag alone, f(s-t) = (1/Mc) sum_m w_m e^{i(s-t)omega_m} --
uniform in CONTENT, not uniform in position.

--- D head (Md = 4, G = 8, dv = 32, gs = dv/G = 4, complex diagonal gate) -------
Gate a_t in C^{M x G}, broadcast over the gs coordinates of each group:

    a_{t,m,g} = exp(-softplus(<gr_{m,g}, h_t> + b)) . exp(i <gi_{m,g}, h_t> + i b')

so |a| <= 1 exactly by construction. State D_t in C^{M x dv}, decay then write,
and note that only the REAL part receives the input:

    D_t = a_t (.) D_{t-1} + V(z_t)          (broadcast over m; V real)

Read with a RANK-ONE query, alpha in C^{M} over channels and beta in C^{dv} over
coordinates, each normalized per element (|alpha_m| = |beta_j| = 1):

    A_t = mean_m Re( conj(alpha_{t,m}) D_{t,m} ),  B_t = mean_m Im( conj(alpha) D )
    u^D_t = RMS( [ br A + bi B , br B - bi A ] )                in R^{2 dv}

--- output ---------------------------------------------------------------------
    x_t <- x_t + W_mix [u^C_t ; u^D_t],   then  x_t <- x_t + FFN(LayerNorm(x_t))

================================================================================
FACTS (pycode, 1024-token blocks, 4 layers, 743.6k layer params, one epoch of
177M tokens, val loss in nats; n = number of seeds)
================================================================================
    Gated DeltaNet, matched params      2.7882 +- 0.0726  (n=3)
    THIS layer                          2.9083 +- 0.0334  (n=3)
    gap                                 +0.1201 +- 0.0461, Welch t=2.61, p~0.086

THIS LAYER PLUS A DELTA RULE ON THE C WRITE (sca2/arch_cdelta.py). Same shape,
same data, same seeds; +516 params for the erase gate, 0.07%. All arms below are
interpolated to a common 168.7M tokens, which moves the two rows above by less
than 0.003.
    delta write, theta init 0.02        2.7480 +- 0.0147  (n=3)
    delta write, theta init 0           2.7879 +- 0.0198  (n=3)
    Gated DeltaNet                      2.7869 +- 0.0733  (n=3)
    THIS layer (additive write)         2.9058 +- 0.0356  (n=3)

"resolved" below means |t| beats the real two-sided 95% t critical value at that
dof, which at n=3 is 2.9-4.3 and NOT 2. A second, seed-PAIRED estimator is quoted
as a check (dump_cdelta.paired); it agrees on every classification.

                              endpoint   paired   resolved?
    delta(0.02) - additive     -0.1577  -0.1865   YES, but spans TWO changes
    delta(0.02) - add(0.02)    -0.1111  -0.1463   YES  <- the delta rule alone
    delta(0)    - additive     -0.1179  -0.1366   YES  <- the delta rule alone
    delta(0.02) - delta(0)     -0.0398  -0.0501   no  (p=0.053)
    add(0.02)   - additive     -0.0467  -0.0369   no  (p=0.140)
    delta(0.02) - GDN          -0.0388  -0.0832   no  (p=0.456)
    additive    - GDN          +0.1189  +0.0994   no  (p=0.088)

ATTRIBUTION. THIS layer's runs use --theta-scale's default of 0.0, so the
-0.1577 row spans the delta rule AND a theta init change and must not be quoted
as the delta rule's effect. theta_ctrl.sh supplied the missing cell -- this layer
at init 0.02, 2.8591 +- 0.0233 (n=3) -- so the delta rule is now measured at BOTH
matched inits: -0.1111 and -0.1179, both resolved. The init itself is worth
-0.037..-0.047 under this additive write, all three seeds negative but not
resolved; the earlier n=1 reading of +0.012 was noise.

TWO CLAIMS THIS FILE MADE THAT DO NOT HOLD:
  * "additive - GDN = +0.1189, the gap, confirmed real" -- NOT resolved, p=0.088.
    An earlier dump_cdelta.py used |t|>2.5, a large-sample habit that is wrong at
    dof~3. THIS LAYER WAS NEVER RESOLVEDLY BEHIND GDN at n=3. It was suggestively
    behind. Everything else in this file stands, but that framing was too strong.
  * the theta init as a resolved effect -- p=0.053, suggestive only.

Not a win over GDN either, and not proven equal: -0.0388 is the better mean but
it does not resolve, and a difference that fails to resolve is absence of
evidence in both directions. Worth noting separately that the delta arm's seed sd
is 0.0147 against GDN's 0.0733, 5x tighter -- it is GDN's variance, not ours,
that prevents these comparisons from resolving.

The erase gate does not sit at its init: beta = sigmoid(<w,z>+b) starts at 0.120
with w = 0, and trains to b -> 0.11..0.27 with |w| = 0.80..1.75. So the erasure
became strongly DATA-DEPENDENT rather than a constant that could be absorbed
elsewhere. beta = 0 reproduces this layer bit-for-bit (checked, 1.1e-15), so the
model could have switched the mechanism off and did the opposite.

Position profile, delta(0.02) minus THIS layer, next to GDN minus THIS layer:
           delta      GDN                        delta      GDN
    0-127  -0.046   -0.123        512-639       -0.173   -0.099
  128-255  -0.094   -0.114        640-767       -0.207   -0.121
  256-383  -0.128   -0.094        768-895       -0.213   -0.140
  384-511  -0.155   -0.091        896-1023      -0.238   -0.165
The two mechanisms are COMPLEMENTARY, not two approximations of one thing. The
delta rule's gain grows monotonically with position and beats GDN by 0.073 in
the last bucket; GDN is ahead by 0.077 in the first. Consistent with erasure
freeing state capacity only where the state is saturated.

Speed is the cost: 68,400 vs 81,400 tok/s, 2521 s/epoch vs 2121 s, so 1.19x and
SCA2 no longer beats GDN's 2401 s. The gain survives the change of axis anyway
-- at an equal 2041 s the arms read 2.7979 / 2.8750 / 2.9114 (delta / GDN /
additive) -- because val is nearly flat in log-tokens here, dval/dln(tokens) =
-0.061, so 19% fewer tokens costs 0.011 against 0.158 gained. All of the 1.19x
is one term, the Gram matrix of the write codes; see arch_cdelta.CHeadDeltaWPos.

Position profile, EARLIER measurement at n=3 vs n=1 (superseded by the n=3 vs
n=3 table above, which is why its GDN column reads -0.123 rather than -0.111 at
0-127; the shape of the conclusion is unchanged):
    0-127   -0.111     384-511  -0.064     768-895   -0.117
  128-255   -0.085     512-639  -0.073     896-1023  -0.138
  256-383   -0.070     640-767  -0.098
GDN is ahead EVERYWHERE. The profile is U-shaped: worst at 896-1023 (-0.138),
second worst at 0-127 (-0.111), flattest in the middle (-0.064). A 128-token
bucket is coarse, so "0-127" is not a context-free measurement; resolving it
would need buckets at 0, 1-7, 8-31, ...

theta is TRAINABLE and does not stay at its zero init: dL/dtheta_m =
<dL/d(theta_m K_m), K_m> is nonzero at theta = 0 (measured 2.1e-4), so theta
bootstraps and K starts learning. Measured over 18 trained checkpoints, all
zero-initialised:
    |theta| mean 0.15 - 0.65,   |xi_m| = |theta_m K_m| mean 0.30 - 1.72, max 3.2
So the trained models DO use content addressing. Re-initialising theta at 0.02
rather than 0 changes the path, not the destination (n=1: val +0.012).
That last clause is now known to be too strong once the write is error
correcting: with the delta rule, init 0.02 beats init 0 by 0.040 at n=3, Welch
t=-2.80. Both arms still drift to large theta (|theta| = 1.8..11.4 measured),
so the init selects WHICH solution is found, not whether content addressing
exists. It is also the only knob here that has never been swept beyond {0, 0.02}.

Ablations, same protocol:
  * depth 2 -> 4 layers: 3.176 -> 2.919, the one large confirmed SCA2 lever.
  * Md 16 -> 4 (funding Mc): +0.100 in favour of small Md. Channel capacity in
    the ungated C head beats gate resolution in the D head.
  * dv: the C-head state is 2.Mc.dv and at equal params it PEAKS at dv=32
    (dv 64/32/16/8 -> state 16384/24192/16128/9056). dv=32 is an interior
    optimum: dv=16 has larger Mc (504) yet loses 0.10 to dv=32.
  * theta re-init 0 -> 0.02 (n=1): val +0.012, but the lag slope steepens -0.277
    -> -0.321, past GDN's -0.304. This changes the initial spectral scale, not
    whether content addressing exists (see above); it costs
    +0.033 at tokens 0-127 and gains -0.011 at 896-1023.
  * seed sd is 0.03-0.07, i.e. larger than most single-seed effects here.

Speed: 2161 s/epoch vs GDN's 2401 s, and decode is O(1) in context length.

================================================================================
THE QUESTION
================================================================================
GDN's state is a MATRIX h in R^{dk x dv} updated by an error-correcting write,
h <- h(diag decay) + beta (v - h^T k) k^T, i.e. it removes what is already
stored at key k. Both of SCA2's states are updated by a DIAGONAL map plus an
additive write; nothing is ever removed. Concretely:

  1. Both heads write with v (.) (a rank-one outer product against a phase or a
     gate) and read with a rank-one query. Is there a capacity theorem that
     separates "diagonal transition + additive write" from "diagonal transition +
     rank-one error-correcting write" at matched state size? The delta rule's
     non-diagonal transition (a - beta P) is exactly what SCA2 lacks.
     Note our keys have CONSTANT amplitude and full support, u_m = e^{i phi_m} /
     sqrt(M), which forces ||Au||^2 = ||Av||^2 = (1/M) sum_m |a_m|^2 for any
     diagonal A: a diagonal gate can reshape read-out interference but cannot
     attenuate one key's energy more than another's. Would selective-SUPPORT
     codes recover selective erasure inside the diagonal class?

     ANSWERED, EMPIRICALLY, AND IT WAS THE RIGHT QUESTION. Measured effect at
     matched theta init: -0.1111 at init 0.02 and -0.1179 at init 0, both
     resolved (see FACTS for why not -0.158).
     The delta rule can be written in the complex domain at no change to the read
     and no change to the state carry. Because the codes have constant amplitude,
     ||phi_t||^2 = M exactly, so the value stored at a key is recovered without
     any normalisation, vhat = Re(phi^H S)/M, and the write
         S_t = S_{t-1} + phi_t (v_t - beta_t.vhat_t)^H
     stacks over a chunk into a UNIT lower triangular system,
         (I + diag(beta).tril(G,-1)) E = V - beta.*R,   G[t,s] = Re(phi_t^H phi_s)/M
     after which S_T = S_0 + sum_t phi_t e_t^H, so the whole mechanism is "solve
     for E, then run the additive layer with v -> E". beta on the erase term only
     (not on the whole write, as DeltaNet does) is what nests the additive
     baseline at beta = 0.
     The constant amplitude also makes this EXACTLY a real unit-norm delta rule:
     with u_t = [cos pw_t ; sin pw_t]/sqrt(M) and H_t = [S^R_t ; S^I_t]/sqrt(M),
         ||u_t|| = 1  and  H_t = (I - beta_t u_t u_t^T) H_{t-1} + u_t v_t^T
     identically (checked to 8.9e-16). So the phase is a PARAMETERISATION of
     normalised keys, not merely a read-out interference device, and the gap to
     GDN is now exact: same algebraic form, different key manifold. GDN's key
     ranges over the unit sphere of R^{dk}; ours is confined to the Clifford torus
     in R^{2M}, on which every coordinate pair carries identical energy 1/M.
     That is the residue of this question, and selective-SUPPORT codes -- the
     original sub-question -- remain the way to test it.
  2. ONE scalar kernel is shared by every value coordinate. At theta = 0, writing
     the pre-RMS linear path through an output projection [P_R, P_I] gives, at
     lag l, H_l = Re(f_l) P_R W_V + Im(f_l) P_I W_V, hence
         dim span{H_l}_l <= 2.
     378 modes and a 24,192-real state, yet the linear branch offers only TWO
     shared temporal profiles. RMS, the D head and depth make the full network
     more expressive, so this is not a bound on the model -- but is it the reason
     raising Mc sharpens the lag kernel while plateauing on loss? Does replacing
     w_m by w_{m,g}, one spectral mixture per value group, lift it? (Cost
     2.Mc.(G-1) params, state unchanged.)

     ANSWERED: NO, and the rank-2 limit is not the binding constraint. w_{m,g}
     with G=2 was run at matched params (ff 364 -> 361 to pay for it), one full
     epoch: val 2.9710 (n=1) against 2.9058 +- 0.0356, i.e. 0.065 WORSE, about
     1.8 seed sd, and worse at EVERY position bucket (+0.045 to +0.085), at
     0.84x the speed. The bound is real and lifting it does not help, so what
     limited this layer was the write, not the number of temporal profiles.
     Method note worth keeping: this arm looked 0.065 BETTER at 15.3M tokens and
     its lag slope was stably better against the control at every probe eval,
     while the end-of-epoch profile was uniformly worse. Mid-descent val and the
     lag slope both pointed the wrong way on the same arm.
  3. Position and content are ADDED in one phase, pw = <theta K, h> + t omega.
     Since kappa(delta, l) = (1/M) sum_m w_m e^{i xi_m . delta} e^{i omega_m l}
     is just a Fourier representation on the joint (delta, l) space, the two
     scales are separately tunable and there is no forced resolution trade-off.
     The exact constraint is a rank one: rank[kappa(delta_i, l_j)] <= M. Is there
     a sharper statement about what a SINGLE shared phase channel can represent,
     versus paired phases (+xi, omega) and (-xi, omega) with tied weights, which
     give cos(xi.delta) e^{i omega l} -- content modulating the amplitude of the
     positional term instead of shifting its phase?
  4. The write address in D is content-independent: every mode receives the same
     v_t, and only future gates can differentiate the traces. (The narrower worry
     that the imaginary state gets no injection is void -- the real system has
     [B, AB] = [[1, a_r], [0, a_i]], rank 2 whenever a_i != 0, so a real input
     does reach both real dimensions.) What is gained by a content-dependent
     complex injection b_{t,m,g} v_t, which keeps the recurrence diagonal?
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(d=128, Mc=378, Md=4, G=8, dv=32, ff=364, layers=4,
           theta_scale=0.0, max_len=1024)


def rms(u, eps=1e-6):
    return u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)


class CHead(nn.Module):
    """Cumulative complex state, phase-indexed, no forgetting."""

    def __init__(self, d, M, dv, max_len, theta_scale=0.0):
        super().__init__()
        self.M, self.dv = M, dv
        self.K = nn.Linear(d, M, False)
        self.V = nn.Linear(d, dv, False)
        self.theta = nn.Parameter(torch.zeros(M) if theta_scale == 0.0
                                  else theta_scale * torch.randn(M))
        k = torch.arange(M, dtype=torch.float32)
        self.register_buffer("omega", math.pi * (10000.0 ** (-k / max(M - 1, 1))))
        self.wr = nn.Parameter(torch.ones(M))
        self.wi = nn.Parameter(torch.zeros(M))

    def forward(self, z, h):
        B, T, _ = z.shape
        p = torch.arange(T, device=z.device, dtype=z.dtype)[:, None]
        pw = self.K(h) * self.theta + p * self.omega            # (B,T,M)
        pq = self.K(z) * self.theta + p * self.omega
        v = self.V(z)[:, :, None]                               # (B,T,1,dv)
        sr = (v * pw.cos()[..., None]).cumsum(1)                # (B,T,M,dv)
        si = (v * pw.sin()[..., None]).cumsum(1)
        qr, qi = pq.cos()[..., None], -pq.sin()[..., None]
        rr, ii = sr * qr - si * qi, sr * qi + si * qr
        wr, wi = self.wr[:, None], self.wi[:, None]
        return rms(torch.cat([(rr * wr - ii * wi).mean(2),
                              (rr * wi + ii * wr).mean(2)], -1))


class DHead(nn.Module):
    """Complex diagonal gate, additive write, rank-one query."""

    def __init__(self, d, M, G, dv):
        super().__init__()
        assert dv % G == 0
        self.M, self.G, self.dv, self.gs = M, G, dv, dv // G
        self.V = nn.Linear(d, dv, False)
        self.gr = nn.Linear(d, M * G)                 # log-magnitude, via softplus
        self.gi = nn.Linear(d, M * G)                 # phase
        self.qa_r = nn.Linear(d, M, False)            # alpha, over channels
        self.qa_i = nn.Linear(d, M, False)
        self.qb_r = nn.Linear(d, dv, False)           # beta, over coordinates
        self.qb_i = nn.Linear(d, dv, False)

    def forward(self, z, h):
        B, T, _ = z.shape
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        v = self.V(z)
        mag = torch.exp(-F.softplus(self.gr(h))).view(B, T, M, G)
        ph = self.gi(h).view(B, T, M, G)
        ar, ai = mag * ph.cos(), mag * ph.sin()
        qar, qai = self.qa_r(z), self.qa_i(z)
        qbr, qbi = self.qb_r(z), self.qb_i(z)
        qar, qai = (lambda n: (qar * n, qai * n))(
            torch.rsqrt(qar.square() + qai.square() + 1e-6))
        qbr, qbi = (lambda n: (qbr * n, qbi * n))(
            torch.rsqrt(qbr.square() + qbi.square() + 1e-6))
        sr = z.new_zeros(B, M, dv)
        si = z.new_zeros(B, M, dv)
        out = []
        for t in range(T):
            a_r, a_i = ar[:, t].unsqueeze(-1), ai[:, t].unsqueeze(-1)
            rg, ig = sr.view(B, M, G, gs), si.view(B, M, G, gs)
            sr = (a_r * rg - a_i * ig).reshape(B, M, dv) + v[:, t, None, :]
            si = (a_r * ig + a_i * rg).reshape(B, M, dv)
            A = (sr * qar[:, t, :, None] + si * qai[:, t, :, None]).mean(1)
            Bv = (si * qar[:, t, :, None] - sr * qai[:, t, :, None]).mean(1)
            out.append(torch.cat([qbr[:, t] * A + qbi[:, t] * Bv,
                                  qbr[:, t] * Bv - qbi[:, t] * A], -1))
        return rms(torch.stack(out, 1))


class BestLayer(nn.Module):
    def __init__(self, d=128, Mc=378, Md=4, G=8, dv=32, ff=364,
                 max_len=1024, theta_scale=0.0):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.c = CHead(d, Mc, dv, max_len, theta_scale)
        self.dh = DHead(d, Md, G, dv)
        self.mix = nn.Linear(4 * dv, d)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        z = self.n(x)
        h = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], 1)
        x = x + self.mix(torch.cat([self.c(z, h), self.dh(z, h)], -1))
        return x + self.ff(self.fn(x))


def _verify():
    """Check this file against the repo path that produced the FACTS above."""
    from sca2.ref import LayerCfg
    from sca2.registry import build

    # theta must be checked NON-ZERO too: at theta=0 the content phase term is
    # multiplied by zero, so a discrepancy there would go unnoticed.
    for ts in (0.0, 0.02):
        cfg = LayerCfg(d=128, Mc=378, Md=4, G=8, ff=364, freq="rope",
                       theta_scale=ts, max_len=1024, dv=32)
        repo = build("v3polarflat", cfg, dtype=torch.float64)
        mine = BestLayer(theta_scale=ts).double()
        keys = mine.load_state_dict(repo.state_dict(), strict=False)
        assert not keys.unexpected_keys and not keys.missing_keys, keys
        torch.manual_seed(1)
        x = torch.randn(2, 96, 128, dtype=torch.float64)
        with torch.no_grad():
            a, b = repo(x), mine(x)
        theta = dict(repo.named_parameters())["c.theta"]
        print(f"theta_scale={ts:<5} |theta|={theta.abs().mean():.5f}  "
              f"params={sum(p.numel() for p in mine.parameters())}  "
              f"max|diff|={(a - b).abs().max().item():.3e}")
        assert torch.allclose(a, b, atol=1e-9), "SEMANTICS DIFFER"
    print("OK: identical function to sca2 v3polarflat")


if __name__ == "__main__":
    _verify()
