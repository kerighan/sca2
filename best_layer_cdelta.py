r"""CURRENT CHAMPION: the SCA2 layer with an error-correcting C write.

Generation 2 of the lineage in WINNERS.md. Generation 1 is `best_layer.py`
(additive write), kept unchanged because it is the control this is measured
against -- do not edit it to match this file.

This is the exact function of the `cdelta` variant at the shape that measured
best. The repo's production path computes the same thing with a chunked closed
form (one triangular solve per chunk) instead of the token loop below;
`python best_layer_cdelta.py` checks this file against that path in float64.

Only the C head's WRITE differs from generation 1. The read, the D head, the
output projection and the FFN are byte-identical, so their notation is not
repeated here -- see best_layer.py. What follows is the one change.

================================================================================
THE CHANGE
================================================================================
Generation 1 wrote additively into a phase-indexed complex state, and never
removed anything:

    S_t = S_{t-1} + phi_t v_t^H,        phi_{t,m} = e^{i pw_{t,m}}

Generation 2 first READS what is already stored at the write code, and writes the
error instead of the value:

    vhat_t^H = Re(phi_t^H S_{t-1}) / M                      (no normalisation!)
    S_t      = S_{t-1} + phi_t (v_t - beta_t vhat_t)^H
    beta_t   = sigmoid(<w, z_t> + b)                        (scalar, per token)

The missing normalisation is the point. A delta rule needs the value stored at
the key, which in general is (k^H S)/||k||^2, and ||k||^2 varies per token. Here
the codes have CONSTANT amplitude, |phi_{t,m}| = 1 for every m, so

    ||phi_t||^2 = M   exactly, for every t and every input,

and the read-back is exact with a constant denominator. The property that made
selective erasure look impossible inside the diagonal class -- constant
amplitude, so a diagonal gate cannot attenuate one key more than another -- is
the same property that makes the delta rule free of a per-token normalisation.

IT IS EXACTLY A REAL RANK-ONE DELTA RULE, ON A CONSTRAINED KEY MANIFOLD. Since
only the real part is read back, set

    u_t = [cos pw_t ; sin pw_t] / sqrt(M)     in R^{2M}
    H_t = [S^R_t ; S^I_t] / sqrt(M)           in R^{2M x dv}

Then ||u_t|| = 1 identically -- that IS the constant-amplitude property -- and
the update above is, with no approximation,

    H_t = (I - beta_t u_t u_t^T) H_{t-1} + u_t v_t^T

which is the textbook delta rule with a unit-norm key. So the complex phase is
not an interference trick here, it is a PARAMETERISATION of normalised keys, and
the difference from GDN is stated precisely: same algebraic form, different key
manifold. GDN's key ranges over the whole unit sphere of R^{dk}; ours is confined
to the Clifford torus {[cos p ; sin p]/sqrt(M)}, a measure-zero submanifold on
which every coordinate PAIR carries identical energy 1/M. That, not the delta
rule, is what open question 1 is about.

WHY beta MULTIPLIES ONLY THE ERASE TERM, AND WHAT IT COSTS. DeltaNet writes
S += beta.phi(v - vhat)^H, gating the whole write. Gating only the erase means
beta = 0 recovers generation 1 EXACTLY (verified at 0.0e+00, bit for bit), so
generation 1 is nested inside generation 2 and the optimiser can switch the
mechanism off. It does the opposite: beta trains from its 0.12 init to bias
0.11..0.27 with weight norm 0.80..1.75, i.e. the erasure becomes strongly
data-dependent rather than a constant that could be absorbed into wr/wi.

The price is that this is NOT exact value replacement. Reading back immediately
after writing gives u_t^T H_t = (1 - beta_t) u_t^T H_{t-1} + v_t^T, so for a key
and value repeated with constant beta the stored value converges to v/beta, not
to v. The layer learns a correction AND an accumulation level (gain 1/beta, so
roughly 4..9 at the trained biases; the following RMS removes the global scale
but not the differences between associations).

This is FORCED, not a stylistic choice. For a write S += a(beta).u(v - b(beta).vhat)^T
the fixed point is v/b(beta), independent of a. Nesting generation 1 at beta = 0
requires a(0) = 1 and b(0) = 0; exact replacement requires b == 1. With b
continuous the two are incompatible, and DeltaNet's choice (a = beta, b = 1)
buys replacement by giving up the write entirely at beta = 0, i.e. by giving up
the nested baseline. Keeping the nesting was a measurement decision -- it is what
makes the difference attributable -- and the 1/beta gain is its consequence.
Do not "fix" the asymmetry without re-measuring: it is part of what worked.

HOW THE REPO REMOVES THE LOOP. Stacking the recurrence over a chunk, with
G[t,s] = Re(phi_t^H phi_s)/M and R the read-back from the incoming state,

    (I + diag(beta) tril(G,-1)) E = V - beta .* R

which is UNIT lower triangular, hence one triangular solve, no inverse and no
iteration. Then S_T = S_0 + sum_t phi_t e_t^H, so the state carry and the read
are generation 1's with v -> E. That is the whole implementation.

================================================================================
FACTS (pycode, 1024-token blocks, 4 layers, one 177M-token epoch, val loss in
nats, arms interpolated to a common 168.7M tokens; params matched to 0.07%)
================================================================================
SHAPE. This file's defaults are Mc=190, dv=56 -- NOT the Mc=378, dv=32 of
generations 1-2. The code is byte-identical to generation 2; only the shape moved,
and it moved the result more than most mechanisms did (see THE SHAPE below).

    THIS layer (dv=56, Mc=190, theta init 0.02)  2.7078 +- 0.0132  (n=3)
    same code at dv=48, Mc=253                   2.7226 +- 0.0431  (n=3)
    Gated DeltaNet, matched params               2.7931 +- 0.0536  (n=6)
    same code at dv=32, Mc=378 (generation 2)    2.7482 +- 0.0147  (n=3)
    delta write, theta init 0, dv=32             2.7879 +- 0.0198  (n=3)
    additive write, theta init 0.02, dv=32       2.8591 +- 0.0233  (n=3)
    generation 1 (additive write, theta init 0)  2.9057 +- 0.0356  (n=3)

"resolved" means |t| exceeds the actual two-sided 95% t critical value at that
dof, which for n=3 is 2.9-4.3, NOT 2. GDN is at n=6 because its seed sd is 4x
this arm's and it supplied ~90% of the variance of every comparison against it.

    THIS - GDN              -0.0854   t=-3.69  dof=6.1  p=0.0101   YES *
    THIS - generation 2     -0.0403   t=-3.53  dof=4.0  p=0.024    YES
    THIS - dv=48            -0.0148   t=-0.57  dof=2.4  p=0.62     no
    dv=48 - GDN             -0.0706   t=-2.13  dof=5.1  p=0.086    no
    gen 2 - GDN             -0.0449   t=-1.92  dof=6.3  p=0.102    no
    gen 2 - additive(0.02)  -0.1111                    p=0.004    YES <- mechanism
    delta(0) - gen 1        -0.1179                    p=0.008    YES <- mechanism
    gen 1 - GDN             +0.1126   t=+3.75  dof=6.0  p=0.0095   YES
    gen 2 - delta(0)        -0.0398                    p=0.053    no
    additive(0.02) - gen 1  -0.0467                    p=0.140     no

* READ THE ASTERISK BEFORE QUOTING THE FIRST ROW. p=0.0101 is uncorrected, and
this shape was picked as the best of FIVE screened (dv=24/40/48/56/64). Bonferroni
over five gives alpha=0.01, whose critical t at dof=6.06 is 3.683; the observed t
is 3.685. It clears multiplicity by 0.002, which is to say it sits exactly on the
boundary and nothing should be leaned on it. The claim is "SCA2 is ahead of GDN,
at the edge of significance after accounting for the shape search" -- not "SCA2
beats GDN", and certainly not by the margin the raw p suggests.

More seeds of THIS arm would barely help: its sd is 0.0132 against GDN's 0.0536,
so GDN owns the variance. n=3 -> n=6 on this arm moves t only from 3.69 to 3.79.
GDN at n=12 would take it to ~4.96. That is the experiment that would settle it,
and it is a GDN experiment.

WHAT IS ESTABLISHED: the mechanism (-0.111 at matched theta init 0.02, -0.118 at
matched init 0, the most strongly resolved comparisons here), and the shape
(-0.040 against generation 2). The win over GDN is marginal, see the asterisk.

THE PAIRED ESTIMATOR IS RETIRED, and this file previously depended on it. It
averaged 6 points over the last 30M tokens and paired by seed; its between-seed sd
looked 8x tighter than the endpoint's on the dv=48 arm, so it was used to re-read
n=1 screens. It does not replicate: on dv=56 its residuals spread 0.032, and
against GDN it predicted -0.083 where n=6 measured -0.038. Two reasons, both now
understood: pairing SCA2 to GDN by seed number is meaningless (different parameter
shapes, no shared nuisance term to cancel), and its apparent precision on dv=48
was three seeds happening to agree. Every n=1 paired number this campaign produced
is uninformative, including a shape-curve reading that was quoted for three hours.
Endpoint interpolation only, from here.

WHAT THIS FILE PREVIOUSLY ASSERTED AND GOT WRONG:
  * the retraction of "gen 1 - GDN is real" is itself now retracted. At GDN n=3 it
    was p=0.088, correctly called unresolved. At n=6 it is p=0.0095: generation 1
    WAS resolvedly behind GDN. The n=3 call was right on its data and wrong about
    the world -- the fix was more GDN seeds, not a softer claim.
  * "THIS - GDN = -0.0388, no (p=0.456)" was computed against GDN at n=3 and with
    a dof error; at n=6 the same comparison is -0.0449, p=0.102.
  * the theta init as a "resolved effect". -0.0398 at p=0.053, suggestive only.

THE SHAPE, which is the whole of generation 3. At matched params Mc trades against
dv linearly (Mc costs 131 params/mode, dv costs 1024/unit across C.V, D.V, D.qb_r,
D.qb_i and mix), so dv=56 buys 24 units of value width by giving up 188 modes. It
wins -0.040 while running FEWER modes, which is the opposite of what the additive
write wanted: under the additive write more modes kept distinct writes from
colliding, and the delta rule corrects that interference instead, so the marginal
value of a mode fell. Note also 2*Mc*dv (the C state) is 21280 here against 24192
at dv=32 -- a SMALLER state scoring better, twice over, since dv=64 holds 16384
and loses. The effect is allocation, not capacity.

ATTRIBUTION. pretrain.py's --theta-scale defaults to 0.0 and generation 1's runs
never override it, so the -0.1577 spans the delta rule AND the init, and was
wrongly quoted as the mechanism's effect here at first. theta_ctrl.sh added the
missing cell (additive write at init 0.02, n=3), which is what makes the -0.1111
row possible. Its own row says the init is worth -0.037..-0.047 under the
additive write, all three seeds negative but not resolved -- consistent with the
two effects being roughly ADDITIVE (-0.118 mechanism plus -0.04 init gives the
-0.158), and inconsistent with the earlier n=1 reading of +0.012, which was
noise.

THROUGHPUT: x1.08 OVER GDN at matched params, measured by sweep_chunk.py in a
blocked design, x1.082 and x1.086 in two independent sessions (185714 params/layer
for GDN against our 185959, so 0.13% in its favour). The 1.19x COST quoted for
generation 2 is withdrawn, not carried over: this shape is faster than GDN, not
slower, which reverses the sign of the speed/quality trade the campaign believed
it was making.

The blocking is not optional. Timed sequentially, the SAME config gave 71.6k /
65.1k / 76.7k tok/s in one session -- 15.2% apart -- while three consecutive
windows of one config agree to 2.2%. Note 71.6 -> 65.1 -> 76.7 is NOT monotone, so
this is not thermal decay and a cold GPU would not fix it; something exogenous
moves the clock on a minute scale. That also means this arm's own 77405 / 76325 /
70702 across seeds was never evidence of the machine heating up, as was claimed
here. Blocking cut the residual noise to 1.4-3.0% and the GDN ratio then
reproduced across sessions to 0.004.

Position profile, val by 128-token bucket, n=3 for THIS and n=6 for GDN:
              THIS     GDN    THIS-GDN                  THIS-GDN   THIS-gen1
    0-127   3.0243  3.0116     +0.013     512-639        -0.100     -0.197
  128-255   2.7947  2.8310     -0.036     640-767        -0.107     -0.226
  256-383   2.7631  2.8306     -0.068     768-895        -0.098     -0.237
  384-511   2.6504  2.7431     -0.093     896-1023       -0.099     -0.263

There is a CROSSOVER, and it is the sharpest structural fact here: this layer is
BEHIND GDN on the first 128 tokens, passes it around position 200, and settles at
-0.10 from position 400 on. Two consequences that pull in opposite directions.

  * The aggregate -0.085 UNDERSTATES the long-context gain. Restricted to
    512-1023, half the window, the gap is -0.1009 at p=0.0078 -- better resolved
    than the aggregate. The head-of-window penalty is +0.0128 at p=0.50, i.e. not
    measurable.
  * That 512 cut was chosen AFTER seeing the profile. The defence is that
    "SCA2 wins in long context" is this project's a-priori hypothesis and its copy
    -task result predicted it, so the direction was not fished for -- but the exact
    cut was, and a post-hoc split cannot be quoted as a headline. Pre-register it
    and re-measure if it is to be used.

Note the profile no longer grows monotonically the way generation 2's did: it is a
level shift that saturates by position 400-500, not a widening gap. Generation 2's
file predicted "the gain should keep growing past 1024"; at this shape it plateaus
within 1024, so that prediction is not supported here and the >1024 test is now
about whether the plateau holds, not whether the gain keeps climbing.

theta does NOT stay near its init in either arm: |theta| runs from 0 to 1.8..11.4
by layer, components up to 1.28 rad. So both arms end up content-addressed, and
the 0.020 between them is the init selecting WHICH solution is found. Do not read
the theta-init-0 arm as a positional-only control; it is not one.

WHERE THE COST IS. Per chunk of C, at this shape (M=190, dv=56):
    K2 kernel (shared with gen 1)   4.C^2.M
    Gram G            (added)       2.C^2.M     <- 50% of the kernel
    triangular solve  (added)       C^2.dv      <- 15% of the Gram at dv=56
The solve is not the cost, so optimising it returns nothing; the Gram is. Halving
Mc from 378 to 190 halved that added term outright, which is why this shape is
faster than generation 2 as well as better.

CHUNK SIZE IS CLOSED AS A LEVER, and the theory pointed the wrong way. Total cost
over T/C chunks is 4.B.T.C.M, LINEAR in C, so halving the chunk halves those FLOPs
and small C should win; halving Mc should additionally have pushed the optimum UP
from the 128 the campaign runs. Blocked sweep, ratios to 128:
    ctx     48     64     96    128    192    256    384
    ratio  0.754  0.886  0.951  1.000  1.016  0.983  0.963
Neither prediction held. The low end is decisively worse (ctx=48 costs 25%), the
high end mildly worse, and 128 sits where it already was; SCA2_D_CHUNK=16 likewise
beats 8 by 5-6% and 32 by 1-3%. Below ~128 this layer is launch- and
occupancy-bound rather than FLOP-bound, so a FLOP argument gets the SIGN wrong.
torch.compile is not a lever either: fullgraph=True already passes, backward
included, so there is no break to fix.

The other lever is CLOSED, and the file used to advertise it as open. Freezing the
write phase makes G a constant Toeplitz matrix computed once (CHeadDeltaWPos),
which did buy back three quarters of the penalty -- and cost 0.9 nats, scoring
3.6182 against 2.7482, with the only positive position slope in the campaign
(+0.19, every other arm improves with position). The read kernel depends on
pq_t - pw_s, so a content-free write phase leaves the query nothing to match
against and destroys associative retrieval across the whole head, not just the
erase. A constant Gram REQUIRES the change that kills the mechanism, so this is
not a tuning failure, it is a dead route.

================================================================================
WHAT IS STILL OPEN
================================================================================
1. The key is confined to the Clifford torus in R^{2M} (see THE CHANGE), whereas
   GDN's ranges over the whole unit sphere of R^{dk}. The delta rule makes
   erasure POSSIBLE; which associations are SEPARABLE is still restricted by that
   manifold. Selective-SUPPORT codes remain untested and are the deeper question
   that best_layer.py question 1 raised.
2. Whether the erase needs content addressing at all, or only lag addressing.
   Undecided: the theta-init-0 arm does not answer it because theta drifts.
3. The shape CURVE, the part the retrade did not settle. dv=48 and dv=56 are
   indistinguishable (-0.0148, p=0.62), and dv=40/64 have only n=1 readings whose
   two estimators disagreed on the SIGN. So the champion shape is a plateau, not a
   peak: do not quote the optimum more precisely than "dv in 48..56", and do not
   quote the n=1 arms at all. The reason this was worth sweeping is now measured
   rather than conjectured -- the additive write's preference for many modes was a
   way to keep writes from colliding, and correcting the collisions instead made
   modes cheaper to give up.
4. Md has never moved from 4 while Mc fell 378 -> 190, so the C/D width RATIO has
   silently gone from 94:1 to 48:1. Nothing here tested whether 4 is still right
   at the new shape.
5. theta_scale has never been swept beyond {0, 0.02}, and it is the only knob
   here that showed even a suggestive effect at matched everything else.
6. GDN at n=12, which is the only cheap way to turn the headline from "ahead at the
   edge of significance" into something that survives multiplicity. See the
   asterisk in FACTS: more seeds of THIS arm are nearly worthless by comparison.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Generation 3's shape. Generations 1-2 ran Mc=378, dv=32; the parameter count is
# matched to 0.03% (185959 vs 186011 per layer), because Mc costs 131 params/mode
# and dv costs 1024/unit, so 188 modes buy 24 units of value width.
CFG = dict(d=128, Mc=190, Md=4, G=8, dv=56, ff=364, layers=4,
           theta_scale=0.02, max_len=1024)
CFG_GEN2 = dict(CFG, Mc=378, dv=32)


def rms(u, eps=1e-6):
    return u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)


class CHeadDelta(nn.Module):
    """Phase-indexed complex state with an error-correcting write.

    Written as the honest token recurrence: unlike generation 1, the write
    depends on the state, so there is no cumsum form. The repo replaces this loop
    with one triangular solve per chunk and gets the same function to 2.7e-15.
    """

    def __init__(self, d, M, dv, max_len, theta_scale=0.0, cdelta_init=-2.0):
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
        # The only new parameters anywhere: 129 per layer (d+1), so 516 across
        # the 4 layers, 0.07% of the 744,044 layer budget.
        self.bproj = nn.Linear(d, 1, True)
        nn.init.zeros_(self.bproj.weight)
        nn.init.constant_(self.bproj.bias, cdelta_init)

    def forward(self, z, h):
        B, T, _ = z.shape
        M = self.M
        p = torch.arange(T, device=z.device, dtype=z.dtype)[:, None]
        pw = self.K(h) * self.theta + p * self.omega            # (B,T,M)
        pq = self.K(z) * self.theta + p * self.omega
        cw, sw = pw.cos(), pw.sin()
        cq, sq = pq.cos(), pq.sin()
        v = self.V(z)                                           # (B,T,dv)
        beta = torch.sigmoid(self.bproj(z))                     # (B,T,1)

        sr = z.new_zeros(B, M, self.dv)
        si = z.new_zeros(B, M, self.dv)
        wr, wi = self.wr[:, None], self.wi[:, None]
        out = []
        for t in range(T):
            cwt, swt = cw[:, t, :, None], sw[:, t, :, None]     # (B,M,1)
            # Read back what is stored at the write code. ||phi||^2 = M exactly,
            # which is why the denominator is a constant and not per-token.
            vhat = (sr * cwt + si * swt).sum(1) / M              # (B,dv)
            e = (v[:, t] - beta[:, t] * vhat)[:, None, :]        # (B,1,dv)
            sr = sr + e * cwt
            si = si + e * swt
            # --- read: identical to generation 1, with the corrected state ---
            qr, qi = cq[:, t, :, None], -sq[:, t, :, None]
            rr, ii = sr * qr - si * qi, sr * qi + si * qr
            out.append(torch.cat([(rr * wr - ii * wi).mean(1),
                                  (rr * wi + ii * wr).mean(1)], -1))
        return rms(torch.stack(out, 1))


class DHead(nn.Module):
    """Unchanged from generation 1: complex diagonal gate, additive write."""

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


class ChampionLayer(nn.Module):
    def __init__(self, d=128, Mc=190, Md=4, G=8, dv=56, ff=364,
                 max_len=1024, theta_scale=0.02):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.c = CHeadDelta(d, Mc, dv, max_len, theta_scale)
        self.dh = DHead(d, Md, G, dv)
        self.mix = nn.Linear(4 * dv, d)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        z = self.n(x)
        h = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], 1)
        x = x + self.mix(torch.cat([self.c(z, h), self.dh(z, h)], -1))
        return x + self.ff(self.fn(x))


def _verify_rank_one():
    """The docstring claims H_t = (I - beta u u^T) H_{t-1} + u v^T exactly.

    Asserted claims get checked here like any other. Both recurrences are written
    out independently -- the complex one in (S^R, S^I) as the layer runs it, the
    real one in R^{2M} -- and compared.
    """
    torch.manual_seed(0)
    B, T, M, dv = 2, 17, 11, 5
    pw = torch.randn(B, T, M, dtype=torch.float64) * 3
    v = torch.randn(B, T, dv, dtype=torch.float64)
    beta = torch.rand(B, T, 1, dtype=torch.float64)
    cw, sw = pw.cos(), pw.sin()

    sr = torch.zeros(B, M, dv, dtype=torch.float64)
    si = torch.zeros(B, M, dv, dtype=torch.float64)
    for t in range(T):
        cwt, swt = cw[:, t, :, None], sw[:, t, :, None]
        vhat = (sr * cwt + si * swt).sum(1) / M
        e = (v[:, t] - beta[:, t] * vhat)[:, None, :]
        sr, si = sr + e * cwt, si + e * swt

    u = torch.cat([cw, sw], -1) / math.sqrt(M)                  # (B,T,2M)
    H = torch.zeros(B, 2 * M, dv, dtype=torch.float64)
    for t in range(T):
        ut = u[:, t, :, None]                                   # (B,2M,1)
        H = H - beta[:, t, :, None] * ut * (ut.transpose(1, 2) @ H) + ut * v[:, t, None, :]

    ref = torch.cat([sr, si], 1) / math.sqrt(M)
    err = (H - ref).abs().max().item()
    nrm = (u.norm(dim=-1) - 1).abs().max().item()
    print(f"real rank-one form           max|diff|={err:.3e}  ||u||-1={nrm:.1e}")
    assert err < 1e-12 and nrm < 1e-12, (err, nrm)


def _verify():
    """Check this file against the repo path that produced the FACTS above."""
    from sca2.ref import LayerCfg
    from sca2.registry import build

    # Both shapes: the champion, and generation 2's, since the FACTS table quotes
    # measurements at each and a shape-specific bug would otherwise hide.
    for ts, shape in ((0.0, CFG), (0.02, CFG), (0.02, CFG_GEN2)):
        cfg = LayerCfg(d=128, Mc=shape["Mc"], Md=4, G=8, ff=364, freq="rope",
                       theta_scale=ts, max_len=1024, dv=shape["dv"])
        repo = build("cdelta", cfg, dtype=torch.float64)
        mine = ChampionLayer(Mc=shape["Mc"], dv=shape["dv"],
                             theta_scale=ts).double()
        keys = mine.load_state_dict(repo.state_dict(), strict=False)
        assert not keys.unexpected_keys and not keys.missing_keys, keys
        torch.manual_seed(1)
        x = torch.randn(2, 96, 128, dtype=torch.float64)
        with torch.no_grad():
            a, b = repo(x), mine(x)
        print(f"Mc={shape['Mc']:<4} dv={shape['dv']:<3} theta_scale={ts:<5} "
              f"params={sum(p.numel() for p in mine.parameters())}"
              f"  max|diff|={(a - b).abs().max().item():.3e}")
        assert torch.allclose(a, b, atol=1e-9), "SEMANTICS DIFFER"

    # beta -> 0 must reproduce generation 1 exactly, or the nesting claim in the
    # docstring is false and every comparison against it is confounded.
    # At generation 1's OWN shape: it is the control the -0.118 is measured
    # against, and nesting is a statement about the function, not the width.
    import best_layer
    mine = ChampionLayer(Mc=378, dv=32, theta_scale=0.02).double()
    nn.init.constant_(mine.c.bproj.bias, -60.0)
    gen1 = best_layer.BestLayer(Mc=378, dv=32, theta_scale=0.02).double()
    gen1.load_state_dict({k: v for k, v in mine.state_dict().items()
                          if not k.startswith("c.bproj")}, strict=False)
    torch.manual_seed(1)
    x = torch.randn(2, 96, 128, dtype=torch.float64)
    with torch.no_grad():
        d0 = (gen1(x) - mine(x)).abs().max().item()
    print(f"beta=0 vs generation 1        max|diff|={d0:.3e}")
    assert d0 < 1e-9, "generation 1 is NOT nested -- comparisons are confounded"
    print("OK: identical function to sca2 cdelta, and nests best_layer.py")


if __name__ == "__main__":
    _verify_rank_one()
    _verify()
