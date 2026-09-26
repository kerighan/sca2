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
    lam_free: bool = False  # FREE MODES. lambda_m = exp(a_m), no pin, no cap other than the fp32
    #   safety ceiling `lam_ceil`. Motivation, measured on the trained d=1024 checkpoint: with
    #   softplus(a).clamp(max=lam_max), 41-100% of the FREE modes of every layer sit exactly AT
    #   the clamp (layer 5: 128/128), where the gradient is exactly zero -- a mode that reaches
    #   it can never come back. Combined with `persist` pinning the other half at exactly 0, the
    #   realised spectrum is a TWO-POINT set: 86-100% of the |w| mass sits on one endpoint or the
    #   other, the |w|-weighted median memory is 1/lam_max (64.0 tokens) in 7 layers out of 8, and
    #   the 8 layers' 16 temporal profiles span an effective rank of 2.82 (97.7% of the singular
    #   mass on 2). GDN, measured the same way, spans 2.5 to 5.8e6 tokens with a monotone depth
    #   gradient (median memory 12.9 -> 7983, x619). The link function is NOT the problem: for
    #   lambda <~ 1/64, softplus(a) ~ exp(a), so |d log tau / da| = 0.94..1.00 across the range --
    #   already scale-free, already the log-rate parameterisation GDN (exp(A_log)), Mamba
    #   (-exp(A_log)) and RWKV (exp(-exp(w))) use. The BOUND is the problem. So: keep the
    #   exponential, drop the pin, and move the ceiling out to where fp32 actually needs it.
    lam_ceil: Optional[float] = None  # fp32 safety ceiling for lam_free. None -> 55/chunk (0.43 at
    #   chunk 128, i.e. a memory floor of 2.3 tokens -- just past GDN's fastest measured head at
    #   2.5). e^{lambda*chunk} is formed in the closed form, and fp32 overflows past ~88.
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
        "batched"  # "chunk" | "triton" (inverse) | "triton_codes" (codes/products)
        # | "triton_fused" (codes/products + inverse + state loop)
    )
    gemm_dtype: Optional[torch.dtype] = (
        None  # dtype of the long-head GEMM OPERANDS (codes, and the state as it is read
        #       and written); None = autocast if active, else fp32. The state itself, the
        #       phases, the Gram's solve and the short head stay fp32 regardless.
    )
    beta_init: float = -2.0  # erase gate bias: sigmoid(-2) = 0.12 at init
    beta_write: bool = False  # also gate new long-head values: e = W @ (beta*v - beta*r).
    # No extra parameters. Requires scalar beta (beta_groups=1); short head unchanged.
    kv_dk: int = 0  # KEY VERIFICATION (0 = off). The long head stores a copy of its own write
    #   key beside the value, reads it back, and gates the output on whether it matches the
    #   query's key: e_s = [V(z_s) ; Kv(h_s)], m_t = cos(Re key part, Kv(z_t)),
    #   g_t = sigmoid(ga.m_t + gb), out = RMS(value part).g_t. A genuine match wrote its key
    #   with h_s ~ z_t so the key read back agrees; a random mixture does not. It exists
    #   because neither the query nor the read's MAGNITUDE distinguishes "found" from "not
    #   found" -- measured, the read norms on new and repeated words are the same
    #   distribution -- so the gate needs evidence the read itself carries. Port of sca2's
    #   CHeadDeltaKV, which was the best arm at d=128. Costs 2.d.dk + 2 parameters.
    layer_scale: bool = False  # a learned gain on each RESIDUAL BRANCH (LayerScale).
    ls_mix_init: float = 1.0   # init value for gs_mix. Two trained checkpoints (lsfree, g2)
    #   converge to a median of ~0.26; starting at 0.25 saves 2h of convergence.
    ls_ff_init: float = 1.0    # init value for gs_ff. Converges to ~0.6 in both runs.
    lam_anchor: float = 0.0    # fraction of modes whose decay is PINNED at its
    # geometric init and receives no gradient. Trained checkpoints show the
    # timescales collapsing: initialised log-uniform over [4, 20000] tokens, a
    # span of 5000x, layers 0, 1, 2 and 6 end with a span of 12x to 52x and a
    # median memory of 7 to 14 tokens. Half the stack forgets everything past a
    # handful of tokens, which leaves ~200 modes redundant with each other and
    # is why 32 mixture kernels span an effective rank of only 4.12. Anchoring
    # guarantees the coverage the initialisation intended, at zero parameters
    # and zero compute -- it is a gradient mask.
    learn_omega: bool = False  # make the rope frequency grid a parameter. It is
    # a buffer today, so when the decays collapse the only remaining source of
    # temporal diversity is a grid nobody optimises. The Triton kernel already
    # computes omega's gradient and returns it (grad[2] in _Codes.backward); it
    # was simply being discarded.
    read_mix: int = 1          # R read weight vectors instead of one, combined per
    # token by alpha(z) = softmax(A z). The read is LINEAR in w, so this equals
    # reading once with w_eff(z_t) = sum_r alpha_r(z_t) w^(r): the kernel becomes
    # sum_r alpha_r(z_t) kappa_r(t,s), still damped exponentials on the same
    # modes, with the state, the write and the transform untouched. What stops
    # being a training-time constant is the SHAPE of the kernel over lag -- the
    # 16 profiles of a trained 8-layer stack span a rank of 2.82. Unlike
    # long_groups, which partitions dv and trades width for diversity, every
    # kernel here reads the full width. See chead_numpy.py.
    decay_softplus: bool = False   # with decay_input: modulate the rate by
    # softplus(lz + b0) instead of exp(lz). See LongHead.lam_t.
    post_norm: bool = False    # RMSNorm on the MIXER OUTPUT before the residual.
    # The inputs of `mix` are normalised (o_norm on the long head, _rms on the
    # short one); its output is a bare Linear, and a single learned scalar has
    # to absorb whatever scale it lands on. Trained checkpoints show the
    # optimiser spending that scalar on damping: gs_mix falls to 0.048, BELOW
    # its 0.1 init, in the middle layers, while gs_ff climbs to 0.445 -- the one
    # component that moves information between positions contributes a twentieth
    # of the residual. This gives the branch a per-channel scale of its own, at
    # d parameters a layer (8192 for the stack, 0.007%), and keeps gs_mix for
    # the overall magnitude. The pattern is Gemma 2's and NormFormer's; neither
    # this layer nor the GDN baseline has it.
    ls_mix_per_channel: bool = False  # gs_mix shape (d,) instead of (): lets each branch
    #   choose WHERE in the residual stream it writes, not just how much.
    #   Measured on the trained 10-layer model: the residual stream grows 29x from layer 0 to
    #   layer 9 (||x|| 25.8 -> 759.4) while each branch emits a roughly CONSTANT norm
    #   (130-216), because the LayerNorm at the head of the branch erases the stream's scale.
    #   So the relative contribution collapses -- layer 0 moves the stream by 785%, layer 8 by
    #   19% -- and the per-layer ablation shows the same collapse in loss terms (+1.59 vs
    #   +0.013). The gradient does NOT vanish: it decays only 3x across the stack, which AdamW
    #   largely absorbs. In principle mix could grow its own weights to compensate and this is
    #   redundant; in practice giving the scale its own parameter and its own gradient, rather
    #   than leaving it entangled in a 4dv x d matrix, is what LayerScale does and it is known
    #   to help deep stacks. 2 parameters per layer. If it changes nothing, the deep layers
    #   genuinely have nothing to say.
    decay_input: bool = False  # DATA-DEPENDENT forgetting. lam becomes lam_max*sigmoid(a_m +
    #   Wd(z_t)_m), a function of the token, instead of a constant per mode. GDN's decay is
    #   g_t = -exp(A_log)*softplus(a(x_t)+dt_bias) and ablates at +1.08 nats there; ours is
    #   frozen after training. The chunked closed form survives: with C_t = sum_{u<=t} lam_u,
    #   decay(t,s) = exp(-(C_t - C_s)), so the ramps become a CUMSUM along the chunk instead
    #   of lam*idx and the factorisation gw_s = exp(C_s), gq_t = exp(-C_t) is unchanged.
    #   Costs d*M parameters and one (B,T,M) tensor. Wd starts at zero, so at init this is
    #   exactly the learn_persist parameterisation (smooth, no clamp, no dead zone).
    conv_silu: bool = False  # SiLU after the causal conv, as GDN does on its q/k/v convs
    #   (F.silu(F.conv1d(...))). Ours was a purely LINEAR convolution, so the whole path from
    #   z to the residual was linear apart from the FFN. Note this breaks identity-at-init:
    #   at init the conv is the identity, so the layer starts from silu(z) rather than z.
    long_groups: int = 1  # same idea on the LONG head: its wr/wi are (M,), one temporal
    #   profile shared by every value channel. G gives it G profiles, group g reading value
    #   channels [g.dvi/G, (g+1).dvi/G). This IS what wg2 tested and lost (+0.065 at d=128,
    #   G=2, generation 2: "the rank-2 bound is real but is NOT the binding constraint"), so
    #   the prior is against it; retried only because the bound scales with dv (one filter
    #   for dv=56 there, 256 here) and the base has changed completely. With key
    #   verification on, the dk-wide key copy lands in the LAST group, so the gate's
    #   evidence is read through that group's kernel.
    w_antipodal: float = 0.0  # symmetry-breaking noise on wr/wi when long_groups > 1.
    #   w0 = 1 + eps*n, w1 = 1 - eps*n (n ~ N(0,1)): the MEAN of the two columns is
    #   exactly the current w, so the layer's average behaviour is unchanged at init, but
    #   the gradient no longer starts from a symmetric fixed point. Measured: the gradient
    #   at the symmetric point already has cos(g0,g1) = -0.71 (anti-correlated), so the
    #   symmetry breaks by real signal, not floating-point noise. But it takes ~4h to reach
    #   cos(kappa0,kappa1) = 0.86; eps = 0.10 starts there instantly.
    short_groups: int = 1  # spectral read weights of the SHORT head, per group of value
    #   channels. 1 = one (L,) filter shared by all dv channels, which is what the layer has
    #   always had. The taps ARE the DFT of the spectral weights (w_m = sum_n taps_n
    #   e^{i n omega_m}), so the short head is one content-dependent L-tap FIR filter applied
    #   identically to every value channel -- and that single shared temporal profile is the
    #   rank bound arch_wgroup.py describes, which gets WORSE with width: one filter for
    #   dv=56 at d=128, one for dv=256 at d=1024. G gives the pillar G profiles instead of 1,
    #   each serving dv/G channels. Identical at init (every group starts at the same w).
    #   NOTE: wg2 refuted this on the LONG head; the short head is where the loss actually
    #   lives (+6.45 nats when muted, against the long head's +2.77) and has never been tried.
    beta_groups: int = 1  # erase-gate granularity. 1 = one scalar per token for ALL M modes,
    #   which is what the layer has always done -- and it is incoherent with its own design:
    #   the spectrum is deliberately heterogeneous (half persistent with infinite memory, a
    #   slow-integrator slice, a damped fast remainder) yet one write strength is forced on
    #   all of it. A fast pole holds short-lived content that should be overwritten hard; a
    #   persistent pole holds document memory that should not. 3 splits it by band --
    #   slow integrators / persistent-but-fast / damped -- for 2*d parameters. Identical at
    #   init (bproj starts at zero weight and a constant bias, so every band agrees).
    kv_gate_pc: bool = False  # PER-CHANNEL key-verification gate: ga, gb become vectors of
    #   length 2*dv instead of scalars, so each output channel gets its own slope and bias on
    #   the SAME evidence m. Both reference architectures gate per channel (GDN's is
    #   silu(gp(x)) over H*dv); ours was one number for all 2*dv channels, which was never a
    #   measured choice. Identical to the scalar gate at init (same 4.0 / 0.0), so it can only
    #   earn its keep. 2*(2*dv) parameters -- 1024 at dv=256.
    gdn_gate: bool = False  # REPLACE our readout with GDN's: LayerNorm(o) * silu(Linear(x)).
    #   Our gate is a SCALAR per token (sigmoid of a cosine match); GDN's is a LEARNED
    #   NON-LINEAR function of x, PER CHANNEL. Each of the 2*dv output channels gets its own
    #   silu(W_gp @ x) gate that can independently amplify or suppress it. This is a ROUTER,
    #   not a confidence gate. Cost: d*(2*dv) + 2*dv + 4*dv = ~525k params at d=1024 dv=256
    #   (LayerNorm 2*dv + Linear d->2*dv). Replaces kv_dk when both are on.
    gdn_gate_scope: str = "long"  # where the gdn-gate applies when gdn_gate=True:
    #   "long"   = long head only (default, what we've tested so far)
    #   "both"   = long head AND short head (separate gp per head)
    #   "concat" = after concat [ul, us] before mix (one gate over 4*dv channels)
    #   "mix"    = after mix projection, before residual add (gate over d channels)
    conv: int = 0  # width of a causal depthwise conv applied to z BEFORE both heads (0 = none).
    #   Every competitive linear mixer has one -- Mamba, GDN (kernel 4 on q/k/v), LFM2 -- and
    #   this layer did not. Initialised to the identity, so at init it is exactly a no-op.
    v_silu: bool = False  # SiLU on V(z) — the values written into the state become non-linear
    #   in z. GDN does this: v = silu(conv(Wv @ z)). Ours was purely linear: v = V(z).
    #   With v_silu the state captures non-linear features rather than linear projections.
    #   Zero params, one elementwise op. NOT identity at init.
    k_silu: bool = False  # SiLU on K(z) — the keys become non-linear in z, giving richer
    #   content features to the phase computation. GDN does q = silu(conv(Wq @ z)),
    #   k = silu(conv(Wk @ z)). Zero params.
    init_v2: bool = False  # CALIBRATED INIT derived from two 16h checkpoints (t2048_gdngate,
    #   t2048_dv384_silu). Every parameter starts where the model converges to, not at the
    #   standard default. Zero params, zero compute, just better starting points.


def _apply_init_v2(layer: "LaplaceAttention"):
    """Calibrated init from converged checkpoints. Every parameter starts where
    the model ends up after 16h of training, not at the standard default.

    Derived from the median of t2048_gdngate (1518M tokens) and t2048_dv384_silu
    (1170M tokens), averaged over 8 layers and 2 checkpoints.
    """
    import math
    cfg = layer.cfg
    d, M, dv, L = cfg.d, cfg.M, cfg.dv, cfg.L

    with torch.no_grad():
        # --- LayerScale ---
        if cfg.layer_scale:
            layer.gs_mix.fill_(0.1)          # converges to 0.12, init was 1.0
            layer.gs_ff.fill_(0.5)           # converges to 0.50, init was 1.0

        # --- Input LayerNorm ---
        layer.n.weight.fill_(0.8)            # converges to 0.78, init was 1.0

        # --- FFN LayerNorm ---
        layer.fn.weight.fill_(0.4)           # converges to 0.36, init was 1.0

        # --- FFN up bias ---
        layer.ff[0].bias.fill_(-0.1)         # converges to -0.10, init was 0.0

        # --- Long head ---
        long = layer.long

        # theta: N(0, 0.3) instead of N(0, 0.02)
        long.theta.copy_(0.3 * torch.randn(M))

        # wr, wi: N(0.3, 0.5) and N(0, 0.5) instead of (ones, zeros)
        long.wr.copy_(0.3 + 0.5 * torch.randn_like(long.wr))
        long.wi.copy_(0.5 * torch.randn_like(long.wi))

        # bproj bias: -1.0 instead of -2.0 (sigmoid(-1)=0.27 vs 0.12)
        nn.init.constant_(long.bproj.bias, -1.0)

        # o_norm weight: 0.8 instead of 1.0
        if hasattr(long, 'o_norm'):
            long.o_norm.weight.fill_(0.8)

        # --- Short head ---
        short = layer.short

        # theta: N(0, 0.1) instead of N(0, 0.02)
        short.theta.copy_(0.1 * torch.randn(L))

        # wr, wi: N(0.5, 0.3) and N(0, 0.3) instead of (ones, zeros)
        short.wr.copy_(0.5 + 0.3 * torch.randn_like(short.wr))
        short.wi.copy_(0.3 * torch.randn_like(short.wi))


def _state_dtype(var: str, default: str) -> torch.dtype:
    """Storage dtype of a decode state. Narrowing these is safe for a reason,
    not by luck: the long head's recurrence is s <- s*damp + e (x) kt with
    damp < 1, a CONTRACTION, so a rounding perturbation decays instead of
    compounding. Measured over 256 decode steps, fp16 and bf16 both reproduce
    every token id; fp16 does it with a tenth of bf16's logit error.
    """
    import os
    return {"fp32": torch.float32, "fp16": torch.float16,
            "bf16": torch.bfloat16}[os.environ.get(var, default)]


def _ring_dtype() -> torch.dtype:
    """Storage dtype of the short head's decode window. $SCA2_RING_DTYPE.

    fp16 by default, and NOT bf16. The window holds cosines, sines and a
    unit-RMS projection, so nothing needs bf16's exponent range -- the largest
    |e| measured over a trained checkpoint is 65.8 against fp16's 65504, a
    thousandfold margin -- while fp16's two extra mantissa bits are worth a
    measured 6.3x on the head's output (7.8e-4 relative against 4.9e-3). The
    reductions stay fp32; only the storage narrows, because storage is what
    multiplies by batch.
    """
    import os
    return {"fp32": torch.float32, "fp16": torch.float16,
            "bf16": torch.bfloat16}[os.environ.get("SCA2_RING_DTYPE", "fp16")]


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
    _has_anchor = False          # class default: the other lam branches never anchor

    def __init__(self, cfg: LaplaceConfig):
        super().__init__()
        if cfg.beta_write and cfg.beta_groups != 1:
            raise ValueError("beta_write requires beta_groups=1 (one write gate per token)")
        d, M, dv = cfg.d, cfg.M, cfg.dv
        # dv is what the head EMITS (2*dv, Re||Im); dvi is what the state carries. With key
        # verification the state also carries the dk-wide key copy, so dvi = dv + dk.
        self.dk = cfg.kv_dk
        self.dvi = dv + self.dk
        self.d, self.M, self.dv, self.cfg = d, M, dv, cfg
        self.K = nn.Linear(d, M, False)
        self.V = nn.Linear(d, dv, False)
        if cfg.gdn_gate:
            self.o_norm = nn.LayerNorm(2 * dv)
            self.gp = nn.Linear(d, 2 * dv)
        elif self.dk:
            self.Kv = nn.Linear(d, self.dk, False)
            # 0-dim when shared (matches sca2's CHeadDeltaKV exactly), (2*dv,) when per-channel
            sh = (2 * dv,) if cfg.kv_gate_pc else ()
            self.ga = nn.Parameter(torch.full(sh, 4.0))  # gate slope on the cosine
            self.gb = nn.Parameter(torch.zeros(sh))      # bias: g = 0.5 at zero evidence
        self.theta = nn.Parameter(
            torch.zeros(M)
            if cfg.theta_scale == 0.0
            else cfg.theta_scale * torch.randn(M)
        )
        # (M,) when shared -- the shape sca2's mirror uses, so the float64 gate still loads --
        # and (M, NG) when grouped.
        self.NG = max(1, cfg.long_groups)
        assert self.dvi % self.NG == 0, \
            f"dv+kv_dk={self.dvi} must divide by long_groups={self.NG}"
        wsh = (M,) if self.NG == 1 else (M, self.NG)
        wr_init = torch.ones(wsh)
        wi_init = torch.zeros(wsh)
        if self.NG > 1 and cfg.w_antipodal > 0:
            # Antipodal symmetry breaking: w_g = 1 ± eps*n. The MEAN across groups is
            # exactly 1, so the layer's average behaviour is unchanged at init. The noise
            # is shared (same n, opposite signs) to maximise the initial separation.
            eps = cfg.w_antipodal
            nr = eps * torch.randn(M)
            ni = eps * torch.randn(M)
            for g in range(self.NG):
                sign = 1.0 - 2.0 * g / (self.NG - 1) if self.NG > 1 else 0.0
                wr_init[:, g] = 1.0 + sign * nr
                wi_init[:, g] = sign * ni
        self.R = max(1, cfg.read_mix)
        if self.R > 1 and self.NG > 1:
            raise ValueError("read_mix > 1 with long_groups > 1 is not implemented: "
                             "they are two different answers to the same question")
        if self.R > 1:
            # R copies of the initialised weight, separated by antipodal noise so
            # the mixture does not start degenerate; alpha starts uniform because
            # A is zero, so step 0 is identical to the single-weight layer.
            base_r, base_i = wr_init.clone(), wi_init.clone()
            nr, ni = 0.05 * torch.randn(self.R, M), 0.05 * torch.randn(self.R, M)
            nr -= nr.mean(0, keepdim=True)
            ni -= ni.mean(0, keepdim=True)
            wr_init = base_r.reshape(1, M).repeat(self.R, 1) + nr
            wi_init = base_i.reshape(1, M).repeat(self.R, 1) + ni
            self.alpha_proj = nn.Linear(d, self.R, bias=False)
            nn.init.zeros_(self.alpha_proj.weight)
        self.wr = nn.Parameter(wr_init)  # spectral read weights, w = wr + i wi
        self.wi = nn.Parameter(wi_init)
        _om = rope_grid(M, cfg.rope_base, cfg.slow_frac, cfg.max_len,
                        cfg.rope_min_period)
        if cfg.learn_omega:
            self.omega = nn.Parameter(_om)
            self.register_buffer("_omega_init", _om.clone())
        else:
            self.register_buffer("omega", _om)
        if cfg.decay_input:
            self.lam_proj = nn.Linear(d, M, False)
            nn.init.zeros_(self.lam_proj.weight)   # at init: identical to learn_persist
        self.bg = max(1, cfg.beta_groups)
        self.bproj = nn.Linear(d, self.bg, True)  # erase gate beta = sigmoid(bproj(z))
        nn.init.zeros_(self.bproj.weight)
        nn.init.constant_(self.bproj.bias, cfg.beta_init)
        lo, hi, self.lam_max = _damp_params(cfg)
        mem = torch.exp(torch.empty(M).uniform_(math.log(lo), math.log(hi)))
        n_pin = int(round(cfg.persist * M))
        low = self.omega.abs().argsort()[:n_pin]  # lowest frequencies persist
        if cfg.lam_free:
            # lambda = exp(a): every mode log-uniform over the WHOLE of mem_range, nothing
            # pinned, nothing masked. `persist` is ignored on purpose -- freeing the modes is
            # the point -- so set mem_range wide (the measured GDN span is 2.5 .. 5.8e6).
            self.lam_ceil = cfg.lam_ceil if cfg.lam_ceil is not None else 55.0 / cfg.chunk
            assert self.lam_ceil * cfg.chunk < 88.0, \
                f"lam_ceil*chunk = {self.lam_ceil*cfg.chunk:.1f} overflows fp32 in e^(lam*chunk)"
            self.lam_raw = nn.Parameter(torch.log(1.0 / mem))     # exp^{-1}
            self.register_buffer("lam_mask", torch.ones(M))
            # Anchors are spread over the SORTED memories, so the pinned set
            # covers the whole range rather than a random clump of it.
            n_anch = int(round(max(0.0, min(1.0, cfg.lam_anchor)) * M))
            am = torch.zeros(M, dtype=torch.bool)
            if n_anch:
                order = mem.argsort()
                am[order[torch.linspace(0, M - 1, n_anch).round().long()]] = True
            self.register_buffer("lam_anchor_mask", am)
            self.register_buffer("lam_anchor_raw", torch.log(1.0 / mem))
            self._has_anchor = bool(n_anch)
        elif cfg.learn_persist:
            # lambda_m = lam_max * sigmoid(a_m): reaches ~0 (a=-8 -> memory > 20k tokens) or the
            # cap within a few hundred steps either way; nothing pinned, the task decides the split.
            a = torch.logit((1.0 / mem / self.lam_max).clamp(1e-4, 1 - 1e-4))
            a[low] = -8.0
            self.lam_raw = nn.Parameter(a)
            self.register_buffer("lam_mask", torch.ones(M))
            self._has_anchor = False
        else:
            self.lam_raw = nn.Parameter(
                torch.log(torch.expm1(1.0 / mem))
            )  # softplus^{-1}(lambda)
            mask = torch.ones(M)
            if n_pin:
                mask[low] = 0.0
            self.register_buffer("lam_mask", mask)
        if self.bg > 1:
            # disjoint bands, in the order the grid is built: 0 = slow integrators (the last
            # n_slow modes), 1 = persistent but on the fast grid, 2 = damped. Extra groups
            # beyond 3 subdivide the damped band evenly.
            n_slow = int(round(cfg.slow_frac * M))
            g = torch.full((M,), min(2, self.bg - 1), dtype=torch.long)
            pinned = torch.zeros(M, dtype=torch.bool)
            if not cfg.lam_free:      # with free modes there is no persistent band to split on
                pinned[low] = True
            g[pinned] = min(1, self.bg - 1)
            if n_slow:
                g[M - n_slow:] = 0
            if self.bg > 3:                       # subdivide the damped band
                dmp = (~pinned).nonzero(as_tuple=True)[0]
                g[dmp] = 2 + (torch.arange(len(dmp)) * (self.bg - 2) // max(len(dmp), 1))
            self.register_buffer("bgroup", g.clamp(max=self.bg - 1))

    # ---- pieces ----------------------------------------------------------- #
    @property
    def wd(self) -> torch.dtype:
        return _wd(self.wr)

    # softplus(log(e-1)) == 1 exactly, so the modulation starts as a no-op and a
    # decay_softplus run is identical to a static-decay run at step 0.
    _SP_B0 = math.log(math.e - 1.0)

    def _lam_raw(self):
        """lam_raw with the anchored modes held at their initial value."""
        # The branch is on a PYTHON bool decided at __init__, never on a tensor.
        # `bool(mask.any())` here is data-dependent control flow and torch.compile
        # cannot trace it: it broke 3 of test_triton_state's 9 cases.
        if not self._has_anchor:
            return self.lam_raw
        return torch.where(self.lam_anchor_mask, self.lam_anchor_raw, self.lam_raw)

    def lam_t(self, lz):
        """(...,M) per-token decay from the ALREADY-PROJECTED lam_proj(z)."""
        if self.cfg.lam_free:
            if self.cfg.decay_softplus:
                # GDN's actual form: rate = exp(A_log) * softplus(a(x) + bias).
                # The exp(a + lz) branch below claimed to be this and is not, and
                # the difference is not cosmetic. exp() grows exponentially into
                # the clamp, and a clamped rate has NO GRADIENT: at lz = +2,
                # 18% of the modes are pinned and lam_proj stops learning for
                # them; at +3, 32%. It bites the FAST modes first -- memory 4
                # saturates at lz > 0.54 while memory 20000 never does -- which
                # are exactly the ones an input-dependent forget gate is for.
                # softplus grows linearly, so the clamp is reached far later and
                # the gradient survives up to it.
                return (torch.exp(self._lam_raw().to(self.wd))
                        * F.softplus(lz.to(self.wd) + self._SP_B0)).clamp(max=self.lam_ceil)
            return torch.exp(self._lam_raw().to(self.wd) + lz.to(self.wd)).clamp(max=self.lam_ceil)
        return self.lam_max * torch.sigmoid(self._lam_raw().to(self.wd) + lz.to(self.wd))

    def omega_stats(self):
        """(drift, log-span, effective count) of the frequency grid, or None.

        Logged with every eval when --learn-omega is on, because the CHECKPOINT
        only ever holds the endpoint and what teaches us how to initialise is
        the trajectory: whether the frequencies spread, collapse, or converge on
        a few values the grid should have started at.

        drift      rms |omega - omega_init| / rms omega_init
        log-span   ln(p95 / p5) of |omega|, the range of periods actually kept
        n_eff      exp(entropy of |omega| / sum |omega|): how many frequencies
                   carry weight, against M if the grid stayed spread
        """
        if not isinstance(self.omega, nn.Parameter):
            return None
        with torch.no_grad():
            om = self.omega.float()
            a = om.abs()
            init = self._omega_init
            drift = ((om - init).pow(2).mean().sqrt()
                     / init.pow(2).mean().sqrt().clamp(min=1e-9))
            q = torch.quantile(a, torch.tensor([0.05, 0.95], device=a.device))
            span = torch.log(q[1].clamp(min=1e-9) / q[0].clamp(min=1e-9))
            pr = a / a.sum().clamp(min=1e-9)
            neff = torch.exp(-(pr * (pr + 1e-12).log()).sum())
        return [round(drift.item(), 4), round(span.item(), 3), round(neff.item(), 1)]

    def w_eff(self, z):
        """(wr, wi) for this token: a point in the span of the R learned weights.

        Mixing the WEIGHTS and mixing the R separate READS are the same thing --
        the read is linear in w -- so doing it here costs one (.., R) x (R, M)
        product and leaves the read GEMM untouched. Doing it the other way would
        have cost R times the read. chead_numpy.py checks the equivalence.
        """
        if self.R == 1:
            return self.wr, self.wi
        a = F.softmax(self.alpha_proj(z.to(self.wd)), -1)        # (..., R)
        # Record how much of the mixture is actually used. The failure mode is
        # alpha collapsing onto one r, which turns this back into a single
        # weight and makes a null result unreadable: a flat curve would not say
        # whether the mechanism does not help or was never engaged. Entropy in
        # nats, ln(R) when uniform, 0 when collapsed. No grad, no sync.
        with torch.no_grad():
            self.alpha_entropy = -(a * (a + 1e-9).log()).sum(-1).mean()
        return a @ self.wr.to(self.wd), a @ self.wi.to(self.wd)  # (..., M)

    def lam(self) -> torch.Tensor:
        if self.cfg.lam_free:
            return torch.exp(self._lam_raw().to(self.wd)).clamp(max=self.lam_ceil)
        if self.cfg.learn_persist:
            return self.lam_max * torch.sigmoid(self._lam_raw().to(self.wd))
        return (
            F.softplus(self._lam_raw().to(self.wd)).clamp(max=self.lam_max)
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

    def _beta(self, bz):
        """sigmoid(bproj(z)) -> (...,1) when shared, (...,M) when banded."""
        b = torch.sigmoid(bz.to(self.wd))
        return b if self.bg == 1 else b[..., self.bgroup]

    def _out(self, u, z, gate=None):
        """(..., 2*dvi) -> (..., 2*dv)."""
        dv, dvi = self.dv, self.dvi
        re, im = u[..., :dvi], u[..., dvi:]
        val = torch.cat([re[..., :dv], im[..., :dv]], -1)
        if self.cfg.gdn_gate:
            return self.o_norm(val) * F.silu(self.gp(z) if gate is None else gate)
        if not self.dk:
            return _rms(torch.cat([u[..., :dv], u[..., dvi:dvi + dv]], -1))
        m = F.cosine_similarity(re[..., dv:], self.Kv(z).to(re.dtype), dim=-1, eps=1e-6)
        return _rms(val) * torch.sigmoid(self.ga * m[..., None] + self.gb)

    def init_state(self, B: int, device) -> State:
        # One (B, 2M, dv) block, rows [Re ; Im]. Keeping the two halves in ONE tensor is
        # what lets every read and write of the state be a single GEMM against a (., 2M)
        # code block instead of a pair -- see _batched.
        return {
            "s": torch.zeros(B, 2 * self.M, self.dvi, device=device, dtype=self.wd),
            "pos": torch.zeros((), device=device, dtype=self.wd),
        }

    def _damp(self, lam, n):
        """(2M,1) decay factors for the packed state."""
        return torch.exp(-lam * n)[:, None].repeat(2, 1)

    # ---- prefill: one chunk against an incoming state ---------------------- #
    def _chunk(self, kz, kh, vz, bz, lz, mw, st: State):
        B, T, _ = kz.shape
        M = self.M
        dev = kz.device
        gd = self._gemm_dtype(kz)
        with _no_autocast(dev):
            di = self.cfg.decay_input
            lam = self.lam()
            idx = torch.arange(T, device=dev, dtype=self.wd)[:, None]
            if di:
                Ct = self.lam_t(lz).cumsum(1)                      # (B,T,M) inclusive
                gw, gq = torch.exp(Ct), torch.exp(-Ct)
                dT_2 = torch.cat([torch.exp(-Ct[:, -1:]), torch.exp(-Ct[:, -1:])], -1)[:, 0]
            else:
                gw, gq = torch.exp(lam * idx), torch.exp(-lam * idx)  # (T,M)
            gT_2 = torch.exp(-lam * (T - 1))[:, None].repeat(2, 1) if not self.cfg.decay_input else None
            p = idx + st["pos"]
            pw, pq = self._phase(kh, p), self._phase(kz, p)  # (B,T,M) fp32
            cw, sw, cq, sq = pw.cos(), pw.sin(), pq.cos(), pq.sin()
            Kk = torch.cat([cw * gw, sw * gw], -1).to(gd)  # (B,T,2M) keys
            Qk = torch.cat([cw * gq, sw * gq], -1).to(gd)  # (B,T,2M) Gram lhs / read-back
            if self.NG == 1:
                wr_, wi_ = (self.wr, self.wi) if mw is None else mw
                c1 = (wr_ * cq + wi_ * sq) * gq
                c2 = (wr_ * sq - wi_ * cq) * gq
                Fq = torch.cat([torch.cat([c1, c2], -1),
                                torch.cat([-c2, c1], -1)], 1).to(gd)  # (B,2T,2M)
            else:
                cg, sg_, gg = cq[..., None], sq[..., None], gq[..., None]
                c1 = (self.wr * cg + self.wi * sg_) * gg               # (B,T,M,NG)
                c2 = (self.wr * sg_ - self.wi * cg) * gg
                f1 = torch.cat([c1, c2], 2).view(B, T, 2 * M, self.NG)
                f2 = torch.cat([-c2, c1], 2).view(B, T, 2 * M, self.NG)
                Fq = torch.cat([f1, f2], 1).permute(0, 3, 1, 2).to(gd)  # (B,NG,2T,2M)
            s0 = (st["s"] if di else st["s"] * self._damp(lam, 1)).to(gd)
            # The erase gate folds into the QUERY-side codes. beta multiplies the read-back
            # inside the sum over modes, so a per-mode beta is a per-mode scale on Qk -- the
            # chunked closed form is unchanged, it just sees pre-scaled queries.
            beta = self._beta(bz)  # (B,T,1) or (B,T,M)
            Qk = (Qk * torch.cat([beta, beta], -1).to(gd)) if self.bg > 1 else Qk
            G = (Qk @ Kk.transpose(1, 2)).to(self.wd) / M
            A = torch.eye(T, device=dev, dtype=self.wd) + (
                G.tril(-1) if self.bg > 1 else beta * G.tril(-1))
            r = (Qk @ s0).to(self.wd) / M
            v = vz.to(self.wd)
            if self.cfg.beta_write:
                v = beta * v
            e = torch.linalg.solve_triangular(
                A, v - (r if self.bg > 1 else beta * r),
                upper=False, unitriangular=True
            )
            ec = e.to(gd)
            Kkt = Kk.transpose(1, 2) if self.NG == 1 else Kk.transpose(1, 2)[:, None]
            K2 = (Fq @ Kkt).masked_fill(
                _causal_mask(T, dev).repeat(2, 1)[None], 0)  # (B,[NG,]2T,T) [Re;Im]
            if self.NG == 1:
                o = (K2 @ ec + Fq @ s0).to(self.wd)  # (B,2T,dvi)
            else:
                NG, w_ = self.NG, self.dvi // self.NG
                ecg = ec.view(B, T, NG, w_).transpose(1, 2)          # (B,NG,T,w)
                s0g = s0.view(B, 2 * M, NG, w_).transpose(1, 2)      # (B,NG,2M,w)
                og = (K2 @ ecg + Fq @ s0g).to(self.wd)               # (B,NG,2T,w)
                o = og.transpose(1, 2).reshape(B, 2 * T, self.dvi)
            o = torch.cat([o[:, :T], o[:, T:]], -1) / M  # (B,T,2dv) = Re || Im
            if di:
                f = dT_2[:, :, None]
                sn = st["s"] * f + (Kk.transpose(1, 2) @ ec).to(self.wd) * f
            else:
                sn = st["s"] * self._damp(lam, T) + (
                    Kk.transpose(1, 2) @ ec).to(self.wd) * gT_2
        return o, {"s": sn, "pos": st["pos"] + T}

    # ---- prefill: all full chunks batched, state loop only ----------------- #
    def _batched(self, kz, kh, vz, bz, lz, mw, st: State, K: int):
        B, T, _ = kz.shape
        M, dv, C = self.M, self.dvi, self.cfg.chunk
        dev = kz.device
        gd = self._gemm_dtype(kz)
        with _no_autocast(dev):
            di = self.cfg.decay_input
            if di:
                # C_t = sum_{u<=t} lam_u along the chunk; decay(t,s) = exp(-(C_t - C_s)),
                # so gw_s = exp(C_s), gq_t = exp(-C_t) and the closed form is unchanged.
                lam_all = self.lam_t(lz).view(B, K, C, M)            # (B,K,C,M)
                Ct = lam_all.cumsum(2)                                # inclusive
                gw = gq = None            # built inside the kernel when it runs
                dC = gT = torch.exp(-Ct[:, :, -1:])                   # (B,K,1,M)
                d1 = None                                             # folded into gq
            else:
                lam = self.lam()
                idx = torch.arange(C, device=dev, dtype=self.wd)[:, None]
                gw, gq = torch.exp(lam * idx), torch.exp(-lam * idx)
                dC, gT, d1 = (
                    torch.exp(-lam * C)[:, None],
                    torch.exp(-lam * (C - 1))[:, None],
                    torch.exp(-lam)[:, None],
                )
            ch = lambda x: x.view(B, K, C, x.shape[-1])
            v, beta = ch(vz.to(self.wd)), ch(self._beta(bz))
            if self.cfg.beta_write:
                # All backends consume gated values. Autograd adds the write
                # contribution to d beta alongside the existing erase/solve terms.
                v = beta * v
            # `di` no longer disqualifies the kernel: it takes the per-token
            # ramp as Ct. Leaving it out cost the whole code path, not just the
            # ramp -- decay_input reverted to PyTorch and ran at 0.79x.
            use_codes = (self.cfg.long_path in ("triton_fused", "triton_codes",
                                                "triton_scan")
                         and kz.is_cuda and self.wd == torch.float32
                         and gd in (torch.float32, torch.bfloat16) and self.bg == 1)
            if use_codes:
                from .triton_phase import phase_codes
                wr_k, wi_k = ((self.wr, self.wi) if mw is None else mw)
                Kk, Qk, Fq = phase_codes(kz, kh, self.theta, self.omega,
                                          self.lam() if di else lam,
                                          wr_k, wi_k, st["pos"], C, gd,
                                          # the compact layout is only read by
                                          # the scan, which decay_input skips
                                          (self.cfg.long_path == "triton_scan") and not di,
                                          ct=Ct if di else None)
                if self.NG == 1:
                    Fq = Fq.squeeze(2)
            else:
                if gw is None:                       # decay_input, kernel off
                    gw, gq = torch.exp(Ct), torch.exp(-Ct)
                p = torch.arange(T, device=dev, dtype=self.wd)[:, None] + st["pos"]
                pw, pq = self._phase(kh, p), self._phase(kz, p)
                cw, sw, cq, sq = map(ch, (pw.cos(), pw.sin(), pq.cos(), pq.sin()))
                # Packed real/imaginary codes share the same state GEMMs.
                Kk = torch.cat([cw * gw, sw * gw], -1).to(gd)
                Qk = torch.cat([cw * gq, sw * gq], -1).to(gd)
                if self.bg > 1:
                    Qk = Qk * torch.cat([beta, beta], -1).to(gd)
                if self.NG == 1:
                    wr_, wi_ = (self.wr, self.wi) if mw is None else (
                        ch(mw[0]), ch(mw[1]))
                    c1 = (wr_ * cq + wi_ * sq) * gq
                    c2 = (wr_ * sq - wi_ * cq) * gq
                    Fq = torch.cat([torch.cat([c1, c2], -1),
                                    torch.cat([-c2, c1], -1)], 2).to(gd)
                else:
                    cg, sg_, gg = cq[..., None], sq[..., None], gq[..., None]
                    c1 = (self.wr * cg + self.wi * sg_) * gg
                    c2 = (self.wr * sg_ - self.wi * cg) * gg
                    f1 = torch.cat([c1, c2], 3).view(B, K, C, 2 * M, self.NG)
                    f2 = torch.cat([-c2, c1], 3).view(B, K, C, 2 * M, self.NG)
                    Fq = torch.cat([f1, f2], 2).permute(0, 1, 4, 2, 3).to(gd)
            # The scan kernel takes d1/dC/gT as PER-MODE constants; with a
            # per-token ramp d1 does not exist and dC/gT are per (b, k, m). That
            # is a separate kernel change, so decay_input takes the codes kernel
            # and the Gram product but not the fused scan -- still far better
            # than the all-PyTorch path it had.
            if use_codes and self.cfg.long_path == "triton_scan" and not di:
                # Gram, inverse, causal kernel and chunk loop are ONE graph node:
                # every code gradient is accumulated in place by the node that
                # produces the next one, so no gradient add pass runs at all.
                # The result is already [Re | Im] packed and divided by M.
                from .triton_scan import long_chunk
                d1_2, dC_2, gT_2 = (x.repeat(2, 1) for x in (d1, dC, gT))
                o, sn = long_chunk(Kk, Qk, Fq.unsqueeze(2) if self.NG == 1 else Fq,
                                   v, beta, st["s"], d1_2, dC_2, gT_2)
                return o, {"s": sn, "pos": st["pos"] + T}
            if use_codes:
                from .triton_product import code_product
                G = code_product(Qk.unsqueeze(2), Kk, gram=True).squeeze(2)
            else:
                G = (Qk @ Kk.transpose(-1, -2)).to(self.wd) / M
            if self.cfg.long_path in ("triton", "triton_fused"):
                from .triton_solve import triangular_inverse
                W = triangular_inverse(G, None if self.bg > 1 else beta)
            else:
                eye = torch.eye(C, device=dev, dtype=self.wd)
                W = torch.linalg.solve_triangular(
                    eye + (G.tril(-1) if self.bg > 1 else beta * G.tril(-1)),
                    eye.expand(B, K, C, C),
                    upper=False,
                    unitriangular=True,
                )  # (B,K,C,C)
            # K2 = Fq Kk^T in one GEMM: [c1|c2] Kk^T is the real part and [-c2|c1] Kk^T
            # the imaginary one, so the (2C,C) block comes out already stacked [Re ; Im]
            # -- the same layout the state read below produces, so the two just add.
            Kkt = Kk.transpose(-1, -2) if self.NG == 1 else Kk.transpose(-1, -2)[:, :, None]
            if use_codes:
                K2 = code_product(Fq.unsqueeze(2) if self.NG == 1 else Fq, Kk)
                if self.NG == 1:
                    K2 = K2.squeeze(2)
            else:
                K2 = (Fq @ Kkt).masked_fill(
                    _causal_mask(C, dev).repeat(2, 1)[None, None], 0)
            if di:
                dC_2 = torch.cat([dC, dC], -1).squeeze(2)             # (B,K,2M)
                gT_2 = dC_2
                d1_2 = None
            else:
                d1_2, dC_2, gT_2 = (x.repeat(2, 1) for x in (d1, dC, gT))
        if (self.cfg.long_path == "triton_fused" and kz.is_cuda
                and self.wd == torch.float32 and gd in (torch.float32, torch.bfloat16)
                and not di and self.bg == 1):
            from .triton_state import state_loop
            fq = Fq.unsqueeze(2) if self.NG == 1 else Fq
            k2 = K2.unsqueeze(2) if self.NG == 1 else K2
            with _no_autocast(dev):
                o, s = state_loop(Kk, Qk, fq, k2, W, v, beta, st["s"], d1_2, dC_2, gT_2)
                o = torch.cat([o[:, :, :C], o[:, :, C:]], -1).reshape(B, K * C, 2 * dv) / M
            return o, {"s": s, "pos": st["pos"] + T}
        Fq, W, v, beta, K2, Kk, Qk = (
            x.unbind(1) for x in (Fq, W, v, beta, K2, Kk, Qk)
        )
        if di:
            dC_l, gT_l = dC_2.unbind(1), gT_2.unbind(1)   # per chunk (B,2M)
        s = st["s"]
        outs = []
        for k in range(K):
            with _no_autocast(dev):
                # with per-token decay the incoming-state factor is already inside gq
                s0 = (s if di else s * d1_2).to(gd)  # (B,2M,dvi)
                r = (Qk[k] @ s0).to(self.wd) / M
                e = W[k] @ (v[k] - (r if self.bg > 1 else beta[k] * r))  # delta-rule solve
                ec = e.to(gd)
                if self.NG == 1:
                    outs.append((K2[k] @ ec + Fq[k] @ s0).to(self.wd))  # (B,2C,dvi) [Re;Im]
                else:
                    NG, w_ = self.NG, self.dvi // self.NG
                    ecg = ec.view(B, C, NG, w_).transpose(1, 2)          # (B,NG,C,w)
                    s0g = s0.view(B, 2 * M, NG, w_).transpose(1, 2)      # (B,NG,2M,w)
                    og = (K2[k] @ ecg + Fq[k] @ s0g).to(self.wd)         # (B,NG,2C,w)
                    outs.append(og.transpose(1, 2).reshape(B, 2 * C, self.dvi))
                if di:
                    s = s * dC_l[k][:, :, None] \
                        + (Kk[k].transpose(-1, -2) @ ec).to(self.wd) * gT_l[k][:, :, None]
                else:
                    s = s * dC_2 + (Kk[k].transpose(-1, -2) @ ec).to(self.wd) * gT_2
        o = torch.stack(outs, 1)  # (B,K,2C,dv)
        o = torch.cat([o[:, :, :C], o[:, :, C:]], -1).reshape(B, K * C, 2 * dv) / M
        return o, {"s": s, "pos": st["pos"] + T}

    def _project(self, z):
        """K/V/beta and optional precomputed output gate (separate by default)."""
        k = self.K(z)
        if self.cfg.k_silu:
            k = F.silu(k)
        v = self.V(z)
        if self.cfg.v_silu:
            v = F.silu(v)
        return k, v, self.bproj(z), None

    def prefill(self, z, z_prev, state: Optional[State] = None):
        """z (B,T,d); z_prev (B,d) is the token before z[:,0] -- the write key at t is z_{t-1}."""
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device)
        # K, V and bproj are linear and follow autocast; doing them once here rather than
        # per chunk also means the write keys K(h) are K(z) shifted by one row, so the
        # second d x M projection and the (B,T,d) shifted copy of z both disappear.
        kz, vz, bz, gate = self._project(z)
        lz = self.lam_proj(z) if self.cfg.decay_input else None
        # Per-token read weights, threaded exactly like lz: one (.., R) x (R, M)
        # product for the whole sequence, then sliced per chunk.
        mw = self.w_eff(z) if self.R > 1 else None
        kh = torch.cat([self.K(z_prev)[:, None], kz[:, :-1]], 1)
        if self.dk:
            # the stored key is Kv(h_s) = Kv of the PREVIOUS token; Kv is linear, so the
            # same shift trick as K applies and no second projection of h is needed.
            kvz = self.Kv(z)
            vz = torch.cat([vz, torch.cat([self.Kv(z_prev)[:, None], kvz[:, :-1]], 1)], -1)
        C = self.cfg.chunk
        K = T // C
        cut = lambda a, b: (kz[:, a:b], kh[:, a:b], vz[:, a:b], bz[:, a:b],
                            lz[:, a:b] if lz is not None else None,
                            None if mw is None else (mw[0][:, a:b], mw[1][:, a:b]))
        if K >= 2 and self.cfg.long_path in ("batched", "triton", "triton_fused",
                                            "triton_codes", "triton_scan"):
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
        return self._out(o, z, gate).to(z.dtype), st

    # ---- decode: one token, all float32 ------------------------------------ #
    def _maybe_fast(self):
        """Attach the fused decode kernel once, if it implements this head.

        Lazy rather than in __init__ so importing the layer never needs Triton,
        and so a head built on CPU still works. $SCA2_LAPA_DECODE=naive forces
        the PyTorch path back, which is how the two are compared.
        """
        import os
        if os.environ.get("SCA2_LAPA_DECODE", "auto") == "naive":
            return None
        try:
            from .triton_decode import supported, long_step
        except Exception:                                     # pragma: no cover
            return None
        if not supported(self):
            return None
        import functools
        return functools.partial(long_step, block_dv=64, block_m=128)

    def step(self, z_t, h_t, state: State):
        if not hasattr(self, "_decode_fast") and z_t.is_cuda:
            self._decode_fast = self._maybe_fast()
        vz = self.V(z_t)
        if self.cfg.v_silu:
            vz = F.silu(vz)
        kh, kz = self.K(h_t), self.K(z_t)
        if self.cfg.k_silu:
            kh, kz = F.silu(kh), F.silu(kz)
        bz = self.bproj(z_t)
        lz = self.lam_proj(z_t) if self.cfg.decay_input else None
        if self.dk:
            vz = torch.cat([vz, self.Kv(h_t)], -1)
        with _no_autocast(z_t.device):
            M = self.M
            if self.cfg.decay_input:
                lt = self.lam_t(lz)                                   # (B,M)
                s0 = state["s"] * torch.cat([lt, lt], -1).neg().exp()[:, :, None]
            else:
                s0 = state["s"] * self._damp(self.lam(), 1)  # (B,2M,dv)
            p = state["pos"]
            pw, pq = self._phase(kh, p), self._phase(kz, p)  # (B,M)
            kt = torch.cat([pw.cos(), pw.sin()], -1)  # (B,2M) write code
            beta = self._beta(bz)
            fast = getattr(self, "_decode_fast", None) if self.R == 1 else None
            if fast is not None:
                # One launch for decay + vhat + rank-1 write + read, instead of
                # four passes over the same 0.5 MB that torch.compile cannot
                # merge (chained reductions of different shapes). See
                # lapa/triton_decode.py; the branch below stays the reference.
                cq, sq = pq.cos(), pq.sin()
                c1 = self.wr * cq + self.wi * sq
                c2 = self.wr * sq - self.wi * cq
                qt = torch.stack([torch.cat([c1, c2], -1),
                                  torch.cat([-c2, c1], -1)], 1)
                sd = _state_dtype("SCA2_LSTATE_DTYPE", "fp16")
                st_in = state["s"]
                if st_in.dtype != sd:
                    st_in = st_in.to(sd)
                s, u = fast(st_in, self._damp(self.lam(), 1), kt, qt,
                            vz.to(self.wd), beta)
                return (self._out(u, z_t).to(z_t.dtype),
                        {"s": s, "pos": state["pos"] + 1})
            ktb = kt * torch.cat([beta, beta], -1) if self.bg > 1 else kt
            vhat = torch.einsum("bm,bmj->bj", ktb, s0) / M
            v = vz.to(self.wd)
            if self.cfg.beta_write:
                v = beta * v
            e = v - (vhat if self.bg > 1 else beta * vhat)
            s = torch.addcmul(s0, e[:, None, :], kt[:, :, None])
            cq, sq = pq.cos(), pq.sin()
            wr_, wi_ = self.w_eff(z_t)
            if self.NG == 1:
                c1, c2 = wr_ * cq + wi_ * sq, wr_ * sq - wi_ * cq
            else:
                cg, sg_ = cq[..., None], sq[..., None]               # (B,M,1)
                c1 = self.wr * cg + self.wi * sg_                     # (B,M,NG)
                c2 = self.wr * sg_ - self.wi * cg
            B_ = z_t.size(0)
            if self.NG == 1:
                qt = torch.stack([torch.cat([c1, c2], -1),
                                  torch.cat([-c2, c1], -1)], 1)  # (B,2,2M)
                u = torch.einsum("bam,bmj->baj", qt, s).reshape(B_, 2 * self.dvi) / M
            else:
                NG, w_ = self.NG, self.dvi // self.NG
                qt = torch.stack([torch.cat([c1, c2], 1),
                                  torch.cat([-c2, c1], 1)], 1)      # (B,2,2M,NG)
                sg = s.view(B_, 2 * self.M, NG, w_)                 # (B,2M,NG,w)
                u = torch.einsum("bamg,bmgj->bagj", qt, sg).reshape(B_, 2 * self.dvi) / M
        return self._out(u, z_t).to(z_t.dtype), {"s": s, "pos": state["pos"] + 1}


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
        # (L,) when shared -- the shape sca2's mirror uses, so the float64 gate still loads --
        # and (L, G) when grouped. w = 1: delta at lag 0 at init (o_t = V(z_t)).
        self.G = max(1, cfg.short_groups)
        assert dv % self.G == 0, f"dv={dv} must divide by short_groups={self.G}"
        sh = (L,) if self.G == 1 else (L, self.G)
        self.wr = nn.Parameter(torch.ones(sh))
        self.wi = nn.Parameter(torch.zeros(sh))
        # built in float64: 2*pi/L rounded in float32 breaks the comb's exact cancellation
        self.register_buffer(
            "omega", (torch.arange(L, dtype=torch.float64) * (2 * math.pi / L)).float()
        )
        if cfg.gdn_gate and cfg.gdn_gate_scope == "both":
            self.o_norm = nn.LayerNorm(2 * dv)
            self.gp = nn.Linear(d, 2 * dv)

    @property
    def wd(self) -> torch.dtype:
        return _wd(self.wr)

    def _gemm_dtype(self, ref: torch.Tensor) -> torch.dtype:
        if self.cfg.gemm_dtype is not None:
            return self.cfg.gemm_dtype
        if torch.is_autocast_enabled() and ref.is_cuda:
            return torch.get_autocast_gpu_dtype()
        return self.wd

    def _beta(self, bz):
        """sigmoid(bproj(z)) -> (...,1) when shared, (...,M) when banded."""
        b = torch.sigmoid(bz.to(self.wd))
        return b if self.bg == 1 else b[..., self.bgroup]

    def _out(self, u, z):
        """(..., 2*dv) -> (..., 2*dv). Short head has no kv_dk."""
        if self.cfg.gdn_gate and self.cfg.gdn_gate_scope == "both":
            return self.o_norm(u) * F.silu(self.gp(z))
        return _rms(u)

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
        if self.G == 1:
            c1, c2 = self.wr * cq + self.wi * sq, self.wr * sq - self.wi * cq
            k_re = (torch.einsum("btwl,btl->btw", cw, c1)
                    + torch.einsum("btwl,btl->btw", sw, c2)) / self.L
            k_im = (torch.einsum("btwl,btl->btw", sw, c1)
                    - torch.einsum("btwl,btl->btw", cw, c2)) / self.L
            return torch.cat([torch.einsum("btw,btwj->btj", k_re, e),
                              torch.einsum("btw,btwj->btj", k_im, e)], -1)
        G, dv = self.G, self.dv
        c1 = self.wr * cq[..., None] + self.wi * sq[..., None]                      # (B,T,L,G)
        c2 = self.wr * sq[..., None] - self.wi * cq[..., None]
        k_re = (torch.einsum("btwl,btlg->btwg", cw, c1)
                + torch.einsum("btwl,btlg->btwg", sw, c2)) / self.L                 # (B,T,W,G)
        k_im = (torch.einsum("btwl,btlg->btwg", sw, c1)
                - torch.einsum("btwl,btlg->btwg", cw, c2)) / self.L
        eg = e.view(*e.shape[:-1], G, dv // G)                                      # (B,T,W,G,dv/G)
        re = torch.einsum("btwg,btwgj->btgj", k_re, eg).reshape(*e.shape[:2], dv)
        im = torch.einsum("btwg,btwgj->btgj", k_im, eg).reshape(*e.shape[:2], dv)
        return torch.cat([re, im], -1)

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
        N = C + L - 1
        Fk = torch.cat([cw, sw], -1).to(gd)                                         # (B,KC+L-1,2L)
        Fk = Fk.unfold(1, N, C).movedim(-1, 2)                                      # (B,K,N,2L)
        ek = e.to(gd).unfold(1, N, C).movedim(-1, 2)                                # (B,K,N,dv)
        i = torch.arange(C, device=cq.device)[:, None]
        j = torch.arange(N, device=cq.device)[None]
        band = ((j - i) >= 0) & ((j - i) <= L - 1)                                  # (C,N)
        band2 = ~torch.cat([band, band], 0)      # True OUTSIDE the band: what gets zeroed
        G, dv = self.G, self.dv
        if G == 1:
            c1 = (self.wr * cq + self.wi * sq).to(gd)                               # (B,KC,L)
            c2 = (self.wr * sq - self.wi * cq).to(gd)
            Fq = torch.cat([torch.cat([c1, c2], -1).view(B, K, C, 2 * L),
                            torch.cat([-c2, c1], -1).view(B, K, C, 2 * L)], 2)      # (B,K,2C,2L)
            S = (Fq @ Fk.transpose(-1, -2) / L).masked_fill(band2[None, None], 0)
            o = (S @ ek).to(self.wd)                                                # (B,K,2C,dv)
        else:
            # one spectral mixture per group of value channels: G kernels instead of one.
            # The state, the phases and the write path are untouched and shared.
            cqg, sqg = cq[..., None], sq[..., None]                                 # (B,KC,L,1)
            c1 = (self.wr * cqg + self.wi * sqg).to(gd)                             # (B,KC,L,G)
            c2 = (self.wr * sqg - self.wi * cqg).to(gd)
            f1 = torch.cat([c1, c2], 2).view(B, K, C, 2 * L, G)
            f2 = torch.cat([-c2, c1], 2).view(B, K, C, 2 * L, G)
            Fq = torch.cat([f1, f2], 2).permute(0, 1, 4, 2, 3)                      # (B,K,G,2C,2L)
            S = (Fq @ Fk.transpose(-1, -2)[:, :, None] / L).masked_fill(
                band2[None, None, None], 0)                                         # (B,K,G,2C,N)
            ekg = ek.view(B, K, N, G, dv // G).permute(0, 1, 3, 2, 4)               # (B,K,G,N,dv/G)
            o = (S @ ekg).to(self.wd)                                               # (B,K,G,2C,dv/G)
            o = o.permute(0, 1, 3, 2, 4).reshape(B, K, 2 * C, dv)
        o = torch.cat([o[:, :, :C], o[:, :, C:]], -1).reshape(B, K * C, 2 * self.dv)
        return o[:, :T]

    def _maybe_fast(self):
        import os
        if os.environ.get("SCA2_LAPA_DECODE", "auto") == "naive":
            return None
        try:
            from .triton_decode import short_supported, short_step
        except Exception:                                     # pragma: no cover
            return None
        return short_step if short_supported(self) else None

    def step(self, z_t, h_t, state: State):
        if not hasattr(self, "_decode_fast") and z_t.is_cuda:
            self._decode_fast = self._maybe_fast()
        kh, kz, vz = self.K(h_t), self.K(z_t), self.V(z_t)
        with _no_autocast(z_t.device):
            B = z_t.size(0)
            p = state["pos"].expand(B)
            phi = self._phase(kh, p)
            psi = self._phase(kz, p)
            # RING BUFFER, not concat-then-slice. The window used to be built by
            # `cat([state, new])` and stored back as `[:, 1:]`, which copies the
            # whole window TWICE per token -- 512 KB a layer, 4 MB a token over
            # 8 layers, purely to shift it by one.
            #
            # It is avoidable exactly, not approximately. The read sums over the
            # window index w with cw[w] and e[w] always paired, so applying the
            # same permutation to both leaves u unchanged; a ring with a moving
            # write pointer applies exactly that permutation.
            L = self.L
            if state["c"].shape[1] != L:                      # from prefill/init
                pad = (0, 0, 0, 1)
                # The ring may be held narrower than the arithmetic. c and s are
                # cosines and sines and e is a unit-RMS projection, so all three
                # are bounded and fp16's 10 mantissa bits beat bf16's 8 at the
                # same size -- bf16 buys range nothing here needs. The window is
                # read back into fp32 and every reduction stays fp32; only the
                # STORAGE narrows, which is what multiplies by batch.
                rd = _ring_dtype()
                cw = F.pad(state["c"], pad).to(rd)
                sw = F.pad(state["s"], pad).to(rd)
                ew = F.pad(state["e"], pad).to(rd)
                ptr = torch.full((), L - 1, device=z_t.device, dtype=torch.long)
            else:
                # In place: the ring is a buffer THIS layer allocated in the
                # branch above, so no caller holds it expecting it unchanged --
                # the prefill state was padded into a fresh tensor, not aliased.
                # A caller that forks decode from one state (beam search) must
                # clone it, which is the same contract a KV cache carries.
                cw, sw, ew = state["c"], state["s"], state["e"]
                ptr = state["ptr"]
            fast = getattr(self, "_decode_fast", None)
            if fast is not None:
                # phase, ring write, kappa, read and the RMS in one launch.
                out = fast(kh, kz, vz.to(self.wd), self.theta, self.omega,
                           self.wr, self.wi, state["pos"], ptr, cw, sw, ew)
                return out.to(z_t.dtype), {
                    "c": cw, "s": sw, "e": ew,
                    "ptr": (ptr + 1) % L, "pos": state["pos"] + 1,
                }
            # A TENSOR index, never a Python int. As an int the pointer is a
            # compile-time constant, so torch.compile respecialises the graph on
            # every token, blows the cache and falls back: measured 407 ms a
            # token against 3.5, a 100x regression from what reads like a
            # harmless `cw[:, ptr] = ...`.
            i = ptr[None]
            cw.index_copy_(1, i, phi.cos()[:, None].to(cw.dtype))
            sw.index_copy_(1, i, phi.sin()[:, None].to(sw.dtype))
            ew.index_copy_(1, i, vz.to(cw.dtype)[:, None])
            u = self._read(
                psi.cos()[:, None],
                psi.sin()[:, None],
                cw[:, None].to(self.wd),
                sw[:, None].to(self.wd),
                ew[:, None].to(self.wd),
            )[:, 0]
        return _rms(u).to(z_t.dtype), {
            "c": cw,
            "s": sw,
            "e": ew,
            "ptr": (ptr + 1) % L,   # stays a tensor
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
        if cfg.gdn_gate and cfg.gdn_gate_scope == "concat":
            self.gate_norm = nn.LayerNorm(4 * dv)
            self.gate_proj = nn.Linear(d, 4 * dv)
        elif cfg.gdn_gate and cfg.gdn_gate_scope == "mix":
            self.gate_norm = nn.LayerNorm(d)
            self.gate_proj = nn.Linear(d, d)
        self.mix_norm = nn.RMSNorm(d) if cfg.post_norm else None
        if cfg.layer_scale:
            mix_shape = (d,) if cfg.ls_mix_per_channel else ()
            self.gs_mix = nn.Parameter(torch.full(mix_shape, cfg.ls_mix_init))
            self.gs_ff = nn.Parameter(torch.full((), cfg.ls_ff_init))
        self.ck = cfg.conv
        if self.ck:
            w = torch.zeros(d, 1, self.ck)
            w[:, 0, -1] = 1.0  # identity at init: the conv starts as a no-op
            self.cw = nn.Parameter(w)
        if cfg.init_v2:
            _apply_init_v2(self)

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
        conv_dtype = (torch.get_autocast_dtype("cuda")
                      if z.is_cuda and torch.is_autocast_enabled("cuda") else z.dtype)
        if (self.cfg.long_path == "triton_scan"
                and z.is_cuda and z.dtype in (torch.float32, torch.bfloat16)
                and conv_dtype in (torch.float32, torch.bfloat16) and self.ck <= 8):
            # SiLU must precede both heads' projections; fusing it after K in
            # phase_codes would change the model. Fuse the conv epilogue instead.
            from .triton_conv import causal_conv
            return causal_conv(zz, self.cw.to(z.dtype), self.cfg.conv_silu), zz[:, -(self.ck - 1):]
        zc = F.conv1d(zz.transpose(1, 2), self.cw.to(z.dtype),
                      groups=self.cfg.d).transpose(1, 2)
        # BACK TO z's dtype: LayerNorm stays fp32 under autocast but conv1d does not, and
        # the long head's triangular solve has no bfloat16 CUDA kernel.
        zc = F.silu(zc) if self.cfg.conv_silu else zc
        return zc.to(z.dtype), zz[:, -(self.ck - 1):]

    def _conv_step(self, z_t, buf):
        """Same filter, one token. out = sum_j w_j . window_j, window = [buf ; z_t]."""
        win = torch.cat([buf.to(z_t.dtype), z_t[:, None]], 1)          # (B,ck,d)
        o = (win.transpose(1, 2) * self.cw.squeeze(1).to(z_t.dtype)).sum(-1)
        o = F.silu(o) if self.cfg.conv_silu else o
        return o.to(win.dtype), win[:, 1:]

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
        cat = torch.cat([ul, us], -1)
        scope = self.cfg.gdn_gate_scope if self.cfg.gdn_gate else ""
        if scope == "concat":
            cat = self.gate_norm(cat) * F.silu(self.gate_proj(z))
        mixer_out = self.mix(cat)
        if scope == "mix":
            mixer_out = self.gate_norm(mixer_out) * F.silu(self.gate_proj(z))
        if self.mix_norm is not None:
            mixer_out = self.mix_norm(mixer_out)
        if self.cfg.layer_scale:
            x = x + self.gs_mix * mixer_out
            x = x + self.gs_ff * self.ff(self.fn(x))
        else:
            x = x + mixer_out
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
        cat = torch.cat([ul, us], -1)
        scope = self.cfg.gdn_gate_scope if self.cfg.gdn_gate else ""
        if scope == "concat":
            cat = self.gate_norm(cat) * F.silu(self.gate_proj(z))
        mixer_out = self.mix(cat)
        if scope == "mix":
            mixer_out = self.gate_norm(mixer_out) * F.silu(self.gate_proj(z))
        if self.mix_norm is not None:
            mixer_out = self.mix_norm(mixer_out)
        if self.cfg.layer_scale:
            y = x_t + self.gs_mix * mixer_out
            y = y + self.gs_ff * self.ff(self.fn(y))
        else:
            y = x_t + mixer_out
            y = y + self.ff(self.fn(y))
        return y, {**new, "long": sl, "short": ss, "z_prev": z}

    def state_floats(self) -> int:
        cfg = self.cfg
        return (2 * cfg.M * (cfg.dv + cfg.kv_dk)
                + (cfg.L - 1) * (2 * cfg.L + cfg.dv) + cfg.d)


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
