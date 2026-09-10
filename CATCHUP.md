# Why does GDN catch up? Conjectures, and the tests that separate them

Status: **conjectures**, written 2026-09-07 before `longrun.sh` has run. Nothing
here is measured except the trajectory in the first table. WINNERS.md holds the
resolved results; this file exists so the reasoning is on record *before* the
data arrives and cannot be fitted to it afterwards.

## The fact to explain

Generation 3 (`cdelta`, Mc=190, dv=56) against GDN at matched parameters, n=3 vs
n=3, pycode, one epoch. `python dump_gap.py`:

| tokens | gap (gen3 − GDN) | gap at positions 0–127 | gap at positions 896–1023 |
|---|---|---|---|
| 20M | −0.02 | +0.04 | −0.02 |
| 50M | **−0.39** | −0.15 | **−0.47** |
| 107M | −0.19 | −0.06 | −0.23 |
| 163M | −0.10 | **−0.01** | **−0.12** |

The gap **opens** to ~50M tokens and then **closes**. Late-half slope +0.30 nats
per ln(tokens); extrapolated, the crossing lands near **210M tokens**, well inside
the 537M of `longrun.sh`. The closing is SCA2 slowing down, not GDN speeding up:
gen3's descent goes from −1.07 to −0.52 per ln(tokens) between the two halves,
GDN's from −0.67 to −0.83.

Two constraints on any explanation:

1. **Short range closed first.** At positions 0–127 the lead is already gone
   (−0.15 → −0.01); at 896+ it is still −0.12. Whatever GDN gains, it gains
   first where there is little to store and nothing to forget. So the primary
   cause is **local precision**, not long memory, and "no forgetting" cannot be
   the first-order story.
2. **The learning rate is constant.** Neither arm is converged anywhere on the
   curve; "final" means "at the budget". A closing gap under constant LR can be
   a schedule artifact (conjecture 4), and every other conjecture has to be read
   with that alternative open.

### A note on a second opinion

Another agent's structural analysis was checked against the code. Its param
count (SCA2 +11%) was wrong — matched to 0.13%, `185959` vs `185714`, in GDN's
favour. Its state count (2.85×) used Mc=128 and Md=16; actual 21729 vs 12420
floats (1.75×). Its main argument, "vector state vs matrix state", is false:
`best_layer_cdelta.py` shows the C state is a 380×56 real matrix updated by
exactly the rank-one delta rule `H ← (I − βuuᵀ)H + uvᵀ` with unit-norm keys.
"SCA2 only adds" describes generation 1, not the champion. The one correct point
is that the C head has no global forget gate, and `c_decay` was already tried
(`runs/gc_ablate.jsonl`: 5.023 vs 5.013, no gain) — but at 80M tokens, at the
*peak* of the lead, not where it closes.

## Results (2026-09-08) — read this before the conjectures below

**The long run (n=2): GDN passes gen3 at ~300M tokens and leads by +0.046 at
440–500M, identical on both seeds to 0.001.** The seed moves each arm's level by
0.02 and the gap not at all. GDN leads at every position bucket: +0.09 at 0–127,
+0.03 at 896+. The gap's slope beyond 150M is +0.10/ln(tokens) and flattening.

**By token class (`catchup.sh`, seed 0, big corpus), gap gen3 − GDN:**

| tokens | aggregate | word_rep | word_new | punct | ws | kw |
|---|---|---|---|---|---|---|
| 40M | −0.37 | **−1.19** | **+0.28** | −0.08 | +0.13 | −0.09 |
| 160M | −0.13 | −0.39 | +0.10 | −0.10 | +0.07 | 0.00 |
| 320M | +0.02 | −0.08 | +0.22 | +0.01 | +0.08 | 0.00 |
| 480M | +0.03 | −0.07 | **+0.14** | +0.04 | +0.10 | +0.08 |

Two facts, and they settle the diagnosis:

1. **gen3's entire early lead was exact retrieval.** On `word_rep` it was 1.19
   nats ahead at 40M. GDN then *learned* retrieval and closed that to −0.07.
2. **gen3 never learned `word_new`.** It is +0.2 to +0.3 behind from the first
   eval to the last; the deficit does not move. Same sign, smaller, on `ws`
   (indentation) and late on `punct`/`kw`.

Decomposition at 480M (share × gap): word_new +0.023, punct +0.012, ws +0.010,
kw +0.006, word_rep −0.022; sum +0.029 against +0.030 measured. **The final gap
is the permanent generalisation deficit, minus a retrieval lead that GDN erased.**

**The intervention failed, cleanly.** `cdelta_bp` (content phase bounded to
±π/2) lost 0.15–0.29 nats to gen3 throughout, all of it at long range (+0.22 to
+0.38 at 896+, +0.04 at 0–127). By class at 305M: `word_new` 6.137 vs gen3
6.134 — **unchanged** — and `word_rep` 2.96 vs 2.70. Bounding the phase
destroyed exact retrieval and bought nothing on new words. Killed at 328M.

**So conjecture 1's diagnosis is right and its remedy is wrong.** The C head's
keys are hashes, and the hash *is* the mechanism: it is what makes an identifier
retrievable without interference 800 tokens later. What gen3 lacks is a
*second* kind of memory — metric, for "this context resembles that one" — not a
softer version of the first. The D head (Md=4, 448 floats of state, never
re-tuned since Mc fell 378→190) is the only place such a thing lives today.

Next, in order: **(a) a metric memory beside the C head**, not instead of it —
grow Md, or give the D head GDN-style free keys (the `keyed` head already exists,
`sca2/arch_keyed.py`), funded from the FFN (conjecture 3's shapes); **(b)** β on
the whole write (conjecture 2), because GDN's retrieval eventually matching the
torus's says exact replacement is not what the torus was winning on either.
Conjecture 5 (forgetting) is not needed to explain anything above.

### The hybrid (launched 2026-09-08, `long_hyb.sh`)

`sca2/arch_hybrid.py`: the cdelta C head unchanged, a **Gated DeltaNet head in
the D slot** (fla's reference, the gdn arm's code) emitting the D head's 2·dv
features, funded from the FFN. Shape chosen from the matched grid:

| H | dk | expand | ff | params/layer | D-slot params | D-slot state |
|---|---|---|---|---|---|---|
| 2 | 48 | 1.0 | 244 | 186,003 (+0.02% vs gen3, +0.16% vs GDN) | 61,668 | 5,472 |

Other matched cells (H, dk, ff): (1, 64, 324), (3, 32, 243), (4, 24, 242). The
FFN drop 364→244 is confounded with the head swap in this one arm; if the hybrid
moves, the FFN-only control is conjecture 3's `ff=260, dv=80, Mc=206`.

Label `catch_hyb_s0`, same protocol and log as `catch_gen3_s0` / `catch_gdn_s0`.
Prediction: `word_new` gap to GDN shrinks toward 0 while `word_rep` keeps gen3's
lead. Iso at float32 (`python -m sca2.iso hyb --self --dtypes float32`); float64
is capped at ~1e-7 by fla's internal float32 cast, as for the gdn arm.

### Hybrid and gated read-out, final (2026-09-08 evening)

| arm | gap to GDN, 440–500M | word_new | word_rep | ws | punct |
|---|---|---|---|---|---|
| gen3 | +0.046 | +0.13..+0.21 | −0.07 | +0.10 | +0.04 |
| **hybrid** (GDN head in D slot) | **−0.014** | +0.11..+0.16 | **−0.06..−0.14** | −0.01..−0.07 | 0.00 |
| gen3 + gated read | +0.090 | +0.17..+0.24 | **+0.04** (lead lost) | +0.11 | +0.07 |

**The hybrid reaches parity with GDN at 500M** (n=1; the long run's gap was
seed-invariant to 0.001, so n=2 — queued as `catch_hyb_s1` — should settle it).
Its gain over gen3 (~0.06) is better retrieval, indentation and punctuation —
**not** `word_new`, which it barely moves. It runs at 59.6k tok/s against GDN's
72k and gen3's 76k: the D-slot GDN head is fla's naive Python reference.

**The gated read-out is dead.** It did nothing for `word_new` and destroyed the
`word_rep` lead. The "unit-RMS ungated read injects noise" story is refuted as
stated. (`sweep_pycode`'s earlier −0.10 for `gatedread` was on the additive layer
with unmatched parameters.)

**The `word_new` deficit is uniform.** `diag_wordnew.py` splits it by the target's
training frequency and by position quarter, gen3 vs GDN:

| target frequency | share | gen3 | GDN | Δ |
|---|---|---|---|---|
| top-300 | 19% | 4.32 | 4.05 | +0.27 |
| 300–1500 | 32% | 5.54 | 5.24 | +0.30 |
| tail | 49% | 6.61 | 6.32 | +0.29 |

The same +0.27..+0.30 in every frequency band **and in every position quarter,
including the first 256 tokens**. It is not about rarity and not about context
length. Invariant so far to: key geometry (bp), D head type (hyb, −0.06 of it),
FFN size (364 / 252 / 244), output gating (gr). Every arm containing the C head
has it; GDN does not.

Two readings remain, and `catch_noc_s0` (Mc=2, the C head crippled, ff matched)
separates them: **harm** — the C head's read is noise on tokens with nothing to
retrieve and the layer cannot reject it (then the deficit vanishes without the C
head); or **missing capability** — GDN has something SCA2 lacks, most plausibly
precise short-lag access (GDN's 4-tap convs on q/k/v; the C head's rope kernel
blurs neighbours, κ[1]=0.81), and the deficit persists at Mc=2. `conv` in
LayerCfg is wired only into the `gc` variants, so a conv arm on cdelta needs code.

### The C head is the cause (2026-09-09 morning)

**`catch_noc_s0` — gen3 with the C head crippled (Mc=2, ff=460 to match):**

| | aggregate vs GDN | word_new | word_rep | punct | ws |
|---|---|---|---|---|---|
| gen3 | +0.05 | **+0.28** | −0.07 | +0.04 | +0.10 |
| no C head | +0.85 | **−0.05..+0.09** | +1.8 | +0.5 | +0.4 |

Without the C head the layer is 0.85 nats worse — retrieval is gone, and so is the
short-range modelling the other classes ride on — **but the `word_new` deficit
disappears entirely.** The C head does not merely fail to help on tokens new to
the window; it *hurts* them, by ~0.28 nats, in every arm that carries it. FFN size
is ruled out as the explanation (GDN has no deficit at ff=260; gen3 has it at 364).

**Proposed mechanism, now testable: `_rms()` erases the match signal.** A hash
read on a novel key is a small random mixture; on an exact repeat it is a
full-size value. RMS normalisation maps both to unit scale, the `mix` downstream
is linear, and the input-driven gate (`gated_read`, failed above) looks at `z`,
not at the read — so nothing in the layer can tell "found" from "not found".
`cdelta_raw` keeps the magnitude (learned per-feature scale, +112 params).
Running as `catch_raw_s0`. Prediction: `word_new` deficit drops toward zero with
`word_rep` intact; that alone is worth ~0.045 aggregate, the whole gap.

**Hybrid at n=2, seed-paired against GDN:** −0.014 (s0) and −0.021 (s1) over
440–520M. Mean **−0.017**: ahead of GDN, where gen3 is +0.046. Modest, same sign
twice, and its word_new deficit is intact — so the hybrid and the raw read are
orthogonal gains if the raw read works.

### The short C head (2026-09-09, `sca2/arch_short.py`, `chead_numpy.py`)

Where the time goes at the champion's shape (eager fwd+bwd, B=8 T=1024, ratios):
C head 52%, **polar D head 44%**, FFN 4%. The D head is a 64-chunk sequential
scan for 448 floats of state, and it has never been ablated.

Replacement, inside the programme: a **second C head on the DFT grid**,
`ω_m = 2πm/L`, whose Dirichlet comb is an *exact* delta at every lag but periodic
in L — so the window is the last L tokens and the state is a ring buffer of the
L−1 previous writes, not an accumulator. Same write (key from h, value from z),
same read, same kernel `κ(t,s)` as the long head; only grid and window differ.
With θ=0 it *is* a learned L-tap causal filter (`w_m = Σ_n taps_n e^{inω_m}`);
θ≠0 makes the taps content-dependent. No delta rule (nothing accumulates), no
forgetting to manage, positions enter mod L (float32-safe at any length),
O(T·L·(L+dv)) with no T×T kernel and no scan. Verified in float64 against the
closed-form window sum, the θ=0 delta, token-by-token decode and a chunk split
(`python -m sca2.arch_short`); iso OK for `cshort` and `cshort_raw`.

Shape: L=16, 9,264 params against the D head's 30,784; ff 364→448 rematches
(186,027). Running as `catch_short_s0`, queued behind `catch_raw_s0`. What it
measures at once: whether local syntax (ws, punct, kw) improves — the conv
hypothesis, in characteristic-function form — and what the polar D head was
worth at medium range (positions 16–200), since it is the first ablation of it.
Timing of the replacement itself is below.

### Raw read refuted; the evidence problem (2026-09-09, 10:30)

`catch_raw_s0` tracked gen3 to within eval noise on every class through 254M
tokens (word_new +0.18 vs +0.18 at 240M) and was stopped. **Keeping the read's
magnitude changes nothing.** Measured why, on the gen3 checkpoint: the C head's
read norm *before* RMS is the same distribution on new and repeated words
(layer 0: 2.86±0.39 vs 3.02±0.32; corr(norm, loss) ≤ 0.23). The magnitude never
carried the "found / not found" signal; RMS erased nothing. And the deficit is
the same in the first quarter of the window as in the last, so it is not a noise
floor growing with the number of writes either: the head hurts new words
uniformly, whatever it holds. The reading that fits: on a novel key the read's
*direction* is arbitrary but deterministic, and downstream layers have learned
to treat it as a feature.

A gate therefore needs evidence that the read itself carries and that the query
does not. **Key verification** (`CHeadDeltaKV`, variant `cdelta_kv`): the write
stores a copy of its own key beside the value, `e_s = [V(z_s); Kv(h_s)]`; the
read returns both; `m_t = cos(key read, Kv(z_t))` is the evidence, `g = σ(a·m + b)`
the gate on the value read. A true match returns its own key; a mixture does not.
+2·d·dk+2 params (dk=16), iso by construction; `cdelta` itself re-verified
bit-identical after the hook refactor.

**Running (user's call: combination first, single-change ablation after):**
`catch_shortkv_s0` = key-verified long head + short dft head, ff=440 (186,021),
then `catch_short_s0` alone. Hooks in `CHeadDelta` (`_value`, `_out`) exist so
subclasses can extend what is stored and what leaves the head.

### Where the noise comes from, and the damped head (2026-09-09, 11:00–12:00)

**The read kernel, reconstructed on the gen3 checkpoint** (`|κ(t,s)|²` over all
writes s ≤ t, positions ≥ 64):

| layer | top-1 write's share | effective # writes read | P(top write's key = current token) |
|---|---|---|---|
| 0 | 0.9% | 435 | 2% (reads itself, lag 0: 40%) |
| 1 | 4.3% | 94 | **97%** |
| 2 | 1.3% | 252 | 6% |
| 3 | 2.5% | 197 | **87%** |

Layers 1 and 3 do induction and **address correctly 97% of the time**; the
addressed write carries **4% of the read**, the other 96% is 100–200 unrelated
writes. Interference floor √(N/M) of a non-forgetting superposition memory; the
delta rule does not touch it (it cleans what a key held *before* its write, not
the cross-talk that arrives *after*). GDN's state is no bigger — it stays clean
because it forgets. This is the mechanism behind both the word_rep plateau and
the word_new harm.

**Decay switched on at inference, gen3 checkpoint, scalar λ, no retraining:**

| 1/λ | val | word_new | word_rep | eff. # writes (layer 1) | top-1 share |
|---|---|---|---|---|---|
| ∞ | 2.310 | 5.712 | 2.000 | 79 | 4.9% |
| 512 | 2.391 | 5.658 | 2.204 | 52 | 8.5% |
| **128** | 2.522 | **5.631** | 2.492 | 19 | 16.3% |

word_new improves by 0.08 (a quarter of the deficit) with a knob the model never
trained with; word_rep collapses because the model relies on undamped retrieval.
Both directions exist; only training decides the net.

**`CHeadDeltaDamp`** (`sca2/arch_damp.py`, variant `cdelta_damp`): learned
per-mode decay `λ_m = softplus(a_m)`, memories `1/λ` init log-uniform in
[64, 4096] tokens (gen3 nearly nested), `LAM_MAX = 1/8`. Decay on the *state*,
codes stay unit-modulus so the delta rule keeps its constant normalisation. The
chunked closed form folds `e^{−λ(t−s)}` into write codes scaled by `e^{+λ(s−t₀)}`
and read codes by `e^{−λ(t−t₀)}`, chunk-relative. Verified to 1e-15 against a
token-loop reference (chunked, single-chunk, decode), `λ→0` reproduces cdelta
bit-for-bit, iso OK. Numpy semantics: third head of `chead_numpy.py`. `long_damp.sh`
is ready and **not launched** — queued only if the short/verification ablation
leaves it worth the GPU time. +190 params/layer (186,149 at ff=364).

### Combo final: parity with GDN, and the gate is real (2026-09-09, 12:40)

`catch_shortkv_s0` (key-verified long head + short dft head, 186,021 params, same
speed as gen3): **val 2.6501 at 535M; gap to GDN over 440–516M −0.017**, against
gen3's +0.052 and the hybrid's −0.014 — at 0.98× gen3's layer cost and 1.9× the
hybrid's decode speed. No crossover: the gap sits at −0.02..−0.04 from 260M on.

| gap to GDN, 440M–end | aggregate | word_new | word_rep | punct | ws | kw |
|---|---|---|---|---|---|---|
| gen3 | +0.052 | +0.176 | −0.026 | +0.039 | +0.129 | +0.066 |
| **combo** | **−0.017** | **+0.061** | **−0.088** | −0.014 | +0.042 | +0.046 |

Position profile (buckets of 128), combo: +0.037 +0.018 −0.013 −0.031 −0.023
−0.042 −0.041 −0.037 — behind GDN on the first 256 tokens, ahead beyond; gen3 was
+0.10 → +0.03, behind everywhere. The combo and the hybrid have the *same* profile
to 0.005 per bucket: two different routes to one limit.

**The gate discriminates where induction happens** (`diag_gate.py`, 20 val batches):

| layer | slope a | gate on word_rep | gate on word_new |
|---|---|---|---|
| 0 | 0.12 | 0.24 | 0.24 (a constant attenuator: the model turned this head down) |
| **1** | **3.88** | **0.59** | **0.35** |
| 2 | 1.81 | 0.42 | 0.43 |
| 3 | 2.69 | 0.71 | 0.67 |

Layer 1 is the layer whose dominant write's key was the current token 97% of the
time; its gate opens on repeats and closes on new words, exactly the designed
behaviour. Layer 0 (the non-induction "reads itself" head) is gated to 0.24
regardless — the gate found a second use as a learned global scale on a head that
was hurting. Layers 2–3 barely use it.

Remaining against GDN: the first 256 positions and half the word_new deficit.
Both are where the long memory has nothing to offer and only adds noise — the
damping's target. Queued on the short-head base (`long_damp_queue.sh`):
`catch_shortdamp_s0` (A: damped half-persistent long head + short head, no gate)
and `catch_shortdampkv_s0` (B: A + gate). A − short = the damping; B − A = whether
the gate still earns its keep once the noise is treated at the source.
Prediction on record: `short` alone keeps the syntax gains and not the word_new
halving; the gate does not survive B − A.

### A: the damped layer beats GDN to the end (2026-09-09, 16:40)

`catch_shortdamp_s0` — long head with the low-frequency half of its modes
persistent and the other half damped, short dft head in the D slot, no gate,
186,217 params, **78.8k tok/s (fastest arm of the campaign)**:

| gap to GDN, 440–516M | | word_new | word_rep | punct | ws | kw |
|---|---|---|---|---|---|---|
| gen3 | +0.052 | +0.176 | −0.026 | +0.039 | +0.129 | +0.066 |
| short | +0.008 | | | | | |
| combo (short + gate) | −0.017 | +0.061 | −0.088 | −0.014 | +0.042 | +0.046 |
| **A (short + damping)** | **−0.034** | +0.083 | **−0.126** | −0.020 | **+0.005** | +0.015 |

Final val 2.6224 against GDN's 2.6694. Over 240–471M the mean gap was −0.066; the
gap's slope beyond 200M is +0.08/ln(tokens) against gen3's +0.30 — closer to a
constant offset than to a closing. By position: +0.032 on the first 128 tokens,
−0.025 → −0.068 beyond. A second seed would turn this from an observation into a
result (the long run's gaps reproduced across seeds to 0.001). Plot:
`plot/A_vs_gdn.png`.

**What the damping learned — and it is not what was initialised.** Damped modes
were initialised with memories 1/λ log-uniform in [64, 4096] tokens. Trained:

| layer | median memory | p90 | modes AT the 1/8 cap (of 95) |
|---|---|---|---|
| 0 | 8 tokens | 26 | 71 |
| 1 | 8 | 20 | 71 |
| 2 | 8 | 24 | 53 |
| 3 | 8 | 22 | 54 |

The optimiser drove the damped half of the spectrum to **forget as fast as it was
allowed**: memories of 8–26 tokens, most modes pinned at the cap. So the trained
long head is really two heads sharing one accumulator: 95 persistent modes (the
long memory) and 95 fast modes (8–26 tokens) — which is exactly the role the polar
D head used to play (a smooth, forgetting, medium-short memory), now absorbed into
the spectrum of the Fourier head. The reviewer's worry ("don't let the long head
turn into a second short head") is what the model chose to do with half of it, and
it paid: every class improved over `short`, retrieval included (−0.126, the best
of any arm) — less noise in the read helps repeats too.

Follow-ups, in order: **(1) lift the cap** — `cshort_damphf` (λ ≤ 60/CTX = 0.47,
memories down to ~2 tokens; the folded closed form overflows float32 beyond
λ·CTX ≈ 88, stress-tested at the cap: prefill/step agree to 1e-6 in float32,
6e-15 in float64), `long_fast.sh` ready, not launched; **(2) the persistent
fraction** as a knob (50% was a prior, not a measurement); **(3) seed 1 of A**.
B (`catch_shortdampkv_s0`, A + gate) runs until ~18:50: with word_new still at
+0.083 for A and the gate's whole effect on that class, B − A is the open question.

### B: gate on top of damping (2026-09-09, 18:40)

`catch_shortdampkv_s0` (A + key verification, ff=440, 186,211 params): **val 2.6132
at 515M; gap to GDN over 440–515M −0.048**, A −0.034, combo −0.017, short +0.008,
gen3 +0.052. B − A over that window: −0.014 (word_new −0.030, word_rep −0.018,
kw −0.014, punct −0.009, ws +0.004); over 160–361M it was −0.031. Eleven
consecutive points of the same sign, but the late ones sit inside a single eval's
noise. Read: the gate adds ~0.015–0.03 on top of damping — real, small, and it
costs +10% of the layer stack, +13% in decode and 6k floats of state. Same
trajectory shape as A (both +0.08/ln(tokens) beyond 200M): it shifts the curve,
it does not change its dynamics. B − GDN by position: +0.03 on the first 128
tokens, −0.04 → −0.08 beyond.

Gate by class on B's checkpoint (`diag_gate.py`, run with `SCA2_CTX_CHUNK=128`:
damped heads overflow float32 at CTX·λ > 60): layer 1 g = 0.66 on word_rep vs
0.41 on word_new (combo: 0.59 vs 0.35) — the evidence is slightly *sharper* once
the read is damped, as guessed; layer 0 again a flat attenuator (0.27). B's damped
modes learned the same thing as A's: median memory 8 tokens, 49–77 of 95 modes at
the 1/8 cap.

Queued: `catch_shortfast_s0` (A with the cap lifted to λ ≤ 0.47, running since
18:40), then `catch_shortdamp_s1` (seed 1 of A, `long_A_s1.sh`).

### The 5-hour convergence run: A holds to 1.4B tokens (2026-09-10, 07:15)

`long5h.sh`, XL corpus (1.57B tokens, single pass, new val split — absolute values
not comparable to the 537M runs), 5 h per arm, seed 0. A: 1423M tokens at 79.1k
tok/s, final val 2.3262. GDN: 1333M at 74.1k, final 2.3550. Plot: `plot/long5h.png`.

| window | median gap A − GDN | mean |
|---|---|---|
| 300–500M | −0.043 | −0.047 |
| 500–700M | −0.053 | −0.055 |
| 700–900M | −0.051 | −0.048 |
| 900–1100M | −0.007 | −0.001 |
| 1100–1300M | −0.043 | −0.036 |

Over 300–1333M: mean −0.032, sd 0.036, slope **+0.042/ln(tokens)** — inside the
noise (the sd is of autocorrelated points). The two positive excursions are
identifiable eval spikes, not trend: A's residual +0.045 at 996M (train 2.14) and
+0.10 at 1328M (its largest of the run; back to 2.315 one eval later), and GDN's
favourable −0.042 at 978M. Outside those, A sits at −0.04..−0.05 from 300M to 1.3B.
Both arms descend at the same pace over the last 400M (−0.07 each). **The crossover
extrapolated from the 537M run (0.6–1.1B) did not happen.** At equal wall-clock
(5 h) A has seen 7% more tokens and ends −0.03..−0.035 below GDN (last clean evals
2.315/2.326 vs 2.355/2.357).

B (`l5_B_s0`) started 06:40, ends ~11:45.

### B at 1.4B, and the standalone file (2026-09-10, 11:45)

`l5_B_s0` final: 1272M tokens, val 2.3180, 73.1k tok/s. Medians over 300M–1.27B:
B − GDN −0.062, A − GDN −0.044, **B − A −0.021**; over the last 270M, B − A −0.024.
The gate's ~0.02 holds to 1.3B tokens, at 7% lower throughput and +13% decode cost:
still an option, not the default. Plot: `plot/long5h.png`.

**`laplace_attention.py`** — the self-contained, dependency-free layer (config
dataclass, `prefill` / `step` / `init_state`, both long-head paths, working-dtype
policy: state and codes/Gram/solve in fp32, the two big GEMMs follow autocast, short
head fp32). Its self-test loads the repo's `cshort_damph` weights and matches it to
9e-16 in float64 on both paths and in decode; float32 decode vs prefill 3e-7. Not yet
in WINNERS.md: that file's bar is n=3, and A is n=1 (at 1.4B tokens).

### Copy capacity at d=128, and the short head as a banded GEMM (2026-09-10 afternoon)

`lapa/benchmarks/copy.py`, 2 layers, lengths 32–512, 4000 steps, token accuracy /
exact-string at the end (`plot/copy_d128*.png`, `runs/copy_d128.jsonl`):

| arm | state/layer | L32 | L64 | L128 | L256 | L512 |
|---|---|---|---|---|---|---|
| attention (ceiling) | KV ~525k | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LapA M=380 | 44k | .996/.85 | .996 | .991/.20 | .981 | **.956** |
| LapA M=190 | 22.7k | .992/.78 | .989 | .979/.07 | .948 | .870 |
| GDN 3×80 | 21.4k | .999/.97 | .994 | .965/.02 | .813 | .515 |
| GDN 3×60 | 12.4k | .999/.97 | .995 | .963 | .792 | .482 |
| LapA M=95 | 12.1k | .981/.54 | .978 | .940 | .861 | .707 |

Two error regimes: GDN is near-exact until its state saturates, then falls off a
cliff; LapA has a small per-token floor (0.5–1.9%, ∝ 1/M — the superposition read's
interference) and no cliff. Crossover in exact-string rate at L=64. Throughput
scales linearly in M (243k / 172k / 114k tok/s for M = 95/190/380; GDN 3×80 120k).

Controls on the L=32 floor, one M=190 arm each: **θ=0: no change** (content phase
neither helps nor hurts copy); **no damping: no change at L=32, +0.08 at L=512**
(0.95 vs 0.87 — the damped half costs range in long copy; the persistent fraction
should be learned, not fixed at 0.5); zero-shot read SNR at lag 33 is ~0.14 with or
without the delta rule and with or without damping — the floor is the rope kernel's
breadth plus training length, not a knob. Delta-rule and rope-base arms pending.

**Short-head window 64**: exact copy at L=32 in 250 steps and at L=64 in 500 (the
Dirichlet comb; nothing to learn but one phase per mode), then — via composition
across the two layers (receptive field 2×63) — L=128 exact 0.96 and L=256 0.36 at
2250 steps where the base had 0.02 / 0.00. The long head starts ~500 steps later
(no easy lengths left to bootstrap on) and catches up. Cost was −40% tok/s with the
unfolded O(T·L²) read; **rewritten as banded GEMMs** (`ShortHead._banded`: chunks
of C=L queries × C+L−1 extended keys, all chunks in one batched matmul, band mask),
exact to 1e-15: L=64 now costs 1.1× L=16 (15.2 → 3.1 ms; unfolded vs banded 0.21).

### Speed pass on Laplace Attention (2026-09-10, in progress)

Profile of A's layer under compile (B=8, T=1024, fwd+bwd): long head 65%, short
head 12%, FFN 11%; zero graph breaks. The long head's time was NOT in GEMMs (7 ms)
but in **2,302 kernel launches per iteration**: 158 generic `add`, 144 `Fill`, 77
`Memcpy` — the autograd signature of slicing a batched tensor inside a Python loop
(each `X[:, k]` backward = a full-size zero fill + copy).

Done, exact to 1e-15 against the token-loop reference, iso OK with the batched path
exercised (`SCA2_CTX_CHUNK=32`):
1. **Chunk-batched prefill** (`CHeadDeltaDamp._prefill_batched`): scaled codes, Gram,
   its triangular inverse `W = A⁻¹` and the intra-chunk kernel computed for all full
   chunks in one call; the sequential loop carries only the state — 5 GEMMs per
   chunk (`r = Rq·S`, `e = W·(v − β r)`, `o = K2·e`, `o += Cq·S₄`, `S += Pw·e`) with the
   state operands stacked so each role is one GEMM. Ragged tail through the old path.
2. **`unbind` before the loop** instead of slicing inside it: 2,302 → 590 launches.
3. Short head read rewritten as contractions on the unfold views (no (B,T,W,L)
   intermediates).
4. `SCA2_LONG_PATH=batched|chunk` knob and `python -m sca2.autotune`, a blocked
   (within-round ratio) timer over path × chunk size that prints the recommended
   setting for the GPU at hand — every number below the shared-GPU line must be
   redone with it once the GPU is idle.

**Clean numbers, idle GPU, blocked design** (`python -m sca2.autotune`, 5 rounds,
round sd 0.3–3%), long head fwd+bwd B=8 T=1024:

| path | CTX 64 | CTX 128 | CTX 256 |
|---|---|---|---|
| chunk (old) | 25.7 ms | 12.4 | 11.4 |
| **batched** | 10.2 | **8.9** | 11.8 |

Batched at 128 is **1.39× faster** than the old path at its own best (12.4 → 8.9 ms).
Recommended `SCA2_LONG_PATH=batched SCA2_CTX_CHUNK=128` on the RTX 2070. Four-layer
stacks, fwd+bwd, blocked: GDN 88.3 ms, gen3 71.2 (0.81), LapA old path 68.4 (0.77),
**LapA batched 62.6 (0.71)** — the layer stack is now 1.41× faster than GDN's (was
1.29× for gen3). **bf16 autocast is 2.7× slower on this GPU** (Turing has no bf16
tensor cores): that lever is for the Spark. The standalone file's bf16 path
deviates from fp32 by 2.6e-3 relative (big GEMMs in bf16, everything else fp32).

## Conjectures, ranked by belief (written before the results)

### 1. Hash keys vs metric keys  ← tested first

The C head's key is `[cos p ; sin p]/√M` with content phase `K(h)·θ` per mode.
`θ` trains to |θ| = 1.8..11.4, so two inputs whose `K(h)` differ slightly land
at phases several turns apart: the key overlap `cos(Δ)` is a **hash** of the
content. Hashing is ideal for retrieving an *exact* repeat — which code is full
of, and which explains the fast start — and useless for "this context resembles
that one". GDN's L2-normalised dot-product keys are a **metric**: nearby contents
share their memory. Late in training, what is left to learn is soft
generalisation between similar contexts, and a hash cannot do it.

**Prediction.** The catch-up lives on tokens where retrieval cannot help: a word
seen for the **first time** in the window (`word_new` in `sca2/tokclass.py`).
On exact repeats (`word_rep`) the torus should hold its lead and lose it last.

**Tests.**
- Zero extra design: `--class-eval` on `pretrain.py` logs the loss by token class
  at every eval; `dump_catchup.py` prints the gap per class and its late slope;
  `diag_tokclass.py` reads saved checkpoints by class × position.
- Intervention: `cdelta_bp` bounds the content phase, `c = b·tanh(K(h)θ/b)`.
  `cos(Δc)` is monotone in |Δc| only on [0, π], so **b = π/2** is the largest
  bound that makes the overlap a genuine similarity; `cdelta_bp2` (b = π) wraps
  once and is the intermediate arm. Same Gram, same solve, same read, same
  parameter count (743836 in 4 layers, identical to gen3); iso-checked
  (`python -m sca2.iso cdelta_bp cdelta_bp2 --self --freq rope --theta-scale 0.5`).
  Confirmation would be a **less positive late slope on `word_new`** with
  `word_rep` not worse. A worse endpoint with a better slope is still a
  confirmation of the mechanism (and a shape question).

Both are in `catchup.sh` (~2 h 50). The class split is informative on its own:
on the gen-2 checkpoint the loss is 5.8 nats on `word_new` against 2.5 on
`word_rep`, i.e. the two populations are nothing alike, and the aggregate
averages them.

### 2. Partial erasure, gain 1/β

β multiplies only the erase term and trains to ≈0.5, so a rewrite removes half
the old value and the stored value converges to `v/β`, not `v`. GDN's β on the
whole write allows exact replacement. Early, interference noise is negligible
against everything still to be learned; late, it is the floor.

**Test.** β on the whole write, DeltaNet-style (gives up the exact nesting of
generation 1, which no longer matters). One layer config, one seed, read the
late slope. Untested by `catchup.sh`; next in line.

### 3. Parameter allocation

At equal total, SCA2 puts 50% in the FFN (93,676 of 185,959), GDN 36%. Token
statistics live in the FFN and are learned early; associative capacity lives in
the mixer and pays late. "Fast start, then stalls" is what an FFN-heavy split
produces.

**Test.** Move budget from `ff` into the mixer at matched parameters, no code
change. Shapes computed against 185,959:

| ff | dv | Mc | params |
|---|---|---|---|
| 260 | 56 | 394 | 185,955 |
| 260 | 64 | 332 | 186,025 |
| 260 | 80 | 206 | 185,903 |
| 300 | 72 | 190 | 185,895 |

Read the slope, not the endpoint. Cheapest test on the list.

### 4. Constant-LR dynamics

The Dirichlet init hands SCA2 a clean positional address at step 0, so it eats
the easy gains before GDN has learned its keys; once they are gone both sit at
the LR's noise floor and GDN, with the freer parameterisation, has more runway.
Not exclusive of 1–3: it is the mechanism that makes them visible.

**Test.** Cosine schedule on the existing 177M budget, both arms. If the closing
disappears it was an artifact; if it persists it is structural. `pretrain.py`
has no schedule yet.

### 5. No global forgetting

The correct point above. Ranked last because the short range closed first,
which forgetting does not explain, and because `c_decay` failed — but that null
was read at the peak of the lead. It becomes a candidate again only if the long
run shows the 896+ bucket closing too.

## Protocol for the next session

1. `bash catchup.sh` — it waits for any running `pretrain.py` first, so it can be
   queued behind `longrun.sh` (which ends around 04:00). gen3 and GDN re-run with the class breakdown (their
   aggregate val must reproduce `shape_dv56_s0` / `gdn4_s3` to eval noise, which
   is the check that `--class-eval` changed nothing), then `cdelta_bp` and
   `cdelta_bp2` as n=1 screens.
2. `python dump_catchup.py` for the per-class gap slopes;
   `python diag_tokclass.py runs/ck_catch_*.pt` for endpoints by class × position.
3. `longrun.sh` was already running (started 19:22 on 2026-09-07) when the class
   breakdown was written, so it answers *when* only; the class data comes from
   `catchup.sh`. A future long run should add `--class-eval` to `COMMON` — it is
   free. (Never edit a `.sh` that bash is executing: it re-reads by byte offset.)

**What would refute conjecture 1:** the gap closes as fast on `word_rep` as on
`word_new`, or `cdelta_bp` shows no slope change on `word_new`. Then go to 3
(cheapest) and 2.

**Rules carried over from WINNERS.md.** n=1 screens rank, they do not measure.
Nothing is a verdict before 90% of an epoch. The bp arms were designed before
seeing any class data, but the *choice* to look at `word_new` was made after
seeing the position profile — so a confirming class split is evidence for the
mechanism, not a headline number until re-measured at n=3.

## Files

- `sca2/tokclass.py` — token classes; `split_repeat` marks a word target as
  `word_rep` iff the same id occurs earlier in the window.
- `pretrain.py --class-eval [--bpe pycode_bpe16k]` — per-class loss in every
  JSONL record (`"cls"`); default behaviour unchanged.
- `sca2/arch_cdelta.py` — `CHeadDeltaBP` (π/2), `CHeadDeltaBP2` (π); variants
  `cdelta_bp[_cc]`, `cdelta_bp2[_cc]`; bound overridable with `SCA2_PHASE_BOUND`.
- `diag_tokclass.py`, `dump_catchup.py`, `dump_gap.py`, `catchup.sh`.
