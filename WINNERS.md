# Lineage

Every generation is a self-contained Python file that reproduces the repo's fast
path in float64 and carries its own measurements. A file is only added when a
result is **resolved at n=3** — not on a promising screen.

Common protocol unless a file says otherwise: pycode, 1024-token blocks, 4
layers, `d=128`, one full epoch of 177.4M tokens, `freq=rope`, params matched to
0.07%, val loss in nats, arms interpolated to a common token count because each
arm's last eval lands wherever the 120s clock put it.

| gen | file | shape | val (n=3) | vs GDN (n=6) | speed vs GDN | what changed |
|---|---|---|---|---|---|---|
| 1 | `best_layer.py` | Mc=378, dv=32 | 2.9057 ± 0.0356 | +0.1126, p=0.0095, **behind** | not remeasured | `v3polarflat`: additive phase-indexed write |
| 2 | `best_layer_cdelta.py` | Mc=378, dv=32 | 2.7482 ± 0.0147 | −0.0449, p=0.102, unresolved | not remeasured | error-correcting C write (delta rule) |
| **3** | **same file, new defaults** | **Mc=190, dv=56** | **2.7078 ± 0.0132** | **−0.0854, p=0.0101, ahead\*** | **x1.08** | shape retrade at matched params — *no code change* |

Generations 1 and 2 say "not remeasured" because throughput here is only
measurable in a blocked same-session design (below) and only generation 3 has been
put through one. Their old figures were withdrawn, not carried forward.

Reference, same protocol: **Gated DeltaNet** at matched params (0.16% *fewer*, in
its favour), 2.7931 ± 0.0536 at **n=6**. It runs six seeds because its seed sd is
4x generation 3's and it therefore supplied ~90% of the variance of every
comparison against it — a GDN seed buys ~25x more resolution per GPU-hour than one
of ours, which is the opposite of the instinct to shore up your own arm.

**\*Read the asterisk before quoting the headline.** p=0.0101 is uncorrected and
generation 3 was the best of **five** shapes screened (dv=24/40/48/56/64).
Bonferroni over five puts α at 0.01, whose critical t at dof=6.06 is 3.683; the
observed t is 3.685. It clears multiplicity **by 0.002**. The defensible claim is
"ahead of GDN, at the edge of significance once the shape search is accounted
for" — not "beats GDN". More seeds of generation 3 barely move it (t 3.69 → 3.79
at n=6, its own variance being already negligible); **GDN at n=12 gives t≈4.96**
and is the only cheap way to settle it.

Each file verifies itself, generation 3 at both shapes:

```bash
python best_layer.py           # == sca2 v3polarflat
python best_layer_cdelta.py    # == sca2 cdelta at Mc=190 AND at Mc=378,
                               #    and nests generation 1 at beta=0 (0.0e+00)
```

Generation 2 nests generation 1 exactly (`beta = 0` gives a bit-for-bit identical
function, measured 0.0e+00), which is what makes a difference between them
attributable to the write rule rather than to some other change.

## What is actually claimed

Two effects, both resolved, and they are independent of each other:

1. **The mechanism is worth −0.111 to −0.118 nats** at matched `theta` init.
2. **The shape is worth a further −0.040**, at matched parameters and *no code
   change* (`Mc` 378→190, `dv` 32→56).

Everything else below is either unresolved or explicitly marginal. "Resolved"
means |t| beats the real two-sided 95% t critical value for its dof — **2.9 to 4.3
at n=3, not 2**. Single estimator now: interpolation to a common 168.7M tokens.

| comparison | Δ | p | resolved |
|---|---|---|---|
| **gen 3 − GDN** | **−0.0854** | **0.0101** | **yes, but marginal under multiplicity — see \*** |
| **gen 3 − gen 2** | **−0.0403** | **0.024** | **yes** — shape |
| **gen 2 − additive(θ.02)** | **−0.1111** | **0.004** | **yes** — mechanism |
| **gen 2(θ0) − gen 1** | **−0.1179** | **0.008** | **yes** — mechanism |
| **gen 1 − GDN** | **+0.1126** | **0.0095** | **yes** — gen 1 was behind |
| gen 3 − gen 2(dv=48) | −0.0148 | 0.62 | no — dv 48 and 56 indistinguishable |
| gen 2(dv=48) − GDN | −0.0706 | 0.086 | no |
| gen 2 − GDN | −0.0449 | 0.102 | no |
| gen 2(θ.02) − gen 2(θ0) | −0.0398 | 0.053 | no |
| additive(θ.02) − gen 1 | −0.0467 | 0.140 | no |

### The paired estimator is retired

`dump_cdelta.paired` averaged 6 points over the last 30M tokens and paired by
seed. It was co-equal with the endpoint reading, and a claim needed both. **It does
not replicate and it is no longer used.**

Its between-seed sd looked 8x tighter than the endpoint's on the `dv=48` arm
(0.005 vs 0.043), which is why it was trusted enough to re-read the n=1 screens
with. On `dv=56` its residuals spread 0.032, six times wider. Against GDN it
predicted −0.083 where n=6 measured −0.038.

Two causes, both now understood. Pairing an SCA2 arm to GDN *by seed number* is
meaningless: different parameter shapes mean the same `--seed` gives incomparable
inits and there is no shared nuisance term for the pairing to cancel. And its
apparent precision on `dv=48` was three seeds happening to agree. **Every n=1
paired number this campaign produced is uninformative**, including a shape-curve
re-reading that was quoted for three hours before `dv=56` at n=3 falsified it.

### A retraction, itself retracted

**Generation 1 *was* resolvedly behind GDN.** This file previously retracted that
claim: at GDN n=3, `gen 1 − GDN = +0.1189` had p=0.088, correctly called
unresolved. At **n=6** the same comparison is +0.1126 at **p=0.0095**.

Worth being precise about what went wrong and what did not. The original claim was
quoted on a *thresholding bug* (`|t|>2.5`, a large-sample habit, wrong at dof≈3),
so retracting it was right on the data available. The fix was never a softer claim
— it was **more GDN seeds**, and nobody thought to spend them for weeks because the
instinct is to add seeds to your own arm.

**The `theta` init remains unresolved**: −0.0398 at p=0.053.

### Attribution

`--theta-scale` defaults to 0.0 and generation 1's runs never override it, so the
−0.1577 spans the delta rule *and* an init change; it was wrongly quoted as the
mechanism's effect at first. `theta_ctrl.sh` supplied the missing cell (additive
write at init 0.02, 2.8591 ± 0.0233, n=3), so the mechanism is now measured at
both matched inits. The init is worth −0.037..−0.047 under the additive write —
all three seeds negative, not resolved — consistent with the two effects being
roughly **additive**, and showing the earlier n=1 reading of +0.012 to be noise.

### The position profile, which the aggregate hides

Generation 3 against GDN, val by 128-token bucket (n=3 vs n=6):

| positions | gen 3 | GDN | Δ | | positions | Δ |
|---|---|---|---|---|---|---|
| 0-127 | 3.0243 | 3.0116 | **+0.013** | | 512-639 | −0.100 |
| 128-255 | 2.7947 | 2.8310 | −0.036 | | 640-767 | −0.107 |
| 256-383 | 2.7631 | 2.8306 | −0.068 | | 768-895 | −0.098 |
| 384-511 | 2.6504 | 2.7431 | −0.093 | | 896-1023 | −0.099 |

**There is a crossover.** Generation 3 is *behind* GDN on the first 128 tokens,
passes it near position 200, and settles at −0.10 from position 400 on. So the
aggregate −0.085 understates the long-context gain: restricted to 512-1023 the gap
is −0.1009 at p=0.0078, better resolved than the aggregate, while the
head-of-window penalty is +0.0128 at p=0.50, not measurable.

**That 512 cut was chosen after seeing the profile** and cannot be quoted as a
headline. The partial defence is that "SCA2 wins in long context" is this project's
a-priori hypothesis, predicted by its copy-task result — the *direction* was not
fished for, the exact cut was. Pre-register and re-measure to use it.

Note this contradicts generation 2's prediction that "the gain should keep growing
past 1024": at generation 3's shape it **plateaus by position 400-500**. The >1024
test is now about whether the plateau holds, not whether the gain keeps climbing.

### Throughput: generation 3 is 1.08x GDN, and every earlier speed number was junk

**`sweep_chunk.py`, blocked design, two independent sessions: x1.082 and x1.086**
at matched params (185714 GDN vs 185959 ours, 0.13% in GDN's favour). Call it
**x1.08**. This is the only throughput claim in this file that rests on a
measurement rather than on an assumption.

Everything quoted before it was junk, including in this file: generation 2 at "68.4k
against generation 1's 81.4k, so 1.19x, SCA2 no longer beats GDN's speed", and the
78.0k that made `cdeltaw` look like a speed win. The reason is worth stating plainly.

**Timing configs sequentially does not work on this machine.** Re-running the *same*
config three times across one session gave **71.6k / 65.1k / 76.7k** tok/s, 15.2%
apart, while three consecutive windows of one config agree to 2.2%. The
between-config variation the machine imposes is ~7x the within-config noise. And
`71.6 → 65.1 → 76.7` is **not monotone**, so it is not thermal decay and "run it on
a cold GPU" would not have fixed it — which also means the earlier reading of
dv=56's `77405 / 76325 / 70702` as *the machine heating up* was itself wrong. It was
never evidence of that.

**The fix is blocking, not averaging harder.** Every round times every config back
to back and only within-round *ratios* are kept, so the clock state is a nuisance
shared inside the block. Measured effect: raw spread 8.7–11.0% → residual ratio sd
**1.4–3.0%**, and the GDN ratio then reproduces across sessions to 0.004.

This is the same pairing idea that **failed** for val loss, and it works here for
exactly the reason it failed there: the nuisance is genuinely shared and genuinely
simultaneous. Two arms trained on different seeds share no such term; two configs
timed three seconds apart share the GPU's clock state. **Pairing is a claim about
what the nuisance is, not a variance-reduction trick, and the claim has to be true.**

### The chunk size is already right, and the theory said otherwise

`SCA2_CTX_CHUNK=128, SCA2_D_CHUNK=16` — what the campaign has been running — is at
the optimum. **No speed-up is available here.**

| ctx (dch=16) | 48 | 64 | 96 | **128** | 192 | 256 | 384 |
|---|---|---|---|---|---|---|---|
| ratio | 0.754 | 0.886 | 0.951 | **1.000** | 1.016 | 0.983 | 0.963 |

`dch=8` is 5–6% worse than 16 everywhere, `dch=32` 1–3% worse. The `ctx=192` cell is
+1.6% ± 3.9% and +0.9% ± 1.3% in the two sessions, i.e. a plateau, not a gain.

**The prediction was wrong, and in an instructive direction.** The C head's dominant
term is `4·B·T·C·Mc`, *linear* in the chunk, so halving `C` halves those FLOPs and
theory says go small. `Mc` had just fallen 378 → 190, halving that term relative to
the fixed launch overhead, so the optimum should have moved *up* from 128. Neither
happened: the low end is decisively **worse** (ctx=48 costs 25%), the high end mildly
worse, and 128 sits where it already was. Below ~128 this layer is launch-bound and
occupancy-bound, not FLOP-bound, so a FLOP argument predicts the wrong sign. Chunk
size is closed as a lever.

One process note: `ctx=64` and `dch=8` were dropped from the first blocked grid
because the *contaminated sequential* run had shown them poor — reusing a
measurement after declaring it void. They were put back and are indeed poor, so
nothing was lost, but the reasoning was invalid at the time.

## Rejected, and why — so they are not tried twice

Kept because each of these looked reasonable on paper, and two of them looked
good mid-run.

| candidate | result | why it is closed |
|---|---|---|
| `cdeltaw` — freeze the C **write** phase so the Gram becomes a constant Toeplitz matrix | 3.618 vs 2.748 (n=1) | Appeared to buy the speed back — a reading now void, since unblocked throughput on this machine varies 15% run to run — and cost **0.9 nats**, ~18 seed sd. The quality verdict stands on its own and is what closes it. Its position slope is **+0.19**, the only positive one in the campaign — every other arm improves with position, this one degrades. Cause: the read kernel depends on `pq_t − pw_s`, so a content-free write phase leaves the query's content term nothing to match against and destroys associative retrieval in the whole head, not just in the erase. **The route is closed, not merely suboptimal**: a constant Gram *requires* the change that kills the mechanism. |
| `wg2` — per-value-group spectral weights `w_{m,g}`, G=2 | 2.9710 (n=1) vs 2.9058 ± 0.0356 | **+0.065, ~1.8 seed sd worse**, and worse at *every* position bucket (+0.045 to +0.085), at 0.84x speed. The rank-2 bound on the shared temporal profile is real but is **not** the binding constraint. |
| `v2c*` — hoisting `q` projections out of the chunk loop | prefill 2.06 vs 2.36 ms, train 54.8 vs 10.6 ms | Wins 10% on prefill, loses **5x on the backward**: it retains `2·B·T·M·dv` of normalised activations. Still registered so the regression stays measurable. |
| the same hoist on the sequential D head | 275 vs 183 ms | 128 strided slices of a big tensor cost more than 128 small GEMMs. |

## Method notes that cost real time to learn

**Mid-descent val and the lag slope can both point the wrong way on the same
arm.** `wg2` was 0.065 *better* than its control at 15.3M tokens and its lag
slope was stably better at every probe eval, yet its end-of-epoch profile was
uniformly worse. `dump_cdelta.py` therefore refuses to print any verdict before
90% of an epoch, and prints the mid-descent column explicitly so a disagreement
stays visible instead of being quoted.

**Screening at n=1 is fine; concluding at n=1 is not.** Seed sd is 0.013–0.054
depending on the arm, which is larger than most single-seed effects measured
here. An arm earns three seeds by beating its control by more than ~1.5 sd on a
full epoch. The exception is an effect of ~18 sd (`cdeltaw`): n=1 screens
decisively when the effect is that large, and spending two more seeds on it would
have been waste, not rigour.

**Calibrate the promotion bar on the CANDIDATE's variance, not the control's.**
`dv=48` screened at −0.067, which is 4.6x `cdelta_t02`'s sd of 0.0147 and looked
like a huge hit. Its *own* endpoint sd turned out to be 0.0430 — so −0.067 was 1.6
of its own sd, i.e. no evidence at all. At n=3 the effect came in at **−0.021,
three times smaller than the screen**. At n=1 the candidate has no sd to use, so
the honest handling is to treat screens as a **ranking only** and never quote the
screen value as an effect size. `dump_cdelta.py` now says so at the print site.

**Selecting the best of k arms inflates the winner, and it needs correcting for.**
Five shapes were screened and the best was promoted; its uncorrected p=0.0101 sits
almost exactly on the Bonferroni-over-five threshold of 0.01. Two distinct errors
are involved and both bit here: the winner's-curse *bias* in the point estimate
(previous note), and the *multiplicity* in the p-value. A sweep followed by "the
winner is significant" is worth roughly one fifth of what it looks like.

**Spend seeds on the arm that owns the variance, which is usually not yours.**
Welch's denominator for gen 3 vs GDN is `0.0132²/3 + 0.0536²/6`: GDN supplies ~90%
of it. A GDN seed therefore bought ~25x more resolution per GPU-hour than one of
ours, and going from n=3 to n=6 on *our* arm would have moved t from 3.69 to only
3.79. Two claims stayed unresolved for weeks because the instinct is to reinforce
your own arm.

**At n=3 the t critical value is 2.9–4.3, not 2.** Thresholding at `|t|>2.5` out
of large-sample habit made two differences look resolved that are not (p=0.088
and p=0.053), and both got quoted in the docs before it was caught. Welch's dof
is often below 3 here because the arms have very unequal variances.

**A single readout point is too noisy for effects under ~0.05, and the obvious fix
made things worse.** Per-eval noise is ±0.05 on 60 eval batches, so interpolating
to one token count inherits all of it: one seed's difference swung from +0.056 to
−0.038, changing sign, purely from moving the readout by 12M tokens. The fix tried
was `dump_cdelta.paired` — 6 points over the last 30M, paired by seed — and it was
**retired for failing to replicate** (see above). The lesson is not "average more
points"; it is that a low-variance estimator must have its variance *verified on a
second arm* before being trusted, especially when it is the one licensing n=1
readings. The real remedy for a noisy readout is **more seeds**, which is boring
and works.

**Block when the nuisance is simultaneous; do not pair when it is not.** These two
notes look contradictory and are the same rule. Val loss: pairing arms by seed
*failed* and was retired, because two runs with different parameter shapes share no
nuisance term for the pairing to cancel. Throughput: pairing configs *within a
timed round* cut the noise 6x (8.7–11% raw → 1.4–3.0% residual), because two
configs timed three seconds apart genuinely do share the GPU's clock state.
Pairing is a claim about what the nuisance is — verify the claim, then use it.

**A knob at its default is not evidence the default was checked.** `SCA2_CTX_CHUNK`
sat at 128 through the whole campaign and turned out to be right, but nobody had
tested it since `Mc` halved, and the argument for re-testing it (cost is linear in
`C`) predicted the wrong *sign*: small chunks are much worse, because below ~128
the layer is launch-bound rather than FLOP-bound. Cheap sweeps of untouched knobs
are worth running precisely because the theory that justifies them can be wrong.

**A null is uninformative unless the mechanism demonstrably switched on.** The
delta rule's gate starts at `beta = 0.12` with a zero weight, and `beta = 0` is
the baseline exactly — so a null result would have been unreadable without
checking that the gate moved. It did: weight norm 0 → 0.80..1.75. Checkpoints are
saved for exactly this read (`dump_cdelta.py` reports it).

**`theta` never stays at a zero init** (it reaches `|theta|` = 1.8..11.4). So a
`theta_scale=0` arm is *not* a positional-only control, and any experiment that
needs one must freeze `theta`, not merely initialise it at zero.

## Open

Ordered by what would change a conclusion, not by interest.

1. **GDN at n=12.** Now the top item. Turns the headline from "ahead at the edge of
   significance" into something that survives multiplicity: t≈4.96 against the
   current 3.685 sitting on a 3.683 threshold. ~4 h, and it is a *GDN* experiment —
   more seeds of generation 3 move t by 0.10.
2. **Where the remaining time actually goes.** Chunk size is closed and the Gram is
   now half what it was, so the old "all of the cost is one term" analysis no longer
   describes this shape. A fresh `bench_heads.py` breakdown at Mc=190/dv=56 would say
   whether the D head is again the training step, as it was at Md=16.
3. **Erasure is confined to the constant-amplitude torus** `{e^{iφ}}`, whereas
   GDN's key direction is free in `R^k`. The delta rule makes erasure *possible*;
   which associations are *separable* is still restricted. Selective-support codes
   are untested, and this is the deepest question on the list.
4. **Whether the erase needs content addressing or only lag addressing.** Still
   undecided, and the route that would have settled it is closed: freezing the
   write phase (`CHeadDeltaWPos`) destroys retrieval across the whole head, so it
   answers nothing. The `theta` init 0 arm does not answer it either, because
   `theta` drifts to 1.8..11.4. A design that freezes `theta` without freezing the
   write phase is needed.
5. **`Md` has never moved from 4** while `Mc` fell 378 → 190, so the C/D width
   ratio silently went from 94:1 to 48:1. Nothing tested whether 4 is still right.
6. **The shape curve**, as opposed to the shape optimum. `dv=48` and `dv=56` are
   indistinguishable (p=0.62) and `dv=40/64` have only n=1 readings on which the
   two estimators disagreed about the *sign*. Quote "dv in 48..56", nothing finer.
7. **`theta_scale` beyond `{0, 0.02}`** — the only knob left that showed even a
   suggestive effect at matched everything else.
8. **Blocks longer than 1024.** The framing has changed: generation 3's advantage
   *plateaus* by position 400-500 rather than growing, so the test is whether the
   plateau holds out to 4k, not whether the gain keeps climbing.
