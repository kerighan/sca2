# sca2 — accelerating the SCA2 layer under an iso constraint

The layer from `bench_tinypython.py`, extracted as a frozen reference and then
optimized. Nothing here changes what the layer computes: every variant is gated
on numerical equivalence with the reference, in prefill **and** in token-by-token
decode.

## Layout

| file | role |
|---|---|
| `ref.py` | frozen reference semantics — the contract. `prefill` / `step` |
| `test_fidelity.py` | proves `ref.py` reproduces `bench_tinypython.py` exactly |
| `iso.py` | the gate: 6 checks per variant / shape / dtype |
| `variants.py` | optimized heads; each registers itself |
| `compiled.py` | `torch.compile` wrapper (prefill + step) |
| `decode.py` | manual CUDA-graph decoder with static state buffers |
| `bench.py`, `bench_heads.py`, `bench_vs_transformer.py` | measurement |
| `DERIVATION.md` | what each optimization exploits, and why |
| `arch_cdelta.py` | **error-correcting C write** — closes the gap to GDN. Not iso: it changes what the layer computes |

## The gate

```bash
python -m sca2.test_fidelity                       # ref == original file
python -m sca2.iso                                 # all variants, fp64 + fp32
python -m sca2.iso v2c8 --device cuda -v           # one variant, every check
```

Six checks, per variant × shape × dtype:

| check | asserts |
|---|---|
| `prefill_iso` | `variant.prefill(x)` == `ref.prefill(x)` |
| `decode_iso` | stepping token-by-token == the variant's own prefill |
| `cross_iso` | stepping token-by-token == the **reference** prefill |
| `split_iso` | `prefill(x[:, :k])` then stepping the tail == full prefill |
| `graph_iso` | the same decode replayed from a captured CUDA graph |
| `grad_iso` | every parameter gradient, and d/dx, == reference gradients |

Shapes cover `T ∈ {1, 3, 7, 64, 128, 129}` (ragged last chunk, single token,
`T` below the chunk size) and `B ∈ {1, 2, 3}`. float64 runs at tolerance 1e-10
to prove the algebra is exact rather than merely close; float32 at 3e-4.

Two notes on the harness, both learned the hard way:

* **Gradient errors are normalized by the layer-wide gradient scale**, not
  per-tensor. `c.wr`'s gradient at `T=1` is 2e-7 while the layer's largest
  gradient is 3.6e-1; a per-tensor denominator reports that as a 6e-4 "failure"
  when the absolute discrepancy is 1e-10.
* **`graph_iso` catches a whole bug class.** `CHeadRef.step` reads
  `float(state["pos"])`. Under CUDA-graph capture that host-side int is frozen,
  so every replayed token decodes at position 0 — 0.52 relative error, and
  completely invisible to an eager test. Variants keep `pos` as a 0-dim tensor,
  and `decode.py` now refuses to capture a state holding a host-side number.

## Results

RTX 2070, **idle** GPU, fp32, `d=128 Mc=64 Md=16 G=8 ff=256`. The GPU must be
idle: a competing process moved these numbers by 2x, in both directions.

Per head, `B=8 T=128`:

| head | fwd | fwd+bwd | peak MB (fwd) |
|---|---|---|---|
| `CHeadRef` | 35.9 ms | 119 ms | 162 |
| `CHeadQuad` | **0.65 ms** | **2.2 ms** | **25** |
| `DHeadRef` | 39.5 ms | 180 ms | 23 |
| `DHeadScan16` | 9.4 ms | 45.8 ms | 35 |

Whole layer, `B=8 T=128`:

| variant | prefill | train | graph decode | prefill MB |
|---|---|---|---|---|
| `ref` | 50.9 ms | 223 ms | — | 165 |
| `v1` (algebra only, eager) | 6.75 ms | 30.7 ms | 266 µs/tok | 36 |
| `v1c8_cc` (+ `torch.compile`) | **2.36 ms** | **10.6 ms** | **101 µs/tok** | **42** |

→ prefill **21.5x**, training step **21.1x**, decode **10.9x**.

At `B=16` (the benchmark's own default): prefill 75.4 -> 4.46 ms (**16.9x**),
train 283 -> 19.1 ms (**14.8x**), train memory 369 -> 134 MB.

Against the attention block it is competing with (same `d`, `ff`, `B`, `T`) —
this is the number `bench_tinypython.py` was trying to report with its broken
`tok/s`:

| layer | params | fwd | fwd+bwd | vs attention |
|---|---|---|---|---|
| `nn.TransformerEncoderLayer` | 132 480 | 0.71 ms | 2.24 ms | 1.00x |
| `ref` | 419 264 | 52.8 ms | 228 ms | **101.7x** |
| `v1c8_cc` | 419 264 | 2.84 ms | 10.9 ms | **4.89x** |

The layer went from 102x the cost of attention to 4.9x, at 3.2x the parameters.

### Rejected, and why

`v2c*` hoists the `q` projections out of the chunk loop, replacing `T/C` small
GEMMs with two big ones. It wins ~10% on prefill (2.06 vs 2.36 ms) and loses 5x
on the backward (train 54.8 vs 10.6 ms), because it retains `2.B.T.M.dv` of
normalized activations. Kept registered so the regression stays measurable
instead of becoming folklore.

The same trick applied to the *sequential* reference D head is also slower
(275 vs 183 ms) — 128 strided slices of a big tensor cost more than 128 small
GEMMs. Neither of these is guessable from the source; both were measured.

## Reproducing

```bash
python -m sca2.bench ref v1 v1c8_cc -B 8 -T 128
python -m sca2.bench_heads -B 8 -T 128          # per-head, so wins aren't masked
python -m sca2.bench_vs_transformer v1c8_cc     # vs the attention block it replaces
```

## Semantics: measured, not asserted

`ref.py` keeps the ORIGINAL semantics as its default. Candidates are opt-in
(`LayerCfg.cand()`, or `--freq` / `--theta-scale`) and become the default only
when a measurement says so. Every fast variant passes iso under **all** of them
— the optimized forms are algebraic identities independent of `omega`'s values,
so the semantic knob and the 21x are orthogonal.

`omega = 2πk/M` is not a bug, which an earlier read of this layer got wrong.
Because `kappa` is a scalar score, at init the C head collapses to a Dirichlet
kernel in the lag, and the `2πk/M` grid makes the geometric sum exact — the head
starts as a *perfect delta*. The period-M aliasing is the price of DFT
orthogonality: off-peak mass is conserved at 1.0 for any equispaced grid, and
`dft` concentrates all of it into one alias spike at lag M.

`python -m sca2.ab_freq` — val loss at 1500 steps, compact vocab, identical RNG
across arms:

| arm | lr 1e-3 | lr 3e-4 |
|---|---|---|
| original (`dft`, θ=0) | 0.8368 | 1.1073 |
| `len` (2πk/L) | 0.7670 (−0.070) | 1.0568 (−0.051) |
| `dft`, θ=0.02 | 0.8784 (+0.042) | 1.1193 (+0.012) |
| **`rope`** | **0.7080 (−0.129)** | **0.9792 (−0.128)** |

Same ranking at both learning rates, and two results land against the theory:
`rope` wins by roughly double `len`'s margin despite being the grid the
delta-at-init argument rules out, and raising `theta` off zero *hurts* (the
zero-gradient-on-K observation was real but irrelevant — `theta` gets gradient
immediately, so it is a one-step delay). One seed per arm; stable across lr, not
yet across seeds. See DERIVATION.md section 4.

The D head's imaginary state receiving no input injection is a *design
limitation*, not a defect — a real input into a complex state just fixes the
injection phase at 0, and the gates rotate it afterwards. Removing it means new
parameters, i.e. a different architecture.

TF32/bf16 matmuls are left off: they would break the fp32 iso tolerance, which
is a precision decision, not a free win.

## The one semantic change that closed the gap to GDN

Everything above preserves the function. This does not, and it is the only
change so far that removed the 0.119-nat deficit against Gated DeltaNet
(`arch_cdelta.py`, pycode, 1024-token blocks, 4 layers, one 177M-token epoch,
params matched to 0.07%):

| arm | val | seed sd | n |
|---|---|---|---|
| **delta write, θ init 0.02** | **2.7480** | 0.0147 | 3 |
| Gated DeltaNet | 2.7869 | 0.0733 | 3 |
| delta write, θ init 0 | 2.7879 | 0.0198 | 3 |
| additive write (`v3polarflat`) | 2.9058 | 0.0356 | 3 |

Two estimators, and a claim counts only if both agree: interpolation to a common
168.7M tokens, and a seed-**paired** mean over the last 30M
(`dump_cdelta.paired`). "Resolved" means |t| beats the true two-sided 95% t
critical value at its dof — **2.9 to 4.3 at n=3, not 2**.

| comparison | endpoint | paired | resolved |
|---|---|---|---|
| delta(θ.02) − additive(θ0) | −0.1577 | −0.1865 | yes, but spans two changes |
| **delta(θ.02) − additive(θ.02)** | **−0.1111** | **−0.1463** | **yes** — mechanism |
| **delta(θ0) − additive(θ0)** | **−0.1179** | **−0.1366** | **yes** — mechanism |
| delta(θ.02) − delta(θ0) | −0.0398 | −0.0501 | no, p=0.053 |
| additive(θ.02) − additive(θ0) | −0.0467 | −0.0369 | no, p=0.140 |
| delta(θ.02) − GDN | −0.0388 | −0.0832 | no, p=0.456 |
| additive − GDN | +0.1189 | +0.0994 | no, p=0.088 |

**Attribution.** `--theta-scale` defaults to 0.0 and the additive arm's runs never
override it, so the −0.1577 row spans the delta rule *and* a θ init change.
`theta_ctrl.sh` supplied the missing cell (additive write at θ init 0.02, 2.8591 ±
0.0233, n=3), so the mechanism is now measured at **both** matched inits: −0.1111
and −0.1179. The init is worth −0.037..−0.047 under the additive write, all three
seeds negative but not resolved — consistent with the two effects being roughly
additive, and showing the earlier n=1 reading of +0.012 to be noise.

**Two retractions.** `additive − GDN = +0.1189` was quoted as "the gap, confirmed
real"; it is **not resolved** (p=0.088). An earlier `dump_cdelta.py` thresholded
at `|t|>2.5`, wrong at dof≈3. So SCA2 was never *resolvedly* behind GDN at n=3 —
only suggestively. The θ init is likewise not a resolved effect (p=0.053).

Against GDN the delta arm has the better mean but nothing resolves, and that is
not evidence of equivalence either — no claim of parity, none of superiority.
What is established is the mechanism against its own baseline.

The write becomes `S += φ_t (v_t − β_t·v̂_t)ᴴ` with `v̂ᴴ = Re(φᴴS)/M`, which is
exact because the codes have constant amplitude (`‖φ‖² = M`). That same constant
amplitude makes this *exactly* a real unit-norm delta rule: with
`u = [cos pʷ ; sin pʷ]/√M` and `H = [Sᴿ ; Sᴵ]/√M`, one has `‖u‖ = 1` and
`H_t = (I − β_t u uᵀ) H_{t−1} + u v_tᵀ` identically (checked to 8.9e-16). So the
phase parameterises normalised keys, and the difference from GDN is exact: same
algebraic form, different key manifold — GDN's key ranges over the unit sphere of
`R^dk`, ours over the Clifford torus in `R^2M`. Over a chunk this
is unit lower triangular, so it costs **one triangular solve** and nothing else:
the read and the state carry are the additive layer's with `v ← E`. `β = 0`
reproduces the additive layer bit-for-bit (1.1e-15), so the baseline is nested
and the gate can switch the mechanism off — it does the opposite, training to
`|w| = 0.80..1.75`, i.e. genuinely data-dependent erasure.

Two things worth carrying forward:

* The gain **grows with position** (−0.046 at tokens 0-127 to −0.238 at
  896-1023) whereas GDN's advantage is flat. The two mechanisms are
  complementary, not two approximations of one.
* It costs **1.19x** in throughput (68.4k vs 81.4k tok/s), so SCA2 no longer
  beats GDN's speed. The gain survives the change of axis (at equal wall clock:
  2.7979 / 2.8750 / 2.9114), and all of the 1.19x is one term — the Gram matrix
  of the write codes, `2·C²·M` against the kernel's `4·C²·M`, with the solve
  itself only 4% of that. `CHeadDeltaWPos` makes that Gram a cached Toeplitz
  matrix by freezing the write phase; whether the gain survives is being
  measured.

`python -m sca2.arch_cdelta` checks the closed form against the sequential
recurrence (2.7e-15), `β = 0` against `CHeadQuad` (1.1e-15), and the cached
Toeplitz against the explicit Gram (3.3e-16).

## Against the Transformer at matched parameters

`python -m sca2.bench_params` — one knob per architecture so nothing is
confounded (SCA2 moves `Md`, the Transformer moves `ff`; budgets agree to 0.2%),
compact vocab, lr swept per architecture and the best reported, 2000 steps, val
loss:

| non-emb params | SCA2 `dft` | SCA2 `rope` | Transformer | best gap |
|---|---|---|---|---|
| 161k | 0.8365 | **0.7024** | 1.0158 (ff=367) | +0.313 |
| 198k | 0.8137 | **0.6907** | 0.9835 (ff=512) | +0.293 |
| 272k | 0.7936 | **0.6756** | 0.9563 (ff=798) | +0.281 |
| 419k | 0.7728 | **0.6593** | 0.9364 (ff=1372) | +0.277 |

SCA2 wins at every matched budget — by 0.16–0.18 nats on the original semantics,
0.28–0.31 with `rope`. And SCA2 at the *smallest* budget beats the Transformer at
the *largest* (0.7024 vs 0.9364).

**The most actionable number in this table is SCA2's own scaling.** Going from
`Md=2` to `Md=16` costs 230k parameters — `dh.qr/qi` grows from 32k to 262k, and
those two projections are 62.6% of the layer — and buys 0.064 nats (`dft`) or
0.043 (`rope`). The q projections are heavily over-provisioned: they carry
`M·dv` = 1024 degrees of freedom per token, exactly as many as the state they
read, while the state's `m`-dependence comes only from the gate phase. A
separable query `q[m,j] = α[m]·β[j]` would cost 20k instead of 262k and free
those parameters for anywhere they earn more. That is the v3 candidate.

Caveats worth stating: 2000 steps, one seed, one dataset, single-layer models,
`T=128`, compact vocab, two-point lr sweep. This is a measurement of these
configurations, not a general claim about the architectures.

## Equal wall clock — the decisive comparison

`python -m sca2.ab_isotime` — 90 s of training loop per arm, compilation and eval
excluded from the clock, all arms compiled, lr swept:

| arm | params | steps | val |
|---|---|---|---|
| transformer ff=256 | 132k | 6849 | 0.6771 |
| transformer ff=1372 | 419k | 4702 | 0.6873 |
| sca2 v1 `Mc64 Md16` | 419k | 5351 | 0.5438 |
| **sca2 sepq `Mc64 Md16`** | **177k** | **7149** | **0.5330** |
| sca2 sepq `Mc16 Md16` | 171k | 7268 | 0.5505 |

SCA2 wins by **0.144 nats at equal seconds**, and the best arm is the *smallest*
one: separable-query at 177k gets MORE steps than the 132k transformer (7149 vs
6849) and a far lower loss. "Powerful but too compute hungry" does not survive
the measurement — the compute is paid for.

Two side results: the bigger transformer is *worse* at equal time than the small
one (fewer steps, no compensation), and `Mc=16` buys only 119 extra steps for
+0.0175 loss, so shrinking `Mc` for throughput is not worth it.

NOTE: the first version of this measurement charged `torch.compile` time to the
budget, which made the step counts meaningless. The warmup is now untimed.

## Separable query, and the shape of its trade

| arm | params | val (`dft`) | val (`rope`) | fwd+bwd | vs attention |
|---|---|---|---|---|---|
| full `Md=16` | 419k | 0.7735 | 0.6576 | 10.05 ms | 4.86x |
| **sepq `Md=16`** | **177k** | 0.8183 | 0.6919 | **7.92 ms** | **3.82x** |
| sepq `Md=64` | 289k | 0.7839 | 0.6672 | 22.02 ms | 10.65x |
| sepq `Md=128` | 437k | 0.7425 | 0.6475 | 44.54 ms | 21.44x |

`Md=16` is the only point on the curve that buys time: −21% time and −58%
parameters for +0.034..0.045 loss. Reinvesting the freed parameters into `Md`
is a bad trade — cost is super-linear (decay matrices are `B·Md·G·C·T`, state is
`Md·dv`), so the arm that wins on quality at matched budget is 4.4x slower.

For scale: a transformer at the same 177k (ff=431, measured directly) reaches
1.0037. sepq `Md=16` is **+0.185** better with `dft` and **+0.312** with `rope`.

## Decode is O(1) — and what that costs

State is fixed-size, so `pos` enters only as a scalar and shapes never change.
Per generated token, `B=8`, eager unless noted:

| context L | SCA2 | SCA2 (graph) | attention | KV cache | attn/SCA2 |
|---|---|---|---|---|---|
| 128 | 1168 µs | 275 µs | 356 µs | 1.0 MB | 0.30x |
| 2 048 | 1245 | 273 | 1 449 | 16.8 | 1.16x |
| 8 192 | 1207 | 266 | 5 813 | 67.1 | 4.82x |
| 131 072 | 1134 | 262 | 332 964 | 1 074 | **293.6x** |

Flat within ±5% across three orders of magnitude; state 0.33 MB constant vs
1074 MB of KV cache at 131k. Eager-vs-eager the crossover is at **L ≈ 1800** —
below that attention is 3x cheaper per token, because SCA2 pays a large fixed
cost for not growing. Prefill is still O(T²) (memory binds first: 4.3 GB of
score matrices at T=8192, B=8), so chunked-linear C-head attention is the next
step for long context.

Note the two results agree: `rope` is both the best-scoring grid and the only one
whose period (2e4) exceeds a context this layer can afford to decode — `dft` and
`len` alias after 64 and 128 positions.
