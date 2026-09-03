# SCA2 layer — what each optimization actually exploits

Semantics are frozen by `sca2/ref.py` and enforced by `python -m sca2.iso`.
Every claim below is verified numerically in float64 (tolerance 1e-10), not
argued.

## 0. Both heads are exactly recurrent, so decode is O(1) per token

C head: the cumulative sums are a running state `(sr, si)` of shape `(B, M, dv)`
plus an absolute position counter. D head: already an explicit scan. So the
layer admits an exact token-by-token decode — `iso.py:cross_iso` measures
3e-16 in float64, i.e. the recurrence is not an approximation.

One trap: the position must live in the state. `CHeadRef.step` reads
`float(state["pos"])`, a host-side int. That is fine eagerly and *silently
wrong* under CUDA-graph capture — the value is frozen at capture time, so every
replayed token decodes at position 0 (`graph_iso` reports 0.52 relative error
for `ref`). Variants store `pos` as a 0-dim tensor; `decode.py` refuses to
capture a state holding a host-side number.

## 1. C head = causal attention with a structured complex kernel

Writing `qr + i.qi = e^{-i pq}` and `sr + i.si = sum_s v[s] e^{i pw[s]}`, the
reference read-out is

    (rr + i.ii)[b,t,m,j] = sum_{s<=t} v[s,j] . e^{i (pw[s,m] - pq[t,m])}
    u_re + i.u_im        = mean_m (rr + i.ii) . (wr + i.wi)
                         = sum_{s<=t} v[s,j] . kappa[b,t,s]

The contraction over `M` has **no `dv` dependence**, so the score

    kappa[b,t,s] = (1/M) sum_m (wr + i.wi)_m e^{i (pw[s,m] - pq[t,m])}

is a *scalar*. Expanding `cos(pw-pq)` and `sin(pw-pq)` turns both parts into one
real matmul over a 2M-dim feature map:

    A   = wr.cos(pw) - wi.sin(pw)        Bm = wr.sin(pw) + wi.cos(pw)
    Kre = [cos(pq), sin(pq)] @ [A , Bm]^T / M
    Kim = [cos(pq), sin(pq)] @ [Bm, -A ]^T / M
    u   = concat(Kre @ v, Kim @ v)          (causally masked)

Cost goes from `B.T.M.dv` elementwise to `B.T^2.(2M+dv)` matmul — more FLOPs at
`T=128`, but the six retained `B.T.M.dv` activations collapse to two `B.T.T`
score matrices. Measured: forward 35.9 -> 0.65 ms, memory 162 -> 25 MB.

For a non-empty incoming state the carry contributes four `B.T.M.dv` einsums;
those are skipped entirely when the state is empty (a python-level `empty` flag,
so it is a compile-time constant rather than a data-dependent branch).

## 2. D head = chunked log-space scan

`s[t] = a[t].s[t-1] + v[t]` with `a = gr + i.gi`, `|a| <= 1`, `v` real. Over a
chunk of length C:

    s[t] = A[t].s_in + sum_{r<=t} D[t,r].v[r]
    A[t] = prod_{u<=t} a[u]              D[t,r] = prod_{r<u<=t} a[u]

`D` is built in log-magnitude / phase space so the products become cumsum
differences: `exp(cla[t] - cla[r])` with `cla` non-increasing, hence always
`<= 1`. No overflow, and — unlike the naive `A[t]/A[r]` — no division by a
decayed prefix. Gradients stay bounded because `d(la)/d(gr) = gr/|a|^2` is
always multiplied by a `D` factor that itself carries `|a[u]|`. The last chunk
is ragged rather than padded, so the closing state comes from the last real
position.

No scalar-kernel reduction exists here: the D head's query depends on `dv`, so
the `(t, m, j)` state genuinely has to be materialized. `B.T.M.dv` is the floor.

Two things decide the speed, in this order:

* **Permute before the outer difference, never after.** Building `dl` by
  permuting the `B.M.G.C.C` difference is a strided read of a 1M-element
  tensor; permuting `cla` while it is still `B.M.G.C` and then broadcasting
  gives identical values contiguously, from a tensor `C` times smaller. Worth
  ~3x on its own.
* **Chunk size trades bandwidth against launches.** Decay traffic is
  `B.M.G.C.T` — linear in C — while launch count is `T/C`. Eager pays per
  launch and prefers large C; fused pays per byte and prefers small. Swept in
  `bench_scan.py`, not guessed.

## 3. The layer is launch-bound, so fusion is the real lever

Profiling the eager chunked D head: 78% of forward time is elementwise kernels
averaging ~20 us on 0.5-1 MB tensors — roughly 5x what their traffic costs. At
`B=8`, `M=16`, `dv=64` the tensors are simply too small to amortize a kernel
launch. `torch.compile` collapses those chains and is worth more than every
algebraic change combined on the decode path.

Decode needs explicit graph capture. `torch.compile(mode="reduce-overhead")`
cannot be used: the state returned by step N is fed back into step N+1, but
cudagraph trees owns those buffers and overwrites them on the next replay
("accessing tensor output of CUDAGraphs that has been overwritten"). So
`decode.py` captures a graph by hand over **static** state buffers, with the
state write-back recorded *inside* the graph — one replay per token, and the
recurrence advances in place.

## 4. What the positional grid actually is (correction)

An earlier read of this layer called `omega = 2*pi*k/M` a bug, on the grounds
that the positional phase is periodic with period M and therefore aliases
positions `p` and `p+64` at the default `block=128`. The periodicity is real,
but calling it a bug was wrong, and section 1 above is why.

Since `kappa[b,t,s]` is a scalar score, at init (`theta=0`, `wr=1`, `wi=0`) it
collapses to a function of the lag alone:

    kappa[n] = (1/M) sum_m cos(n . omega_m),        n = s - t

a Dirichlet kernel. With `omega_m = 2*pi*m/M` the geometric sum is exact,

    sum_{m=0}^{M-1} e^{i 2 pi m n / M} = M . delta(n mod M)

so the grid is precisely what makes the C head an *exact delta* at
initialization. The aliasing is not an oversight, it is the price of DFT
orthogonality: the off-peak mass is conserved at 1.0 for any equispaced grid,
and `dft` buys a perfectly clean main lobe by putting the entire budget into one
alias spike.

Measured at M=64 (`kappa[n]`, theta=0):

| grid | n=0 | n=1 | n=2 | n=64 | max off-peak | total off-peak |
|---|---|---|---|---|---|---|
| `dft` 2πk/M | 1.000 | 0.000 | 0.000 | **1.000** | 1.000 | 1.00 |
| `len` 2πk/L, L=128 | 1.000 | 0.016 | 0.000 | 0.000 | **0.016** | 1.00 |
| `rope` geometric | 1.000 | 0.808 | 0.739 | 0.311 | 0.808 | **50.2** |

So the real trade is *shape*, not presence, of the sidelobe budget: `dft` is a
clean delta plus one full-height alias at lag M; `len` is a delta with 1.6%
ripple and no alias inside the window. `rope` is not a delta at all — the
geometric spacing also leaves the basis numerically near-degenerate (rank 49/128
at T=128), so theory rules it out without an experiment.

### And the theory does not predict trained quality

`sca2/ab_freq.py` measures it on the real cached corpus (compact vocabulary,
otherwise 98% of the loss is vocabulary bookkeeping; identical RNG consumption
across arms). Val loss after 1500 steps:

| arm | lr 1e-3 | lr 3e-4 |
|---|---|---|
| original (`dft`, theta=0) | 0.8368 | 1.1073 |
| `len` | 0.7670 (-0.070) | 1.0568 (-0.051) |
| `dft`, theta=0.02 | 0.8784 (+0.042) | 1.1193 (+0.012) |
| **`rope`** | **0.7080 (-0.129)** | **0.9792 (-0.128)** |

Same ranking at both learning rates. Three things came out of it, two of them
against the prediction above:

1. The aliasing does cost something -- `len` beats `dft` consistently. So the
   clean-delta argument does not justify the alias spike.
2. **`rope` wins by roughly double `len`'s margin**, despite the theory "ruling
   it out" for 50x off-peak mass and a rank-49 basis. The delta-at-init analysis
   is correct about step 0 and simply does not predict where training ends up: a
   delta means the head starts as near-identity and has to learn every lag,
   while a broad kernel has immediate access to a range of lags. The off-peak
   mass is coverage, not noise.
3. Raising `theta` off zero **hurts** (+0.01 to +0.04). The zero-gradient-on-K
   observation was real but irrelevant -- `theta` gets gradient immediately, so
   it is a one-step delay, and a pure-positional init is worth more than
   unblocking K one step earlier. Dropped.

One seed per arm: the ranking is stable across learning rates, not yet across
seeds.

`rope` is additionally the only grid whose period (2e4) exceeds any context this
layer would plausibly decode -- see section 5, where decode is O(1) in context
length while `dft`/`len` alias after 64/128 positions.

Likewise `theta = 0` starves `c.K` of gradient at step 0 — verified,
`|grad c.K| = 0` exactly — but `theta` itself receives gradient immediately, so
it is a one-step delay, not a dead unit. Also an empirical question, also in the
A/B (`theta_scale=0.02` gives `|grad c.K| = 2.2e-3` at init).

`ref.py` therefore keeps the ORIGINAL semantics as its default. The candidates
are reachable via `LayerCfg.cand()` / `--freq` / `--theta-scale`, and become the
default only if the loss says so.

## 4. Not done, and why

* TF32 / bf16 matmuls would speed the C head's `B.T.T` GEMMs further, but they
  change numerics beyond the fp32 iso tolerance. That is a precision decision
  for the owner of the model, not a free win, so it is left off.
* The frozen quirks (`omega` aliasing at period M, zero-init `theta` starving
  `K` of gradient, `si` receiving no input injection) are *semantics*. Fixing
  them belongs in a separate change with its own before/after quality
  measurement — not smuggled into a performance commit.

## 5. O(1) decode is real, and it costs something at short context

The state is `(B, Mc, dv)` plus `(B, Md, dv)` -- fixed size, so `pos` enters only
as a scalar and the tensor shapes never change with context. Measured per
generated token, `B=8`, `d=128`, eager unless noted (RTX 2070, idle):

| context L | SCA2 | SCA2 (CUDA graph) | attention | KV cache | attn/SCA2 |
|---|---|---|---|---|---|
| 128 | 1168 µs | 275 µs | 356 µs | 1.0 MB | 0.30x |
| 512 | 1140 | 281 | 347 | 4.2 | 0.30x |
| 2 048 | 1245 | 273 | 1 449 | 16.8 | 1.16x |
| 8 192 | 1207 | 266 | 5 813 | 67.1 | 4.82x |
| 32 768 | 1142 | 434 | 34 347 | 268 | 30.1x |
| 131 072 | 1134 | 262 | 332 964 | 1 074 | **293.6x** |

Flat within +-5% over three orders of magnitude of context; attention is exactly
linear past the launch-bound regime (1024x the length, 959x the time). State is
0.33 MB constant against 1074 MB of KV cache at 131k -- a factor of 3300.

The honest reading is not "SCA2 wins decode". Eager-vs-eager the crossover is at
**L ~ 1800**; below that attention is 3x cheaper per token, because SCA2 pays a
large fixed cost (`Mc.dv` = 4096 elements per head per batch element) for the
privilege of not growing. What the fixed shape buys beyond the crossover is that
it can be captured in a CUDA graph at all; attention can only be graphed with a
static full-length cache plus a mask, which makes every step cost `O(max_len)`,
including the first.

The attention baseline loads its weights from `nn.TransformerEncoderLayer`, so it
is the same block the benchmark trains, not a lookalike.

### Prefill is still O(T^2)

The quadratic C head (section 1) trades FLOPs for arithmetic efficiency, which
wins big at `T=128` but is asymptotically worse than the linear form. Estimated
crossover is around `T ~ 8k`, and memory binds first: the two `B.T.T` score
matrices are 4.3 GB at `T=8192, B=8`. Chunked-linear attention for the C head is
the next step if long context matters; the derivation in section 1 already
supports it (the state carry term is implemented and iso-tested via
`split_iso`).
