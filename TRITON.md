# Triton Kernel for the Long Head — Handoff Document

## Implementation status (2026-09-15)

Four opt-in implementations are available through `LaplaceConfig.long_path`
or `SCA2_LONG_PATH` in the sca2 adapter. The default remains `batched`.

| Path | Handwritten Triton forward and backward |
|---|---|
| `triton` | Triangular inverse and its analytic adjoint |
| `triton_codes` | Phase/decay/code construction, Gram, causal grouped code product |
| `triton_fused` | All of the above, plus the recurrent state loop |
| `triton_scan` | All of the above as **one** autograd node, on compact read codes |

`triton_scan` is the fastest path in every arm measured and is the one to opt
into; `triton_codes` and `triton_fused` are kept as the earlier milestones the
comparison is against. The CUDA paths use fp32 state/solve/phase arithmetic and
fp32 or bf16 codes. CPU/fp64, banded beta and input-dependent decay retain the
applicable reference components. Projection GEMMs, output RMS/key verification,
LayerScale, short head and FFN remain PyTorch. These are first-order autograd
implementations.

- `lapa/triton_phase.py`: constructs grouped codes directly in their consumed
  layout, avoiding intermediate phase/trig/decay tensors and layout copies;
  backward recomputes the phases and produces projected-key/parameter gradients.
  `compact=True` emits the read codes as `[c1 | c2]` without the rotated block.
- `lapa/triton_product.py`: masked Gram and grouped code products, with causal
  masks and group reductions inside their gradient kernels; `ACC` accumulates
  onto an existing gradient buffer instead of overwriting it.
- `lapa/triton_scan{,_kernel}.py`: the whole chunked form as a single graph
  node — Gram, triangular inverse, causal intra-chunk kernel, chunk loop, and
  every code and matrix gradient. Described below.
- `lapa/triton_state*.py`: the earlier per-chunk fused state loop (`triton_fused`).
- `lapa/triton_solve*.py`: register-tiled forward substitution and two-GEMM
  analytic inverse adjoint, including gate gradients and strict-lower masking.
- `lapa/test_triton_{solve,state,scan}.py`: output/state/input/parameter
  gradients, noncontiguous inputs, odd dimensions, grouped reads, nonzero
  initial states, ragged tails, bf16, fp64 fallback, decay-ceiling stress and
  compiled execution.

```bash
OMP_NUM_THREADS=1 python -m unittest lapa.test_triton_scan lapa.test_triton_solve \
  lapa.test_triton_state -v
OMP_NUM_THREADS=1 SCA2_CTX_CHUNK=8 SCA2_LONG_PATH=triton_scan \
  python -m sca2.iso lapa --self --fast --device cuda
PYTORCH_ALLOC_CONF=expandable_segments:True python -m lapa.benchmarks.triton \
  --paths batched,triton_codes,triton_scan
```

The short chunks in the iso command ensure it actually enters `_batched`;
the default fast harness has T <= 128 and would not exercise a batched
C=128 kernel. The `--self` harness does not compare gradients; the dedicated
tests above do.

### Results

Measured on the idle GB10 at the target configuration (B=8, T=1024, d=1024,
M=256, dv=256, kv_dk=16, C=128, NG=2, conv=4, free decay, scalar LayerScale,
bf16 autocast, `torch.compile(fullgraph=True)`), five warmups per arm, then
**20 paired rounds of 10 iterations** with the arm order alternating each round.
Percentages are the median of within-round ratios, not the ratio of the medians.

| Full layer | `batched` | `triton_codes` | `triton_scan` |
|---|---:|---:|---:|
| Forward, no gradients | 11.212 ms | 10.870 ms (+3.3%) | **9.479 ms (+19.8%)** |
| Forward + backward | 37.117 ms | 34.755 ms (+6.7%) | **32.087 ms (+16.2%)** |
| Forward + backward + AdamW | 37.592 ms | 35.228 ms (+6.9%) | **32.600 ms (+14.8%)** |

| Long head only | `batched` | `triton_codes` | `triton_scan` |
|---|---:|---:|---:|
| Forward, no gradients | 5.110 ms | 4.624 ms (+11.1%) | **3.219 ms (+56.6%)** |
| Forward + backward | 15.752 ms | 14.055 ms (+12.6%) | **10.645 ms (+48.2%)** |

`triton_codes` reproduces the +5.94% full-layer training figure reported for it
earlier, so the comparison is like for like. `triton_scan` is **2.4x that gain**
at layer level and roughly four times it on the long head alone. These are
single-layer synthetic timings, not a full language-model training throughput
claim; the AdamW arm clears gradients each step and uses lr=0 to keep the
weights fixed.

GPU time, measured with `nsys` over six iterations of the long head's
forward+backward, went from **12.0 ms/iter in 323 kernel launches** for
`triton_codes` to **10.7 ms/iter in 114 launches**. Roughly half of the
remaining time is layer work shared by every path -- the input cast, the
projections, the five-way accumulation of the input gradient, the output RMS
and key gate.

All **22 dedicated tests pass** (`test_triton_scan` adds 9 to the existing 13),
covering independent gradients for every input of the fused node, odd
dimensions, C=1, ragged tails, nonzero incoming states, bf16, the fp64 and
banded-beta/input-decay fallbacks, and compiled execution at the training
shape. `long_groups` 1, 2, 4 and 8 all match the reference layer to 2.6e-06 in
fp32 and 5.9e-03 under bf16 autocast, including compiled at the training shape:
the grouped column blocking and the static group reduction in the `dKk` adjoint
are the two places that depend on the group count, and only NG=2 is trained. The GPU iso gate passes in fp64 (5.5e-16
against a tolerance of 1e-10) and fp32 (2.5e-07 against 3e-04). The compiled
bf16 training-shape test's worst normalized deviation is 0.0059, against the
harness's bf16 tolerance of 0.06.

Raw per-round data is in
[`lapa/benchmarks/results/triton_scan_gb10_20260915.json`](lapa/benchmarks/results/triton_scan_gb10_20260915.json).

### Shape-specific tuning and fused convolution (2026-09-21)

`triton_scan` now tunes forward, both reverse kernels, matrix gradients and
compact-code products. `lapa/triton_scan_tune.py` retains the historical TUNE,
GTUNE, KTUNE and FTUNE settings among the candidates. Selection depends on
B/K/C/R/D/G, dtype, device, training/inference and operation kind where relevant.
The grids follow the selected tiles; all tiles are powers of two and the
existing masks handle widths such as 272, 384 and 512 without padding the
whole value/state tensors. Tiny shapes use conservative single-config kernels; fp32/tf32x3 uses a smaller
resource-appropriate candidate set than bf16.

There was no requirement for D itself to be a power of two: D=272 was already
non-power-of-two. The old forward actually used BN=64, TK=32. With NG=1,
D=384 has six full column tiles; with NG=2, D=272 has three tiles per group.
Both yield 48 programs at B=8. Changing T from 1024 to 2048 also doubles the
sequential chunk count, so width alone does not explain the timing difference.

The forward stores its r/h intermediates immediately after the solve and delays
K2@e until after the mode loop, shortening register lifetimes. Gradient scratch
layouts use fixed 16-column/16-row slots; each chosen tile writes its partial
and zeros the remaining slots it owns. No stale partials survive a change of
configuration, and no atomic reductions or whole-tensor reset passes are needed.
Both decay gradients are computed in the mode-parallel reverse kernel, sharing
the same saved-state load instead of reading it again in the reducing kernel.

Autotuning benchmarks restore the live state (`ST`), state adjoint (`DSR`) and
accumulated code gradients (`OUT`). **Do not use `early_config_prune` with these
kernels on PyTorch 2.9**: its Triton wrapper rebuilds the autotuner after pruning
without forwarding `restore_value`. A compiled 16-chunk correctness test exposed
this as a large output error. Separate single-config entry points handle tiny
shapes without that reconstruction. See also the
[Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html).

The first use of a new shape benchmarks/compiles candidates; subsequent calls
use cached choices. Eager uses Triton's disk cache; compiled execution uses
Inductor's cache. `SCA2_SCAN_AUTOTUNE=0` (set **before import**) selects historical
tiles for comparisons, with the new scratch layout and forward scheduling.
It is not a switch back to the entire old implementation.

#### `conv_silu`: preserve the model's ordering

`phase_codes` receives **K(z)**, and conv_silu computes **K(silu(conv(z)))**.
Applying SiLU inside phase construction would instead compute silu(K(conv(z)))
and leave V, gates and the short head inconsistent. The fusion therefore lives
in `lapa/triton_conv.py`, at the causal convolution's epilogue, and is selected
by the `triton_scan` layer path. It supports both values of `LaplaceConfig.conv_silu`.
It emits contiguous [B,T,d] tensors, avoiding layout copies before projections.
Backward recomputes the activation inside the input/weight-gradient kernels.
There is no separate SiLU tensor/pass. bf16 autocast rounds at the same conv,
activation and gradient boundaries as the reference; accumulation stays fp32.

The fast convolution handles CUDA fp32/bf16 and widths 1..8. Other convolution
settings and fp64 keep the PyTorch reference. Decode remains the reference
single-token operation, with the same state/history representation.

#### Reproduce the comparisons

The benchmark's old defaults are preserved (dv=256, kv_dk=16, T=1024, NG=2).
For the new training setup, supply the shape explicitly:

```bash
OMP_NUM_THREADS=1 python -m unittest lapa.test_triton_scan -v
OMP_NUM_THREADS=1 SCA2_CTX_CHUNK=8 SCA2_LONG_PATH=triton_scan \
  python -m sca2.iso lapa --self --fast --device cuda

# dv=384, D=384; full current layer configuration
OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m lapa.benchmarks.triton --paths batched,triton_scan \
  --tokens 2048 --dv 384 --kv-dk 0 --groups 1 --ff 5800 \
  --short-window 128 --gdn-gate
# Add --conv-silu to measure the activation-enabled arm.

# Historical dv=256 / D=272 configuration
OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m lapa.benchmarks.triton --paths batched,triton_scan
# D=512 uses the same selector: --dv 512 --kv-dk 0 --groups 1.
```

The dedicated suite includes fp32 and compiled bf16 at D=256/384/512 without
key verification, alongside the old D=272 tests, grouped reads, ragged tails,
independent input gradients, decay-ceiling stress and fallbacks. Convolution
checks cover SiLU on/off, widths 1/4/8, fp32/bf16, compiled adjoints and carried
history. Architectural hypotheses are in [IDEAS.md](IDEAS.md).

#### Measured outcome (GB10, 2026-09-21)

Same paired protocol as above: five warmups, 20 alternating rounds of ten
iterations, bf16 autocast and fullgraph compilation. First-use compilation and
autotuning are excluded. The full-layer D=384 configuration is the command
above (T=2048, ff=5800, short window=128, GDN gate); D=272 retains the historical
T=1024, ff=4096, short window=64 configuration. Compare paths within each row,
not absolute timings between widths or between separate SiLU runs.

| D / NG | Scope | batched | triton_scan | Paired throughput gain | scan tok/s |
|---|---|---:|---:|---:|---:|
| 384 / 1 | Long head, fwd+bwd | 32.163 ms | 24.721 ms | +30.5% | 662.8k |
| 384 / 1 | Full layer, fwd+bwd | 89.306 ms | 81.376 ms | +9.5% | 201.3k |
| 384 / 1 | Full layer, fwd+bwd+AdamW | 91.076 ms | 83.420 ms | +10.2% | 196.4k |
| 272 / 2 | Full layer, fwd+bwd | 34.151 ms | 29.083 ms | +17.6% | 281.7k |
| 272 / 2 | Full layer, fwd+bwd+AdamW | 36.342 ms | 30.733 ms | +18.3% | 266.6k |
| 384 / 1 + SiLU | Full layer, fwd+bwd | 96.053 ms | 87.921 ms | +9.5% | 186.3k |
| 384 / 1 + SiLU | Full layer, fwd+bwd+AdamW | 94.155 ms | 84.635 ms | +11.1% | 193.6k |

**The 77 ms / approximately 213k tok/s target is not reached.** D=384 full-layer
training measures 81.376 ms (201.3k tok/s), versus 89.306 ms for batched. The
long head improves more than the whole layer; this result does not establish
that further tile tuning alone can close the remaining approximately 4.4 ms.
D=272 retains its historical throughput advantage. D=512 passes correctness
and compiled-gradient checks, but has no end-to-end performance claim here.

All 12 scan/convolution tests pass. A further B=8, M=256, D=384, C=128,
16-chunk compiled bf16 check, including a three-token ragged tail, passes all
output/state/input/parameter-gradient comparisons; worst normalized output
error is 0.006874 against tolerance 0.06. GPU iso also passes (0 failing checks).

Raw per-round timings, configuration arguments, source hashes and validation
results are recorded in
[`triton_scan_widths_gb10_20260921.json`](lapa/benchmarks/results/triton_scan_widths_gb10_20260921.json).

### The scan path: the whole chunked form as one graph node

The measurement that drove the design: at the training shape the long head's
own arithmetic is a rounding error. Profiled with `nsys` (torch.profiler does
not report kernel times on SM 12.1), `triton_codes` spent **49% of its GPU time
in pointwise and elementwise kernels** — gradient adds, dtype casts, layout
clones, per-chunk stacks — and most of the rest in 145 small cuBLAS GEMM
launches averaging 20 microseconds. Kernel launch here costs 3.2 us, bandwidth
is 200 GB/s and bf16 peak is 60 TFLOP/s: 300 FLOP per byte. Nothing in the long
head is compute bound, so the work is to stop moving intermediates.

Five structural changes, each measured:

**One autograd node.** `long_chunk(Kk, Qk, Fc, v, beta, s, d1, dC, gT)` owns the
Gram, the triangular inverse, the causal intra-chunk kernel and the chunk loop.
Because one node owns all of them, the kernel that produces the second
contribution to a code gradient accumulates onto the buffer the first one wrote
(`ACC` in the product kernel, in-place adds in the K2 adjoints). The backward
therefore runs **no gradient-add pass at all**; separate autograd nodes spent
~1.5 ms/iter summing `dFq`, `dKk` and `dQk` across their two or three producers.

**The chunk loop is column-parallel.** It is sequential only in the chunk index:
given the codes, the inverse and the values, every column of the value width is
independent — the state read, the delta solve, the output and the state write
all act column by column. One program owns (batch, read group, column block)
and walks all K chunks, so the forward scan is **one launch** instead of four
per chunk, and the reductions over the chunk length C are single `tl.dot` calls
rather than tiled loops.

**The reverse scan splits by what actually reduces.** `de` and the write adjoint
need the whole mode axis reduced, but `ds0 = Fq^T do + Qk^T dr` is a per-mode
outer product and the state-gradient recursion is elementwise, so the second
half also splits over the 2M modes. Two launches per chunk with the mode-parallel
half at a 4x larger grid beat one launch: 1.65 -> 1.20 ms/iter.

**Compact read codes.** `Fq = [[c1 | c2], [-c2 | c1]]`: the imaginary rows are
the real rows rotated, so `phase_codes(..., compact=True)` stores `[c1 | c2]`
once and every consumer applies the rotation while it has the tile in registers.
That halves the largest tensor in the layer (33.6 -> 16.8 MB) and its gradient,
in the code kernel, the causal product, the scan and the read-code gradient at
once: 12.4 -> 11.1 ms/iter of GPU time.

**Only the state the backward needs is kept.** The live fp32 state ping-pongs
between two slots and stays in L2; what is written per chunk is the rounded,
decayed copy `s0` the code gradients read anyway. The backward recovers the
undecayed state as `s0 / d1` for the two ramp gradients — `d1 = exp(-lambda)`
is bounded below by `exp(-lam_ceil)`, so the division is stable, and the path
is exact when `gemm_dtype` is fp32. The gT gradient is reassociated:
`sum_j ds . (Kk^T e) = sum_c Kk . (e ds^T)`, and the second form is the product
the `dKk` kernel already computes, so the state write is never recomputed.

### Four things about Inductor that cost real time to find

These bit hard enough to be worth writing down; all four are about user-defined
Triton kernels inside `torch.compile`.

1. **Passing the same tensor in two argument slots silently corrupts results.**
   Inductor clones what it believes a user kernel mutates; with an output tensor
   also passed in an unused slot it cloned the wrong one, and the kernel wrote a
   buffer nobody read. Eager was exact, compiled produced NaN. Every launch here
   now passes each tensor once.
2. **A pointwise result is realised once per user kernel that reads it.** The
   scaled output gradient is read by 18 launches and Inductor wrote 18 identical
   8.9 MB copies — 0.94 ms/iter. Writing it with a Triton kernel of its own,
   whose output is already a real buffer, removes all of them.
3. **The eager `OutOfResources` retry does not apply under `torch.compile`.**
   The launch config is baked into the graph, so a configuration that needs more
   shared memory than the SM has is a hard compile failure; the defaults have to
   fit by construction. `_launch` keeps the retry as the eager safety net.
4. **Triton's pipeliner hoists loads across the chunk loop's carried
   dependency.** At `num_stages >= 3` the scan read a chunk's state before the
   previous chunk had written it — wrong by 100% for C in {7, 8, 16}, correct at
   C = 32 and 128, which is exactly the kind of bug that passes a quick check.
   An explicit `tl.debug_barrier()` at the top of the chunk loop fixes it and
   costs nothing measurable.

5. **A `tl.constexpr` with a default value disables the analysis entirely.**
   Adding `ACC: tl.constexpr = False` to the product kernel and leaving the
   existing call sites alone made PyTorch raise `Incorrect number of arguments
   passed to kernel` internally, conclude that the kernel mutates nothing, and
   **drop the launch from the compiled graph**. `triton_codes` then computed
   neither the Gram nor K2 under `torch.compile` — silently wrong, and faster,
   because it was skipping work. Every constexpr is now required and passed at
   every call site. Eager was correct throughout, which is why only the
   compiled tests caught it.

### Earlier milestone: the code/product hybrid (`triton_codes`)

Superseded by `triton_scan` above; kept because it is the baseline the current
numbers are measured against, and because its conclusion -- that more
handwritten kernels are not automatically faster -- still holds.

Measured on the idle GB10 with the target configuration and scalar LayerScale,
five warmups per arm, 20 alternating paired rounds of 10 iterations:

| Full layer | `batched` | `triton_codes` | `triton_fused` |
|---|---:|---:|---:|
| Forward, no gradients | 11.607 ms | 11.294 ms | **10.999 ms** |
| Forward + backward | 37.296 ms | **35.210 ms** | 36.287 ms |
| Forward + backward + AdamW | 37.593 ms | **35.582 ms** | 37.182 ms |

Within-round median throughput improvements over `batched`:

- `triton_codes`: +3.15% forward, **+5.94% forward/backward, +6.09% with AdamW**.
  It beat the reference in all 20 rounds for each measurement.
- `triton_fused`: **+6.26% forward**, +2.43% forward/backward, +1.15% with AdamW.

Recommended opt-in at the time: `SCA2_LONG_PATH=triton_codes` for training,
`triton_fused` for forward-only prefill. These are single-layer
synthetic timings, not a full language-model training throughput claim.
The AdamW arm clears gradients each step and uses lr=0 to keep weights fixed.
Paired speedups need not equal the ratios of the independently reported medians.

Raw per-round data, configuration and invocation are in
[`lapa/benchmarks/results/triton_gb10_20260915.json`](lapa/benchmarks/results/triton_gb10_20260915.json).
All **13 dedicated tests pass**, including independent gradients for every
state-loop input, compiled eight-chunk execution at M=256/dvi=272, and
decay-ceiling stress on padded chunks. Both new paths pass the GPU iso gate
in fp64/fp32. The compiled bf16 eight-chunk test's worst normalized deviation
was 0.00632 (output), below the harness's bf16 tolerance of 0.06.

The target named here at the time -- the state-loop backward's large matrix
gradients -- is what `triton_scan` addresses: not by tiling them better, but by
removing the gradient adds between graph nodes, splitting the reverse scan into
its reducing and mode-parallel halves, and halving the read-code tensor.

Correction to the strategy below: in the current `_batched`, Gram, inverse W
and masked K2 are already computed for all chunks before the sequential loop.
Only the state-dependent reads, delta correction, outputs and writes remain
sequential. Moving the solve into that loop should therefore be benchmarked,
not assumed to save launches.

### Historical inverse-only timing (before the Triton adjoint)

Target LayerScale is now **one scalar per residual branch**, following the
user's ablation results: `ls_mix_per_channel=False`; the FFN already used a
scalar. This is outside the inverse kernel, so no kernel change is needed.

Configuration: B=8, T=1024, d=1024, M=256, dv=256, kv_dk=16, C=128,
NG=2, conv=4, free decay, bf16 autocast, `torch.compile(fullgraph=True)`.
Identical initial weights for batched/triton. Five warmups per arm, then
20 paired rounds of 10 iterations, alternating arm order; synchronized
wall-clock timing. GPU had no other compute process at the start.

| Full layer | batched median ms | triton median ms | Median paired triton/batched |
|---|---:|---:|---:|
| Forward, no gradients | 11.997 | 11.959 | 0.999 |
| Forward + backward | 39.139 | 39.012 | 1.003 |
| Forward + backward + AdamW | 39.739 | 39.505 | 0.992 |

Paired ratios vary across rounds: 0.972–1.039 forward, 0.981–1.023
forward+backward, 0.981–1.015 with AdamW. There is no clear useful gain at
layer level. Ratios are medians of within-round ratios, not ratios of the
two medians. These are **single-layer synthetic benchmarks**, not full LM
training throughput. AdamW uses fused=True and lr=0 to exercise the step
while keeping weights fixed; this arm clears gradients each iteration.

An earlier 10-round run with vector attention LayerScale measured long-head
forward at 5.23/5.21 ms and forward+backward at 16.57/16.32 ms (batched/triton).
The isolated inverse measured 0.170/0.138 ms forward and 0.926/0.631 ms
forward+backward, with substantial timing variation. It is a small fraction
of total time, and that isolated improvement does not establish a layer gain.

These numbers describe the first inverse-forward prototype only. The newer
code/product and state-loop implementations are listed at the top of this file.

## 1. The Opportunity

The long head runs at **3% GPU utilisation** on the GB10 (measured: 1.8 TFLOP/s
out of 56.6 peak bf16). The layer spends 16.7 GFLOPs of actual compute in
9.3 ms; the rest is kernel launch overhead and intermediate memory traffic.

A fused Triton kernel that eliminates intermediates would bring us from
~23,500 tok/s to an estimated ~30,000+ tok/s for the full training loop,
which is worth as much as several loss-reducing architecture changes combined
(at equal wall-clock, every % of speed = a % of loss at no cost).

GDN gets its speed from `fla`'s Triton kernels (×1.66 over PyTorch naive).
We need the equivalent for our closed-form chunked linear attention.

## 2. Where Everything Lives

### Reference implementation (the ground truth to match)
- **`lapa/layer.py`**, class `LongHead`
  - `_batched()` at line ~530: the chunked closed-form, all-chunks-at-once path.
    This is the HOT PATH during training (set by `SCA2_LONG_PATH=batched`).
  - `_chunk()` at line ~450: single-chunk version, easier to read, same math.
  - `prefill()` at line ~639: dispatches to `_batched` or `_chunk`.
  - `step()`: single-token decode (separate, much simpler, not the target).

### Existing Triton prototype
- **`lapa/triton_codes.py`**: fuses phase → cos/sin → decay scales → Kk/Qk/Fq
  construction. Correctness: 0 error vs reference. Speed: 2.05 ms → 0.84 ms
  (×2.4). But this is only 21% of the long head; the rest is in the chunk loop.

### Correctness gate
- **`sca2/iso.py`**: the repo's iso harness. Run with `--self` for architecture
  candidates. Must pass at float64 (tol 1e-10) and float32 (tol 3e-4).
  ```bash
  python -m sca2.iso lapa --self --fast
  ```

### Speed benchmark
- **`lapa/benchmarks/speed.py`**: blocked-design timing with within-round ratios.
  ```bash
  python -m lapa.benchmarks.speed --sections   # long / short / FFN split
  ```

## 3. The Math (one chunk, NG=1 for clarity)

Given per-chunk inputs:
- `kz, kh`: (B, C, d) — key projections of current and previous tokens
- `vz`: (B, C, dvi) — value projection (dvi = dv + kv_dk)
- `bz`: (B, C, 1) or (B, C, M) — erase gate β = sigmoid(bproj(z))
- `state_in`: (B, 2M, dvi) — the recurrent state S

### Step 1: Phase codes (element-wise, fully parallelisable)
```
phi_w = theta * K(z_prev) + pos * omega          # (B, C, M) fp32
phi_q = theta * K(z)      + pos * omega          # (B, C, M) fp32
cw, sw = cos(phi_w), sin(phi_w)                  # (B, C, M)
cq, sq = cos(phi_q), sin(phi_q)                  # (B, C, M)
```

### Step 2: Decay scales
```
# Without decay_input (the common case):
gw = exp(lambda * idx)     # (C, M)    idx = 0..C-1
gq = exp(-lambda * idx)    # (C, M)
# lambda: (M,) learned, exp(lam_raw).clamp(max=lam_ceil)
```

### Step 3: Code blocks
```
Kk = [cw*gw | sw*gw]                              # (B, C, 2M)  write codes
Qk = [cw*gq | sw*gq]                              # (B, C, 2M)  read-back codes

# Read codes with spectral weight w = wr + i*wi folded in:
c1 = (wr*cq + wi*sq) * gq                         # (B, C, M)
c2 = (wr*sq - wi*cq) * gq                         # (B, C, M)
Fq = [[c1 | c2], [-c2 | c1]]                      # (B, 2C, 2M)  [Re ; Im]
```

### Step 4: Gram matrix + triangular solve (the delta-rule WY transform)
```
G = (Qk @ Kk^T) / M                               # (B, C, C)
A = I + beta * tril(G, -1)                         # (B, C, C) lower triangular
W = solve_triangular(A, I)                         # (B, C, C) forward substitution
```

### Step 5: State read + delta-rule correct + output
```
s0 = state_in * exp(-lambda)                       # (B, 2M, dvi)  one-step decay
r  = (Qk @ s0) / M                                 # (B, C, dvi)   state readout
e  = W @ (v - beta * r)                            # (B, C, dvi)   corrected values
K2 = (Fq @ Kk^T).masked_fill(causal_mask, 0)      # (B, 2C, C)    intra-chunk kernel
o  = (K2 @ e + Fq @ s0) / M                        # (B, 2C, dvi)  [Re ; Im] output
```

### Step 6: State update
```
state_out = state_in * exp(-lambda * C) + (Kk^T @ e) * exp(-lambda * (C-1))
```

### The sequential constraint
Steps 4-6 form a **sequential loop over K chunks** because each chunk's `state_out`
is the next chunk's `state_in`. Steps 1-3 are fully parallel over chunks and are
already batched in `_batched()` (all K chunks stacked in a single tensor).

### NG > 1 (long_groups)
When `long_groups = NG > 1`, `wr/wi` have shape `(M, NG)` and Fq becomes
`(B, NG, 2C, 2M)`. The value channels are split into NG groups of `dvi // NG`
each, and steps 5-6 are done per-group. The Gram and solve are shared (same Kk/Qk).
This is the main source of extra overhead (~10% measured).

## 4. Profiling Results

### Per-operation, per chunk (B=8, C=128, M=256, dvi=272):
```
operation (per chunk)              ms      x8 chunks     % of long head
──────────────────────────────────────────────────────────────────────────
code construction (steps 1-3)     0.25     2.00 ms        21%
gram Qk@Kk^T                     0.023    0.18 ms         2%
solve triangular                  0.106    0.84 ms         9%   ← expensive
state read Qk@s0                  0.033    0.26 ms         3%
delta e=W@(v-βr)                  0.034    0.27 ms         3%
Fq@Kk^T + mask                   0.032    0.26 ms         3%
K2@ec                             0.043    0.35 ms         4%
Fq@s0                             0.038    0.30 ms         3%
state write                       0.178    1.42 ms        15%   ← expensive
──────────────────────────────────────────────────────────────────────────
sum of measured ops                        5.88 ms        63%
launch overhead (not in any op)            3.42 ms        37%   ← THE target
──────────────────────────────────────────────────────────────────────────
total measured                             9.30 ms       100%
```

### Full layer fwd+bwd (compiled, B=8, T=1024):
```
config                              ms fwd+bwd     tok/s
nu (no kv_dk, no conv)               61.52 ms    133,151
+ all features, NG=1                  67.91 ms    120,636    +10.4%
+ long_groups=2                       74.88 ms    109,404    +21.7% total
```

### What we tried that doesn't help much:
- **CUDA graphs on fwd only**: +15% on the long head, but only +3% on fwd+bwd
  (backward dominates and already overlaps well)
- **Chunk size sweep** (64, 128, 256, 512): C=128 is already the sweet spot;
  larger chunks make the solve and Gram quadratically more expensive
- **torch.compile**: already used, gives ×2 over eager. Zero graph breaks.

## 5. Kernel Strategy

### Priority 1: Fuse the chunk loop body (steps 4-6)
One Triton kernel per chunk that takes `(Kk_chunk, Qk_chunk, Fq_chunk, v, beta, state_in)`
and produces `(output_chunk, state_out)`. This eliminates:
- 3.42 ms of launch overhead (37% of the long head)
- The solve's LAPACK call (replace with a simple forward-substitution loop)
- The dtype cast intermediates (.to(float32), .to(bfloat16))
- The masked_fill on K2

Target: **5.7 ms saved out of 9.3 ms** = 61% of the long head.

### Priority 2: Fuse code construction (steps 1-3)
Already prototyped in `lapa/triton_codes.py`. Saves 1.2 ms (21% of long head).
The prototype works for NG=1; extending to NG=2 is straightforward.

### Priority 3: Fuse the NG>1 grouped read
The `view → transpose → matmul → transpose → reshape` sequence in the NG>1 path
of step 5 is pure overhead. A kernel that does the grouped matmul natively
(each group reads its slice of dvi) avoids all the reshaping.

### What NOT to fuse
- The linear projections K(z), V(z), Kv(z): these are standard nn.Linear,
  already optimal in cuBLAS
- The FFN: standard nn.Sequential, already optimal
- The short head: 4.3 ms, small, and has a different structure (exact L-tap DFT window)

## 6. Numerical Constraints

- The **state** is always fp32 (accumulates over the full sequence).
- The **Gram, solve, and delta-rule correction** are fp32 (the triangular solve
  has no bf16 CUDA kernel, and accumulation errors in bf16 degrade the solve).
- The **code blocks** (Kk, Qk, Fq) and the **matmuls** can be bf16.
- `exp(lambda * C)` is formed: with `lam_free`, `lambda` can be up to `lam_ceil`
  = 55/128 = 0.43, so `lambda * C` ≤ 55. fp32 exp overflows at ~88. **Safe, but
  not by much** — the kernel must NOT use fp16 for the decay ramps.
- The phases `phi = theta * K(z) + pos * omega` must be fp32 (the grid is
  sensitive to truncation; bf16 quantises it to ~3-bit resolution at high M).

## 7. Shapes at d=1024 (the training config)

```
B = 8, T = 1024, M = 256, dv = 256, dk = 16, dvi = 272
C = 128 (chunk), K = 8 (chunks), NG = 2 (long_groups)
2M = 512

Code blocks:  Kk (B, K, C, 2M) = (8, 8, 128, 512) bf16
              Qk (B, K, C, 2M) = same
              Fq (B, K, [NG,] 2C, 2M) = (8, 8, 2, 256, 512) bf16
State:        s  (B, 2M, dvi) = (8, 512, 272) fp32
Values:       v  (B, K, C, dvi) = (8, 8, 128, 272) fp32
Gram:         G  (B, C, C) = (8, 128, 128) fp32
Solve:        W  (B, C, C) = (8, 128, 128) fp32
Output:       o  (B, K, 2C, dvi) = (8, 8, 256, 272) fp32 → (B, T, 2dv) after cat

Causal mask: (2C, C) = (256, 128), lower-triangular, repeated [Re ; Im].
```

## 8. How to Test

```bash
# 1. Correctness: must match the reference at float64 and float32
python -m sca2.iso lapa --self --fast

# 2. Speed: blocked-design benchmark
python -m lapa.benchmarks.speed --sections

# 3. Full training loop speed (the number that matters)
export PYTORCH_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched
python -c "
import torch, time, statistics
torch._dynamo.config.cache_size_limit = 256
from lapa.layer import LaplaceConfig, LaplaceAttention
d, M, dv, L, T, B = 1024, 256, 256, 64, 1024, 8
cfg = LaplaceConfig(d=d, M=M, dv=dv, L=L, ff=4096, chunk=128,
    rope_base=1000.0, slow_frac=0.25, max_len=T, kv_dk=16, conv=4,
    lam_free=True, mem_range=(4.0, 20000.0), long_groups=2, layer_scale=True,
    ls_mix_init=0.25, ls_ff_init=0.5, ls_mix_per_channel=False, w_antipodal=0.1)
m = LaplaceAttention(cfg).cuda()
x = torch.randn(B, T, d, device='cuda', requires_grad=True)
mc = torch.compile(m, dynamic=False)
for _ in range(5):
    with torch.autocast('cuda', dtype=torch.bfloat16): mc(x).float().square().mean().backward()
torch.cuda.synchronize()
ts = []
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16): mc(x).float().square().mean().backward()
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
print(f'{statistics.median(ts):.2f} ms  {B*T/(statistics.median(ts)/1e3):,.0f} tok/s')
"

# Current baseline: ~75 ms, ~109k tok/s (fwd+bwd, compiled, full config)
# Target: ~58 ms, ~141k tok/s (long head halved)
```

## 9. Hardware

- NVIDIA GB10 (Gigabyte Atom / Spark): SM 12.1, 128GB unified RAM
- **48 SMs, 24 MB L2** (`torch.cuda.get_device_properties`)
- 56.6 TFLOP/s bf16 peak, 194 GB/s measured bandwidth
- **292 FLOP/byte** — extremely compute-dense, bandwidth-starved
- Measured, and worth re-measuring before any redesign: 60 TFLOP/s on a large
  bf16 GEMM, 200 GB/s on a 64 MB copy, and **3.2 us per kernel launch**
  (CPU-side enqueue of a trivial kernel, which is the binding cost for a
  hundreds-of-launches graph). The L2 is large enough to hold the long head's
  whole recurrent state, which is why keeping the live state in a two-slot
  ping-pong buffer costs nothing.
- The profiler (`torch.profiler`) does NOT report CUDA kernel times on this GPU
  (SM 12.1 is barely in range). Use wall-clock timing with `torch.cuda.synchronize()`.
- `torch._dynamo.config.cache_size_limit = 256` is required (default 8 silently
  falls back to eager on our model).
- Triton 3.5.1 is installed and works (the prototype in `lapa/triton_codes.py` runs).
