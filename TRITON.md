# Triton Kernel for the Long Head — Handoff Document

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
    ls_mix_init=0.25, ls_ff_init=0.5, ls_mix_per_channel=True, w_antipodal=0.1)
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
- 56.6 TFLOP/s bf16 peak, 194 GB/s measured bandwidth
- **292 FLOP/byte** — extremely compute-dense, bandwidth-starved
- The profiler (`torch.profiler`) does NOT report CUDA kernel times on this GPU
  (SM 12.1 is barely in range). Use wall-clock timing with `torch.cuda.synchronize()`.
- `torch._dynamo.config.cache_size_limit = 256` is required (default 8 silently
  falls back to eager on our model).
- Triton 3.5.1 is installed and works (the prototype in `lapa/triton_codes.py` runs).
