# Note for the Spark agent — scaling Laplace Attention to d = 1024–2048

You are taking a layer that has been developed and measured at d=128 on an RTX 2070 and
running it at 10–20× the width on a DGX Spark (Blackwell, bf16 tensor cores, Triton
works). This note tells you what is known, what is *not* known, what to measure first,
and where the traps are. Read `CATCHUP.md` for the full chronological record (including
the retractions) and `WINNERS.md` for the method rules — those rules were paid for.

---

## 1. What the layer is

**Laplace Attention v1** = `lapa.LaplaceAttention` with
`LaplaceConfig(rope_base=1000, slow_frac=0.25, L=64)`, or in the training pipeline
`--variant cshort_damph_cc --rope-base 1000 --slow-frac 0.25 --Ls 64`.

Two phase-coded memories sharing one equation (`chead_numpy.py` has all of it in 250
lines of numpy, self-checked):

- **Long head** — a learned Laplace transform of the token stream. Write code
  `c_s = exp(i(θ·K(h_s) + s·ω))`, unit modulus; error-correcting (delta-rule) write;
  per-mode decay `λ_m`. The grid is **mixed**: 25% slow integrators (periods 2T..20T,
  the document memory — they carry 55–85% of the trained LM's state energy) and 75% dense
  rope at base 1000 (the addressing range; rule: *unaliased range 2·base ≈ context*).
  Half the modes are pinned at λ=0, the other half learn to forget.
- **Short head** — a DFT-grid window of `L` tokens whose Dirichlet comb is an *exact* tap
  at every lag < L. Ring-buffer state, additive write, banded-GEMM read. Copies up to
  L−2 tokens exactly by construction, and composes across depth (2 layers ⇒ ~2(L−1)).

Verify any change with **both** gates, they catch different things:

```bash
python -m lapa.layer                      # standalone file == repo fast path, float64, 9e-16
python -m sca2.iso cshort_damph --self    # prefill == token-by-token decode
SCA2_CTX_CHUNK=32 python -m sca2.iso cshort_damph --self   # exercises the batched path
python -m sca2.arch_damp ; python -m sca2.arch_short       # vs independent token-loop references
```

The iso gate alone cannot catch a kernel that is wrong *the same way* on both paths — that
happened once and only the reference test found it.

---

## 2. What is established at d=128 (4 layers, ~186k params/layer, pycode corpus)

| result | number | where |
|---|---|---|
| LM vs Gated DeltaNet, 1.4B tokens, XL corpus | **−0.031 nats** (median of last 25%) | `runs/long5h.jsonl`, `plot_lm.py` |
| copy, exact-string at L=512, 12k steps | **0.85** vs GDN **0.00** | `runs/copy_d128.jsonl` |
| 4-layer stack fwd+bwd, blocked timing | **0.75× GDN's time** | `python -m sca2.autotune` |
| decode | **285 µs/token vs 780** (GDN Triton), state 0.09 MB/layer | `sca2/bench_decode_scaling.py` |

Settled by single-change arms (do not re-litigate; `CATCHUP.md` has each): θ init, damping
on/off, the delta rule, rope 1e5, windows 16/32/128, `gated_read`, raw read, bounded phase.

---

## 3. What is NOT known, in priority order

**(a) Does M have to grow with d?** This is the single most important question and the
whole speed argument rests on it. LapA's mixer is **linear in d** if M and dv stay fixed;
GDN's is **quadratic** (all its projections are d × H·dk). At d=128 LapA wins with a
mixer 1.5× *smaller* than GDN's. At d=1024, if M must scale with d to keep quality, that
advantage evaporates. Run the copy sweep **first**, it is cheap:

```bash
# lapa/benchmarks/README.md has the full protocol; adjust batch to the GPU
C="--d 1024 --layers 2 --ff 4096 --lengths 128,256,512,1024 --symbols 256 \
   --batch 16 --steps 6000 --lr 5e-4 --dtype bf16 --compile --log runs/copy_d1024.jsonl \
   --rope-base 2048 --slow-frac 0.25 --L 64"
for M in 128 256 512 1024; do
  python -m lapa.benchmarks.copy run --arm lapa --label "LapA M=$M" --M $M --dv 256 $C
done
python -m lapa.benchmarks.copy run --arm gdn  --label "GDN 8x128" --gdn-heads 8 --gdn-head-k 128 $C
python -m lapa.benchmarks.copy run --arm gdn2 --label "GDN2 8x128" --gdn-heads 8 --gdn-head-k 128 $C
python -m lapa.benchmarks.copy plot runs/copy_d1024.jsonl --out plot/copy_d1024.png
```

If M=256 still copies L=512 at d=1024, M is set by *capacity*, not by width, and the
linear-in-d mixer is real. If the cliff tracks M/d, report it immediately — it changes the
architecture's story.

**Note the `--rope-base 2048`**: the rule is 2·base ≈ context, and these runs use
T = 2·1024+2 ≈ 2050. At d=128/T=1024 the right base was 1000. Do not carry 1000 over
blindly to a longer context.

**(b) Is GDN's Triton kernel much faster than fla's reference?** Every speed number in
this repo compares our inductor path to **fla's naive PyTorch reference**, because
`fla`'s Triton kernels do not run on sm_75. On the Spark they do. Re-measure honestly:
GDN with `chunk_gated_delta_rule` (Triton) vs LapA with inductor. Expect their mixer to
gain 2–3× at small shapes; our 0.75× could become parity. Then decide whether to write
our own fused kernel (the codes and the K2 matrix are still materialised — that is where
our remaining margin is).

**(c) Does the FFN/mixer split favour us?** At d=128, matched by total parameters, GDN
puts 64% of its budget in the mixer and LapA 41–46%. If the FFN matters less than the
mixer at scale, param-matching *helps GDN* — which makes our win more solid, not less. But
report **three columns side by side** (total params, mixer params, decode state) rather
than picking one axis. There is no shape that matches all three: at d=1024, ff=4096,
matching GDN 8×128's mixer (5.27M) needs M=2000 at dv=512, which gives LapA **15× GDN's
state**. Parameters and state are different resources for these two architectures.

**(d) bf16 in anger.** The layer has a precision policy (`lapa/layer.py` docstring): state,
phases, codes, Gram and triangular solve always fp32; the two big long-head GEMMs follow
autocast. Measured deviation under bf16 autocast vs fp32: 2.6e-3 relative. Never trained in
bf16. On Turing bf16 was *2.7× slower* (no bf16 tensor cores) — on Blackwell it should be
the default. Watch the delta-rule solve: it is the one place where reduced precision could
bite, and it is already forced to fp32.

---

## 4. Suggested shapes

Per-layer parameter counts (mixer / FFN / decode state in floats):

| config | total | mixer | FFN | state |
|---|---|---|---|---|
| LapA d=1024, M=256, dv=256, ff=4096 | 10.30M | 1.91M | 8.40M | 156k |
| LapA d=1024, M=512, dv=256, ff=4096 | 10.56M | 2.17M | 8.40M | 287k |
| LapA d=1024, M=256, dv=512, ff=4096 | 11.87M | 3.48M | 8.40M | 304k |
| GDN d=1024, 8×128, ff=4096 | 13.67M | 5.27M | 8.40M | 140k |
| GDN2 d=1024, 8×128, ff=4096 | ~15M | ~6.7M | 8.40M | 140k |
| LapA d=1536, M=384, dv=384, ff=6144 | 23.12M | 4.23M | 18.89M | 329k |
| LapA d=2048, M=512, dv=512, ff=8192 | 41.05M | 7.48M | 33.57M | 567k |

Whole models, 16 layers, vocab 32k untied (add ~2·32768·d for embedding + head):

| | layers | emb+head | total |
|---|---|---|---|
| d=1024 × 16 | 165M | 67M | **232M** |
| d=1536 × 16 | 370M | 101M | **471M** |
| d=2048 × 16 | 657M | 134M | **791M** |

For a 1–2B model, go to d=2048 with 24–32 layers, or d=2560 with 24. Reference points:
LFM2-1.2B is d=2048, 16 layers, FFN 12288 (SwiGLU, auto-adjusted to ~8192), vocab 65536,
10 conv layers + 6 attention. Their `conv_L_cache=3` is the analogue of our short head, and
their 6 attention layers the analogue of our long head — a natural comparison to make.

**To match GDN at d=1024** (ff 4096 both sides): by total parameters, LapA needs dv=512
with M=2000, which also matches its mixer exactly and gives 15× the state. By a fixed
M=256/dv=256, LapA is 25% *below* GDN in total parameters — the honest thing is to run
both and report the three columns.

---

## 5. Speed: what to do before the long runs

1. **`python -m sca2.autotune`** on an idle GPU. It times `path × chunk size` in a
   **blocked design** (every cell inside every round, only within-round ratios kept) and
   prints `SCA2_LONG_PATH` / `SCA2_CTX_CHUNK` for that machine. At d=128 the answer was
   `batched` / `128` (1.39× the per-chunk path). It may differ on Blackwell. Extend
   `--chunks` to 256,512 at larger d.
2. **Sequential timing is not a measurement.** On the 2070, re-running the *same* config
   three times gave 71.6k / 65.1k / 76.7k tok/s — 15% apart, non-monotone. Cumulative
   tok/s printed by `pretrain.py` is for progress, not for comparisons. Any speed claim
   must come from a blocked design.
3. **Raise `torch._dynamo.config.cache_size_limit`** (64 is set in the benchmarks). At the
   default of 8, Dynamo silently falls back to eager after 8 recompiles and the throughput
   cliff looks like an architectural effect. That cost a day once.
4. **`lam_max · chunk ≲ 60`** — the chunked closed form materialises `e^{λ·chunk}` and
   overflows fp32 beyond ~88. The default `lam_max = 1/L` is safe for any chunk ≤ 4L.
5. Known remaining margin in our code: the long head still materialises the scaled codes
   and the `K2` kernel (B,K,2C,C). A fused kernel is the next 2× if it is worth the time.

---

## 6. Training protocol that produced the results

`pretrain.py`, single pass over a long-document corpus, equal wall-clock per arm, warmup
untimed, both arms compiled:

```bash
python -u pretrain.py --label <name> --seed 0 --variant cshort_damph_cc \
  --data <corpus>.pt --block 1024 --batch 8 --d 128 --layers 4 --Md 4 --G 8 --freq rope \
  --Mc 190 --dv 56 --ff 424 --Ls 64 --theta-scale 0.02 --rope-base 1000 --slow-frac 0.25 \
  --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8 --class-eval \
  --log runs/<log>.jsonl --save runs/ck_<name>
```

- `--class-eval` logs the loss **by token class** (word seen earlier in the window vs new,
  keyword, punctuation, whitespace). It is free and it is what diagnosed every mechanism in
  this campaign. Keep it on. It needs the BPE files (`--bpe <prefix>`).
- `--pos-buckets 8` logs the loss by position in the window: the long-context instrument.
- `--warmup N --cosine` now exist (cosine over one pass of the corpus). The whole campaign
  ran at constant lr 1e-3 without warmup; at scale you should use them, but if you compare
  against the d=128 numbers, know that they did not.
- Read with `python plot_lm.py runs/<log>.jsonl --ref <gdn arm>` — it prints, per arm, the
  median gap over the last quarter, the gap's slope vs ln(tokens), and perplexity.

**How to read the curves** (this is where most mistakes happened):
- Compare at **matched tokens** (the plotter interpolates), and separately at matched wall
  clock if that is the question. They are different claims.
- Use **medians over a window**, never a single eval. Per-eval noise is ±0.02–0.05 nats.
- **Eval spikes** are duplicated documents in the corpus: the train loss at that step drops
  to 0.3 and the next eval recovers. Check the train loss before reading a spike as a trend.
- Nothing is a verdict before ~90% of the budget.
- A seed changes a run's **level** far more than its **shape**: gaps reproduced across seeds
  to 0.001 in the 537M long run. One long convergence run beats two short seeds for reading
  whether a gap closes — but a headline number still wants n≥2.

---

## 7. Traps specific to this repo

- **Never edit a `.sh` while bash is executing it** — bash re-reads by byte offset and the
  next command shifts. Write a new file.
- **`pgrep -f` / `pkill -f` match your own command line.** A queue script that waits with
  `while pgrep -f "pretrain.py"` waits on itself forever, and `pkill -f` kills the shell
  that issued it (exit 144). Wait on a **log marker** (`until grep -q "DONE" runs/x.log`)
  and kill by PID after checking `comm == python`.
- The `sca2/` package is the research pipeline (registry, variants, iso gate); `lapa/` is
  the sanctuary (standalone layer, LM stacking, clean benchmarks). `lapa/layer.py` and
  `sca2/arch_short.py` + `sca2/arch_damp.py` are **mirrors**: a change to one must be
  copied to the other, and `python -m lapa.layer` checks that they agree in float64.
- Corpora and checkpoints are gitignored. `prep_pycode.py` + `prep_longdoc.py` rebuild the
  code corpus; `prep_fineweb.py` the web one. The BPE files *are* tracked, deliberately: the
  token stream must be reproducible across machines.

---

## 8. If you have time for only three things

1. The copy sweep at d=1024 (question (a)) — it is 6 arms × ~1 h and it decides the
   scaling story.
2. One 5 h LM run per arm at d=1024, LapA v1 vs GDN vs GDN-2, three columns reported.
3. The blocked speed comparison against fla's **Triton** kernels, which nobody has ever
   been able to run here.

Everything else (learn_persist, the key-verification gate, seeds, hybrids) is documented in
`CATCHUP.md` and can wait.

---

## 9. Answers so far (spark branch, GB10, 2026-09-11)

### The machine, because it reframes everything above

| | RTX 2070 | GB10 |
|---|---|---|
| bf16 matmul (4096³) | ~28 TFLOP/s | **56.6 TFLOP/s** |
| memory bandwidth (measured) | 448 GB/s | **194 GB/s** |
| **FLOP per byte** | ~62 | **~292** |

The Spark is ~5x more compute-rich *relative to memory* than the machine this layer
was tuned on. A materialised intermediate costs ~5x more relative to a GEMM, and an
fp32 GEMM costs 3.4x its bf16 equivalent (tf32 is off by default: 16.8 / 37.9 / 56.6
TFLOP/s for fp32 / tf32 / bf16). Optimisation targets here are traffic and dtype, not
FLOPs — the long head's GEMMs ran at ~8% of roofline before this work and still do.

### (b) Is GDN's Triton kernel much faster than fla's reference? **Yes, 1.66x.** Settled.

`fla` 0.5.2 installs and its Triton kernels run on Blackwell (`chunk_gated_delta_rule`
agrees with fla's own naive reference to 4.7e-3, i.e. bf16 level). This is the first
time anything in this repo has been measured against them. One layer, B=8, T=1024,
d=1024, fwd+bwd, all four arms in one blocked design, all compiled, all bf16:

| arm | ms | vs GDN-Triton | tok/s |
|---|---|---|---|
| GDN 8x128, fla **Triton** | 42.52 | 1.000x | 193k |
| GDN 8x128, fla naive ref | 70.61 | 1.661x | 116k |
| LapA M=256 dv=256, **before** this branch | 56.56 | **1.330x — behind** | 145k |
| LapA M=256 dv=256, **after** | **29.43** | **0.692x** | **278k** |

Read the third row before quoting the fourth. Against GDN's *real* kernel the layer
as it stood was **1.33x slower** at d=1024; the 0.75x at d=128 was measured against
the naive reference and does not survive the move. After the work on this branch LapA
is 1.44x faster than the Triton kernel. **Every speed claim in CATCHUP.md, WINNERS.md
and `lapa/README.md` that predates this compares against the naive reference and is
worth 1.66x less than it reads.** `--gdn-kernel naive` reproduces the old comparison.

Parameters and state at that shape, since no single axis is matched (§4): LapA 10.30M
total / 1.90M mixer / 156k state, GDN 13.67M / 5.27M / 140k. LapA is faster *and* 25%
smaller with a 2.8x smaller mixer, at 1.11x the state.

### (d) bf16 in anger. **Fine, and it is now the default under autocast.**

The policy is now "the state and the phases are fp32, everything else follows
autocast" (`lapa/layer.py` docstring). Widening it from "the two big GEMMs" to the
codes, the Gram, the state reads/writes and the short head left the deviation vs fp32
at ~3e-3, the figure this repo already documented. Still fp32 and staying so: the
recurrent state, the delta-rule solve that writes into it, and the phase sum (`p.omega`
reaches ~1e3 radians). Set `gemm_dtype=torch.float32` to restore the old behaviour.

The short head was fp32 on the grounds that its Dirichlet comb needs exact
cancellation. Measured rather than assumed: the cancellation is carried by the fp32
accumulator *inside* the tensor-core GEMM, not by the operands, and bf16 codes leave
the exact tap at cosine similarity 0.999998 (min 0.999996). The caution was over-strict.

### Speed protocol on this machine (replaces §5.1 for GB10)

- **Chunk size is flat here**: 128 / 256 / 512 land within 5%, `batched` slightly ahead
  of `chunk`. The 2070's 1.39x for batched/128 does not reproduce; that axis is not a
  lever on GB10 and the default `chunk=128` is fine. (`python -m lapa.benchmarks.speed
  --chunks 64,128,256,512 --paths batched,chunk`)
- **`torch.compile` is worth 1.8x and is not optional.** Eager 53.1 ms vs compiled
  29.2. Also: fp32 autocast-off costs 2.2x (65.6 ms) — never benchmark without bf16.
- `mode="reduce-overhead"` (CUDA graphs) trips on parameter-gradient accumulation.
- CUPTI does not work here: `torch.profiler` returns an empty table. Use ablation
  ladders and the blocked timer in `lapa/benchmarks/speed.py` instead.
- The FFN is now ~half the layer's time and is within ~1.5x of its own GEMM roofline
  (its two GEMMs measure 51-57 TFLOP/s, i.e. at peak). Further layer-level speedup has
  to come from the mixer or from the FFN shape, not from tuning.

### What is NOT answered

**(a) Does M have to grow with d?** Untouched — still the most important question, and
the copy sweep in §3 is still the thing to run first. Nothing here bears on it.

**(c) The FFN/mixer split**, beyond the three columns reported above at one shape.

Also untouched: any LM run at d=1024. The numbers above are one layer, fwd+bwd,
synthetic input. They say the layer is fast; they say nothing about whether it learns
at this width.

### Traps found here, to add to §7

- `sca2/arch_gdn.py` hardcoded one machine's site-packages path to fla's naive
  reference, and `registry._load_variants()` swallows `ImportError`. With `fla` absent
  the registry silently held 45 variants instead of 105 — `cshort_damph` and `gdn`
  among the missing — so **every gate in §1 raised KeyError and the whole variant
  system was a no-op**, quietly. Fixed via importlib, with `$SCA2_FLA_NAIVE` to
  override. If a gate in §1 fails with `KeyError`, suspect this before the layer.
- torch 2.9.1+cu130 warns that sm_121 is outside its supported range (max 12.0) on
  every import. It is noise; everything works, including Triton 3.5.1.
