# lapa — Laplace Attention

A sequence-mixing layer that replaces attention. Two phase-coded memories share
one equation: a long one (a learned Laplace transform of the token stream, half
its spectrum persistent, half forgetting in a few tokens) and a short one (an
exact 16-token window on the DFT grid). No softmax, no KV cache: decode is O(1)
per token with ~23k floats of state per layer at d=128.

Provenance, derivation and every measurement: `../CATCHUP.md`, `../WINNERS.md`,
`../chead_numpy.py` (the math in 250 lines of numpy).

## Use it

```python
from lapa import LaplaceAttention, LaplaceConfig, LaplaceLM

# a block, drop-in for an attention block: (B, T, d) -> (B, T, d)
layer = LaplaceAttention(LaplaceConfig(d=512, M=256, dv=256, L=16, ff=2048))
y = layer(x)

# streaming: prefill a prompt, then one token at a time with constant memory
y, state = layer.prefill(x)
y_t, state = layer.step(x_t, state)

# a whole model
lm = LaplaceLM(vocab=32000, d=512, n_layers=12, M=256, dv=256, ff=2048)
logits = lm(tokens)                       # (B, T, V), for training
ids = lm.generate(prompt_ids, n=200)      # token by token
```

Any module with `prefill(x, state) -> (y, state)`, `step(x_t, state) -> (y_t, state)`
and `init_state(B, device)` stacks in `lapa.LM` — the baselines in `benchmarks/`
do, so a model can mix layer types.

## Config knobs that matter

| knob | default | meaning |
|---|---|---|
| `d` | 128 | model width |
| `M` | 190 | long-head modes = frequency resolution. **Not tied to d**; set by copy capacity (see benchmarks) |
| `dv` | 56 | value width; each head emits 2·dv |
| `L` | 16 | short-head window, tokens; the read is a banded GEMM, so 64 costs ~1.1× 16 (copy up to L−2 tokens is exact by construction) |
| `ff` | 448 | FFN width (GELU) |
| `persist` | 0.5 | fraction of long-head modes with λ = 0 (infinite memory), lowest frequencies |
| `chunk` | 128 | prefill chunk; `lam_max · chunk ≲ 60` keeps float32 safe |
| `long_path` | batched | `batched` (all chunks' intra-chunk work in one call) or `chunk`; pick with `python -m sca2.autotune` on your GPU |
| `gemm_dtype` | None | dtype of the two big GEMMs; None follows `torch.autocast` |

## Precision

State, phases, codes, Gram and triangular solve are always float32 (float64 if
the module is). The two large GEMMs of the long head and all linear layers follow
`torch.autocast`. Under bf16 autocast the output deviates from float32 by ~3e-3
relative. The short head is float32 (its Dirichlet comb relies on exact
cancellation). On GPUs without bf16 tensor cores (Turing), bf16 is *slower*.

## Verify

```bash
python -m lapa.layer        # == repo fast path in float64 (9e-16); decode == prefill; bf16 deviation
```

## Numbers (seed 0, 4 layers, d=128, ~186k params/layer, pycode, matched to Gated DeltaNet)

- val loss −0.03..−0.05 nats below GDN from 300M to 1.4B tokens (GDN had overtaken the previous generation at 300M)
- 4-layer stack fwd+bwd: 0.71× GDN's time (blocked timing, RTX 2070, vs fla's *naive* reference)
- decode: ~2.7× faster than GDN's Triton path, 0.09 MB state per layer

## Speed at d=1024 (DGX Spark / GB10, one layer, fwd+bwd, blocked, both compiled, bf16)

B=8, T=1024, LapA M=256 dv=256 L=64 ff=4096 vs GDN 8×128 ff=4096:

| arm | ms | vs GDN | tok/s | params | mixer | state |
|---|---|---|---|---|---|---|
| GDN, fla **Triton** kernel | 41.7 | 1.00× | 196k | 13.67M | 5.27M | 140k |
| **LapA** | **29.4** | **0.71×** | **279k** | 10.30M | 1.90M | 156k |
| GDN, fla naive reference | 71.2 | 1.70× | 115k | 13.67M | 5.27M | 140k |

Two things to read carefully. First, the bottom row is the comparison every speed
number in this repo used before the Spark, because fla's Triton kernels do not build
on sm_75 — they are worth 1.70× and no claim should rest on the naive reference.
LapA holds 0.71× against the *real* kernel. Second, the three right-hand columns are
not matched and cannot all be: LapA is 25% smaller in total parameters with a 2.8×
smaller mixer, and pays 1.11× the decode state. See `../SPARK.md` §4.

Reproduce: `python -m lapa.benchmarks.vs_gdn`, `python -m lapa.benchmarks.speed`.

The benchmarks that produced these, and the ones to run at scale: `benchmarks/README.md`.
Scaling to d = 1024–2048 (what is known, what is not, shapes, speed protocol, traps):
`../SPARK.md`.
