# Versions

Each module here exports the same three names — `CHead`, `DHead`, `Layer` — plus
a one-line `NOTE`. Parameter names and shapes are identical across versions, so a
`state_dict` is portable between them and switching is a drop-in swap, never a
call-site change.

```python
from sca2 import SCA2Layer, LayerCfg, make_layer
layer = make_layer(LayerCfg(d=128), device="cuda")
y, state = layer.prefill(x)          # (B,T,d) -> (B,T,d)
y_t, state = layer.step(x_t, state)  # (B,d)   -> (B,d)
```

| env | effect |
|---|---|
| `SCA2_VERSION=v0\|v1\|v2` | choose implementation (default: `v2`) |
| `SCA2_COMPILE=0` | disable the `torch.compile` wrapper |
| `SCA2_CHUNK=16` | override the D-head chunk size |

Explicit argument beats environment: `make_layer(cfg, version="v1", chunk=16)`.

| version | status | what changed | outcome |
|---|---|---|---|
| **v0** `v0_ref.py` | reference | — | The semantics contract. `test_fidelity.py` proves it reproduces the original `bench_tinypython.py` exactly. |
| **v1** `v1_quad_scan.py` | **recommended** | C head → causal attention with a structured complex kernel; D head → chunked log-space scan | prefill **21.5x**, training step **21.1x**, graph decode **10.9x** over v0. |
| **v2** `v2_fused.py` | regression | one merged projection gemm, `h`-projections by shift | **Slower.** train 18.7 vs 10.8 ms, graph decode 174 vs 101 µs. Kept for the record. |

`DEFAULT` is always the *recommended* version. A version that measured worse
stays in the tree — `sca2.versions.table()` prints the status — but never becomes
the default.

### Three measured negative results, and the pattern in them

| experiment | idea | outcome |
|---|---|---|
| `x_hoistq*` | pull `q` out of the chunk loop as one big gemm | +10% prefill, **−5x backward**: 16 narrow python slices of a retained `2·B·T·M·dv` tensor each scatter-add |
| `x_loopfree` | chunk axis as a tensor axis, two-level scan, no python loop | **−2.6x prefill**, 89 vs 54 MB: the loop was doing cache blocking, and the eager profile that motivated it had already been fused away by `torch.compile` |
| `v2` | merge the 5 once-per-sequence projections into one gemm | **−1.7x train**: kernel count did drop (1191 vs 1233) but gemm time *rose* 7.03 → 9.24 ms — one wide gemm (K=128, N=448) is less efficient than five narrow ones, and the cat/split boundaries cost 0.22 → 0.62 ms |

All three chased *scheduling*. In the compiled regime inductor has already taken
those wins; what is left is gemm efficiency on tiny matrices, which is a **shape**
problem. The next lever is therefore to change the shapes — see the separable
D-head query in the top-level README.

Every version passes `python -m sca2.iso` — prefill, token-by-token decode,
prefill-then-decode continuation, CUDA-graph decode, and every gradient — at
1e-16..1e-14 in float64. They are the same computation on different schedules.

## Adding a version

1. New module exporting `CHead`, `DHead`, `Layer`, `NOTE`.
2. Add it to `MODULES` and `ORDER` in `__init__.py`; bump `DEFAULT` only once it
   is measured to be better.
3. `python -m sca2.iso <name> --dtypes float64,float32 -v` must be clean. If the
   change alters parameters or semantics it is **not** a version — it belongs
   behind its own A/B (see `ab_freq.py` for the pattern).
4. `python -m sca2.bench v0 v1 v2 <name>` on an **idle** GPU. Contention moved
   these numbers by 2x during development, in both directions.

## Two traps this layer has already sprung

**A host-side number in the state.** `CHeadRef.step` reads
`float(state["pos"])`. Eagerly fine; under CUDA-graph capture the value is frozen
and every replayed token decodes at position 0 — 0.52 relative error, invisible
to an eager test. Keep `pos` a 0-dim tensor; `decode.py` refuses to capture
otherwise, and `iso.py::graph_iso` checks it.

**Masking after `exp`.** Both scan levels build a decay matrix as
`(cumsum difference).exp() * mask`. On kept entries the exponent is ≤ 0, but on
*masked* entries it can be large and positive, and `inf * 0 = NaN`. Zero-padding
a ragged chunk makes this reachable: a padded gate gives `|a|² = tiny`, so
`la = -354` in fp64 and a masked exponent of `+2620`. Clamp to `max=0` **before**
`exp` — exact where it matters, finite where it does not.
