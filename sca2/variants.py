"""
Historical and experimental variants -- NOT versions.

The versions live in sca2/versions/ and are the drop-in path (`SCA2_VERSION`).
What is here is the audit trail: head-isolation variants used to attribute a
speedup to one head rather than the other, the chunk-size sweep, and the
measured-and-rejected experiments. Everything is prefixed `x_` so it can never
shadow a version name.

Class names are re-exported for back-compat with anything that imported them
from this module before the versions/ split.
"""
from .ref import CHeadRef, DHeadRef
from .registry import register
from .compiled import wrap as _compile_wrap
from .versions.v1_quad_scan import (      # noqa: F401  (back-compat re-exports)
    causal_mask, CHeadQuad, DHeadChunk, DHeadScan,
    DHeadScan4, DHeadScan8, DHeadScan16, DHeadScan32, DHeadScan64, DHeadScan128,
    DHeadScanQ4, DHeadScanQ8, DHeadScanQ16,
)

# --- head isolation: one head optimized, the other left at v0 -------------- #
register("c_quad", CHeadQuad, DHeadRef, note="v1 C head only")
register("d_chunk", CHeadRef, DHeadChunk, note="first chunked D scan (bad layout)")
for _c in (4, 8, 16, 32, 64, 128):
    register(f"d_scan{_c}", CHeadRef, globals()[f"DHeadScan{_c}"],
             note=f"v1 D head only, chunk={_c}")

# --- v1 chunk-size sweep --------------------------------------------------- #
# eager prefers large chunks (pays per launch), compiled prefers small (pays per
# byte): the optimum moved from 16 to 8. Measured, not guessed.
for _c in (4, 8, 16, 32, 64):
    register(f"x_v1c{_c}", CHeadQuad, globals()[f"DHeadScan{_c}"],
             note=f"v1 eager, chunk={_c}")
    register(f"x_v1c{_c}_cc", CHeadQuad, globals()[f"DHeadScan{_c}"],
             note=f"v1 compiled, chunk={_c}", wrap=_compile_wrap)

# --- REJECTED: hoisting q out of v1's python chunk loop -------------------- #
# +10% prefill, -5x backward (train 54.8 vs 10.6 ms). 16 narrow python slices of
# a retained 2.B.T.M.dv tensor each scatter-add in the backward. v2 gets the
# same GEMM merge for free by removing the python loop entirely.
for _c in (4, 8, 16):
    register(f"x_hoistq{_c}_cc", CHeadQuad, globals()[f"DHeadScanQ{_c}"],
             note=f"REJECTED hoisted q, chunk={_c}", wrap=_compile_wrap)


# --- REJECTED: loop-free two-level scan (was v2's first design) ------------- #
# 2.6x slower on prefill (6.31 vs 2.43 ms compiled), 89 vs 54 MB. The python
# chunk loop was doing cache blocking, and the eager profile that motivated
# removing it had already been fused away by torch.compile. See
# versions/v2_fused.py's docstring.
from .versions.v2_fused import DHeadLoopFree, CHead as _V2CHead, Layer as _V2Layer
register("x_loopfree", _V2CHead, DHeadLoopFree,
         note="REJECTED loop-free two-level scan", layer_cls=_V2Layer)

# dynamic compilation: one specialization for all shapes, so a shape sweep can
# never exhaust the recompile budget and fall back to eager unnoticed
from .compiled import wrap_dynamic as _cwd  # noqa: E402
from .versions.v1_quad_scan import CHeadQuad as _CQ, DHeadScan8 as _D8  # noqa: E402
register("v1_dyn", _CQ, _D8, note="v1 + compile(dynamic=True)", wrap=_cwd)
