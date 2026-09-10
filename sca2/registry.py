"""Variant registry. Every optimized implementation registers here and is
automatically picked up by iso.py and bench.py."""
from .ref import CHeadRef, DHeadRef, SCA2Layer, LayerCfg
import torch

VARIANTS = {}


def register(name, c_cls=None, d_cls=None, note="", wrap=None, layer_cls=None,
             arch=False):
    """Register a (C head, D head[, Layer]) triple. None means 'use v0'.

    `wrap` is an optional post-build callable (e.g. a torch.compile wrapper); it
    must preserve the prefill/step contract, and iso.py checks that it does.

    `arch=True` marks an ARCHITECTURE CANDIDATE: a different function with a
    different parameter set, not a faster schedule for the same one. It can
    never be iso-checked against the reference -- only against itself, for
    prefill/decode consistency -- and iso.py refuses to pretend otherwise.
    """
    VARIANTS[name] = {"c": c_cls or CHeadRef, "d": d_cls or DHeadRef,
                      "note": note, "wrap": wrap, "layer": layer_cls or SCA2Layer,
                      "arch": arch}
    return name


def build(name, cfg=None, seed=0, device="cpu", dtype=torch.float32):
    v = VARIANTS[name]
    cfg = cfg or LayerCfg()
    torch.manual_seed(seed)
    m = v.get("layer", SCA2Layer)(cfg, v["c"], v["d"]).to(device=device, dtype=dtype)
    return v["wrap"](m) if v.get("wrap") else m


register("ref", note="frozen reference (contract)")


def _register_versions():
    """One canonical name per version, plus its compiled form."""
    from . import versions
    from .compiled import wrap as compile_wrap
    for name in versions.ORDER:
        m = versions.module(name)
        register(name, m.CHead, m.DHead, note=m.NOTE, layer_cls=m.Layer)
        if name != "v0":
            register(f"{name}_cc", m.CHead, m.DHead, note=m.NOTE + " + compile",
                     layer_cls=m.Layer, wrap=compile_wrap)


def _load_variants():
    """Import experimental variants and architecture candidates."""
    from . import variants     # noqa: F401  historical experiments
    from . import arch_sepq     # noqa: F401  architecture candidate
    from . import arch_fixdecay # noqa: F401  convolution-form candidate
    from . import arch_cumsum   # noqa: F401  original SeqCond temporal form
    from . import arch_gdn      # noqa: F401  Gated DeltaNet baseline
    from . import arch_keyed    # noqa: F401  key-addressed delta rule
    from . import arch_gatedc   # noqa: F401  gated multi-head C head
    from . import arch_wgroup   # noqa: F401  per-value-group spectral weights
    from . import arch_cdelta   # noqa: F401  complex error-correcting C write
    from . import arch_hybrid   # noqa: F401  cdelta C head + GDN D head
    from . import arch_short    # noqa: F401  cdelta C head + short dft C head
    from . import arch_damp     # noqa: F401  cdelta with per-mode decay (Laplace)
    from . import arch_gdn2     # noqa: F401  Gated DeltaNet-2 baseline (via lapa)
    from . import fast_dhead    # noqa: F401  loop-free D head (iso with sepq/polar)
    try:
        from . import triton_dhead  # noqa: F401  fused kernel (needs triton)
    except Exception as e:  # triton missing or unsupported GPU
        import warnings
        warnings.warn(f"triton D-head unavailable: {e}")


_register_versions()
try:
    _load_variants()
except ImportError:
    pass
