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


# Imported one at a time, so a module whose optional dependency is absent costs
# only its own variants. This used to be a single try/except ImportError around
# the whole block: on a fresh host where flash-linear-attention had been
# installed --no-deps, arch_mamba2 raised, the except swallowed it, and EVERY
# variant vanished -- `lapa_cc` included, with a KeyError as the only symptom
# and nothing pointing at the real cause.
_VARIANT_MODULES = [
    ("variants",      "historical experiments"),
    ("arch_sepq",     "architecture candidate"),
    ("arch_fixdecay", "convolution-form candidate"),
    ("arch_cumsum",   "original SeqCond temporal form"),
    ("arch_gdn",      "Gated DeltaNet baseline"),
    ("arch_keyed",    "key-addressed delta rule"),
    ("arch_gatedc",   "gated multi-head C head"),
    ("arch_wgroup",   "per-value-group spectral weights"),
    ("arch_cdelta",   "complex error-correcting C write"),
    ("arch_hybrid",   "cdelta C head + GDN D head"),
    ("arch_short",    "cdelta C head + short dft C head"),
    ("arch_damp",     "cdelta with per-mode decay (Laplace)"),
    ("arch_gdn2",     "Gated DeltaNet-2 baseline (via lapa)"),
    ("arch_lapa",     "Laplace Attention v1 (via lapa), the fast path"),
    ("arch_mamba2",   "Mamba2 (SSD) baseline via fla"),
    ("fast_dhead",    "loop-free D head (iso with sepq/polar)"),
    ("triton_dhead",  "fused kernel (needs triton)"),
]

#: {module name: the exception that kept it out}, for diagnosing a thin registry.
UNAVAILABLE: dict[str, str] = {}


def _load_variants():
    """Import experimental variants and architecture candidates."""
    import importlib
    import warnings
    for name, note in _VARIANT_MODULES:
        try:
            importlib.import_module(f".{name}", __package__)
        except Exception as exc:                        # noqa: BLE001
            UNAVAILABLE[name] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"sca2 variant module {name} ({note}) unavailable: {exc}")


_register_versions()
_load_variants()
