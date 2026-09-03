"""
SCA2 layer -- versioned, iso-gated implementations.

Drop-in use. `SCA2Layer` is the ACTIVE version's layer class; the name and the
prefill/step contract never change, so switching implementation is an
environment variable and never a call-site edit:

    from sca2 import SCA2Layer, LayerCfg, make_layer

    layer = make_layer(LayerCfg(d=128), device="cuda")   # honours SCA2_VERSION
    y, state = layer.prefill(x)                          # (B,T,d) -> (B,T,d)
    y_t, state = layer.step(x_t, state)                  # (B,d)   -> (B,d)

    SCA2_VERSION=v0   frozen reference (slow, obvious)
    SCA2_VERSION=v1   quad C head + chunked log-space D scan
    SCA2_VERSION=v2   loop-free two-level scan + merged projection gemm (default)
    SCA2_COMPILE=0    disable the torch.compile wrapper
    SCA2_CHUNK=16     override the D head chunk size

Parameter names and shapes are identical across versions, so a state_dict is
portable between them. `python -m sca2.iso` asserts they agree numerically.
"""
import torch

from .ref import (LayerCfg, freq_grid, CHeadRef, DHeadRef,      # noqa: F401
                  SCA2Layer as ReferenceLayer)
from .registry import VARIANTS, register, build                  # noqa: F401
from . import versions

__all__ = ["SCA2Layer", "LayerCfg", "make_layer", "ReferenceLayer", "active_version",
           "VARIANTS", "register", "build", "freq_grid", "versions"]


def active_version(version=None):
    return versions.active_name(version)


def make_layer(cfg=None, version=None, chunk=None, compile=None, seed=0,
               device="cpu", dtype=torch.float32):
    """Build the active (or requested) version's layer.

    `compile` defaults to on for cuda and off for cpu, and is overridden by
    SCA2_COMPILE either way.
    """
    cfg = cfg or LayerCfg()
    c_cls, d_cls, layer_cls = versions.heads(version, chunk)
    torch.manual_seed(seed)
    m = layer_cls(cfg, c_cls, d_cls).to(device=device, dtype=dtype)
    want = versions.compile_enabled(default=str(device).startswith("cuda")) \
        if compile is None else compile
    if want:
        from .compiled import wrap
        m = wrap(m)
    return m


# the drop-in name: whatever version the environment selected
SCA2Layer = versions.heads()[2]
