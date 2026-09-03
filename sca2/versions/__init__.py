"""
Versioned SCA2 layer implementations.

Every version module exports the SAME three names -- `CHead`, `DHead`, `Layer`
-- plus a one-line `NOTE`, so switching version is a drop-in swap and never a
call-site change. Parameter names and shapes are identical across versions, so a
state_dict written by one loads into any other, and `python -m sca2.iso` checks
that they agree numerically in prefill, in token-by-token decode, under CUDA
graph capture, and in every gradient.

Selecting a version, in priority order:

    sca2.make_layer(cfg, version="v2")     explicit argument
    SCA2_VERSION=v1 python your_script.py  environment
    (nothing)                              DEFAULT below

    SCA2_COMPILE=0                         disable the torch.compile wrapper
    SCA2_CHUNK=16                          override the D head chunk size

  v0  frozen reference. The semantics contract; slow and obvious.
  v1  quad C head + chunked log-space D scan. 21x over v0.
  v2  merged projection gemm -- MEASURED SLOWER, kept for the record.
  v3  v1 + chunked C head, removing the O(T^2) prefill memory wall.
"""
import importlib
import os

MODULES = {"v0": "v0_ref", "v1": "v1_quad_scan", "v2": "v2_fused",
           "v3": "v3_longctx"}
DEFAULT = "v1"   # v2 measured slower as first written; see its module docstring
ORDER = ["v0", "v1", "v2", "v3"]


def _env(name, cast, default=None):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name}={v!r} is not a valid {cast.__name__}")


def active_name(version=None):
    name = version or _env("SCA2_VERSION", str) or DEFAULT
    if name not in MODULES:
        raise ValueError(f"unknown SCA2 version {name!r}; have {sorted(MODULES)}")
    return name


def module(version=None):
    return importlib.import_module(f".{MODULES[active_name(version)]}", __package__)


def heads(version=None, chunk=None):
    """(CHead, DHead, Layer) for a version, with the chunk-size override applied."""
    m = module(version)
    c, d, layer = m.CHead, m.DHead, m.Layer
    ch = chunk if chunk is not None else _env("SCA2_CHUNK", int)
    if ch is not None and hasattr(d, "CHUNK"):
        d = type(f"{d.__name__}C{ch}", (d,), {"CHUNK": ch})
    return c, d, layer


def status(version=None):
    """'reference' | 'recommended' | 'regression'. DEFAULT is always the
    recommended one -- a version that measured worse stays in the tree for the
    record but never becomes the default."""
    return getattr(module(version), "STATUS", "unknown")


def table():
    return [(n, status(n), module(n).NOTE) for n in ORDER]


def compile_enabled(default=True):
    v = _env("SCA2_COMPILE", str)
    return default if v is None else v.lower() not in ("0", "false", "no", "off")
