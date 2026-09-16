"""Mamba2 baseline via fla.layers.mamba2.

Wrapper is identical to GDNLayer: norm -> mixer -> residual -> norm -> FFN.
The mixer is fla's Mamba2 (SSD kernel), which falls back to Triton when
causal_conv1d is not installed (our case on the GB10).

    --variant mamba2_cc --mamba-expand 1

expand=1 gives 11.8M/layer (between LapA's 10.8M and GDN's 13.7M).
expand=2 gives 15.0M/layer (above GDN).
"""
import os
import torch
import torch.nn as nn

from .registry import register
from .compiled import wrap as _cw

# Silence the warnings on import
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fla.layers.mamba2 import Mamba2


class Mamba2Layer(nn.Module):
    """Same wrapper as GDNLayer / SCA2Layer: norm -> mixer -> residual -> norm -> FFN."""

    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__()
        d = cfg.d
        self.cfg = cfg
        expand = getattr(cfg, "mamba_expand", 1)
        self.n = nn.LayerNorm(d)
        self.mix = Mamba2(hidden_size=d, head_dim=64, state_size=128,
                          expand=expand, backend="cuda" if torch.cuda.is_available() else "naive")
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff), nn.GELU(), nn.Linear(cfg.ff, d))

    def init_state(self, B, device, dtype=torch.float32):
        return {}  # Mamba2 handles its own state internally

    def forward(self, x):
        return self.prefill(x)[0]

    def prefill(self, x, state=None):
        y = self.mix(self.n(x))
        if isinstance(y, tuple):
            y = y[0]
        x = x + y
        return x + self.ff(self.fn(x)), {}

    def step(self, x_t, state):
        # fla's Mamba2 doesn't expose a clean step interface;
        # for decode we'd need to handle the conv cache etc.
        # For training (prefill only) this is unused.
        raise NotImplementedError("Mamba2 step-by-step decode not wrapped")


# A dummy c_cls that the registry needs but Mamba2Layer ignores
class _Dummy:
    pass


register("mamba2", _Dummy, None, arch=True, layer_cls=Mamba2Layer,
         note="Mamba2 (SSD) via fla, expand=cfg.mamba_expand (ARCH)")
register("mamba2_cc", _Dummy, None, arch=True, layer_cls=Mamba2Layer, wrap=_cw,
         note="Mamba2 (SSD) + torch.compile")
