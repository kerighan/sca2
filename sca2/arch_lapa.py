"""Laplace Attention for the LM pipeline (pretrain.py), wrapping the sanctuary's
`lapa.LaplaceAttention` into the registry protocol.

Why this exists. `cshort_damph` (arch_short.py + arch_damp.py) and `lapa/layer.py`
compute the SAME function -- `python -m lapa.layer` checks it in float64 every run --
but they are not the same code, and as of the Spark branch only `lapa/layer.py` has
been optimised. The two differ by 1.9x at d=1024 on GB10, so training through the
sca2 mirror would throw the whole speed-up away. `--variant lapa_cc` trains the fast
one; `--variant cshort_damph_cc` still trains the mirror, and the two remain a cross-
check of each other (same loss curve, different implementation).

Shape comes from LayerCfg exactly as the mirror reads it: Mc -> M, Ls -> L, dv, ff,
theta_scale, rope_base, slow_frac, damp_mem -> mem_range, lam_max, max_len. The
prefill chunk follows $SCA2_CTX_CHUNK and the path $SCA2_LONG_PATH, as elsewhere in
this package (see sca2.autotune; on GB10 the chunk axis is flat, see SPARK.md §9).
"""
import os

import torch.nn as nn

from lapa import LaplaceAttention, LaplaceConfig

from .compiled import wrap as _cw
from .registry import register
from .versions.v1_quad_scan import CHeadQuad


def config_from(cfg) -> LaplaceConfig:
    """LayerCfg -> LaplaceConfig. The mapping the float64 gate in lapa/layer.py uses."""
    return LaplaceConfig(
        d=cfg.d,
        M=cfg.Mc,
        dv=cfg.dv if cfg.dv is not None else cfg.d // 2,
        L=cfg.Ls,
        ff=cfg.ff,
        theta_scale=cfg.theta_scale,
        rope_base=cfg.rope_base,
        rope_min_period=cfg.rope_min_period,
        persist=cfg.persist,
        learn_persist=cfg.learn_persist,
        conv=cfg.conv,
        conv_silu=cfg.conv_silu,
        beta_init=cfg.beta_init,
        decay_input=cfg.decay_input,
        kv_dk=cfg.kv_dk,
        kv_gate_pc=cfg.kv_gate_pc,
        beta_groups=cfg.beta_groups,
        short_groups=cfg.short_groups,
        long_groups=cfg.long_groups,
        slow_frac=cfg.slow_frac,
        max_len=cfg.max_len,
        mem_range=cfg.damp_mem,
        lam_max=cfg.lam_max,
        chunk=int(os.environ.get("SCA2_CTX_CHUNK", 128)),
        long_path=os.environ.get("SCA2_LONG_PATH", "batched"),
    )


class LapALayer(nn.Module):
    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__()
        self.cfg = cfg
        self.inner = LaplaceAttention(config_from(cfg))

    def init_state(self, B, device, dtype=None):
        return self.inner.init_state(B, device)

    def forward(self, x):
        return self.inner.prefill(x)[0]

    def prefill(self, x, state=None):
        return self.inner.prefill(x, state)

    def step(self, x_t, state):
        return self.inner.step(x_t, state)

    def state_floats(self):
        return self.inner.state_floats()


register("lapa", CHeadQuad, None, arch=True, layer_cls=LapALayer,
         note="Laplace Attention v1 via lapa/layer.py (== cshort_damph, optimised)")
register("lapa_cc", CHeadQuad, None, arch=True, layer_cls=LapALayer, wrap=_cw,
         note="lapa + compile")
