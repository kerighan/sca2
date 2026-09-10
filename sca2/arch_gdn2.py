"""Gated DeltaNet-2 baseline for the LM pipeline (pretrain.py), wrapping the sanctuary's
lapa.benchmarks.baselines.GDN2Layer into the registry protocol. NVIDIA, arXiv 2605.22791:
channel-wise erase (b, key axis) and write (w, value axis) gates on top of KDA's
channel-wise decay; fla's naive reference (chunk for prefill, recurrent for decode).
Shape from LayerCfg.gdn_heads / gdn_head_k / gdn_expand_v; FFN from cfg.ff."""
import torch.nn as nn

from lapa.benchmarks.baselines import GDN2Layer
from .compiled import wrap as _cw
from .registry import register
from .versions.v1_quad_scan import CHeadQuad


class GDN2LayerMatched(nn.Module):
    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__()
        self.cfg = cfg
        self.inner = GDN2Layer(cfg.d, heads=cfg.gdn_heads, head_k=cfg.gdn_head_k, expand_v=cfg.gdn_expand_v, ff=cfg.ff)

    def init_state(self, B, device, dtype=None):
        return self.inner.init_state(B, device)

    def forward(self, x):
        return self.inner.prefill(x)[0]

    def prefill(self, x, state=None):
        return self.inner.prefill(x, state)

    def step(self, x_t, state):
        return self.inner.step(x_t, state)


register("gdn2", CHeadQuad, None, arch=True, layer_cls=GDN2LayerMatched,
         note="Gated DeltaNet-2 on fla's reference (ARCH: different function)")
register("gdn2_cc", CHeadQuad, None, arch=True, layer_cls=GDN2LayerMatched, wrap=_cw,
         note="gdn2 + compile")
