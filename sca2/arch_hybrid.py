"""HYBRID: the cdelta C head (hash keys, exact long-range retrieval) beside a
Gated DeltaNet head (metric keys, similarity) in the D-head slot.

Why (CATCHUP.md, results). By token class, generation 3's whole lead over GDN was
exact retrieval of repeats (word_rep: -1.19 nats at 40M tokens, which GDN then
learned away to -0.07), while on words NEW to the window it trailed by +0.2..+0.3
from the first eval to the last. Bounding the C head's phase (cdelta_bp) killed
retrieval and did not move word_new at all: the torus code is a hash, the hash is
the mechanism, and what is missing is a SECOND memory that generalises, not a
softer first one. The polar D head (Md=4, 448 floats of state) is where such a
memory would have to live, and it has never been re-tuned since Mc fell 378->190.

So: keep the C head byte-identical, replace the D head by a GatedDeltaNet head
(fla's reference, the same code as the gdn arm) that emits the D head's 2*dv
features, and fund it from the FFN at matched parameters. Everything else in the
layer -- norm, mix, residuals, FFN shape -- is SCA2Layer's.

Shape knobs come from LayerCfg.gdn_heads / gdn_head_k / gdn_expand_v (pretrain.py
--gdn-heads / --gdn-head-k / --gdn-expand-v). Matched shapes against the champion's
185959 params/layer are listed in CATCHUP.md.

Iso: the GDN head's prefill (chunked) and step (recurrent) are fla's two
references, which compute in float32 internally, so float64 checks are capped at
~1e-7 exactly as for the gdn arm; run `python -m sca2.iso hyb --self --dtypes
float32`.
"""
import torch.nn as nn

from .arch_cdelta import CHeadDelta
from .arch_gdn import GatedDeltaNet
from .compiled import wrap as _cw
from .fast_dhead import DHeadSepQPolarFlat
from .ref import SCA2Layer
from .registry import register


class GDNHead(GatedDeltaNet):
    """GatedDeltaNet whose output projection emits `out_dim` features instead of d."""

    def __init__(self, d, heads, head_k, expand_v, out_dim, conv_k=4):
        super().__init__(d, heads, head_k, expand_v, conv_k)
        self.o = nn.Linear(self.value_dim, out_dim, False)


class DHeadGDN(nn.Module):
    """D-head interface (prefill(z, h, state) / step(z_t, h_t, state) / init_state)
    around a GDN head. `h` (the previous token's z) is unused: GDN addresses by
    content of the current token, which is the point."""

    def __init__(self, d, dv, heads, head_k, expand_v):
        super().__init__()
        self.dv = dv
        self.m = GDNHead(d, heads, head_k, expand_v, out_dim=2 * dv)

    def init_state(self, B, device, dtype):
        return self.m.init_state(B, device, dtype)

    def prefill(self, z, h, state):
        return self.m.prefill(z, state)

    def step(self, z_t, h_t, state):
        return self.m.step(z_t, state)


class HybridLayer(SCA2Layer):
    def __init__(self, cfg, c_cls=CHeadDelta, d_cls=None):
        super().__init__(cfg, c_cls, DHeadSepQPolarFlat)     # built, then replaced
        dv = cfg.dv if cfg.dv is not None else cfg.d // 2
        self.dh = DHeadGDN(cfg.d, dv, cfg.gdn_heads, cfg.gdn_head_k, cfg.gdn_expand_v)


register("hyb", CHeadDelta, None, arch=True, layer_cls=HybridLayer,
         note="cdelta C head + GDN head in the D slot (ARCH: different function)")
register("hyb_cc", CHeadDelta, None, arch=True, layer_cls=HybridLayer, wrap=_cw,
         note="hyb + compile")
