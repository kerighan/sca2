"""
Version v3 -- "long context".

v1, with the C head's O(T^2) memory wall removed.

v1's C head is exact causal attention with a structured complex kernel, so its
prefill materializes two B.T.T score matrices: 4.3 GB at T=8192, B=8. That wall,
not FLOPs, is what bounds the context this layer can prefill -- and it is the
only thing stopping the O(1) decode (bench_decode_scaling.py: flat to 131k
tokens) from being usable at length.

The fix needs no new maths. v1's C head already accepts an incoming state and
already returns the closing one -- that path is what `iso.py::split_iso` tests --
so chunking is just calling it once per chunk and letting the state carry:

    per chunk:  intra-chunk quadratic scores  +  rotation of the state entering
                the chunk  ->  state updated with the chunk's own contribution

Memory drops from B.T.T to B.C.C plus a fixed-size state. FLOPs go from
B.T^2.(2M+dv) to B.T.C.(2M+dv) + 2.B.T.M.dv -- at T=8192, C=256 that is 48x
fewer. At T <= C the loop runs once and this IS v1, bit for bit, so there is no
regression at the benchmark's T=128.

Chunk size via SCA2_CTX_CHUNK (default 256). The D head is v1's, reused.
"""
import os
import torch

from ..ref import SCA2Layer
from .v1_quad_scan import CHeadQuad, DHeadScan8


class CHead(CHeadQuad):
    CTX = int(os.environ.get("SCA2_CTX_CHUNK", 256))

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        C = self.CTX
        if T <= C:                      # one chunk: identical to v1
            return super().prefill(z, h, state)
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        outs = []
        for s0 in range(0, T, C):
            o, st = super().prefill(z[:, s0:s0 + C], h[:, s0:s0 + C], st)
            outs.append(o)
        return torch.cat(outs, 1), st


DHead = DHeadScan8
Layer = SCA2Layer
STATUS = "candidate"
NOTE = "v1 + chunked C head (removes the O(T^2) prefill memory wall)"
