"""
Loop-free D head: the chunked scan's Python loop removed, semantics unchanged.

Why. In the real training loop SCA2 runs at 79k tok/s while GDN runs at 97k,
even though a microbenchmark with pre-staged GPU tensors puts them at parity
(97.7k vs 98.8k). The gap is launch overhead, not arithmetic: arch_sepq's
prefill is `for s0 in range(0, T, CHUNK)`, so at T=256, CHUNK=8 it issues
32 iterations x ~15 kernels x 2 layers ~= 1000 launches per forward. As soon as
the CPU is also loading data it stops keeping the GPU fed -- which is exactly
the difference between the microbenchmark and pretrain.py. GDN does not have
this problem because fla's reference folds the chunks into a batch dimension
(`rearrange(x, 'b h (n c) d -> b h n c d')`) and only loops over the carry.

What. Two observations remove the loop entirely.

  1. Everything inside a chunk is independent of the carry, so fold T into
     (n, c) and compute all n chunks in one set of kernels.

  2. The carry itself has the SAME structure one level up. Writing S_k for the
     state at the end of chunk k, the transition is diagonal,

         S_k = D_k . S_{k-1} + L_k,     D_k = exp(sum of log a over chunk k)

     so with E_k = cumsum_j<=k (chunk-k total log-magnitude) -- i.e. the global
     cumulative log-magnitude sampled at chunk ends --

         S_k = sum_{j<=k} exp(E_k - E_j + i(P_k - P_j)) L_j
               + exp(E_k + i P_k) S_init

     which is a lower-triangular n x n matmul: the same closed form the code
     already uses INSIDE a chunk, applied to the chunk index. It is numerically
     safe for the same reason: log|a| <= 0, so E_k - E_j <= 0 for k >= j and
     every exponential is <= 1, no clamp needed beyond the existing one.

Cost of the extra level: B.M.G.n.n.gs = 17 MMAC at T=256, n=32 -- against the
~1000 launches it removes. Launches drop to O(30) for the whole prefill.

This is an IMPLEMENTATION change: same parameters, same function. So unlike
every architecture candidate in this package it is checkable exactly --

    python -m sca2.iso polarflat --against polar
    python -m sca2.iso v3polarflat --against v3polar

`delta_rule="chunk"` is NOT supported here: its correction reads the state at
the chunk start, which reintroduces the very dependency this removes. That path
falls back to the parent's loop.
"""
import torch

from .arch_sepq import DHeadSepQ, DHeadSepQPolar
from .ref import _gated_out
from .registry import register
from .compiled import wrap as _cw
from .versions.v1_quad_scan import CHeadQuad
from .versions.v3_longctx import CHead as CHeadChunked

_TRIL = {}


def _tril(n, device, dtype):
    k = (n, str(device), str(dtype))
    m = _TRIL.get(k)
    if m is None:
        m = torch.tril(torch.ones(n, n, device=device, dtype=dtype))
        _TRIL[k] = m
    return m


class _FlatMixin:
    """Replaces DHeadSepQ.prefill with the loop-free two-level form."""

    def prefill(self, z, h, state=None):
        if self.delta_rule:                       # exact or chunk: parent's path
            return super().prefill(z, h, state)
        B, T, _ = z.shape
        M, G, gs, dv = self.M, self.G, self.gs, self.dv
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        C = min(self.CHUNK, T)
        if T % C:                                 # ragged tail: parent's path
            return super().prefill(z, h, state)
        n = T // C

        v = self.V(z)
        vc = v.view(B, T, G, gs).permute(0, 2, 1, 3).reshape(B, G, n, C, gs)
        lap, php = self._log_polar(h, B, T)               # (B,M,G,T)
        AR, AI, BR, BI = self._alpha_beta(z)

        la = lap.view(B, M, G, n, C)
        ph = php.view(B, M, G, n, C)
        cla = la.cumsum(-1)                               # within chunk
        cph = ph.cumsum(-1)

        # ---- intra-chunk, all n chunks at once ---------------------------- #
        keep = _tril(C, z.device, z.dtype)
        dl = (cla.unsqueeze(-1) - cla.unsqueeze(-2)).clamp(max=0)
        dp = cph.unsqueeze(-1) - cph.unsqueeze(-2)
        mag = dl.exp() * keep
        ire = torch.einsum("bmgnts,bgnsj->bmgntj", mag * dp.cos(), vc)
        iim = torch.einsum("bmgnts,bgnsj->bmgntj", mag * dp.sin(), vc)

        # ---- inter-chunk carry, same closed form on the chunk index ------- #
        E = cla[..., -1].cumsum(-1)                       # (B,M,G,n) global at ends
        P = cph[..., -1].cumsum(-1)
        Lr, Li = ire[..., -1, :], iim[..., -1, :]         # (B,M,G,n,gs)
        keepn = _tril(n, z.device, z.dtype)
        dE = (E.unsqueeze(-1) - E.unsqueeze(-2)).clamp(max=0)
        dP = P.unsqueeze(-1) - P.unsqueeze(-2)
        wm = dE.exp() * keepn
        Wr, Wi = wm * dP.cos(), wm * dP.sin()
        Sr = (torch.einsum("bmgkj,bmgjy->bmgky", Wr, Lr)
              - torch.einsum("bmgkj,bmgjy->bmgky", Wi, Li))
        Si = (torch.einsum("bmgkj,bmgjy->bmgky", Wr, Li)
              + torch.einsum("bmgkj,bmgjy->bmgky", Wi, Lr))
        # incoming state, carried to every chunk end. Applied unconditionally:
        # at init it is exactly zero, and branching on an `empty` flag would make
        # this class's state dict differ from the parent's, which `step` shares.
        zr = st["sr"].view(B, M, G, 1, gs)
        zi = st["si"].view(B, M, G, 1, gs)
        eE = E.exp()
        Gr = (eE * P.cos()).unsqueeze(-1)                 # (B,M,G,n,1)
        Gi = (eE * P.sin()).unsqueeze(-1)
        Sr = Sr + Gr * zr - Gi * zi
        Si = Si + Gr * zi + Gi * zr

        # state ENTERING each chunk: S_{k-1}, with S_init in front
        Inr = torch.cat([zr, Sr[..., :-1, :]], -2).unsqueeze(-2)   # (B,M,G,n,1,gs)
        Ini = torch.cat([zi, Si[..., :-1, :]], -2).unsqueeze(-2)

        am = cla.exp()
        Are = (am * cph.cos()).unsqueeze(-1)              # (B,M,G,n,C,1)
        Aim = (am * cph.sin()).unsqueeze(-1)
        s_re = ire + Are * Inr - Aim * Ini
        s_im = iim + Are * Ini + Aim * Inr

        s_re = s_re.permute(0, 3, 4, 1, 2, 5).reshape(B, T, M, dv)
        s_im = s_im.permute(0, 3, 4, 1, 2, 5).reshape(B, T, M, dv)
        u = self._read(s_re, s_im, AR, AI, BR, BI, 2)
        return _gated_out(self, u, z), {"sr": Sr[..., -1, :].reshape(B, M, dv),
                                        "si": Si[..., -1, :].reshape(B, M, dv)}


class DHeadSepQFlat(_FlatMixin, DHeadSepQ):
    pass


class DHeadSepQPolarFlat(_FlatMixin, DHeadSepQPolar):
    pass


def _cg(layer, **kw):
    """compile + cudagraphs on the prefill (see CompiledLayer.__init__)."""
    return _cw(layer, prefill_mode="reduce-overhead", **kw)


register("sepqflat", CHeadQuad, DHeadSepQFlat, arch=True,
         note="loop-free sepq D head (iso with sepq)")
register("polarflat", CHeadQuad, DHeadSepQPolarFlat, arch=True,
         note="loop-free polar D head (iso with polar)")
register("polarflat_cc", CHeadQuad, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="loop-free polar D head + compile")
register("v3polarflat", CHeadChunked, DHeadSepQPolarFlat, arch=True,
         note="chunked C head + loop-free polar D head (iso with v3polar)")
register("v3polarflat_cc", CHeadChunked, DHeadSepQPolarFlat, arch=True, wrap=_cw,
         note="chunked C head + loop-free polar D head + compile")
# cudagraph-on-prefill pairs, to separate the two effects: the loop removal
# (flat) from the launch-overhead removal (cg). Same function in all four.
register("v3polar_cg", CHeadChunked, DHeadSepQPolar, arch=True, wrap=_cg,
         note="v3polar + compile with cudagraphs on prefill")
register("v3polarflat_cg", CHeadChunked, DHeadSepQPolarFlat, arch=True, wrap=_cg,
         note="loop-free polar D head + cudagraphs on prefill")
