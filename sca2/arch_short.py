r"""SHORT C HEAD: the characteristic-function form of a short causal conv, in the
D-head slot.  Math and numpy walkthrough: chead_numpy.py.  Motivation: CATCHUP.md.

A C head is a windowed Fourier sum,  o_t = sum_{s in window(t)} kappa(t,s) e_s,

    kappa(t,s) = (1/L) sum_m w_m exp( i[ theta_m (K(h_s) - K(z_t))_m + (s - t) omega_m ] )

The long head (cdelta) takes window = everything so far, a rope grid so the comb
never aliases, and keeps the sum as an accumulator S.  THIS head takes the DFT grid
omega_m = 2 pi m / L, whose kernel is EXACTLY delta((s - t) mod L) -- an exact tap
at every lag, but periodic -- so the window must be the last L tokens and the
state is a ring buffer of the L-1 previous writes (codes + values).  Nothing
accumulates, so the write is additive (no delta rule) and there is nothing to
forget.  With theta = 0 and w chosen, the head IS a learned L-tap causal filter
over the values; theta != 0 makes the taps depend on content(z_t) - content(h_s).

Same conventions as the long head: write key from h (previous token), value from
z (current token), read with z -- the one-step induction association.  Positions
enter only through t mod L, so the phase argument stays small at any context
length (float32-safe where t * omega would not be).

Cost: O(T . L . (L + dv)) per sequence -- no T x T kernel, no sequential scan.
Measured against the polar D head it replaces: see CATCHUP.md.

Interface = D head: prefill(z, h, state) -> (u (B,T,2dv), state), step(z_t, h_t,
state) -> (u (B,2dv), state), init_state(B, device, dtype).  ShortLayer swaps it
into SCA2Layer's D slot; LayerCfg.Ls sets L.  Self-test: python -m sca2.arch_short
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_cdelta import CHeadDelta, CHeadDeltaKV, CHeadDeltaRaw
from .compiled import wrap as _cw
from .fast_dhead import DHeadSepQPolarFlat
from .ref import SCA2Layer, _gated_out
from .registry import register


class CHeadShort(nn.Module):
    def __init__(self, d, L, dv, theta_scale=0.02):
        super().__init__()
        self.d, self.L, self.dv = d, L, dv
        self.K = nn.Linear(d, L, False)
        self.V = nn.Linear(d, dv, False)
        self.theta = nn.Parameter(theta_scale * torch.randn(L))
        # w = 1: kernel = delta at lag 0 at init (o_t = V(z_t)); the taps are learned
        self.wr = nn.Parameter(torch.ones(L))
        self.wi = nn.Parameter(torch.zeros(L))
        # built in float64 then cast: 2*pi/L rounded in float32 breaks the exact
        # cancellation of the Dirichlet comb at the 1e-7 level (harmless in float32
        # training, visible in the float64 self-test below)
        self.register_buffer("omega", (torch.arange(L, dtype=torch.float64) * (2 * math.pi / L))
                             .to(torch.get_default_dtype()))                  # (L,)

    # ---- state: the L-1 previous writes ----------------------------------- #
    def init_state(self, B, device, dtype):
        n = self.L - 1
        return {"c": torch.ones(B, n, self.L, device=device, dtype=dtype),   # cos(phi) of past writes
                "s": torch.zeros(B, n, self.L, device=device, dtype=dtype),  # sin(phi)
                "e": torch.zeros(B, n, self.dv, device=device, dtype=dtype), # their values (0 = no write)
                "pos": torch.zeros((), device=device, dtype=torch.long)}

    def _phase(self, x, p):
        """x (..., d), p (...) integer positions -> phase (..., L); positions mod L."""
        return self.K(x) * self.theta + (p % self.L).to(x.dtype)[..., None] * self.omega

    def _read(self, cq, sq, cw, sw, e):
        """cq,sq (B,T,L) read codes; cw,sw (B,T,W,L) write codes of the W window
        slots of each t (unfold views); e (B,T,W,dv).  Returns (B,T,2dv) = [Re | Im].

        Re kappa = sum_l cw.(wr cq + wi sq) + sw.(wr sq - wi cq)   (expand cos(phi-psi))
        Im kappa = sum_l sw.(wr cq + wi sq) - cw.(wr sq - wi cq)
        so the read-side weights are folded into two (B,T,L) vectors and the window
        tensors enter only through contractions -- no (B,T,W,L) intermediate."""
        c1 = self.wr * cq + self.wi * sq                        # (B,T,L)
        c2 = self.wr * sq - self.wi * cq
        k_re = (torch.einsum("btwl,btl->btw", cw, c1) + torch.einsum("btwl,btl->btw", sw, c2)) / self.L
        k_im = (torch.einsum("btwl,btl->btw", sw, c1) - torch.einsum("btwl,btl->btw", cw, c2)) / self.L
        return torch.cat([torch.einsum("btw,btwj->btj", k_re, e),
                          torch.einsum("btw,btwj->btj", k_im, e)], -1)

    # ---- prefill ---------------------------------------------------------- #
    BANDED = True   # prefill read as banded GEMMs (same method as lapa/layer.py ShortHead._banded); False = unfolded einsum

    def _banded(self, cq, sq, cw, sw, e, T):
        """The window read as BANDED GEMMs, all chunks at once (no sequential dependency).

        kappa(t,s) = Fq_t . Fk_s / L  with  Fk_s = [cw_s ; sw_s]  (2L)  and, for the real /
        imaginary parts,  Fq1_t = [c1 ; c2],  Fq2_t = [-c2 ; c1],  c1 = wr cq + wi sq,
        c2 = wr sq - wi cq.  Queries are cut into chunks of C = L; the chunk with queries
        [t0, t0+C) reads extended keys [t0, t0+C+L-1) (the L-1 buffered writes come first
        in the extended arrays, so query t reads extended indices t .. t+L-1 = lags L-1 .. 0).
        S = Fq (B,K,2C,2L) @ Fk_ext (B,K,2L,C+L-1), band mask 0 <= j - i <= L-1, o = S @ e_ext.
        Same numbers as the unfolded contraction (checked against the repo path in __main__),
        dense GEMMs instead of an O(T.L.L) einsum on strided views."""
        B = cq.size(0)
        L = self.L
        C = L
        K = -(-T // C)
        pad = K * C - T
        if pad:                                                   # ragged tail: pad queries and keys
            cq, sq = F.pad(cq, (0, 0, 0, pad)), F.pad(sq, (0, 0, 0, pad))
            cw, sw, e = F.pad(cw, (0, 0, 0, pad)), F.pad(sw, (0, 0, 0, pad)), F.pad(e, (0, 0, 0, pad))
        c1 = self.wr * cq + self.wi * sq                          # (B,KC,L)
        c2 = self.wr * sq - self.wi * cq
        Fq = torch.cat([torch.cat([c1, c2], -1).view(B, K, C, 2 * L),
                        torch.cat([-c2, c1], -1).view(B, K, C, 2 * L)], 2)          # (B,K,2C,2L)
        Fk = torch.cat([cw, sw], -1)                                                # (B,KC+L-1,2L)
        N = C + L - 1
        Fk = Fk.unfold(1, N, C).movedim(-1, 2)                                      # (B,K,N,2L)
        ek = e.unfold(1, N, C).movedim(-1, 2)                                       # (B,K,N,dv)
        S = Fq @ Fk.transpose(-1, -2) / L                                           # (B,K,2C,N)
        i = torch.arange(C, device=cq.device)[:, None]
        j = torch.arange(N, device=cq.device)[None]
        band = ((j - i) >= 0) & ((j - i) <= L - 1)                                  # (C,N)
        S = S.masked_fill(~torch.cat([band, band], 0)[None, None], 0)
        o = S @ ek                                                                  # (B,K,2C,dv)
        o = torch.cat([o[:, :, :C], o[:, :, C:]], -1).reshape(B, K * C, 2 * self.dv)
        return o[:, :T]

    def prefill(self, z, h, state=None):
        B, T, _ = z.shape
        st = state if state is not None else self.init_state(B, z.device, z.dtype)
        L, n = self.L, self.L - 1
        p = torch.arange(T, device=z.device) + st["pos"]                     # (T,)
        phi = self._phase(h, p[None].expand(B, T))                           # write phases (B,T,L)
        cw_new, sw_new, e_new = phi.cos(), phi.sin(), self.V(z)              # (B,T,L) (B,T,L) (B,T,dv)
        # extended sequence: the L-1 buffered writes, then this chunk
        cw = torch.cat([st["c"], cw_new], 1)                                 # (B, n+T, L)
        sw = torch.cat([st["s"], sw_new], 1)
        e = torch.cat([st["e"], e_new], 1)                                   # (B, n+T, dv)
        psi = self._phase(z, p[None].expand(B, T))                           # read phases (B,T,L)
        if self.BANDED:
            u = self._banded(psi.cos(), psi.sin(), cw, sw, e, T)
        else:
            # window slots: query t reads extended indices t .. t+n  (lags L-1 .. 0)
            win = lambda x: x.unfold(1, L, 1).movedim(-1, 2)                 # (B, T, L, C)
            u = self._read(psi.cos(), psi.sin(), win(cw), win(sw), win(e))
        new = {"c": cw[:, -n:], "s": sw[:, -n:], "e": e[:, -n:], "pos": st["pos"] + T}
        return _gated_out(self, u, z), new

    # ---- decode ----------------------------------------------------------- #
    def step(self, z_t, h_t, state):
        B = z_t.size(0)
        p = state["pos"].expand(B)
        phi = self._phase(h_t, p)                                            # (B,L)
        cw = torch.cat([state["c"], phi.cos()[:, None]], 1)                  # (B,L,L)  window incl. this token
        sw = torch.cat([state["s"], phi.sin()[:, None]], 1)
        e = torch.cat([state["e"], self.V(z_t)[:, None]], 1)                 # (B,L,dv)
        psi = self._phase(z_t, p)                                            # (B,L)
        u = self._read(psi.cos()[:, None], psi.sin()[:, None],
                       cw[:, None], sw[:, None], e[:, None])[:, 0]           # (B,2dv)
        new = {"c": cw[:, 1:], "s": sw[:, 1:], "e": e[:, 1:], "pos": state["pos"] + 1}
        return _gated_out(self, u, z_t), new


class ShortLayer(SCA2Layer):
    """cdelta C head + short dft C head in the D slot."""
    C_CLS = CHeadDelta

    def __init__(self, cfg, c_cls=None, d_cls=None):
        super().__init__(cfg, self.C_CLS, DHeadSepQPolarFlat)                # built, then replaced
        dv = cfg.dv if cfg.dv is not None else cfg.d // 2
        self.dh = CHeadShort(cfg.d, cfg.Ls, dv, theta_scale=cfg.theta_scale or 0.02)
        # window-aligned damping (mirror of lapa/layer.py _damp_params): the damped half of the
        # long head starts just beyond the exact window and may never forget faster than it
        if hasattr(self.c, "lam_raw"):
            lam_max = cfg.lam_max if cfg.lam_max is not None else 1.0 / cfg.Ls
            lo, hi = cfg.damp_mem if cfg.damp_mem is not None else (float(cfg.Ls), 32.0 * cfg.Ls)
            self.c.LAM_MAX = lam_max                                          # instance attr shadows the class cap
            with torch.no_grad():
                mem = torch.exp(torch.empty(self.c.M).uniform_(math.log(lo), math.log(hi)))
                self.c.lam_raw.copy_(torch.log(torch.expm1(1.0 / mem)))
        # Causal depthwise conv on z, before BOTH heads (mirror of lapa/layer.py). Every
        # competitive linear mixer has one -- Mamba, GDN (kernel 4 on q/k/v), LFM2 -- and this
        # lineage never did; LayerCfg.conv existed but only arch_gatedc implemented it.
        # Identity at init, so cfg.conv > 0 changes nothing until it is trained.
        self.ck = cfg.conv
        if self.ck:
            w = torch.zeros(cfg.d, 1, self.ck)
            w[:, 0, -1] = 1.0
            self.cw = nn.Parameter(w)

    def init_state(self, B, device, dtype=torch.float32):
        st = super().init_state(B, device, dtype)
        if getattr(self, "ck", 0):
            st["cbuf"] = torch.zeros(B, self.ck - 1, self.cfg.d, device=device, dtype=dtype)
        return st

    def _conv(self, z, buf):
        zz = torch.cat([buf.to(z.dtype), z], 1)
        zc = F.conv1d(zz.transpose(1, 2), self.cw.to(z.dtype),
                      groups=self.cfg.d).transpose(1, 2)
        # BACK TO z's dtype. LayerNorm returns fp32 under autocast, but conv1d is an
        # autocast-eligible op and returns bf16; letting that through hands bf16 to the
        # long head, whose triangular solve has no bfloat16 CUDA kernel at all.
        return zc.to(z.dtype), zz[:, -(self.ck - 1):]

    def prefill(self, x, state=None):
        if not getattr(self, "ck", 0):
            return super().prefill(x, state)
        B = x.size(0)
        if state is None:
            state = self.init_state(B, x.device, x.dtype)
        z = self.n(x)
        z, cbuf = self._conv(z, state["cbuf"])
        h = torch.empty_like(z)
        h[:, 0] = state["z_prev"]
        h[:, 1:] = z[:, :-1]
        uc, cs = self.c.prefill(z, h, state["c"])
        ud, ds = self.dh.prefill(z, h, state["d"])
        x = x + self.mix(torch.cat([uc, ud], -1))
        x = x + self.ff(self.fn(x))
        return x, {"c": cs, "d": ds, "z_prev": z[:, -1], "cbuf": cbuf}

    def step(self, x_t, state):
        if not getattr(self, "ck", 0):
            return super().step(x_t, state)
        z = self.n(x_t)
        win = torch.cat([state["cbuf"].to(z.dtype), z[:, None]], 1)
        z = (win.transpose(1, 2) * self.cw.squeeze(1).to(z.dtype)).sum(-1).to(win.dtype)
        h = state["z_prev"]
        uc, cs = self.c.step(z, h, state["c"])
        ud, ds = self.dh.step(z, h, state["d"])
        y = x_t + self.mix(torch.cat([uc, ud], -1))
        y = y + self.ff(self.fn(y))
        return y, {"c": cs, "d": ds, "z_prev": z, "cbuf": win[:, 1:]}


class ShortLayerRaw(ShortLayer):
    """same, with the long head's read unnormalised (cdelta_raw)."""
    C_CLS = CHeadDeltaRaw


class ShortLayerKV(ShortLayer):
    """short dft head in the D slot + key-verified long head: the two fixes together."""
    C_CLS = CHeadDeltaKV


register("cshort_kv", CHeadDeltaKV, None, arch=True, layer_cls=ShortLayerKV,
         note="cdelta_kv long head + short dft head")
register("cshort_kv_cc", CHeadDeltaKV, None, arch=True, layer_cls=ShortLayerKV, wrap=_cw,
         note="cshort_kv + compile")
from .arch_damp import CHeadDeltaDampHalf, CHeadDeltaDampHalfFast, CHeadDeltaDampHalfKV   # noqa: E402  (after ShortLayer)


class ShortLayerDampH(ShortLayer):
    """short dft head in the D slot + half-persistent DAMPED long head, no verification."""
    C_CLS = CHeadDeltaDampHalf


class ShortLayerDampHKV(ShortLayer):
    """... plus key verification: does the gate still earn its keep once the noise is damped?"""
    C_CLS = CHeadDeltaDampHalfKV


class ShortLayerDampHF(ShortLayer):
    C_CLS = CHeadDeltaDampHalfFast


register("cshort_damphf", CHeadDeltaDampHalfFast, None, arch=True, layer_cls=ShortLayerDampHF,
         note="cshort_damph with the decay cap lifted (lambda <= 2)")
register("cshort_damphf_cc", CHeadDeltaDampHalfFast, None, arch=True, layer_cls=ShortLayerDampHF, wrap=_cw,
         note="cshort_damphf + compile")
register("cshort_damph", CHeadDeltaDampHalf, None, arch=True, layer_cls=ShortLayerDampH,
         note="damped (half-persistent) long head + short dft head")
register("cshort_damph_cc", CHeadDeltaDampHalf, None, arch=True, layer_cls=ShortLayerDampH, wrap=_cw,
         note="cshort_damph + compile")
register("cshort_damphkv", CHeadDeltaDampHalfKV, None, arch=True, layer_cls=ShortLayerDampHKV,
         note="damped (half-persistent) long head + key verification + short dft head")
register("cshort_damphkv_cc", CHeadDeltaDampHalfKV, None, arch=True, layer_cls=ShortLayerDampHKV, wrap=_cw,
         note="cshort_damphkv + compile")
register("cshort", CHeadDelta, None, arch=True, layer_cls=ShortLayer,
         note="cdelta + short dft C head (L-tap characteristic-function conv) in the D slot")
register("cshort_cc", CHeadDelta, None, arch=True, layer_cls=ShortLayer, wrap=_cw,
         note="cshort + compile")
register("cshort_raw", CHeadDeltaRaw, None, arch=True, layer_cls=ShortLayerRaw,
         note="cshort with the long head's read unnormalised")
register("cshort_raw_cc", CHeadDeltaRaw, None, arch=True, layer_cls=ShortLayerRaw, wrap=_cw,
         note="cshort_raw + compile")


if __name__ == "__main__":
    # Semantic checks in float64, against the closed form -- the iso gate cannot
    # catch a kernel that is wrong the same way on both paths.
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    d, L, dv, B, T = 8, 8, 4, 2, 40
    hd = CHeadShort(d, L, dv, theta_scale=0.3).double()
    hd.rms_read = False; hd.rscale = torch.ones(2 * dv, dtype=torch.float64)   # raw read for the check
    z = torch.randn(B, T, d, dtype=torch.float64); h = torch.roll(z, 1, 1); h[:, 0] = 0
    u, _ = hd.prefill(z, h)
    # closed form: o_t = sum_{s=t-L+1..t} kappa(t,s) e_s
    with torch.no_grad():
        p = torch.arange(T)
        phi = hd.K(h) * hd.theta + (p % L).double()[:, None] * hd.omega        # (B,T,L)
        psi = hd.K(z) * hd.theta + (p % L).double()[:, None] * hd.omega
        e = hd.V(z); w = hd.wr + 1j * hd.wi
        ref = torch.zeros(B, T, dv, dtype=torch.complex128)
        for t in range(T):
            for s in range(max(0, t - L + 1), t + 1):
                kap = (w * torch.exp(1j * (phi[:, s] - psi[:, t]))).sum(-1) / L  # (B,)
                ref[:, t] += kap[:, None] * e[:, s]
        err = (u - torch.cat([ref.real, ref.imag], -1)).abs().max().item()
    print(f"prefill vs closed form (window sum): {err:.2e}")
    assert err < 1e-12
    # theta = 0, w = 1  ->  o_t = V(z_t) exactly (Dirichlet delta at lag 0)
    with torch.no_grad():
        hd.theta.zero_(); u0, _ = hd.prefill(z, h)
        err0 = (u0[..., :dv] - hd.V(z)).abs().max().item(); print(f"theta=0, w=1: o_t - V(z_t) = {err0:.2e}")
        assert err0 < 1e-12 and u0[..., dv:].abs().max() < 1e-12
    # prefill == token-by-token, across a chunk boundary
    with torch.no_grad():
        hd.theta.normal_(); u, _ = hd.prefill(z, h)
        st = hd.init_state(B, z.device, z.dtype); outs = []
        for t in range(T):
            o, st = hd.step(z[:, t], h[:, t], st); outs.append(o)
        err1 = (u - torch.stack(outs, 1)).abs().max().item(); print(f"prefill vs step: {err1:.2e}")
        assert err1 < 1e-12
        u1, st1 = hd.prefill(z[:, :13], h[:, :13]); u2, _ = hd.prefill(z[:, 13:], h[:, 13:], st1)
        err2 = (u - torch.cat([u1, u2], 1)).abs().max().item(); print(f"prefill vs split 13|27: {err2:.2e}")
        assert err2 < 1e-12
    with torch.no_grad():
        hd.BANDED = False; u_unf, _ = hd.prefill(z, h); hd.BANDED = True; u_band, _ = hd.prefill(z, h)
        print(f"banded vs unfolded read: {(u_unf - u_band).abs().max().item():.2e}"); assert (u_unf - u_band).abs().max() < 1e-12
    n = sum(p.numel() for p in hd.parameters()); print(f"params (d={d}, L={L}, dv={dv}): {n}; ALL OK")
