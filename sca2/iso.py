"""
Iso harness: every SCA2 variant must be numerically indistinguishable from the
frozen reference, in BOTH modes.

Checks, per variant / shape / dtype:
  1. prefill_iso    variant.prefill(x)                       == ref.prefill(x)
  2. decode_iso     variant stepped token-by-token           == variant.prefill(x)
  3. cross_iso      variant stepped token-by-token           == ref.prefill(x)
  4. split_iso      prefill(x[:, :k]) then step the tail     == variant.prefill(x)
  5. grad_iso       d/dparam and d/dx                        == ref gradients

Run:  python -m sca2.iso            (all variants, fp32+fp64)
      python -m sca2.iso v1 --fast
Exit code is non-zero if any check fails, so it gates every new iteration.
"""
import argparse
import sys
import torch

from .ref import LayerCfg
from .registry import VARIANTS, build

# (dtype-specific) tolerances on the relative error metric below.
TOL = {torch.float64: 1e-10, torch.float32: 3e-4, torch.bfloat16: 6e-2}


def relerr(a, b, scale=None):
    """Max abs deviation, normalized by `scale` (default: max|b| of this tensor).

    Gradient checks pass an explicit model-wide scale: a per-tensor denominator
    reports 1e-3 "error" on a parameter whose gradient is 1e-7 while the layer's
    gradient scale is 1e0, which is noise, not a discrepancy.
    """
    a, b = a.detach().double(), b.detach().double()
    den = (b.abs().max() if scale is None else torch.as_tensor(float(scale))).clamp_min(1e-12)
    return ((a - b).abs().max() / den).item()


def _decode_all(layer, x, state=None):
    """Token-by-token decode of the whole sequence."""
    B, T, _ = x.shape
    st = state if state is not None else layer.init_state(B, x.device, x.dtype)
    ys = []
    for t in range(T):
        y, st = layer.step(x[:, t], st)
        ys.append(y)
    return torch.stack(ys, 1), st


def _decode_all_graph(layer, x):
    """Token-by-token decode through a captured CUDA graph."""
    from .decode import GraphDecoder
    B, T, _ = x.shape
    st = layer.init_state(B, x.device, x.dtype)
    dec = GraphDecoder(layer, st)
    return torch.stack([dec.step(x[:, t]).clone() for t in range(T)], 1), st


def _sync_weights(dst, src):
    (getattr(dst, "layer", dst)).load_state_dict(src.state_dict())


def _flat_grads(layer, y):
    """Gradients keyed by name, with any wrapper prefix stripped so a wrapped
    variant's dict lines up with the reference's."""
    layer.zero_grad(set_to_none=True)
    y.square().mean().backward()
    inner = getattr(layer, "layer", layer)
    return {n: (p.grad.clone() if p.grad is not None else None)
            for n, p in inner.named_parameters()}


def check(name, cfg, B, T, dtype, device, seed=0, want_grad=True, self_mode=False,
          against=None):
    """Returns list of (check_name, relerr, tol, passed).

    self_mode compares a variant against ITSELF instead of against the
    reference. Needed for architecture candidates (different parameters, so no
    reference to compare to): prefill_iso and grad_iso become vacuous and are
    dropped, while decode_iso / split_iso / graph_iso still check that the
    recurrent form matches the parallel one -- the part that actually breaks.
    """
    tol = TOL[dtype]
    ref = build(name if self_mode else (against or "ref"), cfg, seed=seed,
                device=device, dtype=dtype)
    var = build(name, cfg, seed=seed, device=device, dtype=dtype)
    _sync_weights(var, ref)

    torch.manual_seed(seed + 1)
    x = torch.randn(B, T, cfg.d, device=device, dtype=dtype)
    out = []

    with torch.no_grad():
        y_ref, _ = ref.prefill(x)
        y_var, _ = var.prefill(x)
        if not self_mode:
            out.append(("prefill_iso", relerr(y_var, y_ref)))

        y_dec, _ = _decode_all(var, x)
        out.append(("decode_iso", relerr(y_dec, y_var)))
        out.append(("cross_iso", relerr(y_dec, y_ref)))

        if T > 1:
            k = max(1, T // 2)
            y_head, st = var.prefill(x[:, :k])
            y_tail, _ = _decode_all(var, x[:, k:], st)
            out.append(("split_iso", relerr(torch.cat([y_head, y_tail], 1), y_ref)))

        if x.is_cuda:
            from .decode import GraphDecoder, _check_capturable
            try:
                _check_capturable(var.init_state(B, x.device, dtype))
                y_g, _ = _decode_all_graph(var, x)
                out.append(("graph_iso", relerr(y_g, y_ref)))
            except TypeError as e:
                out.append(("graph_iso[unsupported]", float("nan")))

    if want_grad and not self_mode:
        xr = x.clone().requires_grad_(True)
        xv = x.clone().requires_grad_(True)
        gr = _flat_grads(ref, ref.prefill(xr)[0])
        gv = _flat_grads(var, var.prefill(xv)[0])
        # one shared denominator: the largest gradient anywhere in the layer
        gscale = max([g.abs().max().item() for g in gr.values() if g is not None]
                     + [xr.grad.abs().max().item()])
        worst, where = 0.0, ""
        for n in gr:
            if gr[n] is None and gv[n] is None:
                continue
            if (gr[n] is None) != (gv[n] is None):
                worst, where = float("inf"), n
                break
            e = relerr(gv[n], gr[n], gscale)
            if e > worst:
                worst, where = e, n
        e = relerr(xv.grad, xr.grad, gscale)
        if e > worst:
            worst, where = e, "input"
        out.append((f"grad_iso[{where}]", worst))

    # Only a check explicitly labelled "[unsupported]" may be skipped. A NaN
    # anywhere else is a failure, not an absent measurement -- swallowing it hid
    # a real overflow-to-NaN in v2 at ragged T.
    return [(cn, e, tol, (e <= tol) if e == e else "unsupported" in cn)
            for cn, e in out]


SHAPES = [(1, 1), (2, 3), (3, 7), (2, 64), (2, 128), (2, 129)]
FAST_SHAPES = [(1, 1), (2, 5), (2, 128)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fast", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="print every check, not just the worst")
    p.add_argument("--against", default=None,
                   help="compare against this variant instead of the reference; use "
                        "when two implementations share parameters and semantics")
    p.add_argument("--self", dest="self_mode", action="store_true",
                   help="architecture candidates: check against the variant itself")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--Mc", type=int, default=16)
    p.add_argument("--Md", type=int, default=8)
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--ff", type=int, default=128)
    p.add_argument("--dtypes", default="float64,float32")
    p.add_argument("--freq", default="dft", help="C head positional grid: dft|len|rope")
    p.add_argument("--theta-scale", type=float, default=0.0)
    a = p.parse_args(argv)

    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff, freq=a.freq,
                   theta_scale=a.theta_scale, max_len=128)
    names = a.variants or [n for n in VARIANTS if n != "ref"] or ["ref"]
    dtypes = [getattr(torch, s) for s in a.dtypes.split(",")]
    shapes = FAST_SHAPES if a.fast else SHAPES

    fails = 0
    for name in names:
        if VARIANTS[name].get("arch") and not (a.self_mode or a.against):
            print(f"\n=== {name}  ({VARIANTS[name]['note']})")
            print("  REFUSED: this is an architecture candidate -- a different "
                  "function with a\n           different parameter set, so there is no "
                  "reference to be iso with.\n           Re-run with --self to check its "
                  "prefill/decode consistency, and use\n           an A/B "
                  "(sca2/ab_sepq.py) to judge whether the change is worth it.")
            fails += 1
            continue
        print(f"\n=== {name}  ({VARIANTS[name]['note']})")
        for dtype in dtypes:
            for B, T in shapes:
                rows = check(name, cfg, B, T, dtype, a.device, self_mode=a.self_mode,
                             against=a.against)
                bad = [r for r in rows if not r[3]]
                tag = "FAIL" if bad else "ok  "
                worst = max(rows, key=lambda r: r[1] / r[2])
                print(f"  {tag} {str(dtype).split('.')[-1]:>9s} B={B:<2d} T={T:<4d} "
                      f"worst={worst[0]:<22s} rel={worst[1]:.3e} (tol {worst[2]:.0e})")
                if a.verbose:
                    for cn, e, tol, okc in rows:
                        mark = ("skip" if (e != e and okc) else
                                ("ok" if okc else "FAIL"))
                        print(f"        {mark:<4s} {cn:<24s} rel={e:.3e}")
                for cn, e, tol, _ in bad:
                    print(f"       -> {cn}: rel={e:.3e} > tol={tol:.0e}")
                fails += len(bad)
    print(f"\n{'ISO FAILED' if fails else 'ISO OK'}  ({fails} failing checks)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
