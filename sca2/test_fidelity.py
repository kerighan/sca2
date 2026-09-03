"""Proves sca2/ref.py under LayerCfg.legacy() reproduces the original
bench_tinypython.py exactly (same weights).

ref.py now DEFAULTS to fixed semantics (see freq_grid / theta_scale), so this
test pins the legacy path -- it is the provenance proof, not a claim about the
current default."""
import importlib.util, os, sys, torch
from .ref import CHeadRef, DHeadRef, SCA2Layer, LayerCfg

def _orig():
    # byte-exact snapshot of bench_tinypython.py as first received, so this proof
    # survives any later edit to the benchmark itself
    p = os.path.join(os.path.dirname(__file__), "_original_bench.py")
    s = importlib.util.spec_from_file_location("bench_orig", os.path.abspath(p))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
    o = _orig()
    torch.set_default_dtype(torch.float64)
    d, Mc, Md, G, ff, V = 64, 16, 8, 4, 128, 97
    B, T = 3, 33
    # legacy(): the pre-fix semantics the original file implemented
    torch.manual_seed(0); ref = SCA2Layer(LayerCfg.legacy(d=d, Mc=Mc, Md=Md, G=G, ff=ff)).double()
    torch.manual_seed(0); orig = o.SCA2(V, d, Mc, Md, G, ff).double()

    # transplant: identical submodule names in both
    ref.load_state_dict({k: v for k, v in orig.state_dict().items()
                         if not k.startswith(("e.", "o.", "on."))}, strict=True)

    x = torch.randn(B, T, d, dtype=torch.float64)
    # replicate the original layer body around the transplanted weights
    z = orig.n(x); h = torch.zeros_like(z); h[:, 1:] = z[:, :-1]
    y_o = x + orig.mix(torch.cat([orig.c(z, h), orig.dh(z, h)], -1))
    y_o = y_o + orig.ff(orig.fn(y_o))
    y_r, _ = ref.prefill(x)
    e_pre = (y_r - y_o).abs().max().item()

    st = ref.init_state(B, x.device, x.dtype); ys = []
    for t in range(T):
        y, st = ref.step(x[:, t], st); ys.append(y)
    e_dec = (torch.stack(ys, 1) - y_o).abs().max().item()

    print(f"ref.prefill vs original : max|delta| = {e_pre:.3e}")
    print(f"ref.step    vs original : max|delta| = {e_dec:.3e}")
    ok = e_pre < 1e-12 and e_dec < 1e-12
    print("FIDELITY OK" if ok else "FIDELITY FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
