"""A/B: separable D-head query vs the full one, at equal and at reduced budget."""
import argparse, sys, torch
from .ref import LayerCfg
from .bench_params import SCA2LM, train_one
from .ab_freq import load_compact

# (label, variant, Md)
ARMS = [("full  Md=16", "v1", 16),
        ("sepq  Md=16", "sepq", 16),
        ("sepq  Md=64", "sepq", 64),
        ("sepq  Md=128", "sepq", 128)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--block", type=int, default=128)
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--G", type=int, default=8); p.add_argument("--ff", type=int, default=256)
    p.add_argument("--lrs", default="1e-3,3e-4")
    p.add_argument("--freq", default="dft")
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, V = load_compact()
    print(f"compact vocab {V}  device {device}  steps {a.steps}  freq {a.freq}")
    print(f"{'arm':<14s} {'params':>8s} {'val':>8s} {'vs full':>9s}")
    base = None
    for label, variant, Md in ARMS:
        cfg = LayerCfg(a.d, a.Mc, Md, a.G, a.ff, freq=a.freq, theta_scale=0.0,
                       max_len=a.block)
        best, npar = float("inf"), None
        for lr in [float(s) for s in a.lrs.split(",")]:
            m = SCA2LM(V, cfg, variant)
            npar = m.core_params()
            v, _ = train_one(m, tr, va, a, device, lr)
            best = min(best, v)
            del m
            if device == "cuda":
                torch.cuda.empty_cache()
        d = "" if base is None else f"{best - base:+.4f}"
        if base is None:
            base = best
        print(f"{label:<14s} {npar:>8d} {best:>8.4f} {d:>9s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
