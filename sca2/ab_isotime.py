"""
The comparison that matters when the complaint is "too slow": EQUAL WALL CLOCK.

bench_params answers "who wins per parameter" (SCA2, by 0.16-0.31 nats at every
matched budget). That is the wrong axis if the architecture is compute hungry:
the Transformer gets more steps for the same seconds, and more steps may buy more
than a better layer does.

So instead of a fixed step count, every arm gets a fixed number of SECONDS of
training loop -- eval excluded from the clock -- and reports how far it got.
Both arms are torch.compile'd so neither is handicapped by python overhead.

Run:  python -m sca2.ab_isotime --seconds 120
"""
import argparse, sys, time
import torch
import torch.nn.functional as F

from .ref import LayerCfg
from .bench_params import SCA2LM, TrfLM
from .ab_freq import load_compact, get_batch, evaluate


def train_seconds(m, tr, va, a, device, lr, seconds, warm=8):
    """Fixed budget of *training loop* seconds.

    The warmup runs BEFORE the clock starts. Without it torch.compile's first
    call is charged to the budget, and since compilation takes tens of seconds
    the step counts end up measuring who compiles fastest rather than who
    computes fastest -- which is exactly how the first version of this
    measurement produced an unusable table.
    """
    m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(123)
    for _ in range(warm):
        x, y = get_batch(tr, a.batch, a.block, g, device)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    # the warmup steps are real training, given equally to every arm; only the
    # data stream is rewound so each arm sees the same batches from step 0
    g = torch.Generator().manual_seed(123)
    step, spent = 0, 0.0
    while spent < seconds:
        x, y = get_batch(tr, a.batch, a.block, g, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        spent += time.perf_counter() - t0
        step += 1
    return step, evaluate(m, va, a.batch, a.block, device)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--block", type=int, default=128)
    p.add_argument("--d", type=int, default=128); p.add_argument("--ff", type=int, default=256)
    p.add_argument("--G", type=int, default=8); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lrs", default="1e-3,3e-4")
    p.add_argument("--freq", default="rope")
    p.add_argument("--g-sweep", default=None, dest="g_sweep",
                   help="comma list of G values to sweep (uses sepq Mc64 Md16)")
    p.add_argument("--warm-steps", type=int, default=8, dest="warm_steps",
                   help="untimed steps before the clock starts (absorbs compilation)")
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, V = load_compact()
    lrs = [float(s) for s in a.lrs.split(",")]

    # (label, builder) -- SCA2 arms use rope, the grid measured best over 3 seeds
    def sca(variant, Mc, Md, freq=None, G=None):
        freq = freq or a.freq
        G = G if G is not None else a.G
        return lambda: SCA2LM(V, LayerCfg(a.d, Mc, Md, G, a.ff, freq=freq,
                                          theta_scale=0.0, max_len=a.block), variant)
    if a.g_sweep:
        # G is the number of gate groups over dv. Decay-matrix traffic is
        # B.Md.G.C.T, i.e. linear in G, so this is the last untested speed knob;
        # G=1 shares one gate per m across all dv channels, which is also the
        # least expressive, hence the equal-time framing.
        ARMS = [("transformer ff=256", lambda: TrfLM(V, a.d, a.heads, 256, a.block))]
        for gg in [int(x) for x in a.g_sweep.split(",")]:
            ARMS.append((f"sca2 sepq G={gg}", sca("sepq_cc", 64, 16, G=gg)))
    else:
        ARMS = [
            ("transformer ff=256", lambda: TrfLM(V, a.d, a.heads, 256, a.block)),
            ("transformer ff=1372", lambda: TrfLM(V, a.d, a.heads, 1372, a.block)),
            ("sca2 v1  Mc64 Md16", sca("v1_cc", 64, 16)),
            ("sca2 sepq Mc64 Md16", sca("sepq_cc", 64, 16)),
            ("sca2 sepq Mc16 Md16", sca("sepq_cc", 16, 16)),
        ]

    print(f"compact vocab {V}  device {device}  budget {a.seconds:.0f}s of training "
          f"loop per arm (eval excluded)  lrs {lrs}")
    print(f"{'arm':<22s} {'params':>8s} {'steps':>7s} {'val':>8s} {'vs trf':>8s}")
    base = None
    for label, mk in ARMS:
        best, npar, bstep = float("inf"), None, 0
        for lr in lrs:
            m = mk()
            npar = m.core_params()
            if device == "cuda":
                m = torch.compile(m) if "transformer" in label else m
            st, v = train_seconds(m, tr, va, a, device, lr, a.seconds,
                                  warm=a.warm_steps)
            if v < best:
                best, bstep = v, st
            del m
            if device == "cuda":
                torch.cuda.empty_cache()
        d = "" if base is None else f"{best - base:+.4f}"
        if base is None:
            base = best
        print(f"{label:<22s} {npar:>8d} {bstep:>7d} {best:>8.4f} {d:>8s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
