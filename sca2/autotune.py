"""Blocked timing of the long head's implementation choices on THIS GPU.

    python -m sca2.autotune [--rounds 5] [-B 8] [-T 1024] [--variant cshort_damph]

Times every (path, chunk size) cell back to back inside each round and keeps
within-round ratios, so the GPU's clock state is a shared nuisance (WINNERS.md:
sequential timing on this machine varies 15% run to run; blocked ratios reproduce
to 0.4%). Prints the recommended SCA2_LONG_PATH / SCA2_CTX_CHUNK. Run it on an
idle GPU; the numbers are meaningless while something else trains.
"""
import argparse, os, statistics, time, torch

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-B", type=int, default=8); p.add_argument("-T", type=int, default=1024)
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=190)
    p.add_argument("--dv", type=int, default=56); p.add_argument("--variant", default="cshort_damph")
    p.add_argument("--paths", default="chunk,batched"); p.add_argument("--chunks", default="64,128,256")
    p.add_argument("--rounds", type=int, default=5); p.add_argument("--iters", type=int, default=6)
    a = p.parse_args(argv)
    from .ref import LayerCfg; from . import registry
    torch._dynamo.config.cache_size_limit = 64
    dev = "cuda"
    cfg = LayerCfg(a.d, a.Mc, 4, 8, 448, freq="rope", theta_scale=0.02, dv=a.dv, Ls=16, max_len=a.T)
    head = registry.build(a.variant, cfg, device=dev).c
    z = torch.randn(a.B, a.T, a.d, device=dev, requires_grad=True); h = torch.roll(z.detach(), 1, 1)
    cells = [(pth, int(c)) for pth in a.paths.split(",") for c in a.chunks.split(",")]
    fns = {}
    for pth, C in cells:
        head.LONG_PATH, head.CTX = pth, C
        f = torch.compile(head.prefill, dynamic=False)
        def run(f=f, pth=pth, C=C):
            head.LONG_PATH, head.CTX = pth, C
            f(z, h)[0].square().mean().backward()
        for _ in range(3): run()                       # compile + warm
        fns[(pth, C)] = run
    torch.cuda.synchronize()
    res = {c: [] for c in cells}
    for r in range(a.rounds):
        for c in cells:                                 # every cell inside every round
            fn = fns[c]; torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(a.iters): fn()
            torch.cuda.synchronize(); res[c].append((time.perf_counter() - t0) / a.iters * 1e3)
    ref = statistics.median(res[cells[0]])
    print(f"{a.variant} long head fwd+bwd, B={a.B} T={a.T} Mc={a.Mc} dv={a.dv}, {a.rounds} blocked rounds")
    print(f"{'path':>8} {'CTX':>4} {'median ms':>10} {'ratio':>6} {'round sd':>9}")
    best = None
    for c in cells:
        med = statistics.median(res[c]); sd = statistics.pstdev([v / statistics.median(res[cells[0]]) for v in res[c]])
        print(f"{c[0]:>8} {c[1]:>4} {med:10.2f} {med/ref:6.3f} {sd:9.3f}")
        if best is None or med < best[1]: best = (c, med)
    print(f"\nrecommended: SCA2_LONG_PATH={best[0][0]} SCA2_CTX_CHUNK={best[0][1]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
