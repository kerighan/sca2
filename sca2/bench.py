"""Timing harness: prefill (fwd, fwd+bwd) and token-by-token decode."""
import argparse, time, sys, torch
from .ref import LayerCfg
from .registry import VARIANTS, build


def _sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def _time(fn, dev, iters, warmup=3, trials=3):
    for _ in range(warmup):
        fn()
    _sync(dev)
    best = float("inf")
    for _ in range(trials):
        _sync(dev); t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(dev)
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def bench_one(name, cfg, B, T, device, dtype, decode_steps, do_bwd=True):
    m = build(name, cfg, device=device, dtype=dtype)
    x = torch.randn(B, T, cfg.d, device=device, dtype=dtype)
    r = {"name": name}

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        r["prefill_ms"] = _time(lambda: m.prefill(x), device, 5) * 1e3
    if device.startswith("cuda"):
        r["prefill_MB"] = torch.cuda.max_memory_allocated() / 1e6

    if do_bwd:
        xg = x.clone().requires_grad_(True)
        def fb():
            m.zero_grad(set_to_none=True)
            m.prefill(xg)[0].square().mean().backward()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        r["train_ms"] = _time(fb, device, 5) * 1e3
        if device.startswith("cuda"):
            r["train_MB"] = torch.cuda.max_memory_allocated() / 1e6

    # decode: warm the state with a prefill, then step
    with torch.no_grad():
        _, st0 = m.prefill(x)
        xt = torch.randn(B, cfg.d, device=device, dtype=dtype)
        def dec():
            st = st0
            for _ in range(decode_steps):
                _, st = m.step(xt, st)
        r["decode_us_per_tok"] = _time(dec, device, 3, warmup=3, trials=5) / decode_steps * 1e6

        # same recurrence, replayed from a captured CUDA graph
        r["graph_us_per_tok"] = float("nan")
        if device.startswith("cuda"):
            from .decode import GraphDecoder, _check_capturable
            try:
                _check_capturable(m.init_state(B, x.device, dtype))
                dec_g = GraphDecoder(m, m.init_state(B, x.device, dtype))
                def decg():
                    for _ in range(decode_steps):
                        dec_g.step(xt)
                # graph replay is cheap enough that 1 iteration is mostly noise
                r["graph_us_per_tok"] = _time(decg, device, 5, warmup=5, trials=5) / decode_steps * 1e6
            except TypeError:
                pass
    del m
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return r


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="*", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("-B", type=int, default=8)
    p.add_argument("-T", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--no-bwd", action="store_true")
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--Md", type=int, default=16); p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=256)
    a = p.parse_args(argv)

    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff)
    dtype = getattr(torch, a.dtype)
    names = a.variants or list(VARIANTS)
    print(f"device={a.device} dtype={a.dtype} B={a.B} T={a.T} cfg={cfg}")
    hdr = (f"{'variant':<16s} {'prefill ms':>11s} {'MB':>7s} {'train ms':>9s} {'MB':>7s} "
           f"{'decode us/tok':>14s} {'graph us/tok':>13s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for n in names:
        try:
            r = bench_one(n, cfg, a.B, a.T, a.device, dtype, a.decode_steps, not a.no_bwd)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{n:<12s} {'OOM':>11s}"); continue
        rows.append(r)
        g = r.get('graph_us_per_tok', float('nan'))
        print(f"{r['name']:<16s} {r['prefill_ms']:>11.2f} {r.get('prefill_MB',0):>7.0f} "
              f"{r.get('train_ms',float('nan')):>9.2f} {r.get('train_MB',0):>7.0f} "
              f"{r['decode_us_per_tok']:>14.1f} "
              f"{('-' if g!=g else f'{g:.1f}'):>13s}")
    if len(rows) > 1:
        base = rows[0]
        print("\nspeedup vs " + base["name"])
        for r in rows[1:]:
            s = lambda k: base[k] / r[k] if r.get(k) else float("nan")
            g = base['decode_us_per_tok'] / r['graph_us_per_tok'] if r['graph_us_per_tok'] == r['graph_us_per_tok'] else float('nan')
            print(f"  {r['name']:<16s} prefill x{s('prefill_ms'):.2f}  "
                  f"train x{s('train_ms'):.2f}  decode x{s('decode_us_per_tok'):.2f}  "
                  f"graph-decode x{g:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
