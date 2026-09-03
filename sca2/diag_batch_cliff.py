"""
Diagnose the B>=32 cliff: sepq goes 6.77 -> 24.45 us/token from B=16 to B=32,
while attention only degrades 8% over the same range. A bigger batch should
improve occupancy, so this is anomalous and probably a bug, not a limit.

Four hypotheses, cheapest first:
  1. THERMAL/CLOCK. The sweep runs B ascending, so large B always runs on a
     hotter GPU. Test: sweep descending too. If the cliff follows the ORDER it
     is thermal; if it follows B it is real.
  2. MEMORY. Peak allocation per B, plus the reported SM clock.
  3. COMPILE. Inductor re-specializes per batch size and may pick a worse
     kernel above some B. Test: eager vs compiled at the cliff.
  4. WHICH OP. Per-kernel profile just below and just above.
"""
import argparse, subprocess, sys, time
import torch
import torch.nn as nn

from .ref import LayerCfg
from .registry import build


def clocks():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out
    except Exception:
        return "n/a"


def step_time(m, B, T, d, iters=5, warmup=3):
    x = torch.randn(B, T, d, device="cuda", requires_grad=True)
    def fb():
        m.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        m.prefill(x)[0].square().mean().backward()
    for _ in range(warmup):
        fb()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    best = float("inf")
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fb()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    mem = torch.cuda.max_memory_allocated() / 1e6
    del x
    torch.cuda.empty_cache()
    return best / (B * T) * 1e6, mem      # us per token, peak MB


def sweep(name, Bs, cfg, T, cool):
    print(f"\n--- {name}, B order {Bs} ---")
    print(f"{'B':>5s} {'us/tok':>9s} {'peak MB':>9s}   clocks/temp/throttle")
    for B in Bs:
        m = build(name, cfg, device="cuda")
        try:
            us, mem = step_time(m, B, T, cfg.d)
            print(f"{B:>5d} {us:>9.3f} {mem:>9.0f}   {clocks()}")
        except torch.OutOfMemoryError:
            print(f"{B:>5d} {'OOM':>9s}")
        del m
        torch.cuda.empty_cache()
        if cool:
            time.sleep(cool)


def main(argv=None):
    p = argparse.ArgumentParser()
    # eager by default: the sweeps re-specialize per B, so a compiled variant
    # pays 10+ compilations for no extra signal -- allocator and clock effects
    # show up identically in eager.
    p.add_argument("--variant", default="sepq")
    p.add_argument("-T", type=int, default=128)
    p.add_argument("--cool", type=float, default=8, help="seconds idle between points")
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--Md", type=int, default=16); p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=256)
    a = p.parse_args(argv)
    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff)
    Bs = [4, 8, 16, 32, 64]

    print("=== 1+2. order effect, memory, clocks ===")
    sweep(a.variant, Bs, cfg, a.T, a.cool)
    sweep(a.variant, list(reversed(Bs)), cfg, a.T, a.cool)

    print("\n=== 3. compiled vs eager at the cliff ===")
    # explicit pair: `variant.replace("_cc","")` silently compared eager to
    # eager when --variant was already eager, which is how this check first
    # produced two identical rows and no information.
    base = a.variant[:-3] if a.variant.endswith("_cc") else a.variant
    for name in (base, base + "_cc"):
        for B in (16, 32):
            m = build(name, cfg, device="cuda")
            us, mem = step_time(m, B, a.T, cfg.d)
            print(f"  {name:<10s} B={B:<4d} {us:8.3f} us/tok  {mem:6.0f} MB")
            del m; torch.cuda.empty_cache()
        time.sleep(a.cool)

    print("\n=== 4. per-kernel, B=16 vs B=32 ===")
    from torch.profiler import profile, ProfilerActivity
    for B in (16, 32):
        m = build(a.variant, cfg, device="cuda")
        x = torch.randn(B, a.T, cfg.d, device="cuda", requires_grad=True)
        def fb():
            m.zero_grad(set_to_none=True)
            if x.grad is not None: x.grad = None
            m.prefill(x)[0].square().mean().backward()
        for _ in range(4): fb()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(3): fb()
            torch.cuda.synchronize()
        ka = pr.key_averages()
        tot = sum(e.self_device_time_total for e in ka)
        print(f"  B={B}: {tot/1e3/3:.2f} ms/step total, {sum(e.count for e in ka)/3:.0f} kernels")
        for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:6]:
            print(f"      {e.self_device_time_total/1e3/3:7.3f} ms  n={e.count/3:5.1f}  {e.key[:58]}")
        del m, x; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
