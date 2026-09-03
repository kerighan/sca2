"""
Interleaved timing.

Run-to-run drift on this box reached 45% on an identical config, which is more
than most of the effects being measured. Sequential arm-after-arm sweeps
attribute that drift to whichever arm happened to run during it. Interleaving
round-robin makes drift hit every arm equally, and reporting the median over
rounds plus the spread makes it visible instead of silent.
"""
import time
import torch


def interleaved(arms, rounds=5, iters=5, warmup=3):
    """arms: dict name -> zero-arg callable performing one step.

    Returns dict name -> (median_seconds, min, max).
    """
    for fn in arms.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples = {k: [] for k in arms}
    for _ in range(rounds):
        for name, fn in arms.items():          # round-robin, not arm-at-a-time
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - t0) / iters)
    out = {}
    for k, v in samples.items():
        v = sorted(v)
        out[k] = (v[len(v) // 2], v[0], v[-1])
    return out


def fmt(out, scale=1.0, unit=""):
    w = max(len(k) for k in out)
    lines = []
    for k, (med, lo, hi) in out.items():
        lines.append(f"  {k:<{w}s} {med*scale:9.2f}{unit}  "
                     f"[{lo*scale:.2f}-{hi*scale:.2f}, spread {100*(hi-lo)/med:.0f}%]")
    return "\n".join(lines)
