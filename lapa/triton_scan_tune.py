"""Power-of-two candidate sets; historical tiles remain available.

SCA2_SCAN_AUTOTUNE=0 selects the historical settings. The default tunes by
shape, device and dtype, with Triton's (or Inductor's) persistent cache.
Do not add early_config_prune: PyTorch 2.9 reconstructs the autotuner after
pruning and drops restore_value, corrupting recurrent states during tuning.
"""

import os
import triton

TUNE = {"fwd": (64, 32, 8, 3), "bwd": (16, 128, 8, 2),
        "bwd2": (64, 32, 4, 3)}
GTUNE = {0: (32, 128, 64, 8, 2), 1: (128, 128, 64, 8, 2),
         2: (128, 128, 64, 8, 2), 4: (128, 128, 64, 8, 2)}
KTUNE = (64, 128, 32, 8, 3)
FTUNE = (128, 128, 32, 8, 3)
AUTOTUNE = os.environ.get("SCA2_SCAN_AUTOTUNE", "1") != "0"


def scan_tuner(which, small=False, fp32=False):
    choices = [TUNE[which]]
    if small:
        choices = [(16, 16, 4, 1)]
    elif AUTOTUNE and fp32:
        # tf32x3 operands make the large bf16 candidates needlessly expensive
        # to compile (and often exceed shared memory). Tune a bounded fp32 set.
        choices += [(16, 32, 4, 2), (32, 32, 8, 3) if which == "fwd" else (32, 32, 4, 2)]
    elif AUTOTUNE:
        choices += [x for x in (
            (16, 32, 4, 2), (16, 64, 4, 2), (16, 128, 4, 2),
            (32, 32, 4, 2), (32, 64, 4, 2), (32, 64, 8, 2),
            (64, 32, 4, 2), (64, 64, 4, 2), (64, 64, 8, 2),
            (32, 32, 8, 3), (64, 32, 8, 1), (128, 32, 8, 1),
        ) if x != choices[0]]
    if which == "fwd" and AUTOTUNE and not small and not fp32:
        choices += [(16, 32, 8, 1), (32, 16, 8, 1), (64, 32, 16, 1), (32, 32, 16, 1)]
    return triton.autotune(
        configs=tuple(triton.Config(dict(BN=bn, TK=tk), num_warps=nw, num_stages=ns)
                      for bn, tk, nw, ns in choices),
        key=["B", "K", "C", "R", "D", "G", "SAVE", "DEV"],
        restore_value=["ST"] if which == "fwd" else (["DSR"] if which == "bwd2" else []),
        cache_results=True,
    )


def matrix_tuner(which, small=False, fp32=False):
    names = ("TM", "TN", "TK" if which == "k2" else "TW")
    old = KTUNE if which == "k2" else (GTUNE[0] if which == "grad0" else
                                      GTUNE[1] if which == "grad" else FTUNE)
    if fp32 and old[0] >= 64:
        old = (*old[:4], 1)  # The old eager _launch could retry at depth one.
    choices = [old]
    if small:
        choices = [(16, 16, 16, 4, 1)]
    elif AUTOTUNE and fp32:
        choices += [(32, 64, 32, 4, 2), (64, 64, 32, 4, 2)]
    elif AUTOTUNE:
        choices += [x for x in (
            (32, 64, 32, 4, 2), (64, 64, 32, 4, 2),
            (64, 128, 32, 4, 2), (128, 64, 32, 4, 2),
            (128, 128, 32, 8, 2), (128, 64, 64, 4, 2),
            (128, 128, 64, 8, 2), (32, 128, 64, 8, 2),
        ) if x != old]
    return triton.autotune(
        configs=tuple(triton.Config(dict(zip(names, x[:3])), num_warps=x[3], num_stages=x[4])
                      for x in choices),
        key=["B", "K", "C", "R", "D", "G", "KIND", "MODE", "DEV"],
        restore_value=["OUT"] if which == "k2" else [],
        cache_results=True,
    )
