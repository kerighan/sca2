"""Read runs/wgroup.jsonl: does a per-value-group spectral read help the C head?

The control is WG=1, which IS v3polarflat, i.e. the md4_dv32 arm already logged
at n=3 -- so it is not rerun. Every arm is interpolated to the same token count,
which matters here because WG costs throughput (1.6x at WG=4), so equal
wall-clock is NOT equal tokens.

    python dump_wg.py
"""
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
FF = {"wg2": 361, "wg4": 355, "wg8": 343}


def series(path, label):
    if not (RUNS / path).exists():
        return []
    v = [json.loads(line) for line in (RUNS / path).open()
         if json.loads(line)["model"] == label]
    return sorted(v, key=lambda r: r["step"])


def main():
    arms = {k: series("wgroup.jsonl", k) for k in FF}
    arms = {k: v for k, v in arms.items() if v}
    ctrl = series("md_axis.jsonl", "md4_dv32")
    if not arms or not ctrl:
        print("nothing logged yet")
        return

    T = min(min(v[-1]["tokens"] for v in arms.values()), ctrl[-1]["tokens"])
    def at(v, k):
        return float(np.interp(T, [r["tokens"] for r in v], [r[k] for r in v]))

    def bucket(v, i):
        return float(np.interp(T, [r["tokens"] for r in v],
                               [r["pos"][i] for r in v]))

    print(f"all arms interpolated to {T/1e6:.1f}M tokens, seed 0, ~743.4k params\n")
    print(f"{'arm':6} {'WG':>3} {'ff':>4} {'val':>8} {'vs WG=1':>8} "
          f"{'slope':>8} {'tok/s':>7}")
    print("-" * 50)
    v1 = at(ctrl, "val")
    s1 = bucket(ctrl, 7) - bucket(ctrl, 0)
    print(f"{'wg1':6} {1:>3} {364:>4} {v1:>8.4f} {'--':>8} {s1:>+8.3f} "
          f"{ctrl[-1]['tok_s']:>7}")
    for k, v in arms.items():
        sl = bucket(v, 7) - bucket(v, 0)
        print(f"{k:6} {int(k[2:]):>3} {FF[k]:>4} {at(v,'val'):>8.4f} "
              f"{at(v,'val')-v1:>+8.4f} {sl:>+8.3f} {v[-1]['tok_s']:>7}")

    print("\nposition profile, delta vs WG=1 (negative = better)")
    hdr = "".join(f"{k:>9}" for k in arms)
    print(f"{'bucket':>10}{hdr}")
    for i in range(8):
        b = bucket(ctrl, i)
        row = "".join(f"{bucket(v,i)-b:>+9.3f}" for v in arms.values())
        print(f"{i*128:>5}-{i*128+127:<4}{row}")

    print("\nseed sd on this corpus is 0.03-0.07: a single-seed delta smaller "
          "than that\nis not resolved, whatever its sign.")


if __name__ == "__main__":
    main()
