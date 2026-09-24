"""Measure each host's speed on one fixed workload, so wall-clock comparisons
across hosts mean something.

    python -m vast.calibrate                  # every slot, dv256, ~4 min each
    python -m vast.calibrate --arm dv256 --minutes 3

Why this exists. The whole conclusion rests on comparing arms at equal WALL
CLOCK. Run sequentially on one host that is exact by construction. Run in
parallel on three rented hosts and it is not: two 5090 machines differ by
5-20% depending on CPU, PCIe, cooling and whether the card is shared, and a
10% throughput difference moves the loss by about 0.1 * 0.25 = 0.025 nats at
our fitted slope -- larger than the effect being measured. Without this step a
parallel run would report the luck of the draw in hosts.

The fix is to time ONE reference arm, identical everywhere, and record
tok/s per host. Elapsed time is then converted to reference-host seconds:

    t_ref = t_actual * (this host's tok/s) / (reference host's tok/s)

The residual error is the calibration's own noise, a few percent, worth under
0.01 nats -- below the 0.02 floor two runs of the same function already show.

Written into vast/runtime/instances.json as `calib_tok_s`, and read by
`vast.curves` when it merges logs from several hosts.
"""
from __future__ import annotations

import argparse
import json
import re
import time

from .common import POOL_FILE, REMOTE, instances, live, run_remote, save_instances

PATTERN = re.compile(r"^\s+(\S+)\s+[\d,]+\s+[\d.]+\s+([\d,]+)\s", re.M)


def calibrate(slot: int, arm: str, rounds: int, block: int, batch: int) -> float:
    info = live(slot)
    cmd = (f"cd {REMOTE}; PYTORCH_ALLOC_CONF=expandable_segments:True "
           f"SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan "
           f"python -u bench_vocab.py --vocabs 32000 --arms {arm} "
           f"--blocks {block} --batch {batch} --rounds {rounds} --iters 3")
    out = run_remote(info, cmd, timeout=1800, check=False)
    text = out.stdout
    m = PATTERN.search(text)
    if not m:
        raise RuntimeError(f"slot {slot}: could not parse a rate from:\n{text[-800:]}")
    return float(m.group(2).replace(",", ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="dv256",
                   help="the reference workload; must be the SAME on every host")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--slots", default=None, help="comma list; default all")
    a = p.parse_args()

    pool = instances()
    slots = ([int(s) for s in a.slots.split(",")] if a.slots
             else list(range(len(pool))))
    print(f"calibrating {len(slots)} host(s) on {a.arm} "
          f"(B={a.batch} T={a.block}, identical everywhere)")
    for slot in slots:
        t0 = time.perf_counter()
        rate = calibrate(slot, a.arm, a.rounds, a.block, a.batch)
        pool[slot]["calib_tok_s"] = rate
        pool[slot]["calib_arm"] = a.arm
        print(f"  slot {slot} (instance {pool[slot]['id']}, "
              f"machine {pool[slot].get('machine_id')}): "
              f"{rate:,.0f} tok/s  ({time.perf_counter()-t0:.0f}s)")
    save_instances(pool)

    rates = [p_["calib_tok_s"] for p_ in pool if p_.get("calib_tok_s")]
    if len(rates) > 1:
        ref = rates[0]
        spread = (max(rates) - min(rates)) / min(rates) * 100
        print(f"\nhost spread: {spread:.1f}%  (reference = slot 0 at {ref:,.0f} tok/s)")
        for p_ in pool:
            if p_.get("calib_tok_s"):
                f = p_["calib_tok_s"] / ref
                print(f"  slot {p_['slot']}: x{f:.4f}  "
                      f"-> multiply its elapsed seconds by {f:.4f} to compare")
        if spread > 25:
            print("\nWARNING: a spread this large makes a parallel wall-clock "
                  "comparison fragile even after normalisation. Consider "
                  "running the arms sequentially on the fastest host.")
    print(f"\nwritten to {POOL_FILE}")


if __name__ == "__main__":
    main()
