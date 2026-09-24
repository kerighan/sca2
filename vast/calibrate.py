"""Measure each host's speed on one fixed workload, so wall-clock comparisons
across hosts mean something.

    python -m vast.calibrate                  # every slot, dv256, ~5 min each
    python -m vast.calibrate --slots 0,2

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

The measurement runs the REAL pretrain.py on the REAL corpus, not a
throughput bench. A bench is a different program: bench_vocab materialises
(B, T, V) logits in fp32 for its loss and OOMs at B=6 T=4096 where training
does not, and a proxy that dies at the shape being calibrated is no proxy at
all. Running the actual command also exercises the memmap dataloader, which
is where a slow host would show up.

The residual error is the calibration's own noise, a few percent, worth under
0.01 nats -- below the 0.02 floor two runs of the same function already show.

Written into vast/runtime/instances.json as `calib_tok_s`, and read by
`vast.curves` when it merges logs from several hosts.
"""
from __future__ import annotations

import argparse
import json
import time

from .common import POOL_FILE, REMOTE, instances, live, run_remote, save_instances
from .submit import build_command


def calibrate(slot: int, arm: str, seconds: int, block: int, batch: int,
              corpus: str) -> float:
    """tok/s of the reference arm on this host, from the real training loop."""
    info = live(slot)
    train = (build_command(arm, seconds / 3600, corpus, block, batch, "calib")
             .replace("--eval-every 1200", f"--eval-every {seconds // 2}")
             .replace("--eval-batches 60", "--eval-batches 5")
             .replace("--save-every 7200", "--save-every 999999")
             .replace("--pos-buckets 16", "--pos-buckets 0")
             .replace(" --class-eval", ""))
    cmd = (f"cd {REMOTE}; rm -f runs/calib.jsonl; "
           f"export PYTORCH_ALLOC_CONF=expandable_segments:True; "
           f"export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan; "
           f"{train} > runs/calib.log 2>&1; "
           f"tail -c 2000 runs/calib.log; echo ---; cat runs/calib.jsonl")
    out = run_remote(info, cmd, timeout=2400, check=False)
    rates = [json.loads(l)["tok_s"] for l in out.stdout.splitlines()
             if l.startswith("{") and "tok_s" in l]
    if not rates:
        raise RuntimeError(f"slot {slot}: no eval produced:\n{out.stdout[-900:]}")
    # The last eval, never the first: the first carries the compile time inside
    # its elapsed seconds and understates the host by a wide margin.
    return float(rates[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="dv256",
                   help="the reference workload; must be the SAME on every host")
    p.add_argument("--seconds", type=int, default=180,
                   help="training seconds timed on each host, after compile")
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--corpus", default="zyda32k")
    p.add_argument("--slots", default=None, help="comma list; default all")
    a = p.parse_args()

    pool = instances()
    slots = ([int(s) for s in a.slots.split(",")] if a.slots
             else list(range(len(pool))))
    print(f"calibrating {len(slots)} host(s) on {a.arm} "
          f"(B={a.batch} T={a.block}, identical everywhere)")
    for slot in slots:
        t0 = time.perf_counter()
        rate = calibrate(slot, a.arm, a.seconds, a.block, a.batch, a.corpus)
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
