"""Rent a GPU, wait for SSH, and prove a real CUDA allocation before spending.

    python -m vast.provision --dry-run          # preview offers only
    python -m vast.provision --max-hourly 0.80

The validation is not cosmetic. The layer runs a hand-written Triton kernel
(`SCA2_LONG_PATH=triton_scan`) and the GDN baseline runs flash-linear-attention's
own kernels; both need a recent toolchain, and a host that merely answers SSH
can still fail to compile them. This allocates a real tensor and reports the
compute capability so the kernel port is validated before any long job.
"""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import signal
import subprocess
import json
import time

from .common import (PROJECT, POOL_FILE, save_instances, ssh_target, vast,
                     wait_for_instance, account)


class Interrupted(Exception):
    """A signal, raised where the rent/validate loop can clean up after it.

    `timeout 1500 python -m vast.provision` sends SIGTERM, whose default action
    is to die immediately. An instance created moments earlier is then neither
    destroyed by the `except` nor recorded by the `finally`: it bills with
    nothing pointing at it, and every local check still looks clean. Turning the
    signal into an ordinary exception puts it back inside the handling that
    already exists. `vast.adopt` recovers the older orphans.
    """


def _raise_on_signal(signum, _frame):
    raise Interrupted(f"signal {signum}")


# A CUDA-enabled Torch is already in the image; never let pip replace it.
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"

# 100 GB: the Zyda-2 corpus is built on the instance (~30 GB at 15B tokens as
# uint16) and each arm saves a ~0.5 GB checkpoint five times.
DISK_GB = 100

# 5090 only, and gpu_ram>=30. The earlier query took either family at
# gpu_ram>=24, which was written before the batch size was measured: at B=6
# T=4096 every arm peaks at 22.2-22.3 GiB allocated, and a 24 GB 4090 has 23.99
# GiB total. Subtract the CUDA context and the allocator's fragmentation and it
# does not fit -- for ANY arm, since the peak is set by the residual and the
# FFN, which are identical across them. A rented 4090 would OOM hours in, after
# the corpus was built on it.
#
# It also keeps the pool homogeneous. Comparing arms at equal wall clock across
# a 4090 (sm 8.9) and a 5090 (sm 12.0) would lean on `vast.calibrate` to
# normalise a gap far larger than the few percent it is meant for.
QUERY = (
    "rentable=true verified=true num_gpus=1 "
    "gpu_name=RTX_5090 gpu_ram>=30 "
    "reliability>0.98 inet_down>=500 inet_up>=100 "
    "cuda_max_good>=12.8 disk_space>=120"
)

VALIDATE = (
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && "
    "python -c \""
    "import torch;"
    "assert torch.cuda.is_available(), 'no CUDA';"
    "d=torch.device('cuda');"
    "x=torch.randn(4096,4096,device=d,dtype=torch.bfloat16);"
    "y=(x@x).float().sum().item();"
    "cap=torch.cuda.get_device_capability();"
    "print('torch', torch.__version__, 'sm', '%d.%d' % cap, "
    "'name', torch.cuda.get_device_name(), 'matmul ok', y==y)\""
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hourly", type=float, default=0.90)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--append", action="store_true",
                        help="add to the recorded pool instead of replacing it")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    # Saving a fresh pool over an existing one does not stop the old instances:
    # it only stops us from KNOWING about them, and an untracked instance bills
    # until someone notices. Either reuse what is already rented or say so.
    existing: list[dict] = []
    if POOL_FILE.exists():
        existing = json.loads(POOL_FILE.read_text())
    if existing and not args.append:
        raise SystemExit(
            f"{len(existing)} instance(s) already recorded: "
            + ", ".join(str(p["id"]) for p in existing)
            + "\npass --append to keep them and rent alongside, or "
              "`python -m vast.teardown --yes` to release them first.")
    base = len(existing)

    who = account()
    print(f"account {who.get('id')}  credit ${who.get('credit', 0):.2f}  "
          f"balance ${who.get('balance', 0):.2f}")

    offers = vast("search", "offers", QUERY, "-o", "dlperf_usd-", "--limit", "30")
    offers = [o for o in offers if float(o["dph_total"]) <= args.max_hourly]
    if not offers:
        raise RuntimeError(f"no offer at or under ${args.max_hourly}/h for: {QUERY}")

    preview, seen_machines = [], set()
    for offer in offers:
        if offer.get("machine_id") in seen_machines:
            continue
        seen_machines.add(offer.get("machine_id"))
        preview.append(offer)
        if len(preview) == args.count:
            break
    if len(preview) < args.count:
        raise RuntimeError(f"only {len(preview)} distinct hosts found")

    for slot, offer in enumerate(preview, start=base):
        print(f"slot {slot}: offer {offer['id']}  {offer['gpu_name']} "
              f"{offer['gpu_ram']/1024:.0f} GB  "
              f"down {offer.get('inet_down', 0):.0f} Mb/s  "
              f"${offer['dph_total']:.3f}/h")
    total = sum(float(o["dph_total"]) for o in preview)
    print(f"pool: ${total:.3f}/h  =  ${total*24:.2f}/day  =  ${total*64:.2f} for a 64 h arm")
    if args.dry_run:
        return

    key = Path.home() / ".ssh" / "id_ed25519.pub"
    if not key.exists():
        raise FileNotFoundError(f"missing SSH public key: {key}")
    env = os.environ.copy()
    env["VAST_API_KEY"] = __import__("vast.common", fromlist=["api_key"]).api_key()
    encoded = base64.b64encode(key.read_bytes()).decode()
    onstart = ("mkdir -p /root/.ssh; echo " + encoded
               + " | base64 -d >> /root/.ssh/authorized_keys; "
                 "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _raise_on_signal)

    pool, attempted = [], set()
    try:
        for offer in offers:
            if len(pool) == args.count:
                break
            machine = offer.get("machine_id")
            if machine in attempted:
                continue
            attempted.add(machine)
            slot, instance_id = base + len(pool), None
            try:
                created = vast("create", "instance", str(offer["id"]), "--image", IMAGE,
                               "--disk", str(DISK_GB), "--ssh", "--direct",
                               "--onstart-cmd", onstart,
                               "--label", f"{PROJECT}-{slot}")
                instance_id = int(created["new_contract"])
                info = wait_for_instance(instance_id, timeout=420)
                subprocess.run(["vastai", "attach", "ssh", str(instance_id), str(key)],
                               env=env, check=True, capture_output=True)
                ok = False
                for _ in range(40):
                    check = subprocess.run([*ssh_target(info), VALIDATE],
                                           text=True, capture_output=True, timeout=120)
                    if check.returncode == 0:
                        print(f"slot {slot} instance {instance_id} validated:")
                        for line in check.stdout.strip().splitlines():
                            print("   ", line)
                        ok = True
                        break
                    time.sleep(15)
                if not ok:
                    raise RuntimeError(f"instance {instance_id} never validated CUDA")
                pool.append({"id": instance_id, "slot": slot,
                             "gpu": offer["gpu_name"], "dph": float(offer["dph_total"]),
                             "machine_id": machine, "calib_tok_s": None})
            except Exception as exc:                      # noqa: BLE001
                stop = isinstance(exc, Interrupted)
                print(f"offer {offer['id']} ({machine}) "
                      f"{'interrupted' if stop else 'unusable'}: {exc}")
                if instance_id is not None:
                    try:
                        vast("destroy", "instance", str(instance_id), "--yes")
                    except Exception:                     # noqa: BLE001
                        print(f"  WARNING: could not destroy {instance_id}, "
                              f"recover it with `python -m vast.adopt`")
                if stop:
                    raise
        if len(pool) < args.count:
            raise RuntimeError(f"only provisioned {len(pool)}/{args.count}")
    finally:
        if pool:
            save_instances(existing + pool)
            print(f"\nrecorded {len(existing) + len(pool)} instance(s) in vast/runtime/ ({len(pool)} new)")
            print("next: python -m vast.setup")


if __name__ == "__main__":
    main()
