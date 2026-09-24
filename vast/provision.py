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
import subprocess
import time

from .common import PROJECT, save_instances, ssh_target, vast, wait_for_instance, account

# A CUDA-enabled Torch is already in the image; never let pip replace it.
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"

# 100 GB: the Zyda-2 corpus is built on the instance (~30 GB at 15B tokens as
# uint16) and each arm saves a ~0.5 GB checkpoint five times.
DISK_GB = 100

# Two families are acceptable. Ada (4090) has more dense bf16 than Blackwell
# consumer (5090) but a third of its bandwidth; the layer is launch-bound on
# the GB10 rather than roofline-bound, so neither spec predicts our throughput
# and the choice is made on price. cuda_max_good>=12.8 keeps the toolchain able
# to build for both.
QUERY = (
    "rentable=true verified=true num_gpus=1 "
    "gpu_name in [RTX_4090,RTX_5090] gpu_ram>=24 "
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

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

    for slot, offer in enumerate(preview):
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

    pool, attempted = [], set()
    try:
        for offer in offers:
            if len(pool) == args.count:
                break
            machine = offer.get("machine_id")
            if machine in attempted:
                continue
            attempted.add(machine)
            slot, instance_id = len(pool), None
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
                print(f"offer {offer['id']} ({machine}) unusable: {exc}")
                if instance_id is not None:
                    try:
                        vast("destroy", "instance", str(instance_id), "--yes")
                    except Exception:                     # noqa: BLE001
                        print(f"  WARNING: could not destroy {instance_id}, "
                              f"check `vastai show instances`")
        if len(pool) < args.count:
            raise RuntimeError(f"only provisioned {len(pool)}/{args.count}")
    finally:
        if pool:
            save_instances(pool)
            print(f"\nrecorded {len(pool)} instance(s) in vast/runtime/")
            print("next: python -m vast.setup")


if __name__ == "__main__":
    main()
