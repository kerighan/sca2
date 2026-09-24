"""Launch a training arm detached and record its PID.

    python -m vast.submit --name z_dv256 --arm dv256 --hours 64
    python -m vast.submit --name corpus 'python prep_zyda.py --tokens 15000000000 ...'

Tracked by the PID written at launch, never by a `pgrep -f` pattern: the
pattern matches the grep itself and a watcher built on it reports a dead run
as alive until the deadline. The PID goes to vast/runtime/<name>.pid and the
collector reads it.

Arms share one GPU, so they are queued rather than run concurrently: at these
shapes a second process halves both their throughputs and the wall-clock
comparison -- the only metric that matters here -- becomes meaningless.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import REMOTE, ROOT, live, run_remote

RUNTIME = ROOT / "vast" / "runtime"

# The arms of the decisive experiment. Everything except Mc/dv/ff is shared,
# so what differs between them is exactly the mixer size.
ARMS = {
    "dv128": "--Mc 128 --dv 128 --ff 4096",
    "dv256": "--Mc 256 --dv 256 --ff 4096",
    "dv384": "--Mc 384 --dv 384 --ff 4096",
}
LAPA_FLAGS = ("--variant lapa_cc --Ls 128 --theta-scale 0.02 --rope-base 2048 "
              "--slow-frac 0.25 --conv 4 --layer-scale --lam-free --damp-mem 4,20000 "
              "--gdn-gate --v-silu --init-v2")
GDN_FLAGS = "--variant gdn_cc --gdn-heads 8 --gdn-head-k 128 --gdn-expand-v 1.0"


def build_command(arm: str, hours: float, corpus: str, block: int, batch: int,
                  log: str) -> str:
    common = (
        f"--data {corpus}.json --block {block} --batch {batch} --d 1024 --layers 8 "
        f"--Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0 "
        f"--seconds {int(hours*3600)} --eval-batches 60 --eval-every 1200 "
        f"--pos-buckets 16 --samples 0 --only sca2 --log runs/{log}.jsonl "
        f"--class-eval --save-every 7200"
    )
    if arm == "gdn":
        return f"python -u pretrain.py --label z_gdn --seed 0 {common} {GDN_FLAGS} --ff 4096"
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; known: {', '.join(ARMS)} , gdn")
    return (f"python -u pretrain.py --label z_{arm} --seed 0 {common} "
            f"{LAPA_FLAGS} {ARMS[arm]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default=None,
                        help="raw remote command; omit and pass --arm instead")
    parser.add_argument("--name", required=True)
    parser.add_argument("--arm", default=None, help=f"one of {', '.join(ARMS)}, gdn")
    parser.add_argument("--hours", type=float, default=64.0)
    parser.add_argument("--corpus", default="zyda32k")
    parser.add_argument("--block", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--log", default="zyda")
    parser.add_argument("--slot", type=int, default=0)
    args = parser.parse_args()

    if (args.command is None) == (args.arm is None):
        parser.error("pass exactly one of a raw command or --arm")
    command = args.command or build_command(
        args.arm, args.hours, args.corpus, args.block, args.batch, args.log)

    info = live(args.slot)
    if args.arm:
        check = run_remote(info, f"test -s {REMOTE}/{args.corpus}.json", check=False)
        if check.returncode != 0:
            raise SystemExit(f"{args.corpus}.json is not on the host yet; "
                             f"finish the corpus build first "
                             f"(python -m vast.logs --name prep_zyda)")

    busy = run_remote(info, f"pgrep -c -f 'pretrain.py' || true", check=False)
    if busy.stdout.strip() not in ("", "0"):
        raise SystemExit("a pretrain.py is already running on this host; "
                         "arms are queued one at a time on a single GPU")

    remote = (
        f"cd {REMOTE} && mkdir -p runs && "
        f"nohup env PYTORCH_ALLOC_CONF=expandable_segments:True "
        f"SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan "
        f"sh -c {json.dumps(command)} > runs/{args.name}.log 2>&1 < /dev/null & echo $!"
    )
    out = run_remote(info, remote, timeout=120)
    pid = out.stdout.strip().splitlines()[-1]
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / f"{args.name}.pid").write_text(pid)
    (RUNTIME / f"{args.name}.cmd").write_text(command)
    print(f"submitted {args.name}  remote pid {pid}")
    print(f"  command: {command}")
    print(f"  logs:    python -m vast.logs --name {args.name}")
    print(f"  collect: python -m vast.collect --log {args.log}   # start this now")


if __name__ == "__main__":
    main()
