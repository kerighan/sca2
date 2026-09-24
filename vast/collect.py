"""Pull the eval log and the checkpoints on an interval, and validate them.

    python -m vast.collect --log zyda                 # one pass
    python -m vast.collect --log zyda --watch 900     # every 15 min until idle

Collection happens on an interval, not only at completion: a 64 h run can
vanish at any point -- the host is reclaimed, the process is OOM-killed, the
network drops -- and a JSONL pulled every quarter hour turns a total loss into
the loss of fifteen minutes.

Validation means loading the artifact, not sizing the file. A truncated
checkpoint has a plausible size and fails only when something tries to read it,
which on the usual schedule is after the instance is gone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from .common import REMOTE, ROOT, live, run_remote, ssh_target

OUT = ROOT / "runs"


def _download(info: dict, remote: str, target: Path, timeout: int = 1800) -> bool:
    """Atomic: a partial transfer never replaces a good local copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=accept-new",
                        "-o", "ConnectTimeout=20", "-P", str(info["ssh_port"]),
                        f"root@{info['ssh_host']}:{remote}", str(tmp)],
                       check=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  {Path(remote).name}: transfer failed ({type(exc).__name__})")
        return False
    tmp.replace(target)
    return True


def _validate_jsonl(path: Path) -> int:
    rows = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                json.loads(line)          # raises on a truncated tail
                rows += 1
    return rows


def _validate_ckpt(path: Path) -> str:
    import torch
    ck = torch.load(path, map_location="cpu")
    n = sum(v.numel() for v in ck["model"].values())
    return f"{n:,} params, label {ck['cfg'].get('label')}"


def collect_once(slot: int, log_name: str, checkpoints: bool) -> bool:
    """True when at least one artifact was pulled and validated."""
    info = live(slot)
    got = False

    remote_log = f"{REMOTE}/runs/{log_name}.jsonl"
    if run_remote(info, f"test -s {remote_log}", check=False).returncode == 0:
        target = OUT / f"{log_name}.jsonl"
        if _download(info, remote_log, target, timeout=300):
            try:
                rows = _validate_jsonl(target)
                print(f"  {target.name}: {rows} evals")
                got = True
            except json.JSONDecodeError as exc:
                print(f"  {target.name}: INVALID ({exc}); keeping the instance")

    if checkpoints:
        listing = run_remote(info, f"ls -1 {REMOTE}/runs/ck_*.pt 2>/dev/null || true",
                             check=False)
        for remote_ck in [p for p in listing.stdout.split() if p.endswith(".pt")]:
            target = OUT / Path(remote_ck).name
            if _download(info, remote_ck, target):
                try:
                    print(f"  {target.name}: {_validate_ckpt(target)}")
                    got = True
                except Exception as exc:                      # noqa: BLE001
                    print(f"  {target.name}: INVALID ({exc}); keeping the instance")
    return got


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="zyda")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--watch", type=int, default=0,
                        help="seconds between passes; 0 = one pass and exit")
    parser.add_argument("--no-checkpoints", action="store_true", dest="no_ckpt")
    parser.add_argument("--deadline-hours", type=float, default=80.0,
                        help="stop watching after this long, so nothing waits forever")
    args = parser.parse_args()

    if not args.watch:
        collect_once(args.slot, args.log, not args.no_ckpt)
        return

    end = time.time() + args.deadline_hours * 3600
    while time.time() < end:
        print(f"[{time.strftime('%H:%M')}] collecting")
        try:
            collect_once(args.slot, args.log, not args.no_ckpt)
        except Exception as exc:                              # noqa: BLE001
            print(f"  pass failed: {exc}")
        info = live(args.slot)
        busy = run_remote(info, "pgrep -c -f pretrain.py || true", check=False)
        if busy.stdout.strip() in ("", "0"):
            print("no pretrain.py running: final pass, then stopping")
            collect_once(args.slot, args.log, not args.no_ckpt)
            print("teardown when you have checked the artifacts:\n"
                  "  python -m vast.teardown --yes")
            return
        time.sleep(args.watch)
    print(f"deadline of {args.deadline_hours} h reached; the instance is STILL BILLING")


if __name__ == "__main__":
    main()
