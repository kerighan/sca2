"""Tail a remote log and report whether the job is actually alive.

    python -m vast.logs --name z_dv256
    python -m vast.logs --name prep_zyda --lines 40

Liveness is read from the PID recorded at submission, not from a process-name
match. Both are reported: a log that stopped growing while the PID is gone is
a crash, and a log that stopped growing while the PID lives is a hang -- they
need different responses and the distinction is invisible from the tail alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import REMOTE, ROOT, live, run_remote

RUNTIME = ROOT / "vast" / "runtime"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--lines", type=int, default=25)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--grep", default=None, help="filter the tail")
    args = parser.parse_args()

    info = live(args.slot)
    pid_file = RUNTIME / f"{args.name}.pid"
    pid = pid_file.read_text().strip() if pid_file.exists() else None

    if pid:
        alive = run_remote(info, f"test -d /proc/{pid} && echo alive || echo gone",
                           check=False)
        state = alive.stdout.strip()
        print(f"pid {pid}: {state}")
    else:
        print(f"no recorded pid for {args.name} (was it submitted from here?)")

    gpu = run_remote(info,
                     "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu "
                     "--format=csv,noheader", check=False)
    if gpu.returncode == 0:
        print(f"gpu: {gpu.stdout.strip()}")

    tail = f"tail -n {args.lines} {REMOTE}/runs/{args.name}.log"
    if args.grep:
        tail += f" | grep -E {args.grep!r}"
    out = run_remote(info, tail, check=False)
    print("-" * 70)
    print(out.stdout.rstrip() or out.stderr.rstrip() or "(log empty)")


if __name__ == "__main__":
    main()
