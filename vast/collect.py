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


def _key(row: dict):
    """What makes an eval unique: which arm, and where it was in its run."""
    return (row.get("model"), row.get("seed"), row.get("step"), row.get("tokens"))


def _merge_jsonl(fresh: Path, archive: Path) -> tuple[int, int]:
    """Union the newly pulled evals into an append-only local archive.

    Replacing the local file with the remote one is not safe: the remote is
    truncated when a run is relaunched with `: > log`, a partial transfer can
    land mid-line, and either way the history that only existed locally is
    gone. Merging on (model, seed, step, tokens) means a pull can only ever ADD
    evals, so no accident upstream can destroy what has already been collected.

    Returns (rows in the archive, rows this pass added).
    """
    seen, out = {}, []
    if archive.exists():
        with archive.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # a truncated local tail is dropped
                k = _key(row)
                if k not in seen:
                    seen[k] = True
                    out.append(line if line.endswith("\n") else line + "\n")
    added = 0
    with fresh.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                          # the remote tail can be mid-write
            k = _key(row)
            if k not in seen:
                seen[k] = True
                out.append(line if line.endswith("\n") else line + "\n")
                added += 1
    tmp = archive.with_suffix(archive.suffix + ".part")
    tmp.write_text("".join(out))
    tmp.replace(archive)
    return len(out), added


def _validate_ckpt(path: Path) -> str:
    import torch
    ck = torch.load(path, map_location="cpu")
    # Count STORAGES, not keys. A state_dict lists every name, and several name
    # the same tensor: weight tying puts the embedding under `e.weight` and
    # `o.weight`, and `self.layer = self.layers[0]` puts layer 0 under both.
    # Summing numel over keys reported 163.6M for a 120.0M model, which reads
    # like a corrupt or mismatched checkpoint and is only double counting.
    seen, n = set(), 0
    for v in ck["model"].values():
        key = v.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            n += v.numel()
    return f"{n:,} params, label {ck['cfg'].get('label')}"


def collect_once(slot: int, log_name: str, checkpoints: bool) -> bool:
    """True when at least one artifact was pulled and validated."""
    info = live(slot)
    got = False

    remote_log = f"{REMOTE}/runs/{log_name}.jsonl"
    if run_remote(info, f"test -s {remote_log}", check=False).returncode == 0:
        archive = OUT / f"{log_name}.jsonl"
        staged = OUT / f".{log_name}.jsonl.remote"
        if _download(info, remote_log, staged, timeout=300):
            total, added = _merge_jsonl(staged, archive)
            staged.unlink(missing_ok=True)
            print(f"  {archive.name}: {total} evals (+{added})")
            got = True

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
    parser.add_argument("--ckpt-every", type=int, default=8, dest="ckpt_every",
                        help="pull checkpoints every Nth pass (they are ~600 MB; "
                             "the eval log comes every pass)")
    parser.add_argument("--no-checkpoints", action="store_true", dest="no_ckpt")
    parser.add_argument("--deadline-hours", type=float, default=80.0,
                        help="stop watching after this long, so nothing waits forever")
    args = parser.parse_args()

    if not args.watch:
        collect_once(args.slot, args.log, not args.no_ckpt)
        return

    # The eval log is a few hundred KB and IS the scientific result, so it is
    # pulled every pass. A checkpoint is ~600 MB and only supports the
    # structural analysis, so it is pulled every Nth. What bounds the loss when
    # a host disappears is cadence, not completeness.
    end = time.time() + args.deadline_hours * 3600
    n = 0
    while time.time() < end:
        n += 1
        want_ckpt = (not args.no_ckpt) and (n % args.ckpt_every == 0)
        print(f"[{time.strftime('%H:%M')}] pass {n}"
              f"{' (with checkpoints)' if want_ckpt else ''}")
        try:
            collect_once(args.slot, args.log, want_ckpt)
        except Exception as exc:                              # noqa: BLE001
            # A transient ssh or API failure must never end the watch: that
            # would leave the run uncollected AND the instance billing.
            print(f"  pass failed ({type(exc).__name__}: {exc}); retrying next pass")
            time.sleep(args.watch)
            continue
        try:
            info = live(args.slot)
            # bracketed so the pattern cannot match the shell carrying it
            busy = run_remote(info, "pgrep -c -f '[p]retrain[.]py' || true",
                              check=False)
            idle = busy.stdout.strip() in ("", "0")
        except Exception:                                     # noqa: BLE001
            idle = False                                      # unknown is not idle
        if idle:
            print("no pretrain.py running: final pass with checkpoints, then stopping")
            try:
                collect_once(args.slot, args.log, not args.no_ckpt)
            except Exception as exc:                          # noqa: BLE001
                print(f"  final pass failed: {exc}; the instance is KEPT")
                return
            print("teardown when you have checked the artifacts:\n"
                  "  python -m vast.teardown --yes")
            return
        time.sleep(args.watch)
    print(f"deadline of {args.deadline_hours} h reached; the instance is STILL BILLING")


if __name__ == "__main__":
    main()
