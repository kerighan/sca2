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
import re
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
# Experimental arms, all dv256 so the control is the dv256 already running.
EXTRA = {
    "dsoft":    "--decay-input --decay-softplus",   # GDN's actual gate form
    "dexp":     "--decay-input",                    # the exp() form that was dropped
    "postnorm": "--post-norm",                      # RMSNorm on the mixer output
    "mix4":     "--read-mix 4 --post-norm",         # mixture of Laplace kernels
    "mix4only": "--read-mix 4",
    # mix4 plus the two things the trained checkpoints say are missing: the
    # decays collapse from a 5000x span of timescales to 12-52x on half the
    # layers, and the frequency grid they would otherwise have to compensate
    # for is a frozen buffer.
    "mixanch":  "--read-mix 4 --post-norm --lam-anchor 0.5 --learn-omega",
    # mix4 plus an unbounded per-token write weight, the seqcond/nautile idea:
    # sigmoid can only attenuate a write, softplus lets a salient token dominate.
    "mixsal":   "--read-mix 4 --post-norm --beta-write --beta-softplus",
}
GDN_FLAGS = "--variant gdn_cc --gdn-heads 8 --gdn-head-k 128 --gdn-expand-v 1.0"
BPE = "zyda_bpe32k"


def build_command(arm: str, hours: float, corpus: str, block: int, batch: int,
                  log: str, bpe: str = BPE) -> str:
    common = (
        f"--data {corpus}.json --block {block} --batch {batch} --d 1024 --layers 8 "
        f"--Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0 "
        f"--seconds {int(hours*3600)} --eval-batches 60 --eval-every 1200 "
        f"--pos-buckets 16 --samples 0 --only sca2 --log runs/{log}.jsonl "
        # --class-eval reads the BPE to split "new word" from "repeated word",
        # and its default prefix is the old 16k codeparrot table, which is not
        # on the host: the arm then dies in its first second. --tie-embed is
        # worse, because it does NOT fail -- it quietly adds 33 M untied output
        # params to every arm and changes what the comparison measures.
        # --save-every is INERT without --save: pretrain.py sets its next-save
        # deadline to infinity when no path is given, so neither the mid-run
        # checkpoints nor the final one are written, and a 40 h arm ends with
        # its curve and no weights.
        f"--class-eval --bpe {bpe} --tie-embed "
        f"--save runs/ck_{log} --save-every 7200"
    )
    if arm == "gdn":
        return f"python -u pretrain.py --label z_gdn --seed 0 {common} {GDN_FLAGS} --ff 4096"
    if arm in EXTRA:
        return (f"python -u pretrain.py --label z_{arm} --seed 0 {common} "
                f"{LAPA_FLAGS} {ARMS['dv256']} {EXTRA[arm]}")
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; known: "
                         f"{', '.join(list(ARMS) + list(EXTRA))} , gdn")
    return (f"python -u pretrain.py --label z_{arm} --seed 0 {common} "
            f"{LAPA_FLAGS} {ARMS[arm]}")


def unknown_flags(command: str, src: str | None = None) -> list[str]:
    """Flags in `command` that pretrain.py does not declare.

    `src` is the REMOTE pretrain.py when the caller can read it, and that is the
    point: checking the local copy passed --beta-softplus on a host whose
    pretrain.py predated the flag, and the arm died seconds after being
    detached. The file that runs is the file to ask. argparse is the authority,
    not the config dataclass -- flags have twice existed in one and not the
    other.
    """
    if src is None:
        src = (ROOT / "pretrain.py").read_text()
    declared = set(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9-]+)"', src))
    used = {w for w in command.split() if w.startswith("--")}
    return sorted(used - declared)


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
    parser.add_argument("--bpe", default=BPE)
    parser.add_argument("--slot", type=int, default=0)
    args = parser.parse_args()

    if (args.command is None) == (args.arm is None):
        parser.error("pass exactly one of a raw command or --arm")
    command = args.command or build_command(
        args.arm, args.hours, args.corpus, args.block, args.batch, args.log,
        args.bpe)

    info = live(args.slot)
    # Read the pretrain.py that will actually run, not the one on this machine.
    remote_src = run_remote(info, f"cat {REMOTE}/pretrain.py", timeout=120,
                            check=False).stdout
    bad = unknown_flags(command, remote_src or None)
    if bad:
        raise SystemExit(
            f"the pretrain.py ON SLOT {args.slot} does not declare: "
            + ", ".join(bad) + "\n(ship the code first: it is probably stale)")
    if args.arm:
        # Every input the command names must exist BEFORE the job is detached:
        # once it is, a missing file shows up only as an empty log, and the
        # card bills while nothing runs.
        need = [f"{REMOTE}/{args.corpus}.json", f"{REMOTE}/{args.corpus}.bin",
                f"{REMOTE}/{args.bpe}-vocab.json", f"{REMOTE}/{args.bpe}-merges.txt"]
        check = run_remote(
            info, " ; ".join(f"test -s {p} || echo MISSING {p}" for p in need),
            check=False)
        missing = [ln.split()[1] for ln in check.stdout.split("\n") if "MISSING" in ln]
        if missing:
            raise SystemExit("not on the host yet:\n  " + "\n  ".join(missing)
                             + "\nfinish the corpus build first "
                               "(python -m vast.logs --name prep_zyda)")

    # The bracket keeps the pattern from matching the shell that carries it:
    # pgrep -f sees every command line including its own, so a bare
    # 'pretrain.py' always reports at least one match and every submission
    # would be refused.
    busy = run_remote(info, "pgrep -c -f '[p]retrain[.]py' || true", check=False)
    if busy.stdout.strip() not in ("", "0"):
        raise SystemExit("a pretrain.py is already running on this host; "
                         "arms are queued one at a time on a single GPU")

    # Separated by `;`, not `&&`. In `A && B && nohup C & echo $!` the `&`
    # applies to the WHOLE chain, so bash backgrounds a subshell that still
    # holds ssh's stdout and stderr open -- ssh then waits for the job to end
    # instead of returning, the submit times out, and no PID is ever recorded
    # while the job runs on regardless. With `;` the `&` binds to the nohup
    # alone, whose streams are redirected, and ssh returns at once.
    remote = (
        f"cd {REMOTE}; mkdir -p runs; "
        f"nohup env PYTORCH_ALLOC_CONF=expandable_segments:True "
        f"SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan "
        # `exec` so sh REPLACES itself with python instead of waiting on it.
        # Without it the launch produces two processes: bash forks a subshell
        # for the redirections, `$!` names that subshell, and python is its
        # grandchild. `kill $!` then removes the wrapper and leaves python
        # orphaned onto init, still training and still holding 25 GiB of the
        # card -- the liveness check reads "stopped" while the job runs on.
        f"sh -c {json.dumps('exec ' + command)} "
        f"> runs/{args.name}.log 2>&1 < /dev/null & "
        # POLL for the training process rather than sleeping a fixed time and
        # hoping. `python -u pretrain.py` spends seconds importing torch before
        # it is visible, and a single `sleep 3` silently fell back to `$!` --
        # recording the subshell again, two below the PID that matters.
        f"echo $!; "
        f"for i in $(seq 1 30); do "
        # Match the PROCESS NAME, not just the command line. `sh -c "exec python
        # ... pretrain.py ..."` carries "pretrain.py" in its own cmdline until
        # the exec lands, so a pgrep -f can return the wrapper -- it did, and
        # the recorded PID was two below the real one while the job ran fine.
        f"  p=$(ps -eo pid,comm,args | awk '$2==\"python\" && /pretrain[.]py/ {{print $1}}' | head -1); "
        f"  if [ -n \"$p\" ]; then echo JOBPID=$p; break; fi; sleep 2; "
        f"done"
    )
    out = run_remote(info, remote, timeout=120)
    lines = [l.strip() for l in out.stdout.strip().splitlines()]
    # The PID of the training process itself, never the shell that started it:
    # killing the wrapper leaves python orphaned onto init, still training and
    # still holding the card, while the liveness check reads "stopped". submit
    # refuses to run when another pretrain.py is up, so this cannot match
    # somebody else's job.
    found = [l[7:] for l in lines if l.startswith("JOBPID=") and l[7:].isdigit()]
    if not found:
        raise SystemExit(f"launched but could not resolve the training PID; "
                         f"check `python -m vast.logs --name {args.name}`:\n"
                         + "\n".join(lines[-5:]))
    pid = found[-1]
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / f"{args.name}.pid").write_text(pid)
    (RUNTIME / f"{args.name}.cmd").write_text(command)
    print(f"submitted {args.name}  remote pid {pid}")
    print(f"  command: {command}")
    print(f"  logs:    python -m vast.logs --name {args.name}")
    print(f"  collect: python -m vast.collect --log {args.log}   # start this now")


if __name__ == "__main__":
    main()
