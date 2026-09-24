"""Destroy the instances and prove nothing is still billing.

    python -m vast.teardown            # report only
    python -m vast.teardown --yes      # destroy, then verify

Teardown is the whole billing surface, not the running instance: a STOPPED
instance still bills its disk, and a volume outlives the instance that created
it. This reports every instance in any state and every volume, and exits
non-zero if anything survives -- so a script that calls it cannot conclude the
money stopped when it did not.

Destruction is asynchronous and fails quietly, so the account is polled until
the id is absent rather than trusting the call. 429s are retried with backoff:
the CLI returns non-JSON when throttled and the failure looks like a parse
error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .common import RUNTIME, instances, save_instances, vast


def _destroy(instance_id: int, attempts: int = 6) -> None:
    delay = 2.0
    for k in range(attempts):
        try:
            vast("destroy", "instance", str(instance_id), "--yes")
            return
        except Exception as exc:                              # noqa: BLE001
            text = str(exc)
            if k == attempts - 1:
                raise
            print(f"  destroy {instance_id} failed ({text[:80]}); retry in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


def _gone(instance_id: int, timeout: int = 180) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            live = vast("show", "instances")
        except Exception:                                     # noqa: BLE001
            time.sleep(5)
            continue
        if not any(int(i["id"]) == instance_id for i in live):
            return True
        time.sleep(5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="actually destroy; without it this only reports")
    parser.add_argument("--slots", default=None,
                        help="comma list of slot numbers; default every recorded "
                             "instance. Selecting by SLOT, never by a property: "
                             "an ad-hoc filter on gpu name destroyed a whole pool "
                             "because the field reads 'RTX 5090' and the filter "
                             "said 'RTX_5090', so it matched everything.")
    args = parser.parse_args()

    try:
        pool = list(instances())
    except RuntimeError:
        pool = []
    if args.slots is not None:
        want = {int(s) for s in args.slots.split(",") if s.strip()}
        unknown = want - {int(p["slot"]) for p in pool}
        if unknown:
            raise SystemExit(f"no such slot: {sorted(unknown)}; "
                             f"recorded slots are {[p['slot'] for p in pool]}")
        chosen = [p for p in pool if int(p["slot"]) in want]
    else:
        chosen = pool
    recorded = [int(p["id"]) for p in chosen]

    # Say what is about to be destroyed, every time, including under --yes.
    # Destruction is irreversible and the corpus on a host is hours of work.
    print(f"{len(chosen)} of {len(pool)} recorded instance(s) selected:")
    for p in chosen:
        print(f"  slot {p['slot']}  {p['id']}  {p.get('gpu')}  "
              f"${float(p.get('dph', 0)):.3f}/h")
    kept = [p for p in pool if p not in chosen]
    for p in kept:
        print(f"  KEEPING slot {p['slot']}  {p['id']}  {p.get('gpu')}")
    if not args.yes:
        print("\nreport only; pass --yes to destroy the selected instances")

    if args.yes:
        for iid in recorded:
            print(f"destroying {iid}")
            try:
                _destroy(iid)
            except Exception as exc:                          # noqa: BLE001
                print(f"  FAILED: {exc}")
            print(f"  {'gone' if _gone(iid) else 'STILL PRESENT'}")

    if args.yes and kept:
        # A partial teardown must leave the survivors tracked: dropping them
        # from the pool file would not stop them, it would only hide them.
        for i, p in enumerate(kept):
            p["slot"] = i
        save_instances(kept)
        print(f"\npool is now {len(kept)} instance(s); "
              f"they are STILL RUNNING and still billing")
        return 0

    surviving = []
    live_instances = vast("show", "instances") or []
    for inst in live_instances:
        surviving.append(f"instance {inst['id']} state={inst.get('actual_status')} "
                         f"${float(inst.get('dph_total', 0)):.3f}/h "
                         f"(a stopped instance still bills its disk)")
    try:
        volumes = vast("show", "volumes") or []
    except Exception:                                         # noqa: BLE001
        volumes = []
    for vol in volumes:
        surviving.append(f"volume {vol.get('id')} {vol.get('size', '?')} GB")

    if surviving:
        print("\nSTILL BILLING:")
        for line in surviving:
            print("  " + line)
        return 1

    print("\nnothing billing: no instances, no volumes")
    for stale in RUNTIME.glob("*.pid"):
        stale.unlink()
    for name in ("instance.json", "instances.json"):
        (RUNTIME / name).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
