"""Find instances this project rented but never recorded, and take them back.

    python -m vast.adopt              # report only
    python -m vast.adopt --yes        # validate and record them

An instance becomes untracked whenever `vast.provision` dies between renting
and saving: a SIGTERM from a `timeout` wrapper, a dropped connection, a
KeyboardInterrupt. Its `except` clause destroys an instance that fails to come
up, and its `finally` records the ones that validated, but a signal arrives
inside neither -- so a machine that was still `loading` keeps billing with
nothing pointing at it. That is the worst state to be in, because the usual
checks all look clean: the pool file is consistent, teardown reports success,
and the only evidence is on the invoice.

Adoption is not simply "add the id to the file". An untracked instance has
never passed the CUDA validation, so it goes through the same allocation test a
freshly rented one does, and is destroyed if it fails. Recovering it is worth
more than renting again: the offer is already paid for and the host is already
past the slowest part of its startup.

Instances are recognised by the label provision.py sets, so another project's
machines on the same account are never touched.
"""
from __future__ import annotations

import argparse
import subprocess
import time

from .common import (PROJECT, instances, save_instances, ssh_target, vast,
                     wait_for_instance)
from .provision import VALIDATE


def untracked() -> list[dict]:
    try:
        known = {p["id"] for p in instances()}
    except RuntimeError:
        known = set()
    mine = []
    for inst in vast("show", "instances"):
        label = inst.get("label") or ""
        if label.startswith(f"{PROJECT}-") and inst["id"] not in known:
            mine.append(inst)
    return mine


def validate(info: dict, tries: int = 40) -> str | None:
    for _ in range(tries):
        try:
            check = subprocess.run([*ssh_target(info), VALIDATE],
                                   text=True, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            time.sleep(15)
            continue
        if check.returncode == 0:
            return check.stdout.strip()
        time.sleep(15)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yes", action="store_true",
                   help="validate and record; without it this only reports")
    p.add_argument("--destroy-failed", action="store_true",
                   help="destroy an adopted instance that cannot validate CUDA")
    a = p.parse_args()

    loose = untracked()
    if not loose:
        print("no untracked instance: the pool file matches the account")
        return

    hourly = sum(float(i.get("dph_total", 0)) for i in loose)
    print(f"{len(loose)} untracked instance(s), billing ${hourly:.3f}/h "
          f"(${hourly*24:.2f}/day):")
    for i in loose:
        print(f"  {i['id']}  {i.get('actual_status')}  {i.get('gpu_name')}  "
              f"${float(i.get('dph_total', 0)):.3f}/h  label {i.get('label')}")
    if not a.yes:
        print("\npass --yes to validate and record them, or destroy them with\n"
              "  vastai destroy instance <id> --yes")
        return

    pool = list(instances())
    for i in loose:
        print(f"\nadopting {i['id']}")
        try:
            info = wait_for_instance(i["id"], timeout=600)
        except TimeoutError as exc:
            print(f"  never came up: {exc}")
            continue
        out = validate(info)
        if out is None:
            print("  FAILED the CUDA validation")
            if a.destroy_failed:
                vast("destroy", "instance", str(i["id"]), "--yes")
                print("  destroyed")
            else:
                print("  left running; destroy it with "
                      f"`vastai destroy instance {i['id']} --yes`")
            continue
        for line in out.splitlines():
            print("   ", line)
        pool.append({"id": i["id"], "slot": len(pool),
                     "gpu": i.get("gpu_name"), "dph": float(i.get("dph_total", 0)),
                     "machine_id": i.get("machine_id"), "calib_tok_s": None})
        save_instances(pool)
        print(f"  recorded as slot {len(pool)-1}")

    print(f"\npool is now {len(pool)} instance(s)")


if __name__ == "__main__":
    main()
