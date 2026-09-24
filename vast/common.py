"""Shared vast.ai plumbing: key resolution, instance store, ssh targets.

Kept deliberately close to the skill's reference module -- the invariants it
encodes were learned the expensive way. The project-specific parts live in the
other modules.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "vast" / "runtime"
INSTANCE_FILE = RUNTIME / "instance.json"
POOL_FILE = RUNTIME / "instances.json"

PROJECT = "sca2"
REMOTE = f"/workspace/{PROJECT}"


def api_key() -> str:
    """Resolve the token the CLI itself was configured with.

    The store comes before the environment on purpose: exported copies outlive
    `vastai set api-key`, which writes only the store, and a stale copy opens a
    different account with no credit -- discovered only after provisioning.
    """
    for candidate in (
        _read(Path.home() / ".config" / "vastai" / "vast_api_key"),
        _read(Path.home() / ".vast_api_key"),
        os.environ.get("VAST_API_KEY"),
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    raise RuntimeError(
        "no vast.ai API key: run `vastai set api-key <key>` or export VAST_API_KEY"
    )


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def vast(*args: str) -> object:
    env = os.environ.copy()
    env["VAST_API_KEY"] = api_key()
    result = subprocess.run(
        ["vastai", *args, "--raw"], env=env, check=True,
        text=True, capture_output=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def account() -> dict:
    """Credit and balance are separate fields; a funded account can show 0 balance."""
    return vast("show", "user")


def instances() -> list[dict]:
    if POOL_FILE.exists():
        return json.loads(POOL_FILE.read_text())
    if INSTANCE_FILE.exists():
        return [json.loads(INSTANCE_FILE.read_text())]
    raise RuntimeError("no instance recorded: run `python -m vast.provision` first")


def instance(slot: int = 0) -> dict:
    return instances()[slot]


def save_instances(pool: list[dict]) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    POOL_FILE.write_text(json.dumps(pool, indent=2))
    if pool:
        INSTANCE_FILE.write_text(json.dumps(pool[0], indent=2))


def live(slot: int = 0) -> dict:
    """Current connection details, re-read from the API (ports move on restart)."""
    return vast("show", "instance", str(instance(slot)["id"]))


def ssh_target(info: dict) -> list[str]:
    return [
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
        "-p", str(info["ssh_port"]), f"root@{info['ssh_host']}",
    ]


def run_remote(info: dict, command: str, timeout: int = 120,
               check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([*ssh_target(info), command], text=True,
                          capture_output=True, timeout=timeout, check=check)


def wait_for_instance(instance_id: int, timeout: int = 300) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = vast("show", "instance", str(instance_id))
        if info.get("actual_status") == "running" and info.get("ssh_host"):
            return info
        time.sleep(10)
    raise TimeoutError(f"instance {instance_id} did not come up within {timeout}s")
