"""Small shared helpers. Standard library only, Python 3.9+."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union


class CommandError(RuntimeError):
    pass


def run(
    cmd: Union[str, Sequence[str]],
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    shell: bool = False,
    env: Optional[dict] = None,
    input_bytes: Optional[bytes] = None,
) -> subprocess.CompletedProcess:
    """Run a command, capturing output as bytes. Raises CommandError on failure if check."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        shell=shell,
        env=env,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        shown = cmd if isinstance(cmd, str) else " ".join(cmd)
        raise CommandError(
            "command failed (%d): %s\n%s"
            % (proc.returncode, shown, proc.stderr.decode("utf-8", "replace")[-2000:])
        )
    return proc


def git(args: Sequence[str], cwd: Union[str, Path], check: bool = True) -> str:
    """Run git and return stdout as text."""
    proc = run(["git", *args], cwd=cwd, check=check)
    return proc.stdout.decode("utf-8", "replace")


def git_bytes(args: Sequence[str], cwd: Union[str, Path]) -> bytes:
    return run(["git", *args], cwd=cwd).stdout


def read_json(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Union[str, Path], data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0
