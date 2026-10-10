"""Isolated, leak-free workspaces for benchmark runs, plus scoring.

The agent works in a fresh directory holding the files of the task's base
commit and nothing else: no git history, so the real change and its tests
cannot be read from the object database. The held-out tests are copied in
only after the agent has finished.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from .util import CommandError, git, git_bytes, run

# Never counted as part of the agent's change.
EXCLUDES = [
    ".venv/", "venv/", "node_modules/", "__pycache__/", ".pytest_cache/",
    ".mypy_cache/", ".ruff_cache/", "target/", "dist/", "build/", "*.egg-info/",
    ".DS_Store",
]

GIT_ID = ["-c", "user.name=selfbench", "-c", "user.email=selfbench@localhost",
          "-c", "commit.gpgsign=false"]


def prepare(task: Dict, dest: Path) -> None:
    """Materialise the task's base commit into `dest` as a one-commit repo."""
    # Absolute paths matter: git runs inside the source repo, so a relative
    # destination would be created there instead of where the caller meant.
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise CommandError("workspace is not empty: %s" % dest)
    dest.mkdir(parents=True, exist_ok=True)
    repo = Path(task["repo"]).resolve()

    # Export via a throwaway index so the source repo's index and HEAD are untouched.
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        run(["git", "read-tree", task["base_commit"]], cwd=repo, env=env)
        run(
            ["git", "checkout-index", "-a", "-f", "--prefix=" + str(dest) + os.sep],
            cwd=repo,
            env=env,
        )

    if not any(dest.iterdir()):
        raise CommandError("workspace came out empty for %s at %s" % (task.get("id"), dest))
    run(["git", "init", "-q"], cwd=dest)
    run(["git", "add", "-A", "-f"], cwd=dest)
    run(["git", *GIT_ID, "commit", "-q", "--allow-empty", "-m", "selfbench baseline"], cwd=dest)
    exclude = dest / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with open(exclude, "a", encoding="utf-8") as f:
        f.write("\n".join(EXCLUDES) + "\n")


def capture_patch(ws: Path, out_patch: Path) -> Dict:
    """Record everything the agent changed relative to the baseline commit."""
    run(["git", "add", "-A"], cwd=ws)
    patch = git_bytes(["diff", "--cached", "--binary", "HEAD"], cwd=ws)
    out_patch.write_bytes(patch)
    numstat = git(["diff", "--cached", "--numstat", "HEAD"], cwd=ws)
    files, added, removed = [], 0, 0
    for ln in numstat.splitlines():
        cols = ln.split("\t")
        if len(cols) != 3:
            continue
        a, d, path = cols
        files.append(path)
        if a != "-":
            added += int(a)
        if d != "-":
            removed += int(d)
    return {"files_changed": files, "lines_added": added, "lines_removed": removed}


def _write_from_commit(repo: Path, commit: str, path: str, ws: Path) -> None:
    data = git_bytes(["show", "%s:%s" % (commit, path)], cwd=repo)
    target = ws / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def overlay_tests(task: Dict, ws: Path) -> None:
    """Replace the test files with the versions from the real change."""
    repo = Path(task["repo"])
    for path in task.get("test_files", []):
        _write_from_commit(repo, task["gold_commit"], path, ws)
    for path in task.get("removed_test_files", []):
        target = ws / path
        if target.exists():
            target.unlink()


def overlay_gold_change(task: Dict, ws: Path) -> None:
    """Apply the real change in full (validation only; never used during a run)."""
    repo = Path(task["repo"])
    status = git(
        ["diff", "--no-renames", "--name-status", task["base_commit"], task["gold_commit"]],
        cwd=repo,
    )
    for ln in status.splitlines():
        cols = ln.split("\t")
        if len(cols) != 2:
            continue
        st, path = cols
        if st == "D":
            target = ws / path
            if target.exists():
                target.unlink()
        else:
            _write_from_commit(repo, task["gold_commit"], path, ws)


def _shell(cmd: str, ws: Path, log: Path, timeout: float) -> Dict:
    start = time.monotonic()
    timed_out = False
    with open(log, "wb") as f:
        proc = subprocess.Popen(
            cmd, cwd=str(ws), shell=True, stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            code: Optional[int] = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            code = None
    return {
        "exit_code": code,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - start, 3),
    }


def _kill_group(proc: subprocess.Popen) -> None:
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_setup(task: Dict, ws: Path, log: Path, timeout: float = 1800) -> Optional[Dict]:
    cmd = task.get("setup_cmd")
    if not cmd:
        return None
    return _shell(cmd, ws, log, timeout)


def check_command(task: Dict) -> str:
    cmd = task.get("check_cmd")
    if not cmd:
        raise CommandError("task %s has no check_cmd" % task.get("id"))
    files = " ".join(shlex.quote(f) for f in task.get("test_files", []))
    return cmd.replace("{test_files}", files)


DEFAULT_CHECK_TIMEOUT_S = 120


def run_check(task: Dict, ws: Path, log: Path, timeout: Optional[float] = None) -> Dict:
    """Run the hidden tests. A hang counts as a failure (some bugs are infinite loops)."""
    limit = timeout or task.get("check_timeout_s") or DEFAULT_CHECK_TIMEOUT_S
    res = _shell(check_command(task), ws, log, limit)
    res["passed"] = res["exit_code"] == 0 and not res["timed_out"]
    return res


def classify_failure(log_text: str) -> str:
    """How the hidden tests failed before the fix: test_failure, compile_error or other."""
    if "test result: FAILED" in log_text:
        return "test_failure"
    if "test result:" not in log_text and re.search(
        r"could not compile|^error(\[E\d+\])?:", log_text, re.M
    ):
        return "compile_error"
    return "other"


def validate(task: Dict, workdir: Path) -> Dict:
    """Confirm the task is real: hidden tests fail before the change and pass after it."""
    workdir = Path(workdir).resolve()
    ws = workdir / "ws"
    logs = workdir
    prepare(task, ws)
    setup = run_setup(task, ws, logs / "setup.log")
    if setup and (setup["exit_code"] != 0 or setup["timed_out"]):
        return {"validated": False, "reason": "setup_cmd failed"}
    overlay_tests(task, ws)
    before = run_check(task, ws, logs / "check_before.log")
    if before["passed"]:
        return {"validated": False, "reason": "hidden tests already pass before the change"}
    log_text = (logs / "check_before.log").read_text(encoding="utf-8", errors="replace")
    kind = classify_failure(log_text)
    overlay_gold_change(task, ws)
    after = run_check(task, ws, logs / "check_after.log")
    if not after["passed"]:
        return {"validated": False, "reason": "hidden tests fail even with the real change",
                "fails_before_as": kind}
    if before["timed_out"]:
        return {"validated": True, "reason": "hangs before, passes after", "fails_before_as": "other"}
    return {"validated": True, "reason": "fails before, passes after", "fails_before_as": kind}
