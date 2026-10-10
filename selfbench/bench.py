"""Run tasks x agent configurations and write one result.json per run."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import __version__, metrics, usage, workspace
from .mine import is_test_file
from .runner import run_agent
from .util import read_json, write_json

DEFAULT_PROMPT_SUFFIX = (
    "\n\nWork only inside this repository. Do not add or change test files: "
    "your change will be checked by tests you cannot see. Stop when the change is complete."
)


def load_tasks(path: str) -> Tuple[Dict, List[Dict]]:
    data = read_json(path)
    if isinstance(data, list):
        return {}, data
    return {k: v for k, v in data.items() if k != "tasks"}, data.get("tasks", [])


def save_tasks(path: str, meta: Dict, tasks: List[Dict]) -> None:
    out = dict(meta)
    out["tasks"] = tasks
    write_json(path, out)


def load_configs(path: str) -> List[Dict]:
    data = read_json(path)
    configs = data.get("configs", data) if isinstance(data, dict) else data
    names = [c.get("name") for c in configs]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("every config needs a unique 'name'")
    return configs


def run_dir(out_root: Path, task_id: str, config_name: str, trial: int) -> Path:
    return Path(out_root) / "runs" / task_id / config_name / ("trial-%d" % trial)


def _tail(path: Path, lines: int = 8) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])[:1500]


def _read_record(path: Path) -> Optional[Dict]:
    try:
        return read_json(path)
    except (ValueError, OSError):
        return None


def _fetch_run_record(config: Dict, run_id: str, dest: Path) -> Optional[str]:
    """Best effort: save what Warp's CLI reports for this run. Raw, unparsed."""
    try:
        proc = subprocess.run(
            [config.get("bin", "oz"), "run", "get", run_id, "--output-format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "run get failed: %s" % exc
    if proc.returncode != 0:
        return "run get exited %d" % proc.returncode
    dest.write_bytes(proc.stdout)
    return None


def run_one(
    task: Dict,
    config: Dict,
    trial: int,
    out_root: Path,
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
    agent_timeout: Optional[float] = None,
    warp_db: Optional[str] = usage.DEFAULT_DB,
    keep_workspace: bool = False,
) -> Dict:
    out_root = Path(out_root).resolve()
    rdir = run_dir(out_root, task["id"], config["name"], trial)
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    ws = rdir / "ws"
    result: Dict = {
        "selfbench_version": __version__,
        "task_id": task["id"],
        "config": config["name"],
        "config_detail": {k: v for k, v in config.items() if k != "env"},
        "trial": trial,
        "status": "started",
        "passed": None,
    }

    def finish(status: str) -> Dict:
        result["status"] = status
        write_json(rdir / "result.json", result)
        if not keep_workspace and ws.exists():
            shutil.rmtree(ws, ignore_errors=True)
        return result

    workspace.prepare(task, ws)
    setup = workspace.run_setup(task, ws, rdir / "setup.log")
    if setup is not None:
        result["setup"] = setup
        if setup["exit_code"] != 0 or setup["timed_out"]:
            return finish("setup_failed")

    timeout = agent_timeout or task.get("timeout_s") or 900
    # A suffix may name the task's crate: "{crate}" is filled per task.
    # "crate_args" carries extra cargo flags a crate needs, e.g. "warpui_core --features tui".
    suffix = (prompt_suffix or "").replace("{crate}", str(task.get("crate_args") or task.get("crate", "")))
    prompt = task["prompt"] + suffix
    result["prompt"] = prompt
    agent = run_agent(
        config, prompt, ws, task["id"], rdir / "events.ndjson", rdir / "agent_stderr.log", timeout
    )
    result["agent"] = agent
    if not agent.get("launched"):
        return finish("agent_not_found")
    if agent["exit_code"] not in (0, None) and agent["stdout_lines"] == 0:
        # The agent died before doing anything (bad flags, not logged in, ...).
        # That is a broken setup, not a failed attempt, so it is not scored.
        return finish("agent_failed_to_start")

    kind = config.get("kind", "warp")
    m = metrics.warp_metrics(rdir / "events.ndjson") if kind == "warp" else metrics.generic_metrics(
        rdir / "events.ndjson"
    )
    if agent["wall_s"] and m.get("tool_time_s") is not None:
        m["non_tool_time_s"] = round(max(0.0, agent["wall_s"] - m["tool_time_s"]), 3)
    result["metrics"] = m
    if (
        kind == "warp"
        and agent["exit_code"] not in (0, None)
        and m["tool_calls"] == 0
        and not m["event_types"].get("agent")
    ):
        # Exited with an error before the model did anything (e.g. out of credits).
        result["agent_error"] = _tail(rdir / "agent_stderr.log")
        return finish("agent_failed_to_start")

    patch = workspace.capture_patch(ws, rdir / "agent.patch")
    patch["touched_test_files"] = [f for f in patch["files_changed"] if is_test_file(f)]
    result["patch"] = patch

    if kind == "warp":
        # Preferred source: Warp's own record of the run. Fallback: the app's local database.
        found = None
        if m.get("run_id"):
            for attempt in range(3):
                note = _fetch_run_record(config, m["run_id"], rdir / "warp_run_get.json")
                if note:
                    result["run_record_note"] = note
                    break
                result.pop("run_record_note", None)
                try:
                    found = usage.from_run_record(read_json(rdir / "warp_run_get.json"))
                except (ValueError, OSError):
                    found = None
                if found:
                    break
                time.sleep(2)
        record = _read_record(rdir / "warp_run_get.json")
        if record and str(record.get("state", "")).upper() == "ERROR":
            # Warp itself ended the run with an error (e.g. agent_stream_failure). Not the
            # model's attempt failing: recorded, kept out of the pass rate, rerun on resume.
            result["warp_error"] = record.get("status_message")
        if found:
            result["usage"] = found
            result["usage_note"] = "ok"
        else:
            looked = usage.lookup(
                warp_db, workspace=str(ws),
                conversation_id=m.get("conversation_id"), run_id=m.get("run_id"),
            )
            result["usage"] = looked["usage"]
            result["usage_note"] = looked["note"]
    else:
        result["usage"] = None
        result["usage_note"] = "token usage is only collected for kind 'warp'"

    workspace.overlay_tests(task, ws)
    check = workspace.run_check(task, ws, rdir / "check.log")
    result["check"] = check
    result["passed"] = bool(check["passed"])
    if result.get("warp_error"):
        result["passed_if_counted"] = result["passed"]
        result["passed"] = None
        return finish("warp_error")
    if agent["timed_out"]:
        return finish("agent_timeout")
    return finish("ok")


def plan(
    tasks: List[Dict],
    configs: List[Dict],
    trials: int,
    only: Optional[Iterable[str]] = None,
    include_unvalidated: bool = False,
) -> List[Tuple[Dict, Dict, int]]:
    wanted = set(only) if only else None
    jobs = []
    for task in tasks:
        if wanted is not None and task["id"] not in wanted:
            continue
        if not include_unvalidated and task.get("validated") is not True:
            continue
        for config in configs:
            for trial in range(1, trials + 1):
                jobs.append((task, config, trial))
    return jobs


def run_all(
    jobs: List[Tuple[Dict, Dict, int]],
    out_root: Path,
    resume: bool = True,
    log=print,
    **kwargs,
) -> List[Dict]:
    out_root = Path(out_root).resolve()
    results = []
    for i, (task, config, trial) in enumerate(jobs, 1):
        existing = run_dir(out_root, task["id"], config["name"], trial) / "result.json"
        if resume and existing.exists():
            prior = read_json(existing)
            if prior.get("status") not in ("started", "agent_not_found", "agent_failed_to_start",
                                           "warp_error"):
                log("[%d/%d] %s x %s #%d: already done, skipping" % (
                    i, len(jobs), task["id"], config["name"], trial))
                results.append(prior)
                continue
        log("[%d/%d] %s x %s #%d: running..." % (i, len(jobs), task["id"], config["name"], trial))
        started = time.monotonic()
        res = run_one(task, config, trial, out_root, **kwargs)
        verdict = {True: "PASS", False: "FAIL", None: "n/a"}[res.get("passed")]
        log("        -> %s (%s) in %.0fs" % (verdict, res["status"], time.monotonic() - started))
        results.append(res)
        if res["status"] == "agent_not_found":
            log("        agent command not found; stopping. Fix the config 'bin' and rerun.")
            break
        if res["status"] == "agent_failed_to_start":
            log("        agent exited at once with no output; stopping. See agent_stderr.log in %s"
                % run_dir(out_root, task["id"], config["name"], trial))
            break
    return results
