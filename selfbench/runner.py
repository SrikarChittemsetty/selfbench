"""Launch a coding agent on a workspace and record its event stream with timestamps.

Two kinds of agent configuration:

  kind "warp"     Warp's local agent CLI:
                    <bin> agent run --prompt P --cwd DIR --output-format ndjson
                    [--model M] [--profile ID] [--harness H]
                  The flags and the NDJSON event shapes come from Warp's
                  open-source client (crates/warp_cli/src/agent.rs and
                  app/src/ai/agent_sdk/driver/output.rs).

  kind "command"  Any other headless agent, given as an argv template with
                  {prompt}, {cwd} and {task_id} placeholders, e.g.
                    ["claude", "-p", "{prompt}"]
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List


def build_command(config: Dict, prompt: str, ws: Path, task_id: str) -> List[str]:
    kind = config.get("kind", "warp")
    if kind == "warp":
        cmd = [
            config.get("bin", "oz"), "agent", "run",
            "--prompt", prompt,
            "--cwd", str(ws),
            "--output-format", "ndjson",
        ]
        if config.get("model"):
            cmd += ["--model", config["model"]]
        if config.get("profile"):
            cmd += ["--profile", config["profile"]]
        if config.get("harness"):
            cmd += ["--harness", config["harness"]]
        cmd += list(config.get("extra_args", []))
        return cmd
    if kind == "command":
        argv = config.get("argv")
        if not argv:
            raise ValueError("config %r: kind 'command' needs an argv list" % config.get("name"))
        out = []
        for arg in argv:
            out.append(
                arg.replace("{prompt}", prompt)
                .replace("{cwd}", str(ws))
                .replace("{task_id}", task_id)
            )
        return out
    raise ValueError("config %r: unknown kind %r" % (config.get("name"), kind))


def run_agent(
    config: Dict,
    prompt: str,
    ws: Path,
    task_id: str,
    events_path: Path,
    stderr_path: Path,
    timeout: float,
) -> Dict:
    """Run the agent to completion (or timeout). Every stdout line is saved with
    the number of seconds since launch, so step timings can be rebuilt later."""
    cmd = build_command(config, prompt, ws, task_id)
    env = dict(os.environ)
    env["SELFBENCH_TASK_ID"] = task_id
    for key, value in (config.get("env") or {}).items():
        # "$PATH" inside a value expands to the inherited PATH, so a config can
        # prepend a directory: {"PATH": "/some/venv/bin:$PATH"}.
        env[str(key)] = str(value).replace("$PATH", os.environ.get("PATH", ""))

    start_wall = time.time()
    start = time.monotonic()
    n_lines = 0
    if not os.path.dirname(cmd[0]) and shutil.which(cmd[0], path=env.get("PATH")) is None:
        return {
            "launched": False, "error": "agent command not found on PATH: %s" % cmd[0],
            "exit_code": None, "timed_out": False, "wall_s": 0.0, "stdout_lines": 0,
            "started_at": start_wall,
        }
    try:
        with open(stderr_path, "wb") as err:
            proc = subprocess.Popen(
                cmd, cwd=str(ws), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=err, start_new_session=True,
            )
    except FileNotFoundError as exc:
        return {
            "launched": False, "error": "could not launch %s: %s" % (cmd[0], exc),
            "exit_code": None, "timed_out": False, "wall_s": 0.0, "stdout_lines": 0,
            "started_at": start_wall,
        }

    def pump() -> None:
        nonlocal n_lines
        with open(events_path, "w", encoding="utf-8") as out:
            for raw in iter(proc.stdout.readline, b""):
                t = round(time.monotonic() - start, 3)
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                out.write(json.dumps({"t": t, "line": line}) + "\n")
                out.flush()
                n_lines += 1

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = False
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        code = None
    wall = round(time.monotonic() - start, 3)
    reader.join(timeout=10)
    return {
        "launched": True, "exit_code": code, "timed_out": timed_out,
        "wall_s": wall, "stdout_lines": n_lines, "started_at": start_wall,
    }
