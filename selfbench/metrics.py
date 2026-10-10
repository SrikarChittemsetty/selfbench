"""Turn a recorded agent event stream into step-level measurements.

Understands the NDJSON that Warp's `agent run --output-format ndjson` prints
(event shapes from app/src/ai/agent_sdk/driver/output.rs in warpdotdev/warp).
Lines that are not recognised are counted, never guessed at.
"""
from __future__ import annotations

import json
import re
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional

TOOL_END_TYPES = {"tool_result", "tool_error", "tool_canceled"}

# Commands that look like the agent running tests on its own work. A heuristic:
# an agent can also test through a script this does not recognise.
TEST_COMMAND = re.compile(
    r"(\bpytest\b|\bunittest\b|\btox\b|\bnox\b|\bnpm (run )?test\b|\byarn test\b|"
    r"\bpnpm test\b|\bgo test\b|\bcargo test\b|\bmake (test|check)\b|\bjest\b|\bvitest\b|\brspec\b)"
)


# Result lines that cargo and pytest print. Exit codes alone are not trusted:
# `cargo test 2>&1 | tail` exits with tail's code, not cargo's.
CARGO_RESULT = re.compile(r"^test result: (ok|FAILED)\b", re.M)
CARGO_FAILED = re.compile(r"^error: (could not compile|test failed)\b", re.M)
PYTEST_SUMMARY = re.compile(r"\b\d+ (passed|failed)\b.* in [\d.]+s\b")
PYTEST_FAILED = re.compile(r"\b\d+ (failed|errors?)\b")
COMPOUND = re.compile(r"\||;")  # covers |, || and ;

OUTPUT_KEEP = 4000

CLAIM = re.compile(
    r"\b(task is (now )?complete|is complete|completed|done|fixed|implemented|successfully|"
    r"all (\d+ )?tests? pass|tests? (now )?pass|`? pass(es|ed)\b|\d+ passed[,;] 0 failed)",
    re.I,
)


def _command_verdict(cmd: Dict) -> Dict:
    """What one test command showed: (verdict, basis)."""
    out = cmd.get("output") or ""
    if cmd.get("exit_code") is None and cmd.get("status") == "running" and not out.strip():
        # Warp handed a long command to a subagent; its output never reaches the stream.
        return {"verdict": "unclear", "basis": "backgrounded"}
    cargo = CARGO_RESULT.findall(out)
    if CARGO_FAILED.search(out) or "FAILED" in cargo:
        return {"verdict": "failed", "basis": "cargo_output"}
    if cargo:
        return {"verdict": "passed", "basis": "cargo_output"}
    summaries = [ln for ln in out.splitlines() if PYTEST_SUMMARY.search(ln)]
    if summaries:
        failed = PYTEST_FAILED.search(summaries[-1])
        return {"verdict": "failed" if failed else "passed", "basis": "pytest_output"}
    if not COMPOUND.search(cmd.get("command") or "") and cmd.get("exit_code") is not None:
        return {"verdict": "passed" if cmd["exit_code"] == 0 else "failed", "basis": "exit_code"}
    return {"verdict": "unclear", "basis": "none"}


def self_check(commands: List[Dict]) -> Dict:
    """Did the agent test its own change, and what did it see last?

    The verdict is read from the test runner's own result lines when the output
    has them, and from the exit code only for a plain command with no pipe or
    `;` (whose exit code may belong to another program). Otherwise "unclear",
    including commands Warp moved to a background subagent (status "running"),
    whose output is not in the event stream.
    "none": it never ran a recognisable test command.
    """
    runs = [c for c in commands if c.get("command") and TEST_COMMAND.search(c["command"])
            and (c.get("exit_code") is not None or c.get("status") == "running")]
    if not runs:
        return {"verdict": "none", "test_commands": 0}
    last = runs[-1]
    out = dict(_command_verdict(last))
    out.update(test_commands=len(runs), last_test_command=last["command"][:200])
    return out


def claimed_success(final_message: Optional[str]) -> bool:
    """Keyword check: does the agent's last message say the work is done or tests pass?"""
    return bool(final_message and CLAIM.search(final_message))


def _pop_matching(pending: deque, tool: Optional[str], output: Optional[str] = None) -> Optional[Dict]:
    # Output with a test runner's result lines identifies itself: give it to the newest
    # unfinished test command rather than to an older call that may never return.
    if output and (CARGO_RESULT.search(output) or CARGO_FAILED.search(output)
                   or PYTEST_SUMMARY.search(output)):
        for i in range(len(pending) - 1, -1, -1):
            call = pending[i]
            if call["tool"] == tool and call.get("command") and TEST_COMMAND.search(call["command"]):
                del pending[i]
                return call
    for i, call in enumerate(pending):
        if tool is None or call["tool"] == tool:
            del pending[i]
            return call
    return None


def _int_or_none(value) -> Optional[int]:
    """Warp has sent exit codes both as numbers and as strings ("0")."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_events(events_path: Path) -> List[Dict]:
    """Return [{"t": seconds, "event": dict | None, "line": str}] in stream order."""
    out: List[Dict] = []
    if not Path(events_path).exists():
        return out
    with open(events_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            line = rec.get("line", "")
            event = None
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    event = parsed
            except ValueError:
                pass
            out.append({"t": rec.get("t"), "event": event, "line": line})
    return out


def warp_metrics(events_path: Path) -> Dict:
    records = read_events(events_path)
    types: Counter = Counter()
    tools: Counter = Counter()
    pending: deque = deque()
    commands: List[Dict] = []
    tool_time = 0.0
    first_action: Optional[float] = None
    conversation_id = run_id = run_url = None
    last_text: Optional[str] = None
    unparsed = 0
    last_t: Optional[float] = None

    for rec in records:
        t, ev = rec["t"], rec["event"]
        if t is not None:
            last_t = t
        if ev is None or "type" not in ev:
            unparsed += 1
            continue
        typ = ev["type"]
        types[typ] += 1
        if typ == "system":
            kind = ev.get("event_type")
            if kind == "conversation_started":
                conversation_id = ev.get("conversation_id")
            elif kind == "run_started":
                run_id = ev.get("run_id")
                run_url = ev.get("run_url")
        elif typ == "tool_call":
            tool = ev.get("tool", "unknown")
            tools[tool] += 1
            if first_action is None:
                first_action = t
            pending.append({"t": t, "tool": tool, "command": ev.get("command")})
        elif typ in TOOL_END_TYPES:
            # Events carry no ids and calls can run in parallel or never finish, so
            # pair each result with the oldest unfinished call of the same tool.
            call = _pop_matching(pending, ev.get("tool"), ev.get("output"))
            if call and t is not None and call["t"] is not None:
                tool_time += max(0.0, t - call["t"])
            if typ == "tool_result" and ev.get("tool") == "run_command":
                commands.append(
                    {
                        "command": (call or {}).get("command"),
                        "status": ev.get("status"),
                        "exit_code": _int_or_none(ev.get("exit_code")),
                        "output": (ev.get("output") or "")[-OUTPUT_KEEP:],
                    }
                )
        elif typ == "agent":
            last_text = ev.get("text")

    finished_cmds = [c for c in commands if c.get("exit_code") is not None]
    return {
        "format": "warp_ndjson",
        "events": len(records),
        "unparsed_lines": unparsed,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "run_url": run_url,
        "event_types": dict(types),
        "tool_calls": sum(tools.values()),
        "tool_calls_by_tool": dict(tools),
        "tool_errors": types.get("tool_error", 0),
        "tool_canceled": types.get("tool_canceled", 0),
        "commands_run": len(finished_cmds),
        "commands_failed": sum(1 for c in finished_cmds if c["exit_code"] != 0),
        "edits": tools.get("edit_files", 0),
        "tool_time_s": round(tool_time, 3),
        "first_action_s": first_action,
        "last_event_s": last_t,
        "final_message": (last_text or "")[:2000] or None,
        "self_check": self_check(commands),
        "claimed_success": claimed_success(last_text),
    }


def generic_metrics(events_path: Path) -> Dict:
    records = read_events(events_path)
    return {
        "format": "raw",
        "events": len(records),
        "last_event_s": records[-1]["t"] if records else None,
    }
