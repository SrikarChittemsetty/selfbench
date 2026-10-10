#!/usr/bin/env python3
"""Stand-in for Warp's agent CLI, used only by the test suite.

Accepts the same arguments selfbench passes to the real CLI:
    stub_agent.py agent run --prompt P --cwd DIR --output-format ndjson [--model M]
and prints events in the NDJSON shapes defined by Warp's client
(app/src/ai/agent_sdk/driver/output.rs). The "model" picks a behaviour.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# Trimmed copy of a real `oz run get <id> --output-format json` record (2026-10-06).
RUN_RECORD = {
    "agent_config": {"model_id": "claude-5-5-sonnet-medium"},
    "conversation_id": "stub-conv-1",
    "execution_location": "LOCAL",
    "request_usage": {
        "inference_cost": 0,
        "inference_cost_usd": 0,
        "model_token_usage": [
            {
                "model_id": "claude-5-5-sonnet-medium",
                "tokens_by_category": {"primary_agent": 154481},
                "total_tokens": 154481,
                "usage_type": "byok",
            }
        ],
    },
    "run_id": "stub-run-1",
    "run_time": "PT9S",
    "source": "CLI",
    "state": "SUCCEEDED",
}


def main():
    if sys.argv[1:3] == ["run", "get"]:
        rec = dict(RUN_RECORD)
        if os.path.exists(os.path.join(os.environ.get("TMPDIR", "/tmp"), "stub-warp-error")):
            # Shape of a real record when Warp killed the run (2026-10-07).
            rec["state"] = "ERROR"
            rec["status_message"] = {"error_code": "agent_stream_failure", "retryable": False,
                                     "message": "Response stream finished unexpectedly"}
        print(json.dumps(rec))
        return
    p = argparse.ArgumentParser()
    p.add_argument("agent")
    p.add_argument("run")
    p.add_argument("--prompt")
    p.add_argument("--cwd")
    p.add_argument("--output-format")
    p.add_argument("--model", default="solver")
    p.add_argument("--profile")
    args = p.parse_args()
    os.chdir(args.cwd)
    task_id = os.environ.get("SELFBENCH_TASK_ID", "")
    behaviour = args.model

    if behaviour == "nocredits":
        # What the real CLI did on 2026-10-06 when the account had no credits.
        emit({"type": "system", "event_type": "run_started", "run_id": "stub-run-1",
              "run_url": "https://example.invalid/run/stub-run-1"})
        sys.stderr.write("Error: You're out of AI credits.\n")
        sys.exit(1)

    emit({"type": "system", "event_type": "conversation_started", "conversation_id": "stub-conv-1"})
    emit({"type": "system", "event_type": "run_started", "run_id": "stub-run-1",
          "run_url": "https://example.invalid/run/stub-run-1"})
    sys.stdout.write("this line is not JSON\n")
    sys.stdout.flush()

    emit({"type": "tool_call", "tool": "run_command", "command": "ls"})
    time.sleep(0.2)
    emit({"type": "tool_result", "tool": "run_command", "status": "complete",
          "exit_code": 0, "output": "README.md"})

    if behaviour == "sleeper":
        time.sleep(60)
    elif behaviour == "lazy":
        emit({"type": "agent", "text": "Looks fine to me, nothing to change."})
    elif behaviour == "crasher":
        emit({"type": "tool_call", "tool": "run_command", "command": "false"})
        emit({"type": "tool_result", "tool": "run_command", "status": "complete",
              "exit_code": 1, "output": ""})
        sys.exit(3)
    else:
        emit({"type": "tool_call", "tool": "edit_files", "title": "change", "file_paths": ["x"]})
        time.sleep(0.1)
        gold_dir = os.environ.get("STUB_GOLD_DIR")
        patch = os.path.join(gold_dir, task_id + ".patch") if gold_dir else None
        if behaviour == "solver" and patch and os.path.exists(patch):
            subprocess.run(["git", "apply", "--whitespace=nowarn", patch], check=True)
        elif behaviour == "solver":
            with open("greeting.py", "w") as f:
                f.write('"""Greeting helpers."""\n\n\ndef greet(name):\n    return "hi " + name\n')
        elif behaviour == "breaker":
            with open("greeting.py", "w") as f:
                f.write('def greet(name):\n    return "hello " + name\n')
        elif behaviour == "cheater":
            os.makedirs("tests", exist_ok=True)
            with open("tests/test_greeting.py", "w") as f:
                f.write('print("ok")\n')
        emit({"type": "tool_result", "tool": "edit_files", "diff": "..."})
        # Every one of these behaviours reports a green test run, right or wrong.
        emit({"type": "tool_call", "tool": "run_command", "command": "python -m pytest -q"})
        emit({"type": "tool_result", "tool": "run_command", "status": "complete",
              "exit_code": 0, "output": "1 passed"})
        emit({"type": "agent", "text": "Done."})


if __name__ == "__main__":
    main()
