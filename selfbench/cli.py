"""Command line entry point: python3 -m selfbench <command>."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from . import __version__, bench, report, usage, workspace
from .mine import mine
from .util import run

# Only the Oz CLI has the scripted `agent run` command. The `warp` binary is the
# interactive agent and has no headless mode, so it is deliberately not a candidate.
CANDIDATE_BINS = ["oz", "oz-preview"]


def find_bin(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit if shutil.which(explicit) or os.path.exists(explicit) else None
    for name in CANDIDATE_BINS:
        if shutil.which(name):
            return name
    return None


def _probe(label: str, cmd: List[str], timeout: float = 90) -> bool:
    print("\n$ " + " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("  not found")
        return False
    except subprocess.TimeoutExpired:
        print("  timed out after %ds" % timeout)
        return False
    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace").strip()
    for ln in (out or err or "(no output)").splitlines()[:25]:
        print("  " + ln[:200])
    print("  [%s: exit %d]" % (label, proc.returncode))
    return proc.returncode == 0


def cmd_doctor(args) -> int:
    print("selfbench %s, Python %s" % (__version__, sys.version.split()[0]))
    ok = _probe("git", ["git", "--version"])
    binary = find_bin(args.bin)
    if not binary:
        print("\nNo Oz CLI found on PATH (looked for: %s)." % ", ".join(CANDIDATE_BINS))
        print("In Warp: Command Palette -> 'Install Oz CLI Command', or pass --bin /path/to/oz.")
        print("Note: the `warp` command is the interactive agent and cannot run scripted tasks.")
        return 1
    print("\nUsing agent CLI: %s (%s)" % (binary, shutil.which(binary) or binary))
    ok &= _probe("version", [binary, "--version"])
    ok &= _probe("login", [binary, "whoami"])
    ok &= _probe("models", [binary, "model", "list"])
    _probe("profiles", [binary, "agent", "profile", "list"])
    db = args.warp_db or usage.DEFAULT_DB
    print("\nWarp local database: %s" % db)
    print("  exists: %s, wal file: %s" % (os.path.exists(db), os.path.exists(db + "-wal")))
    print("\n%s" % ("Basics look fine. Next: python3 -m selfbench smoke" if ok
                    else "Something above failed; fix it before running the bench."))
    return 0 if ok else 1


def cmd_mine(args) -> int:
    found = mine(args.repo, limit=args.limit, max_source_lines=args.max_lines)
    meta, tasks = ({}, [])
    if args.append and os.path.exists(args.output):
        meta, tasks = bench.load_tasks(args.output)
    known = {t["id"] for t in tasks}
    added = [t for t in found if t["id"] not in known]
    bench.save_tasks(args.output, meta, tasks + added)
    weak = sum(1 for t in added if t["needs_better_prompt"])
    print("%d candidate tasks from %s (%d added to %s)." % (
        len(found), args.repo, len(added), args.output))
    if weak:
        print("%d have a very short commit message; rewrite their 'prompt' by hand." % weak)
    print("Review each prompt and check_cmd, then run: python3 -m selfbench validate %s" % args.output)
    return 0


def cmd_validate(args) -> int:
    meta, tasks = bench.load_tasks(args.tasks)
    wanted = set(args.only) if args.only else None
    good = 0
    for task in tasks:
        if wanted is not None and task["id"] not in wanted:
            continue
        with tempfile.TemporaryDirectory(prefix="selfbench-validate-") as tmp:
            try:
                verdict = workspace.validate(task, Path(tmp))
            except Exception as exc:  # report and keep going
                verdict = {"validated": False, "reason": "error: %s" % str(exc)[:300]}
        task["validated"] = verdict["validated"]
        task["validation_note"] = verdict["reason"]
        good += 1 if verdict["validated"] else 0
        print("%-40s %s  (%s)" % (
            task["id"], "OK  " if verdict["validated"] else "SKIP", verdict["reason"]))
        bench.save_tasks(args.tasks, meta, tasks)
    print("\n%d task(s) validated. Only validated tasks are run by default." % good)
    return 0


def cmd_run(args) -> int:
    meta, tasks = bench.load_tasks(args.tasks)
    configs = bench.load_configs(args.configs)
    jobs = bench.plan(tasks, configs, args.trials, args.only, args.include_unvalidated)
    if not jobs:
        print("Nothing to run. Are the tasks validated? (python3 -m selfbench validate ...)")
        return 1
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("%d runs planned: %d task(s) x %d config(s) x %d trial(s)." % (
        len(jobs), len({j[0]['id'] for j in jobs}), len(configs), args.trials))
    bench.run_all(
        jobs, out, resume=not args.no_resume,
        prompt_suffix=meta.get("prompt_suffix", bench.DEFAULT_PROMPT_SUFFIX),
        agent_timeout=args.timeout,
        warp_db=None if args.no_usage else (args.warp_db or usage.DEFAULT_DB),
        keep_workspace=args.keep_workspaces,
    )
    agg = report.write_report(out)
    _print_summary(agg)
    print("\nReport: %s" % (out / "report.html"))
    return 0


def _print_summary(agg) -> None:
    print("\n%-28s %9s %8s %12s %12s" % ("config", "pass rate", "passed", "median time", "median tok"))
    for c in agg["configs"]:
        print("%-28s %9s %8s %12s %12s" % (
            c["config"][:28], report._fmt_rate(c["pass_rate"]),
            "%d/%d" % (c["passed"], c["scored_runs"]),
            report._fmt_secs(c["median_wall_s"]), report._fmt_int(c["median_tokens"])))


    gate = agg.get("self_check") or {}
    if any(g["runs"] for g in gate.values()):
        print("\nagent's own check vs hidden tests (runs / hidden pass / hidden fail):")
        for key, label in (("passed", "self-check passed"), ("failed", "self-check failed"),
                           ("unclear", "self-check unclear"), ("none", "no self-check")):
            g = gate.get(key) or {"runs": 0, "hidden_pass": 0, "hidden_fail": 0}
            print("  %-18s %3d / %3d / %3d" % (label, g["runs"], g["hidden_pass"], g["hidden_fail"]))


def cmd_report(args) -> int:
    out = Path(args.out).resolve()
    agg = report.write_report(out, title=args.title, tasks_path=args.tasks)
    _print_summary(agg)
    print("\nReport: %s" % (out / "report.html"))
    return 0


def make_demo_repo(dest: Path) -> Path:
    """A two-commit repo whose second commit adds a function and its test."""
    dest.mkdir(parents=True, exist_ok=True)
    ident = ["-c", "user.name=selfbench", "-c", "user.email=selfbench@localhost",
             "-c", "commit.gpgsign=false"]
    run(["git", "init", "-q"], cwd=dest)
    (dest / "README.md").write_text("# demo\n\nA tiny repo used by the selfbench smoke test.\n")
    (dest / "greeting.py").write_text('"""Greeting helpers."""\n')
    run(["git", "add", "-A"], cwd=dest)
    run(["git", *ident, "commit", "-q", "-m", "Initial commit"], cwd=dest)
    (dest / "greeting.py").write_text(
        '"""Greeting helpers."""\n\n\ndef greet(name):\n    return "hi " + name\n'
    )
    (dest / "tests").mkdir()
    (dest / "tests" / "test_greeting.py").write_text(
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        "from greeting import greet\n\n"
        'assert greet("sam") == "hi sam"\n'
        'print("ok")\n'
    )
    run(["git", "add", "-A"], cwd=dest)
    run(["git", *ident, "commit", "-q", "-m",
         "Add greet(name) to greeting.py\n\nIt returns the string 'hi ' followed by the name."],
        cwd=dest)
    return dest


def cmd_smoke(args) -> int:
    binary = find_bin(args.bin)
    if not binary:
        print("No Warp agent CLI found. Run: python3 -m selfbench doctor")
        return 1
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    demo = out / "demo-repo"
    if demo.exists():
        shutil.rmtree(demo)
    make_demo_repo(demo)
    tasks = mine(str(demo))
    for t in tasks:
        t["check_cmd"] = "python3 {test_files}"
    if len(tasks) != 1:
        print("internal error: expected 1 demo task, got %d" % len(tasks))
        return 1
    with tempfile.TemporaryDirectory(prefix="selfbench-validate-") as tmp:
        verdict = workspace.validate(tasks[0], Path(tmp))
    tasks[0]["validated"] = verdict["validated"]
    print("demo task check: %s" % verdict["reason"])
    if not verdict["validated"]:
        return 1
    config = {"name": "smoke", "kind": "warp", "bin": binary}
    if args.model:
        config["model"] = args.model
    if args.profile:
        config["profile"] = args.profile
    print("Running one tiny task with: %s agent run ... (timeout %ds)" % (binary, args.timeout))
    res = bench.run_one(tasks[0], config, 1, out, agent_timeout=args.timeout,
                        warp_db=args.warp_db or usage.DEFAULT_DB, keep_workspace=True)
    rdir = bench.run_dir(out, tasks[0]["id"], "smoke", 1)
    m = res.get("metrics") or {}
    print("\nstatus:        %s" % res["status"])
    print("hidden tests:  %s" % {True: "PASS", False: "FAIL", None: "n/a"}[res.get("passed")])
    print("agent exit:    %s after %ss" % (
        (res.get("agent") or {}).get("exit_code"), (res.get("agent") or {}).get("wall_s")))
    print("events parsed: %s (%s unrecognised lines), tool calls: %s" % (
        m.get("events"), m.get("unparsed_lines"), m.get("tool_calls")))
    print("files changed: %s" % (res.get("patch") or {}).get("files_changed"))
    print("token usage:   %s" % (
        (res.get("usage") or {}).get("total_tokens") if res.get("usage") else res.get("usage_note")))
    if res["status"] != "ok":
        err = rdir / "agent_stderr.log"
        if err.exists():
            tail = err.read_text(errors="replace").strip().splitlines()[-15:]
            if tail:
                print("\nagent stderr (last lines):")
                for ln in tail:
                    print("  " + ln[:200])
    print("\nEverything from this run is in: %s" % rdir)
    return 0 if res["status"] == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="selfbench", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check that the agent CLI and its login work")
    d.add_argument("--bin")
    d.add_argument("--warp-db")
    d.set_defaults(func=cmd_doctor)

    m = sub.add_parser("mine", help="propose tasks from a repo's git history")
    m.add_argument("repo")
    m.add_argument("-o", "--output", default="tasks.json")
    m.add_argument("--limit", type=int, default=300, help="commits to scan")
    m.add_argument("--max-lines", type=int, default=300, help="largest source change to accept")
    m.add_argument("--append", action="store_true", help="add to an existing tasks file")
    m.set_defaults(func=cmd_mine)

    v = sub.add_parser("validate", help="keep only tasks whose tests fail before and pass after")
    v.add_argument("tasks")
    v.add_argument("--only", nargs="*")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="run tasks x configurations")
    r.add_argument("--tasks", required=True)
    r.add_argument("--configs", required=True)
    r.add_argument("--out", default="bench-out")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--only", nargs="*")
    r.add_argument("--include-unvalidated", action="store_true")
    r.add_argument("--timeout", type=float, help="seconds per agent run (default: task timeout_s)")
    r.add_argument("--warp-db")
    r.add_argument("--no-usage", action="store_true", help="do not read Warp's local database")
    r.add_argument("--keep-workspaces", action="store_true")
    r.add_argument("--no-resume", action="store_true", help="rerun even if a result exists")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="rebuild results.json and report.html")
    rp.add_argument("out")
    rp.add_argument("--title", default="selfbench results")
    rp.add_argument("--tasks", help="tasks file, for a per-task table with PR links")
    rp.set_defaults(func=cmd_report)

    s = sub.add_parser("smoke", help="one tiny end-to-end run against the real agent")
    s.add_argument("--bin")
    s.add_argument("--model")
    s.add_argument("--profile")
    s.add_argument("--out", default="selfbench-smoke")
    s.add_argument("--timeout", type=float, default=600)
    s.add_argument("--warp-db")
    s.set_defaults(func=cmd_smoke)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
