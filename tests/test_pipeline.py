"""End-to-end tests with a stand-in agent. Run: python3 tests/test_pipeline.py"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from selfbench import bench, metrics, report, usage, workspace  # noqa: E402
from selfbench.cli import make_demo_repo  # noqa: E402
from selfbench.mine import is_test_file, mine  # noqa: E402
from selfbench.util import git  # noqa: E402

STUB = str(HERE / "stub_agent.py")
os.chmod(STUB, 0o755)


def cfg(name, model=None, **extra):
    c = {"name": name, "kind": "warp", "bin": STUB}
    if model:
        c["model"] = model
    c.update(extra)
    return c


def demo_task(tmp):
    repo = make_demo_repo(Path(tmp) / "demo")
    tasks = mine(str(repo))
    assert len(tasks) == 1, tasks
    task = tasks[0]
    task["check_cmd"] = "python3 {test_files}"
    return repo, task


def test_classification():
    assert is_test_file("tests/test_x.py")
    assert is_test_file("pkg/foo_test.go")
    assert is_test_file("src/a/b.spec.ts")
    assert is_test_file("src/__tests__/a.js")
    assert not is_test_file("src/contest.py")
    assert not is_test_file("src/latest/main.py")


def test_mine_and_validate():
    with tempfile.TemporaryDirectory() as tmp:
        repo, task = demo_task(tmp)
        assert task["test_files"] == ["tests/test_greeting.py"]
        assert task["source_files"] == ["greeting.py"]
        assert task["prompt"].startswith("Add greet(name)")
        assert task["base_commit"] != task["gold_commit"]
        verdict = workspace.validate(task, Path(tmp) / "v")
        assert verdict["validated"] is True, verdict
        assert verdict["fails_before_as"] == "other", verdict  # python, not cargo

        # A task whose tests already pass at the base commit must be rejected.
        fake = dict(task, check_cmd="true")
        verdict = workspace.validate(fake, Path(tmp) / "v2")
        assert verdict["validated"] is False and "already pass" in verdict["reason"], verdict

        # A task whose tests fail even with the real change must be rejected.
        fake = dict(task, check_cmd="false")
        verdict = workspace.validate(fake, Path(tmp) / "v3")
        assert verdict["validated"] is False and "even with the real change" in verdict["reason"]


def test_runs_and_scoring():
    with tempfile.TemporaryDirectory() as tmp:
        repo, task = demo_task(tmp)
        task["validated"] = True
        head_before = git(["rev-parse", "HEAD"], cwd=repo).strip()
        out = Path(tmp) / "out"
        expected = {"solver": True, "lazy": False, "breaker": False, "cheater": False}
        for model, want in expected.items():
            res = bench.run_one(task, cfg(model, model), 1, out, warp_db=None, keep_workspace=True)
            assert res["status"] == "ok", res
            assert res["passed"] is want, (model, res["check"])
            m = res["metrics"]
            assert m["conversation_id"] == "stub-conv-1" and m["run_id"] == "stub-run-1"
            assert m["unparsed_lines"] == 1, m
            assert m["commands_failed"] == 0
            assert res["usage"]["source"] == "warp_run_get" and res["usage_note"] == "ok"
            assert res["usage"]["total_tokens"] == 154481
            assert res["usage"]["tokens_by_model"][0]["usage_type"] == "byok"
            if model == "lazy":
                assert m["tool_calls"] == 1 and res["patch"]["files_changed"] == []
                assert m["commands_run"] == 1 and m["self_check"]["verdict"] == "none"
            else:
                assert m["tool_calls"] == 3 and m["edits"] == 1 and m["commands_run"] == 2
                assert m["tool_time_s"] >= 0.3
                assert m["self_check"] == {"verdict": "passed", "basis": "exit_code",
                                           "test_commands": 1,
                                           "last_test_command": "python -m pytest -q"}
                assert m["claimed_success"] is True  # final message "Done."
            if model == "cheater":
                assert res["patch"]["touched_test_files"] == ["tests/test_greeting.py"]
            if model == "solver":
                assert res["patch"]["files_changed"] == ["greeting.py"]
                rdir = bench.run_dir(out, task["id"], model, 1)
                assert (rdir / "agent.patch").read_text().count("+def greet") == 1
                # The workspace must not contain the real change anywhere in its history.
                ws = rdir / "ws"
                log = git(["log", "--all", "--oneline"], cwd=ws).strip().splitlines()
                assert len(log) == 1, log
                probe = subprocess.run(
                    ["git", "cat-file", "-e", task["gold_commit"]], cwd=ws,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                assert probe.returncode != 0, "gold commit is reachable from the workspace"

        # The source repository must be untouched by all of the above.
        assert git(["status", "--porcelain"], cwd=repo).strip() == ""
        assert git(["rev-parse", "HEAD"], cwd=repo).strip() == head_before

        # Failure modes.
        res = bench.run_one(task, cfg("sleeper", "sleeper"), 1, out, warp_db=None, agent_timeout=2)
        assert res["status"] == "agent_timeout" and res["passed"] is False, res
        assert res["agent"]["wall_s"] < 10
        res = bench.run_one(task, cfg("crasher", "crasher"), 1, out, warp_db=None)
        assert res["status"] == "ok" and res["agent"]["exit_code"] == 3 and res["passed"] is False
        assert res["metrics"]["commands_failed"] == 1
        res = bench.run_one(task, {"name": "ghost", "bin": "/nonexistent/agent"}, 1, out, warp_db=None)
        assert res["status"] == "agent_not_found" and res["passed"] is None
        # An agent that dies at once with no output is a broken setup, not a failed attempt.
        res = bench.run_one(
            task, {"name": "badflags", "kind": "command", "argv": ["sh", "-c", "echo no >&2; exit 1"]},
            1, out, warp_db=None)
        assert res["status"] == "agent_failed_to_start" and res["passed"] is None, res
        res = bench.run_one(task, cfg("nocredits", "nocredits"), 1, out, warp_db=None)
        assert res["status"] == "agent_failed_to_start" and res["passed"] is None, res
        assert "out of AI credits" in res["agent_error"]
        res = bench.run_one(
            task, {"name": "generic", "kind": "command",
                   "argv": ["sh", "-c", "echo working on {task_id} in {cwd}; echo \"$0\"", "{prompt}"]},
            1, out, warp_db=None)
        assert res["status"] == "ok" and res["passed"] is False
        assert res["metrics"]["format"] == "raw" and res["metrics"]["events"] >= 2

        # A "{crate}" placeholder in the suffix is filled per task, and the prompt is kept.
        res = bench.run_one(dict(task, crate="pkg_a"), cfg("solver", "solver"), 1, Path(tmp) / "out-suffix", warp_db=None,
                            prompt_suffix="\n\nRun `cargo test -p {crate}`.")
        assert res["prompt"].endswith("Run `cargo test -p pkg_a`."), res["prompt"]
        # Workspaces are removed unless asked to keep them.
        assert not (bench.run_dir(out, task["id"], "crasher", 1) / "ws").exists()

        agg = report.write_report(out)
        by = {c["config"]: c for c in agg["configs"]}
        assert by["solver"]["pass_rate"] == 1.0 and by["lazy"]["pass_rate"] == 0.0
        assert by["sleeper"]["timeouts"] == 1 and by["sleeper"]["passed"] == 0
        assert by["ghost"]["scored_runs"] == 0 and by["ghost"]["pass_rate"] is None
        assert by["ghost"]["errors"] == 1
        assert by["solver"]["median_tokens"] == 154481
        assert by["generic"]["median_tokens"] is None
        page = (out / "report.html").read_text()
        assert "solver" in page and "<svg" in page and "n/a" in page
        # Self-check table: solver was right; breaker and cheater saw green and were wrong.
        gate = agg["self_check"]
        assert gate["passed"] == {"runs": 3, "hidden_pass": 1, "hidden_fail": 2}, gate
        assert gate["none"]["hidden_pass"] == 0 and gate["none"]["runs"] >= 2, gate
        assert "Agent's own check against the hidden tests" in page
        # Claim table: breaker and cheater said "Done." and the hidden tests failed.
        claimed = agg["claims"]["all"]["claimed"]
        assert claimed["hidden_fail"] >= 2 and agg["claims"]["lazy"]["claimed"]["runs"] == 0, agg["claims"]
        (out / "verified_cases.json").write_text('[{"task_id": "t", "config": "c", "text": "x <b>"}]')
        (out / "limits.json").write_text('["one trial each"]')
        page2 = report.write_report(out) and (out / "report.html").read_text()
        assert "x &lt;b&gt;" in page2 and "one trial each" in page2
        assert metrics.self_check([{"command": "pytest -q", "exit_code": 1}])["verdict"] == "failed"
        assert metrics.self_check([{"command": "ls", "exit_code": 0}])["verdict"] == "none"
        assert metrics.self_check([{"command": "pytest", "exit_code": 1},
                                   {"command": "python -m pytest tests", "exit_code": 0}])["verdict"] == "passed"
        # Older results without a stored self-check get it re-derived from events.
        rj = bench.run_dir(out, task["id"], "breaker", 1) / "result.json"
        import json as _json
        stored = _json.loads(rj.read_text()); del stored["metrics"]["self_check"]
        rj.write_text(_json.dumps(stored))
        assert report.aggregate(report.collect(out))["self_check"] == gate
        # A record with no token data must yield None, never zeros.
        assert usage.from_run_record({"state": "FAILED"}) is None
        assert usage.from_run_record({"request_usage": {"model_token_usage": []}}) is None


def test_plan_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        repo, task = demo_task(tmp)
        configs = [cfg("solver", "solver"), cfg("lazy", "lazy")]
        assert bench.plan([task], configs, 1) == []  # not validated yet
        task["validated"] = True
        jobs = bench.plan([task], configs, 2)
        assert len(jobs) == 4
        out = Path(tmp) / "out"
        lines = []
        bench.run_all(jobs, out, log=lines.append, warp_db=None)
        assert sum("running" in ln for ln in lines) == 4
        lines.clear()
        bench.run_all(jobs, out, log=lines.append, warp_db=None)
        assert sum("already done" in ln for ln in lines) == 4
        agg = report.aggregate(report.collect(out))
        assert agg["total_runs"] == 4
        cell = agg["matrix"][0]["cells"]
        assert cell["solver"] == dict(cell["solver"], scored=2, passed=2)
        assert cell["lazy"]["passed"] == 0


def test_relative_out_path():
    """A relative output directory must not leak into the source repository."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, task = demo_task(tmp)
        task["validated"] = True
        here = os.getcwd()
        os.chdir(tmp)
        try:
            res = bench.run_one(task, cfg("solver", "solver"), 1, Path("rel-out"), warp_db=None)
        finally:
            os.chdir(here)
        assert res["status"] == "ok" and res["passed"] is True, res
        assert (Path(tmp) / "rel-out" / "runs").is_dir()
        assert not (repo / "rel-out").exists()
        assert git(["status", "--porcelain"], cwd=repo).strip() == ""


def test_self_check_reads_output():
    """A piped test command's exit code belongs to the last program, not the tests."""
    sc = metrics.self_check
    cargo_fail = ("running 3 tests\ntest a ... ok\ntest b ... FAILED\n\n"
                  "test result: FAILED. 2 passed; 1 failed; 0 ignored\n"
                  "error: test failed, to rerun pass `--lib`")
    cargo_ok = "test result: ok. 154 passed; 0 failed\ntest result: ok. 0 passed; 0 failed"
    # `| tail` exits 0 even though the tests failed: the output decides.
    r = sc([{"command": "cargo test -p x 2>&1 | tail -20", "exit_code": 0, "output": cargo_fail}])
    assert r["verdict"] == "failed" and r["basis"] == "cargo_output", r
    r = sc([{"command": "cargo test -p x | tail -5", "exit_code": 0, "output": cargo_ok}])
    assert r["verdict"] == "passed" and r["basis"] == "cargo_output", r
    # A compile error has no `test result:` line at all.
    r = sc([{"command": "cargo test -p x 2>&1 | head", "exit_code": 0,
             "output": "error[E0425]: cannot find value\nerror: could not compile `x`"}])
    assert r["verdict"] == "failed", r
    # Piped and nothing recognisable in the output: unclear, never "passed".
    r = sc([{"command": "cargo test -p x 2>&1 | grep -c ok", "exit_code": 0, "output": "12"}])
    assert r["verdict"] == "unclear" and r["basis"] == "none", r
    r = sc([{"command": "cargo build; cargo test", "exit_code": 0, "output": ""}])
    assert r["verdict"] == "unclear", r
    r = sc([{"command": "cargo test || true", "exit_code": 0, "output": ""}])
    assert r["verdict"] == "unclear", r
    # Plain command, no result lines: the exit code is trusted.
    r = sc([{"command": "cd ws && cargo test -p x", "exit_code": 101, "output": ""}])
    assert r["verdict"] == "failed" and r["basis"] == "exit_code", r
    # pytest summaries are read the same way.
    r = sc([{"command": "pytest -q 2>&1 | tail -3", "exit_code": 0,
             "output": "FAILED tests/t.py::x\n1 failed, 254 passed, 1211 subtests passed in 0.37s"}])
    assert r["verdict"] == "failed" and r["basis"] == "pytest_output", r
    r = sc([{"command": "pytest -q | tail -1", "exit_code": 0, "output": "619 passed in 2.44s"}])
    assert r["verdict"] == "passed", r
    # Only the last test command counts.
    r = sc([{"command": "cargo test", "exit_code": 101, "output": cargo_fail},
            {"command": "cargo test 2>&1 | tail", "exit_code": 0, "output": cargo_ok}])
    assert r["verdict"] == "passed" and r["test_commands"] == 2, r
    # Warp moved the command to a subagent: no exit code, no output. Unclear, not "none".
    r = sc([{"command": "cargo test -p x", "exit_code": 0, "output": cargo_ok},
            {"command": "cargo test -p x", "exit_code": None, "status": "running", "output": ""}])
    assert r["verdict"] == "unclear" and r["basis"] == "backgrounded" and r["test_commands"] == 2, r
    assert metrics.claimed_success("HTML comments are now stripped. `cargo test -p md` passes: 146 passed, 0 failed.")
    assert metrics.claimed_success("The task is complete. All 765 tests pass.")
    assert metrics.claimed_success("Fixed the overflow in reversed().")
    assert not metrics.claimed_success("I could not find the cause.")
    assert not metrics.claimed_success(None)


def test_rust_check_cmd_and_failure_kind():
    from selfbench.mine import guess_check_cmd
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        git(["init", "-q"], cwd=repo)
        crate = repo / "crates" / "dir_name" / "src"
        crate.mkdir(parents=True)
        (repo / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/*"]\n')
        (crate.parent / "Cargo.toml").write_text(
            '[package]\nname = "pkg_name"\nversion = "0.1.0"\n\n[dependencies]\nname = "decoy"\n')
        (crate / "foo_tests.rs").write_text("")
        git(["add", "-A"], cwd=repo)
        git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], cwd=repo)
        sha = git(["rev-parse", "HEAD"], cwd=repo).strip()
        got = guess_check_cmd(["crates/dir_name/src/foo_tests.rs"], repo, sha)
        assert got == "cargo test -p pkg_name", got  # package name, not directory name
        assert guess_check_cmd(["src/tests.rs"]) == "cargo test"  # no repo given
    classify = workspace.classify_failure
    assert classify("test x ... FAILED\ntest result: FAILED. 1 passed; 1 failed") == "test_failure"
    assert classify("error[E0599]: no method named `foo`\nerror: could not compile `x`") == "compile_error"
    assert classify("test result: ok. 3 passed\nerror: something else") == "other"
    assert classify("Killed") == "other"


def test_event_pairing_and_string_exit_codes():
    """Real stream shape (2026-10-07): a run_command call that never returns, parallel calls,
    and exit codes sent as strings. Results must pair with the right command."""
    import json as _json
    evs = [
        {"type": "tool_call", "tool": "run_command", "command": "cargo test -p md"},
        {"type": "tool_result", "tool": "run_command", "status": "complete", "exit_code": "0",
         "output": "test result: ok. 141 passed; 0 failed"},
        {"type": "tool_call", "tool": "run_command", "command": "cat >> tests.rs <<EOF"},
        {"type": "tool_call", "tool": "edit_files", "title": "Add tests"},
        {"type": "tool_call", "tool": "run_command", "command": "cargo test -p md 2>&1 | tail"},
        {"type": "tool_result", "tool": "edit_files", "diff": "..."},
        {"type": "tool_result", "tool": "run_command", "status": "complete", "exit_code": "1",
         "output": "test result: FAILED. 145 passed; 1 failed"},
        {"type": "agent", "text": "Done."},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "events.ndjson"
        p.write_text("".join(_json.dumps({"t": i, "line": _json.dumps(e)}) + "\n" for i, e in enumerate(evs)))
        m = metrics.warp_metrics(p)
    # The cat call never returned; the last result belongs to it under FIFO pairing, but the
    # failing cargo output must still count, and as a test command.
    assert m["self_check"]["test_commands"] == 2, m["self_check"]
    assert m["self_check"]["verdict"] == "failed", m["self_check"]
    assert m["commands_failed"] == 1 and m["commands_run"] == 2, m
    assert metrics.self_check([{"command": "cargo test", "exit_code": metrics._int_or_none("0"),
                                "output": ""}])["verdict"] == "passed"


def test_warp_error_runs_are_not_scored():
    """When Warp's run record says ERROR, the run is kept out of the pass rate and rerun on resume."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, task = demo_task(tmp)
        task["validated"] = True
        flag = Path(os.environ.get("TMPDIR", "/tmp")) / "stub-warp-error"
        out = Path(tmp) / "out"
        flag.write_text("1")
        try:
            res = bench.run_one(task, cfg("breaker", "breaker"), 1, out, warp_db=None)
        finally:
            flag.unlink()
        assert res["status"] == "warp_error" and res["passed"] is None, res
        assert res["warp_error"]["error_code"] == "agent_stream_failure"
        agg = report.aggregate(report.collect(out))
        assert agg["configs"][0]["scored_runs"] == 0 and agg["configs"][0]["errors"] == 1
        lines = []
        bench.run_all([(task, cfg("breaker", "breaker"), 1)], out, log=lines.append, warp_db=None)
        assert any("running" in ln for ln in lines), lines  # rerun, not skipped


def test_usage_reads_wal():
    """Uses a real copy of a Warp database if one is provided."""
    db = os.environ.get("SELFBENCH_TEST_DB")
    if not db:
        print("  (skipped: set SELFBENCH_TEST_DB to a copied warp.sqlite to run)")
        return
    wd = os.environ["SELFBENCH_TEST_DB_WORKDIR"]
    got = usage.lookup(db, workspace=wd, retries=1)
    assert got["usage"] is not None, got
    u = got["usage"]
    assert u["matched_by"] == "working_directory"
    assert u["total_tokens"] == int(os.environ["SELFBENCH_TEST_DB_TOKENS"]), u
    assert u["warp_counts"]["files_changed"] == 3, u
    by_token = usage.lookup(db, conversation_id=os.environ["SELFBENCH_TEST_DB_TOKEN"], retries=1)
    assert by_token["usage"]["matched_by"] == "conversation_id"
    miss = usage.lookup(db, workspace="/nowhere", retries=1)
    assert miss["usage"] is None and "no matching" in miss["note"]
    assert usage.lookup("/no/such/db")["usage"] is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print("running %s" % t.__name__)
        t()
    print("\nall %d tests passed" % len(tests))
