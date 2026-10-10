"""Mine benchmark tasks from a repository's own git history.

A commit becomes a candidate task when it changes both test files and source
files. The task is then: start from the parent commit, give the agent the
commit message as the request, and judge the result with the tests that the
real commit added or changed (which the agent never sees).

This is the same construction SWE-bench uses, applied to your own repos.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .util import git

CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".rb",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".swift", ".php", ".scala", ".sh",
}

TEST_PATTERNS = [
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)__tests__/"),
    re.compile(r"(^|/)spec/"),
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"(^|/)[^/]+_test\.(py|go|rb|rs)$"),
    re.compile(r"(^|/)[^/]+_tests\.rs$"),  # Rust sibling test modules (foo_tests.rs)
    re.compile(r"(^|/)tests\.rs$"),
    re.compile(r"(^|/)[^/]+\.(test|spec)\.(js|jsx|ts|tsx)$"),
    re.compile(r"(^|/)[^/]+Tests?\.(java|kt|cs|swift)$"),
    re.compile(r"(^|/)conftest\.py$"),
]

TRAILER = re.compile(
    r"^(co-authored-by|signed-off-by|reviewed-by|change-id|claude-session|"
    r"generated with|\U0001F916 generated)\b.*$",
    re.IGNORECASE,
)


def is_test_file(path: str) -> bool:
    return any(p.search(path) for p in TEST_PATTERNS)


def is_code_file(path: str) -> bool:
    return Path(path).suffix.lower() in CODE_EXTS


def clean_message(subject: str, body: str) -> str:
    lines = [ln for ln in body.splitlines() if not TRAILER.match(ln.strip())]
    body = "\n".join(lines).strip()
    return (subject.strip() + ("\n\n" + body if body else "")).strip()


CARGO_NAME = re.compile(r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"', re.M | re.S)


def cargo_package(repo: Path, commit: str, path: str) -> Optional[str]:
    """Name of the Cargo package that owns `path` at `commit` (nearest Cargo.toml with [package])."""
    parts = Path(path).parent.parts
    for i in range(len(parts), -1, -1):
        toml = "/".join(parts[:i] + ("Cargo.toml",))
        try:
            text = git(["show", "%s:%s" % (commit, toml)], cwd=repo)
        except Exception:
            continue
        m = CARGO_NAME.search(text)
        if m:
            return m.group(1)
    return None


def guess_check_cmd(
    test_files: List[str], repo: Optional[Path] = None, commit: Optional[str] = None
) -> Optional[str]:
    """Best-effort test command. Always review it; it is a guess."""
    exts = {Path(f).suffix.lower() for f in test_files}
    if ".py" in exts:
        return "python3 -m pytest -q {test_files}"
    if exts & {".js", ".jsx", ".ts", ".tsx"}:
        return "npm test -- {test_files}"
    if ".go" in exts:
        return "go test ./..."
    if ".rs" in exts:
        # The directory name and the package name differ for some crates, so read Cargo.toml.
        pkgs = []
        if repo is not None and commit:
            for f in test_files:
                name = cargo_package(Path(repo), commit, f) if f.endswith(".rs") else None
                if name and name not in pkgs:
                    pkgs.append(name)
        return "cargo test " + " ".join("-p %s" % p for p in pkgs) if pkgs else "cargo test"
    if ".rb" in exts:
        return "bundle exec rspec {test_files}"
    return None


def mine(
    repo: str,
    limit: int = 300,
    max_source_lines: int = 300,
    max_source_files: int = 8,
    min_prompt_chars: int = 25,
) -> List[Dict]:
    """Return candidate tasks from the last `limit` non-merge commits of `repo`."""
    repo_path = Path(repo).resolve()
    name = repo_path.name
    fmt = "%H%x00%P%x00%s%x00%b%x1e"
    raw = git(["log", "--no-merges", "-n", str(limit), "--format=" + fmt], cwd=repo_path)
    tasks: List[Dict] = []
    for record in raw.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split("\x00")
        if len(parts) < 4:
            continue
        sha, parents, subject, body = parts[0], parts[1], parts[2], parts[3]
        parent_list = parents.split()
        if len(parent_list) != 1:
            continue  # root commit or merge
        parent = parent_list[0]

        status = git(
            ["diff", "--no-renames", "--name-status", parent, sha], cwd=repo_path
        )
        numstat = git(["diff", "--no-renames", "--numstat", parent, sha], cwd=repo_path)
        lines_by_file: Dict[str, int] = {}
        for ln in numstat.splitlines():
            cols = ln.split("\t")
            if len(cols) != 3:
                continue
            added, deleted, path = cols
            if added == "-" or deleted == "-":
                lines_by_file[path] = 0  # binary
            else:
                lines_by_file[path] = int(added) + int(deleted)

        test_files: List[str] = []
        removed_test_files: List[str] = []
        source_files: List[str] = []
        for ln in status.splitlines():
            cols = ln.split("\t")
            if len(cols) != 2:
                continue
            st, path = cols
            if is_test_file(path):
                if st == "D":
                    removed_test_files.append(path)
                else:
                    test_files.append(path)
            elif is_code_file(path):
                source_files.append(path)

        if not test_files or not source_files:
            continue
        source_lines = sum(lines_by_file.get(f, 0) for f in source_files)
        if source_lines == 0 or source_lines > max_source_lines:
            continue
        if len(source_files) > max_source_files:
            continue

        message = clean_message(subject, body)
        tasks.append(
            {
                "id": "%s-%s" % (name, sha[:8]),
                "repo": str(repo_path),
                "base_commit": parent,
                "gold_commit": sha,
                "prompt": message,
                "needs_better_prompt": len(message) < min_prompt_chars,
                "test_files": sorted(test_files),
                "removed_test_files": sorted(removed_test_files),
                "source_files": sorted(source_files),
                "source_lines_changed": source_lines,
                "setup_cmd": None,
                "check_cmd": guess_check_cmd(test_files, repo_path, sha),
                "timeout_s": 900,
                "validated": None,
            }
        )
    return tasks
