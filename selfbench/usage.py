"""Read token usage for a run from Warp's local conversation store.

Warp keeps its SQLite database in write-ahead-log mode, so the newest data
lives in `warp.sqlite-wal`, not in `warp.sqlite`. Reading the main file alone
returns stale numbers. This module always copies the database together with
its -wal and -shm files and reads the copy, so Warp's live files are never
opened or modified.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

DEFAULT_DB = os.path.expanduser(
    "~/Library/Group Containers/2BBY89MBSN.dev.warp/"
    "Library/Application Support/dev.warp.Warp-Stable/warp.sqlite"
)


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return a == b


def _summarise(conv_id: str, data: Dict, summary: Dict, matched_by: str) -> Dict:
    usage = data.get("conversation_usage_metadata") or {}
    per_model = []
    total = 0
    for entry in usage.get("token_usage") or []:
        warp = int(entry.get("warp_tokens") or 0)
        byok = int(entry.get("byok_tokens") or 0)
        custom = int(entry.get("custom_endpoint_tokens") or 0)
        total += warp + byok + custom
        per_model.append(
            {
                "model": entry.get("model_id"),
                "warp_tokens": warp,
                "byok_tokens": byok,
                "custom_endpoint_tokens": custom,
            }
        )
    tool_stats = usage.get("tool_usage_metadata") or {}
    diff = tool_stats.get("apply_file_diff_stats") or {}
    cmds = tool_stats.get("run_command_stats") or {}
    return {
        "source": "warp_local_db",
        "matched_by": matched_by,
        "conversation_id": conv_id,
        "title": summary.get("title"),
        "total_tokens": total,
        "tokens_by_model": per_model,
        "credits_spent": usage.get("credits_spent"),
        "provider_cost_cents": usage.get("total_provider_cost_in_cents"),
        "context_window_usage": usage.get("context_window_usage"),
        "warp_counts": {
            "file_diffs": diff.get("count"),
            "files_changed": diff.get("files_changed"),
            "lines_added": diff.get("lines_added"),
            "lines_removed": diff.get("lines_removed"),
            "commands_requested": cmds.get("count"),
            "commands_executed": cmds.get("commands_executed"),
        },
    }


def from_run_record(record: Dict) -> Optional[Dict]:
    """Usage from `oz run get <run_id> --output-format json` (shape observed 2026-10-06).

    Returns None when the record carries no token data, so callers can fall back
    instead of reporting zeros.
    """
    req = record.get("request_usage") or {}
    entries = req.get("model_token_usage") or []
    per_model = []
    total = 0
    for entry in entries:
        tokens = entry.get("total_tokens")
        if tokens is None:
            continue
        total += int(tokens)
        per_model.append(
            {
                "model": entry.get("model_id"),
                "total_tokens": int(tokens),
                "usage_type": entry.get("usage_type"),
                "tokens_by_category": entry.get("tokens_by_category") or {},
            }
        )
    if not per_model:
        return None
    return {
        "source": "warp_run_get",
        "conversation_id": record.get("conversation_id"),
        "total_tokens": total,
        "tokens_by_model": per_model,
        "inference_cost_usd": req.get("inference_cost_usd"),
        "run_state": record.get("state"),
        "run_time": record.get("run_time"),
        "execution_location": record.get("execution_location"),
    }


def _lookup_once(
    db_path: str,
    workspace: Optional[str],
    conversation_id: Optional[str],
    run_id: Optional[str],
) -> Optional[Dict]:
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "warp.sqlite"
        shutil.copyfile(db_path, copy)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                shutil.copyfile(db_path + suffix, str(copy) + suffix)
        conn = sqlite3.connect(str(copy))
        try:
            rows = conn.execute(
                "SELECT conversation_id, conversation_data, summary "
                "FROM agent_conversations ORDER BY last_modified_at DESC"
            ).fetchall()
        finally:
            conn.close()
    for conv_id, raw_data, raw_summary in rows:
        try:
            data = json.loads(raw_data) if raw_data else {}
            summary = json.loads(raw_summary) if raw_summary else {}
        except ValueError:
            continue
        matched = None
        if conversation_id and conversation_id in (
            conv_id, data.get("server_conversation_token")
        ):
            matched = "conversation_id"
        elif run_id and data.get("run_id") == run_id:
            matched = "run_id"
        elif workspace and _same_path(summary.get("initial_working_directory"), workspace):
            matched = "working_directory"
        if matched:
            return _summarise(conv_id, data, summary, matched)
    return None


def lookup(
    db_path: Optional[str],
    workspace: Optional[str] = None,
    conversation_id: Optional[str] = None,
    run_id: Optional[str] = None,
    retries: int = 4,
    delay: float = 1.5,
) -> Dict:
    """Return {"usage": dict | None, "note": str}. Never raises; never invents numbers."""
    if not db_path:
        return {"usage": None, "note": "usage lookup disabled"}
    if not os.path.exists(db_path):
        return {"usage": None, "note": "Warp database not found at %s" % db_path}
    last_error = None
    for attempt in range(retries):
        try:
            found = _lookup_once(db_path, workspace, conversation_id, run_id)
            if found:
                return {"usage": found, "note": "ok"}
        except (sqlite3.Error, OSError) as exc:  # torn copy while Warp is writing
            last_error = exc
        if attempt < retries - 1:
            time.sleep(delay)
    if last_error:
        return {"usage": None, "note": "could not read Warp database: %s" % last_error}
    return {"usage": None, "note": "no matching conversation in Warp's local database"}
