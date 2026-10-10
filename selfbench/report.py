"""Aggregate result.json files into results.json and a self-contained HTML report."""
from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import warp_metrics
from .util import median, read_json, write_json

COUNTED = ("ok", "agent_timeout")  # runs that produced a verdict


def collect(out_root: Path) -> List[Dict]:
    results = []
    for path in sorted(Path(out_root).glob("runs/*/*/trial-*/result.json")):
        try:
            res = read_json(path)
        except ValueError:
            continue
        # Always re-derive the self-check from the saved event stream, so runs
        # recorded with older logic are judged the same way (no rerun, no tokens).
        m = res.get("metrics") or {}
        events = path.parent / "events.ndjson"
        if m.get("format") == "warp_ndjson" and events.exists():
            fresh = warp_metrics(events)
            m["self_check"] = fresh["self_check"]
            m["claimed_success"] = fresh["claimed_success"]
            res["metrics"] = m
        results.append(res)
    return results


def aggregate(results: List[Dict]) -> Dict:
    by_config: Dict[str, List[Dict]] = defaultdict(list)
    cells: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_config[r["config"]].append(r)
        cells[r["task_id"]][r["config"]].append(r)

    configs = []
    for name in sorted(by_config):
        runs = by_config[name]
        scored = [r for r in runs if r.get("status") in COUNTED and r.get("passed") is not None]
        passed = sum(1 for r in scored if r["passed"])
        tokens = [
            r["usage"]["total_tokens"] for r in scored
            if r.get("usage") and r["usage"].get("total_tokens") is not None
        ]
        configs.append(
            {
                "config": name,
                "detail": runs[0].get("config_detail", {}),
                "runs": len(runs),
                "scored_runs": len(scored),
                "passed": passed,
                "pass_rate": (passed / len(scored)) if scored else None,
                "median_wall_s": median(r["agent"]["wall_s"] for r in scored if r.get("agent")),
                "median_tool_calls": median(
                    (r.get("metrics") or {}).get("tool_calls") for r in scored
                ),
                "median_tool_time_s": median(
                    (r.get("metrics") or {}).get("tool_time_s") for r in scored
                ),
                "timeouts": sum(1 for r in runs if r.get("status") == "agent_timeout"),
                "errors": sum(1 for r in runs if r.get("status") not in COUNTED),
                "runs_with_token_data": len(tokens),
                "median_tokens": median(tokens),
                "total_tokens": sum(tokens) if tokens else None,
            }
        )

    matrix = []
    for task_id in sorted(cells):
        row = {"task_id": task_id, "cells": {}}
        for name in sorted(by_config):
            runs = cells[task_id].get(name, [])
            scored = [r for r in runs if r.get("status") in COUNTED and r.get("passed") is not None]
            row["cells"][name] = {
                "runs": len(runs),
                "scored": len(scored),
                "passed": sum(1 for r in scored if r["passed"]),
                "median_wall_s": median(r["agent"]["wall_s"] for r in scored if r.get("agent")),
            }
        matrix.append(row)
    gate = {v: {"runs": 0, "hidden_pass": 0, "hidden_fail": 0} for v in ("passed", "failed", "unclear", "none")}
    for r in results:
        if r.get("status") not in COUNTED or r.get("passed") is None:
            continue
        verdict = ((r.get("metrics") or {}).get("self_check") or {}).get("verdict")
        if verdict not in gate:
            continue
        gate[verdict]["runs"] += 1
        gate[verdict]["hidden_pass" if r["passed"] else "hidden_fail"] += 1
    claims: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in results:
        if r.get("status") not in COUNTED or r.get("passed") is None:
            continue
        said = "claimed" if (r.get("metrics") or {}).get("claimed_success") else "no_claim"
        for key in (r["config"], "all"):
            c = claims.setdefault(key, {k: {"runs": 0, "hidden_pass": 0, "hidden_fail": 0}
                                        for k in ("claimed", "no_claim")})
            c[said]["runs"] += 1
            c[said]["hidden_pass" if r["passed"] else "hidden_fail"] += 1
    return {"configs": configs, "matrix": matrix, "total_runs": len(results), "self_check": gate,
            "claims": claims}


def _fmt_secs(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return "%dm %02ds" % (int(v) // 60, int(v) % 60) if v >= 60 else "%.0fs" % v


def _fmt_int(v) -> str:
    return "n/a" if v is None else "{:,}".format(int(round(v)))


def _fmt_rate(v: Optional[float]) -> str:
    return "n/a" if v is None else "%.0f%%" % (v * 100)


def _scatter(configs: List[Dict], x_key: str, x_label: str, fmt) -> str:
    pts = [c for c in configs if c.get(x_key) is not None and c.get("pass_rate") is not None]
    if len(pts) < 1:
        return ""
    w, h, pad_l, pad_r, pad_t, pad_b = 640, 320, 56, 24, 16, 44
    xmax = max(c[x_key] for c in pts) * 1.1 or 1.0
    px = lambda v: pad_l + (w - pad_l - pad_r) * (v / xmax)
    py = lambda v: pad_t + (h - pad_t - pad_b) * (1 - v)
    parts = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Pass rate against %s">' % (
        w, h, html.escape(x_label))]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = py(frac)
        parts.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (
            pad_l, w - pad_r, y, y))
        parts.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%d%%</text>' % (
            pad_l - 8, y + 4, frac * 100))
    for frac in (0, 0.5, 1.0):
        v = xmax * frac
        parts.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (
            px(v), h - pad_b + 18, html.escape(fmt(v))))
    parts.append('<text class="axis" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (
        (pad_l + w - pad_r) / 2, h - 6, html.escape(x_label)))
    for c in pts:
        x, y = px(c[x_key]), py(c["pass_rate"])
        parts.append('<circle class="pt" cx="%.1f" cy="%.1f" r="6"/>' % (x, y))
        anchor = "end" if x > w * 0.7 else "start"
        dx = -10 if anchor == "end" else 10
        parts.append('<text class="lbl" x="%.1f" y="%.1f" text-anchor="%s">%s</text>' % (
            x + dx, y - 9, anchor, html.escape(c["config"])))
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1c1d21;--muted:#6b6f7a;--line:#e2e2de;
--pass:#1a7f4b;--passbg:#e3f4ea;--fail:#b3261e;--failbg:#fbe9e7;--mix:#8a6100;--mixbg:#fdf2d0;--accent:#3b5bdb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121316;--card:#1b1d22;
--ink:#ececf0;--muted:#9a9fac;--line:#2c2f37;--pass:#5fd598;--passbg:#153324;--fail:#ff8a80;
--failbg:#3a1a18;--mix:#f2c94c;--mixbg:#3a2f10;--accent:#8ea2ff}}
:root[data-theme="dark"]{--bg:#121316;--card:#1b1d22;--ink:#ececf0;--muted:#9a9fac;--line:#2c2f37;
--pass:#5fd598;--passbg:#153324;--fail:#ff8a80;--failbg:#3a1a18;--mix:#f2c94c;--mixbg:#3a2f10;--accent:#8ea2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:24px 16px}
main{max-width:960px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 12px}
.sub{color:var(--muted);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:18px;margin-bottom:16px}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.num,th.num{text-align:right}.cell{display:inline-block;min-width:54px;text-align:center;
padding:2px 8px;border-radius:6px;font-weight:600;font-size:13px}
.p{color:var(--pass);background:var(--passbg)}.f{color:var(--fail);background:var(--failbg)}
.m{color:var(--mix);background:var(--mixbg)}.e{color:var(--muted)}
svg{width:100%;height:auto}.grid{stroke:var(--line)}.tick,.axis{fill:var(--muted);font-size:11px}
.pt{fill:var(--accent)}.lbl{fill:var(--ink);font-size:12px}.note{color:var(--muted);font-size:13px;margin:10px 0 0}
"""


def _lead_html(agg: Dict) -> str:
    c = (agg.get("claims") or {}).get("all")
    if not c:
        return ""
    said, wrong = c["claimed"]["runs"], c["claimed"]["hidden_fail"]
    scored = said + c["no_claim"]["runs"]
    return ("<section class=card><p style='font-size:18px;margin:0'><b>The agent said the work was done "
            "in %d of %d runs. In %d of those, the tests that shipped with the real fix failed.</b></p>"
            "</section>" % (said, scored, wrong))


def _claims_html(agg: Dict) -> str:
    claims = (agg.get("claims") or {}).get("all")
    if not claims:
        return ""
    rows = []
    for key, label in (("claimed", "Final message says done / tests pass"),
                       ("no_claim", "No such claim")):
        c = claims[key]
        rows.append("<tr><td>%s</td><td class=num>%d</td><td class=num>%d</td><td class=num>%d</td></tr>"
                    % (label, c["runs"], c["hidden_pass"], c["hidden_fail"]))
    per = []
    for name in sorted(k for k in agg["claims"] if k != "all"):
        c = agg["claims"][name]["claimed"]
        per.append("%s: %d of %d claims wrong" % (html.escape(name), c["hidden_fail"], c["runs"]))
    return (
        "<section class=card><h2>What the agent said against the hidden tests</h2><div class=scroll>"
        "<table><thead><tr><th>Agent's final message</th><th class=num>Runs</th>"
        "<th class=num>Hidden tests passed</th><th class=num>Hidden tests failed</th></tr></thead>"
        "<tbody>%s</tbody></table></div><p class=note>%s. The claim is a keyword check on the "
        "agent's final message; every counted wrong claim is checked by hand below.</p></section>"
        % ("".join(rows), "; ".join(per)))


def _tasks_html(agg: Dict, tasks: List[Dict], names: List[str], repo_url: Optional[str]) -> str:
    by_id = {row["task_id"]: row for row in agg["matrix"]}
    rows = []
    for t in tasks:
        row = by_id.get(t["id"])
        if row is None:
            continue
        pr = t.get("pr")
        pr_html = ('<a href="%s/pull/%d">#%d</a>' % (html.escape(repo_url), pr, pr)
                   if pr and repo_url else (str(pr) if pr else ""))
        subject = html.escape(t["prompt"].splitlines()[0][:70])
        cells = []
        for n in names:
            cell = row["cells"].get(n) or {"scored": 0, "passed": 0}
            if not cell["scored"]:
                cells.append('<td><span class="cell e">n/a</span></td>')
            else:
                ok = cell["passed"] == cell["scored"]
                cells.append('<td><span class="cell %s">%s</span></td>' % ("p" if ok else "f",
                                                                         "pass" if ok else "fail"))
        rows.append("<tr><td>%s</td><td>%s</td><td>%s</td><td class=num>%s</td><td>%s</td>%s</tr>" % (
            pr_html, subject, html.escape(t.get("crate", "")), t.get("source_lines_changed", ""),
            "yes" if t.get("agent_co_authored") else "none", "".join(cells)))
    head = "".join("<th>%s</th>" % html.escape(n) for n in names)
    return (
        "<section class=card><h2>Tasks</h2><div class=scroll><table><thead><tr><th>PR</th>"
        "<th>Change</th><th>Crate</th><th class=num>Source lines</th><th>Warp-agent trailer</th>%s"
        "</tr></thead><tbody>%s</tbody></table></div><p class=note>Warp-agent trailer: the merged "
        "commit carries a co-author trailer from Warp's own agent (Oz, Warp, or warp-agent-staging). "
        "'None' does not mean written by hand: outside contributors may have used any tool without "
        "leaving a trace in the commit.</p></section>" % (head, "".join(rows)))


def _verified_html(cases: List[Dict]) -> str:
    if cases is None:
        return ""
    if not cases:
        return ("<section class=card><h2>Verified cases: said done, hidden tests failed</h2>"
                "<p>None. No run survived the hand check.</p></section>")
    def tag(c):
        rec = c.get("own_output_recorded")
        if rec is None:
            return ""
        return (' <span class="cell p">own tests green in log</span>' if rec
                else ' <span class="cell e">own test output not in log</span>')
    items = "".join(
        "<li><b>%s, %s.</b>%s %s</li>" % (html.escape(c["task_id"]), html.escape(c["config"]),
                                          tag(c), html.escape(c["text"])) for c in cases)
    recorded = sum(1 for c in cases if c.get("own_output_recorded"))
    return ("<section class=card><h2>Verified cases: said done, hidden tests failed (%d)</h2>"
            "<ul>%s</ul><p class=note>Each case was checked by hand: the agent's final message and test "
            "commands in events.ndjson, the failing hidden test in check.log, and the tested patch in "
            "agent.patch. In %d of %d cases the agent's own green cargo output is in the log; in the rest "
            "Warp ran the command in a background subagent and only the agent's claim is recorded.</p></section>"
            % (len(cases), items, recorded, len(cases)))


def _limits_html(limits: List[str]) -> str:
    if not limits:
        return ""
    return ("<section class=card><h2>Limits</h2><ul>%s</ul></section>"
            % "".join("<li>%s</li>" % html.escape(x) for x in limits))


def render_html(agg: Dict, title: str = "selfbench results", extras: Optional[Dict] = None) -> str:
    extras = extras or {}
    configs, matrix = agg["configs"], agg["matrix"]
    names = [c["config"] for c in configs]
    rows = []
    for c in configs:
        rows.append(
            "<tr><td>%s</td><td class=num>%s</td><td class=num>%d/%d</td><td class=num>%s</td>"
            "<td class=num>%s</td><td class=num>%s</td><td class=num>%s</td><td class=num>%d</td></tr>"
            % (
                html.escape(c["config"]), _fmt_rate(c["pass_rate"]), c["passed"], c["scored_runs"],
                _fmt_secs(c["median_wall_s"]), _fmt_int(c["median_tool_calls"]),
                _fmt_int(c["median_tokens"]) + (
                    "" if c["runs_with_token_data"] in (0, c["scored_runs"])
                    else " (%d of %d runs)" % (c["runs_with_token_data"], c["scored_runs"])
                ),
                _fmt_int(c["total_tokens"]), c["timeouts"] + c["errors"],
            )
        )
    head = "".join("<th>%s</th>" % html.escape(n) for n in names)
    body = []
    for row in matrix:
        tds = []
        for n in names:
            cell = row["cells"].get(n) or {"scored": 0, "passed": 0, "runs": 0}
            if cell["scored"] == 0:
                tds.append('<td><span class="cell e">%s</span></td>' % (
                    "error" if cell["runs"] else "not run"))
                continue
            cls = "p" if cell["passed"] == cell["scored"] else ("f" if cell["passed"] == 0 else "m")
            label = ("pass" if cls == "p" else "fail") if cell["scored"] == 1 else "%d/%d" % (
                cell["passed"], cell["scored"])
            tds.append('<td><span class="cell %s">%s</span></td>' % (cls, label))
        body.append("<tr><td>%s</td>%s</tr>" % (html.escape(row["task_id"]), "".join(tds)))

    gate = agg.get("self_check") or {}
    gate_rows = []
    labels = {"passed": "Agent ran tests and they passed", "failed": "Agent ran tests and they failed",
              "unclear": "Agent ran tests, result unclear", "none": "Agent ran no tests"}
    for key in ("passed", "failed", "unclear", "none"):
        g = gate.get(key) or {"runs": 0, "hidden_pass": 0, "hidden_fail": 0}
        gate_rows.append("<tr><td>%s</td><td class=num>%d</td><td class=num>%d</td><td class=num>%d</td></tr>"
                         % (labels[key], g["runs"], g["hidden_pass"], g["hidden_fail"]))
    gate_html = (
        "<section class=card><h2>Agent's own check against the hidden tests</h2><div class=scroll>"
        "<table><thead><tr><th>What the agent saw</th><th class=num>Runs</th>"
        "<th class=num>Hidden tests passed</th><th class=num>Hidden tests failed</th></tr></thead>"
        "<tbody>%s</tbody></table></div><p class=note>If a passing self-check were used to skip human "
        "review, the top-right cell is what would have slipped through. Test commands are detected by "
        "pattern, so a run can be miscounted as 'no tests'. The result is read from cargo's or pytest's own "
        "result lines; the exit code is used only for a command with no pipe or ';'. Otherwise it is 'unclear'.</p></section>" % "".join(gate_rows)
    ) if any((gate.get(k) or {}).get("runs") for k in labels) else ""

    time_chart = _scatter(configs, "median_wall_s", "median time per task", _fmt_secs)
    token_chart = _scatter(configs, "median_tokens", "median tokens per task", _fmt_int)
    n_tasks = len(matrix)
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>%s</title><style>%s</style></head><body><main>"
        "<h1>%s</h1><p class=sub>%d tasks, %d configurations, %d runs. "
        "A run passes only if the hidden tests from the real change pass.</p>"
        "%s%s%s"
        "<section class=card><h2>By configuration</h2><div class=scroll><table><thead><tr>"
        "<th>Configuration</th><th class=num>Pass rate</th><th class=num>Passed</th>"
        "<th class=num>Median time</th><th class=num>Median tool calls</th>"
        "<th class=num>Median tokens</th><th class=num>Total tokens</th>"
        "<th class=num>Timeouts / errors</th></tr></thead><tbody>%s</tbody></table></div>"
        "<p class=note>Tokens are Warp's own count for each run (<code>oz run get</code>), shown as n/a "
        "when a run has none. Small samples: treat differences of a few tasks as noise.</p></section>"
        "%s%s%s%s"
        "<section class=card><h2>By task</h2><div class=scroll><table><thead><tr><th>Task</th>%s"
        "</tr></thead><tbody>%s</tbody></table></div></section>"
        "</main></body></html>"
    ) % (
        html.escape(title), CSS, html.escape(title), n_tasks, len(configs), agg["total_runs"],
        _lead_html(agg), _claims_html(agg), _verified_html(extras.get("verified")),
        "".join(rows),
        ("<section class=card><h2>Pass rate against time</h2>%s</section>" % time_chart)
        if time_chart else "",
        ("<section class=card><h2>Pass rate against tokens</h2>%s</section>" % token_chart)
        if token_chart else "",
        gate_html,
        (_tasks_html(agg, extras["tasks"], names, extras.get("repo_url")) if extras.get("tasks") else "")
        + _limits_html(extras.get("limits") or []),
        head, "".join(body),
    )


def write_report(out_root: Path, title: str = "selfbench results",
                 tasks_path: Optional[str] = None) -> Dict:
    """Optional extras, if present next to the results: verified_cases.json (hand-checked
    cases, [{task_id, config, text}]) and limits.json (list of strings)."""
    results = collect(out_root)
    agg = aggregate(results)
    write_json(Path(out_root) / "results.json", agg)
    extras: Dict = {}
    vpath, lpath = Path(out_root) / "verified_cases.json", Path(out_root) / "limits.json"
    if vpath.exists():
        extras["verified"] = read_json(vpath)
    if lpath.exists():
        extras["limits"] = read_json(lpath)
    if tasks_path:
        data = read_json(tasks_path)
        extras["tasks"] = data["tasks"] if isinstance(data, dict) else data
        extras["repo_url"] = data.get("repo_url") if isinstance(data, dict) else None
    (Path(out_root) / "report.html").write_text(render_html(agg, title, extras), encoding="utf-8")
    return agg
