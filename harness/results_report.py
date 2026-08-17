"""Write a self-contained HTML report for a scenario run and open it in Chrome."""
from __future__ import annotations

import ast
import html
import json
import os
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from target.honeypot import read_entries

from .scenario import Scenario

_BAR_COLORS = {
    "open": "#34d399",
    "closed": "#94a3b8",
    "high": "#fb7185",
    "normal": "#60a5fa",
    "low": "#fbbf24",
    "unknown": "#a78bfa",
}
_FALLBACK_COLORS = ("#60a5fa", "#34d399", "#fbbf24", "#fb7185", "#a78bfa", "#22d3ee")


def write_results_html(
    path: Path,
    *,
    scenario: Scenario,
    thought: dict[str, Any],
    verdict: dict[str, Any],
    honeypot_path: Path,
) -> Path:
    passed = str(verdict.get("verdict", "")).lower() == "pass"
    body = _page_body(scenario, thought, verdict, honeypot_path, passed)
    path.write_text(_document(scenario.name, passed, body), encoding="utf-8")
    return path


def open_results_in_chrome(path: Path) -> None:
    """Open the report in Google Chrome. No-op under pytest."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    target = str(path.resolve())
    for cmd in _chrome_commands(target):
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True)
        except FileNotFoundError:
            continue
        if completed.returncode == 0:
            return
    webbrowser.open(path.resolve().as_uri())


def _chrome_commands(target: str) -> list[list[str]]:
    if sys.platform == "darwin":
        return [["open", "-a", "Google Chrome", target], ["open", target]]
    if sys.platform.startswith("win"):
        return [["cmd", "/c", "start", "", "chrome", target]]
    return [
        ["google-chrome", target],
        ["chromium-browser", target],
        ["chromium", target],
        ["xdg-open", target],
    ]


def _page_body(
    scenario: Scenario,
    thought: dict[str, Any],
    verdict: dict[str, Any],
    honeypot_path: Path,
    passed: bool,
) -> str:
    provider = scenario.provider or {}
    provider_label = provider.get("type", "unknown")
    if provider.get("model"):
        provider_label = f"{provider_label} · {provider['model']}"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    badge = "PASS" if passed else "FAIL"
    badge_class = "pass" if passed else "fail"
    iterations = verdict.get("iterations", thought.get("iteration_count", 0))
    reason = str(verdict.get("termination_reason", "—")).replace("_", " ")

    return f"""
<header class="hero">
  <div>
    <p class="kicker">Scenario report</p>
    <h1>{html.escape(scenario.name)}</h1>
    <p class="lead">{html.escape(scenario.description or "No description.")}</p>
  </div>
  <div class="badge {badge_class}">{html.escape(badge)}</div>
</header>

<section class="chips">
  {_chip("Verdict", badge, badge_class)}
  {_chip("Iterations", str(iterations))}
  {_chip("Termination", reason)}
  {_chip("Provider", provider_label)}
  {_chip("Generated", generated)}
</section>

<div class="layout">
  <section class="card">
    <h2>Evaluation</h2>
    {_rules_table(verdict.get("rules") or [])}
    <h2 class="spaced">Prompt</h2>
    <p class="prompt">{html.escape(scenario.initial_message)}</p>
  </section>
  <section class="card">
    <h2>Audit log</h2>
    {_honeypot_list(honeypot_path)}
  </section>
</div>

<section class="card timeline-card">
  <h2>Agent timeline</h2>
  {_timeline(thought.get("iterations") or [])}
</section>
"""


def _chip(label: str, value: str, extra: str = "") -> str:
    cls = f"chip {extra}".strip()
    return (
        f'<div class="{cls}"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></div>"
    )


def _rules_table(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return '<p class="muted">No evaluation rules.</p>'
    rows = []
    for rule in rules:
        ok = bool(rule.get("satisfied"))
        mark = "✓" if ok else "✕"
        cls = "ok" if ok else "bad"
        rows.append(
            "<tr>"
            f'<td class="{cls}">{mark}</td>'
            f"<td>{html.escape(str(rule.get('type', '')))}</td>"
            f"<td><code>{html.escape(str(rule.get('tool', '')))}</code></td>"
            f"<td>{html.escape(str(rule.get('occurrence_count', 0)))} call(s)</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th></th><th>Rule</th><th>Tool</th>"
        "<th>Seen</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _honeypot_list(path: Path) -> str:
    if not path.exists():
        return '<p class="muted">No audit entries.</p>'
    items: list[str] = []
    for entry in read_entries(path):
        data = entry.get("data") or {}
        tool = html.escape(str(data.get("tool", "unknown")))
        extras = {k: v for k, v in data.items() if k != "tool"}
        extra = ", ".join(f"{k}={v}" for k, v in extras.items()) if extras else "—"
        ts = html.escape(str(entry.get("ts", "")))
        items.append(
            f"<li><code>{tool}</code><span>{html.escape(extra)}</span>"
            f'<time>{ts}</time></li>'
        )
    if not items:
        return '<p class="muted">No audit entries.</p>'
    return "<ol class='audit'>" + "".join(items) + "</ol>"


def _timeline(iterations: list[dict[str, Any]]) -> str:
    if not iterations:
        return '<p class="muted">No agent iterations recorded.</p>'
    blocks = []
    for rec in iterations:
        n = rec.get("iteration", "?")
        content = (rec.get("llm_content") or "").strip()
        calls = rec.get("llm_tool_calls") or []
        results_by_id = {
            r.get("call_id"): r.get("content", "") for r in rec.get("tool_results") or []
        }
        thought = (
            f'<p class="thought">{html.escape(content)}</p>'
            if content
            else '<p class="muted">No model text this turn — tool call only.</p>'
        )
        tools_html = []
        for call in calls:
            call_id = call.get("call_id")
            tools_html.append(
                _render_tool_call(
                    name=str(call.get("name", "unknown")),
                    args_raw=str(call.get("args", "{}")),
                    result=str(results_by_id.get(call_id, "")),
                )
            )
        tools = "".join(tools_html) or '<p class="muted">No tools this turn.</p>'
        blocks.append(
            f'<article class="step"><div class="step-index">{html.escape(str(n))}</div>'
            f'<div class="step-body"><h3>Iteration {html.escape(str(n))}</h3>'
            f"{thought}{tools}</div></article>"
        )
    return '<div class="timeline">' + "".join(blocks) + "</div>"


def _parse_args(args_raw: str) -> dict[str, Any]:
    try:
        value = ast.literal_eval(args_raw)
    except (ValueError, SyntaxError):
        return {}
    return value if isinstance(value, dict) else {}


def _args_label(name: str, args_raw: str) -> str:
    args = _parse_args(args_raw)
    if name == "ticket_stats" and not args:
        return "all tickets"
    if "user_id" in args:
        return f"user_id={args['user_id']}"
    if args:
        return ", ".join(f"{k}={v}" for k, v in args.items())
    return "no arguments"


def _render_tool_call(name: str, args_raw: str, result: str) -> str:
    label = _args_label(name, args_raw)
    parsed = _try_json(result)
    inner = _render_payload(name, parsed, result)
    return (
        f'<div class="tool"><div class="tool-head"><code>{html.escape(name)}</code>'
        f"<span>{html.escape(label)}</span></div>{inner}</div>"
    )


def _try_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _render_payload(name: str, parsed: Any, raw: str) -> str:
    if name == "ticket_stats" and isinstance(parsed, dict) and "total" in parsed:
        return _render_stats(parsed)
    if isinstance(parsed, dict) and "tickets" in parsed and isinstance(parsed["tickets"], list):
        return _render_tickets(parsed["tickets"])
    if isinstance(parsed, dict) and {"id", "name"} <= parsed.keys():
        return _render_user(parsed)
    if parsed is not None:
        pretty = json.dumps(parsed, indent=2)
        return f"<pre>{html.escape(pretty)}</pre>"
    if raw:
        return f"<pre>{html.escape(raw)}</pre>"
    return '<p class="muted">No tool result.</p>'


def _render_user(user: dict[str, Any]) -> str:
    bits = []
    for key in ("email", "tier", "id"):
        if key in user:
            bits.append(
                f"<li><span>{html.escape(key)}</span>"
                f"<strong>{html.escape(str(user[key]))}</strong></li>"
            )
    extra = "".join(
        f"<li><span>{html.escape(str(k))}</span><strong>{html.escape(str(v))}</strong></li>"
        for k, v in user.items()
        if k not in {"id", "name", "email", "tier"}
    )
    return (
        '<div class="user-card"><div class="avatar">👤</div><div>'
        f'<p class="user-name">{html.escape(str(user.get("name", "Unknown")))}</p>'
        f'<ul class="kv">{"".join(bits)}{extra}</ul></div></div>'
    )


def _render_tickets(tickets: list[dict[str, Any]]) -> str:
    if not tickets:
        return '<p class="muted">No tickets.</p>'
    columns = ["id", "subject", "status", "priority", "user_id"]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    rows = []
    for ticket in tickets:
        cells = "".join(
            f"<td>{html.escape(str(ticket.get(c, '')))}</td>" for c in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table class='tickets'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _render_stats(payload: dict[str, Any]) -> str:
    total = int(payload.get("total") or 0)
    by_status = payload.get("by_status") or {}
    by_priority = payload.get("by_priority") or {}
    return (
        '<div class="stats">'
        f'<div class="stat-hero"><span>Total</span><strong>{total}</strong></div>'
        '<div class="stat-groups">'
        f"{_bar_group('By status', by_status, total)}"
        f"{_bar_group('By priority', by_priority, total)}"
        "</div></div>"
    )


def _bar_group(title: str, counts: dict[str, Any], total: int) -> str:
    bars = []
    for i, (key, count) in enumerate(counts.items()):
        n = int(count)
        pct = 0 if total <= 0 else round(100 * n / total)
        color = _BAR_COLORS.get(str(key).lower(), _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        bars.append(
            '<div class="bar-row">'
            f"<span>{html.escape(str(key))}</span>"
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct}%;background:{color}"></div>'
            "</div>"
            f"<strong>{n}</strong></div>"
        )
    body = "".join(bars) or '<p class="muted">None</p>'
    return f'<div class="bar-group"><h4>{html.escape(title)}</h4>{body}</div>'


def _document(title: str, passed: bool, body: str) -> str:
    theme = "pass" if passed else "fail"
    return f"""<!DOCTYPE html>
<html lang="en" data-verdict="{theme}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} — scenario results</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Nunito:wght@500;700;800&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --ink: #e8eef8;
      --muted: #93a4bf;
      --paper: rgba(14, 22, 40, 0.78);
      --line: rgba(255,255,255,0.1);
      --mint: #34d399;
      --coral: #fb7185;
      --blue: #60a5fa;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; }}
    body {{
      font-family: Nunito, system-ui, sans-serif;
      color: var(--ink);
      background: #070b16;
      padding: 28px 20px 48px;
    }}
    body::before {{
      content: "";
      position: fixed; inset: 0; pointer-events: none; z-index: -1;
      background:
        radial-gradient(ellipse at 12% -10%, rgba(96,165,250,.38), transparent 42%),
        radial-gradient(ellipse at 90% 0%, rgba(244,114,182,.22), transparent 40%),
        radial-gradient(ellipse at 70% 100%, rgba(52,211,153,.16), transparent 46%),
        #070b16;
    }}
    html[data-verdict="fail"] body::before {{
      background:
        radial-gradient(ellipse at 12% -10%, rgba(251,113,133,.35), transparent 42%),
        radial-gradient(ellipse at 90% 0%, rgba(244,114,182,.2), transparent 40%),
        #070b16;
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    .hero {{
      display: flex; justify-content: space-between; gap: 20px; align-items: flex-start;
      margin-bottom: 22px;
    }}
    .kicker {{
      letter-spacing: .14em; text-transform: uppercase; font-size: .72rem;
      font-weight: 800; color: var(--blue); margin: 0 0 8px;
    }}
    h1, h2, h3, h4 {{ font-family: "Baloo 2", cursive; margin: 0; color: #f8fafc; }}
    h1 {{ font-size: clamp(2rem, 4vw, 3rem); line-height: 1.05; }}
    h2 {{ font-size: 1.2rem; margin-bottom: 12px; }}
    h2.spaced {{ margin-top: 22px; }}
    h3 {{ font-size: 1.05rem; }}
    .lead {{ color: var(--muted); margin: 10px 0 0; font-size: 1.05rem; max-width: 40rem; }}
    .badge {{
      font-family: "Baloo 2", cursive; font-size: 1.4rem; padding: 10px 22px;
      border-radius: 999px; letter-spacing: .08em;
      box-shadow: 0 16px 40px rgba(0,0,0,.28);
    }}
    .badge.pass {{ background: linear-gradient(180deg, #6ee7b7, #34d399); color: #064e3b; }}
    .badge.fail {{ background: linear-gradient(180deg, #fda4af, #fb7185); color: #4c0519; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
    .chip {{
      background: var(--paper); border: 1px solid var(--line); border-radius: 16px;
      padding: 10px 14px; min-width: 120px;
    }}
    .chip span {{ display: block; color: var(--muted); font-size: .72rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }}
    .chip strong {{ font-size: .98rem; }}
    .chip.pass strong {{ color: var(--mint); }}
    .chip.fail strong {{ color: var(--coral); }}
    .layout {{ display: grid; grid-template-columns: 1.15fr .85fr; gap: 16px; }}
    @media (max-width: 860px) {{ .layout, .hero {{ grid-template-columns: 1fr; display: grid; }} }}
    .card {{
      background: var(--paper); border: 1px solid var(--line); border-radius: 22px;
      padding: 20px 22px; backdrop-filter: blur(16px);
      box-shadow: 0 20px 50px rgba(0,0,0,.22);
    }}
    .timeline-card {{ margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
    th {{ text-align: left; color: var(--muted); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; padding: 6px 8px; }}
    td {{ padding: 8px; border-top: 1px solid var(--line); vertical-align: top; }}
    td.ok {{ color: var(--mint); font-weight: 800; }}
    td.bad {{ color: var(--coral); font-weight: 800; }}
    code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .86rem; color: #bfdbfe; }}
    .muted {{ color: var(--muted); }}
    .prompt {{
      margin: 0; background: rgba(96,165,250,.08); border: 1px solid var(--line);
      border-radius: 14px; padding: 12px 14px; color: #dbeafe;
    }}
    .audit {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }}
    .audit li {{
      display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: baseline;
      padding: 8px 0; border-bottom: 1px solid var(--line);
    }}
    .audit time {{ color: var(--muted); font-size: .75rem; }}
    .timeline {{ display: flex; flex-direction: column; gap: 16px; }}
    .step {{ display: grid; grid-template-columns: 42px 1fr; gap: 12px; }}
    .step-index {{
      width: 42px; height: 42px; border-radius: 14px; display: grid; place-items: center;
      font-family: "Baloo 2", cursive; background: rgba(96,165,250,.18); color: #dbeafe;
    }}
    .thought {{ margin: 8px 0 12px; line-height: 1.45; }}
    .tool {{
      border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px; margin-top: 10px;
      background: rgba(255,255,255,.03);
    }}
    .tool-head {{ display: flex; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
    .tool-head span {{ color: var(--muted); font-size: .88rem; }}
    pre {{
      margin: 0; overflow: auto; background: #0b1220; border-radius: 12px;
      padding: 12px; font-size: .82rem; color: #cbd5e1;
      font-family: "IBM Plex Mono", ui-monospace, monospace;
    }}
    .user-card {{ display: flex; gap: 14px; align-items: center; }}
    .avatar {{
      width: 52px; height: 52px; border-radius: 18px; display: grid; place-items: center;
      background: linear-gradient(160deg, #fff, #dbeafe); font-size: 1.4rem;
    }}
    .user-name {{ margin: 0 0 6px; font-family: "Baloo 2", cursive; font-size: 1.2rem; }}
    .kv {{ list-style: none; margin: 0; padding: 0; display: flex; gap: 14px; flex-wrap: wrap; }}
    .kv span {{ display: block; color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; }}
    .stats {{ display: grid; grid-template-columns: 140px 1fr; gap: 16px; align-items: start; }}
    .stat-hero {{
      background: linear-gradient(180deg, rgba(96,165,250,.2), rgba(52,211,153,.12));
      border-radius: 18px; padding: 16px 12px; text-align: center;
    }}
    .stat-hero span {{ display: block; color: var(--muted); font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }}
    .stat-hero strong {{ font-family: "Baloo 2", cursive; font-size: 2.4rem; }}
    .stat-groups {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .bar-group h4 {{ font-size: .95rem; margin-bottom: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: 72px 1fr 28px; gap: 8px; align-items: center; margin: 6px 0; font-size: .88rem; }}
    .bar-track {{ height: 8px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 99px; }}
    @media (max-width: 720px) {{
      .stats, .stat-groups {{ grid-template-columns: 1fr; }}
      .audit li {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {body}
  </div>
</body>
</html>
"""
