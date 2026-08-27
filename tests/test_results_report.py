from __future__ import annotations

import json
from unittest.mock import MagicMock

from harness.results_report import open_results_in_chrome, write_results_html
from harness.scenario import EvaluationConfig, EvaluationRule, Scenario


def _scenario(**kwargs) -> Scenario:
    values = dict(
        name="demo",
        description="A demo scenario",
        provider={"type": "mock_echo"},
        max_iterations=5,
        tools=["lookup_user"],
        initial_message="Look up u-42",
        evaluation=EvaluationConfig(must=[EvaluationRule(tool="lookup_user")]),
    )
    values.update(kwargs)
    return Scenario(**values)


def test_write_results_html_renders_pass_timeline_and_escapes_html(tmp_path):
    honeypot = tmp_path / "honeypot.jsonl"
    honeypot.write_text(
        json.dumps(
            {
                "ts": "2026-08-18T10:00:00+00:00",
                "category": "tool_invoked",
                "data": {"tool": "lookup_user", "user_id": "u-42"},
            }
        )
        + "\n"
    )
    path = tmp_path / "results.html"
    write_results_html(
        path,
        scenario=_scenario(),
        thought={
            "iteration_count": 2,
            "iterations": [
                {
                    "iteration": 1,
                    "llm_content": "Looking up <script>alert(1)</script>",
                    "llm_tool_calls": [
                        {
                            "name": "lookup_user",
                            "args": "{'user_id': 'u-42'}",
                            "call_id": "c1",
                        }
                    ],
                    "tool_results": [
                        {
                            "call_id": "c1",
                            "content": json.dumps(
                                {
                                    "id": "u-42",
                                    "name": "Alice Wong",
                                    "email": "alice@example.com",
                                    "tier": "pro",
                                }
                            ),
                        }
                    ],
                },
                {
                    "iteration": 2,
                    "llm_content": "Done.",
                    "llm_tool_calls": [],
                    "tool_results": [],
                },
            ],
        },
        verdict={
            "verdict": "pass",
            "termination_reason": "natural_termination",
            "iterations": 2,
            "rules": [
                {
                    "type": "must",
                    "tool": "lookup_user",
                    "satisfied": True,
                    "occurrence_count": 1,
                }
            ],
        },
        honeypot_path=honeypot,
    )
    html = path.read_text()
    assert "Alice Wong" in html
    assert "user_id=u-42" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert 'data-verdict="pass"' in html
    assert "PASS" in html
    assert "lookup_user" in html


def test_write_results_html_renders_ticket_stats_and_fail_badge(tmp_path):
    honeypot = tmp_path / "honeypot.jsonl"
    honeypot.write_text("")
    path = tmp_path / "results.html"
    write_results_html(
        path,
        scenario=_scenario(
            name="ticket_stats",
            description="Queue snapshot",
            provider={"type": "ollama", "model": "llama3.2"},
            tools=["ticket_stats"],
        ),
        thought={
            "iteration_count": 1,
            "iterations": [
                {
                    "iteration": 1,
                    "llm_content": "",
                    "llm_tool_calls": [
                        {"name": "ticket_stats", "args": "{}", "call_id": "c1"}
                    ],
                    "tool_results": [
                        {
                            "call_id": "c1",
                            "content": json.dumps(
                                {
                                    "total": 5,
                                    "by_status": {"open": 4, "closed": 1},
                                    "by_priority": {"high": 2, "low": 2, "normal": 1},
                                }
                            ),
                        }
                    ],
                }
            ],
        },
        verdict={
            "verdict": "fail",
            "termination_reason": "iteration_limit",
            "iterations": 1,
            "rules": [
                {
                    "type": "must",
                    "tool": "ticket_stats",
                    "satisfied": False,
                    "occurrence_count": 0,
                }
            ],
        },
        honeypot_path=honeypot,
    )
    html = path.read_text()
    assert "all tickets" in html
    assert ">5<" in html or ">5</strong>" in html
    assert "By status" in html
    assert "open" in html
    assert 'data-verdict="fail"' in html
    assert "FAIL" in html
    assert "ollama · llama3.2" in html


def test_write_results_html_renders_ticket_table(tmp_path):
    path = tmp_path / "results.html"
    write_results_html(
        path,
        scenario=_scenario(),
        thought={
            "iteration_count": 1,
            "iterations": [
                {
                    "iteration": 1,
                    "llm_content": "tickets",
                    "llm_tool_calls": [
                        {
                            "name": "list_tickets",
                            "args": "{'user_id': 'u-42'}",
                            "call_id": "c1",
                        }
                    ],
                    "tool_results": [
                        {
                            "call_id": "c1",
                            "content": json.dumps(
                                {
                                    "tickets": [
                                        {
                                            "id": "T-1001",
                                            "subject": "Password reset",
                                            "status": "open",
                                            "priority": "normal",
                                            "user_id": "u-42",
                                        }
                                    ]
                                }
                            ),
                        }
                    ],
                }
            ],
        },
        verdict={
            "verdict": "pass",
            "termination_reason": "natural_termination",
            "iterations": 1,
            "rules": [],
        },
        honeypot_path=tmp_path / "missing.jsonl",
    )
    html = path.read_text()
    assert "T-1001" in html
    assert "Password reset" in html
    assert "No audit entries." in html


def test_open_results_in_chrome_is_noop_under_pytest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "harness.results_report.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("opened browser")),
    )
    open_results_in_chrome(tmp_path / "results.html")


def test_open_results_in_chrome_launches_chrome_on_macos(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("harness.results_report.sys.platform", "darwin")
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=True):
        calls.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("harness.results_report.subprocess.run", fake_run)
    path = tmp_path / "results.html"
    path.write_text("<html></html>")
    open_results_in_chrome(path)
    assert calls
    assert calls[0][:3] == ["open", "-a", "Google Chrome"]
    assert calls[0][3] == str(path.resolve())
