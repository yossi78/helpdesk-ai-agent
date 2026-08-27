from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from run import main
from tests.conftest import SCENARIOS_DIR, free_port
from target.app import start_target as real_start_target


def test_main_happy_path_writes_outputs_and_exits_zero(tmp_path, monkeypatch, capsys):
    port = free_port()

    def _start_target(**kwargs):
        kwargs["port"] = port
        return real_start_target(**kwargs)

    monkeypatch.setattr("run.start_target", _start_target)
    code = main([str(SCENARIOS_DIR / "happy_path.yaml"), "--output-dir", str(tmp_path)])
    assert code == 0
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "pass"
    assert verdict["termination_reason"] == "natural_termination"
    assert (tmp_path / "thought_process.json").exists()
    assert (tmp_path / "honeypot.jsonl").exists()
    report = (tmp_path / "results.html").read_text()
    assert "happy_path" in report
    assert "Alice Wong" in report
    assert "PASS" in report
    captured = capsys.readouterr()
    assert "Ticket stats:" not in captured.out
    assert "Results report:" in captured.out


def test_main_returns_one_when_expected_verdict_mismatches(tmp_path, monkeypatch):
    port = free_port()

    def _start_target(**kwargs):
        kwargs["port"] = port
        return real_start_target(**kwargs)

    monkeypatch.setattr("run.start_target", _start_target)
    scenario = (
        Path(__file__).parent / "fixtures" / "run_only" / "expected_mismatch.yaml"
    )
    code = main([str(scenario), "--output-dir", str(tmp_path)])
    assert code == 1
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "fail"
    report = (tmp_path / "results.html").read_text()
    assert "FAIL" in report
    assert 'data-verdict="fail"' in report


def test_main_clears_stale_output_files(tmp_path, monkeypatch, capsys):
    port = free_port()

    def _start_target(**kwargs):
        kwargs["port"] = port
        return real_start_target(**kwargs)

    monkeypatch.setattr("run.start_target", _start_target)
    stale = tmp_path / "thought_process.json"
    stale.write_text("stale")
    (tmp_path / "results.html").write_text("stale-report")
    code = main([str(SCENARIOS_DIR / "happy_path.yaml"), "--output-dir", str(tmp_path)])
    assert code == 0
    thought = json.loads((tmp_path / "thought_process.json").read_text())
    assert thought["iteration_count"] == 3
    assert "stale-report" not in (tmp_path / "results.html").read_text()
    captured = capsys.readouterr()
    assert f"Removed previous report: {tmp_path / 'results.html'}" in captured.out


def test_main_ticket_stats_ollama_batched_tools(tmp_path, monkeypatch, capsys):
    port = free_port()

    def _start_target(**kwargs):
        kwargs["port"] = port
        return real_start_target(**kwargs)

    monkeypatch.setattr("run.start_target", _start_target)

    first = MagicMock()
    first.raise_for_status.return_value = None
    first.json.return_value = {
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "ticket_stats", "arguments": {}}},
                {
                    "function": {
                        "name": "ticket_stats",
                        "arguments": {"user_id": "u-42"},
                    }
                },
            ],
        }
    }
    second = MagicMock()
    second.raise_for_status.return_value = None
    second.json.return_value = {
        "message": {"content": "Queue has 5; Alice has 2.", "tool_calls": []}
    }

    with patch("harness.providers.ollama.requests.post") as post:
        post.side_effect = [first, second]
        code = main(
            [str(SCENARIOS_DIR / "ticket_stats.yaml"), "--output-dir", str(tmp_path)]
        )

    assert code == 0
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "pass"
    assert verdict["termination_reason"] == "natural_termination"
    assert verdict["iterations"] == 2
    assert verdict["rules"][0]["occurrence_count"] == 2
    thought = json.loads((tmp_path / "thought_process.json").read_text())
    assert thought["iteration_count"] == 2
    assert len(thought["iterations"][0]["llm_tool_calls"]) == 2
    captured = capsys.readouterr()
    assert "Ticket stats:" in captured.out
    assert '"total": 5' in captured.out
    assert '"total": 2' in captured.out
    assert "Agent summary:" in captured.out
    assert "Queue has 5; Alice has 2." in captured.out
    report = (tmp_path / "results.html").read_text()
    assert "ticket_stats" in report
    assert "all tickets" in report
    assert "user_id=u-42" in report
    assert "Queue has 5; Alice has 2." in report


def test_main_opens_results_unless_no_open(tmp_path, monkeypatch):
    port = free_port()

    def _start_target(**kwargs):
        kwargs["port"] = port
        return real_start_target(**kwargs)

    monkeypatch.setattr("run.start_target", _start_target)
    with patch("run.open_results_in_chrome") as opener:
        main([str(SCENARIOS_DIR / "happy_path.yaml"), "--output-dir", str(tmp_path)])
        opener.assert_called_once()
        assert opener.call_args.args[0] == tmp_path / "results.html"

        opener.reset_mock()
        main(
            [
                str(SCENARIOS_DIR / "happy_path.yaml"),
                "--output-dir",
                str(tmp_path),
                "--no-open",
            ]
        )
        opener.assert_not_called()
