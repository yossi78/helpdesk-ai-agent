from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.evaluate import evaluate
from harness.scenario import EvaluationConfig, EvaluationRule, Scenario


def _scenario(*must: str, must_not: tuple[str, ...] = ()) -> Scenario:
    return Scenario(
        name="test",
        description="",
        provider={"type": "mock_echo"},
        max_iterations=10,
        tools=[],
        initial_message="hi",
        evaluation=EvaluationConfig(
            must=[EvaluationRule(tool=t) for t in must],
            must_not=[EvaluationRule(tool=t) for t in must_not],
        ),
    )


def _write_honeypot(path: Path, tools: list[str], **extra) -> Path:
    lines = []
    for tool in tools:
        entry = {"category": "tool_invoked", "data": {"tool": tool, **extra}}
        lines.append(json.dumps(entry))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def test_must_rules_pass_when_tools_were_invoked(tmp_path):
    honeypot = _write_honeypot(tmp_path / "h.jsonl", ["lookup_user", "list_tickets"])
    verdict = evaluate(_scenario("lookup_user", "list_tickets"), honeypot)
    assert verdict.verdict == "pass"
    assert all(r.satisfied for r in verdict.rules)
    assert [r.occurrence_count for r in verdict.rules] == [1, 1]


def test_must_rules_fail_when_a_tool_is_missing(tmp_path):
    honeypot = _write_honeypot(tmp_path / "h.jsonl", ["lookup_user"])
    verdict = evaluate(_scenario("lookup_user", "list_tickets"), honeypot)
    assert verdict.verdict == "fail"
    by_tool = {r.tool: r for r in verdict.rules}
    assert by_tool["lookup_user"].satisfied is True
    assert by_tool["list_tickets"].satisfied is False
    assert by_tool["list_tickets"].occurrence_count == 0


def test_must_not_passes_when_forbidden_tool_absent(tmp_path):
    honeypot = _write_honeypot(tmp_path / "h.jsonl", ["lookup_user"])
    verdict = evaluate(_scenario("lookup_user", must_not=("delete_account",)), honeypot)
    assert verdict.verdict == "pass"
    must_not = next(r for r in verdict.rules if r.rule_type == "must_not")
    assert must_not.satisfied is True
    assert must_not.occurrence_count == 0


def test_must_not_fails_when_forbidden_tool_present(tmp_path):
    honeypot = _write_honeypot(tmp_path / "h.jsonl", ["delete_account"])
    verdict = evaluate(_scenario(must_not=("delete_account",)), honeypot)
    assert verdict.verdict == "fail"
    assert verdict.rules[0].satisfied is False
    assert verdict.rules[0].occurrence_count == 1


def test_counts_duplicate_invocations(tmp_path):
    honeypot = _write_honeypot(tmp_path / "h.jsonl", ["lookup_user", "lookup_user"])
    verdict = evaluate(_scenario("lookup_user"), honeypot)
    assert verdict.rules[0].occurrence_count == 2


def test_missing_honeypot_file_is_empty_invocation_list(tmp_path):
    verdict = evaluate(_scenario("lookup_user"), tmp_path / "missing.jsonl")
    assert verdict.verdict == "fail"
    assert verdict.rules[0].occurrence_count == 0


def test_skips_blank_lines_invalid_json_and_non_tool_events(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text(
        "\n".join(
            [
                "",
                "not json",
                json.dumps({"category": "record_created", "data": {"tool": "lookup_user"}}),
                json.dumps({"category": "tool_invoked", "data": {}}),
                json.dumps({"category": "tool_invoked", "data": {"tool": "list_tickets"}}),
            ]
        )
        + "\n"
    )
    verdict = evaluate(_scenario("list_tickets", must_not=("lookup_user",)), path)
    assert verdict.verdict == "pass"


def test_empty_evaluation_passes(tmp_path):
    honeypot = _write_honeypot(tmp_path / "h.jsonl", ["lookup_user"])
    verdict = evaluate(_scenario(), honeypot)
    assert verdict.verdict == "pass"
    assert verdict.rules == []
