from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import iter_scenario_files, run_scenario
from harness.scenario import load_scenario


SCENARIO_FILES = iter_scenario_files()


@pytest.mark.parametrize(
    "scenario_path",
    SCENARIO_FILES,
    ids=[p.stem if p.stem != "scenario" else p.parent.name for p in SCENARIO_FILES],
)
def test_scenario_matches_expected_verdict(scenario_path: Path, tmp_path):
    scenario = load_scenario(scenario_path)
    assert scenario.expected_verdict, f"{scenario_path} needs expected_verdict"
    result = run_scenario(scenario_path, tmp_path)
    assert result["verdict"] == scenario.expected_verdict.lower()
    verdict_file = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict_file["verdict"] == result["verdict"]
    assert (tmp_path / "thought_process.json").exists()


def test_happy_path_terminates_naturally_after_three_iterations(tmp_path):
    from tests.conftest import SCENARIOS_DIR

    result = run_scenario(SCENARIOS_DIR / "happy_path.yaml", tmp_path)
    assert result["termination_reason"] == "natural_termination"
    assert result["iterations"] == 3
    thought = json.loads((tmp_path / "thought_process.json").read_text())
    assert thought["iteration_count"] == 3


def test_ticket_stats_invokes_tool_twice_then_stops(tmp_path):
    from tests.conftest import SCENARIOS_DIR

    result = run_scenario(SCENARIOS_DIR / "ticket_stats.yaml", tmp_path)
    assert result["verdict"] == "pass"
    assert result["termination_reason"] == "natural_termination"
    assert result["iterations"] == 3
    by_tool = {r["tool"]: r for r in result["rules"]}
    assert by_tool["ticket_stats"]["occurrence_count"] == 2


def test_iteration_limit_stops_before_second_tool(tmp_path):
    path = (
        Path(__file__).parent
        / "fixtures"
        / "scenarios"
        / "iteration_limit.yaml"
    )
    result = run_scenario(path, tmp_path)
    assert result["termination_reason"] == "iteration_limit"
    assert result["iterations"] == 2
    by_tool = {r["tool"]: r for r in result["rules"]}
    assert by_tool["lookup_user"]["occurrence_count"] == 1
    assert by_tool["list_tickets"]["occurrence_count"] == 0
