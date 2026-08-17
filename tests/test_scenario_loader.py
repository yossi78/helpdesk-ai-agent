from __future__ import annotations

import pytest

from harness.scenario import load_scenario
from tests.conftest import SCENARIOS_DIR


def test_load_happy_path():
    scenario = load_scenario(SCENARIOS_DIR / "happy_path.yaml")
    assert scenario.name == "happy_path"
    assert scenario.tools == ["lookup_user", "list_tickets"]
    assert [r.tool for r in scenario.evaluation.must] == ["lookup_user", "list_tickets"]
    assert scenario.evaluation.must_not == []
    assert scenario.expected_verdict == "pass"
    assert scenario.provider["type"] == "mock_echo"
    assert scenario.max_iterations == 10
    assert scenario.mount_path is None
    assert scenario.target_url == "http://127.0.0.1:5555"
    assert scenario.source_path == (SCENARIOS_DIR / "happy_path.yaml").resolve()


def test_load_ticket_stats_uses_ollama():
    scenario = load_scenario(SCENARIOS_DIR / "ticket_stats.yaml")
    assert scenario.name == "ticket_stats"
    assert scenario.provider["type"] == "ollama"
    assert scenario.provider["model"] == "llama3.2"
    assert scenario.tools == ["ticket_stats"]
    assert [r.tool for r in scenario.evaluation.must] == ["ticket_stats"]
    assert scenario.evaluation.must[0].min_count == 2


def test_defaults_when_optional_fields_omitted(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text(
        "\n".join(
            [
                "name: minimal",
                "provider:",
                "  type: mock_echo",
                "initial_message: hello",
            ]
        )
        + "\n"
    )
    scenario = load_scenario(path)
    assert scenario.description == ""
    assert scenario.max_iterations == 20
    assert scenario.tools == []
    assert scenario.system_prompt == ""
    assert scenario.expected_verdict is None
    assert scenario.evaluation.must == []
    assert scenario.evaluation.must_not == []


def test_empty_evaluation_block(tmp_path):
    path = tmp_path / "empty_eval.yaml"
    path.write_text(
        "\n".join(
            [
                "name: empty_eval",
                "provider: {type: mock_echo}",
                "initial_message: hello",
                "evaluation:",
            ]
        )
        + "\n"
    )
    scenario = load_scenario(path)
    assert scenario.evaluation.must == []
    assert scenario.evaluation.must_not == []


def test_mount_path_must_be_relative(tmp_path):
    path = tmp_path / "abs.yaml"
    path.write_text(
        "\n".join(
            [
                "name: abs",
                "provider: {type: mock_echo}",
                "initial_message: hello",
                "mount_path: /tmp/extra",
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="must be relative"):
        load_scenario(path)


def test_mount_path_home_prefix_rejected(tmp_path):
    path = tmp_path / "home.yaml"
    path.write_text(
        "\n".join(
            [
                "name: home",
                "provider: {type: mock_echo}",
                "initial_message: hello",
                "mount_path: ~/extra",
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="must be relative"):
        load_scenario(path)


def test_mount_path_cannot_escape_scenario_dir(tmp_path):
    path = tmp_path / "escape.yaml"
    path.write_text(
        "\n".join(
            [
                "name: escape",
                "provider: {type: mock_echo}",
                "initial_message: hello",
                "mount_path: ../outside",
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="escapes scenario directory"):
        load_scenario(path)


def test_valid_mount_path_resolves_under_scenario_dir(tmp_path):
    extra = tmp_path / "data"
    extra.mkdir()
    path = tmp_path / "mounted.yaml"
    path.write_text(
        "\n".join(
            [
                "name: mounted",
                "provider: {type: mock_echo}",
                "initial_message: hello",
                "mount_path: data",
            ]
        )
        + "\n"
    )
    scenario = load_scenario(path)
    assert scenario.mount_path == extra.resolve()
