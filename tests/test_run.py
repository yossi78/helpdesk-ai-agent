from __future__ import annotations

import json
from pathlib import Path

from run import main
from tests.conftest import SCENARIOS_DIR, free_port
from target.app import start_target as real_start_target


def test_main_happy_path_writes_outputs_and_exits_zero(tmp_path, monkeypatch):
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


def test_main_clears_stale_output_files(tmp_path, monkeypatch):
    port = free_port()

    def _start_target(**kwargs):
        kwargs["port"] = port
        return real_start_target(**kwargs)

    monkeypatch.setattr("run.start_target", _start_target)
    stale = tmp_path / "thought_process.json"
    stale.write_text("stale")
    main([str(SCENARIOS_DIR / "happy_path.yaml"), "--output-dir", str(tmp_path)])
    thought = json.loads((tmp_path / "thought_process.json").read_text())
    assert thought["iteration_count"] == 3
