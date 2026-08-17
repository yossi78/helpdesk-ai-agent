from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest
import requests

from evaluator.evaluate import evaluate
from harness.agent_loop import run_agent_loop
from harness.providers import make_provider
from harness.scenario import load_scenario
from harness.thought_process import ThoughtProcessRecorder
from harness.tools import ToolHandler
from target.app import start_target

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = ROOT / "scenarios"
FIXTURE_SCENARIOS_DIR = Path(__file__).parent / "fixtures" / "scenarios"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthy(base_url: str, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=0.2)
            if resp.ok:
                return
        except requests.RequestException:
            time.sleep(0.05)
    raise RuntimeError(f"target at {base_url} did not become healthy")


def iter_scenario_files() -> list[Path]:
    files = sorted(SCENARIOS_DIR.glob("*.yaml"))
    files.extend(sorted(FIXTURE_SCENARIOS_DIR.rglob("*.yaml")))
    return files


def run_scenario(scenario_path: Path, output_dir: Path, port: int | None = None) -> dict:
    """Run a scenario the same way run.py does, on an ephemeral port."""
    output_dir.mkdir(parents=True, exist_ok=True)
    thought_path = output_dir / "thought_process.json"
    honeypot_path = output_dir / "honeypot.jsonl"
    verdict_path = output_dir / "verdict.json"

    scenario = load_scenario(scenario_path)
    if port is None:
        port = free_port()
    target = start_target(
        mount_path=scenario.mount_path,
        honeypot_path=honeypot_path,
        port=port,
    )
    wait_healthy(target.base_url)
    try:
        provider = make_provider(scenario.provider)
        tool_handler = ToolHandler(scenario.tools, target_url=target.base_url)
        recorder = ThoughtProcessRecorder()
        result = run_agent_loop(
            provider=provider,
            tool_handler=tool_handler,
            recorder=recorder,
            system_prompt=scenario.system_prompt,
            initial_message=scenario.initial_message,
            max_iterations=scenario.max_iterations,
        )
        recorder.export(thought_path)
        verdict = evaluate(scenario, honeypot_path)
        payload = {
            "verdict": verdict.verdict,
            "termination_reason": result.reason,
            "iterations": result.iterations,
            "rules": [
                {
                    "type": r.rule_type,
                    "tool": r.tool,
                    "satisfied": r.satisfied,
                    "occurrence_count": r.occurrence_count,
                }
                for r in verdict.rules
            ],
        }
        verdict_path.write_text(json.dumps(payload, indent=2))
        return payload
    finally:
        target.stop()


@pytest.fixture
def target_server(tmp_path):
    honeypot_path = tmp_path / "honeypot.jsonl"
    server = start_target(honeypot_path=honeypot_path, port=free_port())
    wait_healthy(server.base_url)
    yield server
    server.stop()
