from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvaluationRule:
    tool: str
    min_count: int = 1


@dataclass
class EvaluationConfig:
    must: list[EvaluationRule] = field(default_factory=list)
    must_not: list[EvaluationRule] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    description: str
    provider: dict[str, Any]
    max_iterations: int
    tools: list[str]
    initial_message: str
    evaluation: EvaluationConfig
    system_prompt: str = ""
    expected_verdict: str | None = None
    mount_path: Path | None = None
    target_url: str = "http://127.0.0.1:5555"
    source_path: Path = field(default_factory=lambda: Path("."))


def load_scenario(path: str | Path) -> Scenario:
    scenario_path = Path(path).resolve()
    scenario_dir = scenario_path.parent
    raw = yaml.safe_load(scenario_path.read_text())

    eval_raw = raw.get("evaluation", {}) or {}
    evaluation = EvaluationConfig(
        must=[EvaluationRule(**r) for r in (eval_raw.get("must") or [])],
        must_not=[EvaluationRule(**r) for r in (eval_raw.get("must_not") or [])],
    )

    mount_path = None
    if raw.get("mount_path"):
        mount_path = _validate_mount_path(raw["mount_path"], scenario_dir)

    return Scenario(
        name=raw["name"],
        description=raw.get("description", ""),
        provider=raw["provider"],
        max_iterations=raw.get("max_iterations", 20),
        tools=raw.get("tools", []),
        initial_message=raw["initial_message"],
        evaluation=evaluation,
        system_prompt=raw.get("system_prompt", ""),
        expected_verdict=raw.get("expected_verdict"),
        mount_path=mount_path,
        target_url=raw.get("target_url", "http://127.0.0.1:5555"),
        source_path=scenario_path,
    )


def _validate_mount_path(mount_path: str, scenario_dir: Path) -> Path:
    if mount_path.startswith(("/", "~")):
        raise ValueError(
            f"mount_path must be relative to the scenario directory: {mount_path!r}"
        )
    resolved = (scenario_dir / mount_path).resolve()
    scenario_dir_resolved = scenario_dir.resolve()
    if not resolved.is_relative_to(scenario_dir_resolved):
        raise ValueError(
            f"mount_path escapes scenario directory: {mount_path!r} -> {resolved}"
        )
    return resolved
