"""Help-desk scenario harness — entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluator.evaluate import evaluate
from harness.agent_loop import run_agent_loop
from harness.providers import make_provider
from harness.scenario import load_scenario
from harness.thought_process import ThoughtProcessRecorder
from harness.tools import ToolHandler
from target.app import start_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run")
    parser.add_argument("scenario", help="Path to a scenario YAML file.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for thought_process.json, honeypot.jsonl, verdict.json.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thought_path = output_dir / "thought_process.json"
    honeypot_path = output_dir / "honeypot.jsonl"
    verdict_path = output_dir / "verdict.json"

    for p in (thought_path, honeypot_path, verdict_path):
        if p.exists():
            p.unlink()

    ############# LOAD SCENARIO FROM YAML #########################
    scenario = load_scenario(args.scenario)   
    print(f"Loaded scenario: {scenario.name}")

    target = start_target(
        mount_path=scenario.mount_path,
        honeypot_path=honeypot_path,
    )
    print(f"Target service running at {target.base_url}")

    try:
        provider = make_provider(scenario.provider)
        tool_handler = ToolHandler(scenario.tools, target_url=target.base_url)
        recorder = ThoughtProcessRecorder()

     ############# RUN AGENT LOOP   #########################
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

        verdict_path.write_text(
            json.dumps(
                {
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
                },
                indent=2,
            )
        )

        print()
        print("Scenario complete.")
        print(f"Termination: {result.reason} after {result.iterations} iteration(s)")
        print(f"Evaluator verdict: {verdict.verdict.upper()}")

        if scenario.expected_verdict:
            expected = scenario.expected_verdict.lower()
            if expected != verdict.verdict:
                print(f"Expected verdict:  {expected.upper()}")
                print("Mismatch — see output/ for details.")
                return 1

        return 0
    finally:
        target.stop()


if __name__ == "__main__":
    sys.exit(main())
