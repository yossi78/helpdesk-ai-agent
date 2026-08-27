"""Help-desk scenario harness — entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluator.evaluate import evaluate
from harness.agent_loop import run_agent_loop
from harness.providers import make_provider
from harness.results_report import open_results_in_chrome, write_results_html
from harness.scenario import load_scenario
from harness.thought_process import ThoughtProcessRecorder
from harness.tools import ToolHandler
from target.app import start_target


def _print_ticket_stats(recorder: ThoughtProcessRecorder) -> None:
    """Print ticket_stats tool payloads so they show up on the console."""
    results: list[tuple[str, str]] = []
    summary = ""
    for rec in recorder._records:
        if (rec.llm_content or "").strip():
            summary = rec.llm_content.strip()
        by_id = {tc["call_id"]: tc for tc in rec.llm_tool_calls}
        for result in rec.tool_results:
            tc = by_id.get(result["call_id"])
            if tc and tc["name"] == "ticket_stats":
                results.append((tc["args"], result["content"]))
    if not results:
        return

    print()
    print("Ticket stats:")
    for args, content in results:
        label = "all tickets" if args in ("{}",) else str(args)
        print(f"  {label}")
        try:
            pretty = json.dumps(json.loads(content), indent=2)
        except (json.JSONDecodeError, TypeError):
            pretty = content
        for line in pretty.splitlines():
            print(f"    {line}")
        print()
    if summary:
        print("Agent summary:")
        print(f"  {summary}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run")
    parser.add_argument("scenario", help="Path to a scenario YAML file.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for thought_process.json, honeypot.jsonl, verdict.json, results.html.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write results.html but do not open it in Chrome.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    thought_path = output_dir / "thought_process.json"
    honeypot_path = output_dir / "honeypot.jsonl"
    verdict_path = output_dir / "verdict.json"
    results_path = output_dir / "results.html"

    if results_path.exists():
        results_path.unlink()
        print(f"Removed previous report: {results_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for p in (thought_path, honeypot_path, verdict_path):
        if p.exists():
            p.unlink()

    ############# [1] - LOAD SCENARIOS FROM YAML #########################
    scenario = load_scenario(args.scenario)   
    print(f"Loaded scenario: {scenario.name}")

    ############# [2] - START FLASK #########################
    target = start_target(
        mount_path=scenario.mount_path,
        honeypot_path=honeypot_path,
    )
    print(f"Target service running at {target.base_url}")

    try:
        provider = make_provider(scenario.provider)
        tool_handler = ToolHandler(scenario.tools, target_url=target.base_url)
        recorder = ThoughtProcessRecorder()

     ############# [3] - RUN AGENT LOOP   #########################
        result = run_agent_loop(
            provider=provider,
            tool_handler=tool_handler,
            recorder=recorder,
            system_prompt=scenario.system_prompt,
            initial_message=scenario.initial_message,
            max_iterations=scenario.max_iterations,
        )

        recorder.export(thought_path)
        _print_ticket_stats(recorder)
    #  [7] EVALUATOR (evaluate.py) CHECKS honeypot.jsonl vs YAML rules  ###     
        verdict = evaluate(scenario, honeypot_path)

    # [8] - WRITE SCORE OF RUN (verdict + rule results, plus termination_reason and iterations)
    #  into verdict.json ###### ######
        verdict_payload = {
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
        verdict_path.write_text(json.dumps(verdict_payload, indent=2))
        thought = json.loads(thought_path.read_text())
        write_results_html(
            results_path,
            scenario=scenario,
            thought=thought,
            verdict=verdict_payload,
            honeypot_path=honeypot_path,
        )

        print()
        print("Scenario complete.")
        print(f"Termination: {result.reason} after {result.iterations} iteration(s)")
        print(f"Evaluator verdict: {verdict.verdict.upper()}")
        print(f"Results report: {results_path}")
        if not args.no_open:
            open_results_in_chrome(results_path)

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
