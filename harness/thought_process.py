from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers.base import ProviderResponse


_MAX_RECORDED_ITERATIONS = 100


@dataclass
class IterationRecord:
    iteration: int
    llm_content: str
    llm_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]


class ThoughtProcessRecorder:
    def __init__(self) -> None:
        self._records: list[IterationRecord] = []

    def record_iteration(
        self,
        iteration: int,
        response: ProviderResponse,
        tool_results: list[dict[str, Any]],
    ) -> None:
        if len(self._records) >= _MAX_RECORDED_ITERATIONS:
            return
        self._records.append(
            IterationRecord(
                iteration=iteration,
                llm_content=response.content,
                llm_tool_calls=[
                    {
                        "name": tc.name,
                        "args": repr(tc.args),
                        "call_id": tc.call_id,
                    }
                    for tc in response.tool_calls
                ],
                tool_results=tool_results,
            )
        )

    def export(self, path: Path) -> None:
        payload = {
            "iteration_count": len(self._records),
            "iterations": [
                {
                    "iteration": r.iteration,
                    "llm_content": r.llm_content,
                    "llm_tool_calls": r.llm_tool_calls,
                    "tool_results": r.tool_results,
                }
                for r in self._records
            ],
        }
        path.write_text(json.dumps(payload, indent=2))
