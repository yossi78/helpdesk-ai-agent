from __future__ import annotations

from typing import Any

from .base import Provider, ProviderResponse, ToolCall


class MockEchoProvider(Provider):
    """Deterministic provider that returns a pre-programmed sequence of
    responses, one per call to complete()."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._index = 0

    def complete(self, messages, available_tools):
        if self._index >= len(self._responses):
            raise RuntimeError(
                f"MockEchoProvider exhausted: scenario asked for response "
                f"#{self._index + 1} but only {len(self._responses)} are configured"
            )
      #  [4] - PROVIDER (mock_echo) PICK  THE NEXT CANNED ACTION (a real LLM would decide here)  #
        spec = self._responses[self._index]
        self._index += 1
        tool_calls = [
            ToolCall(
                name=tc["name"],
                args=tc.get("args", {}),
                call_id=f"mock-{self._index}-{i}",
            )
            for i, tc in enumerate(spec.get("tool_calls", []) or [])
        ]
        return ProviderResponse(
            content=spec.get("content", ""),
            tool_calls=tool_calls,
        )

    @classmethod
    def from_config(cls, config):
        return cls(responses=config.get("responses", []))
