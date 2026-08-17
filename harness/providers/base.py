from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str


@dataclass
class ProviderResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class Provider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "Provider":
        ...
