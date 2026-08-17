from __future__ import annotations

from typing import Any

from .base import Provider, ProviderResponse, ToolCall
from .mock_echo import MockEchoProvider
from .openai_compatible import OpenAICompatibleProvider


_REGISTRY: dict[str, type[Provider]] = {
    "mock_echo": MockEchoProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def make_provider(config: dict[str, Any]) -> Provider:
    ptype = config.get("type")
    if ptype not in _REGISTRY:
        raise ValueError(f"unknown provider type: {ptype!r}")
    return _REGISTRY[ptype].from_config(config)


__all__ = [
    "Provider",
    "ProviderResponse",
    "ToolCall",
    "MockEchoProvider",
    "OpenAICompatibleProvider",
    "make_provider",
]
