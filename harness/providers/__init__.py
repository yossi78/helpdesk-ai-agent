from __future__ import annotations

from typing import Any

from .base import Provider, ProviderResponse, ToolCall
from .mock_echo import MockEchoProvider
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider


_REGISTRY: dict[str, type[Provider]] = {
    "mock_echo": MockEchoProvider,
    "ollama": OllamaProvider,
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
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "make_provider",
]
