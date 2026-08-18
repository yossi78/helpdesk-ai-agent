from __future__ import annotations

import json
from typing import Any

import requests

from .base import Provider, ProviderResponse, ToolCall


_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "llama3.2"


class OllamaProvider(Provider):
    """Ollama native chat provider (`POST /api/chat`).

    The agent loop speaks OpenAI-style messages; this class translates
    them to Ollama's native tool-calling format so the rest of the
    harness does not need to know which backend is in use.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, messages, available_tools):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_ollama_messages(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if available_tools:
            payload["tools"] = [
                {"type": "function", "function": t} for t in available_tools
            ]

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed at {self.base_url}: {exc}. "
                "Start it with `docker compose up ollama`."
            ) from exc
        resp.raise_for_status()
        msg = resp.json().get("message") or {}

        tool_calls: list[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    name=fn.get("name") or "",
                    args=_parse_args(fn.get("arguments")),
                    call_id=tc.get("id") or f"ollama-{i}",
                )
            )

        return ProviderResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            base_url=config.get("base_url", _DEFAULT_BASE_URL),
            model=config.get("model", _DEFAULT_MODEL),
            timeout=float(config.get("timeout", 120.0)),
            temperature=float(config.get("temperature", 0.0)),
        )


def _parse_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    return {"_raw": raw}


def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style chat messages to Ollama's native shape."""
    call_names: dict[str, str] = {}
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        out: dict[str, Any] = {"role": role, "content": msg.get("content") or ""}
        if role == "assistant" and msg.get("tool_calls"):
            ollama_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                call_id = tc.get("id")
                if call_id:
                    call_names[call_id] = name
                ollama_calls.append(
                    {"function": {"name": name, "arguments": _parse_args(fn.get("arguments"))}}
                )
            out["tool_calls"] = ollama_calls
        elif role == "tool":
            name = msg.get("tool_name") or call_names.get(msg.get("tool_call_id") or "", "")
            if name:
                out["tool_name"] = name
        converted.append(out)
    return converted
