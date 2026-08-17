from __future__ import annotations

import json
import os
from typing import Any

import requests

from .base import Provider, ProviderResponse, ToolCall


_PROBE_TIMEOUT_SECONDS = 15.0


class OpenAICompatibleProvider(Provider):
    """OpenAI-compatible chat completions provider.

    Determines whether the configured model supports tool-use at
    construction time and caches the result for the lifetime of the
    instance.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._supports_tools = self._probe_tool_support()

    def _probe_tool_support(self) -> bool:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Capability probe.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._auth_headers(),
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            return False
        return resp.status_code < 400

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, messages, available_tools):
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if self._supports_tools and available_tools:
            payload["tools"] = [
                {"type": "function", "function": t} for t in available_tools
            ]

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._auth_headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(name=tc["function"]["name"], args=args, call_id=tc["id"])
            )

        return ProviderResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
        )

    @classmethod
    def from_config(cls, config):
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"environment variable {api_key_env} is not set"
            )
        return cls(
            base_url=config["base_url"],
            model=config["model"],
            api_key=api_key,
        )
