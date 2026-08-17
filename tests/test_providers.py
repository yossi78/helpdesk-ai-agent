from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from harness.providers import make_provider
from harness.providers.mock_echo import MockEchoProvider
from harness.providers.openai_compatible import OpenAICompatibleProvider


def test_make_provider_unknown_type():
    with pytest.raises(ValueError, match="unknown provider type"):
        make_provider({"type": "nope"})


def test_mock_echo_from_config_and_sequence():
    provider = make_provider(
        {
            "type": "mock_echo",
            "responses": [
                {
                    "content": "one",
                    "tool_calls": [{"name": "lookup_user", "args": {"user_id": "u-42"}}],
                },
                {"content": "two"},
            ],
        }
    )
    first = provider.complete([], [])
    assert first.content == "one"
    assert first.tool_calls[0].name == "lookup_user"
    assert first.tool_calls[0].call_id == "mock-1-0"
    second = provider.complete([], [])
    assert second.content == "two"
    assert second.tool_calls == []


def test_mock_echo_exhausted():
    provider = MockEchoProvider(responses=[{"content": "only"}])
    provider.complete([], [])
    with pytest.raises(RuntimeError, match="exhausted"):
        provider.complete([], [])


def test_openai_from_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider.from_config(
            {"base_url": "http://example", "model": "gpt"}
        )


def test_openai_complete_parses_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    probe = MagicMock()
    probe.status_code = 200
    complete = MagicMock()
    complete.raise_for_status.return_value = None
    complete.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "calling",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "lookup_user",
                                "arguments": '{"user_id": "u-42"}',
                            },
                        },
                        {
                            "id": "call-2",
                            "function": {"name": "broken", "arguments": "not-json"},
                        },
                    ],
                }
            }
        ]
    }

    with patch("harness.providers.openai_compatible.requests.post") as post:
        post.side_effect = [probe, complete]
        provider = OpenAICompatibleProvider.from_config(
            {
                "base_url": "http://llm.example/v1/",
                "model": "gpt-test",
                "api_key_env": "OPENAI_API_KEY",
            }
        )
        response = provider.complete(
            [{"role": "user", "content": "hi"}],
            [{"name": "lookup_user", "description": "x", "parameters": {}}],
        )

    assert provider._supports_tools is True
    assert response.content == "calling"
    assert response.tool_calls[0].args == {"user_id": "u-42"}
    assert response.tool_calls[1].args == {"_raw": "not-json"}
    complete_payload = post.call_args_list[1].kwargs["json"]
    assert complete_payload["tools"][0]["function"]["name"] == "lookup_user"


def test_openai_skips_tools_when_probe_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    complete = MagicMock()
    complete.raise_for_status.return_value = None
    complete.json.return_value = {
        "choices": [{"message": {"content": "no tools", "tool_calls": None}}]
    }

    with patch("harness.providers.openai_compatible.requests.post") as post:
        post.side_effect = [requests.ConnectionError("down"), complete]
        provider = OpenAICompatibleProvider(
            base_url="http://llm.example/v1",
            model="gpt-test",
            api_key="sk-test",
        )
        response = provider.complete(
            [{"role": "user", "content": "hi"}],
            [{"name": "lookup_user", "description": "x", "parameters": {}}],
        )

    assert provider._supports_tools is False
    assert response.content == "no tools"
    assert response.tool_calls == []
    assert "tools" not in post.call_args_list[1].kwargs["json"]
