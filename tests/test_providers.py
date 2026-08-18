from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from harness.agent_loop import run_agent_loop
from harness.providers import make_provider
from harness.providers.mock_echo import MockEchoProvider
from harness.providers.ollama import OllamaProvider, _parse_args, _to_ollama_messages
from harness.providers.openai_compatible import OpenAICompatibleProvider
from harness.thought_process import ThoughtProcessRecorder
from harness.tools import ToolHandler


def _ollama_http(message: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": message}
    return resp


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


def test_ollama_from_config_defaults():
    provider = make_provider({"type": "ollama"})
    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://127.0.0.1:11434"
    assert provider.model == "llama3.2"
    assert provider.timeout == 120.0
    assert provider.temperature == 0.0


def test_ollama_complete_parses_native_tool_calls():
    complete = MagicMock()
    complete.raise_for_status.return_value = None
    complete.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "checking the queue",
            "tool_calls": [
                {
                    "function": {
                        "name": "ticket_stats",
                        "arguments": {"user_id": "u-42"},
                    }
                },
                {
                    "function": {
                        "name": "broken",
                        "arguments": "not-json",
                    }
                },
            ],
        }
    }

    with patch("harness.providers.ollama.requests.post") as post:
        post.return_value = complete
        provider = OllamaProvider.from_config(
            {
                "base_url": "http://ollama.example/",
                "model": "llama3.2",
            }
        )
        response = provider.complete(
            [{"role": "user", "content": "queue stats"}],
            [{"name": "ticket_stats", "description": "x", "parameters": {}}],
        )

    assert response.content == "checking the queue"
    assert response.tool_calls[0].name == "ticket_stats"
    assert response.tool_calls[0].args == {"user_id": "u-42"}
    assert response.tool_calls[0].call_id == "ollama-0"
    assert response.tool_calls[1].args == {"_raw": "not-json"}
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False
    assert payload["tools"][0]["function"]["name"] == "ticket_stats"
    assert post.call_args.args[0] == "http://ollama.example/api/chat"


def test_ollama_translates_openai_style_history():
    complete = MagicMock()
    complete.raise_for_status.return_value = None
    complete.json.return_value = {
        "message": {"role": "assistant", "content": "done", "tool_calls": None}
    }

    history = [
        {"role": "user", "content": "stats please"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "ticket_stats",
                        "arguments": '{"user_id": "u-42"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"total": 2}',
        },
    ]

    with patch("harness.providers.ollama.requests.post") as post:
        post.return_value = complete
        provider = OllamaProvider(base_url="http://ollama.example", model="llama3.2")
        response = provider.complete(history, [])

    assert response.content == "done"
    assert response.tool_calls == []
    messages = post.call_args.kwargs["json"]["messages"]
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"user_id": "u-42"}
    assert "id" not in messages[1]["tool_calls"][0]
    assert "tool_call_id" not in messages[2]
    assert messages[2]["tool_name"] == "ticket_stats"
    assert "tools" not in post.call_args.kwargs["json"]


def test_ollama_complete_wraps_connection_errors():
    with patch("harness.providers.ollama.requests.post") as post:
        post.side_effect = requests.ConnectionError("down")
        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="docker compose up ollama"):
            provider.complete([{"role": "user", "content": "hi"}], [])


def test_ollama_complete_wraps_timeouts():
    with patch("harness.providers.ollama.requests.post") as post:
        post.side_effect = requests.Timeout("slow")
        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            provider.complete([{"role": "user", "content": "hi"}], [])


def test_ollama_http_errors_from_raise_for_status_propagate():
    complete = _ollama_http({"content": "unused"})
    complete.raise_for_status.side_effect = requests.HTTPError("500")
    with patch("harness.providers.ollama.requests.post", return_value=complete):
        provider = OllamaProvider()
        with pytest.raises(requests.HTTPError, match="500"):
            provider.complete([{"role": "user", "content": "hi"}], [])


def test_ollama_from_config_overrides():
    provider = OllamaProvider.from_config(
        {
            "base_url": "http://custom:11434/",
            "model": "qwen2.5",
            "timeout": 30,
            "temperature": 0.4,
        }
    )
    assert provider.base_url == "http://custom:11434"
    assert provider.model == "qwen2.5"
    assert provider.timeout == 30.0
    assert provider.temperature == 0.4


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, {}),
        ({}, {}),
        ({"user_id": "u-42"}, {"user_id": "u-42"}),
        ("", {}),
        ('{"user_id": "u-42"}', {"user_id": "u-42"}),
        ("not-json", {"_raw": "not-json"}),
        ("[1, 2]", {"_raw": "[1, 2]"}),
        (42, {"_raw": 42}),
    ],
)
def test_ollama_parse_args(raw, expected):
    assert _parse_args(raw) == expected


def test_to_ollama_messages_defaults_and_tool_name_lookup():
    converted = _to_ollama_messages(
        [
            {"content": None},
            {"role": "system", "content": "rules"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "ticket_stats", "arguments": "{}"},
                    },
                    {"function": {}},
                ],
            },
            {"role": "assistant", "content": "no tools"},
            {"role": "tool", "tool_call_id": "call-1", "content": '{"total": 5}'},
            {"role": "tool", "tool_call_id": "missing", "content": "orphan"},
            {
                "role": "tool",
                "tool_name": "explicit",
                "tool_call_id": "call-1",
                "content": "prefer explicit",
            },
        ]
    )
    assert converted[0] == {"role": "user", "content": ""}
    assert converted[1] == {"role": "system", "content": "rules"}
    assert converted[2]["tool_calls"] == [
        {"function": {"name": "ticket_stats", "arguments": {}}},
        {"function": {"name": "", "arguments": {}}},
    ]
    assert "tool_calls" not in converted[3]
    assert converted[4]["tool_name"] == "ticket_stats"
    assert "tool_call_id" not in converted[4]
    assert "tool_name" not in converted[5]
    assert converted[6]["tool_name"] == "explicit"


def test_ollama_complete_handles_empty_message_and_preserves_native_id():
    complete = _ollama_http(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "native-9",
                    "function": {"name": "ticket_stats", "arguments": None},
                }
            ],
        }
    )
    with patch("harness.providers.ollama.requests.post") as post:
        post.return_value = complete
        provider = OllamaProvider(timeout=45.0, temperature=0.2)
        response = provider.complete([{"role": "user", "content": "hi"}], [])

    assert response.content == ""
    assert response.tool_calls[0].call_id == "native-9"
    assert response.tool_calls[0].args == {}
    payload = post.call_args.kwargs["json"]
    assert payload["options"]["temperature"] == 0.2
    assert post.call_args.kwargs["timeout"] == 45.0
    assert "tools" not in payload


def test_ollama_complete_empty_response_body():
    complete = MagicMock()
    complete.raise_for_status.return_value = None
    complete.json.return_value = {}
    with patch("harness.providers.ollama.requests.post", return_value=complete):
        response = OllamaProvider().complete([], [])
    assert response.content == ""
    assert response.tool_calls == []


def test_ollama_batched_tool_calls_are_two_agent_iterations(target_server):
    first = _ollama_http(
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "ticket_stats", "arguments": {}}},
                {
                    "function": {
                        "name": "ticket_stats",
                        "arguments": {"user_id": "u-42"},
                    }
                },
            ],
        }
    )
    second = _ollama_http(
        {"content": "Queue has 5 tickets; Alice has 2 open.", "tool_calls": []}
    )
    with patch("harness.providers.ollama.requests.post") as post:
        post.side_effect = [first, second]
        result = run_agent_loop(
            provider=OllamaProvider(),
            tool_handler=ToolHandler(
                ["ticket_stats"], target_url=target_server.base_url
            ),
            recorder=ThoughtProcessRecorder(),
            system_prompt="",
            initial_message="stats",
            max_iterations=10,
        )

    assert result.reason == "natural_termination"
    assert result.iterations == 2
    second_messages = post.call_args_list[1].kwargs["json"]["messages"]
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]
    assert [m["tool_name"] for m in tool_msgs] == ["ticket_stats", "ticket_stats"]
    assert json.loads(tool_msgs[0]["content"])["total"] == 5
    assert json.loads(tool_msgs[1]["content"])["total"] == 2
