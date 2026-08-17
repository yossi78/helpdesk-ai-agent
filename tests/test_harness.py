from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from harness.agent_loop import run_agent_loop
from harness.providers.base import ProviderResponse, ToolCall
from harness.thought_process import ThoughtProcessRecorder, _MAX_RECORDED_ITERATIONS
from harness.tools import TOOL_REGISTRY, FatalToolError, ToolHandler


def _provider(*responses: ProviderResponse):
    provider = MagicMock()
    provider.complete.side_effect = list(responses)
    return provider


def test_natural_termination_without_tools():
    recorder = ThoughtProcessRecorder()
    handler = MagicMock()
    handler.available_tools_for_llm.return_value = []
    result = run_agent_loop(
        provider=_provider(ProviderResponse(content="done")),
        tool_handler=handler,
        recorder=recorder,
        system_prompt="be helpful",
        initial_message="hello",
        max_iterations=5,
    )
    assert result.reason == "natural_termination"
    assert result.iterations == 1
    handler.execute.assert_not_called()
    assert recorder._records[0].llm_content == "done"


def test_executes_tools_then_stops():
    recorder = ThoughtProcessRecorder()
    handler = MagicMock()
    handler.available_tools_for_llm.return_value = []
    handler.execute.return_value = '{"ok": true}'
    call = ToolCall(name="lookup_user", args={"user_id": "u-42"}, call_id="c1")
    result = run_agent_loop(
        provider=_provider(
            ProviderResponse(content="looking", tool_calls=[call]),
            ProviderResponse(content="done"),
        ),
        tool_handler=handler,
        recorder=recorder,
        system_prompt="",
        initial_message="look up u-42",
        max_iterations=5,
    )
    assert result.reason == "natural_termination"
    assert result.iterations == 2
    handler.execute.assert_called_once_with(call)
    assert recorder._records[0].tool_results == [{"call_id": "c1", "content": '{"ok": true}'}]


def test_iteration_limit_skips_tool_execution_on_last_turn():
    recorder = ThoughtProcessRecorder()
    handler = MagicMock()
    handler.available_tools_for_llm.return_value = []
    call = ToolCall(name="lookup_user", args={"user_id": "u-42"}, call_id="c1")
    result = run_agent_loop(
        provider=_provider(ProviderResponse(content="looking", tool_calls=[call])),
        tool_handler=handler,
        recorder=recorder,
        system_prompt="",
        initial_message="hi",
        max_iterations=1,
    )
    assert result.reason == "iteration_limit"
    assert result.iterations == 1
    handler.execute.assert_not_called()


def test_recorder_caps_at_max_iterations(tmp_path):
    recorder = ThoughtProcessRecorder()
    response = ProviderResponse(content="x")
    for i in range(_MAX_RECORDED_ITERATIONS + 5):
        recorder.record_iteration(i + 1, response, [])
    assert len(recorder._records) == _MAX_RECORDED_ITERATIONS
    out = tmp_path / "thought.json"
    recorder.export(out)
    payload = json.loads(out.read_text())
    assert payload["iteration_count"] == _MAX_RECORDED_ITERATIONS
    assert payload["iterations"][0]["llm_tool_calls"] == []


def test_recorder_uses_repr_for_tool_args():
    recorder = ThoughtProcessRecorder()
    recorder.record_iteration(
        1,
        ProviderResponse(
            content="x",
            tool_calls=[ToolCall(name="lookup_user", args={"user_id": "u-42"}, call_id="c1")],
        ),
        [],
    )
    assert recorder._records[0].llm_tool_calls[0]["args"] == repr({"user_id": "u-42"})


def test_unknown_tool_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown tool"):
        ToolHandler(["not_a_tool"], target_url="http://127.0.0.1:1")


def test_execute_unknown_to_scenario_is_fatal():
    handler = ToolHandler(["lookup_user"], target_url="http://127.0.0.1:1")
    with pytest.raises(FatalToolError, match="not available"):
        handler.execute(ToolCall(name="list_tickets", args={}, call_id="c1"))


def test_execute_wraps_non_fatal_errors(target_server):
    handler = ToolHandler(["lookup_user"], target_url=target_server.base_url)
    result = handler.execute(
        ToolCall(name="lookup_user", args={"user_id": "no-such-user"}, call_id="c1")
    )
    assert result.startswith("Error: HTTPError:")
    summary = handler.error_summary()
    assert len(summary) == 1
    assert summary[0].tool == "lookup_user"
    assert summary[0].exception_class == "HTTPError"
    assert summary[0].count == 1


def test_all_registered_tools_round_trip(target_server):
    handler = ToolHandler(list(TOOL_REGISTRY), target_url=target_server.base_url)
    specs = {t["name"]: t for t in handler.available_tools_for_llm()}
    assert set(specs) == set(TOOL_REGISTRY)

    lookup = json.loads(
        handler.execute(ToolCall("lookup_user", {"user_id": "u-42"}, "c1"))
    )
    assert lookup["name"] == "Alice Wong"

    tickets = json.loads(
        handler.execute(ToolCall("list_tickets", {"user_id": "u-42"}, "c2"))
    )
    assert len(tickets["tickets"]) == 2

    all_tickets = json.loads(handler.execute(ToolCall("list_tickets", {}, "c3")))
    assert len(all_tickets["tickets"]) == 5

    refunds = json.loads(
        handler.execute(ToolCall("lookup_refunds", {"user_id": "u-77"}, "c4"))
    )
    assert refunds["refunds"][0]["id"] == "R-501"

    notify = json.loads(
        handler.execute(
            ToolCall("send_notification", {"user_id": "u-77", "message": "hi"}, "c5")
        )
    )
    assert notify["sent"] is True

    cleanup = json.loads(handler.execute(ToolCall("delete_old_records", {}, "c6")))
    assert cleanup["deleted_count"] == 12

    deleted = json.loads(
        handler.execute(ToolCall("delete_account", {"user_id": "u-103"}, "c7"))
    )
    assert deleted["deleted"] is True


def test_fatal_error_from_tool_fn_is_not_wrapped(target_server):
    handler = ToolHandler(["lookup_user"], target_url=target_server.base_url)

    def boom(args, url):
        raise FatalToolError("hard fail")

    handler._tools["lookup_user"] = replace(TOOL_REGISTRY["lookup_user"], fn=boom)
    with pytest.raises(FatalToolError, match="hard fail"):
        handler.execute(ToolCall("lookup_user", {"user_id": "u-42"}, "c1"))
