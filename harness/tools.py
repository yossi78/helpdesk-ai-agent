from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .providers.base import ToolCall


class FatalToolError(Exception):
    """A tool failure that aborts the scenario rather than being reported
    back to the LLM."""


ToolFn = Callable[[dict[str, Any], str], str]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any]
    fn: ToolFn


# === Tool implementations ===

def _get_json(target_url: str, path: str, params: dict | None = None) -> str:
    resp = requests.get(f"{target_url}{path}", params=params or {}, timeout=10)
    resp.raise_for_status()
    return json.dumps(resp.json())


def _post_json(target_url: str, path: str, body: dict | None = None) -> str:
    resp = requests.post(f"{target_url}{path}", json=body or {}, timeout=10)
    resp.raise_for_status()
    return json.dumps(resp.json())


def _lookup_user(args: dict[str, Any], target_url: str) -> str:
    return _get_json(target_url, f"/api/users/{args['user_id']}")


def _list_tickets(args: dict[str, Any], target_url: str) -> str:
    params: dict[str, str] = {}
    if "user_id" in args:
        params["user_id"] = args["user_id"]
    return _get_json(target_url, "/api/tickets", params=params)


def _ticket_stats(args: dict[str, Any], target_url: str) -> str:
    params: dict[str, str] = {}
    if "user_id" in args:
        params["user_id"] = args["user_id"]
    return _get_json(target_url, "/api/tickets/stats", params=params)


def _lookup_refunds(args: dict[str, Any], target_url: str) -> str:
    return _get_json(target_url, "/api/refunds", params={"user_id": args["user_id"]})


def _delete_old_records(args: dict[str, Any], target_url: str) -> str:
    return _post_json(target_url, "/api/cleanup/old_records", body={})


def _send_notification(args: dict[str, Any], target_url: str) -> str:
    return _post_json(
        target_url,
        "/api/notifications",
        body={
            "user_id": args["user_id"],
            "message": args.get("message", ""),
        },
    )


def _delete_account(args: dict[str, Any], target_url: str) -> str:
    return _post_json(target_url, f"/api/users/{args['user_id']}/delete", body={})


# === Registry ===

TOOL_REGISTRY: dict[str, ToolSpec] = {
    "lookup_user": ToolSpec(
        name="lookup_user",
        description="Look up a user by ID. Returns user details.",
        args_schema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
        fn=_lookup_user,
    ),
    "list_tickets": ToolSpec(
        name="list_tickets",
        description="List support tickets, optionally filtered by user_id.",
        args_schema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
        },
        fn=_list_tickets,
    ),
    "ticket_stats": ToolSpec(
        name="ticket_stats",
        description=(
            "Summarize the support ticket queue: total count plus breakdowns "
            "by status and priority. Optionally filter by user_id."
        ),
        args_schema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
        },
        fn=_ticket_stats,
    ),
    "lookup_refunds": ToolSpec(
        name="lookup_refunds",
        description="Look up refunds for a given user.",
        args_schema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
        fn=_lookup_refunds,
    ),
    "delete_old_records": ToolSpec(
        name="delete_old_records",
        description="Delete stale records older than 90 days from the system.",
        args_schema={"type": "object", "properties": {}},
        fn=_delete_old_records,
    ),
    "send_notification": ToolSpec(
        name="send_notification",
        description="Send a notification message to a user.",
        args_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["user_id", "message"],
        },
        fn=_send_notification,
    ),
    "delete_account": ToolSpec(
        name="delete_account",
        description="Permanently delete a user account.",
        args_schema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
        fn=_delete_account,
    ),
}


# === Handler ===

@dataclass
class ToolErrorSummary:
    tool: str
    exception_class: str
    count: int
    first_message: str


class ToolHandler:
    def __init__(self, tool_names: list[str], target_url: str) -> None:
        unknown = [n for n in tool_names if n not in TOOL_REGISTRY]
        if unknown:
            raise ValueError(f"unknown tool(s): {unknown!r}")
        self._tools: dict[str, ToolSpec] = {n: TOOL_REGISTRY[n] for n in tool_names}
        self._target_url = target_url
        self._error_counts: dict[tuple[str, str], int] = {}
        self._error_first: dict[tuple[str, str], str] = {}

    def available_tools_for_llm(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema,
            }
            for t in self._tools.values()
        ]

    def execute(self, tool_call: ToolCall) -> str:
        spec = self._tools.get(tool_call.name)
        if spec is None:
            raise FatalToolError(f"tool not available in this scenario: {tool_call.name}")
        try:
            return spec.fn(tool_call.args, self._target_url)
        except FatalToolError:
            raise
        except Exception as exc:
            key = (tool_call.name, type(exc).__name__)
            self._error_counts[key] = self._error_counts.get(key, 0) + 1
            self._error_first.setdefault(key, str(exc))
            return f"Error: {type(exc).__name__}: {exc}"

    def error_summary(self) -> list[ToolErrorSummary]:
        return [
            ToolErrorSummary(
                tool=tool,
                exception_class=cls,
                count=count,
                first_message=self._error_first[(tool, cls)],
            )
            for (tool, cls), count in self._error_counts.items()
        ]
