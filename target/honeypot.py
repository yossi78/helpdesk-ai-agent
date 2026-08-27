from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import InMemoryStore


log = logging.getLogger(__name__)


def read_entries(path: Path) -> list[dict[str, Any]]:
    """Read every entry the honeypot wrote, newest last.

    Accepts both the indented format written by `Honeypot.record` and the
    single-line-per-entry format the log used previously. Malformed text is
    skipped by resyncing to the next line rather than aborting the read.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    entries: list[dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        try:
            entry, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            newline = text.find("\n", pos)
            if newline == -1:
                break
            pos = newline + 1
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


class Honeypot:
    """Append-only recorder of security-relevant events.

    Each entry is written as an indented JSON object followed by a newline, so
    the log stays readable by hand; use `read_entries` to parse it back.

    Failures during recording are logged but never raised — the entity
    should keep serving even if its audit channel is broken.
    """

    def __init__(self, log_path: Path | str | None = None):
        if log_path is None:
            env_path = os.environ.get("HONEYPOT_LOG_PATH")
            log_path = Path(env_path) if env_path else None
        elif isinstance(log_path, str):
            log_path = Path(log_path)
        self.log_path: Path | None = log_path

    def record(self, category: str, data: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "category": category,
                "data": data,
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
      ######  [6] - WRITE AUDIT LOG (honeypot.jsonl)     ########    
            with self.log_path.open("a") as f:
                f.write(json.dumps(entry, indent=2) + "\n")
        except Exception as exc:
            log.warning("honeypot record failed: %s", exc)


def _query_recording_enabled(store: InMemoryStore) -> bool:
    if store.vulnerable:
        return True
    return os.environ.get("TARGET_HONEYPOT_QUERIES", "").lower() in (
        "1",
        "true",
        "yes",
    )


def attach_to_store(honeypot: Honeypot, store: InMemoryStore) -> None:
    """Wrap the store's mutating and reading methods to emit honeypot
    events on success.

    Idempotent — subsequent calls after the first are no-ops.
    """
    if getattr(store, "_honeypot_attached", False):
        return
    store._honeypot_attached = True  # type: ignore[attr-defined]

    orig_insert = store.insert
    orig_query = store.query
    orig_get_by_id = store.get_by_id

    def _wrapped_insert(category: str, item: dict[str, Any]) -> dict[str, Any]:
        result = orig_insert(category, item)
        honeypot.record(
            "record_created",
            {"category": category, "item_id": item.get("id")},
        )
        return result

    def _wrapped_query(category: str, **filters: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        success = False
        try:
            result = orig_query(category, **filters)
            success = True
            return result
        finally:
            if _query_recording_enabled(store):
                honeypot.record(
                    "record_queried",
                    {
                        "category": category,
                        "filters": filters,
                        "result_count": len(result),
                        "success": success,
                    },
                )

    def _wrapped_get_by_id(category: str, id_value: Any) -> dict[str, Any] | None:
        result: dict[str, Any] | None = None
        success = False
        try:
            result = orig_get_by_id(category, id_value)
            success = True
            return result
        finally:
            if _query_recording_enabled(store):
                honeypot.record(
                    "record_queried",
                    {
                        "category": category,
                        "id": id_value,
                        "result_count": 1 if result else 0,
                        "success": success,
                    },
                )

    store.insert = _wrapped_insert  # type: ignore[method-assign]
    store.query = _wrapped_query  # type: ignore[method-assign]
    store.get_by_id = _wrapped_get_by_id  # type: ignore[method-assign]
