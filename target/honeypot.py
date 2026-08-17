from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import InMemoryStore


log = logging.getLogger(__name__)


class Honeypot:
    """Append-only JSONL recorder of security-relevant events.

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
            with self.log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
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
