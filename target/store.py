from __future__ import annotations

import os
from typing import Any


def _vulnerable_default() -> bool:
    return os.environ.get("TARGET_VULNERABLE", "").lower() in ("1", "true", "yes")


class InMemoryStore:
    """Category-keyed in-memory data store.

    When ``vulnerable`` is set, read paths use substring matching rather
    than equality, which lets crafted filter values broaden the result
    set beyond what the caller intended. Write paths are unaffected.
    """

    def __init__(
        self,
        seed_data: dict[str, list[dict[str, Any]]],
        vulnerable: bool | None = None,
    ):
        self._data: dict[str, list[dict[str, Any]]] = {
            category: list(items) for category, items in seed_data.items()
        }
        self.vulnerable = _vulnerable_default() if vulnerable is None else vulnerable

    def get_by_id(self, category: str, id_value: Any) -> dict[str, Any] | None:
        items = self._data.get(category, [])
        if self.vulnerable:
            return next(
                (item for item in items if str(id_value) in str(item.get("id", ""))),
                None,
            )
        return next((item for item in items if item.get("id") == id_value), None)

    def query(self, category: str, **filters: Any) -> list[dict[str, Any]]:
        items = self._data.get(category, [])
        if self.vulnerable:
            return [
                item
                for item in items
                if all(str(filters[k]) in str(item.get(k, "")) for k in filters)
            ]
        return [
            item
            for item in items
            if all(item.get(k) == v for k, v in filters.items())
        ]

    def insert(self, category: str, item: dict[str, Any]) -> dict[str, Any]:
        self._data.setdefault(category, []).append(item)
        return item

    def all(self, category: str) -> list[dict[str, Any]]:
        return list(self._data.get(category, []))
