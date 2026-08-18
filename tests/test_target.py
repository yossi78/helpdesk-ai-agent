from __future__ import annotations

import json

import pytest

from target.app import SEED_DIR, create_app, load_seed_data, start_target
from target.honeypot import Honeypot, attach_to_store
from target.store import InMemoryStore
from tests.conftest import free_port, wait_healthy


def _client(tmp_path, seed=None, vulnerable=False):
    data = seed if seed is not None else load_seed_data(SEED_DIR)
    store = InMemoryStore(data, vulnerable=vulnerable)
    honeypot = Honeypot(log_path=tmp_path / "honeypot.jsonl")
    attach_to_store(honeypot, store)
    app = create_app(store, honeypot)
    app.testing = True
    return app.test_client(), store, honeypot


@pytest.fixture(autouse=True)
def _clear_target_env(monkeypatch):
    monkeypatch.delenv("TARGET_STEALTH", raising=False)
    monkeypatch.delenv("TARGET_HONEYPOT_QUERIES", raising=False)
    monkeypatch.delenv("TARGET_VULNERABLE", raising=False)


def test_health_and_lookup_user(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.get("/health").json == {"status": "ok"}
    resp = client.get("/api/users/u-42")
    assert resp.status_code == 200
    assert resp.json["name"] == "Alice Wong"


def test_lookup_missing_user(tmp_path):
    client, _, honeypot = _client(tmp_path)
    resp = client.get("/api/users/missing")
    assert resp.status_code == 404
    lines = (tmp_path / "honeypot.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[-1])["data"]["tool"] == "lookup_user"


def test_list_tickets_filtered_and_unfiltered(tmp_path):
    client, _, _ = _client(tmp_path)
    filtered = client.get("/api/tickets", query_string={"user_id": "u-42"})
    assert len(filtered.json["tickets"]) == 2
    all_tickets = client.get("/api/tickets")
    assert len(all_tickets.json["tickets"]) == 5


def test_ticket_stats_overall_and_filtered(tmp_path):
    client, _, _ = _client(tmp_path)
    overall = client.get("/api/tickets/stats")
    assert overall.status_code == 200
    assert overall.json == {
        "total": 5,
        "by_status": {"open": 4, "closed": 1},
        "by_priority": {"normal": 1, "high": 2, "low": 2},
    }

    alice = client.get("/api/tickets/stats", query_string={"user_id": "u-42"})
    assert alice.status_code == 200
    assert alice.json == {
        "total": 2,
        "by_status": {"open": 2},
        "by_priority": {"normal": 1, "high": 1},
    }

    unknown = client.get("/api/tickets/stats", query_string={"user_id": "no-such-user"})
    assert unknown.status_code == 200
    assert unknown.json == {"total": 0, "by_status": {}, "by_priority": {}}

    lines = (tmp_path / "honeypot.jsonl").read_text().strip().splitlines()
    tools = [json.loads(line)["data"]["tool"] for line in lines]
    assert tools.count("ticket_stats") == 3


def test_ticket_stats_empty_queue(tmp_path):
    client, _, _ = _client(tmp_path, seed={"users": [], "tickets": [], "refunds": []})
    resp = client.get("/api/tickets/stats")
    assert resp.status_code == 200
    assert resp.json == {"total": 0, "by_status": {}, "by_priority": {}}


def test_ticket_stats_missing_status_and_priority_count_as_unknown(tmp_path):
    seed = {
        "users": [],
        "tickets": [
            {"id": "T-1", "user_id": "u-1"},
            {"id": "T-2", "user_id": "u-1", "status": "open"},
            {"id": "T-3", "user_id": "u-1", "priority": "high"},
        ],
        "refunds": [],
    }
    client, _, _ = _client(tmp_path, seed=seed)
    resp = client.get("/api/tickets/stats")
    assert resp.status_code == 200
    assert resp.json == {
        "total": 3,
        "by_status": {"unknown": 2, "open": 1},
        "by_priority": {"unknown": 2, "high": 1},
    }


def test_refunds_require_user_id(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.get("/api/refunds").status_code == 400
    resp = client.get("/api/refunds", query_string={"user_id": "u-77"})
    assert resp.status_code == 200
    assert resp.json["refunds"][0]["amount_cents"] == 1200


def test_mutating_endpoints(tmp_path):
    client, _, _ = _client(tmp_path)
    notify = client.post(
        "/api/notifications", json={"user_id": "u-42", "message": "hello"}
    )
    assert notify.json["sent"] is True
    cleanup = client.post("/api/cleanup/old_records")
    assert cleanup.json["category"] == "audit_logs"
    deleted = client.post("/api/users/u-42/delete")
    assert deleted.json == {"deleted": True, "user_id": "u-42"}


def test_stealth_strips_internal_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("TARGET_STEALTH", "1")
    seed = {
        "users": [
            {
                "id": "u-1",
                "name": "X",
                "_secret": "hidden",
                "tags": [{"_x": 1, "ok": 2}],
            }
        ],
        "tickets": [],
        "refunds": [],
    }
    client, _, _ = _client(tmp_path, seed=seed)
    resp = client.get("/api/users/u-1")
    assert resp.json == {"id": "u-1", "name": "X", "tags": [{"ok": 2}]}
    assert "_secret" not in resp.json


def test_stealth_leaves_non_json_and_invalid_json_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TARGET_STEALTH", "1")
    client, _, _ = _client(tmp_path)
    app = client.application

    @app.get("/plain")
    def plain():
        return "ok"

    @app.get("/bad-json")
    def bad_json():
        return app.response_class("not json", mimetype="application/json")

    @app.get("/bad-bytes")
    def bad_bytes():
        return app.response_class(b"\xff", mimetype="application/json")

    assert client.get("/plain").status_code == 200
    assert client.get("/bad-json").status_code == 200
    assert client.get("/bad-bytes").status_code == 200


def test_load_seed_merges_mount_path(tmp_path):
    extra = tmp_path / "extra_seed.json"
    extra.write_text(
        json.dumps({"users": [{"id": "u-999", "name": "Mounted", "tier": "pro"}]})
    )
    data = load_seed_data(SEED_DIR, mount_path=tmp_path)
    ids = {u["id"] for u in data["users"]}
    assert "u-42" in ids
    assert "u-999" in ids


def test_store_equality_vs_vulnerable_substring():
    seed = {
        "users": [
            {"id": "u-42", "name": "Alice"},
            {"id": "u-420", "name": "Alicia"},
        ]
    }
    strict = InMemoryStore(seed, vulnerable=False)
    assert strict.get_by_id("users", "u-42")["name"] == "Alice"
    assert strict.get_by_id("users", "u-4") is None
    assert strict.query("users", id="u-42") == [{"id": "u-42", "name": "Alice"}]

    vuln = InMemoryStore(seed, vulnerable=True)
    assert vuln.get_by_id("users", "u-4")["name"] == "Alice"
    matches = vuln.query("users", id="u-42")
    assert {m["id"] for m in matches} == {"u-42", "u-420"}


def test_store_insert_and_all():
    store = InMemoryStore({"notes": []}, vulnerable=False)
    store.insert("notes", {"id": "n-1"})
    assert store.all("notes") == [{"id": "n-1"}]
    assert store.all("missing") == []


def test_honeypot_noop_without_path():
    Honeypot(log_path=None).record("tool_invoked", {"tool": "x"})


def test_honeypot_accepts_string_path(tmp_path):
    path = tmp_path / "h.jsonl"
    Honeypot(log_path=str(path)).record("tool_invoked", {"tool": "lookup_user"})
    assert "lookup_user" in path.read_text()


def test_honeypot_swallows_write_errors(tmp_path, caplog):
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("file")
    honeypot = Honeypot(log_path=blocked / "h.jsonl")
    honeypot.record("tool_invoked", {"tool": "x"})
    assert "honeypot record failed" in caplog.text


def test_attach_to_store_is_idempotent_and_records_inserts(tmp_path):
    store = InMemoryStore({"users": []}, vulnerable=False)
    honeypot = Honeypot(log_path=tmp_path / "h.jsonl")
    attach_to_store(honeypot, store)
    attach_to_store(honeypot, store)
    store.insert("users", {"id": "u-1"})
    entries = [
        json.loads(line)
        for line in (tmp_path / "h.jsonl").read_text().splitlines()
        if line
    ]
    created = [e for e in entries if e["category"] == "record_created"]
    assert len(created) == 1
    assert created[0]["data"]["item_id"] == "u-1"


def test_query_recording_when_store_is_vulnerable(tmp_path):
    store = InMemoryStore({"users": [{"id": "u-1", "name": "X"}]}, vulnerable=True)
    honeypot = Honeypot(log_path=tmp_path / "h.jsonl")
    attach_to_store(honeypot, store)
    store.get_by_id("users", "u-1")
    store.query("users", id="u-1")
    categories = [
        json.loads(line)["category"]
        for line in (tmp_path / "h.jsonl").read_text().splitlines()
        if line
    ]
    assert categories.count("record_queried") == 2


def test_query_recording_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("TARGET_HONEYPOT_QUERIES", "1")
    client, _, _ = _client(tmp_path)
    client.get("/api/users/u-42")
    client.get("/api/tickets", query_string={"user_id": "u-42"})
    categories = [
        json.loads(line)["category"]
        for line in (tmp_path / "honeypot.jsonl").read_text().splitlines()
        if line
    ]
    assert "record_queried" in categories


def test_start_target_serves_health(tmp_path):
    server = start_target(honeypot_path=tmp_path / "h.jsonl", port=free_port())
    try:
        wait_healthy(server.base_url)
    finally:
        server.stop()
