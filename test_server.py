"""Smoke tests for memory server endpoints.

Uses FastAPI TestClient with a temporary ChromaDB and incoming dir.
Seeds data via /ingest, then tests all search and stats endpoints.
"""

import json
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Spin up the app with isolated ChromaDB + incoming dirs, seed data."""
    tmp = tmp_path_factory.mktemp("memory")
    chroma_dir = str(tmp / "chromadb")
    incoming_dir = str(tmp / "incoming")

    with patch.dict(os.environ, {
        "CHROMA_DIR": chroma_dir,
        "INCOMING_DIR": incoming_dir,
        "EMBED_DEVICE": "cpu",
        "WORKER_INTERVAL": "0.1",
    }):
        # Import after patching env so module-level constants pick up overrides
        import importlib
        import server as srv
        importlib.reload(srv)

        with TestClient(srv.app) as c:
            # Seed test data
            resp = c.post("/ingest", json={"chunks": [
                {
                    "text": "User: How does the syneme parser work?\n\nAssistant: The parser uses Python's tokenize module to build a tree of every token.",
                    "user_text": "How does the syneme parser work?",
                    "session_id": "session-aaa",
                    "timestamp": "2025-06-01T10:00:00Z",
                    "project": "/home/user/syneme",
                    "turn_number": 0,
                    "branch": "main",
                },
                {
                    "text": "User: What is reference distance?\n\nAssistant: Reference distance measures how far the reader has to look to find where a name is defined. It uses jedi for static analysis.",
                    "user_text": "What is reference distance?",
                    "session_id": "session-aaa",
                    "timestamp": "2025-06-01T10:05:00Z",
                    "project": "/home/user/syneme",
                    "turn_number": 1,
                    "branch": "main",
                },
                {
                    "text": "User: Explain how Python decorators work\n\nAssistant: A decorator is a function that takes another function and extends its behavior without modifying it.",
                    "user_text": "Explain how Python decorators work",
                    "session_id": "session-bbb",
                    "timestamp": "2025-06-02T14:00:00Z",
                    "project": "/home/user/other-project",
                    "turn_number": 0,
                    "branch": "feature-x",
                },
            ]})
            assert resp.status_code == 201
            assert resp.json()["queued"] == 3

            # Wait for the background worker to process all 3 chunks
            for _ in range(50):
                time.sleep(0.2)
                s = c.get("/stats").json()
                if s["total_documents"] >= 3 and s["total_user_inputs"] >= 3:
                    break
            else:
                pytest.fail(f"Worker didn't finish: {s}")

            yield c


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_counts(client):
    s = client.get("/stats").json()
    assert s["total_documents"] == 3
    assert s["total_subchunks"] > 0
    assert s["total_user_inputs"] == 3
    assert s["queue_pending"] == 0


def test_stats_collection_names(client):
    s = client.get("/stats").json()
    assert s["collection_name"] == "conversations"
    assert s["subchunk_collection_name"] == "subchunks"
    assert s["user_input_collection_name"] == "user_inputs"


# ---------------------------------------------------------------------------
# /search (conversations collection)
# ---------------------------------------------------------------------------

def test_search_post_returns_results(client):
    resp = client.post("/search", json={"q": "syneme parser tokenize", "k": 3})
    results = resp.json()["results"]
    assert len(results) > 0
    assert "text" in results[0]
    assert "distance" in results[0]
    assert "session_id" in results[0]


def test_search_get_returns_results(client):
    resp = client.get("/search", params={"q": "parser", "k": 2})
    results = resp.json()["results"]
    assert len(results) > 0


def test_search_project_filter(client):
    resp = client.post("/search", json={
        "q": "parser",
        "k": 10,
        "project": "/home/user/other-project",
    })
    results = resp.json()["results"]
    assert all(r["project"] == "/home/user/other-project" for r in results)


def test_search_respects_k(client):
    resp = client.post("/search", json={"q": "python", "k": 1})
    results = resp.json()["results"]
    assert len(results) == 1


# ---------------------------------------------------------------------------
# /search_subchunks
# ---------------------------------------------------------------------------

def test_search_subchunks_returns_results(client):
    resp = client.post("/search_subchunks", json={"q": "tokenize module", "k": 3})
    results = resp.json()["results"]
    assert len(results) > 0
    assert results[0]["chunk_type"] == "subchunk"
    assert "parent_chunk_id" in results[0]
    assert "window_index" in results[0]


def test_search_subchunks_text_is_short(client):
    """Subchunks should be at most window_size chars (500)."""
    resp = client.post("/search_subchunks", json={"q": "jedi static analysis", "k": 5})
    for r in resp.json()["results"]:
        assert len(r["text"]) <= 500


def test_search_subchunks_session_filter(client):
    resp = client.post("/search_subchunks", json={
        "q": "parser",
        "k": 10,
        "session_id": "session-bbb",
    })
    results = resp.json()["results"]
    assert all(r["session_id"] == "session-bbb" for r in results)


# ---------------------------------------------------------------------------
# /search_user_inputs
# ---------------------------------------------------------------------------

def test_search_user_inputs_returns_results(client):
    resp = client.post("/search_user_inputs", json={"q": "how does the parser work", "k": 3})
    results = resp.json()["results"]
    assert len(results) > 0
    assert results[0]["chunk_type"] == "user_input"
    assert "parent_chunk_id" in results[0]


def test_search_user_inputs_contains_only_user_text(client):
    """User inputs should not contain assistant responses."""
    resp = client.post("/search_user_inputs", json={"q": "decorators", "k": 3})
    for r in resp.json()["results"]:
        assert "Assistant:" not in r["text"]


def test_search_user_inputs_project_filter(client):
    resp = client.post("/search_user_inputs", json={
        "q": "parser",
        "k": 10,
        "project": "/home/user/syneme",
    })
    results = resp.json()["results"]
    assert all(r["project"] == "/home/user/syneme" for r in results)


# ---------------------------------------------------------------------------
# /ingest_summary
# ---------------------------------------------------------------------------

def test_ingest_summary(client):
    resp = client.post("/ingest_summary", json={
        "text": "This session discussed syneme parser internals and reference distance.",
        "session_id": "session-aaa",
        "timestamp": "2025-06-01T11:00:00Z",
        "project": "/home/user/syneme",
    })
    assert resp.status_code == 201
    assert resp.json()["queued"] == 1


# ---------------------------------------------------------------------------
# MMR diversity
# ---------------------------------------------------------------------------

def test_mmr_produces_diverse_results(client):
    """When k > 1, results should come from different sessions when possible."""
    resp = client.post("/search", json={"q": "python code", "k": 3})
    results = resp.json()["results"]
    if len(results) >= 2:
        session_ids = {r["session_id"] for r in results}
        # With 2 distinct sessions in seed data, MMR should surface both
        assert len(session_ids) >= 2, f"Expected diverse sessions, got {session_ids}"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_duplicate_ingest_does_not_create_duplicates(client):
    """Re-ingesting the same chunk (same session_id + turn_number) should not duplicate."""
    # Drain queue first — previous tests may have queued items still being processed
    for _ in range(30):
        s = client.get("/stats").json()
        if s["queue_pending"] == 0:
            break
        time.sleep(0.2)
    else:
        pytest.fail(f"Queue didn't drain: {s['queue_pending']} pending")

    before = client.get("/stats").json()["total_documents"]

    # Re-ingest the same chunk from seed data
    resp = client.post("/ingest", json={"chunks": [
        {
            "text": "User: How does the syneme parser work?\n\nAssistant: The parser uses Python's tokenize module to build a tree of every token.",
            "user_text": "How does the syneme parser work?",
            "session_id": "session-aaa",
            "timestamp": "2025-06-01T10:00:00Z",
            "project": "/home/user/syneme",
            "turn_number": 0,
            "branch": "main",
        },
    ]})
    assert resp.status_code == 201

    # Wait for worker to process
    for _ in range(30):
        time.sleep(0.2)
        s = client.get("/stats").json()
        if s["queue_pending"] == 0:
            break
    else:
        pytest.fail(f"Worker didn't finish: {s['queue_pending']} pending")

    after = client.get("/stats").json()["total_documents"]
    assert after == before, f"Expected {before} docs, got {after} — duplicate was inserted"
