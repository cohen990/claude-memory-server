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
                    "text": "User: How does the acme parser work?\n\nAssistant: The parser uses Python's tokenize module to build a tree of every token.",
                    "user_text": "How does the acme parser work?",
                    "session_id": "session-aaa",
                    "timestamp": "2025-06-01T10:00:00Z",
                    "project": "/home/user/acme",
                    "turn_number": 0,
                    "branch": "main",
                },
                {
                    "text": "User: What is reference distance?\n\nAssistant: Reference distance measures how far the reader has to look to find where a name is defined. It uses jedi for static analysis.",
                    "user_text": "What is reference distance?",
                    "session_id": "session-aaa",
                    "timestamp": "2025-06-01T10:05:00Z",
                    "project": "/home/user/acme",
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
    resp = client.post("/search", json={"q": "acme parser tokenize", "k": 3})
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
        "project": "/home/user/acme",
    })
    results = resp.json()["results"]
    assert all(r["project"] == "/home/user/acme" for r in results)


# ---------------------------------------------------------------------------
# /ingest_summary
# ---------------------------------------------------------------------------

def test_ingest_summary(client):
    resp = client.post("/ingest_summary", json={
        "text": "This session discussed acme parser internals and reference distance.",
        "session_id": "session-aaa",
        "timestamp": "2025-06-01T11:00:00Z",
        "project": "/home/user/acme",
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

# ---------------------------------------------------------------------------
# /chunks/undreamed and /chunks/mark_dreamed
# ---------------------------------------------------------------------------

def test_chunks_undreamed_returns_all_chunks(client):
    """All seeded chunks should be undreamed initially."""
    resp = client.get("/chunks/undreamed")
    data = resp.json()
    assert resp.status_code == 200
    assert data["total"] >= 3
    for chunk in data["chunks"]:
        assert "id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk


def test_chunks_undreamed_days_filter(client):
    """days=1 should exclude chunks from 2025."""
    resp = client.get("/chunks/undreamed", params={"days": 1})
    data = resp.json()
    # Seed data is from 2025, so with days=1 from now (2026), nothing matches
    assert data["total"] == 0


def test_chunks_mark_dreamed(client):
    """Marking a chunk as dreamed should remove it from undreamed results."""
    # Get undreamed chunks
    resp = client.get("/chunks/undreamed")
    chunks = resp.json()["chunks"]
    assert len(chunks) >= 1

    target = chunks[0]
    target_id = target["id"]
    target_meta = target["metadata"]

    # Mark it as dreamed
    resp = client.post("/chunks/mark_dreamed", json={
        "ids": [target_id],
        "metadatas": [{**target_meta, "dreamed": 1}],
    })
    assert resp.status_code == 200
    assert resp.json()["marked"] == 1

    # Verify it's gone from undreamed
    resp = client.get("/chunks/undreamed")
    undreamed_ids = [c["id"] for c in resp.json()["chunks"]]
    assert target_id not in undreamed_ids

    # Restore it for other tests
    resp = client.post("/chunks/mark_dreamed", json={
        "ids": [target_id],
        "metadatas": [{**target_meta, "dreamed": 0}],
    })
    assert resp.json()["marked"] == 1


def test_chunks_mark_dreamed_empty(client):
    """Empty ids list should return marked=0."""
    resp = client.post("/chunks/mark_dreamed", json={"ids": [], "metadatas": []})
    assert resp.json()["marked"] == 0


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
            "text": "User: How does the acme parser work?\n\nAssistant: The parser uses Python's tokenize module to build a tree of every token.",
            "user_text": "How does the acme parser work?",
            "session_id": "session-aaa",
            "timestamp": "2025-06-01T10:00:00Z",
            "project": "/home/user/acme",
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


# ---------------------------------------------------------------------------
# /delete
# ---------------------------------------------------------------------------

def test_delete_requires_filter(client):
    """Must specify session_ids or project."""
    resp = client.post("/delete", json={"dry_run": True})
    assert resp.status_code == 400


def test_delete_dry_run_by_session(client):
    """Dry run should report counts without deleting anything."""
    before = client.get("/stats").json()

    resp = client.post("/delete", json={
        "session_ids": ["session-bbb"],
        "dry_run": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True
    assert data["chunks"] == 1       # session-bbb has 1 chunk
    assert data["user_inputs"] == 1
    assert data["subchunks"] > 0

    # Nothing actually deleted
    after = client.get("/stats").json()
    assert after["total_documents"] == before["total_documents"]


def test_delete_dry_run_by_project(client):
    """Dry run by project should find matching chunks."""
    resp = client.post("/delete", json={
        "project": "/home/user/acme",
        "dry_run": True,
    })
    data = resp.json()
    assert data["chunks"] >= 2  # session-aaa has 2+ chunks in acme (summary may add more)


def test_delete_by_session(client):
    """Deleting session-bbb should remove it from all collections."""
    before = client.get("/stats").json()

    resp = client.post("/delete", json={
        "session_ids": ["session-bbb"],
        "dry_run": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is False
    assert data["chunks"] == 1

    after = client.get("/stats").json()
    assert after["total_documents"] == before["total_documents"] - 1
    assert after["total_user_inputs"] == before["total_user_inputs"] - 1
    assert after["total_subchunks"] == before["total_subchunks"] - data["subchunks"]

    # session-bbb should no longer appear in search
    resp = client.post("/search", json={
        "q": "decorators",
        "k": 10,
        "session_id": "session-bbb",
    })
    assert len(resp.json()["results"]) == 0

    # session-aaa should still be intact
    resp = client.post("/search", json={
        "q": "parser",
        "k": 10,
        "session_id": "session-aaa",
    })
    assert len(resp.json()["results"]) > 0
