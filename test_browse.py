"""Tests for browse.py — memory graph browser API proxy.

Uses FastAPI TestClient against browse.py with mocked upstream responses.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake upstream data
# ---------------------------------------------------------------------------

FAKE_NODES = [
    {"id": "n1", "type": "vibe", "text": "test vibe one",
     "created_at": "2025-01-01T00:00:00", "updated_at": "2025-01-01T00:00:00",
     "source_ids": ["s1"]},
    {"id": "n2", "type": "detail", "text": "test detail one",
     "created_at": "2025-01-01T00:00:00", "updated_at": "2025-01-01T00:00:00",
     "source_ids": []},
    {"id": "n3", "type": "vibe", "text": "test vibe two",
     "created_at": "2025-01-01T00:00:00", "updated_at": "2025-01-01T00:00:00",
     "source_ids": []},
]

FAKE_EDGES = [
    {"source_id": "n1", "target_id": "n2", "weight": 0.7,
     "created_at": "2025-01-01T00:00:00", "last_activated": None,
     "activation_count": 0},
    {"source_id": "n1", "target_id": "n3", "weight": 0.4,
     "created_at": "2025-01-01T00:00:00", "last_activated": None,
     "activation_count": 0},
]

FAKE_RECALL = {
    "recall_id": "r1", "created_at": "2025-01-01T00:00:00",
    "session_id": "test-session-1",
    "results": [
        {"node_id": "n1", "similarity": 0.9, "source": "seed",
         "connected_via": None, "reflection": "U", "type": "vibe",
         "text": "test vibe one"},
    ],
}

FAKE_MAIN_STATS = {
    "total_documents": 100, "total_subchunks": 500, "total_user_inputs": 80,
    "collection_name": "conversations", "subchunk_collection_name": "subchunks",
    "user_input_collection_name": "user_inputs",
    "queue_pending": 0, "queue_failed": 0,
    "graph_nodes": 3, "graph_vibes": 2, "graph_details": 1,
    "graph_edges": 2, "graph_activated_edges": 0,
}


# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------

class MockTransport(httpx.AsyncBaseTransport):
    """Fake HTTP transport that returns canned responses for server endpoints."""

    def __init__(self):
        self.routes = {}

    def add(self, method: str, path: str, json_body, status_code: int = 200):
        self.routes[(method, path)] = (json_body, status_code)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        # Strip query string for route matching
        path_no_query = path.split("?")[0]
        key = (request.method, path_no_query)
        if key in self.routes:
            body, status = self.routes[key]
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def mock_transport():
    transport = MockTransport()

    # /graph/full
    transport.add("GET", "/graph/full",
                  {"nodes": FAKE_NODES, "edges": FAKE_EDGES})

    # /graph/nodes (list)
    transport.add("GET", "/graph/nodes",
                  {"nodes": FAKE_NODES, "total": 3})

    # /graph/nodes/n1
    transport.add("GET", "/graph/nodes/n1",
                  {"node": FAKE_NODES[0], "edges": FAKE_EDGES})

    # /graph/nodes/nonexistent
    transport.add("GET", "/graph/nodes/nonexistent",
                  {"error": "Node not found"}, status_code=404)

    # /graph/nodes/n1/neighbors
    transport.add("GET", "/graph/nodes/n1/neighbors",
                  {"neighbors": [
                      {"node": FAKE_NODES[1], "edge": FAKE_EDGES[0]},
                      {"node": FAKE_NODES[2], "edge": FAKE_EDGES[1]},
                  ]})

    # /graph/nodes/n2/neighbors
    transport.add("GET", "/graph/nodes/n2/neighbors",
                  {"neighbors": [
                      {"node": FAKE_NODES[0], "edge": FAKE_EDGES[0]},
                  ]})

    # /graph/nodes/nonexistent/neighbors
    transport.add("GET", "/graph/nodes/nonexistent/neighbors",
                  {"error": "Node not found"}, status_code=404)

    # /graph/recalls
    transport.add("GET", "/graph/recalls",
                  {"recalls": [FAKE_RECALL]})

    # /graph/reflections
    transport.add("GET", "/graph/reflections", {"U": 1, "I": 1})

    # /graph/reflection-timeline
    transport.add("GET", "/graph/reflection-timeline", [
        {"bucket": "2025-01-01T00:00:00", "U": 2, "I": 1, "N": 0, "D": 0, "M": 0},
        {"bucket": "2025-01-01T01:00:00", "U": 0, "I": 0, "N": 3, "D": 1, "M": 0},
    ])

    # /stats
    transport.add("GET", "/stats", FAKE_MAIN_STATS)

    return transport


@pytest.fixture
def client(mock_transport):
    import browse
    original_client = browse.http_client
    browse.http_client = httpx.AsyncClient(
        transport=mock_transport, base_url="http://fake:8420",
    )
    c = TestClient(browse.app)
    yield c
    browse.http_client = original_client


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Memory Browser" in resp.text


# ---------------------------------------------------------------------------
# /api/graph
# ---------------------------------------------------------------------------

def test_full_graph_structure(client):
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2
    for node in data["nodes"]:
        assert "embedding" not in node


# ---------------------------------------------------------------------------
# /api/nodes
# ---------------------------------------------------------------------------

def test_list_nodes_default(client):
    resp = client.get("/api/nodes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert len(data["nodes"]) == 3


# ---------------------------------------------------------------------------
# /api/nodes/{id}
# ---------------------------------------------------------------------------

def test_get_node(client):
    resp = client.get("/api/nodes/n1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node"]["id"] == "n1"
    assert len(data["edges"]) == 2


def test_get_node_not_found(client):
    resp = client.get("/api/nodes/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/nodes/{id}/neighbors
# ---------------------------------------------------------------------------

def test_get_neighbors(client):
    resp = client.get("/api/nodes/n1/neighbors")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["neighbors"]) == 2


def test_get_neighbors_one_edge(client):
    resp = client.get("/api/nodes/n2/neighbors")
    data = resp.json()
    assert len(data["neighbors"]) == 1


def test_get_neighbors_not_found(client):
    resp = client.get("/api/nodes/nonexistent/neighbors")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/recalls
# ---------------------------------------------------------------------------

def test_recalls_endpoint(client):
    resp = client.get("/api/recalls")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["recalls"]) >= 1
    assert data["recalls"][0]["session_id"] == "test-session-1"


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

def test_stats_endpoint(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["graph"]["total_nodes"] == 3
    assert data["reflections"]["U"] == 1
    assert data["chromadb"]["total_documents"] == 100


# ---------------------------------------------------------------------------
# /api/reflection-timeline
# ---------------------------------------------------------------------------

def test_reflection_timeline(client):
    resp = client.get("/api/reflection-timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["U"] == 2
    assert data[1]["N"] == 3
