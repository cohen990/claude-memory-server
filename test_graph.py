"""Tests for graph memory layer.

Following test_server.py patterns — pytest, temp dirs, isolated DB per module.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from graph import GraphStore, DreamLog, embedding_to_blob, blob_to_embedding, EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Create a GraphStore with a temporary database."""
    db_path = str(tmp_path / "test_graph.db")
    gs = GraphStore(db_path=db_path)
    yield gs
    gs.close()


def _random_embedding(seed=None):
    """Generate a random normalized embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _similar_embedding(base, blend=0.95, seed=None):
    """Generate an embedding with cosine similarity ~blend to base.

    Uses spherical interpolation: blend*base + (1-blend)*random, renormalized.
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(EMBEDDING_DIM).astype(np.float32)
    noise /= np.linalg.norm(noise)
    mixed = blend * base + (1 - blend) * noise
    mixed /= np.linalg.norm(mixed)
    return mixed


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_embedding_roundtrip():
    vec = _random_embedding(42)
    blob = embedding_to_blob(vec)
    assert len(blob) == EMBEDDING_DIM * 4
    restored = blob_to_embedding(blob)
    np.testing.assert_array_almost_equal(vec, restored)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_created(store):
    """Tables and indexes should exist after init."""
    tables = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "nodes" in table_names
    assert "edges" in table_names


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

def test_add_and_get_node(store):
    emb = _random_embedding(1)
    node_id = store.add_node("vibe", "test vibe text", emb, source_ids=["src1", "src2"])
    assert node_id  # non-empty string

    node = store.get_node(node_id)
    assert node is not None
    assert node["type"] == "vibe"
    assert node["text"] == "test vibe text"
    assert node["source_ids"] == ["src1", "src2"]
    np.testing.assert_array_almost_equal(node["embedding"], emb, decimal=5)


def test_get_nonexistent_node(store):
    assert store.get_node("nonexistent-id") is None


def test_update_node_embedding(store):
    emb1 = _random_embedding(10)
    emb2 = _random_embedding(20)
    node_id = store.add_node("detail", "some detail", emb1)

    store.update_node_embedding(node_id, emb2)
    node = store.get_node(node_id)
    np.testing.assert_array_almost_equal(node["embedding"], emb2, decimal=5)


def test_update_node_text(store):
    emb1 = _random_embedding(30)
    emb2 = _random_embedding(31)
    node_id = store.add_node("vibe", "old text", emb1)

    store.update_node_text(node_id, "new text", emb2)
    node = store.get_node(node_id)
    assert node["text"] == "new text"
    np.testing.assert_array_almost_equal(node["embedding"], emb2, decimal=5)


def test_merge_node_embedding(store):
    base_emb = _random_embedding(40)
    node_id = store.add_node("detail", "detail text", base_emb,
                             source_ids=["s1", "s2", "s3"])

    new_emb = _random_embedding(41)
    store.merge_node_embedding(node_id, new_emb, new_source_ids=["s4"])

    node = store.get_node(node_id)
    # Weight should be 3:1 (3 existing sources)
    expected = (base_emb * 3 + new_emb) / 4
    np.testing.assert_array_almost_equal(node["embedding"], expected, decimal=4)
    assert node["source_ids"] == ["s1", "s2", "s3", "s4"]


def test_merge_node_nonexistent(store):
    """Merging into nonexistent node should be a no-op."""
    store.merge_node_embedding("nope", _random_embedding(99))


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------

def test_find_similar_above_threshold(store):
    base = _random_embedding(50)
    node_id = store.add_node("vibe", "base node", base)

    similar = _similar_embedding(base, blend=0.97, seed=51)
    result = store.find_similar(similar, threshold=0.9)
    assert result is not None
    assert result["id"] == node_id
    assert result["similarity"] > 0.9


def test_find_similar_below_threshold(store):
    base = _random_embedding(60)
    store.add_node("vibe", "base node", base)

    different = _random_embedding(61)  # random = low similarity
    result = store.find_similar(different, threshold=0.95)
    assert result is None


def test_find_similar_with_type_filter(store):
    emb = _random_embedding(70)
    store.add_node("vibe", "a vibe", emb, source_ids=["a"])
    store.add_node("detail", "a detail", _similar_embedding(emb, blend=0.98, seed=71))

    result = store.find_similar(emb, threshold=0.8, node_type="detail")
    assert result is not None
    assert result["type"] == "detail"


def test_find_similar_empty_store(store):
    result = store.find_similar(_random_embedding(80))
    assert result is None


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------

def test_add_and_get_edges(store):
    id1 = store.add_node("vibe", "node 1", _random_embedding(100))
    id2 = store.add_node("detail", "node 2", _random_embedding(101))
    store.add_edge(id1, id2, weight=0.7)

    edges = store.get_edges(id1)
    assert len(edges) == 1
    assert edges[0]["source_id"] == id1
    assert edges[0]["target_id"] == id2
    assert edges[0]["weight"] == pytest.approx(0.7)
    assert edges[0]["activation_count"] == 0

    # Should also show up when querying the target
    edges2 = store.get_edges(id2)
    assert len(edges2) == 1


def test_add_edge_duplicate_ignored(store):
    id1 = store.add_node("vibe", "n1", _random_embedding(110))
    id2 = store.add_node("vibe", "n2", _random_embedding(111))
    store.add_edge(id1, id2, weight=0.5)
    store.add_edge(id1, id2, weight=0.9)  # should be ignored

    edges = store.get_edges(id1)
    assert len(edges) == 1
    assert edges[0]["weight"] == pytest.approx(0.5)  # original weight preserved


def test_bump_edge_activation(store):
    id1 = store.add_node("vibe", "n1", _random_embedding(120))
    id2 = store.add_node("vibe", "n2", _random_embedding(121))
    store.add_edge(id1, id2)

    store.bump_edge_activation(id1, id2)
    store.bump_edge_activation(id1, id2)

    edges = store.get_edges(id1)
    assert edges[0]["activation_count"] == 2
    assert edges[0]["last_activated"] is not None


def test_get_activated_edges(store):
    id1 = store.add_node("vibe", "n1", _random_embedding(130))
    id2 = store.add_node("vibe", "n2", _random_embedding(131))
    id3 = store.add_node("vibe", "n3", _random_embedding(132))

    store.add_edge(id1, id2)
    store.add_edge(id2, id3)
    store.bump_edge_activation(id1, id2)

    activated = store.get_activated_edges()
    assert len(activated) == 1
    assert activated[0]["source_id"] == id1
    assert activated[0]["target_id"] == id2


def test_reset_activation_counts(store):
    id1 = store.add_node("vibe", "n1", _random_embedding(140))
    id2 = store.add_node("vibe", "n2", _random_embedding(141))
    store.add_edge(id1, id2)
    store.bump_edge_activation(id1, id2)

    store.reset_activation_counts()
    activated = store.get_activated_edges()
    assert len(activated) == 0


def test_update_edge_weight(store):
    id1 = store.add_node("vibe", "n1", _random_embedding(150))
    id2 = store.add_node("vibe", "n2", _random_embedding(151))
    store.add_edge(id1, id2, weight=0.5)

    store.update_edge_weight(id1, id2, 0.8)
    edges = store.get_edges(id1)
    assert edges[0]["weight"] == pytest.approx(0.8)


def test_update_edge_weight_clamped(store):
    id1 = store.add_node("vibe", "n1", _random_embedding(160))
    id2 = store.add_node("vibe", "n2", _random_embedding(161))
    store.add_edge(id1, id2)

    store.update_edge_weight(id1, id2, 1.5)
    edges = store.get_edges(id1)
    assert edges[0]["weight"] == pytest.approx(1.0)

    store.update_edge_weight(id1, id2, -0.3)
    edges = store.get_edges(id1)
    assert edges[0]["weight"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search_basic(store):
    """Search should return nodes sorted by similarity."""
    target = _random_embedding(200)
    close = _similar_embedding(target, blend=0.95, seed=201)
    far = _random_embedding(202)

    store.add_node("vibe", "close node", close)
    store.add_node("vibe", "far node", far)

    results = store.search(target, k=2, expand_neighbors=False)
    assert len(results) > 0
    assert results[0]["text"] == "close node"
    assert results[0]["source"] == "seed"


def test_search_with_neighbor_expansion(store):
    """Neighbors connected by edges should appear in results."""
    seed_emb = _random_embedding(210)
    neighbor_emb = _random_embedding(211)  # dissimilar to query

    # Add many decoy nodes so the neighbor doesn't land in top k*2 seeds
    for i in range(20):
        store.add_node("vibe", f"decoy {i}", _random_embedding(2100 + i))

    seed_id = store.add_node("vibe", "seed node", seed_emb)
    neighbor_id = store.add_node("detail", "neighbor detail", neighbor_emb)
    store.add_edge(seed_id, neighbor_id, weight=0.9)

    # Search with k=1 so only 2 seeds are taken (k*2=2), neighbor unlikely among them
    query = _similar_embedding(seed_emb, blend=0.99, seed=212)
    results = store.search(query, k=3, expand_neighbors=True)
    result_ids = {r["id"] for r in results}
    assert seed_id in result_ids
    assert neighbor_id in result_ids

    neighbor_result = next(r for r in results if r["id"] == neighbor_id)
    assert neighbor_result["source"] == "neighbor"
    assert "edge_weight" in neighbor_result
    assert neighbor_result["connected_via"] == seed_id


def test_search_does_not_activate_edges(store):
    """Search is a pure read — should NOT increment activation counts."""
    id1 = store.add_node("vibe", "node1", _random_embedding(220))
    id2 = store.add_node("detail", "node2", _random_embedding(221))
    store.add_edge(id1, id2, weight=0.7)

    # Search near node1 to trigger expansion
    store.search(_similar_embedding(_random_embedding(220), blend=0.99, seed=222),
                 k=5, expand_neighbors=True)

    activated = store.get_activated_edges()
    assert len(activated) == 0


def test_search_empty_store(store):
    results = store.search(_random_embedding(230))
    assert results == []


def test_search_returns_hydrated_results(store):
    emb = _random_embedding(240)
    store.add_node("vibe", "hydrated text", emb, source_ids=["x"])

    results = store.search(emb, k=1, expand_neighbors=False)
    assert len(results) == 1
    assert results[0]["text"] == "hydrated text"
    assert results[0]["type"] == "vibe"
    assert results[0]["source_ids"] == ["x"]
    assert "similarity" in results[0]
    assert "score" in results[0]


# ---------------------------------------------------------------------------
# Cache consistency
# ---------------------------------------------------------------------------

def test_cache_consistent_after_add(store):
    """Cache should reflect newly added nodes."""
    emb = _random_embedding(300)
    node_id = store.add_node("vibe", "cached", emb)

    assert node_id in store._id_to_idx
    idx = store._id_to_idx[node_id]
    assert idx < store._embeddings.shape[0]


def test_cache_consistent_after_update(store):
    """Cache should reflect updated embeddings."""
    emb1 = _random_embedding(310)
    emb2 = _random_embedding(311)
    node_id = store.add_node("vibe", "will update", emb1)

    store.update_node_embedding(node_id, emb2)

    # Search should find the node with the new embedding
    results = store.search(emb2, k=1, expand_neighbors=False)
    assert len(results) == 1
    assert results[0]["id"] == node_id


def test_cache_rebuild(store):
    """_rebuild_cache should produce the same state as incremental updates."""
    emb1 = _random_embedding(320)
    emb2 = _random_embedding(321)
    store.add_node("vibe", "node a", emb1)
    store.add_node("detail", "node b", emb2)

    # Save current state
    old_ids = list(store._node_ids)
    old_shape = store._embeddings.shape

    # Rebuild
    store._rebuild_cache()

    assert store._node_ids == old_ids
    assert store._embeddings.shape == old_shape


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats(store):
    store.add_node("vibe", "v1", _random_embedding(400))
    store.add_node("vibe", "v2", _random_embedding(401))
    store.add_node("detail", "d1", _random_embedding(402))

    id1 = store._node_ids[0]
    id2 = store._node_ids[1]
    store.add_edge(id1, id2)

    s = store.stats()
    assert s["total_nodes"] == 3
    assert s["nodes_by_type"]["vibe"] == 2
    assert s["nodes_by_type"]["detail"] == 1
    assert s["total_edges"] == 1
    assert s["activated_edges"] == 0


def test_stats_empty(store):
    s = store.stats()
    assert s["total_nodes"] == 0
    assert s["total_edges"] == 0


# ---------------------------------------------------------------------------
# Recall tracking
# ---------------------------------------------------------------------------

def test_create_recall(store):
    """create_recall should store recall + results and return an ID."""
    emb = _random_embedding(500)
    id1 = store.add_node("vibe", "node1", _random_embedding(501))
    id2 = store.add_node("detail", "node2", _random_embedding(502))

    results = [
        {"id": id1, "similarity": 0.9, "source": "seed"},
        {"id": id2, "similarity": 0.7, "source": "neighbor", "connected_via": id1},
    ]
    recall_id = store.create_recall(emb, results)
    assert recall_id  # non-empty

    # Verify stored
    row = store._conn.execute(
        "SELECT COUNT(*) FROM recall_results WHERE recall_id = ?",
        (recall_id,),
    ).fetchone()
    assert row[0] == 2


def test_reflect_on_recall(store):
    """reflect_on_recall should set reflection codes by position."""
    id1 = store.add_node("vibe", "n1", _random_embedding(510))
    id2 = store.add_node("vibe", "n2", _random_embedding(511))

    recall_id = store.create_recall(_random_embedding(512), [
        {"id": id1, "similarity": 0.8, "source": "seed"},
        {"id": id2, "similarity": 0.6, "source": "seed"},
    ])

    store.reflect_on_recall(recall_id, ["U", "N"])

    rows = store._conn.execute(
        "SELECT position, reflection FROM recall_results WHERE recall_id = ? ORDER BY position",
        (recall_id,),
    ).fetchall()
    assert rows[0] == (0, "U")
    assert rows[1] == (1, "N")


def test_get_reflected_recalls(store):
    """get_reflected_recalls returns only recalls with reflections."""
    id1 = store.add_node("vibe", "n1", _random_embedding(520))
    id2 = store.add_node("vibe", "n2", _random_embedding(521))

    # Create two recalls, only rate one
    emb1 = _random_embedding(522)
    r1 = store.create_recall(emb1, [
        {"id": id1, "similarity": 0.9, "source": "seed"},
    ])
    store.create_recall(_random_embedding(523), [
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])

    store.reflect_on_recall(r1, ["U"])

    reflected = store.get_reflected_recalls()
    assert len(reflected) == 1
    assert reflected[0]["recall_id"] == r1
    assert reflected[0]["results"][0]["reflection"] == "U"
    assert reflected[0]["results"][0]["node_id"] == id1
    np.testing.assert_array_almost_equal(reflected[0]["query_embedding"], emb1, decimal=5)


def test_get_reflected_recalls_empty(store):
    """get_reflected_recalls returns empty list when nothing is reflected."""
    assert store.get_reflected_recalls() == []


def test_clear_reflected_recalls(store):
    """clear_reflected_recalls marks reflected recalls as dreamed but leaves unreflected ones."""
    id1 = store.add_node("vibe", "n1", _random_embedding(530))
    id2 = store.add_node("vibe", "n2", _random_embedding(531))

    r1 = store.create_recall(_random_embedding(532), [
        {"id": id1, "similarity": 0.9, "source": "seed"},
    ])
    r2 = store.create_recall(_random_embedding(533), [
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])

    store.reflect_on_recall(r1, ["M"])
    store.clear_reflected_recalls()

    # r1 should be marked as dreamed
    row = store._conn.execute(
        "SELECT dreamed_at FROM recalls WHERE id = ?", (r1,)
    ).fetchone()
    assert row is not None
    assert row[0] is not None

    # r2 should remain un-dreamed
    row = store._conn.execute(
        "SELECT dreamed_at FROM recalls WHERE id = ?", (r2,)
    ).fetchone()
    assert row is not None
    assert row[0] is None


def test_clear_processed_recalls(store):
    """clear_processed_recalls marks all recalls as dreamed, not deleted."""
    id1 = store.add_node("vibe", "n1", _random_embedding(540))
    id2 = store.add_node("vibe", "n2", _random_embedding(541))

    r1 = store.create_recall(_random_embedding(542), [
        {"id": id1, "similarity": 0.9, "source": "seed"},
    ])
    r2 = store.create_recall(_random_embedding(543), [
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])

    # Rate only r1
    store.reflect_on_recall(r1, ["M"])

    store.clear_processed_recalls()

    # Both should still exist but have dreamed_at set
    for rid in (r1, r2):
        row = store._conn.execute(
            "SELECT dreamed_at FROM recalls WHERE id = ?", (rid,)
        ).fetchone()
        assert row is not None
        assert row[0] is not None  # dreamed_at is set

    # recall_results should still be intact
    row = store._conn.execute("SELECT COUNT(*) FROM recall_results").fetchone()
    assert row[0] == 2


def test_schema_has_recall_tables(store):
    """Recall tables should exist after init."""
    tables = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "recalls" in table_names
    assert "recall_results" in table_names


# ---------------------------------------------------------------------------
# session_id on recalls
# ---------------------------------------------------------------------------

def test_create_recall_with_session_id(store):
    """create_recall should store session_id when provided."""
    id1 = store.add_node("vibe", "n1", _random_embedding(600))
    recall_id = store.create_recall(
        _random_embedding(601),
        [{"id": id1, "similarity": 0.9, "source": "seed"}],
        session_id="sess-abc-123",
    )

    row = store._conn.execute(
        "SELECT session_id FROM recalls WHERE id = ?", (recall_id,)
    ).fetchone()
    assert row[0] == "sess-abc-123"


def test_create_recall_without_session_id(store):
    """create_recall should store NULL session_id when not provided."""
    id1 = store.add_node("vibe", "n1", _random_embedding(610))
    recall_id = store.create_recall(
        _random_embedding(611),
        [{"id": id1, "similarity": 0.9, "source": "seed"}],
    )

    row = store._conn.execute(
        "SELECT session_id FROM recalls WHERE id = ?", (recall_id,)
    ).fetchone()
    assert row[0] is None


# ---------------------------------------------------------------------------
# list_recalls
# ---------------------------------------------------------------------------

def test_list_recalls_by_session(store):
    """list_recalls should return recalls filtered by session_id."""
    id1 = store.add_node("vibe", "vibe text", _random_embedding(620))
    id2 = store.add_node("detail", "detail text", _random_embedding(621))

    # Create recall for session A
    r1 = store.create_recall(
        _random_embedding(622),
        [{"id": id1, "similarity": 0.9, "source": "seed"}],
        session_id="session-A",
    )
    # Create recall for session B
    store.create_recall(
        _random_embedding(623),
        [{"id": id2, "similarity": 0.8, "source": "seed"}],
        session_id="session-B",
    )

    recalls = store.list_recalls(session_id="session-A", limit=10)
    assert len(recalls) == 1
    assert recalls[0]["recall_id"] == r1
    assert recalls[0]["session_id"] == "session-A"


def test_list_recalls_joins_node_data(store):
    """list_recalls results should include node type and text."""
    id1 = store.add_node("vibe", "my vibe text", _random_embedding(630))

    store.create_recall(
        _random_embedding(631),
        [{"id": id1, "similarity": 0.85, "source": "seed"}],
        session_id="session-X",
    )

    recalls = store.list_recalls(session_id="session-X")
    assert len(recalls) == 1
    r = recalls[0]["results"][0]
    assert r["type"] == "vibe"
    assert r["text"] == "my vibe text"
    assert r["similarity"] == pytest.approx(0.85)
    assert r["source"] == "seed"


def test_list_recalls_includes_reflections(store):
    """list_recalls should show reflections after reflect_on_recall is called."""
    id1 = store.add_node("detail", "detail node", _random_embedding(640))
    id2 = store.add_node("vibe", "vibe node", _random_embedding(641))

    recall_id = store.create_recall(
        _random_embedding(642),
        [
            {"id": id1, "similarity": 0.9, "source": "seed"},
            {"id": id2, "similarity": 0.7, "source": "neighbor", "connected_via": id1},
        ],
        session_id="session-Y",
    )
    store.reflect_on_recall(recall_id, ["U", "N"])

    recalls = store.list_recalls(session_id="session-Y")
    assert len(recalls) == 1
    assert recalls[0]["results"][0]["reflection"] == "U"
    assert recalls[0]["results"][1]["reflection"] == "N"


def test_list_recalls_limit(store):
    """list_recalls should respect the limit parameter."""
    id1 = store.add_node("vibe", "n1", _random_embedding(650))

    for i in range(5):
        store.create_recall(
            _random_embedding(660 + i),
            [{"id": id1, "similarity": 0.8, "source": "seed"}],
            session_id="session-Z",
        )

    recalls = store.list_recalls(session_id="session-Z", limit=2)
    assert len(recalls) == 2


def test_list_recalls_no_session_filter(store):
    """list_recalls without session_id returns most recent across all sessions."""
    id1 = store.add_node("vibe", "n1", _random_embedding(670))

    store.create_recall(
        _random_embedding(671),
        [{"id": id1, "similarity": 0.9, "source": "seed"}],
        session_id="sess-1",
    )
    store.create_recall(
        _random_embedding(672),
        [{"id": id1, "similarity": 0.8, "source": "seed"}],
        session_id="sess-2",
    )

    recalls = store.list_recalls(limit=10)
    assert len(recalls) == 2


def test_list_recalls_empty(store):
    """list_recalls returns empty list when no recalls exist."""
    assert store.list_recalls(session_id="nonexistent") == []


# ---------------------------------------------------------------------------
# list_nodes
# ---------------------------------------------------------------------------

def test_list_nodes_basic(store):
    """list_nodes returns nodes with expected fields, no embedding blob."""
    store.add_node("vibe", "a vibe", _random_embedding(700))
    store.add_node("detail", "a detail", _random_embedding(701))

    result = store.list_nodes()
    assert result["total"] == 2
    assert len(result["nodes"]) == 2
    for node in result["nodes"]:
        assert "id" in node
        assert "type" in node
        assert "text" in node
        assert "created_at" in node
        assert "updated_at" in node
        assert "source_ids" in node
        assert "embedding" not in node


def test_list_nodes_type_filter(store):
    """type='vibe' only returns vibes."""
    store.add_node("vibe", "v1", _random_embedding(710))
    store.add_node("vibe", "v2", _random_embedding(711))
    store.add_node("detail", "d1", _random_embedding(712))

    result = store.list_nodes(node_type="vibe")
    assert result["total"] == 2
    assert all(n["type"] == "vibe" for n in result["nodes"])


def test_list_nodes_pagination(store):
    """limit/offset work, total count is correct."""
    for i in range(5):
        store.add_node("detail", f"node {i}", _random_embedding(720 + i))

    page1 = store.list_nodes(limit=2, offset=0)
    assert page1["total"] == 5
    assert len(page1["nodes"]) == 2

    page2 = store.list_nodes(limit=2, offset=2)
    assert page2["total"] == 5
    assert len(page2["nodes"]) == 2

    # IDs should not overlap
    ids1 = {n["id"] for n in page1["nodes"]}
    ids2 = {n["id"] for n in page2["nodes"]}
    assert ids1.isdisjoint(ids2)


def test_list_nodes_empty(store):
    """Returns {nodes: [], total: 0} on empty store."""
    result = store.list_nodes()
    assert result == {"nodes": [], "total": 0}


# ---------------------------------------------------------------------------
# get_full_graph
# ---------------------------------------------------------------------------

def test_get_full_graph(store):
    """Returns both nodes and edges, no embedding blobs."""
    id1 = store.add_node("vibe", "n1", _random_embedding(730))
    id2 = store.add_node("detail", "n2", _random_embedding(731))
    store.add_edge(id1, id2, weight=0.6)

    result = store.get_full_graph()
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    for node in result["nodes"]:
        assert "embedding" not in node
    assert result["edges"][0]["weight"] == pytest.approx(0.6)


def test_get_full_graph_respects_limits(store):
    """node_limit/edge_limit are honored."""
    ids = []
    for i in range(5):
        ids.append(store.add_node("vibe", f"n{i}", _random_embedding(740 + i)))
    for i in range(4):
        store.add_edge(ids[i], ids[i + 1])

    result = store.get_full_graph(node_limit=2, edge_limit=1)
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1


def test_get_full_graph_empty(store):
    """Empty store returns {nodes: [], edges: []}."""
    result = store.get_full_graph()
    assert result == {"nodes": [], "edges": []}


# ---------------------------------------------------------------------------
# compute_layout
# ---------------------------------------------------------------------------

def test_compute_layout_basic(store, tmp_path):
    """compute_layout returns positions for all nodes."""
    id1 = store.add_node("vibe", "n1", _random_embedding(780))
    id2 = store.add_node("detail", "n2", _random_embedding(781))
    store.add_edge(id1, id2, weight=0.6)

    cache_path = str(tmp_path / "layout.json")
    positions = store.compute_layout(cache_path=cache_path)

    assert id1 in positions
    assert id2 in positions
    assert "x" in positions[id1]
    assert "y" in positions[id1]

    # Cache file should exist
    import json
    with open(cache_path) as f:
        cached = json.load(f)
    assert cached == positions


def test_compute_layout_disconnected_components(store, tmp_path):
    """Disconnected components get tiled without overlapping."""
    # Component 1
    a1 = store.add_node("vibe", "a1", _random_embedding(790))
    a2 = store.add_node("vibe", "a2", _random_embedding(791))
    store.add_edge(a1, a2)

    # Component 2 (disconnected)
    b1 = store.add_node("detail", "b1", _random_embedding(792))
    b2 = store.add_node("detail", "b2", _random_embedding(793))
    store.add_edge(b1, b2)

    cache_path = str(tmp_path / "layout.json")
    positions = store.compute_layout(cache_path=cache_path)

    assert len(positions) == 4
    # All four nodes should have positions
    for nid in [a1, a2, b1, b2]:
        assert nid in positions


def test_compute_layout_empty(store, tmp_path):
    """Empty graph produces empty positions."""
    cache_path = str(tmp_path / "layout.json")
    positions = store.compute_layout(cache_path=cache_path)
    assert positions == {}


def test_compute_layout_isolated_nodes(store, tmp_path):
    """Nodes with no edges still get positions."""
    id1 = store.add_node("vibe", "isolated1", _random_embedding(795))
    id2 = store.add_node("detail", "isolated2", _random_embedding(796))

    cache_path = str(tmp_path / "layout.json")
    positions = store.compute_layout(cache_path=cache_path)

    assert id1 in positions
    assert id2 in positions


def test_compute_layout_deterministic(store, tmp_path):
    """Layout with seed=42 should be deterministic across runs."""
    id1 = store.add_node("vibe", "n1", _random_embedding(797))
    id2 = store.add_node("detail", "n2", _random_embedding(798))
    store.add_edge(id1, id2)

    cache1 = str(tmp_path / "layout1.json")
    cache2 = str(tmp_path / "layout2.json")
    pos1 = store.compute_layout(cache_path=cache1)
    pos2 = store.compute_layout(cache_path=cache2)

    assert pos1 == pos2


# ---------------------------------------------------------------------------
# get_full_graph with positions
# ---------------------------------------------------------------------------

def test_get_full_graph_includes_positions(store, tmp_path):
    """get_full_graph includes cached positions when available."""
    from graph import LAYOUT_CACHE_PATH
    id1 = store.add_node("vibe", "n1", _random_embedding(850))
    id2 = store.add_node("detail", "n2", _random_embedding(851))
    store.add_edge(id1, id2)

    # Compute layout to the default cache path
    store.compute_layout()

    result = store.get_full_graph()
    for node in result["nodes"]:
        assert "position" in node
        assert node["position"] is not None
        assert "x" in node["position"]
        assert "y" in node["position"]


def test_get_full_graph_null_positions_without_cache(store, tmp_path):
    """get_full_graph returns null positions when no cache exists."""
    # Use a store that won't have a cache file
    id1 = store.add_node("vibe", "n1", _random_embedding(860))

    result = store.get_full_graph()
    for node in result["nodes"]:
        assert "position" in node
        assert node["position"] is None


# ---------------------------------------------------------------------------
# reflection_distribution
# ---------------------------------------------------------------------------

def test_reflection_distribution(store):
    """Returns correct counts per reflection code."""
    id1 = store.add_node("vibe", "n1", _random_embedding(750))
    id2 = store.add_node("vibe", "n2", _random_embedding(751))

    r1 = store.create_recall(_random_embedding(752), [
        {"id": id1, "similarity": 0.9, "source": "seed"},
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])
    store.reflect_on_recall(r1, ["U", "N"])

    r2 = store.create_recall(_random_embedding(753), [
        {"id": id1, "similarity": 0.7, "source": "seed"},
    ])
    store.reflect_on_recall(r2, ["U"])

    dist = store.reflection_distribution()
    assert dist["U"] == 2
    assert dist["N"] == 1


def test_reflection_distribution_empty(store):
    """Returns empty dict when nothing reflected."""
    assert store.reflection_distribution() == {}


def test_reflection_timeline(store):
    """Returns hourly-bucketed reflection counts."""
    id1 = store.add_node("vibe", "n1", _random_embedding(760))
    id2 = store.add_node("detail", "n2", _random_embedding(761))

    r1 = store.create_recall(_random_embedding(762), [
        {"id": id1, "similarity": 0.9, "source": "seed"},
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])
    store.reflect_on_recall(r1, ["U", "I"])

    r2 = store.create_recall(_random_embedding(763), [
        {"id": id1, "similarity": 0.7, "source": "seed"},
    ])
    store.reflect_on_recall(r2, ["N"])

    timeline = store.reflection_timeline()
    assert len(timeline) >= 1
    # All created in the same second, so one bucket
    bucket = timeline[0]
    assert "bucket" in bucket
    assert bucket["U"] == 1
    assert bucket["I"] == 1
    assert bucket["N"] == 1
    assert bucket["D"] == 0
    assert bucket["M"] == 0


def test_reflection_timeline_empty(store):
    """Returns empty list when no reflections exist."""
    assert store.reflection_timeline() == []


# ---------------------------------------------------------------------------
# Reflections survive dreaming
# ---------------------------------------------------------------------------

def test_reflections_survive_clear_processed_recalls(store):
    """Reflection timeline/distribution should still work after clear_processed_recalls."""
    id1 = store.add_node("vibe", "n1", _random_embedding(770))
    id2 = store.add_node("detail", "n2", _random_embedding(771))

    r1 = store.create_recall(_random_embedding(772), [
        {"id": id1, "similarity": 0.9, "source": "seed"},
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])
    store.reflect_on_recall(r1, ["U", "I"])

    # Mark all recalls as dreamed (as dream reconsolidation does)
    store.clear_processed_recalls()

    # Recall data still exists
    count = store._conn.execute("SELECT COUNT(*) FROM recall_results").fetchone()[0]
    assert count == 2

    # Reflection data persists for charts
    dist = store.reflection_distribution()
    assert dist["U"] == 1
    assert dist["I"] == 1

    timeline = store.reflection_timeline()
    assert len(timeline) >= 1
    assert timeline[0]["U"] == 1
    assert timeline[0]["I"] == 1

    # But get_reflected_recalls returns nothing (already dreamed)
    assert store.get_reflected_recalls() == []


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# get_recalled_nodes_for_sessions
# ---------------------------------------------------------------------------

def test_get_recalled_nodes_basic(store):
    """Returns nodes that were recalled during specified sessions."""
    id1 = store.add_node("vibe", "functional programming", _random_embedding(800))
    id2 = store.add_node("detail", "uses nomic embeddings", _random_embedding(801))

    store.create_recall(
        _random_embedding(802),
        [{"id": id1, "similarity": 0.9, "source": "seed"}],
        session_id="sess-A",
    )
    store.create_recall(
        _random_embedding(803),
        [{"id": id2, "similarity": 0.8, "source": "seed"}],
        session_id="sess-B",
    )

    results = store.get_recalled_nodes_for_sessions(["sess-A"])
    assert len(results) == 1
    assert results[0]["id"] == id1
    assert results[0]["type"] == "vibe"
    assert results[0]["text"] == "functional programming"


def test_get_recalled_nodes_multiple_sessions(store):
    """Collects nodes across multiple sessions."""
    id1 = store.add_node("vibe", "v1", _random_embedding(810))
    id2 = store.add_node("detail", "d1", _random_embedding(811))

    store.create_recall(
        _random_embedding(812),
        [{"id": id1, "similarity": 0.9, "source": "seed"}],
        session_id="sess-1",
    )
    store.create_recall(
        _random_embedding(813),
        [{"id": id2, "similarity": 0.8, "source": "seed"}],
        session_id="sess-2",
    )

    results = store.get_recalled_nodes_for_sessions(["sess-1", "sess-2"])
    result_ids = {r["id"] for r in results}
    assert result_ids == {id1, id2}


def test_get_recalled_nodes_deduplicates(store):
    """Same node recalled in multiple sessions appears only once."""
    id1 = store.add_node("vibe", "shared vibe", _random_embedding(820))

    store.create_recall(
        _random_embedding(821),
        [{"id": id1, "similarity": 0.9, "source": "seed"}],
        session_id="sess-X",
    )
    store.create_recall(
        _random_embedding(822),
        [{"id": id1, "similarity": 0.85, "source": "seed"}],
        session_id="sess-Y",
    )

    results = store.get_recalled_nodes_for_sessions(["sess-X", "sess-Y"])
    assert len(results) == 1
    assert results[0]["id"] == id1


def test_get_recalled_nodes_empty_sessions(store):
    """Returns empty list for empty session_ids input."""
    assert store.get_recalled_nodes_for_sessions([]) == []


def test_get_recalled_nodes_no_recalls(store):
    """Returns empty list when sessions have no recalls."""
    results = store.get_recalled_nodes_for_sessions(["nonexistent-session"])
    assert results == []


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migrate_adds_session_id(tmp_path):
    """Migration should add session_id column to an existing DB without it."""
    db_path = str(tmp_path / "migrate_test.db")

    # Create a DB with the old schema (no session_id)
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE recalls (
            id TEXT PRIMARY KEY,
            query_embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE recall_results (
            recall_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            node_id TEXT NOT NULL,
            similarity REAL NOT NULL,
            source TEXT NOT NULL,
            connected_via TEXT,
            rating TEXT,
            PRIMARY KEY (recall_id, position)
        );
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_ids TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            last_activated TEXT,
            activation_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (source_id, target_id)
        );
    """)
    conn.commit()
    conn.close()

    # Open with GraphStore — migration should add session_id and rename rating→reflection
    gs = GraphStore(db_path=db_path)
    recall_cols = {
        row[1] for row in
        gs._conn.execute("PRAGMA table_info(recalls)").fetchall()
    }
    assert "session_id" in recall_cols

    rr_cols = {
        row[1] for row in
        gs._conn.execute("PRAGMA table_info(recall_results)").fetchall()
    }
    assert "reflection" in rr_cols
    assert "rating" not in rr_cols
    gs.close()


# ---------------------------------------------------------------------------
# DreamLog
# ---------------------------------------------------------------------------

@pytest.fixture
def dream_log(store):
    """Create a DreamLog backed by the test store."""
    return DreamLog(store)


def test_dream_schema_created(store):
    """dream_runs and dream_operations tables should exist."""
    tables = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "dream_runs" in table_names
    assert "dream_operations" in table_names


def test_start_run(dream_log):
    """start_run inserts a row and returns a UUID."""
    run_id = dream_log.start_run("consolidate")
    assert run_id
    assert len(run_id) == 36  # UUID format

    row = dream_log._graph_store._conn.execute(
        "SELECT type, started_at, finished_at FROM dream_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert row[0] == "consolidate"
    assert row[1] is not None
    assert row[2] is None  # not finished yet


def test_log_operation(dream_log):
    """log_operation inserts operations correctly."""
    run_id = dream_log.start_run("consolidate")
    dream_log.log_operation(run_id, "node_created", "node-abc", "vibe",
                            {"text": "some text", "source_count": 3})
    dream_log.log_operation(run_id, "edge_created", None, None,
                            {"source_id": "a", "target_id": "b", "weight": 0.5})

    ops = dream_log.get_run_operations(run_id)
    assert len(ops) == 2
    assert ops[0]["operation"] == "node_created"
    assert ops[0]["node_id"] == "node-abc"
    assert ops[0]["node_type"] == "vibe"
    assert ops[0]["detail"]["text"] == "some text"
    assert ops[1]["operation"] == "edge_created"
    assert ops[1]["node_id"] is None


def test_finish_run(dream_log):
    """finish_run updates counts and finished_at."""
    run_id = dream_log.start_run("reconsolidate")
    dream_log.finish_run(run_id, edges_adjusted=5, nodes_resynthesized=2)

    runs = dream_log.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["id"] == run_id
    assert r["type"] == "reconsolidate"
    assert r["finished_at"] is not None
    assert r["edges_adjusted"] == 5
    assert r["nodes_resynthesized"] == 2
    assert r["error"] is None


def test_finish_run_with_error(dream_log):
    """finish_run records error string."""
    run_id = dream_log.start_run("consolidate")
    dream_log.finish_run(run_id, error="something broke", chunks_processed=10)

    runs = dream_log.list_runs()
    assert runs[0]["error"] == "something broke"
    assert runs[0]["chunks_processed"] == 10


def test_list_runs_ordering(dream_log):
    """list_runs returns newest first."""
    r1 = dream_log.start_run("consolidate")
    dream_log.finish_run(r1)
    r2 = dream_log.start_run("reconsolidate")
    dream_log.finish_run(r2)

    runs = dream_log.list_runs()
    assert len(runs) == 2
    # r2 was started after r1, so it comes first
    assert runs[0]["id"] == r2
    assert runs[1]["id"] == r1


def test_list_runs_limit(dream_log):
    """list_runs respects the limit parameter."""
    for _ in range(5):
        rid = dream_log.start_run("consolidate")
        dream_log.finish_run(rid)

    runs = dream_log.list_runs(limit=2)
    assert len(runs) == 2


def test_list_runs_empty(dream_log):
    """list_runs returns empty list when no runs exist."""
    assert dream_log.list_runs() == []


def test_get_run_operations_empty(dream_log):
    """get_run_operations returns empty list for run with no ops."""
    run_id = dream_log.start_run("consolidate")
    assert dream_log.get_run_operations(run_id) == []


def test_get_run_operations_nonexistent(dream_log):
    """get_run_operations returns empty list for nonexistent run."""
    assert dream_log.get_run_operations("nonexistent-id") == []
