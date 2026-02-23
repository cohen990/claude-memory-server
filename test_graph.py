"""Tests for graph memory layer.

Following test_server.py patterns — pytest, temp dirs, isolated DB per module.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from graph import GraphStore, embedding_to_blob, blob_to_embedding, EMBEDDING_DIM


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
    tables = store.conn.execute(
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
    row = store.conn.execute(
        "SELECT COUNT(*) FROM recall_results WHERE recall_id = ?",
        (recall_id,),
    ).fetchone()
    assert row[0] == 2


def test_rate_recall(store):
    """rate_recall should set rating codes by position."""
    id1 = store.add_node("vibe", "n1", _random_embedding(510))
    id2 = store.add_node("vibe", "n2", _random_embedding(511))

    recall_id = store.create_recall(_random_embedding(512), [
        {"id": id1, "similarity": 0.8, "source": "seed"},
        {"id": id2, "similarity": 0.6, "source": "seed"},
    ])

    store.rate_recall(recall_id, ["U", "N"])

    rows = store.conn.execute(
        "SELECT position, rating FROM recall_results WHERE recall_id = ? ORDER BY position",
        (recall_id,),
    ).fetchall()
    assert rows[0] == (0, "U")
    assert rows[1] == (1, "N")


def test_get_rated_recalls(store):
    """get_rated_recalls returns only recalls with ratings."""
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

    store.rate_recall(r1, ["U"])

    rated = store.get_rated_recalls()
    assert len(rated) == 1
    assert rated[0]["recall_id"] == r1
    assert rated[0]["results"][0]["rating"] == "U"
    assert rated[0]["results"][0]["node_id"] == id1
    np.testing.assert_array_almost_equal(rated[0]["query_embedding"], emb1, decimal=5)


def test_get_rated_recalls_empty(store):
    """get_rated_recalls returns empty list when nothing is rated."""
    assert store.get_rated_recalls() == []


def test_clear_rated_recalls(store):
    """clear_rated_recalls deletes rated recalls but leaves unrated ones."""
    id1 = store.add_node("vibe", "n1", _random_embedding(530))
    id2 = store.add_node("vibe", "n2", _random_embedding(531))

    r1 = store.create_recall(_random_embedding(532), [
        {"id": id1, "similarity": 0.9, "source": "seed"},
    ])
    r2 = store.create_recall(_random_embedding(533), [
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])

    store.rate_recall(r1, ["M"])
    store.clear_rated_recalls()

    # r1 should be gone
    row = store.conn.execute(
        "SELECT COUNT(*) FROM recalls WHERE id = ?", (r1,)
    ).fetchone()
    assert row[0] == 0

    # r2 should remain
    row = store.conn.execute(
        "SELECT COUNT(*) FROM recalls WHERE id = ?", (r2,)
    ).fetchone()
    assert row[0] == 1


def test_clear_processed_recalls(store):
    """clear_processed_recalls deletes ALL recalls — rated and unrated."""
    id1 = store.add_node("vibe", "n1", _random_embedding(540))
    id2 = store.add_node("vibe", "n2", _random_embedding(541))

    r1 = store.create_recall(_random_embedding(542), [
        {"id": id1, "similarity": 0.9, "source": "seed"},
    ])
    r2 = store.create_recall(_random_embedding(543), [
        {"id": id2, "similarity": 0.8, "source": "seed"},
    ])

    # Rate only r1
    store.rate_recall(r1, ["M"])

    store.clear_processed_recalls()

    # Both should be gone
    for rid in (r1, r2):
        row = store.conn.execute(
            "SELECT COUNT(*) FROM recalls WHERE id = ?", (rid,)
        ).fetchone()
        assert row[0] == 0

    # recall_results should be empty too
    row = store.conn.execute("SELECT COUNT(*) FROM recall_results").fetchone()
    assert row[0] == 0


def test_schema_has_recall_tables(store):
    """Recall tables should exist after init."""
    tables = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "recalls" in table_names
    assert "recall_results" in table_names
