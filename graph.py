"""Graph memory layer — sqlite-backed associative memory graph.

Standalone module, no FastAPI dependency. Owns the sqlite DB and all
graph operations. Nodes hold synthesized memories (vibes/details) with
embeddings; edges encode weighted associations between them. Search
walks the graph neighborhood, and retrieval strengthens edges for
reconsolidation during the next dream cycle.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np


GRAPH_DB_PATH = os.environ.get(
    "GRAPH_DB_PATH",
    os.path.expanduser("~/.memory-server/graph.db"),
)

EMBEDDING_DIM = 768  # nomic-embed-text-v1.5 Matryoshka 768-dim
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a float32 numpy vector to raw bytes."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize raw bytes to a float32 numpy vector."""
    return np.frombuffer(blob, dtype=np.float32).copy()


class GraphStore:
    """Graph-based long-term memory store backed by sqlite."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or GRAPH_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._create_schema()
        self._rebuild_cache()

    def _create_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_ids TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                last_activated TEXT,
                activation_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (source_id, target_id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
        """)
        # Partial index for activated edges — sqlite requires separate statement
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_activated "
                "ON edges(activation_count) WHERE activation_count > 0"
            )
        except sqlite3.OperationalError:
            pass  # partial indexes not supported on all builds
        self.conn.commit()

    def _rebuild_cache(self):
        """Load all node embeddings into a pre-normalized numpy matrix."""
        rows = self.conn.execute(
            "SELECT id, embedding FROM nodes"
        ).fetchall()

        self._node_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}

        if not rows:
            self._embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            return

        embeddings = []
        for node_id, blob in rows:
            self._id_to_idx[node_id] = len(self._node_ids)
            self._node_ids.append(node_id)
            embeddings.append(blob_to_embedding(blob))

        self._embeddings = np.vstack(embeddings).astype(np.float32)
        # Pre-normalize for cosine similarity via dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = self._embeddings / norms

    def _append_to_cache(self, node_id: str, embedding: np.ndarray):
        """Add a single node to the in-memory cache."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._id_to_idx[node_id] = len(self._node_ids)
        self._node_ids.append(node_id)
        if self._embeddings.shape[0] == 0:
            self._embeddings = vec
        else:
            self._embeddings = np.vstack([self._embeddings, vec])

    def _update_cache_embedding(self, node_id: str, embedding: np.ndarray):
        """Update a single node's embedding in the cache."""
        idx = self._id_to_idx.get(node_id)
        if idx is None:
            return
        vec = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._embeddings[idx] = vec

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, type: str, text: str, embedding: np.ndarray,
                 source_ids: list[str] | None = None) -> str:
        """Insert a new node and update the cache. Returns the node ID."""
        node_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        source_ids_json = json.dumps(source_ids or [])

        self.conn.execute(
            "INSERT INTO nodes (id, type, text, embedding, created_at, updated_at, source_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (node_id, type, text, embedding_to_blob(embedding), now, now, source_ids_json),
        )
        self.conn.commit()
        self._append_to_cache(node_id, embedding)
        return node_id

    def get_node(self, node_id: str) -> dict | None:
        """Fetch a single node by ID."""
        row = self.conn.execute(
            "SELECT id, type, text, embedding, created_at, updated_at, source_ids "
            "FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "type": row[1],
            "text": row[2],
            "embedding": blob_to_embedding(row[3]),
            "created_at": row[4],
            "updated_at": row[5],
            "source_ids": json.loads(row[6]),
        }

    def update_node_embedding(self, node_id: str, embedding: np.ndarray):
        """Replace a node's embedding (for dream reconsolidation)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE nodes SET embedding = ?, updated_at = ? WHERE id = ?",
            (embedding_to_blob(embedding), now, node_id),
        )
        self.conn.commit()
        self._update_cache_embedding(node_id, embedding)

    def update_node_text(self, node_id: str, text: str, embedding: np.ndarray):
        """Replace a node's text and embedding (when re-synthesized)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE nodes SET text = ?, embedding = ?, updated_at = ? WHERE id = ?",
            (text, embedding_to_blob(embedding), now, node_id),
        )
        self.conn.commit()
        self._update_cache_embedding(node_id, embedding)

    def merge_node_embedding(self, node_id: str, new_embedding: np.ndarray,
                             new_source_ids: list[str] | None = None):
        """Blend a new embedding into an existing node, weighted by source count.

        Weight = len(existing_source_ids) : 1 — established nodes are harder to shift.
        """
        node = self.get_node(node_id)
        if not node:
            return

        existing_sources = node["source_ids"]
        weight = len(existing_sources)
        blended = (node["embedding"] * weight + np.asarray(new_embedding, dtype=np.float32)) / (weight + 1)

        merged_sources = existing_sources + (new_source_ids or [])
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "UPDATE nodes SET embedding = ?, source_ids = ?, updated_at = ? WHERE id = ?",
            (embedding_to_blob(blended), json.dumps(merged_sources), now, node_id),
        )
        self.conn.commit()
        self._update_cache_embedding(node_id, blended)

    def find_similar(self, embedding: np.ndarray, threshold: float = 0.85,
                     node_type: str | None = None) -> dict | None:
        """Find the most similar node above threshold. For merge-vs-create decisions."""
        if self._embeddings.shape[0] == 0:
            return None

        query = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        similarities = self._embeddings @ query
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim < threshold:
            return None

        best_id = self._node_ids[best_idx]

        # Optional type filter
        if node_type:
            row = self.conn.execute(
                "SELECT type FROM nodes WHERE id = ?", (best_id,)
            ).fetchone()
            if not row or row[0] != node_type:
                # Find next best matching the type
                sorted_indices = np.argsort(-similarities)
                for idx in sorted_indices:
                    if float(similarities[idx]) < threshold:
                        return None
                    nid = self._node_ids[int(idx)]
                    row = self.conn.execute(
                        "SELECT type FROM nodes WHERE id = ?", (nid,)
                    ).fetchone()
                    if row and row[0] == node_type:
                        best_id = nid
                        best_sim = float(similarities[int(idx)])
                        break
                else:
                    return None

        node = self.get_node(best_id)
        if node:
            node["similarity"] = best_sim
        return node

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, source_id: str, target_id: str, weight: float = 0.5):
        """Insert an edge, ignoring if it already exists."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (source_id, target_id, weight, created_at, activation_count) "
            "VALUES (?, ?, ?, ?, 0)",
            (source_id, target_id, weight, now),
        )
        self.conn.commit()

    def get_edges(self, node_id: str) -> list[dict]:
        """Get all edges connected to a node (both directions)."""
        rows = self.conn.execute(
            "SELECT source_id, target_id, weight, created_at, last_activated, activation_count "
            "FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        ).fetchall()
        return [
            {
                "source_id": r[0],
                "target_id": r[1],
                "weight": r[2],
                "created_at": r[3],
                "last_activated": r[4],
                "activation_count": r[5],
            }
            for r in rows
        ]

    def bump_edge_activation(self, source_id: str, target_id: str):
        """Increment activation count and update timestamp for a single edge."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE edges SET activation_count = activation_count + 1, "
            "last_activated = ? WHERE source_id = ? AND target_id = ?",
            (now, source_id, target_id),
        )
        self.conn.commit()

    def get_activated_edges(self) -> list[dict]:
        """Get all edges with activation_count > 0 (for dream processing)."""
        rows = self.conn.execute(
            "SELECT source_id, target_id, weight, created_at, last_activated, activation_count "
            "FROM edges WHERE activation_count > 0"
        ).fetchall()
        return [
            {
                "source_id": r[0],
                "target_id": r[1],
                "weight": r[2],
                "created_at": r[3],
                "last_activated": r[4],
                "activation_count": r[5],
            }
            for r in rows
        ]

    def reset_activation_counts(self):
        """Zero out all activation counts (called after dream processing)."""
        self.conn.execute(
            "UPDATE edges SET activation_count = 0 WHERE activation_count > 0"
        )
        self.conn.commit()

    def update_edge_weight(self, source_id: str, target_id: str, weight: float):
        """Set edge weight, clamped to [0, 1]."""
        weight = max(0.0, min(1.0, weight))
        self.conn.execute(
            "UPDATE edges SET weight = ? WHERE source_id = ? AND target_id = ?",
            (weight, source_id, target_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, k: int = 10,
               expand_neighbors: bool = True) -> list[dict]:
        """Search the graph by vector similarity with optional neighbor expansion.

        1. Brute-force cosine across all nodes
        2. Take top k*2 seeds
        3. For top k seeds, expand neighbors scored by edge_weight*0.3 + similarity*0.7
        4. Merge, deduplicate, return top k
        5. Batch-activate all traversed edges
        """
        if self._embeddings.shape[0] == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        similarities = self._embeddings @ query
        n_seeds = min(k * 2, len(self._node_ids))
        top_indices = np.argsort(-similarities)[:n_seeds]

        # Build seed results
        results: dict[str, dict] = {}
        for idx in top_indices:
            node_id = self._node_ids[int(idx)]
            sim = float(similarities[int(idx)])
            results[node_id] = {
                "id": node_id,
                "similarity": sim,
                "score": sim,
                "source": "seed",
            }

        # Expand neighbors for top k seeds
        activated_edges: list[tuple[str, str]] = []
        if expand_neighbors:
            seed_ids = [self._node_ids[int(i)] for i in top_indices[:k]]
            for seed_id in seed_ids:
                edges = self.get_edges(seed_id)
                for edge in edges:
                    neighbor_id = (
                        edge["target_id"] if edge["source_id"] == seed_id
                        else edge["source_id"]
                    )
                    # Track edge for activation (always use canonical order)
                    activated_edges.append((edge["source_id"], edge["target_id"]))

                    if neighbor_id in results:
                        continue  # already a seed or neighbor

                    # Get neighbor similarity
                    neighbor_idx = self._id_to_idx.get(neighbor_id)
                    if neighbor_idx is None:
                        continue
                    neighbor_sim = float(similarities[neighbor_idx])
                    neighbor_score = edge["weight"] * 0.3 + neighbor_sim * 0.7

                    results[neighbor_id] = {
                        "id": neighbor_id,
                        "similarity": neighbor_sim,
                        "score": neighbor_score,
                        "source": "neighbor",
                        "edge_weight": edge["weight"],
                        "connected_via": seed_id,
                    }

        # Batch activate traversed edges
        if activated_edges:
            now = datetime.now(timezone.utc).isoformat()
            self.conn.executemany(
                "UPDATE edges SET activation_count = activation_count + 1, "
                "last_activated = ? WHERE source_id = ? AND target_id = ?",
                [(now, src, tgt) for src, tgt in activated_edges],
            )
            self.conn.commit()

        # Sort by score, take top k, hydrate with text/type
        sorted_results = sorted(results.values(), key=lambda r: r["score"], reverse=True)[:k]

        # Hydrate with node text and type
        hydrated = []
        for r in sorted_results:
            row = self.conn.execute(
                "SELECT type, text FROM nodes WHERE id = ?", (r["id"],)
            ).fetchone()
            if row:
                r["type"] = row[0]
                r["text"] = row[1]
                hydrated.append(r)

        return hydrated

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return node and edge counts by type."""
        node_counts = {}
        for row in self.conn.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type"):
            node_counts[row[0]] = row[1]

        total_nodes = sum(node_counts.values())
        total_edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        activated_edges = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE activation_count > 0"
        ).fetchone()[0]

        return {
            "total_nodes": total_nodes,
            "nodes_by_type": node_counts,
            "total_edges": total_edges,
            "activated_edges": activated_edges,
        }

    def close(self):
        self.conn.close()
