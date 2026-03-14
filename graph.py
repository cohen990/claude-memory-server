"""Graph memory layer — sqlite-backed associative memory graph.

Standalone module, no FastAPI dependency. Owns the sqlite DB and all
graph operations. Nodes hold synthesized memories (vibes/details) with
embeddings; edges encode weighted associations between them. Search
walks the graph neighborhood; recalls and agent reflections drive edge
weight changes during dream reconsolidation.

Thread safety: all database access is serialized through _db(), a
context manager that acquires a lock and yields the raw connection.
This lets GraphStore be used safely from both the async event loop
and background executor threads (e.g. the queue worker).
"""

import json
import math
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import networkx as nx
import numpy as np


GRAPH_DB_PATH = os.environ.get(
    "GRAPH_DB_PATH",
    os.path.expanduser("~/.memory-server/graph.db"),
)
LAYOUT_CACHE_PATH = os.environ.get(
    "LAYOUT_CACHE",
    os.path.expanduser("~/.memory-server/layout_cache.json"),
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

    def __init__(self, db_path: str | None = None,
                 check_same_thread: bool = True):
        self.db_path = db_path or GRAPH_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path,
                                     check_same_thread=check_same_thread)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()
        self._create_schema()
        self._rebuild_cache()

    @contextmanager
    def _db(self):
        """Acquire the lock and yield the raw connection.

        All database access goes through this so that GraphStore is
        thread-safe when used from both the async event loop and
        background executor threads.
        """
        with self._lock:
            yield self._conn

    def _create_schema(self):
        self._conn.executescript("""
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

            CREATE TABLE IF NOT EXISTS dream_runs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                chunks_processed INTEGER NOT NULL DEFAULT 0,
                nodes_created INTEGER NOT NULL DEFAULT 0,
                nodes_merged INTEGER NOT NULL DEFAULT 0,
                edges_created INTEGER NOT NULL DEFAULT 0,
                edges_adjusted INTEGER NOT NULL DEFAULT 0,
                nodes_resynthesized INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS dream_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                operation TEXT NOT NULL,
                node_id TEXT,
                node_type TEXT,
                detail TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES dream_runs(id)
            );

            CREATE TABLE IF NOT EXISTS recalls (
                id TEXT PRIMARY KEY,
                query_embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                session_id TEXT
            );

            CREATE TABLE IF NOT EXISTS recall_results (
                recall_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                node_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                source TEXT NOT NULL,
                connected_via TEXT,
                reflection TEXT,
                PRIMARY KEY (recall_id, position),
                FOREIGN KEY (recall_id) REFERENCES recalls(id),
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            );

            CREATE TABLE IF NOT EXISTS markers (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                label TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS english_word_freqs (
                word TEXT PRIMARY KEY,
                log_prob REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS personal_word_counts (
                word TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            );

        """)
        # Partial index for activated edges — sqlite requires separate statement
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_activated "
                "ON edges(activation_count) WHERE activation_count > 0"
            )
        except sqlite3.OperationalError:
            pass  # partial indexes not supported on all builds
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        """Run schema migrations for columns added after initial release."""
        cols = {
            row[1] for row in
            self._conn.execute("PRAGMA table_info(recalls)").fetchall()
        }
        if "session_id" not in cols:
            self._conn.execute("ALTER TABLE recalls ADD COLUMN session_id TEXT")
        if "query_text" not in cols:
            self._conn.execute("ALTER TABLE recalls ADD COLUMN query_text TEXT")
        if "dreamed_at" not in cols:
            self._conn.execute("ALTER TABLE recalls ADD COLUMN dreamed_at TEXT")
        if "general_surprisal" not in cols:
            self._conn.execute("ALTER TABLE recalls ADD COLUMN general_surprisal REAL")
        if "personal_surprisal" not in cols:
            self._conn.execute("ALTER TABLE recalls ADD COLUMN personal_surprisal REAL")
        self._conn.execute("DROP TABLE IF EXISTS rating_events")

        # Add contested columns to nodes
        node_cols = {
            row[1] for row in
            self._conn.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "contested_at" not in node_cols:
            self._conn.execute("ALTER TABLE nodes ADD COLUMN contested_at TEXT")
        if "contested_correction" not in node_cols:
            self._conn.execute("ALTER TABLE nodes ADD COLUMN contested_correction TEXT")

        # Rename rating → reflection in recall_results
        rr_cols = {
            row[1] for row in
            self._conn.execute("PRAGMA table_info(recall_results)").fetchall()
        }
        if "rating" in rr_cols and "reflection" not in rr_cols:
            self._conn.execute(
                "ALTER TABLE recall_results RENAME COLUMN rating TO reflection"
            )

    def _rebuild_cache(self):
        """Load all node embeddings into a pre-normalized numpy matrix."""
        rows = self._conn.execute(
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

        with self._db() as conn:
            conn.execute(
                "INSERT INTO nodes (id, type, text, embedding, created_at, updated_at, source_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (node_id, type, text, embedding_to_blob(embedding), now, now, source_ids_json),
            )
            conn.commit()
        self._append_to_cache(node_id, embedding)
        return node_id

    def get_node(self, node_id: str) -> dict | None:
        """Fetch a single node by ID."""
        with self._db() as conn:
            row = conn.execute(
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
        with self._db() as conn:
            conn.execute(
                "UPDATE nodes SET embedding = ?, updated_at = ? WHERE id = ?",
                (embedding_to_blob(embedding), now, node_id),
            )
            conn.commit()
        self._update_cache_embedding(node_id, embedding)

    def update_node_text(self, node_id: str, text: str, embedding: np.ndarray):
        """Replace a node's text and embedding (when re-synthesized)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as conn:
            conn.execute(
                "UPDATE nodes SET text = ?, embedding = ?, updated_at = ? WHERE id = ?",
                (text, embedding_to_blob(embedding), now, node_id),
            )
            conn.commit()
        self._update_cache_embedding(node_id, embedding)

    def contest_node(self, node_id_prefix: str, correction: str) -> dict:
        """Mark a node as contested with a proposed correction.

        Accepts a node ID prefix (at least 8 chars) and resolves to the full
        UUID. Sets contested_at and contested_correction for dream adjudication.

        Returns {"node_id": str, "text": str} on success.
        Raises ValueError if prefix is ambiguous or not found.
        """
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, text FROM nodes WHERE id LIKE ?",
                (node_id_prefix + "%",),
            ).fetchall()

        if not rows:
            raise ValueError(f"No node found matching prefix {node_id_prefix!r}")
        if len(rows) > 1:
            raise ValueError(
                f"Ambiguous prefix {node_id_prefix!r} — matches {len(rows)} nodes. "
                f"Use a longer prefix."
            )

        node_id, current_text = rows[0]
        now = datetime.now(timezone.utc).isoformat()

        with self._db() as conn:
            conn.execute(
                "UPDATE nodes SET contested_at = ?, contested_correction = ? WHERE id = ?",
                (now, correction, node_id),
            )
            conn.commit()

        return {"node_id": node_id, "text": current_text}

    def get_contested_nodes(self) -> list[dict]:
        """Return all nodes with pending contest corrections."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, type, text, contested_at, contested_correction, source_ids "
                "FROM nodes WHERE contested_at IS NOT NULL",
            ).fetchall()

        return [
            {
                "id": row[0],
                "type": row[1],
                "text": row[2],
                "contested_at": row[3],
                "contested_correction": row[4],
                "source_ids": json.loads(row[5]),
            }
            for row in rows
        ]

    def resolve_contest(self, node_id: str, new_text: str | None,
                        new_embedding: np.ndarray | None):
        """Clear a contest, optionally updating the node text.

        If new_text is provided, the node text and embedding are updated.
        If None, the contest is dismissed (original text kept).
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as conn:
            if new_text is not None and new_embedding is not None:
                conn.execute(
                    "UPDATE nodes SET text = ?, embedding = ?, "
                    "contested_at = NULL, contested_correction = NULL, "
                    "updated_at = ? WHERE id = ?",
                    (new_text, embedding_to_blob(new_embedding), now, node_id),
                )
            else:
                conn.execute(
                    "UPDATE nodes SET contested_at = NULL, "
                    "contested_correction = NULL WHERE id = ?",
                    (node_id,),
                )
            conn.commit()

        if new_embedding is not None:
            self._update_cache_embedding(node_id, new_embedding)

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

        with self._db() as conn:
            conn.execute(
                "UPDATE nodes SET embedding = ?, source_ids = ?, updated_at = ? WHERE id = ?",
                (embedding_to_blob(blended), json.dumps(merged_sources), now, node_id),
            )
            conn.commit()
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
            with self._db() as conn:
                row = conn.execute(
                    "SELECT type FROM nodes WHERE id = ?", (best_id,)
                ).fetchone()
            if not row or row[0] != node_type:
                # Find next best matching the type
                sorted_indices = np.argsort(-similarities)
                for idx in sorted_indices:
                    if float(similarities[idx]) < threshold:
                        return None
                    nid = self._node_ids[int(idx)]
                    with self._db() as conn:
                        row = conn.execute(
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
        """Insert an edge, ignoring if it already exists or is a self-edge."""
        if source_id == target_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, weight, created_at, activation_count) "
                "VALUES (?, ?, ?, ?, 0)",
                (source_id, target_id, weight, now),
            )
            conn.commit()

    def get_edges(self, node_id: str) -> list[dict]:
        """Get all edges connected to a node (both directions)."""
        with self._db() as conn:
            rows = conn.execute(
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
        with self._db() as conn:
            conn.execute(
                "UPDATE edges SET activation_count = activation_count + 1, "
                "last_activated = ? WHERE source_id = ? AND target_id = ?",
                (now, source_id, target_id),
            )
            conn.commit()

    def get_activated_edges(self) -> list[dict]:
        """Get all edges with activation_count > 0 (for dream processing)."""
        with self._db() as conn:
            rows = conn.execute(
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
        with self._db() as conn:
            conn.execute(
                "UPDATE edges SET activation_count = 0 WHERE activation_count > 0"
            )
            conn.commit()

    def update_edge_weight(self, source_id: str, target_id: str, weight: float):
        """Set edge weight, clamped to [0, 1]."""
        weight = max(0.0, min(1.0, weight))
        with self._db() as conn:
            conn.execute(
                "UPDATE edges SET weight = ? WHERE source_id = ? AND target_id = ?",
                (weight, source_id, target_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, k: int = 10,
               expand_neighbors: bool = True) -> list[dict]:
        """Search the graph by vector similarity with optional neighbor expansion.

        Pure read — no side effects. Recalls and reflections drive edge weight
        changes during dream reconsolidation.

        1. Brute-force cosine across all nodes
        2. Take top k*2 seeds
        3. For top k seeds, expand neighbors scored by edge_weight*0.3 + similarity*0.7
        4. Merge, deduplicate, return top k
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
        if expand_neighbors:
            seed_ids = [self._node_ids[int(i)] for i in top_indices[:k]]
            for seed_id in seed_ids:
                edges = self.get_edges(seed_id)
                for edge in edges:
                    neighbor_id = (
                        edge["target_id"] if edge["source_id"] == seed_id
                        else edge["source_id"]
                    )

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

        # Sort by score, take top k, hydrate with text/type
        sorted_results = sorted(results.values(), key=lambda r: r["score"], reverse=True)[:k]

        # Hydrate with node text and type
        hydrated = []
        with self._db() as conn:
            for r in sorted_results:
                row = conn.execute(
                    "SELECT type, text, source_ids FROM nodes WHERE id = ?", (r["id"],)
                ).fetchone()
                if row:
                    r["type"] = row[0]
                    r["text"] = row[1]
                    r["source_ids"] = json.loads(row[2]) if row[2] else []
                    hydrated.append(r)

        return hydrated

    # ------------------------------------------------------------------
    # Recall tracking
    # ------------------------------------------------------------------

    def create_recall(self, query_embedding: np.ndarray,
                      results: list[dict],
                      session_id: str | None = None,
                      query_text: str | None = None,
                      general_surprisal: float | None = None,
                      personal_surprisal: float | None = None) -> str:
        """Store a recall (search event) and its ordered results. Returns recall ID."""
        recall_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._db() as conn:
            conn.execute(
                "INSERT INTO recalls "
                "(id, query_embedding, created_at, session_id, query_text, "
                " general_surprisal, personal_surprisal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (recall_id, embedding_to_blob(query_embedding), now, session_id,
                 query_text, general_surprisal, personal_surprisal),
            )
            for i, r in enumerate(results):
                conn.execute(
                    "INSERT INTO recall_results "
                    "(recall_id, position, node_id, similarity, source, connected_via) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (recall_id, i, r["id"], r.get("similarity", 0.0),
                     r.get("source", "seed"), r.get("connected_via")),
                )
            conn.commit()
        return recall_id

    def reflect_on_recall(self, recall_id: str, reflections: list[str]):
        """Set reflection codes on recall_results by position.

        reflections: list of single-char codes like ['U', 'I', 'N', 'N', 'M'].
        """
        with self._db() as conn:
            for i, code in enumerate(reflections):
                conn.execute(
                    "UPDATE recall_results SET reflection = ? "
                    "WHERE recall_id = ? AND position = ?",
                    (code, recall_id, i),
                )
            conn.commit()

    def reflect_on_node(self, recall_id: str, node_id_prefix: str,
                        reflection: str) -> bool:
        """Update the reflection for a single node within a recall.

        Looks up the node by ID prefix within the recall's results.
        Returns True if a match was found and updated.
        """
        with self._db() as conn:
            rows = conn.execute(
                "SELECT node_id FROM recall_results "
                "WHERE recall_id = ? AND node_id LIKE ?",
                (recall_id, node_id_prefix + "%"),
            ).fetchall()

        if not rows:
            return False
        if len(rows) > 1:
            raise ValueError(
                f"Ambiguous prefix {node_id_prefix!r} in recall {recall_id} — "
                f"matches {len(rows)} results"
            )

        node_id = rows[0][0]
        with self._db() as conn:
            conn.execute(
                "UPDATE recall_results SET reflection = ? "
                "WHERE recall_id = ? AND node_id = ?",
                (reflection, recall_id, node_id),
            )
            conn.commit()
        return True

    def get_reflected_recalls(self) -> list[dict]:
        """Return un-dreamed recalls that have at least one reflected result.

        Returns list of dicts with keys: recall_id, query_embedding, results.
        Each result has: node_id, similarity, source, connected_via, reflection.
        """
        with self._db() as conn:
            # Find recall IDs that have reflections and haven't been dreamed yet
            reflected_ids = conn.execute(
                "SELECT DISTINCT rr.recall_id FROM recall_results rr "
                "JOIN recalls r ON r.id = rr.recall_id "
                "WHERE rr.reflection IS NOT NULL AND r.dreamed_at IS NULL"
            ).fetchall()
            if not reflected_ids:
                return []

            recalls = []
            for (recall_id,) in reflected_ids:
                row = conn.execute(
                    "SELECT query_embedding FROM recalls WHERE id = ?",
                    (recall_id,),
                ).fetchone()
                if not row:
                    continue

                results = conn.execute(
                    "SELECT node_id, similarity, source, connected_via, reflection "
                    "FROM recall_results WHERE recall_id = ? ORDER BY position",
                    (recall_id,),
                ).fetchall()

                recalls.append({
                    "recall_id": recall_id,
                    "query_embedding": blob_to_embedding(row[0]),
                    "results": [
                        {
                            "node_id": r[0],
                            "similarity": r[1],
                            "source": r[2],
                            "connected_via": r[3],
                            "reflection": r[4],
                        }
                        for r in results
                    ],
                })

        return recalls

    def list_recalls(self, session_id: str | None = None,
                     limit: int = 1) -> list[dict]:
        """Return recent recalls with full details for display.

        Joins recalls → recall_results → nodes, ordered by created_at DESC.
        Returns list of dicts with keys: recall_id, created_at, session_id,
        results (each with node_id, type, text, similarity, source, reflection).
        """
        with self._db() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT id, created_at, session_id, query_text, "
                    "general_surprisal, personal_surprisal FROM recalls "
                    "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, created_at, session_id, query_text, "
                    "general_surprisal, personal_surprisal FROM recalls "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            recalls = []
            for recall_id, created_at, sess_id, query_text, gen_s, pers_s in rows:
                results = conn.execute(
                    "SELECT rr.node_id, rr.similarity, rr.source, "
                    "rr.connected_via, rr.reflection, n.type, n.text "
                    "FROM recall_results rr "
                    "LEFT JOIN nodes n ON rr.node_id = n.id "
                    "WHERE rr.recall_id = ? ORDER BY rr.position",
                    (recall_id,),
                ).fetchall()

                recalls.append({
                    "recall_id": recall_id,
                    "created_at": created_at,
                    "session_id": sess_id,
                    "query_text": query_text,
                    "general_surprisal": gen_s,
                    "personal_surprisal": pers_s,
                    "results": [
                        {
                            "node_id": r[0],
                            "similarity": r[1],
                            "source": r[2],
                            "connected_via": r[3],
                            "reflection": r[4],
                            "type": r[5],
                            "text": r[6],
                        }
                        for r in results
                    ],
                })

        return recalls

    def clear_reflected_recalls(self):
        """Mark reflected recalls as dreamed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as conn:
            reflected_ids = conn.execute(
                "SELECT DISTINCT recall_id FROM recall_results WHERE reflection IS NOT NULL"
            ).fetchall()
            if not reflected_ids:
                return

            ids = [r[0] for r in reflected_ids]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE recalls SET dreamed_at = ? WHERE id IN ({placeholders})",
                [now] + ids,
            )
            conn.commit()

    def get_recalled_nodes_for_sessions(self, session_ids: list[str]) -> list[dict]:
        """Fetch unique nodes that were recalled during the given sessions.

        Joins recalls → recall_results → nodes filtered by session_id IN (...).
        Returns list of {id, type, text} — no embeddings. Deduplicated by node_id.
        """
        if not session_ids:
            return []

        placeholders = ",".join("?" * len(session_ids))
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT n.id, n.type, n.text "
                f"FROM recalls r "
                f"JOIN recall_results rr ON r.id = rr.recall_id "
                f"JOIN nodes n ON rr.node_id = n.id "
                f"WHERE r.session_id IN ({placeholders})",
                session_ids,
            ).fetchall()

        return [{"id": r[0], "type": r[1], "text": r[2]} for r in rows]

    def clear_processed_recalls(self):
        """Mark all recalls as dreamed instead of deleting them."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as conn:
            conn.execute(
                "UPDATE recalls SET dreamed_at = ? WHERE dreamed_at IS NULL",
                (now,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Read-only queries (for browse UI)
    # ------------------------------------------------------------------

    def list_nodes(self, node_type: str | None = None,
                   limit: int = 50, offset: int = 0) -> dict:
        """Paginated node listing without embedding blobs."""
        with self._db() as conn:
            if node_type:
                total = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = ?", (node_type,)
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT id, type, text, created_at, updated_at, source_ids "
                    "FROM nodes WHERE type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (node_type, limit, offset),
                ).fetchall()
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM nodes"
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT id, type, text, created_at, updated_at, source_ids "
                    "FROM nodes ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()

        return {
            "nodes": [
                {
                    "id": r[0],
                    "type": r[1],
                    "text": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                    "source_ids": json.loads(r[5]),
                }
                for r in rows
            ],
            "total": total,
        }

    def compute_layout(self, cache_path: str | None = None) -> dict[str, dict]:
        """Compute x/y positions for all nodes using networkx spring layout.

        Detects connected components, lays out each independently, then tiles
        them in a grid (largest-first, left-to-right with wrapping).
        Caches result to JSON. Returns {node_id: {"x": float, "y": float}}.
        """
        cache_path = cache_path or LAYOUT_CACHE_PATH

        with self._db() as conn:
            node_rows = conn.execute("SELECT id FROM nodes").fetchall()
            edge_rows = conn.execute(
                "SELECT source_id, target_id, weight FROM edges"
            ).fetchall()

        if not node_rows:
            positions: dict[str, dict] = {}
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(positions, f)
            return positions

        G = nx.Graph()
        for (node_id,) in node_rows:
            G.add_node(node_id)
        for src, tgt, weight in edge_rows:
            if G.has_node(src) and G.has_node(tgt):
                # Compress weight range to reduce spacing variance.
                # Raw weights span [0,1] → layout_weight in [0.7, 1.0].
                # Weak edges still attract substantially, keeping the graph
                # compact while preserving relative topology.
                layout_weight = 0.7 + weight * 0.3
                G.add_edge(src, tgt, weight=layout_weight)

        # Layout each connected component independently
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        component_positions: list[dict[str, tuple[float, float]]] = []
        component_sizes: list[int] = []

        total_nodes = len(node_rows)
        for comp_nodes in components:
            subgraph = G.subgraph(comp_nodes)
            n = len(comp_nodes)
            pos = nx.spring_layout(
                subgraph, weight="weight", iterations=100,
                seed=42, scale=math.sqrt(n),
            )

            component_positions.append({n: (float(x), float(y)) for n, (x, y) in pos.items()})
            component_sizes.append(len(comp_nodes))

        # Tile components in a grid: left-to-right with wrapping
        cols = max(1, int(math.ceil(math.sqrt(len(components)))))
        spacing = 3.0  # gap between component bounding boxes (in layout units)
        positions = {}
        offset_x = 0.0
        offset_y = 0.0
        row_height = 0.0
        col_idx = 0

        for comp_pos in component_positions:
            if not comp_pos:
                continue

            xs = [p[0] for p in comp_pos.values()]
            ys = [p[1] for p in comp_pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y

            for node_id, (x, y) in comp_pos.items():
                positions[node_id] = {
                    "x": round(x - min_x + offset_x, 2),
                    "y": round(y - min_y + offset_y, 2),
                }

            row_height = max(row_height, height)
            offset_x += width + spacing
            col_idx += 1

            if col_idx >= cols:
                col_idx = 0
                offset_x = 0.0
                offset_y += row_height + spacing
                row_height = 0.0

        # Scale to pixel-space.
        # FR with scale=sqrt(N) gives roughly constant inter-node distance
        # in layout units, but higher edge density compresses the layout.
        # Compensate: px_per_unit grows with density relative to a baseline.
        # Calibrated at 1520 nodes / 2000 edges (density ~1.3), px=75.
        total_edges = len(edge_rows)
        density = total_edges / max(total_nodes, 1)
        baseline_density = 1.0
        px_per_unit = 75 * max(1.0, density / baseline_density)
        for pos in positions.values():
            pos["x"] = round(pos["x"] * px_per_unit, 1)
            pos["y"] = round(pos["y"] * px_per_unit, 1)

        # Local repulsion pass — inflate dense clusters so nodes are readable.
        # Like a Mercator projection: distorts distances but preserves
        # relative topology while making dense regions navigable.
        # min_dist is per-pair: sum of radii + padding, so large nodes
        # (high degree) get more clearance.
        degree = dict(G.degree())
        def _node_radius(node_id: str) -> float:
            """Match frontend: min(8 + degree*3, 40) / 2."""
            d = degree.get(node_id, 1)
            return min(8 + d * 3, 40) / 2

        padding = 10.0  # extra px clearance beyond touching
        ids = list(positions.keys())
        n_nodes = len(ids)
        if n_nodes > 1:
            radii = [_node_radius(nid) for nid in ids]
            for _ in range(5):  # iterate to convergence
                moved = 0
                for i in range(n_nodes):
                    pi = positions[ids[i]]
                    for j in range(i + 1, n_nodes):
                        pj = positions[ids[j]]
                        required = radii[i] + radii[j] + padding
                        dx = pi["x"] - pj["x"]
                        dy = pi["y"] - pj["y"]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < required and dist > 0:
                            push = (required - dist) / 2
                            ux, uy = dx / dist, dy / dist
                            pi["x"] += ux * push
                            pi["y"] += uy * push
                            pj["x"] -= ux * push
                            pj["y"] -= uy * push
                            moved += 1
                        elif dist == 0:
                            import random
                            angle = random.random() * 2 * math.pi
                            pi["x"] += math.cos(angle) * required / 2
                            pi["y"] += math.sin(angle) * required / 2
                            moved += 1
                if moved == 0:
                    break
            # Round after repulsion
            for pos in positions.values():
                pos["x"] = round(pos["x"], 1)
                pos["y"] = round(pos["y"], 1)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(positions, f)

        return positions

    def get_full_graph(self) -> dict:
        """All nodes + edges for Cytoscape.js initial load (no embeddings)."""
        with self._db() as conn:
            node_rows = conn.execute(
                "SELECT id, type, text, created_at, updated_at, source_ids "
                "FROM nodes",
            ).fetchall()
            edge_rows = conn.execute(
                "SELECT source_id, target_id, weight, created_at, last_activated, activation_count "
                "FROM edges",
            ).fetchall()

        # Load cached layout positions
        layout_positions: dict[str, dict] = {}
        try:
            with open(LAYOUT_CACHE_PATH) as f:
                layout_positions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return {
            "nodes": [
                {
                    "id": r[0],
                    "type": r[1],
                    "text": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                    "source_ids": json.loads(r[5]),
                    "position": layout_positions.get(r[0]),
                }
                for r in node_rows
            ],
            "edges": [
                {
                    "source_id": r[0],
                    "target_id": r[1],
                    "weight": r[2],
                    "created_at": r[3],
                    "last_activated": r[4],
                    "activation_count": r[5],
                }
                for r in edge_rows
            ],
        }

    def reflection_distribution(self) -> dict:
        """Count of each reflection code across all recall results."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT reflection, COUNT(*) FROM recall_results "
                "WHERE reflection IS NOT NULL GROUP BY reflection"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def reflection_timeline(self) -> list[dict]:
        """Reflection counts bucketed by hour.

        Returns list of {bucket, U, I, N, D, M} sorted chronologically.
        Bucket is ISO timestamp truncated to the hour.
        """
        with self._db() as conn:
            rows = conn.execute(
                "SELECT substr(r.created_at, 1, 13) AS bucket, "
                "       rr.reflection, COUNT(*) "
                "FROM recall_results rr "
                "JOIN recalls r ON r.id = rr.recall_id "
                "WHERE rr.reflection IS NOT NULL "
                "GROUP BY bucket, rr.reflection "
                "ORDER BY bucket"
            ).fetchall()

        buckets: dict[str, dict] = {}
        for bucket_str, reflection, count in rows:
            if bucket_str not in buckets:
                buckets[bucket_str] = {
                    "bucket": bucket_str + ":00:00",
                    "U": 0, "I": 0, "N": 0, "D": 0, "M": 0,
                }
            if reflection in buckets[bucket_str]:
                buckets[bucket_str][reflection] = count

        return list(buckets.values())

    # ------------------------------------------------------------------
    # Markers (deploy / change annotations for charts)
    # ------------------------------------------------------------------

    def create_marker(self, label: str) -> dict:
        """Create a timestamp marker with a label."""
        marker_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO markers (id, created_at, label) VALUES (?, ?, ?)",
                (marker_id, now, label),
            )
            conn.commit()
        return {"id": marker_id, "created_at": now, "label": label}

    def list_markers(self) -> list[dict]:
        """Return all markers, sorted chronologically."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id, created_at, label FROM markers ORDER BY created_at"
            ).fetchall()
        return [{"id": r[0], "created_at": r[1], "label": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Word frequency tables (for surprisal gate)
    # ------------------------------------------------------------------

    def load_english_freqs(self, freqs: dict[str, float]):
        """Bulk-load english word frequencies. freqs: {word: log_prob}.

        Replaces any existing data.
        """
        with self._db() as conn:
            conn.execute("DELETE FROM english_word_freqs")
            conn.executemany(
                "INSERT INTO english_word_freqs (word, log_prob) VALUES (?, ?)",
                freqs.items(),
            )
            conn.commit()

    def get_english_log_prob(self, word: str) -> float | None:
        """Return log_prob for a single english word, or None if unseen."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT log_prob FROM english_word_freqs WHERE word = ?", (word,)
            ).fetchone()
        return row[0] if row else None

    def get_english_log_probs(self, words: list[str]) -> dict[str, float]:
        """Return {word: log_prob} for words found in the english table."""
        if not words:
            return {}
        placeholders = ",".join("?" for _ in words)
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT word, log_prob FROM english_word_freqs "
                f"WHERE word IN ({placeholders})", words
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def english_freq_count(self) -> int:
        """Number of words in the english frequency table."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM english_word_freqs"
            ).fetchone()
        return row[0]

    def update_personal_word_counts(self, words: list[str]):
        """Increment counts for each word in the personal corpus."""
        if not words:
            return
        with self._db() as conn:
            conn.executemany(
                "INSERT INTO personal_word_counts (word, count) VALUES (?, 1) "
                "ON CONFLICT(word) DO UPDATE SET count = count + 1",
                [(w,) for w in words],
            )
            conn.commit()

    def get_personal_word_counts(self, words: list[str]) -> dict[str, int]:
        """Return {word: count} for words found in the personal table."""
        if not words:
            return {}
        placeholders = ",".join("?" for _ in words)
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT word, count FROM personal_word_counts "
                f"WHERE word IN ({placeholders})", words
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def personal_corpus_total(self) -> int:
        """Total word count across all personal entries."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(count), 0) FROM personal_word_counts"
            ).fetchone()
        return row[0]

    def personal_vocab_size(self) -> int:
        """Number of distinct words in the personal corpus."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM personal_word_counts"
            ).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return node and edge counts by type."""
        with self._db() as conn:
            node_counts = {}
            for row in conn.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type"):
                node_counts[row[0]] = row[1]

            total_nodes = sum(node_counts.values())
            total_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            activated_edges = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE activation_count > 0"
            ).fetchone()[0]

        return {
            "total_nodes": total_nodes,
            "nodes_by_type": node_counts,
            "total_edges": total_edges,
            "activated_edges": activated_edges,
        }

    def pending_dream_count(self) -> int:
        """Count of reflected recalls that haven't been dreamed yet."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT rr.recall_id) FROM recall_results rr "
                "JOIN recalls r ON r.id = rr.recall_id "
                "WHERE rr.reflection IS NOT NULL AND r.dreamed_at IS NULL"
            ).fetchone()
        return row[0]

    def close(self):
        with self._db() as conn:
            conn.close()


class DreamLog:
    """Append-only dream log backed by the same sqlite DB as GraphStore."""

    def __init__(self, graph_store: GraphStore):
        self._graph_store = graph_store

    def start_run(self, run_type: str) -> str:
        """Insert a dream_runs row and return the run UUID."""
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._graph_store._db() as conn:
            conn.execute(
                "INSERT INTO dream_runs (id, type, started_at) VALUES (?, ?, ?)",
                (run_id, run_type, now),
            )
            conn.commit()
        return run_id

    def log_operation(self, run_id: str, operation: str,
                      node_id: str | None, node_type: str | None,
                      detail: dict):
        """Insert a dream_operations row."""
        now = datetime.now(timezone.utc).isoformat()
        with self._graph_store._db() as conn:
            conn.execute(
                "INSERT INTO dream_operations "
                "(run_id, timestamp, operation, node_id, node_type, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, now, operation, node_id, node_type, json.dumps(detail)),
            )
            conn.commit()

    def finish_run(self, run_id: str, *, error: str | None = None,
                   chunks_processed: int = 0, nodes_created: int = 0,
                   nodes_merged: int = 0, edges_created: int = 0,
                   edges_adjusted: int = 0, nodes_resynthesized: int = 0):
        """Update a dream_runs row with final counts and finished_at."""
        now = datetime.now(timezone.utc).isoformat()
        with self._graph_store._db() as conn:
            conn.execute(
                "UPDATE dream_runs SET finished_at = ?, error = ?, "
                "chunks_processed = ?, nodes_created = ?, nodes_merged = ?, "
                "edges_created = ?, edges_adjusted = ?, nodes_resynthesized = ? "
                "WHERE id = ?",
                (now, error, chunks_processed, nodes_created, nodes_merged,
                 edges_created, edges_adjusted, nodes_resynthesized, run_id),
            )
            conn.commit()

    def list_runs(self, limit: int = 20) -> list[dict]:
        """Return recent dream runs, newest first."""
        with self._graph_store._db() as conn:
            rows = conn.execute(
                "SELECT id, type, started_at, finished_at, chunks_processed, "
                "nodes_created, nodes_merged, edges_created, edges_adjusted, "
                "nodes_resynthesized, error "
                "FROM dream_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "type": r[1],
                "started_at": r[2],
                "finished_at": r[3],
                "chunks_processed": r[4],
                "nodes_created": r[5],
                "nodes_merged": r[6],
                "edges_created": r[7],
                "edges_adjusted": r[8],
                "nodes_resynthesized": r[9],
                "error": r[10],
            }
            for r in rows
        ]

    def get_run_operations(self, run_id: str) -> list[dict]:
        """Return all operations for a given run."""
        with self._graph_store._db() as conn:
            rows = conn.execute(
                "SELECT id, run_id, timestamp, operation, node_id, node_type, detail "
                "FROM dream_operations WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "run_id": r[1],
                "timestamp": r[2],
                "operation": r[3],
                "node_id": r[4],
                "node_type": r[5],
                "detail": json.loads(r[6]),
            }
            for r in rows
        ]
