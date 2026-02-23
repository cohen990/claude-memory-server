"""Memory server — FastAPI + ChromaDB + nomic-embed-text-v1.5.

Runs on the secondary machine. Embeds conversation chunks and stores
them in a persistent ChromaDB collection. Provides /ingest, /ingest_summary,
and /search endpoints.

Ingestion is async: /ingest writes chunks to an incoming/ directory and
returns 201 immediately. A background worker embeds and stores them.
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.expanduser("~/.memory-server/chromadb"))
INCOMING_DIR = os.environ.get("INCOMING_DIR", os.path.expanduser("~/.memory-server/incoming"))
MODEL_NAME = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
COLLECTION_NAME = "conversations"
SUBCHUNK_COLLECTION_NAME = "subchunks"
USER_INPUT_COLLECTION_NAME = "user_inputs"
SUBCHUNK_WINDOW = 500   # characters per subchunk
SUBCHUNK_OVERLAP = 100  # character overlap between consecutive subchunks
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")
BIND_HOST = os.environ.get("BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("BIND_PORT", "8420"))
WORKER_INTERVAL = float(os.environ.get("WORKER_INTERVAL", "2.0"))  # seconds between queue checks

# nomic-embed-text-v1.5 expects task-prefixed inputs
SEARCH_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "

MMR_LAMBDA = 0.7        # relevance vs diversity tradeoff (1.0 = pure relevance)
MMR_OVERFETCH = 3       # fetch this many times k candidates for MMR pool


# ---------------------------------------------------------------------------
# MMR re-ranking
# ---------------------------------------------------------------------------

def mmr_rerank(
    embeddings: list[list[float]],
    distances: list[float],
    k: int,
    lam: float = MMR_LAMBDA,
) -> list[int]:
    """Maximal Marginal Relevance — select k diverse, relevant indices.

    ChromaDB distances are cosine distances (0 = identical, 2 = opposite).
    Convert to similarity = 1 - distance for the MMR score.

    Returns indices into the original results list.
    """
    if len(embeddings) <= k:
        return list(range(len(embeddings)))

    embs = np.array(embeddings)
    # Normalise for cosine similarity via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embs = embs / norms

    relevance = np.array([1.0 - d for d in distances])

    selected = []
    remaining = set(range(len(embeddings)))

    for _ in range(k):
        best_idx = -1
        best_score = -float("inf")
        for idx in remaining:
            rel = relevance[idx]
            if selected:
                sims = embs[idx] @ embs[selected].T
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0
            score = lam * rel - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

model: SentenceTransformer = None  # type: ignore[assignment]
collection: chromadb.Collection = None  # type: ignore[assignment]
subchunk_collection: chromadb.Collection = None  # type: ignore[assignment]
user_input_collection: chromadb.Collection = None  # type: ignore[assignment]
worker_task: asyncio.Task = None  # type: ignore[assignment]
graph_store = None  # GraphStore instance, initialized at startup


# ---------------------------------------------------------------------------
# Queue worker
# ---------------------------------------------------------------------------

def make_subchunks(text: str, window: int = SUBCHUNK_WINDOW, overlap: int = SUBCHUNK_OVERLAP) -> list[str]:
    """Split text into rolling windows of `window` chars with `overlap` overlap."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + window])
        start += window - overlap
    return chunks


def process_one_file(path: Path):
    """Embed a queued chunk file and store it in ChromaDB. Deletes the file on success."""
    with open(path) as f:
        item = json.load(f)

    chunk_type = item.get("chunk_type", "turn_pair")
    text = item["text"]
    embedding = model.encode([DOCUMENT_PREFIX + text], show_progress_bar=False).tolist()

    session_id = item.get("session_id", "")
    turn_number = item.get("turn_number", -1)
    doc_id = f"{session_id}_{turn_number}"
    metadata = {
        "session_id": session_id,
        "timestamp": item.get("timestamp", ""),
        "project": item.get("project", ""),
        "turn_number": turn_number,
        "branch": item.get("branch", ""),
        "chunk_type": chunk_type,
        "dreamed": 0,
    }
    collection.upsert(
        ids=[doc_id],
        embeddings=embedding,
        documents=[text],
        metadatas=[metadata],
    )

    # Generate and store subchunks
    windows = make_subchunks(text)
    if windows:
        sc_ids = []
        sc_documents = []
        sc_metadatas = []
        for i, window_text in enumerate(windows):
            sc_ids.append(f"{doc_id}_sc{i}")
            sc_documents.append(window_text)
            sc_metadatas.append({
                "session_id": session_id,
                "timestamp": metadata["timestamp"],
                "project": metadata["project"],
                "turn_number": turn_number,
                "branch": metadata["branch"],
                "chunk_type": "subchunk",
                "parent_chunk_id": doc_id,
                "window_index": i,
            })
        # Batch embed all subchunks at once
        sc_embeddings = model.encode(
            [DOCUMENT_PREFIX + t for t in sc_documents],
            show_progress_bar=False,
        ).tolist()
        subchunk_collection.upsert(
            ids=sc_ids,
            embeddings=sc_embeddings,
            documents=sc_documents,
            metadatas=sc_metadatas,
        )

    # Store user input separately for prompt-to-prompt matching
    user_text = item.get("user_text", "")
    if user_text:
        ui_embedding = model.encode(
            [DOCUMENT_PREFIX + user_text], show_progress_bar=False
        ).tolist()
        user_input_collection.upsert(
            ids=[f"{doc_id}_ui"],
            embeddings=ui_embedding,
            documents=[user_text],
            metadatas=[{
                "session_id": session_id,
                "timestamp": metadata["timestamp"],
                "project": metadata["project"],
                "turn_number": turn_number,
                "branch": metadata["branch"],
                "chunk_type": "user_input",
                "parent_chunk_id": doc_id,
            }],
        )

    path.unlink()


async def queue_worker():
    """Background loop that processes the incoming queue."""
    incoming = Path(INCOMING_DIR)
    processed_total = 0
    while True:
        try:
            files = sorted(incoming.glob("*.json"))
            if files:
                print(f"Queue: {len(files)} pending")
            for i, path in enumerate(files):
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, process_one_file, path)
                    processed_total += 1
                    remaining = len(files) - i - 1
                    if processed_total % 50 == 0:
                        print(f"Queue: embedded {processed_total} total, {remaining} remaining in batch")
                except Exception as e:
                    print(f"Worker error processing {path.name}: {e}")
                    failed_dir = incoming.parent / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    try:
                        path.rename(failed_dir / path.name)
                    except OSError:
                        pass
        except Exception as e:
            print(f"Worker loop error: {e}")

        await asyncio.sleep(WORKER_INTERVAL)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, collection, subchunk_collection, user_input_collection, worker_task, graph_store

    print(f"Loading embedding model: {MODEL_NAME} (device={DEVICE})")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    # Use Matryoshka 768-dim (full) embeddings
    model.truncate_dim = 768
    print("Model loaded.")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(INCOMING_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    subchunk_collection = client.get_or_create_collection(
        name=SUBCHUNK_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    user_input_collection = client.get_or_create_collection(
        name=USER_INPUT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' ready ({collection.count()} documents).")
    print(f"ChromaDB collection '{SUBCHUNK_COLLECTION_NAME}' ready ({subchunk_collection.count()} subchunks).")
    print(f"ChromaDB collection '{USER_INPUT_COLLECTION_NAME}' ready ({user_input_collection.count()} user inputs).")

    # Initialize graph store
    from graph import GraphStore
    graph_store = GraphStore()
    gs = graph_store.stats()
    print(f"Graph store ready ({gs['total_nodes']} nodes, {gs['total_edges']} edges).")

    # Start background queue worker
    worker_task = asyncio.create_task(queue_worker())
    print("Queue worker started.")

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="memory-server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    text: str
    user_text: Optional[str] = None
    session_id: str
    timestamp: str  # ISO 8601
    project: Optional[str] = None
    turn_number: Optional[int] = None
    branch: Optional[str] = None


class IngestRequest(BaseModel):
    chunks: list[Chunk]


class SummaryRequest(BaseModel):
    text: str
    session_id: str
    timestamp: str
    project: Optional[str] = None


class SearchRequest(BaseModel):
    q: str
    k: int = 10
    project: Optional[str] = None
    session_id: Optional[str] = None
    exclude_session_id: Optional[str] = None


class SearchSubchunksRequest(BaseModel):
    q: str
    k: int = 10
    project: Optional[str] = None
    session_id: Optional[str] = None
    exclude_session_id: Optional[str] = None


class SearchUserInputsRequest(BaseModel):
    q: str
    k: int = 10
    project: Optional[str] = None
    exclude_session_id: Optional[str] = None


class SearchResult(BaseModel):
    text: str
    session_id: str
    timestamp: str
    project: Optional[str] = None
    turn_number: Optional[int] = None
    branch: Optional[str] = None
    distance: float
    chunk_type: str


class GraphSearchRequest(BaseModel):
    q: str
    k: int = 10
    expand_neighbors: bool = True
    node_type: Optional[str] = None


class EmbedRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", status_code=201)
async def ingest(req: IngestRequest):
    """Accept conversation chunks into the queue for async embedding."""
    if not req.chunks:
        return {"queued": 0}

    incoming = Path(INCOMING_DIR)
    for chunk in req.chunks:
        turn = chunk.turn_number if chunk.turn_number is not None else -1
        filename = f"{chunk.session_id}_{turn}.json"
        item = {
            "text": chunk.text,
            "user_text": chunk.user_text or "",
            "session_id": chunk.session_id,
            "timestamp": chunk.timestamp,
            "project": chunk.project or "",
            "turn_number": turn,
            "branch": chunk.branch or "",
            "chunk_type": "turn_pair",
        }
        # Write to temp file then rename for atomicity
        # Deterministic filename = natural dedup at queue level
        tmp_path = incoming / f".{filename}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(item, f)
        tmp_path.rename(incoming / filename)

    return {"queued": len(req.chunks)}


@app.post("/ingest_summary", status_code=201)
async def ingest_summary(req: SummaryRequest):
    """Accept a session summary into the queue for async embedding."""
    incoming = Path(INCOMING_DIR)
    filename = f"{req.session_id}_summary.json"
    item = {
        "text": req.text,
        "session_id": req.session_id,
        "timestamp": req.timestamp,
        "project": req.project or "",
        "turn_number": -1,
        "branch": "",
        "chunk_type": "session_summary",
    }
    tmp_path = incoming / f".{filename}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(item, f)
    tmp_path.rename(incoming / filename)

    return {"queued": 1}


def _do_search(q: str, k: int, project: Optional[str],
               session_id: Optional[str] = None,
               exclude_session_id: Optional[str] = None) -> dict:
    """Shared search logic for GET and POST endpoints."""
    query_text = SEARCH_PREFIX + q
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()

    where_clauses = []
    if project:
        where_clauses.append({"project": project})
    if session_id:
        where_clauses.append({"session_id": session_id})
    if exclude_session_id:
        where_clauses.append({"session_id": {"$ne": exclude_session_id}})

    where_filter = None
    if len(where_clauses) == 1:
        where_filter = where_clauses[0]
    elif len(where_clauses) > 1:
        where_filter = {"$and": where_clauses}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k * MMR_OVERFETCH,
        where=where_filter,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    items = []
    if results["documents"] and results["documents"][0]:
        selected = mmr_rerank(results["embeddings"][0], results["distances"][0], k)
        for idx in selected:
            doc = results["documents"][0][idx]
            meta = results["metadatas"][0][idx]
            dist = results["distances"][0][idx]
            items.append(SearchResult(
                text=doc,
                session_id=meta["session_id"],
                timestamp=meta["timestamp"],
                project=meta.get("project", ""),
                turn_number=meta.get("turn_number"),
                branch=meta.get("branch", ""),
                distance=dist,
                chunk_type=meta.get("chunk_type", "unknown"),
            ))

    resp = {"results": [item.model_dump() for item in items]}

    # If session_id filter was used and no results, check if the session exists
    if not items and session_id:
        try:
            existing = collection.get(
                where={"session_id": session_id},
                limit=1,
                include=[],
            )
            if not existing["ids"]:
                resp["warning"] = f"No session found with id '{session_id}'"
            else:
                resp["warning"] = f"Session '{session_id}' exists but no results matched your query"
        except Exception:
            pass

    return resp


@app.get("/search")
async def search_get(
    q: str = Query(..., description="Search query"),
    k: int = Query(10, ge=1, le=100, description="Number of results"),
    project: Optional[str] = Query(None, description="Filter by project path"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    exclude_session_id: Optional[str] = Query(None, description="Exclude this session ID"),
):
    """Search via GET (convenience for curl)."""
    return _do_search(q, k, project, session_id, exclude_session_id)


@app.post("/search")
async def search_post(req: SearchRequest):
    """Search via POST (for long queries)."""
    return _do_search(req.q, req.k, req.project, req.session_id, req.exclude_session_id)


@app.post("/search_subchunks")
async def search_subchunks(req: SearchSubchunksRequest):
    """Search the subchunks collection for fine-grained context."""
    query_text = SEARCH_PREFIX + req.q
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()

    where_clauses = []
    if req.project:
        where_clauses.append({"project": req.project})
    if req.session_id:
        where_clauses.append({"session_id": req.session_id})
    if req.exclude_session_id:
        where_clauses.append({"session_id": {"$ne": req.exclude_session_id}})

    where_filter = None
    if len(where_clauses) == 1:
        where_filter = where_clauses[0]
    elif len(where_clauses) > 1:
        where_filter = {"$and": where_clauses}

    # Overfetch extra to account for per-turn dedup
    results = subchunk_collection.query(
        query_embeddings=query_embedding,
        n_results=req.k * MMR_OVERFETCH * 2,
        where=where_filter,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    items = []
    if results["documents"] and results["documents"][0]:
        selected = mmr_rerank(results["embeddings"][0], results["distances"][0],
                              req.k * 2)  # get extra for dedup

        # Per-turn dedup: keep only the best-scoring subchunk per parent_chunk_id
        seen_parents = set()
        for idx in selected:
            meta = results["metadatas"][0][idx]
            parent_id = meta.get("parent_chunk_id", "")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            doc = results["documents"][0][idx]
            dist = results["distances"][0][idx]
            items.append({
                "text": doc,
                "session_id": meta["session_id"],
                "timestamp": meta["timestamp"],
                "project": meta.get("project", ""),
                "turn_number": meta.get("turn_number"),
                "branch": meta.get("branch", ""),
                "distance": dist,
                "chunk_type": meta.get("chunk_type", "subchunk"),
                "parent_chunk_id": parent_id,
                "window_index": meta.get("window_index", 0),
            })
            if len(items) >= req.k:
                break

    resp = {"results": items}

    # If session_id filter was used and no results, check if the session exists at all
    if not items and req.session_id:
        try:
            existing = subchunk_collection.get(
                where={"session_id": req.session_id},
                limit=1,
                include=[],
            )
            if not existing["ids"]:
                resp["warning"] = f"No session found with id '{req.session_id}'"
            else:
                resp["warning"] = f"Session '{req.session_id}' exists but no results matched your query"
        except Exception:
            pass

    return resp


@app.post("/search_user_inputs")
async def search_user_inputs(req: SearchUserInputsRequest):
    """Search the user_inputs collection — matches prompt against past user prompts."""
    query_text = SEARCH_PREFIX + req.q
    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()

    where_clauses = []
    if req.project:
        where_clauses.append({"project": req.project})
    if req.exclude_session_id:
        where_clauses.append({"session_id": {"$ne": req.exclude_session_id}})

    where_filter = None
    if len(where_clauses) == 1:
        where_filter = where_clauses[0]
    elif len(where_clauses) > 1:
        where_filter = {"$and": where_clauses}

    results = user_input_collection.query(
        query_embeddings=query_embedding,
        n_results=req.k * MMR_OVERFETCH,
        where=where_filter,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    items = []
    if results["documents"] and results["documents"][0]:
        selected = mmr_rerank(results["embeddings"][0], results["distances"][0], req.k)
        for idx in selected:
            doc = results["documents"][0][idx]
            meta = results["metadatas"][0][idx]
            dist = results["distances"][0][idx]
            items.append({
                "text": doc,
                "session_id": meta["session_id"],
                "timestamp": meta["timestamp"],
                "project": meta.get("project", ""),
                "turn_number": meta.get("turn_number"),
                "branch": meta.get("branch", ""),
                "distance": dist,
                "chunk_type": meta.get("chunk_type", "user_input"),
                "parent_chunk_id": meta.get("parent_chunk_id", ""),
            })

    return {"results": items}


@app.post("/search_graph")
async def search_graph(req: GraphSearchRequest):
    """Search the graph memory layer."""
    query_text = SEARCH_PREFIX + req.q
    query_embedding = model.encode([query_text], show_progress_bar=False)[0]

    results = graph_store.search(
        query_embedding, k=req.k, expand_neighbors=req.expand_neighbors,
    )

    # Optional type post-filter
    if req.node_type:
        results = [r for r in results if r.get("type") == req.node_type]

    return {"results": results}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Return the embedding for a text string. Used by dream.py."""
    embedding = model.encode(
        [DOCUMENT_PREFIX + req.text], show_progress_bar=False,
    )[0]
    return {"embedding": embedding.tolist()}


@app.post("/graph/reload_cache")
async def reload_graph_cache():
    """Rebuild the graph store's in-memory embedding cache."""
    graph_store._rebuild_cache()
    s = graph_store.stats()
    return {"status": "ok", "nodes": s["total_nodes"], "edges": s["total_edges"]}


@app.get("/stats")
async def stats():
    """Return collection statistics."""
    incoming = Path(INCOMING_DIR)
    pending = len(list(incoming.glob("*.json")))
    failed_dir = incoming.parent / "failed"
    failed = len(list(failed_dir.glob("*.json"))) if failed_dir.exists() else 0

    gs = graph_store.stats() if graph_store else {}

    return {
        "total_documents": collection.count(),
        "total_subchunks": subchunk_collection.count(),
        "total_user_inputs": user_input_collection.count(),
        "collection_name": COLLECTION_NAME,
        "subchunk_collection_name": SUBCHUNK_COLLECTION_NAME,
        "user_input_collection_name": USER_INPUT_COLLECTION_NAME,
        "queue_pending": pending,
        "queue_failed": failed,
        "graph_nodes": gs.get("total_nodes", 0),
        "graph_vibes": gs.get("nodes_by_type", {}).get("vibe", 0),
        "graph_details": gs.get("nodes_by_type", {}).get("detail", 0),
        "graph_edges": gs.get("total_edges", 0),
        "graph_activated_edges": gs.get("activated_edges", 0),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
