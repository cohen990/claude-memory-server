"""Memory server — FastAPI + ChromaDB + nomic-embed-text-v1.5.

Runs on the secondary machine. Embeds conversation chunks and stores
them in a persistent ChromaDB collection. Provides /ingest, /ingest_summary,
and /search endpoints.

Ingestion is async: /ingest writes chunks to an incoming/ directory and
returns 201 immediately. A background worker embeds and stores them.
"""

import asyncio
import json
import multiprocessing
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from surprisal import tokenize, should_retrieve


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
# Embed worker process — runs in a separate process to avoid GIL contention
# ---------------------------------------------------------------------------

_worker_model = None


def _init_embed_worker():
    """Initialize the embedding model in the worker process."""
    global _worker_model
    from sentence_transformers import SentenceTransformer
    _worker_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    _worker_model.truncate_dim = 768
    print(f"Embed worker process initialized (device={DEVICE}, pid={os.getpid()})")


def _embed_one_file_in_worker(path: Path) -> dict:
    """Read a queued chunk file and compute embeddings in the worker process.

    Identical logic to the old _embed_one_file, but uses the worker-local model.
    """
    from surprisal import tokenize as _tokenize

    with open(path) as f:
        item = json.load(f)

    text = item["text"]
    embedding = _worker_model.encode([DOCUMENT_PREFIX + text], show_progress_bar=False).tolist()

    session_id = item.get("session_id", "")
    turn_number = item.get("turn_number", -1)
    doc_id = f"{session_id}_{turn_number}"
    metadata = {
        "session_id": session_id,
        "timestamp": item.get("timestamp", ""),
        "project": item.get("project", ""),
        "turn_number": turn_number,
        "branch": item.get("branch", ""),
        "chunk_type": item.get("chunk_type", "turn_pair"),
        "dreamed": 0,
    }

    result = {
        "doc_id": doc_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata,
        "session_id": session_id,
        "turn_number": turn_number,
    }

    # Compute subchunk embeddings
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
        sc_embeddings = _worker_model.encode(
            [DOCUMENT_PREFIX + t for t in sc_documents],
            show_progress_bar=False,
        ).tolist()
        result["subchunks"] = {
            "ids": sc_ids,
            "embeddings": sc_embeddings,
            "documents": sc_documents,
            "metadatas": sc_metadatas,
        }

    # Compute user input embedding
    user_text = item.get("user_text", "")
    if user_text:
        ui_embedding = _worker_model.encode(
            [DOCUMENT_PREFIX + user_text], show_progress_bar=False
        ).tolist()
        result["user_input"] = {
            "id": f"{doc_id}_ui",
            "embedding": ui_embedding,
            "text": user_text,
            "metadata": {
                "session_id": session_id,
                "timestamp": metadata["timestamp"],
                "project": metadata["project"],
                "turn_number": turn_number,
                "branch": metadata["branch"],
                "chunk_type": "user_input",
                "parent_chunk_id": doc_id,
            },
        }
        result["user_words"] = _tokenize(user_text)

    return result


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
embed_pool: ProcessPoolExecutor = None  # type: ignore[assignment]
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



def _store_one_file(data: dict, path: Path):
    """Upsert pre-computed embeddings into ChromaDB. Runs on the event loop thread.

    ChromaDB is not thread-safe — all access must be serialized on the same thread.
    """
    collection.upsert(
        ids=[data["doc_id"]],
        embeddings=data["embedding"],
        documents=[data["text"]],
        metadatas=[data["metadata"]],
    )

    if "subchunks" in data:
        sc = data["subchunks"]
        subchunk_collection.upsert(
            ids=sc["ids"],
            embeddings=sc["embeddings"],
            documents=sc["documents"],
            metadatas=sc["metadatas"],
        )

    if "user_input" in data:
        ui = data["user_input"]
        user_input_collection.upsert(
            ids=[ui["id"]],
            embeddings=ui["embedding"],
            documents=[ui["text"]],
            metadatas=[ui["metadata"]],
        )

        words = data.get("user_words", [])
        if words and graph_store:
            graph_store.update_personal_word_counts(words)

    path.unlink()


async def queue_worker():
    """Background loop that processes the incoming queue.

    Embedding runs in a separate process (ProcessPoolExecutor) so the GIL
    in the worker can't block the server's event loop. ChromaDB upserts run
    on the event loop thread — ChromaDB's PersistentClient is not thread-safe.
    """
    incoming = Path(INCOMING_DIR)
    processed_total = 0
    loop = asyncio.get_event_loop()
    while True:
        try:
            files = sorted(incoming.glob("*.json"))
            if files:
                print(f"Queue: {len(files)} pending")
            for i, path in enumerate(files):
                try:
                    # Embed in worker process (separate GIL)
                    data = await loop.run_in_executor(
                        embed_pool, _embed_one_file_in_worker, path,
                    )
                    # Store in ChromaDB on event loop thread (not thread-safe)
                    _store_one_file(data, path)
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
    global model, collection, subchunk_collection, user_input_collection, worker_task, graph_store, embed_pool

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
    graph_store = GraphStore(check_same_thread=False)
    gs = graph_store.stats()
    print(f"Graph store ready ({gs['total_nodes']} nodes, {gs['total_edges']} edges).")

    # Start embed worker process pool (separate process = separate GIL)
    # Use spawn context — fork is not safe with MPS/Metal on macOS
    ctx = multiprocessing.get_context("spawn")
    embed_pool = ProcessPoolExecutor(max_workers=1, initializer=_init_embed_worker, mp_context=ctx)
    print("Embed worker process pool started.")

    # Start background queue worker
    worker_task = asyncio.create_task(queue_worker())
    print("Queue worker started.")

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    embed_pool.shutdown(wait=False)


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
    session_id: Optional[str] = None
    min_similarity: Optional[float] = None
    general_surprisal: Optional[float] = None
    personal_surprisal: Optional[float] = None


class ListRecallsRequest(BaseModel):
    session_id: Optional[str] = None
    limit: int = 1


class ReflectOnRecallRequest(BaseModel):
    recall_id: str
    reflections: str  # comma-separated codes like "U,I,N,N,M"


class DeleteRequest(BaseModel):
    session_ids: list[str] | None = None
    project: str | None = None
    dry_run: bool = True  # default to dry run for safety


class MarkDreamedRequest(BaseModel):
    ids: list[str]
    metadatas: list[dict]


class EmbedRequest(BaseModel):
    text: str


class SurprisalRequest(BaseModel):
    text: str


class ContestNodeRequest(BaseModel):
    node_id_prefix: str
    correction: str


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

    # Optional post-filters
    if req.min_similarity is not None:
        results = [r for r in results if r.get("similarity", 0) >= req.min_similarity]
    if req.node_type:
        results = [r for r in results if r.get("type") == req.node_type]

    # Store recall for reflection feedback — only stores results that survived
    # filtering, so recall_results count matches what the caller actually sees.
    recall_id = None
    if results:
        recall_id = graph_store.create_recall(
            np.asarray(query_embedding, dtype=np.float32), results,
            session_id=req.session_id, query_text=req.q,
            general_surprisal=req.general_surprisal,
            personal_surprisal=req.personal_surprisal,
        )

    return {"results": results, "recall_id": recall_id}


@app.post("/reflect_on_recall")
async def reflect_on_recall(req: ReflectOnRecallRequest):
    """Record reflections on the results of a previous graph search recall."""
    codes = [c.strip() for c in req.reflections.split(",")]

    valid_codes = {"U", "I", "N", "D", "M"}
    for c in codes:
        if c not in valid_codes:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid reflection code: {c!r}. Valid: {valid_codes}"},
            )

    # Verify recall exists and check result count
    with graph_store._db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM recall_results WHERE recall_id = ?",
            (req.recall_id,),
        ).fetchone()
    if not row or row[0] == 0:
        return JSONResponse(
            status_code=404,
            content={"error": f"Recall {req.recall_id} not found"},
        )
    if len(codes) != row[0]:
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected {row[0]} reflections, got {len(codes)}"},
        )

    graph_store.reflect_on_recall(req.recall_id, codes)
    return {"status": "ok", "recall_id": req.recall_id, "reflected": len(codes)}


@app.post("/reflect_on_node")
async def reflect_on_node(req: dict):
    """Update the reflection for a single node within a recall.

    Body: {"recall_id": str, "node_id_prefix": str, "reflection": str}
    """
    recall_id = req.get("recall_id", "")
    node_id_prefix = req.get("node_id_prefix", "")
    reflection = req.get("reflection", "")

    valid_codes = {"U", "I", "N", "D", "M"}
    if reflection not in valid_codes:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid reflection code: {reflection!r}. Valid: {valid_codes}"},
        )

    try:
        found = graph_store.reflect_on_node(recall_id, node_id_prefix, reflection)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not found:
        return JSONResponse(
            status_code=404,
            content={"error": f"No result matching prefix {node_id_prefix!r} in recall {recall_id}"},
        )

    return {"status": "ok", "recall_id": recall_id, "node_id_prefix": node_id_prefix}


@app.post("/list_recalls")
async def list_recalls(req: ListRecallsRequest):
    """List recent recalls with full details for the /memories skill."""
    recalls = graph_store.list_recalls(
        session_id=req.session_id, limit=req.limit,
    )
    return {"recalls": recalls}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Return the embedding for a text string. Used by dream.py."""
    embedding = model.encode(
        [DOCUMENT_PREFIX + req.text], show_progress_bar=False,
    )[0]
    return {"embedding": embedding.tolist()}


@app.post("/surprisal")
async def surprisal(req: SurprisalRequest):
    """Compute surprisal scores for query text against both corpora.

    Used by the prompt hook to gate memory retrieval.
    """
    words = tokenize(req.text)
    english_probs = graph_store.get_english_log_probs(words)
    personal_counts = graph_store.get_personal_word_counts(words)
    corpus_total = graph_store.personal_corpus_total()
    vocab_size = graph_store.personal_vocab_size()

    result = should_retrieve(
        req.text, english_probs, personal_counts, corpus_total, vocab_size,
    )
    return result


@app.get("/chunks/undreamed")
async def chunks_undreamed(
    days: Optional[int] = Query(None, ge=1, description="Limit to last N days"),
    limit: int = Query(10000, ge=1, description="Max chunks to return"),
):
    """Return conversation chunks that haven't been processed by dream yet."""
    total = collection.count()
    if total == 0:
        return {"chunks": [], "total": 0}

    results = collection.get(
        where={"dreamed": {"$ne": 1}},
        include=["documents", "metadatas"],
        limit=total,  # get all undreamed, apply limit after optional date filter
    )

    chunks = []
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        text = results["documents"][i]

        if days is not None:
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            ts = meta.get("timestamp", "")
            if ts < cutoff.isoformat():
                continue

        chunks.append({"id": doc_id, "text": text, "metadata": meta})
        if len(chunks) >= limit:
            break

    return {"chunks": chunks, "total": len(chunks)}


@app.post("/chunks/mark_dreamed")
async def chunks_mark_dreamed(req: MarkDreamedRequest):
    """Mark chunks as dreamed by updating their metadata."""
    if not req.ids:
        return {"marked": 0}
    collection.update(ids=req.ids, metadatas=req.metadatas)
    return {"marked": len(req.ids)}


@app.post("/chunks/by_ids")
async def chunks_by_ids(req: dict):
    """Fetch specific chunks by their IDs."""
    ids = req.get("ids", [])
    if not ids:
        return {"chunks": []}
    results = collection.get(ids=ids, include=["documents", "metadatas"])
    chunks = []
    for i, doc_id in enumerate(results["ids"]):
        chunks.append({
            "id": doc_id,
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return {"chunks": chunks}


@app.post("/graph/reload_cache")
async def reload_graph_cache():
    """Rebuild the graph store's in-memory embedding cache."""
    graph_store._rebuild_cache()
    s = graph_store.stats()
    return {"status": "ok", "nodes": s["total_nodes"], "edges": s["total_edges"]}


@app.post("/graph/recompute_layout")
async def recompute_layout():
    """Recompute graph layout positions and cache them."""
    positions = graph_store.compute_layout()
    return {"status": "ok", "nodes_positioned": len(positions)}


# ---------------------------------------------------------------------------
# Graph browse endpoints (read-only, used by browse.py)
# ---------------------------------------------------------------------------

@app.get("/graph/full")
async def graph_full():
    """Full graph for Cytoscape.js initial load."""
    return graph_store.get_full_graph()


@app.get("/graph/nodes")
async def graph_list_nodes(
    type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Paginated node listing."""
    return graph_store.list_nodes(node_type=type, limit=limit, offset=offset)


@app.get("/graph/nodes/{node_id}")
async def graph_get_node(node_id: str):
    """Single node with edges, no embedding."""
    node = graph_store.get_node(node_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": "Node not found"})
    node.pop("embedding", None)
    edges = graph_store.get_edges(node_id)
    return {"node": node, "edges": edges}


@app.get("/graph/nodes/{node_id}/neighbors")
async def graph_get_neighbors(node_id: str):
    """Edges + hydrated neighbor nodes."""
    node = graph_store.get_node(node_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": "Node not found"})

    edges = graph_store.get_edges(node_id)
    neighbors = []
    for edge in edges:
        neighbor_id = (
            edge["target_id"] if edge["source_id"] == node_id
            else edge["source_id"]
        )
        neighbor = graph_store.get_node(neighbor_id)
        if neighbor:
            neighbor.pop("embedding", None)
            neighbors.append({"node": neighbor, "edge": edge})

    return {"neighbors": neighbors}


@app.post("/graph/contest")
async def contest_node(req: ContestNodeRequest):
    """Mark a graph node as contested with a proposed correction."""
    try:
        result = graph_store.contest_node(req.node_id_prefix, req.correction)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return {
        "status": "contested",
        "node_id": result["node_id"],
        "current_text": result["text"],
        "correction": req.correction,
    }


@app.get("/graph/recalls")
async def graph_list_recalls(
    session_id: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """Recent recalls with full details."""
    return {"recalls": graph_store.list_recalls(session_id=session_id, limit=limit)}


@app.get("/graph/reflections")
async def graph_reflection_distribution():
    """Reflection distribution across all recalls."""
    return graph_store.reflection_distribution()


@app.get("/graph/reflection-timeline")
async def graph_reflection_timeline():
    """Reflection counts bucketed by hour for timeline chart."""
    return graph_store.reflection_timeline()


class CreateMarkerRequest(BaseModel):
    label: str


@app.post("/graph/marker")
async def create_marker(req: CreateMarkerRequest):
    """Create a timestamp marker for chart annotations."""
    return graph_store.create_marker(req.label)


@app.get("/graph/markers")
async def list_markers():
    """List all markers."""
    return graph_store.list_markers()


@app.get("/graph/dream-runs")
async def graph_dream_runs(
    limit: int = Query(20, ge=1, le=100),
):
    """Recent dream runs."""
    from graph import DreamLog
    dream_log = DreamLog(graph_store)
    return {"runs": dream_log.list_runs(limit=limit)}


@app.get("/graph/dream-runs/{run_id}/operations")
async def graph_dream_run_operations(run_id: str):
    """Operations for a specific dream run."""
    from graph import DreamLog
    dream_log = DreamLog(graph_store)
    return {"operations": dream_log.get_run_operations(run_id)}


@app.post("/delete")
async def delete_chunks(req: DeleteRequest):
    """Delete chunks by session IDs or project.

    Removes matching documents from all three ChromaDB collections
    (conversations, subchunks, user_inputs) and cleans up graph nodes
    whose source_ids become empty after removal.

    Defaults to dry_run=true — set dry_run=false to actually delete.
    """
    if not req.session_ids and not req.project:
        return JSONResponse(
            status_code=400,
            content={"error": "Must specify session_ids or project"},
        )

    # Build where filter
    if req.session_ids:
        if len(req.session_ids) == 1:
            where = {"session_id": req.session_ids[0]}
        else:
            where = {"session_id": {"$in": req.session_ids}}
    else:
        where = {"project": req.project}

    # Find matching IDs in each collection
    chunk_ids = collection.get(where=where, include=[])["ids"]
    subchunk_ids = subchunk_collection.get(where=where, include=[])["ids"]
    user_input_ids = user_input_collection.get(where=where, include=[])["ids"]

    result = {
        "chunks": len(chunk_ids),
        "subchunks": len(subchunk_ids),
        "user_inputs": len(user_input_ids),
        "dry_run": req.dry_run,
    }

    # Check graph nodes that reference these chunk IDs
    orphaned_nodes = []
    if graph_store and chunk_ids:
        with graph_store._db() as conn:
            cursor = conn.execute(
                "SELECT id, source_ids FROM nodes"
            )
            for node_id, source_ids_json in cursor:
                source_ids = json.loads(source_ids_json) if source_ids_json else []
                remaining = [s for s in source_ids if s not in set(chunk_ids)]
                if not remaining and source_ids:
                    orphaned_nodes.append(node_id)
        result["orphaned_graph_nodes"] = len(orphaned_nodes)

    if req.dry_run:
        return result

    # Actually delete
    if chunk_ids:
        # ChromaDB .delete() has a batch limit; chunk in groups
        for i in range(0, len(chunk_ids), 500):
            collection.delete(ids=chunk_ids[i:i+500])
    if subchunk_ids:
        for i in range(0, len(subchunk_ids), 500):
            subchunk_collection.delete(ids=subchunk_ids[i:i+500])
    if user_input_ids:
        for i in range(0, len(user_input_ids), 500):
            user_input_collection.delete(ids=user_input_ids[i:i+500])

    # Remove orphaned graph nodes and their edges
    if graph_store and orphaned_nodes:
        with graph_store._db() as conn:
            for node_id in orphaned_nodes:
                conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                             (node_id, node_id))
                conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.commit()
        # Rebuild the in-memory cache
        graph_store._rebuild_cache()
        result["graph_nodes_deleted"] = len(orphaned_nodes)

    # Also remove recalls for deleted sessions
    if graph_store and req.session_ids:
        with graph_store._db() as conn:
            for sid in req.session_ids:
                conn.execute("DELETE FROM recall_results WHERE recall_id IN "
                             "(SELECT recall_id FROM recalls WHERE session_id = ?)", (sid,))
                conn.execute("DELETE FROM recalls WHERE session_id = ?", (sid,))
            conn.commit()

    return result


@app.get("/stats")
async def stats():
    """Return collection statistics."""
    incoming = Path(INCOMING_DIR)
    pending = len(list(incoming.glob("*.json")))
    failed_dir = incoming.parent / "failed"
    failed = len(list(failed_dir.glob("*.json"))) if failed_dir.exists() else 0

    gs = graph_store.stats() if graph_store else {}
    pending_dream = graph_store.pending_dream_count() if graph_store else 0

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
        "pending_dream": pending_dream,
    }


@app.get("/dream/health")
async def dream_health():
    """Lightweight dream health check for the prompt hook."""
    from graph import DreamLog
    dream_log = DreamLog(graph_store)
    runs = dream_log.list_runs(limit=5)

    stuck = [r for r in runs if r["finished_at"] is None]
    # Check if the most recent consolidation failed (sort by finished_at
    # since started_at can be misleading with concurrent/killed runs)
    consolidations = [r for r in runs if r["type"] == "consolidate"
                      and r["finished_at"]]
    consolidations.sort(key=lambda r: r["finished_at"], reverse=True)
    last_consolidation = consolidations[0] if consolidations else None

    status = "ok"
    message = None
    if stuck:
        status = "stuck"
        oldest = min(r["started_at"] for r in stuck)
        message = f"{len(stuck)} dream run(s) started but never finished (oldest: {oldest})"
    elif last_consolidation and last_consolidation.get("error"):
        status = "failing"
        message = f"Last consolidation failed: {last_consolidation['error']}"

    return {"status": status, "message": message}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
