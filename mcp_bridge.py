"""MCP bridge — exposes search_memory tool to Claude Code.

Runs on the primary machine via stdio. Claude Code spawns this process.
Proxies search requests to the memory server over HTTP.

Configure MEMORY_SERVER_URL env var to point at the secondary machine.
"""

import os
import sys
import logging
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, Annotations

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("memory-bridge")

SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")

mcp = FastMCP("memory")


@mcp.tool()
async def search_memory(query: str, k: int = 10, project: Optional[str] = None,
                        session_id: Optional[str] = None,
                        exclude_session_id: Optional[str] = None) -> str:
    """Search past conversations for relevant context.

    Returns the top-k most similar conversation chunks from the vector store.
    Use this to recall previous discussions, decisions, solutions, and context
    from past sessions.

    Args:
        query: Natural language search query describing what you're looking for.
        k: Number of results to return (default 10, max 100).
        project: Optional project directory path to filter results.
        session_id: Optional session ID to restrict search to a specific conversation.
        exclude_session_id: Optional session ID to exclude (use your current session ID to avoid self-referential results).
    """
    body = {"q": query, "k": k}
    if project:
        body["project"] = project
    if session_id:
        body["session_id"] = session_id
    if exclude_session_id:
        body["exclude_session_id"] = exclude_session_id

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{SERVER_URL}/search", json=body)
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Cannot connect to memory server at {SERVER_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        return f"Error: Memory server returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    data = resp.json()
    results = data.get("results", [])
    warning = data.get("warning")

    if not results:
        if warning:
            return f"No results found. {warning}"
        return "No results found."

    parts = []
    if warning:
        parts.append(f"Warning: {warning}")
    for i, r in enumerate(results, 1):
        header = f"--- Result {i} (distance: {r['distance']:.3f}) ---"
        meta_parts = []
        if r.get("project"):
            meta_parts.append(f"project: {r['project']}")
        if r.get("session_id"):
            meta_parts.append(f"session: {r['session_id']}")
        if r.get("timestamp"):
            meta_parts.append(f"time: {r['timestamp']}")
        if r.get("branch"):
            meta_parts.append(f"branch: {r['branch']}")

        meta = " | ".join(meta_parts)
        parts.append(f"{header}\n{meta}\n\n{r['text']}")

    return "\n\n".join(parts)


@mcp.tool()
async def search_memory_detail(
    query: str, k: int = 10, project: Optional[str] = None,
    session_id: Optional[str] = None,
    exclude_session_id: Optional[str] = None,
) -> str:
    """Search past conversations at fine granularity (~500 char windows).

    Returns the top-k most similar subchunks — small, focused excerpts rather
    than full turn pairs. Use this for precise recall of specific details,
    decisions, or code snippets.

    Args:
        query: Natural language search query describing what you're looking for.
        k: Number of results to return (default 10, max 100).
        project: Optional project directory path to filter results.
        session_id: Optional session ID to restrict search to a specific conversation.
        exclude_session_id: Optional session ID to exclude (use your current session ID to avoid self-referential results).
    """
    body = {"q": query, "k": k}
    if project:
        body["project"] = project
    if session_id:
        body["session_id"] = session_id
    if exclude_session_id:
        body["exclude_session_id"] = exclude_session_id

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{SERVER_URL}/search_subchunks", json=body)
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Cannot connect to memory server at {SERVER_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        return f"Error: Memory server returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    data = resp.json()
    results = data.get("results", [])
    warning = data.get("warning")

    if not results:
        if warning:
            return f"No results found. {warning}"
        return "No results found."

    parts = []
    if warning:
        parts.append(f"Warning: {warning}")
    for i, r in enumerate(results, 1):
        header = f"--- Result {i} (distance: {r['distance']:.3f}) ---"
        meta_parts = []
        if r.get("project"):
            meta_parts.append(f"project: {r['project']}")
        if r.get("session_id"):
            meta_parts.append(f"session: {r['session_id']}")
        if r.get("timestamp"):
            meta_parts.append(f"time: {r['timestamp']}")
        if r.get("branch"):
            meta_parts.append(f"branch: {r['branch']}")

        meta = " | ".join(meta_parts)
        parts.append(f"{header}\n{meta}\n\n{r['text']}")

    return "\n\n".join(parts)


@mcp.tool()
async def search_memory_graph(
    query: str, k: int = 10, expand_neighbors: bool = True,
    node_type: str | None = None,
) -> str:
    """Search the graph memory layer for synthesized long-term memories.

    The graph contains vibes (high-level themes) and details (specific facts)
    connected by weighted edges. Search walks the graph neighborhood —
    retrieval strengthens associations for future consolidation.

    Args:
        query: Natural language search query describing what you're looking for.
        k: Number of results to return (default 10, max 100).
        expand_neighbors: Whether to include graph neighbors of top results (default True).
        node_type: Optional filter: "vibe" or "detail".
    """
    body: dict = {"q": query, "k": k, "expand_neighbors": expand_neighbors}
    if node_type:
        body["node_type"] = node_type

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{SERVER_URL}/search_graph", json=body)
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Cannot connect to memory server at {SERVER_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        return f"Error: Memory server returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    data = resp.json()
    results = data.get("results", [])

    if not results:
        return "No graph memory results found."

    parts = []
    for i, r in enumerate(results, 1):
        node_type_str = r.get("type", "unknown")
        sim = r.get("similarity", 0)
        score = r.get("score", 0)
        source = r.get("source", "")

        header = f"--- Result {i} [{node_type_str}] (sim: {sim:.3f}, score: {score:.3f}, via: {source}) ---"
        extras = []
        if r.get("edge_weight") is not None:
            extras.append(f"edge_weight: {r['edge_weight']:.2f}")
        if r.get("connected_via"):
            extras.append(f"connected_via: {r['connected_via'][:8]}...")
        source_ids = r.get("source_ids", [])
        if source_ids:
            extras.append(f"sources: {', '.join(source_ids[:5])}" +
                         (f" (+{len(source_ids) - 5} more)" if len(source_ids) > 5 else ""))

        meta = " | ".join(extras) if extras else ""
        text = r.get("text", "")
        if meta:
            parts.append(f"{header}\n{meta}\n\n{text}")
        else:
            parts.append(f"{header}\n\n{text}")

    return "\n\n".join(parts)


@mcp.tool()
async def memory_stats() -> str:
    """Return statistics about the memory store (total documents, etc.)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SERVER_URL}/stats")
            resp.raise_for_status()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    data = resp.json()
    lines = [
        f"Total documents: {data['total_documents']}",
        f"Total subchunks: {data.get('total_subchunks', 'N/A')}",
        f"Total user inputs: {data.get('total_user_inputs', 'N/A')}",
        f"Collection: {data['collection_name']}",
        f"Graph nodes: {data.get('graph_nodes', 0)} (vibes: {data.get('graph_vibes', 0)}, details: {data.get('graph_details', 0)})",
        f"Graph edges: {data.get('graph_edges', 0)} (activated: {data.get('graph_activated_edges', 0)})",
    ]
    return "\n".join(lines)


@mcp.tool()
async def list_recalls(session_id: str, limit: int = 1) -> str:
    """List recent memory recalls for a session, with full details and ratings.

    Shows what memories were injected by the prompt hook and how they were rated.
    Use with the /memories skill to let the user see their memory activity.

    Args:
        session_id: The session ID to filter recalls for.
        limit: Number of recent recalls to return (default 1).
    """
    body = {"session_id": session_id, "limit": limit}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{SERVER_URL}/list_recalls", json=body)
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Cannot connect to memory server at {SERVER_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        return f"Error: Memory server returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    data = resp.json()
    recalls = data.get("recalls", [])

    if not recalls:
        return "No recalls found for this session."

    rating_labels = {
        "U": "USED", "I": "INTERESTING", "N": "NOISE",
        "D": "DISTRACTING", "M": "MISLEADING",
    }

    parts = []
    for recall in recalls:
        ts = recall.get("created_at", "?")
        rid = recall.get("recall_id", "?")
        results = recall.get("results", [])

        lines = [f"=== Recall {rid[:12]}... at {ts} ==="]
        for i, r in enumerate(results, 1):
            ntype = r.get("type", "?")
            sim = r.get("similarity", 0)
            text = (r.get("text") or "")[:150]
            rating = r.get("rating")
            source = r.get("source", "seed")

            rating_str = f" [{rating} = {rating_labels.get(rating, '?')}]" if rating else " [unrated]"
            lines.append(f"  {i}. [{ntype}] (sim: {sim:.2f}, via: {source}){rating_str}")
            lines.append(f"     {text}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


RATING_LABELS = {
    "U": ("USED", 2),
    "I": ("INTERESTING", 1),
    "N": ("NOISE", 0),
    "D": ("DISTRACTING", -1),
    "M": ("MISLEADING", -2),
}


@mcp.tool()
async def rate_memories(ratings: str):
    """Rate memories from the most recent graph memory injection.

    Call this once per turn when graph memories are injected via the prompt hook.
    The prompt hook output includes a recall ID — pass it back with ratings.

    Format: recall_id:U,I,N,N,M
    One letter per memory, in order. Codes: U=used, I=interesting, N=noise, D=distracting, M=misleading.

    Args:
        ratings: Compact rating string like "abc123:U,I,N,N,M"
    """
    if ":" not in ratings:
        return "Error: expected format recall_id:U,I,N,N,M"

    recall_id, codes_str = ratings.split(":", 1)
    recall_id = recall_id.strip()
    codes_str = codes_str.strip()

    if not recall_id or not codes_str:
        return "Error: expected format recall_id:U,I,N,N,M"

    # Validate codes locally before sending
    codes = [c.strip() for c in codes_str.split(",")]
    valid_codes = set(RATING_LABELS.keys())
    for c in codes:
        if c not in valid_codes:
            return f"Error: invalid code {c!r}. Valid: U, I, N, D, M"

    # POST to server
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SERVER_URL}/rate_recall",
                json={"recall_id": recall_id, "ratings": codes_str},
            )
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Cannot connect to memory server at {SERVER_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code}: {e.response.json().get('error', e.response.text)}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    return TextContent(
        type="text", text="",
        annotations=Annotations(audience=["assistant"]),
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
