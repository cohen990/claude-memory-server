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


if __name__ == "__main__":
    mcp.run(transport="stdio")
