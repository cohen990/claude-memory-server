"""Memory graph browser — read-only web UI for exploring the graph.

Thin frontend server on port 8421. All data comes from the main memory
server at :8420 over HTTP — no direct database access.
"""

import os

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

MAIN_SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")
BIND_HOST = os.environ.get("BROWSE_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("BROWSE_PORT", "8421"))

app = FastAPI(title="memory-browser")
http_client = httpx.AsyncClient(base_url=MAIN_SERVER_URL, timeout=10.0)


# ---------------------------------------------------------------------------
# API proxy — forward /api/* to the main server's /graph/* and /stats
# ---------------------------------------------------------------------------

async def _proxy(path: str, params: dict | None = None) -> JSONResponse:
    """Forward a GET request to the main server."""
    try:
        resp = await http_client.get(path, params=params)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.ConnectError:
        return JSONResponse(
            status_code=502,
            content={"error": f"Cannot reach memory server at {MAIN_SERVER_URL}"},
        )


@app.get("/api/graph")
async def full_graph(request: Request):
    return await _proxy("/graph/full", dict(request.query_params))


@app.get("/api/nodes")
async def list_nodes(request: Request):
    return await _proxy("/graph/nodes", dict(request.query_params))


@app.get("/api/nodes/{node_id}")
async def get_node(node_id: str):
    return await _proxy(f"/graph/nodes/{node_id}")


@app.get("/api/nodes/{node_id}/neighbors")
async def get_neighbors(node_id: str):
    return await _proxy(f"/graph/nodes/{node_id}/neighbors")


@app.get("/api/recalls")
async def list_recalls(request: Request):
    return await _proxy("/graph/recalls", dict(request.query_params))


@app.get("/api/reflection-timeline")
async def reflection_timeline():
    return await _proxy("/graph/reflection-timeline")


@app.get("/api/dream-runs")
async def dream_runs(request: Request):
    return await _proxy("/graph/dream-runs", dict(request.query_params))


@app.get("/api/dream-runs/{run_id}/operations")
async def dream_run_operations(run_id: str):
    return await _proxy(f"/graph/dream-runs/{run_id}/operations")


@app.get("/api/stats")
async def stats():
    """Combine graph stats, reflections, and main server stats."""
    try:
        graph_resp, reflections_resp, stats_resp = await gather_stats()
    except httpx.ConnectError:
        return JSONResponse(
            status_code=502,
            content={"error": f"Cannot reach memory server at {MAIN_SERVER_URL}"},
        )

    return {
        "graph": graph_resp,
        "reflections": reflections_resp,
        "chromadb": stats_resp,
    }


async def gather_stats():
    """Fetch graph stats, reflections, and chromadb stats in parallel."""
    import asyncio
    graph_task = http_client.get("/stats")
    reflections_task = http_client.get("/graph/reflections")
    stats_task = http_client.get("/stats")

    results = await asyncio.gather(graph_task, reflections_task, stats_task,
                                   return_exceptions=True)

    def safe_json(r):
        if isinstance(r, Exception):
            return None
        if r.status_code == 200:
            return r.json()
        return None

    # /stats has both chromadb and graph info; extract graph portion
    main_stats = safe_json(results[0])
    graph_stats = None
    if main_stats:
        graph_stats = {
            "total_nodes": main_stats.get("graph_nodes", 0),
            "nodes_by_type": {
                "vibe": main_stats.get("graph_vibes", 0),
                "detail": main_stats.get("graph_details", 0),
            },
            "total_edges": main_stats.get("graph_edges", 0),
            "activated_edges": main_stats.get("graph_activated_edges", 0),
        }

    return graph_stats, safe_json(results[1]), safe_json(results[2])


# ---------------------------------------------------------------------------
# Static files + SPA fallback
# ---------------------------------------------------------------------------

_base = os.path.dirname(os.path.abspath(__file__))
dist_dir = os.path.join(_base, "frontend", "dist")
assets_dir = os.path.join(dist_dir, "assets")

if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
async def index():
    """Serve the SPA shell."""
    index_path = os.path.join(dist_dir, "index.html")
    return FileResponse(index_path, media_type="text/html")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
