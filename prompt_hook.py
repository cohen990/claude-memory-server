"""UserPromptSubmit hook — searches memory for context relevant to the user's prompt.

Injects compact metadata summaries so Claude can decide whether to fetch full details
via the search_memory MCP tool.

Reads hook JSON from stdin, searches the memory server, prints results to stdout
if any pass the relevance threshold.
"""

import json
import os
import sys
import urllib.request
import urllib.error

SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")
DISTANCE_THRESHOLD = float(os.environ.get("MEMORY_DISTANCE_THRESHOLD", "0.5"))
MAX_RESULTS = int(os.environ.get("MEMORY_MAX_RESULTS", "5"))


def search(query: str, k: int = MAX_RESULTS, project: str = "",
           exclude_session_id: str = "") -> list[dict]:
    body = {"q": query, "k": k}
    if project:
        body["project"] = project
    if exclude_session_id:
        body["exclude_session_id"] = exclude_session_id

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/search_user_inputs",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("results", [])
    except Exception:
        return []


def search_graph(query: str, k: int = 3) -> list[dict]:
    """Search the graph memory layer for synthesized long-term memories."""
    body = {"q": query, "k": k, "expand_neighbors": True}
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/search_graph",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read()).get("results", [])
    except Exception:
        return []


def first_line(text: str, max_len: int = 150) -> str:
    """Extract first meaningful line from a turn pair, truncated."""
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("Assistant:"):
            # Strip "User: " prefix for compactness
            if line.startswith("User: "):
                line = line[6:]
            if len(line) > max_len:
                return line[:max_len] + "..."
            return line
    return text[:max_len]


def main():
    import urllib.parse

    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        return

    prompt = hook_input.get("prompt", "")
    if not prompt or len(prompt) < 10:
        return

    # Skip slash commands
    if prompt.lstrip().startswith("/"):
        return

    current_session = hook_input.get("session_id", "")
    results = search(prompt, exclude_session_id=current_session)
    if not results:
        return

    relevant = [r for r in results if r["distance"] < DISTANCE_THRESHOLD]

    lines = []
    if relevant:
        lines.append(f"[memory] {len(relevant)} result{'s' if len(relevant) != 1 else ''} from past conversations:")
        if current_session:
            lines.append(f"  (current session: {current_session} — pass as exclude_session_id to avoid self-references)")
        for r in relevant:
            ts = r.get("timestamp", "")[:10]  # just the date
            sid = r.get("session_id", "")
            dist = r.get("distance", 1.0)
            preview = first_line(r.get("text", ""))
            lines.append(f'  {dist:.2f} | {ts} | {sid} | "{preview}"')

    # Graph memory search
    graph_results = search_graph(prompt, k=3)
    graph_relevant = [r for r in graph_results if r.get("similarity", 0) > 0.5]
    if graph_relevant:
        lines.append(f"[graph] {len(graph_relevant)} synthesized memor{'ies' if len(graph_relevant) != 1 else 'y'}:")
        for r in graph_relevant:
            ntype = r.get("type", "?")
            sim = r.get("similarity", 0)
            text = r.get("text", "")[:150]
            lines.append(f"  {sim:.2f} | [{ntype}] {text}")

    if lines:
        lines.insert(0, "--- AGENT MEMORY (not visible to user) ---")
        lines.append("These are your private memories from past sessions. The user cannot see this section.")
        lines.append("If these hints look relevant, use search_memory or search_memory_graph tools to retrieve full context.")
        lines.append("--- END AGENT MEMORY ---")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
