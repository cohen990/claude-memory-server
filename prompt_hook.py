"""UserPromptSubmit hook — searches graph memory for context relevant to the user's prompt.

Injects compact synthesized memories so Claude can decide whether to fetch full details
via the search_memory MCP tool. Source chunk IDs are included so the agent can trace
back to original conversations.

Reads hook JSON from stdin, searches the memory server, prints results to stdout
if any pass the relevance threshold.
"""

import json
import os
import sys
import urllib.request
import urllib.error

SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")


def search_graph(query: str, k: int = 5) -> list[dict]:
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
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("results", [])
    except Exception:
        return []


def main():
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

    # Graph memory search
    graph_results = search_graph(prompt, k=5)
    graph_relevant = [r for r in graph_results if r.get("similarity", 0) > 0.5]
    if not graph_relevant:
        return

    lines = ["--- AGENT MEMORY (not visible to user) ---"]
    if current_session:
        lines.append(f"(current session: {current_session} — pass as exclude_session_id to avoid self-references)")

    lines.append(f"[graph] {len(graph_relevant)} synthesized memor{'ies' if len(graph_relevant) != 1 else 'y'}:")
    for r in graph_relevant:
        node_id = r.get("id", "?")
        ntype = r.get("type", "?")
        sim = r.get("similarity", 0)
        text = r.get("text", "")[:200]
        source_ids = r.get("source_ids", [])
        lines.append(f"  {sim:.2f} | {node_id[:12]} | [{ntype}] {text}")
        if source_ids:
            # Show first few source chunk IDs for traceability
            shown = source_ids[:3]
            extra = f" (+{len(source_ids) - 3} more)" if len(source_ids) > 3 else ""
            lines.append(f"    sources: {', '.join(shown)}{extra}")

    lines.append("These are your private memories from past sessions. The user cannot see this section.")
    lines.append("Detail memories are associative recall, not authoritative records — validate specifics against source conversations before relying on them.")
    lines.append("Use search_memory or search_memory_detail tools with session_id to retrieve full source conversations.")
    lines.append("--- END AGENT MEMORY ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
