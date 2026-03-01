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
# SURPRISAL_GATE: "1" = enforce gate, "0" = log-only (always retrieve)
SURPRISAL_GATE = os.environ.get("SURPRISAL_GATE", "0")


def check_surprisal(text: str) -> dict | None:
    """Ask the server whether this query should trigger retrieval.

    Returns the gate decision dict, or None on error (fail-open).
    """
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/surprisal",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def search_graph(query: str, k: int = 5, session_id: str = "",
                  min_similarity: float = 0.5,
                  general_surprisal: float | None = None,
                  personal_surprisal: float | None = None) -> dict:
    """Search the graph memory layer for synthesized long-term memories.

    Returns {"results": [...], "recall_id": str|None}.
    Results below min_similarity are filtered server-side and excluded from
    the stored recall, so recall_results count matches what we display.
    """
    body = {"q": query, "k": k, "expand_neighbors": True,
            "min_similarity": min_similarity}
    if general_surprisal is not None:
        body["general_surprisal"] = general_surprisal
    if personal_surprisal is not None:
        body["personal_surprisal"] = personal_surprisal
    if session_id:
        body["session_id"] = session_id
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/search_graph",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return {"results": [], "recall_id": None}


def get_last_assistant_text(transcript_path: str) -> str:
    """Read the transcript and return the last assistant message text.

    Used to give the graph search more context than just the user's bare prompt.
    """
    if not transcript_path:
        return ""
    try:
        from ingest import read_transcript, extract_text
        messages = read_transcript(transcript_path)
        # Walk backwards to find the last assistant message
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                text = extract_text(msg)
                if text:
                    # Truncate — we just need enough for context, not the whole response
                    return text[:500]
        return ""
    except Exception:
        return ""


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        return

    prompt = hook_input.get("prompt", "")
    if not prompt:
        return

    # Skip slash commands
    if prompt.lstrip().startswith("/"):
        return

    current_session = hook_input.get("session_id", "")

    # Build search query from user prompt + previous assistant context
    transcript_path = hook_input.get("transcript_path", "")
    prev_context = get_last_assistant_text(transcript_path)
    if prev_context:
        query = prev_context + "\n\n" + prompt
    else:
        query = prompt

    # Surprisal gate — skip retrieval for filler or novel-topic queries
    gate = check_surprisal(prompt)  # gate on raw prompt, not prompt+context
    if gate:
        gen_s = gate.get("general_surprisal", 0)
        pers_s = gate.get("personal_surprisal", 0)
        reason = gate.get("reason", "?")
        if not gate.get("retrieve", True):
            if SURPRISAL_GATE == "1":
                # Gate enforced — skip retrieval
                print(f"[surprisal] skipped: {reason} "
                      f"(gen={gen_s:.1f}, pers={pers_s:.1f})",
                      file=sys.stderr)
                return
            else:
                # Log-only mode — continue but log the decision
                print(f"[surprisal] would skip: {reason} "
                      f"(gen={gen_s:.1f}, pers={pers_s:.1f})",
                      file=sys.stderr)

    # Graph memory search — pass session_id and surprisal scores so they're
    # stored alongside the recall for calibration analysis.
    gate_gen = gate.get("general_surprisal") if gate else None
    gate_pers = gate.get("personal_surprisal") if gate else None
    graph_data = search_graph(query, k=5, session_id=current_session,
                              general_surprisal=gate_gen,
                              personal_surprisal=gate_pers)
    graph_results = graph_data.get("results", [])
    recall_id = graph_data.get("recall_id")
    if not graph_results:
        return

    lines = ["--- AGENT MEMORY (not visible to user) ---"]
    if current_session:
        lines.append(f"(current session: {current_session} — pass as exclude_session_id to avoid self-references)")

    recall_tag = f" recall:{recall_id}" if recall_id else ""
    lines.append(f"[graph]{recall_tag} | {len(graph_results)} synthesized memor{'ies' if len(graph_results) != 1 else 'y'}:")
    for r in graph_results:
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
    if recall_id:
        example_codes = ",".join(["U"] + ["N"] * (len(graph_results) - 1)) if graph_results else "U"
        lines.append(f"Reflect on these memories using the reflect tool: {recall_id}:{example_codes} (one code per memory above, in order)")
    lines.append("--- END AGENT MEMORY ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
