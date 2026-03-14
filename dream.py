"""Dream pipeline — nightly consolidation of raw memories into graph nodes.

CLI script run on the server machine. Imports graph.py directly for DB access,
calls the memory server HTTP API for embeddings, and the Claude CLI for
LLM synthesis (uses OAuth creds, no API key needed).

Usage:
    python dream.py consolidate --days 7
    python dream.py reconsolidate
    python dream.py full --days 7
    python dream.py stats
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

import numpy as np

from graph import GraphStore, DreamLog

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")
DREAM_MODEL = os.environ.get("DREAM_MODEL", "sonnet")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.85"))
STALENESS_THRESHOLD = float(os.environ.get("STALENESS_THRESHOLD", "0.15"))
REFLECTION_SCALE = float(os.environ.get("REFLECTION_SCALE", "0.02"))
REFLECTION_EDGE_FLOOR = 0.05
REFLECTION_EDGE_CAP = 0.95

REFLECTION_VALUES = {
    "U": 2,
    "I": 1,
    "N": 0,
    "D": -1,
    "M": -2,
}



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blend_embeddings(node_emb: np.ndarray,
                      neighbor_embs: list[np.ndarray],
                      neighbor_weights: list[float]) -> np.ndarray:
    """Weighted average: node (weight=1) + neighbors (by edge weight)."""
    total_weight = 1.0 + sum(neighbor_weights)
    blended = node_emb.copy()
    for emb, w in zip(neighbor_embs, neighbor_weights):
        blended += emb * w
    blended /= total_weight
    return blended


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors: 1.0 - cos(a, b)."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return 1.0 - float(np.dot(a_norm, b_norm))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _apply_reflection_delta(old_weight: float, delta: float,
                            floor: float = REFLECTION_EDGE_FLOOR,
                            cap: float = REFLECTION_EDGE_CAP) -> float:
    """Apply a reflection-derived delta to an edge weight, clamped to [floor, cap]."""
    return max(floor, min(cap, old_weight + delta))


def embed_text(text: str) -> np.ndarray:
    """Get embedding from memory server's /embed endpoint."""
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return np.array(data["embedding"], dtype=np.float32)


def reload_cache():
    """Tell the server to rebuild its graph embedding cache."""
    req = urllib.request.Request(
        f"{SERVER_URL}/graph/reload_cache",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"Warning: Could not reload server cache: {e}")
        return None


def recompute_layout():
    """Tell the server to recompute graph layout positions."""
    req = urllib.request.Request(
        f"{SERVER_URL}/graph/recompute_layout",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            print(f"  Layout recomputed: {data.get('nodes_positioned', 0)} nodes positioned")
            return data
    except Exception as e:
        print(f"Warning: Could not recompute layout: {e}")
        return None


SYNTHESIS_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "vibes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "source_indices"],
            },
        },
        "details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "source_indices"],
            },
        },
        "connections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_idx": {"type": "integer"},
                    "from_type": {"type": "string"},
                    "to_idx": {"type": "integer"},
                    "to_type": {"type": "string"},
                    "from_existing_idx": {"type": "integer"},
                    "to_existing_idx": {"type": "integer"},
                    "weight": {"type": "number"},
                },
                "required": ["weight"],
            },
        },
    },
    "required": ["vibes", "details", "connections"],
})


def _claude(prompt: str, json_schema: str | None = None) -> str | dict:
    """Run a prompt through the Claude CLI and return the response.

    Uses -p (print mode) with --no-session-persistence so it doesn't pollute
    conversation history. Uses OAuth creds — no API key needed.

    When json_schema is provided, returns the parsed dict from structured_output.
    Otherwise returns the text result string.
    """
    cmd = [
        "claude", "-p",
        "--model", DREAM_MODEL,
        "--no-session-persistence",
        "--output-format", "json",
        "--max-turns", "7",
    ]
    if json_schema:
        cmd.extend(["--json-schema", json_schema])

    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={result.returncode}): {result.stderr[:500]}")

    envelope = json.loads(result.stdout)

    if envelope.get("is_error") or envelope.get("subtype") == "error_max_turns":
        raise RuntimeError(f"claude CLI error: {envelope.get('subtype', 'unknown')}")

    # --json-schema puts structured output in structured_output field
    if json_schema and "structured_output" in envelope:
        return envelope["structured_output"]

    return envelope.get("result", "")


def _build_synthesis_prompt(chunks: list[str],
                            recalled_nodes: list[dict] | None = None) -> str:
    """Build the synthesis prompt from chunks and optional recalled nodes."""
    combined = "\n\n---\n\n".join(chunks)

    prompt = f"""Analyze these conversation excerpts and synthesize them into long-term memory nodes.

Extract two types of nodes:
- vibes: High-level themes, patterns, preferences, or recurring topics (e.g., "prefers functional programming", "frequently works on memory systems")
- details: Specific facts, decisions, solutions, or technical details worth remembering (e.g., "uses nomic-embed-text-v1.5 with 768-dim Matryoshka embeddings")

Also suggest connections between the nodes you create.

source_indices refer to which conversation excerpts (0-indexed) contributed to each node.
Connection weight should be 0.0-1.0 indicating strength of association.

Conversation excerpts:

{combined}"""

    if recalled_nodes:
        prompt += "\n\nExisting memories that were recalled during these conversations:\n\n"
        for i, node in enumerate(recalled_nodes):
            prompt += f'[{i}] ({node["type"]}, id={node["id"]}) "{node["text"]}"\n'
        prompt += """
You may suggest connections between new nodes you create and these existing nodes.
For connections to existing nodes, use "from_existing_idx" or "to_existing_idx" (0-indexed into the list above) instead of "from_idx"/"from_type" or "to_idx"/"to_type" for that side of the connection.
A connection can link two new nodes, a new node to an existing node, or an existing node to a new node."""

    return prompt


def synthesize(chunks: list[str],
               recalled_nodes: list[dict] | None = None) -> dict:
    """Send chunks to Claude CLI for synthesis into vibes and details.

    When recalled_nodes is provided, includes them as context so Claude can
    suggest cross-temporal connections between new and existing nodes.

    Returns: {"vibes": [...], "details": [...], "connections": [...]}
    """
    prompt = _build_synthesis_prompt(chunks, recalled_nodes)
    # _claude returns parsed dict when json_schema is provided
    return _claude(prompt, json_schema=SYNTHESIS_SCHEMA)


def resynthesize_text(node_text: str, neighbor_texts: list[str]) -> str:
    """Re-synthesize a node's text given its neighbors for context."""
    context = "\n".join(f"- {t}" for t in neighbor_texts[:5])

    prompt = f"""This memory node's text has drifted from its embedding due to reinforcement from related memories. Rewrite it to better reflect its current associations.

Current text: {node_text}

Related memories:
{context}

Write a single updated description that integrates insights from the related memories. Keep it concise (1-3 sentences). Respond with ONLY the updated text, no explanation."""

    return _claude(prompt).strip()


def adjudicate_contest(node_text: str, correction: str,
                       source_chunks: list[str]) -> dict:
    """Ask Claude to adjudicate between a node's text and a proposed correction.

    Returns {"verdict": "accept"|"reject"|"revise", "text": str, "reasoning": str}.
    """
    sources_section = ""
    if source_chunks:
        source_text = "\n---\n".join(source_chunks[:5])
        sources_section = f"""

Source conversations (the original context this memory was synthesized from):
{source_text}
"""

    prompt = f"""A memory node has been contested by the agent. Decide whether the correction is valid.

Current memory text:
{node_text}

Proposed correction:
{correction}
{sources_section}
Decide:
- ACCEPT: The correction is right. Replace the memory with corrected text.
- REVISE: Both have partial truth. Write a revised version that incorporates the correction.
- REJECT: The original is correct. Dismiss the correction.

Respond with ONLY a JSON object (no markdown fences):
{{"verdict": "accept|revise|reject", "text": "the final text (corrected, revised, or original)", "reasoning": "brief explanation"}}"""

    ADJUDICATION_SCHEMA = json.dumps({
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["accept", "revise", "reject"]},
            "text": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "text", "reasoning"],
    })

    return _claude(prompt, json_schema=ADJUDICATION_SCHEMA)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _fetch_chunks_by_ids(ids: list[str]) -> list[str]:
    """Fetch chunk texts by ID from the server API."""
    if not ids:
        return []
    body = json.dumps({"ids": ids}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/chunks/by_ids",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return [c["text"] for c in data.get("chunks", [])]
    except Exception:
        return []


def _fetch_undreamed_chunks(days: int | None = None) -> tuple[list[str], list[str], list[dict]]:
    """Fetch un-dreamed chunks from the server API.

    Returns (docs, ids, metas) lists.
    """
    params = {}
    if days is not None:
        params["days"] = str(days)
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{SERVER_URL}/chunks/undreamed"
    if qs:
        url += f"?{qs}"

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    docs = [c["text"] for c in data["chunks"]]
    ids = [c["id"] for c in data["chunks"]]
    metas = [c["metadata"] for c in data["chunks"]]
    return docs, ids, metas


def _mark_chunks_dreamed(batch_ids: list[str], batch_metas: list[dict]):
    """Mark chunks as dreamed via the server API."""
    marked_metas = [{**m, "dreamed": 1} for m in batch_metas]
    body = json.dumps({"ids": batch_ids, "metadatas": marked_metas}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/chunks/mark_dreamed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def cmd_consolidate(args):
    """Read recent conversations and synthesize into graph nodes."""
    if args.days:
        print(f"Consolidating last {args.days} days of conversations...")
    else:
        print("Consolidating all un-dreamed conversations...")

    docs, ids, metas = _fetch_undreamed_chunks(days=args.days)

    if not ids:
        print("No un-dreamed conversations found.")
        return

    print(f"Found {len(docs)} un-dreamed chunks.")

    # Process in batches to stay within context limits
    batch_size = 20
    graph = GraphStore()
    dream_log = DreamLog(graph)
    run_id = dream_log.start_run("consolidate")
    total_vibes = 0
    total_details = 0
    total_merged = 0
    total_edges = 0
    run_error = None

    try:
        for batch_start in range(0, len(docs), batch_size):
            batch_docs = docs[batch_start:batch_start + batch_size]
            batch_ids = ids[batch_start:batch_start + batch_size]
            batch_metas = metas[batch_start:batch_start + batch_size]
            print(f"\nProcessing batch {batch_start // batch_size + 1} ({len(batch_docs)} chunks)...")

            # Collect session IDs from this batch and fetch recalled nodes
            batch_session_ids = list({
                m["session_id"] for m in batch_metas
                if m.get("session_id")
            })
            recalled_nodes = []
            if batch_session_ids:
                recalled_nodes = graph.get_recalled_nodes_for_sessions(batch_session_ids)
                if recalled_nodes:
                    print(f"  Found {len(recalled_nodes)} recalled nodes from {len(batch_session_ids)} sessions")

            try:
                synthesis = synthesize(batch_docs, recalled_nodes=recalled_nodes or None)
            except Exception as e:
                print(f"  Synthesis error: {e}")
                dream_log.log_operation(run_id, "error", None, None,
                                        {"message": f"Synthesis error: {e}"})
                continue

            # Process vibes
            for vibe in synthesis.get("vibes", []):
                text = vibe["text"]
                source_indices = vibe.get("source_indices", [])
                source_ids = [batch_ids[i] for i in source_indices if i < len(batch_ids)]

                try:
                    embedding = embed_text(text)
                except Exception as e:
                    print(f"  Embed error for vibe: {e}")
                    continue

                existing = graph.find_similar(embedding, threshold=SIMILARITY_THRESHOLD, node_type="vibe")
                if existing:
                    graph.merge_node_embedding(existing["id"], embedding, new_source_ids=source_ids)
                    print(f"  Merged vibe into {existing['id'][:8]}...: {text[:60]}")
                    dream_log.log_operation(run_id, "node_merged", existing["id"], "vibe",
                                            {"text": text, "similarity": existing.get("similarity", 0),
                                             "source_count": len(source_ids)})
                    total_merged += 1
                else:
                    node_id = graph.add_node("vibe", text, embedding, source_ids=source_ids)
                    print(f"  Created vibe {node_id[:8]}...: {text[:60]}")
                    dream_log.log_operation(run_id, "node_created", node_id, "vibe",
                                            {"text": text, "source_count": len(source_ids)})
                    total_vibes += 1

            # Process details
            for detail in synthesis.get("details", []):
                text = detail["text"]
                source_indices = detail.get("source_indices", [])
                source_ids = [batch_ids[i] for i in source_indices if i < len(batch_ids)]

                try:
                    embedding = embed_text(text)
                except Exception as e:
                    print(f"  Embed error for detail: {e}")
                    continue

                existing = graph.find_similar(embedding, threshold=SIMILARITY_THRESHOLD, node_type="detail")
                if existing:
                    graph.merge_node_embedding(existing["id"], embedding, new_source_ids=source_ids)
                    print(f"  Merged detail into {existing['id'][:8]}...: {text[:60]}")
                    dream_log.log_operation(run_id, "node_merged", existing["id"], "detail",
                                            {"text": text, "similarity": existing.get("similarity", 0),
                                             "source_count": len(source_ids)})
                    total_merged += 1
                else:
                    node_id = graph.add_node("detail", text, embedding, source_ids=source_ids)
                    print(f"  Created detail {node_id[:8]}...: {text[:60]}")
                    dream_log.log_operation(run_id, "node_created", node_id, "detail",
                                            {"text": text, "source_count": len(source_ids)})
                    total_details += 1

            # Process connections
            # Build index of nodes created in this batch for connection lookup
            all_nodes = []
            for v in synthesis.get("vibes", []):
                # Find the node we just created/merged
                try:
                    emb = embed_text(v["text"])
                    found = graph.find_similar(emb, threshold=0.9)
                    if found:
                        all_nodes.append(("vibe", found["id"]))
                    else:
                        all_nodes.append(("vibe", None))
                except Exception:
                    all_nodes.append(("vibe", None))

            for d in synthesis.get("details", []):
                try:
                    emb = embed_text(d["text"])
                    found = graph.find_similar(emb, threshold=0.9)
                    if found:
                        all_nodes.append(("detail", found["id"]))
                    else:
                        all_nodes.append(("detail", None))
                except Exception:
                    all_nodes.append(("detail", None))

            # Build lookup for existing (recalled) nodes
            existing_node_ids = [n["id"] for n in recalled_nodes]

            for conn in synthesis.get("connections", []):
                weight = conn.get("weight", 0.5)
                from_id = None
                to_id = None

                # Resolve "from" side
                if "from_existing_idx" in conn:
                    idx = conn["from_existing_idx"]
                    if 0 <= idx < len(existing_node_ids):
                        from_id = existing_node_ids[idx]
                elif "from_idx" in conn and "from_type" in conn:
                    from_type = conn["from_type"]
                    from_idx = conn["from_idx"]
                    typed_nodes = [(i, n) for i, (t, n) in enumerate(all_nodes) if t == from_type]
                    if from_idx < len(typed_nodes):
                        from_id = typed_nodes[from_idx][1]

                # Resolve "to" side
                if "to_existing_idx" in conn:
                    idx = conn["to_existing_idx"]
                    if 0 <= idx < len(existing_node_ids):
                        to_id = existing_node_ids[idx]
                elif "to_idx" in conn and "to_type" in conn:
                    to_type = conn["to_type"]
                    to_idx = conn["to_idx"]
                    typed_nodes = [(i, n) for i, (t, n) in enumerate(all_nodes) if t == to_type]
                    if to_idx < len(typed_nodes):
                        to_id = typed_nodes[to_idx][1]

                if from_id and to_id:
                    graph.add_edge(from_id, to_id, weight=weight)
                    from_node = graph.get_node(from_id)
                    to_node = graph.get_node(to_id)
                    dream_log.log_operation(run_id, "edge_created", None, None,
                                            {"source_id": from_id, "target_id": to_id, "weight": weight,
                                             "source_text": from_node["text"] if from_node else None,
                                             "target_text": to_node["text"] if to_node else None})
                    total_edges += 1

            # Mark batch chunks as dreamed so they won't be reprocessed
            _mark_chunks_dreamed(batch_ids, batch_metas)
    except Exception as e:
        run_error = str(e)
        dream_log.log_operation(run_id, "error", None, None, {"message": str(e)})
        raise
    finally:
        dream_log.finish_run(
            run_id, error=run_error,
            chunks_processed=len(docs),
            nodes_created=total_vibes + total_details,
            nodes_merged=total_merged,
            edges_created=total_edges,
        )

    # Rebuild cache on server and recompute layout
    graph._rebuild_cache()
    reload_cache()
    recompute_layout()

    s = graph.stats()
    print(f"\nConsolidation complete.")
    print(f"  New vibes: {total_vibes}, New details: {total_details}, New edges: {total_edges}")
    print(f"  Total graph: {s['total_nodes']} nodes, {s['total_edges']} edges")


def cmd_reconsolidate(args):
    """Process reflected recalls: adjust edge weights and reconsolidate affected embeddings.

    Recalls are the sole signal for edge weight changes. Every search creates
    a recall; the agent reflects on it. Only explicitly reflected recalls with non-zero
    reflection values affect edge weights. Unreflected recalls default to 0 (no effect).
    """
    graph = GraphStore()
    dream_log = DreamLog(graph)
    run_id = dream_log.start_run("reconsolidate")
    run_error = None

    # 0. Adjudicate contested nodes
    contested = graph.get_contested_nodes()
    contests_resolved = 0

    if contested:
        print(f"Adjudicating {len(contested)} contested node(s)...")

        for node in contested:
            node_id = node["id"]
            try:
                source_chunks = _fetch_chunks_by_ids(node["source_ids"][:5])
                result = adjudicate_contest(
                    node["text"], node["contested_correction"], source_chunks,
                )
                verdict = result["verdict"]
                new_text = result["text"]
                reasoning = result["reasoning"]

                print(f"  {node_id[:8]}... [{node['type']}]: {verdict}")
                print(f"    Original: {node['text'][:80]}")
                print(f"    Correction: {node['contested_correction'][:80]}")
                print(f"    Reasoning: {reasoning[:120]}")

                if verdict in ("accept", "revise"):
                    new_embedding = embed_text(new_text)
                    graph.resolve_contest(node_id, new_text, new_embedding)
                    print(f"    → Updated: {new_text[:80]}")
                else:
                    graph.resolve_contest(node_id, None, None)
                    print(f"    → Dismissed, original kept")

                dream_log.log_operation(
                    run_id, "contest_resolved", node_id, node["type"],
                    {"verdict": verdict, "old_text": node["text"],
                     "correction": node["contested_correction"],
                     "new_text": new_text, "reasoning": reasoning})
                contests_resolved += 1
            except Exception as e:
                print(f"  Warning: contest adjudication failed for {node_id[:8]}: {e}")
                dream_log.log_operation(run_id, "error", node_id, None,
                                        {"message": f"Contest adjudication failed: {e}"})

    # 1. Get reflected recalls — the only signal for edge weight changes
    reflected_recalls = graph.get_reflected_recalls()
    edges_adjusted = 0
    affected_nodes: set[str] = set()
    resynthesized = 0

    try:
        if reflected_recalls:
            print(f"Processing {len(reflected_recalls)} reflected recalls...")

            for recall in reflected_recalls:
                query_emb = recall["query_embedding"]

                for result in recall["results"]:
                    reflection_code = result.get("reflection")
                    if not reflection_code or reflection_code not in REFLECTION_VALUES:
                        continue

                    reflection_value = REFLECTION_VALUES[reflection_code]
                    if reflection_value == 0:
                        continue  # NOISE = no change

                    node_id = result["node_id"]
                    affected_nodes.add(node_id)
                    edges = graph.get_edges(node_id)
                    if not edges:
                        continue

                    for edge in edges:
                        neighbor_id = (
                            edge["target_id"] if edge["source_id"] == node_id
                            else edge["source_id"]
                        )
                        neighbor = graph.get_node(neighbor_id)
                        if not neighbor:
                            continue

                        sim = _cosine_sim(query_emb, neighbor["embedding"])
                        if sim <= 0:
                            continue  # only adjust edges pointing toward the query

                        delta = reflection_value * sim * REFLECTION_SCALE
                        new_weight = _apply_reflection_delta(edge["weight"], delta)

                        if new_weight != edge["weight"]:
                            graph.update_edge_weight(
                                edge["source_id"], edge["target_id"], new_weight,
                            )
                            source_node = graph.get_node(edge["source_id"])
                            target_node = graph.get_node(edge["target_id"])
                            dream_log.log_operation(
                                run_id, "edge_adjusted", node_id, None,
                                {"source_id": edge["source_id"], "target_id": edge["target_id"],
                                 "old_weight": edge["weight"], "new_weight": new_weight,
                                 "reflection": reflection_code, "delta": delta,
                                 "source_text": source_node["text"] if source_node else None,
                                 "target_text": target_node["text"] if target_node else None})
                            edges_adjusted += 1
                            if abs(delta) >= 0.01:
                                print(f"  Edge {edge['source_id'][:8]}→{edge['target_id'][:8]}: "
                                      f"{edge['weight']:.3f} → {new_weight:.3f} "
                                      f"(reflection={reflection_code}, sim={sim:.2f})")
        else:
            print("No reflected recalls to process.")

        # 2. Reconsolidate affected node embeddings
        if affected_nodes:
            print(f"\nReconsolidating {len(affected_nodes)} affected nodes...")

            for node_id in affected_nodes:
                node = graph.get_node(node_id)
                if not node:
                    continue

                edges = graph.get_edges(node_id)
                if not edges:
                    continue

                # Compute weighted blend of self + neighbors
                neighbor_embeddings = []
                neighbor_weights = []
                neighbor_texts = []
                for edge in edges:
                    neighbor_id = (
                        edge["target_id"] if edge["source_id"] == node_id
                        else edge["source_id"]
                    )
                    neighbor = graph.get_node(neighbor_id)
                    if neighbor:
                        neighbor_embeddings.append(neighbor["embedding"])
                        neighbor_weights.append(edge["weight"])
                        neighbor_texts.append(neighbor["text"])

                if not neighbor_embeddings:
                    continue

                blended = _blend_embeddings(node["embedding"], neighbor_embeddings, neighbor_weights)
                graph.update_node_embedding(node_id, blended)

                # Check staleness: is the text still a good description of the embedding?
                try:
                    text_embedding = embed_text(node["text"])
                    cosine_dist = _cosine_distance(text_embedding, blended)

                    if cosine_dist > STALENESS_THRESHOLD:
                        print(f"  Node {node_id[:8]}... stale (dist={cosine_dist:.3f}), re-synthesizing text...")
                        new_text = resynthesize_text(node["text"], neighbor_texts)
                        new_embedding = embed_text(new_text)
                        graph.update_node_text(node_id, new_text, new_embedding)
                        dream_log.log_operation(
                            run_id, "node_resynthesized", node_id, node["type"],
                            {"old_text": node["text"], "new_text": new_text,
                             "staleness": cosine_dist})
                        resynthesized += 1
                        print(f"    → {new_text[:80]}")
                except Exception as e:
                    print(f"  Warning: staleness check failed for {node_id[:8]}: {e}")
                    dream_log.log_operation(run_id, "error", node_id, None,
                                            {"message": f"Staleness check failed: {e}"})
    except Exception as e:
        run_error = str(e)
        dream_log.log_operation(run_id, "error", None, None, {"message": str(e)})
        raise
    finally:
        dream_log.finish_run(
            run_id, error=run_error,
            edges_adjusted=edges_adjusted,
            nodes_resynthesized=resynthesized,
        )

    # 3. Clear all recalls (reflected and unreflected) to prevent buildup
    graph.clear_processed_recalls()

    # 4. Rebuild cache
    graph._rebuild_cache()
    reload_cache()

    s = graph.stats()
    print(f"\nReconsolidation complete.")
    print(f"  Contests adjudicated: {contests_resolved}")
    print(f"  Nodes reconsolidated: {len(affected_nodes)}")
    print(f"  Texts re-synthesized: {resynthesized}")
    print(f"  Edges adjusted by reflections: {edges_adjusted}")
    print(f"  Total graph: {s['total_nodes']} nodes, {s['total_edges']} edges")


def cmd_full(args):
    """Run full dream cycle: consolidate then reconsolidate."""
    cmd_consolidate(args)
    print("\n" + "=" * 60 + "\n")
    cmd_reconsolidate(args)


def cmd_stats(args):
    """Print graph statistics."""
    graph = GraphStore()
    s = graph.stats()

    print("Graph Memory Statistics")
    print("=" * 40)
    print(f"Total nodes: {s['total_nodes']}")
    for ntype, count in s.get("nodes_by_type", {}).items():
        print(f"  {ntype}: {count}")
    print(f"Total edges: {s['total_edges']}")
    print(f"Activated edges: {s['activated_edges']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dream pipeline — nightly memory consolidation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_consolidate = sub.add_parser("consolidate", help="Synthesize recent conversations into graph nodes")
    p_consolidate.add_argument("--days", type=int, default=None, help="Limit to last N days (default: all un-dreamed)")

    p_reconsolidate = sub.add_parser("reconsolidate", help="Process activated edges and reconsolidate embeddings")

    p_full = sub.add_parser("full", help="Full dream cycle: consolidate + reconsolidate")
    p_full.add_argument("--days", type=int, default=None, help="Limit to last N days (default: all un-dreamed)")

    p_stats = sub.add_parser("stats", help="Print graph statistics")

    args = parser.parse_args()

    commands = {
        "consolidate": cmd_consolidate,
        "reconsolidate": cmd_reconsolidate,
        "full": cmd_full,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
