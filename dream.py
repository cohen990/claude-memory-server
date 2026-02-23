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

from graph import GraphStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")
DREAM_MODEL = os.environ.get("DREAM_MODEL", "sonnet")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.85"))
STALENESS_THRESHOLD = float(os.environ.get("STALENESS_THRESHOLD", "0.15"))
RATING_SCALE = float(os.environ.get("RATING_SCALE", "0.02"))
RATING_EDGE_FLOOR = 0.05
RATING_EDGE_CAP = 0.95

RATING_VALUES = {
    "U": 2,
    "I": 1,
    "N": 0,
    "D": -1,
    "M": -2,
}

CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.expanduser("~/.memory-server/chromadb"))
COLLECTION_NAME = "conversations"


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


def _apply_rating_delta(old_weight: float, delta: float,
                        floor: float = RATING_EDGE_FLOOR,
                        cap: float = RATING_EDGE_CAP) -> float:
    """Apply a rating-derived delta to an edge weight, clamped to [floor, cap]."""
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
                    "weight": {"type": "number"},
                },
                "required": ["from_idx", "from_type", "to_idx", "to_type", "weight"],
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


def synthesize(chunks: list[str]) -> dict:
    """Send chunks to Claude CLI for synthesis into vibes and details.

    Returns: {"vibes": [...], "details": [...], "connections": [...]}
    """
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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_consolidate(args):
    """Read recent conversations from ChromaDB and synthesize into graph nodes."""
    import chromadb

    if args.days:
        print(f"Consolidating last {args.days} days of conversations...")
    else:
        print("Consolidating all un-dreamed conversations...")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total = collection.count()
    if total == 0:
        print("No conversations found.")
        return

    # Filter to chunks not yet processed by dream.
    # Chunks ingested before the dreamed field was added won't have it,
    # so we use $ne 1 (matches both dreamed=0 and missing field).
    results = collection.get(
        where={"dreamed": {"$ne": 1}},
        include=["documents", "metadatas"],
        limit=total,
    )

    docs = []
    ids = []
    metas = []

    if args.days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        cutoff_str = cutoff.isoformat()
        for i, meta in enumerate(results["metadatas"]):
            ts = meta.get("timestamp", "")
            if ts >= cutoff_str:
                docs.append(results["documents"][i])
                ids.append(results["ids"][i])
                metas.append(meta)
    else:
        docs = results["documents"]
        ids = results["ids"]
        metas = results["metadatas"]

    if not ids:
        print("No un-dreamed conversations found.")
        return

    print(f"Found {len(docs)} un-dreamed chunks (of {total} total).")

    # Process in batches to stay within context limits
    batch_size = 20
    graph = GraphStore()
    total_vibes = 0
    total_details = 0
    total_edges = 0

    for batch_start in range(0, len(docs), batch_size):
        batch_docs = docs[batch_start:batch_start + batch_size]
        batch_ids = ids[batch_start:batch_start + batch_size]
        batch_metas = metas[batch_start:batch_start + batch_size]
        print(f"\nProcessing batch {batch_start // batch_size + 1} ({len(batch_docs)} chunks)...")

        try:
            synthesis = synthesize(batch_docs)
        except Exception as e:
            print(f"  Synthesis error: {e}")
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
            else:
                node_id = graph.add_node("vibe", text, embedding, source_ids=source_ids)
                print(f"  Created vibe {node_id[:8]}...: {text[:60]}")
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
            else:
                node_id = graph.add_node("detail", text, embedding, source_ids=source_ids)
                print(f"  Created detail {node_id[:8]}...: {text[:60]}")
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

        for conn in synthesis.get("connections", []):
            from_type = conn.get("from_type", "vibe")
            to_type = conn.get("to_type", "detail")
            from_idx = conn.get("from_idx", 0)
            to_idx = conn.get("to_idx", 0)
            weight = conn.get("weight", 0.5)

            # Map indices to the all_nodes list
            vibes_list = [(i, n) for i, (t, n) in enumerate(all_nodes) if t == "vibe"]
            details_list = [(i, n) for i, (t, n) in enumerate(all_nodes) if t == "detail"]

            from_nodes = vibes_list if from_type == "vibe" else details_list
            to_nodes = details_list if to_type == "detail" else vibes_list

            if from_idx < len(from_nodes) and to_idx < len(to_nodes):
                from_id = from_nodes[from_idx][1]
                to_id = to_nodes[to_idx][1]
                if from_id and to_id:
                    graph.add_edge(from_id, to_id, weight=weight)
                    total_edges += 1

        # Mark batch chunks as dreamed so they won't be reprocessed
        # ChromaDB update() replaces metadata entirely, so carry forward existing fields
        marked_metas = [{**m, "dreamed": 1} for m in batch_metas]
        collection.update(
            ids=batch_ids,
            metadatas=marked_metas,
        )

    # Rebuild cache on server
    graph._rebuild_cache()
    reload_cache()

    s = graph.stats()
    print(f"\nConsolidation complete.")
    print(f"  New vibes: {total_vibes}, New details: {total_details}, New edges: {total_edges}")
    print(f"  Total graph: {s['total_nodes']} nodes, {s['total_edges']} edges")


def cmd_reconsolidate(args):
    """Process rated recalls: adjust edge weights and reconsolidate affected embeddings.

    Recalls are the sole signal for edge weight changes. Every search creates
    a recall; the agent rates it. Only explicitly rated recalls with non-zero
    rating values affect edge weights. Unrated recalls default to 0 (no effect).
    """
    graph = GraphStore()

    # 1. Get rated recalls — the only signal for edge weight changes
    rated_recalls = graph.get_rated_recalls()
    edges_adjusted = 0
    affected_nodes: set[str] = set()

    if rated_recalls:
        print(f"Processing {len(rated_recalls)} rated recalls...")

        for recall in rated_recalls:
            query_emb = recall["query_embedding"]

            for result in recall["results"]:
                rating_code = result.get("rating")
                if not rating_code or rating_code not in RATING_VALUES:
                    continue

                rating_value = RATING_VALUES[rating_code]
                if rating_value == 0:
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

                    delta = rating_value * sim * RATING_SCALE
                    new_weight = _apply_rating_delta(edge["weight"], delta)

                    if new_weight != edge["weight"]:
                        graph.update_edge_weight(
                            edge["source_id"], edge["target_id"], new_weight,
                        )
                        edges_adjusted += 1
                        if abs(delta) >= 0.01:
                            print(f"  Edge {edge['source_id'][:8]}→{edge['target_id'][:8]}: "
                                  f"{edge['weight']:.3f} → {new_weight:.3f} "
                                  f"(rating={rating_code}, sim={sim:.2f})")
    else:
        print("No rated recalls to process.")

    # 2. Reconsolidate affected node embeddings
    resynthesized = 0

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
                    resynthesized += 1
                    print(f"    → {new_text[:80]}")
            except Exception as e:
                print(f"  Warning: staleness check failed for {node_id[:8]}: {e}")

    # 3. Clear all recalls (rated and unrated) to prevent buildup
    graph.clear_processed_recalls()

    # 4. Rebuild cache
    graph._rebuild_cache()
    reload_cache()

    s = graph.stats()
    print(f"\nReconsolidation complete.")
    print(f"  Nodes reconsolidated: {len(affected_nodes)}")
    print(f"  Texts re-synthesized: {resynthesized}")
    print(f"  Edges adjusted by ratings: {edges_adjusted}")
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
