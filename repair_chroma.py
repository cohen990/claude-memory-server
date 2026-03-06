#!/usr/bin/env python3
"""Repair corrupted ChromaDB HNSW indices by re-embedding all documents.

When HNSW segment files are corrupted (e.g. from concurrent access),
ChromaDB's API returns "Error finding id" on both reads and writes.
The document text and metadata are still intact in ChromaDB's SQLite.

This script:
1. Reads all documents + metadata directly from ChromaDB's internal SQLite
2. Deletes and recreates each collection
3. Re-embeds all documents using the same model
4. Upserts them into the fresh collections

Run this with the server STOPPED — only one process should access ChromaDB at a time.

Usage:
    python repair_chroma.py [--chroma-dir DIR] [--model MODEL] [--batch-size N]
"""

import argparse
import json
import os
import sqlite3
import sys

import chromadb
from sentence_transformers import SentenceTransformer


DOCUMENT_PREFIX = "search_document: "
DEFAULT_CHROMA_DIR = os.path.expanduser("~/.memory-server/chromadb")
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")


def extract_documents_from_sqlite(chroma_dir: str) -> dict[str, list[dict]]:
    """Extract all documents and metadata from ChromaDB's internal SQLite."""
    db_path = os.path.join(chroma_dir, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)

    # Map collection names to their metadata segment IDs
    collections = {}
    rows = conn.execute("""
        SELECT c.name, s.id as segment_id, c.id as collection_id
        FROM segments s
        JOIN collections c ON s.collection = c.id
        WHERE s.scope = 'METADATA'
    """).fetchall()

    for name, segment_id, collection_id in rows:
        collections[name] = {
            "segment_id": segment_id,
            "collection_id": collection_id,
            "docs": [],
        }

    for name, info in collections.items():
        segment_id = info["segment_id"]

        # Get all embedding IDs for this segment
        embedding_rows = conn.execute(
            "SELECT id, embedding_id FROM embeddings WHERE segment_id = ?",
            (segment_id,),
        ).fetchall()

        print(f"  {name}: {len(embedding_rows)} documents")

        for embed_internal_id, embedding_id in embedding_rows:
            # Get document text from FTS
            doc_row = conn.execute(
                "SELECT string_value FROM embedding_metadata WHERE id = ? AND key = 'chroma:document'",
                (embed_internal_id,),
            ).fetchone()
            doc_text = doc_row[0] if doc_row else ""

            # Get all metadata for this embedding
            meta_rows = conn.execute(
                "SELECT key, string_value, int_value, float_value FROM embedding_metadata WHERE id = ? AND key != 'chroma:document'",
                (embed_internal_id,),
            ).fetchall()

            metadata = {}
            for key, str_val, int_val, float_val in meta_rows:
                if str_val is not None:
                    metadata[key] = str_val
                elif int_val is not None:
                    metadata[key] = int_val
                elif float_val is not None:
                    metadata[key] = float_val

            info["docs"].append({
                "id": embedding_id,
                "text": doc_text,
                "metadata": metadata,
            })

    conn.close()
    return collections


def repair(chroma_dir: str, model_name: str, device: str, batch_size: int):
    print(f"ChromaDB dir: {chroma_dir}")
    print(f"Model: {model_name} (device={device})")
    print()

    # Step 1: Extract all documents from SQLite
    print("Extracting documents from SQLite...")
    collections = extract_documents_from_sqlite(chroma_dir)
    print()

    # Step 2: Load embedding model
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    model.truncate_dim = 768
    print("Model loaded.")
    print()

    # Step 3: Delete corrupted collections and recreate
    print("Recreating collections...")
    client = chromadb.PersistentClient(path=chroma_dir)

    for name in list(collections.keys()):
        client.delete_collection(name)
        print(f"  Deleted {name}")

    new_collections = {}
    for name in collections:
        new_collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  Created {name}")
    print()

    # Step 4: Re-embed and upsert in batches
    for name, info in collections.items():
        docs = info["docs"]
        col = new_collections[name]
        total = len(docs)
        print(f"Re-embedding {name} ({total} documents)...")

        for i in range(0, total, batch_size):
            batch = docs[i:i + batch_size]
            ids = [d["id"] for d in batch]
            texts = [d["text"] for d in batch]
            metadatas = [d["metadata"] for d in batch]

            embeddings = model.encode(
                [DOCUMENT_PREFIX + t for t in texts],
                show_progress_bar=False,
            ).tolist()

            col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            done = min(i + batch_size, total)
            print(f"  {done}/{total}")

        # Verify
        try:
            sample = col.get(limit=1, include=["embeddings"])
            if sample["embeddings"]:
                col.query(query_embeddings=[sample["embeddings"][0]], n_results=1)
                print(f"  Verified OK ({col.count()} docs)")
            else:
                print(f"  Warning: no embeddings to verify")
        except Exception as e:
            print(f"  VERIFICATION FAILED: {e}")
        print()

    print("Repair complete.")


def main():
    parser = argparse.ArgumentParser(description="Repair corrupted ChromaDB HNSW indices")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    repair(args.chroma_dir, args.model, args.device, args.batch_size)


if __name__ == "__main__":
    main()
