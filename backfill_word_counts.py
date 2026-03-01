"""One-time backfill: populate word frequency tables in graph.db.

Two operations:
1. Load english word frequencies from the wordfreq library into
   the english_word_freqs table. Requires `pip install wordfreq`.
2. Scan existing ChromaDB user_inputs and populate the
   personal_word_counts table.

Usage:
    # On the server machine (has access to ChromaDB and graph.db):
    python backfill_word_counts.py [--english-only] [--personal-only]

    # Default: both operations
    python backfill_word_counts.py
"""

import argparse
import math
import os
import re
import sys

from graph import GraphStore
from surprisal import tokenize

CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.expanduser("~/.memory-server/chromadb"))
USER_INPUT_COLLECTION_NAME = "user_inputs"


def load_english_freqs(store: GraphStore):
    """Load english word frequencies from the wordfreq library."""
    try:
        from wordfreq import iter_wordlist, word_frequency
    except ImportError:
        print("Error: wordfreq not installed. Run: pip install wordfreq")
        sys.exit(1)

    print("Loading english word frequencies from wordfreq...")
    words = list(iter_wordlist('en', wordlist='best'))
    # Filter to alpha-only (matches our tokenizer)
    alpha_words = [w for w in words if re.match(r'^[a-z]+$', w)]

    freqs = {}
    for w in alpha_words:
        prob = word_frequency(w, 'en')
        if prob > 0:
            freqs[w] = math.log(prob)

    print(f"  {len(freqs)} words extracted from wordfreq")
    store.load_english_freqs(freqs)
    print(f"  Loaded into english_word_freqs table")


def backfill_personal_counts(store: GraphStore):
    """Scan existing ChromaDB user_inputs and populate personal_word_counts."""
    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed.")
        sys.exit(1)

    print(f"Opening ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(USER_INPUT_COLLECTION_NAME)
    except Exception:
        print(f"  Collection '{USER_INPUT_COLLECTION_NAME}' not found. Nothing to backfill.")
        return

    # Get all documents
    total = collection.count()
    print(f"  Found {total} user input documents")

    if total == 0:
        return

    # Process in batches to avoid memory issues
    batch_size = 1000
    total_words = 0
    offset = 0

    while offset < total:
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents"],
        )
        docs = results.get("documents", [])
        if not docs:
            break

        all_words = []
        for doc in docs:
            if doc:
                all_words.extend(tokenize(doc))

        if all_words:
            store.update_personal_word_counts(all_words)
            total_words += len(all_words)

        offset += len(docs)
        print(f"  Processed {min(offset, total)}/{total} documents ({total_words} words)")

    print(f"  Backfill complete: {total_words} total word occurrences indexed")
    print(f"  Vocabulary size: {store.personal_vocab_size()}")


def main():
    parser = argparse.ArgumentParser(description="Backfill word frequency tables")
    parser.add_argument("--english-only", action="store_true",
                        help="Only load english frequencies")
    parser.add_argument("--personal-only", action="store_true",
                        help="Only backfill personal word counts")
    args = parser.parse_args()

    store = GraphStore()

    if not args.personal_only:
        load_english_freqs(store)

    if not args.english_only:
        backfill_personal_counts(store)

    store.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
