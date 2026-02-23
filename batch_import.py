"""Batch importer — ingests historical Claude Code transcripts into the memory server.

Reads JSONL transcripts from ~/.claude/projects/, chunks them into turn pairs,
and sends them to the server. Tracks which sessions have been ingested to avoid
duplicates on re-runs.

Usage:
    python batch_import.py                    # ingest all unprocessed transcripts
    python batch_import.py --project syneme   # filter by project name substring
    python batch_import.py --include-subagents  # also ingest subagent transcripts
    python batch_import.py --dry-run          # show what would be ingested
    python batch_import.py --reset            # clear the tracking file and re-ingest everything
"""

import argparse
import json
import os
import sys
from pathlib import Path

from ingest import read_transcript, chunk_transcript, ingest_chunks

CLAUDE_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
TRACKING_FILE = os.path.expanduser("~/.memory-server/ingested_sessions.json")
SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")


def find_transcripts(include_subagents: bool = False,
                     project_filter: str = "") -> list[Path]:
    """Find all JSONL transcript files under ~/.claude/projects/."""
    projects_dir = Path(CLAUDE_PROJECTS_DIR)
    if not projects_dir.exists():
        print(f"Projects directory not found: {CLAUDE_PROJECTS_DIR}", file=sys.stderr)
        return []

    transcripts = []
    for path in projects_dir.rglob("*.jsonl"):
        # Skip subagent transcripts unless requested
        if not include_subagents and "subagents" in path.parts:
            continue

        # Filter by project name if specified
        if project_filter:
            # The project dir is the first component after projects/
            rel = path.relative_to(projects_dir)
            project_dir = str(rel.parts[0]) if rel.parts else ""
            if project_filter.lower() not in project_dir.lower():
                continue

        transcripts.append(path)

    return sorted(transcripts)


def load_tracking() -> set[str]:
    """Load the set of already-ingested session file paths."""
    if not os.path.exists(TRACKING_FILE):
        return set()
    with open(TRACKING_FILE) as f:
        data = json.load(f)
    return set(data.get("ingested", []))


def save_tracking(ingested: set[str]):
    """Save the set of ingested session file paths."""
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    with open(TRACKING_FILE, "w") as f:
        json.dump({"ingested": sorted(ingested)}, f, indent=2)


def derive_project_name(transcript_path: Path) -> str:
    """Extract the project directory name from the transcript path.

    ~/.claude/projects/-home-user-projects-myapp/abc123.jsonl
    → /home/user/projects/myapp
    """
    try:
        rel = transcript_path.relative_to(CLAUDE_PROJECTS_DIR)
        project_dir = str(rel.parts[0])
        # Convert -home-user-projects-foo to /home/user/projects/foo
        return "/" + project_dir.lstrip("-").replace("-", "/")
    except (ValueError, IndexError):
        return ""


def main():
    parser = argparse.ArgumentParser(description="Batch import Claude Code transcripts")
    parser.add_argument("--project", default="", help="Filter by project name substring")
    parser.add_argument("--include-subagents", action="store_true",
                        help="Also ingest subagent transcripts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be ingested without doing it")
    parser.add_argument("--reset", action="store_true",
                        help="Clear tracking file and re-ingest everything")
    parser.add_argument("--server", default=SERVER_URL,
                        help=f"Memory server URL (default: {SERVER_URL})")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of chunks to send per request (default: 50)")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)
            print("Tracking file cleared.")

    transcripts = find_transcripts(
        include_subagents=args.include_subagents,
        project_filter=args.project,
    )
    print(f"Found {len(transcripts)} transcript files.")

    already_ingested = load_tracking()
    to_process = [t for t in transcripts if str(t) not in already_ingested]
    print(f"Already ingested: {len(already_ingested)}, new: {len(to_process)}")

    if not to_process:
        print("Nothing to ingest.")
        return

    if args.dry_run:
        for t in to_process:
            messages = read_transcript(str(t))
            chunks = chunk_transcript(messages)
            project = derive_project_name(t)
            print(f"  {t.name}: {len(chunks)} chunks (project: {project})")
        return

    total_chunks = 0
    total_files = 0
    newly_ingested = set()

    for i, transcript_path in enumerate(to_process):
        try:
            messages = read_transcript(str(transcript_path))
            chunks = chunk_transcript(messages)
        except Exception as e:
            print(f"  Error reading {transcript_path.name}: {e}", file=sys.stderr)
            continue

        if not chunks:
            newly_ingested.add(str(transcript_path))
            continue

        # Fill in project name from path if not in chunk metadata
        project_name = derive_project_name(transcript_path)
        for chunk in chunks:
            if not chunk.get("project"):
                chunk["project"] = project_name

        # Send in batches
        for batch_start in range(0, len(chunks), args.batch_size):
            batch = chunks[batch_start:batch_start + args.batch_size]
            result = ingest_chunks(batch, server_url=args.server)
            if "error" in result:
                print(f"  Error ingesting {transcript_path.name}: {result['error']}",
                      file=sys.stderr)
                break
        else:
            # All batches succeeded
            total_chunks += len(chunks)
            total_files += 1
            newly_ingested.add(str(transcript_path))

        # Progress
        if (i + 1) % 10 == 0 or i == len(to_process) - 1:
            print(f"  Progress: {i + 1}/{len(to_process)} files, {total_chunks} chunks ingested")

    # Update tracking
    all_ingested = already_ingested | newly_ingested
    save_tracking(all_ingested)

    print(f"\nDone. Ingested {total_chunks} chunks from {total_files} files.")
    print(f"Total tracked sessions: {len(all_ingested)}")


if __name__ == "__main__":
    main()
