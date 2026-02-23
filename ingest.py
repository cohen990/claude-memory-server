"""Ingestion library — chunks conversation transcripts and sends them to the memory server.

Two modes of operation:
1. As a Claude Code Stop hook: reads hook context from stdin, extracts the latest
   turn pair from the transcript, and POSTs it to the server.
2. As a library: import chunk_transcript() and ingest_chunks() for batch use.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def read_transcript(path: str) -> list[dict]:
    """Read a JSONL transcript file and return all message lines."""
    messages = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages.append(obj)
            except json.JSONDecodeError:
                continue
    return messages


def extract_text(message: dict) -> str:
    """Extract human-readable text from a message's content field.

    Handles both string content and list-of-blocks content.
    Skips tool_use, tool_result, and other non-text blocks.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def chunk_transcript(messages: list[dict]) -> list[dict]:
    """Convert a transcript into turn-pair chunks.

    A turn pair is a user message + the concatenated text of all subsequent
    assistant messages until the next user message. Tool use/result messages
    are skipped (they're noise for retrieval — the assistant's text summary
    is what matters).

    Returns a list of chunk dicts with keys:
        text, session_id, timestamp, project, turn_number, branch
    """
    chunks = []
    turn_number = 0
    i = 0

    while i < len(messages):
        msg = messages[i]
        msg_type = msg.get("type", "")

        # Find user messages
        if msg_type == "user" and isinstance(msg.get("message"), dict):
            inner = msg["message"]
            if inner.get("role") != "user":
                i += 1
                continue

            user_text = extract_text(inner)
            if not user_text:
                i += 1
                continue

            # Collect metadata from the user message
            session_id = msg.get("sessionId", "")
            timestamp = msg.get("timestamp", "")
            project = msg.get("cwd", "")
            branch = msg.get("gitBranch", "")

            # Collect subsequent assistant text responses
            assistant_parts = []
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                next_type = next_msg.get("type", "")

                if next_type == "user" and isinstance(next_msg.get("message"), dict):
                    if next_msg["message"].get("role") == "user":
                        # tool_result messages also have type=user, role=user isn't set for those
                        # actually, tool_results have content as list with tool_result blocks
                        inner_next = next_msg["message"]
                        content = inner_next.get("content", "")
                        # Check if this is a real user message (string or has text blocks)
                        if isinstance(content, str) and content:
                            break
                        if isinstance(content, list):
                            has_text = any(b.get("type") == "text" for b in content)
                            has_tool_result = any(b.get("type") == "tool_result" for b in content)
                            if has_text and not has_tool_result:
                                break
                            if has_tool_result:
                                j += 1
                                continue
                            break

                if next_type == "assistant" and isinstance(next_msg.get("message"), dict):
                    text = extract_text(next_msg["message"])
                    if text:
                        assistant_parts.append(text)

                j += 1

            if assistant_parts:
                pair_text = f"User: {user_text}\n\nAssistant: {chr(10).join(assistant_parts)}"

                # Truncate very long chunks (embedding model has 8192 token limit)
                if len(pair_text) > 16000:
                    pair_text = pair_text[:16000] + "\n\n[truncated]"

                chunks.append({
                    "text": pair_text,
                    "user_text": user_text,
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "project": project,
                    "turn_number": turn_number,
                    "branch": branch,
                })

            turn_number += 1
            i = j
        else:
            i += 1

    return chunks


def get_latest_turn_pair(messages: list[dict]) -> Optional[dict]:
    """Extract only the last turn pair from the transcript.

    Used by the Stop hook to ingest incrementally.
    """
    chunks = chunk_transcript(messages)
    if chunks:
        return chunks[-1]
    return None


# ---------------------------------------------------------------------------
# HTTP client (stdlib only — no external deps for the hook script)
# ---------------------------------------------------------------------------

def ingest_chunks(chunks: list[dict], server_url: str = SERVER_URL) -> dict:
    """POST chunks to the memory server's /ingest endpoint."""
    payload = json.dumps({"chunks": chunks}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/ingest",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def ingest_summary(text: str, session_id: str, timestamp: str,
                   project: str = "", server_url: str = SERVER_URL) -> dict:
    """POST a session summary to the memory server."""
    payload = json.dumps({
        "text": text,
        "session_id": session_id,
        "timestamp": timestamp,
        "project": project,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/ingest_summary",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Stop hook entrypoint
# ---------------------------------------------------------------------------

def run_as_hook():
    """Called when this script runs as a Claude Code Stop hook.

    Reads hook context from stdin, extracts the latest turn pair from the
    transcript, and ingests it.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        print("Failed to parse hook input", file=sys.stderr)
        return

    # Don't ingest if this is a recursive stop (hook-triggered response)
    if hook_input.get("stop_hook_active"):
        return

    transcript_path = hook_input.get("transcript_path")
    if not transcript_path:
        print("No transcript_path in hook input", file=sys.stderr)
        return

    transcript_path = os.path.expanduser(transcript_path)
    if not os.path.exists(transcript_path):
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)
        return

    messages = read_transcript(transcript_path)
    chunk = get_latest_turn_pair(messages)
    if not chunk:
        return

    # Use session_id from hook input if available (more reliable)
    session_id = hook_input.get("session_id", chunk.get("session_id", ""))
    if session_id:
        chunk["session_id"] = session_id

    result = ingest_chunks([chunk])
    if "error" in result:
        print(f"Ingestion error: {result['error']}", file=sys.stderr)
    else:
        print(f"Ingested 1 turn pair (session {session_id[:8]})", file=sys.stderr)


if __name__ == "__main__":
    run_as_hook()
