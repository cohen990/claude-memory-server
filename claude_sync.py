"""Claude.ai conversation sync — pulls conversations from claude.ai into the memory server.

Two modes:
    python claude_sync.py login  — headed browser, user logs into claude.ai via Google SSO
    python claude_sync.py sync   — fetches conversations via HTTP and ingests into memory server

Login uses Playwright with a persistent browser profile at ~/.claude-sync/browser-profile/.
After login, the sessionKey cookie is extracted and saved to ~/.claude-sync/session.json.
Sync uses plain HTTP with the saved cookie (no browser needed).

Sync state is tracked in ~/.claude-sync/synced_conversations.json.
Logs go to ~/.claude-sync/sync.log.

Cron entry:
    0 * * * * cd ~/memory-server && .venv/bin/python claude_sync.py sync
"""

import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SYNC_DIR = Path(os.path.expanduser("~/.claude-sync"))
BROWSER_PROFILE_DIR = SYNC_DIR / "browser-profile"
SESSION_FILE = SYNC_DIR / "session.json"
TRACKING_FILE = SYNC_DIR / "synced_conversations.json"
LOG_FILE = SYNC_DIR / "sync.log"

SERVER_URL = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8420")
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.5  # seconds between conversation fetches

BASE_URL = "https://claude.ai"

logger = logging.getLogger("claude_sync")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    SYNC_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def save_session(session_key: str):
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"sessionKey": session_key}, f)
    tmp.rename(SESSION_FILE)


def load_session() -> str | None:
    if not SESSION_FILE.exists():
        return None
    with open(SESSION_FILE) as f:
        data = json.load(f)
    return data.get("sessionKey")


# ---------------------------------------------------------------------------
# Tracking state
# ---------------------------------------------------------------------------

def load_tracking() -> dict[str, str]:
    """Load conversation_uuid -> last_updated_at mapping."""
    if not TRACKING_FILE.exists():
        return {}
    with open(TRACKING_FILE) as f:
        return json.load(f)


def save_tracking(state: dict[str, str]):
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TRACKING_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(TRACKING_FILE)


# ---------------------------------------------------------------------------
# HTTP client for claude.ai API
# ---------------------------------------------------------------------------

def claude_api_get(path: str, session_key: str) -> dict | list | None:
    """GET a claude.ai API endpoint using the session cookie."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={
        "Cookie": f"sessionKey={session_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logger.error("API error %s %s: %s", e.code, path, e.reason)
        return None
    except Exception as e:
        logger.error("Request failed %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Memory server client (stdlib only, like ingest.py)
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


# ---------------------------------------------------------------------------
# Message tree walking
# ---------------------------------------------------------------------------

def extract_active_branch(chat_messages: list[dict],
                          current_leaf_uuid: str) -> list[dict]:
    """Walk the message tree to extract the active branch in order.

    Conversations can have branches (regenerated responses). We follow only
    the branch that leads to current_leaf_message_uuid.
    """
    by_uuid = {m["uuid"]: m for m in chat_messages}

    # Walk backwards from leaf to root to find the active path
    active_uuids = set()
    current = current_leaf_uuid
    while current and current in by_uuid:
        active_uuids.add(current)
        current = by_uuid[current].get("parent_message_uuid")

    # Collect in forward order (root to leaf)
    ordered = [m for m in chat_messages if m["uuid"] in active_uuids]
    # Sort by index to preserve conversation order
    ordered.sort(key=lambda m: m.get("index", 0))

    return ordered


def extract_message_text(message: dict) -> str:
    """Extract text content from a claude.ai message."""
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def conversation_to_chunks(conversation: dict) -> list[dict]:
    """Convert a claude.ai conversation to memory server chunks."""
    conv_uuid = conversation["uuid"]
    current_leaf = conversation.get("current_leaf_message_uuid", "")
    chat_messages = conversation.get("chat_messages", [])

    if not chat_messages or not current_leaf:
        return []

    messages = extract_active_branch(chat_messages, current_leaf)

    # Pair human + assistant messages into turns
    chunks = []
    turn_number = 0
    i = 0

    while i < len(messages):
        msg = messages[i]
        if msg.get("sender") != "human":
            i += 1
            continue

        user_text = extract_message_text(msg)
        if not user_text:
            i += 1
            continue

        # Collect subsequent assistant messages
        assistant_parts = []
        j = i + 1
        while j < len(messages):
            next_msg = messages[j]
            if next_msg.get("sender") == "human":
                break
            if next_msg.get("sender") == "assistant":
                text = extract_message_text(next_msg)
                if text:
                    assistant_parts.append(text)
            j += 1

        if assistant_parts:
            assistant_text = "\n".join(assistant_parts)
            pair_text = f"User: {user_text}\n\nAssistant: {assistant_text}"

            # Truncate very long chunks (embedding model has 8192 token limit)
            if len(pair_text) > 16000:
                pair_text = pair_text[:16000] + "\n\n[truncated]"

            # Use the human message's created_at as the timestamp
            timestamp = msg.get("created_at", "")

            chunks.append({
                "text": pair_text,
                "user_text": user_text,
                "session_id": f"claude-web-{conv_uuid}",
                "timestamp": timestamp,
                "project": "claude.ai",
                "turn_number": turn_number,
                "branch": "",
            })

        turn_number += 1
        i = j

    return chunks


# ---------------------------------------------------------------------------
# Login mode
# ---------------------------------------------------------------------------

def do_login():
    """Launch a headed browser for the user to log into claude.ai."""
    from playwright.sync_api import sync_playwright

    BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Launching browser for login...")
    logger.info("Log into claude.ai via Google SSO, then close the browser.")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(BROWSER_PROFILE_DIR),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(f"{BASE_URL}/login")

        # Wait for the user to complete login and land on the main page
        logger.info("Waiting for login to complete (navigate to claude.ai)...")
        try:
            page.wait_for_url(f"{BASE_URL}/new**", timeout=300_000)
            logger.info("Login successful!")
        except Exception:
            logger.info("Browser closed or timed out.")

        # Extract sessionKey cookie
        cookies = context.cookies(BASE_URL)
        session_key = None
        for cookie in cookies:
            if cookie["name"] == "sessionKey":
                session_key = cookie["value"]
                break

        context.close()

    if session_key:
        save_session(session_key)
        logger.info("Session key saved to %s", SESSION_FILE)
    else:
        logger.error("No sessionKey cookie found — login may have failed.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sync mode
# ---------------------------------------------------------------------------

def do_sync():
    """Fetch conversations from claude.ai via HTTP and ingest into memory server."""
    session_key = load_session()
    if not session_key:
        logger.error("No session found. Run 'claude_sync.py login' first.")
        sys.exit(1)

    # Step 1: Get org ID (also validates session)
    orgs = claude_api_get("/api/organizations", session_key)
    if not isinstance(orgs, list) or not orgs:
        logger.error("Failed to get orgs — session may be expired. Run 'claude_sync.py login'.")
        sys.exit(1)

    org_id = orgs[0].get("uuid")
    if not org_id:
        logger.error("No org UUID in response.")
        sys.exit(1)

    logger.info("Org ID: %s", org_id)

    # Step 2: Load tracking state
    tracking = load_tracking()
    logger.info("Tracking %d previously synced conversations", len(tracking))

    # Step 3: List conversations
    conversations = list_conversations(org_id, session_key)
    logger.info("Found %d conversations on claude.ai", len(conversations))

    # Step 4: Filter to new/updated
    to_sync = []
    for conv in conversations:
        uuid = conv["uuid"]
        updated_at = conv.get("updated_at", "")
        tracked = tracking.get(uuid)
        if not tracked or tracked < updated_at:
            to_sync.append(conv)

    logger.info("%d conversations need syncing", len(to_sync))

    if not to_sync:
        logger.info("Nothing to sync.")
        return

    # Step 5: Fetch and ingest each conversation
    total_chunks = 0
    synced_count = 0

    for i, conv in enumerate(to_sync):
        uuid = conv["uuid"]
        name = conv.get("name", "(unnamed)")

        try:
            full_conv = claude_api_get(
                f"/api/organizations/{org_id}/chat_conversations/{uuid}"
                f"?tree=True&rendering_mode=messages&render_all_tools=true",
                session_key,
            )
            if not full_conv:
                logger.warning("Failed to fetch conversation %s (%s)", uuid[:8], name)
                continue

            chunks = conversation_to_chunks(full_conv)
            if not chunks:
                # Conversation has no turn pairs (maybe empty or all tool use)
                tracking[uuid] = conv.get("updated_at", "")
                synced_count += 1
                continue

            # Ingest in batches
            success = True
            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[batch_start:batch_start + BATCH_SIZE]
                result = ingest_chunks(batch)
                if "error" in result:
                    logger.error("Ingest error for %s: %s", uuid[:8], result["error"])
                    success = False
                    break

            if success:
                tracking[uuid] = conv.get("updated_at", "")
                total_chunks += len(chunks)
                synced_count += 1
                logger.info("  [%d/%d] %s — %d chunks", i + 1, len(to_sync), name, len(chunks))

        except Exception as e:
            logger.error("Error processing %s (%s): %s", uuid[:8], name, e)
            continue

        # Rate limit
        if i < len(to_sync) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    # Step 6: Save tracking
    save_tracking(tracking)
    logger.info("Sync complete. %d conversations, %d chunks ingested.", synced_count, total_chunks)


def list_conversations(org_id: str, session_key: str) -> list[dict]:
    """List all conversations, paginating through the API."""
    all_conversations = []
    cursor = None

    while True:
        path = f"/api/organizations/{org_id}/chat_conversations"
        if cursor:
            path += f"?cursor={cursor}"

        data = claude_api_get(path, session_key)
        if data is None:
            break

        # The API may return a list directly or a dict with cursor info
        if isinstance(data, list):
            all_conversations.extend(data)
            break  # No pagination info — we got everything
        elif isinstance(data, dict):
            items = data.get("data", data.get("conversations", []))
            all_conversations.extend(items)
            cursor = data.get("cursor") or data.get("next_cursor")
            has_more = data.get("has_more", False)
            if not cursor or not has_more:
                break
        else:
            break

    return all_conversations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    global SERVER_URL

    parser = argparse.ArgumentParser(description="Sync claude.ai conversations to memory server")
    parser.add_argument("command", choices=["login", "sync"], help="login or sync")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument("--server", default=SERVER_URL, help="Memory server URL")
    args = parser.parse_args()

    SERVER_URL = args.server

    setup_logging(verbose=args.verbose)

    if args.command == "login":
        do_login()
    elif args.command == "sync":
        do_sync()


if __name__ == "__main__":
    main()
