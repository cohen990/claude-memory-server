#!/usr/bin/env bash
# Restart memory-server on the remote machine without redeploying files.
# Usage: ./restart-server.sh
#
# Configure DEPLOY_REMOTE in .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

REMOTE="${DEPLOY_REMOTE:?Set DEPLOY_REMOTE in .env (e.g. user@192.168.1.100)}"
SERVER_URL="${MEMORY_SERVER_URL:-http://localhost:8420}"

echo "Restarting server..."
ssh "$REMOTE" "systemctl --user restart memory-server"

echo "Waiting for server..."
for i in $(seq 1 30); do
  if curl -sf "$SERVER_URL/stats" >/dev/null 2>&1; then
    echo "Server is up."
    exit 0
  fi
  sleep 2
done

echo "ERROR: Server did not become healthy within 60s."
echo "Check logs: ssh $REMOTE 'journalctl --user -u memory-server -n 50'"
exit 1
