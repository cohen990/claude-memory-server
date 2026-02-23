#!/usr/bin/env bash
# Deploy memory-server to the remote machine.
# Usage: ./deploy.sh [--restart]
#
# Configure DEPLOY_REMOTE and DEPLOY_REMOTE_DIR in .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

REMOTE="${DEPLOY_REMOTE:?Set DEPLOY_REMOTE in .env (e.g. user@192.168.1.100)}"
REMOTE_DIR="${DEPLOY_REMOTE_DIR:-~/memory-server}"
SERVER_URL="${MEMORY_SERVER_URL:-http://localhost:8420}"

echo "Syncing files to $REMOTE..."
rsync -avz \
  --exclude=__pycache__ \
  --exclude=.pytest_cache \
  --exclude='*.pyc' \
  --exclude=.venv \
  --exclude=.git \
  "$SCRIPT_DIR/" "$REMOTE:$REMOTE_DIR/"

if [[ "${1:-}" == "--restart" ]]; then
  echo "Restarting server..."
  ssh "$REMOTE" "systemctl --user restart memory-server"
  echo "Waiting for server..."
  for i in $(seq 1 30); do
    if curl -sf "$SERVER_URL/stats" >/dev/null 2>&1; then
      echo "Server is up."
      break
    fi
    sleep 2
  done
fi

echo "Done."
