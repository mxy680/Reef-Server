#!/usr/bin/env bash
set -euo pipefail

# Usage: ./deploy.sh deploy@YOUR_SERVER_IP
SERVER="${1:?Usage: ./deploy.sh user@host}"
REMOTE_DIR="/opt/reef"

echo "==> Syncing to $SERVER:$REMOTE_DIR ..."
rsync -az --delete \
    --exclude .venv \
    --exclude .git \
    --exclude .env \
    --exclude data/ \
    --exclude tests/ \
    --exclude __pycache__ \
    --exclude .pytest_cache \
    . "$SERVER:$REMOTE_DIR/"

echo "==> Building and deploying ..."
ssh "$SERVER" "cd $REMOTE_DIR && docker compose build app && docker compose up -d"

echo "==> Checking status ..."
ssh "$SERVER" "cd $REMOTE_DIR && docker compose ps"

echo "==> Health check ..."
sleep 5
ssh "$SERVER" "cd $REMOTE_DIR && docker compose exec app curl -sf http://localhost:8000/health && echo ' OK' || echo ' FAILED'"

echo "==> Done."
