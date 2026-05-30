#!/usr/bin/env bash
# Launches a single GPU worker for the distributed multi-server miner.
# Run this on each of the 3 GPU servers with a different WORKER_STRATEGY.
#
# Usage:
#   bash scripts/run_worker.sh sigma_a   # GPU server 1
#   bash scripts/run_worker.sh sigma_b   # GPU server 2
#   bash scripts/run_worker.sh jsma      # GPU server 3
#
# Or set WORKER_STRATEGY in scripts/worker.env and just run:
#   bash scripts/run_worker.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"
# Allow per-worker env file via WORKER_ENV (e.g. worker_sigma_a.env).
# Default to plain worker.env for single-worker setups.
ENV_FILE="$SCRIPT_DIR/${WORKER_ENV:-worker.env}"
EXAMPLE_ENV="$SCRIPT_DIR/worker.env.example"

if [[ ! -f "$ENV_FILE" && -f "$EXAMPLE_ENV" ]]; then
  echo "Missing $ENV_FILE — copy template:"
  echo "  cp \"$EXAMPLE_ENV\" \"$ENV_FILE\""
  echo "Then edit WORKER_STRATEGY, WORKER_PORT, etc. before re-running."
  exit 1
fi
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi

# CLI override: first positional arg is the strategy.
if [[ -n "${1:-}" ]]; then
  export WORKER_STRATEGY="$1"
fi

WORKER_STRATEGY="${WORKER_STRATEGY:-sigma_a}"
WORKER_PORT="${WORKER_PORT:-9200}"
WORKER_HOST="${WORKER_HOST:-0.0.0.0}"

# Validate strategy early before launching uvicorn.
case "$WORKER_STRATEGY" in
  sigma_a|sigma_b|sigma_c|sigma_d|sigma_e|sigma_hard_a|sigma_hard_b|sigma_max|jsma|jsma_strong|square|sparse_rs|sparse_rs_v2) ;;
  *)
    echo "Invalid WORKER_STRATEGY=$WORKER_STRATEGY (allowed: sigma_a/b/c/d/e/hard_a/hard_b/max, jsma, jsma_strong, square, sparse_rs, sparse_rs_v2)"
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip > /dev/null
python -m pip install -r requirements.txt > /dev/null
python -m pip install -e . > /dev/null
python -m pip install fastapi uvicorn > /dev/null

PM2_NAME="perturb-worker-${WORKER_STRATEGY}"

echo "Starting worker '${PM2_NAME}' on ${WORKER_HOST}:${WORKER_PORT}..."
if command -v pm2 >/dev/null 2>&1; then
  if pm2 describe "$PM2_NAME" >/dev/null 2>&1; then
    pm2 delete "$PM2_NAME"
  fi
  # Pass env to PM2 explicitly because pm2 ignores parent shell env vars.
  pm2 start ".venv/bin/python" --name "$PM2_NAME" \
    --update-env \
    -- -m tools.worker_service
  pm2 save
  pm2 status "$PM2_NAME"
else
  # Foreground fallback if PM2 not installed.
  exec python -m tools.worker_service
fi
