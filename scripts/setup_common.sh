#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROLE="${1:-validator}"
ROLE="$(echo "$ROLE" | tr '[:upper:]' '[:lower:]')"
DEFAULT_OLLAMA_MODEL="${PERTURB_LLM_ENDPOINT_MODEL:-qwen2.5:1.5b-instruct}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ and rerun."
  exit 1
fi

if [[ "$ROLE" != "miner" && "$ROLE" != "validator" ]]; then
  echo "Usage: bash ./scripts/setup_common.sh [miner|validator]"
  exit 1
fi

if [[ "$ROLE" == "validator" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm not found. Install Node.js (which includes npm) and rerun."
    echo "macOS: brew install node"
    echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y nodejs npm"
    exit 1
  fi

  echo "Installing PM2..."
  npm install -g pm2

  if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    if ! command -v curl >/dev/null 2>&1; then
      echo "curl not found. Install curl first, then rerun setup."
      exit 1
    fi
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  echo "Ensuring Ollama server is running..."
  if ! curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    if pm2 describe perturb-ollama >/dev/null 2>&1; then
      pm2 restart perturb-ollama
    else
      pm2 start "ollama serve" --name perturb-ollama
    fi
    pm2 save
  fi

  for _ in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
    echo "Ollama server is not reachable on http://127.0.0.1:11434"
    exit 1
  fi

  echo "Ensuring Ollama model is available: ${DEFAULT_OLLAMA_MODEL}"
  ollama pull "${DEFAULT_OLLAMA_MODEL}"
fi

echo "Creating/updating virtual environment..."
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install bittensor bittensor-cli
python -m pip install -e .

echo "Setup complete for role: ${ROLE}"
