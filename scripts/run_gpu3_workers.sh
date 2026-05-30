#!/usr/bin/env bash
# Launches the 3 hard-image-only workers on GPU3 (RTX 3090, 24 GB).
# These workers ONLY receive dispatch when phase_e gap > 3 (hard image),
# so they don't consume GPU cycles on easy cases.
#
# Setup before running:
#   1. git clone repo to /root/linf_26
#   2. install CUDA driver 545+ and Python venv
#   3. start NVIDIA MPS: nvidia-cuda-mps-control -d
#
# This script starts:
#   port 9221 — sigma_hard_a   (σ-zero B=4 zeros-init, n_iter=500)
#   port 9222 — sigma_hard_b   (σ-zero B=4 random seed=99, n_iter=400)
#   port 9223 — jsma_strong    (JSMA + backward_eliminate, low K)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting GPU3 hard-image workers (3 workers on ports 9221-9223)..."
echo ""

WORKER_ENV=worker_sigma_hard_a.env bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_hard_b.env bash scripts/run_worker.sh
WORKER_ENV=worker_jsma_strong.env  bash scripts/run_worker.sh

echo ""
echo "All 3 GPU3 hard-image workers launched. Verify with:"
echo "  curl http://localhost:9221/health  # sigma_hard_a"
echo "  curl http://localhost:9222/health  # sigma_hard_b"
echo "  curl http://localhost:9223/health  # jsma_strong"
echo ""
echo "Then on the COORDINATOR (GPU1) machine, ADD to scripts/miner.env:"
echo "  export PERTURB_HARD_WORKER_URLS=\"\\"
echo "    http://<GPU3_IP>:9221,\\"
echo "    http://<GPU3_IP>:9222,\\"
echo "    http://<GPU3_IP>:9223\""
echo ""
echo "Then on coordinator: bash scripts/run_miner.sh"
