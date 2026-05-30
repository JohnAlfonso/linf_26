#!/usr/bin/env bash
# Launches the 4 additional workers on GPU2 (RTX PRO 6000).
# Run this on the GPU2 server AFTER:
#   1. git clone repo to /root/linf_26
#   2. install CUDA driver 545+ and Python venv
#   3. start NVIDIA MPS: nvidia-cuda-mps-control -d
#
# This script starts:
#   port 9211 — sigma_c    (σ-zero B=8, max Pareto-K)
#   port 9212 — sigma_d    (σ-zero random seed=17)
#   port 9213 — sigma_e    (σ-zero random seed=42)
#   port 9214 — sparse_rs_v2 (Sparse-RS seed=99)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting GPU2 workers (4 workers on ports 9211-9214)..."
echo ""

WORKER_ENV=worker_sigma_c.env       bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_d.env       bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_e.env       bash scripts/run_worker.sh
WORKER_ENV=worker_sparse_rs_v2.env  bash scripts/run_worker.sh

echo ""
echo "All 4 GPU2 workers launched. Verify with:"
echo "  curl http://localhost:9211/health  # sigma_c"
echo "  curl http://localhost:9212/health  # sigma_d"
echo "  curl http://localhost:9213/health  # sigma_e"
echo "  curl http://localhost:9214/health  # sparse_rs_v2"
echo ""
echo "Then on the COORDINATOR (GPU1) machine, edit scripts/miner.env:"
echo '  export PERTURB_WORKER_URLS="\'
echo '    http://127.0.0.1:9201,\'
echo '    http://127.0.0.1:9202,\'
echo '    http://127.0.0.1:9203,\'
echo '    http://127.0.0.1:9205,\'
echo '    http://<GPU2_IP>:9211,\'
echo '    http://<GPU2_IP>:9212,\'
echo '    http://<GPU2_IP>:9213,\'
echo '    http://<GPU2_IP>:9214"'
echo ""
echo "Then on coordinator: bash scripts/run_miner.sh"
