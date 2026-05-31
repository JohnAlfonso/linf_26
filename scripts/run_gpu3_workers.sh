#!/usr/bin/env bash
# Launches the hard-image-only workers on GPU3 (RTX PRO 6000, 96 GB).
# These workers ONLY receive dispatch when phase_e gap > 3 (hard image),
# so they don't consume GPU cycles on easy cases.
#
# Setup before running:
#   1. git clone repo to /root/linf_26
#   2. install CUDA driver 545+ and Python venv
#   3. start NVIDIA MPS: nvidia-cuda-mps-control -d
#
# This script starts:
#   port 9221 — sigma_hard_a    (σ-zero B=8 zeros-init, n_iter=400)
#   port 9222 — sigma_hard_b    (σ-zero B=8 random seed=99, n_iter=400)
#   port 9223 — jsma_strong     (JSMA + backward_eliminate, low K)
#   port 9224 — sigma_max       (σ-zero B=12 zeros-init, n_iter=300)
#   port 9225 — sparse_pgd      (Sparse-PGD + MALT pre-stage)
#   port 9226 — sigma_max_malt  (σ-zero B=12 with MALT targets)
#   port 9227 — sigma_grind     (σ-zero B=1 single-target, n=900)
#   port 9228 — greedy_fool     (GreedyFool 2-stage + MALT)         [NEW]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting GPU3 workers (8 workers on ports 9221-9228)..."
echo ""

WORKER_ENV=worker_sigma_hard_a.env    bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_hard_b.env    bash scripts/run_worker.sh
WORKER_ENV=worker_jsma_strong.env     bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_max.env       bash scripts/run_worker.sh
WORKER_ENV=worker_sparse_pgd.env      bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_max_malt.env  bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_grind.env     bash scripts/run_worker.sh
WORKER_ENV=worker_greedy_fool.env     bash scripts/run_worker.sh

echo ""
echo "All 8 GPU3 workers launched. Verify with:"
echo "  curl http://localhost:9221/health  # sigma_hard_a (B=8)"
echo "  curl http://localhost:9222/health  # sigma_hard_b (B=8)"
echo "  curl http://localhost:9223/health  # jsma_strong"
echo "  curl http://localhost:9224/health  # sigma_max (B=12)"
echo "  curl http://localhost:9225/health  # sparse_pgd + MALT"
echo "  curl http://localhost:9226/health  # sigma_max_malt (B=12 + MALT)"
echo "  curl http://localhost:9227/health  # sigma_grind (B=1, n=900)"
echo "  curl http://localhost:9228/health  # greedy_fool 2-stage + MALT"
echo ""
echo "Then on the COORDINATOR (GPU1) machine, ADD the GPU3 URLs to"
echo "PERTURB_HARD_WORKER_URLS in scripts/miner.env (replace <GPU3_IP> and"
echo "the public-port mapping your provider gives):"
echo "  export PERTURB_HARD_WORKER_URLS=\"\\"
echo "    http://<GPU3_IP>:<pub9221>,\\"
echo "    http://<GPU3_IP>:<pub9222>,\\"
echo "    http://<GPU3_IP>:<pub9223>,\\"
echo "    http://<GPU3_IP>:<pub9224>,\\"
echo "    http://<GPU3_IP>:<pub9225>,\\"
echo "    http://<GPU3_IP>:<pub9226>\""
echo ""
echo "Then on coordinator: bash scripts/run_miner.sh"
