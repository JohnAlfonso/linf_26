#!/usr/bin/env bash
# Launches all 3 workers on the same machine (different ports).
# Useful when you have one beefy GPU (e.g. RTX PRO 6000 with 96 GB)
# and want to colocate the workers + coordinator.
#
# NOTE on performance: the 3 workers will share the same GPU SMs. With
# default CUDA context isolation each worker runs concurrently but
# fights for kernel-launch slots. For better concurrency, enable
# NVIDIA MPS (Multi-Process Service):
#     sudo nvidia-cuda-mps-control -d
# This lets multiple processes share one CUDA context with much better
# kernel interleaving — often 1.5-2× the throughput vs no MPS.
#
# Watch GPU contention while it runs:
#     watch -n 1 nvidia-smi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Launch each worker with its own env file. PM2 names them distinctly.
WORKER_ENV=worker_sigma_a.env bash scripts/run_worker.sh
WORKER_ENV=worker_sigma_b.env bash scripts/run_worker.sh
WORKER_ENV=worker_jsma.env bash scripts/run_worker.sh

echo
echo "All 3 workers launched. Verify:"
echo "  curl http://localhost:9201/health  # sigma_a"
echo "  curl http://localhost:9202/health  # sigma_b"
echo "  curl http://localhost:9203/health  # jsma"
echo
echo "Now point the coordinator at all 3:"
echo "  In scripts/miner.env:"
echo '  export PERTURB_WORKER_URLS="http://127.0.0.1:9201,http://127.0.0.1:9202,http://127.0.0.1:9203"'
echo
echo "Then restart the coordinator:"
echo "  pm2 restart perturb-miner"
