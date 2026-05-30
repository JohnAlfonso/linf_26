# GPU2 Deployment Guide (Workers 5-8)

8-worker architecture: GPU1 hosts the coordinator + workers 1-4, GPU2 hosts workers 5-8. The coordinator dispatches to all 8 in parallel via HTTP, scores each via FP32 verify, picks the best.

## Pre-deployment requirements

### Network
- **Sub-10ms RTT** from GPU2 → GPU1 (test with `ping`)
- **1 Gbps+** sustained bandwidth (each request ~2 MB)
- **Same datacenter / same region strongly recommended**
- Open inbound TCP ports 9211-9214 on GPU2 from GPU1's IP

### GPU2 software stack
- NVIDIA driver 545+
- Python 3.12+
- CUDA toolkit (matching driver)
- `nvidia-cuda-mps-control` binary
- Git, build tools

## Deployment steps

### On GPU2 (the new RTX PRO 6000)

```bash
# 1. Clone repo
cd /root
git clone <YOUR_REPO_URL> linf_26
cd /root/linf_26

# 2. Install Python venv (creates .venv)
apt install -y python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install fastapi uvicorn

# 3. Set up NVIDIA MPS (CRITICAL — must be done before workers start)
mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Verify MPS daemon running:
ps aux | grep nvidia-cuda-mps-control | grep -v grep

# 4. Launch all 4 workers
bash scripts/run_gpu2_workers.sh

# 5. Wait ~30 seconds, then verify each is healthy
for PORT in 9211 9212 9213 9214; do
  echo "Port $PORT:"
  curl -s http://localhost:$PORT/health | python3 -m json.tool
done
```

### On GPU1 (existing coordinator)

```bash
# Edit miner.env to add GPU2 worker URLs
nano /root/linf_26/scripts/miner.env

# Find the PERTURB_WORKER_URLS line and replace with:
export PERTURB_WORKER_URLS="http://127.0.0.1:9201,http://127.0.0.1:9202,http://127.0.0.1:9203,http://127.0.0.1:9205,http://<GPU2_IP>:9211,http://<GPU2_IP>:9212,http://<GPU2_IP>:9213,http://<GPU2_IP>:9214"

# Replace <GPU2_IP> with the actual GPU2 server IP address.

# Restart coordinator
bash scripts/run_miner.sh

# Verify all 8 workers are being dispatched
pm2 logs perturb-miner --lines 30 | grep "workers:"
# Should see: "workers: 8/8 ok"
```

## Worker layout

| Worker | Port | Strategy | What it does | GPU |
|---|---|---|---|---|
| 1 | 9201 | sigma_a | σ-zero zeros-init, B=2 | GPU1 |
| 2 | 9202 | sigma_b | σ-zero random-init B=4, seed=1 | GPU1 |
| 3 | 9203 | jsma | JSMA cluster_kernel=3 | GPU1 |
| 4 | 9205 | sparse_rs | σ-zero seed + Sparse-RS seed=7 | GPU1 |
| **5** | **9211** | **sigma_c** | **σ-zero B=8 zeros-init (max Pareto)** | **GPU2** |
| **6** | **9212** | **sigma_d** | **σ-zero random-init B=4, seed=17** | **GPU2** |
| **7** | **9213** | **sigma_e** | **σ-zero random-init B=4, seed=42** | **GPU2** |
| **8** | **9214** | **sparse_rs_v2** | **Sparse-RS seed=99 (2nd restart)** | **GPU2** |

## Expected gain

- Before: 4 workers, mean ~0.944 (validator ~0.948)
- After 8 workers: mean ~0.946-0.949 (validator ~0.949-0.951)
- Hard-image cases improve more than easy cases (more parallel restart chances)

## Failsafe rollback

If GPU2 is having issues (network down, workers crashed), the coordinator's `WORKER_REQUEST_TIMEOUT_S=11.5` will skip dead workers. But if you want to disable GPU2 entirely:

```bash
# On GPU1 — revert to 4 workers
sed -i 's|http://<GPU2_IP>:[0-9]*,||g' /root/linf_26/scripts/miner.env
# Or manually edit miner.env to remove the GPU2 URLs.
bash /root/linf_26/scripts/run_miner.sh
```

## Monitoring

On GPU1 (coordinator):
```bash
pm2 logs perturb-miner --nostream --lines 100 | grep "workers:"
# Watch for "workers: 8/8 ok" — should be the steady state.
# If you see "workers: 4/8 ok" or fewer, GPU2 is failing.
```

On GPU2:
```bash
pm2 list  # check all 4 workers online
nvidia-smi  # check VRAM usage (~25 GB across 4 workers)
nvidia-smi dmon -i 0 -c 6 -s u  # 6 samples, GPU utilization
```
