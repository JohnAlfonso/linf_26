# GPU3 Deployment — Hard-Image-Only Workers (Workers 9-11)

11-worker architecture: GPU1 (coord+4w) + GPU2 (4w) + **GPU3 (3 hard-only workers)**.

GPU3 workers ONLY fire when the coordinator detects a hard image (`phase_e_gap > 3`). On easy images they're idle — saves their compute for the cases that need it.

## Architecture flow

```
Every validator request:
  1. coordinator computes phase_e gap
  2. always dispatch 8 base workers (GPU1 + GPU2)
  3. IF gap > 3:
       ALSO dispatch 3 hard-only workers (GPU3)
       (total 11 in parallel, wall bounded by slowest)
     ELSE:
       skip GPU3 dispatch
  4. coordinator FP32-verifies all returned candidates
  5. picks best by FP32 score
  6. boost / dense_shrink / ship
```

## Pre-deployment requirements (RTX 3090)

| Component | Requirement |
|---|---|
| VRAM | 24 GB (3 workers × ~6 GB each = 18 GB used, 6 GB headroom) |
| GPU Compute Cap | 8.6+ (RTX 3090 is 8.6, fine) |
| Memory BW | 936 GB/s (RTX 3090 ✓) |
| CUDA driver | 545+ |
| Network to GPU1 coordinator | < 10ms RTT |
| Open ports | 9221, 9222, 9223 (TCP, inbound from GPU1 IP) |

## Deployment steps

### On GPU3 server (RTX 3090):

```bash
# 1. Clone repo
cd /root && git clone <YOUR_REPO_URL> linf_26 && cd /root/linf_26

# 2. Install deps + venv (same as GPU1/GPU2)
apt install -y python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install fastapi uvicorn

# 3. Start MPS daemon (CRITICAL for multi-worker GPU sharing)
mkdir -p /tmp/nvidia-mps /tmp/nvidia-log
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Verify MPS daemon running:
ps aux | grep nvidia-cuda-mps-control | grep -v grep

# 4. Launch 3 hard-image workers
bash scripts/run_gpu3_workers.sh

# 5. Wait ~30 seconds, then verify health
for PORT in 9221 9222 9223; do
  echo "Port $PORT:"
  curl -s http://localhost:$PORT/health | python3 -m json.tool
done
```

### On GPU1 (coordinator):

Edit `/root/linf_26/scripts/miner.env`:

```bash
# Add this line (replace <GPU3_IP> with actual IP):
export PERTURB_HARD_WORKER_URLS="http://<GPU3_IP>:9221,http://<GPU3_IP>:9222,http://<GPU3_IP>:9223"
```

Restart coordinator:
```bash
cd /root/linf_26 && bash scripts/run_miner.sh

# Verify hard-only dispatch is active:
pm2 logs perturb-miner --lines 30 --nostream | grep "workers:"
# On easy images (gap≤3): "workers: 8/8 ok (hard=False)"
# On hard images (gap>3): "workers: 11/11 ok (hard=True)"
```

## Worker layout (full 11-worker matrix)

| Worker | Port | Strategy | GPU | Always-on? |
|---|---|---|---|---|
| 1 | 9201 | sigma_a | GPU1 | yes |
| 2 | 9202 | sigma_b | GPU1 | yes |
| 3 | 9203 | jsma | GPU1 | yes |
| 4 | 9205 | sparse_rs | GPU1 | yes |
| 5 | 9211 | sigma_c | GPU2 | yes |
| 6 | 9212 | sigma_d | GPU2 | yes |
| 7 | 9213 | sigma_e | GPU2 | yes |
| 8 | 9214 | sparse_rs_v2 | GPU2 | yes |
| **9** | **9221** | **sigma_hard_a** | **GPU3** | **hard-only (gap>3)** |
| **10** | **9222** | **sigma_hard_b** | **GPU3** | **hard-only (gap>3)** |
| **11** | **9223** | **jsma_strong** | **GPU3** | **hard-only (gap>3)** |

## Expected impact

Targeted improvement: **min_score lift on hard images**.

| Scenario | Before GPU3 | After GPU3 |
|---|---|---|
| Easy images (gap≤3) | 0.948-0.951 | 0.948-0.951 (unchanged) |
| Hard images (gap>3) | 0.92-0.94 | **0.93-0.95** (+0.01 typical) |
| Validator min_score | 0.9036 | **0.92+** projected |
| Validator avg_score | 0.9435 | **0.945-0.948** projected |

## Failsafe rollback

If GPU3 has issues:
```bash
# On GPU1 — comment out the hard URLs line
sed -i 's|^export PERTURB_HARD_WORKER_URLS|# export PERTURB_HARD_WORKER_URLS|' /root/linf_26/scripts/miner.env
cd /root/linf_26 && bash scripts/run_miner.sh
# Coordinator returns to 8-worker dispatch immediately.
```

## Monitoring

On GPU1:
```bash
pm2 logs perturb-miner --nostream --lines 100 | grep "workers:"
# Watch the hard=True/False ratio. Should be ~30% True (hard image rate).
```

On GPU3:
```bash
pm2 list  # check all 3 workers online
nvidia-smi  # check VRAM usage (~18 GB across 3 workers)
```
