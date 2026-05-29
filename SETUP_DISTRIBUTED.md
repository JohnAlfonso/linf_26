# Distributed Multi-GPU Mining Setup

This guide walks through deploying the miner across **3 GPU servers + 1 coordinator** to compete with multi-server top miners.

## Architecture

```
                   Validator
                       │ (Linf adversarial request)
                       ▼
            ┌────────────────────────┐
            │   COORDINATOR          │  Server 0 (the wallet host)
            │   neurons/miner.py     │  bittensor axon + boost + scoring
            └────────────────────────┘
                  │ HTTP/JSON (async parallel)
       ┌──────────┼─────────────────┐
       ▼          ▼                 ▼
   ┌──────┐  ┌──────┐           ┌──────┐
   │ GPU 1│  │ GPU 2│           │ GPU 3│
   │sigma_a│ │sigma_b│          │ jsma │  Servers 1-3 (GPU workers)
   └──────┘  └──────┘           └──────┘
   tools/worker_service.py
```

Each worker runs a **different attack** — diversity (not redundancy) closes the gap to the top miner. Coordinator picks the best of 3 + applies boost.

## Quickstart

### Step 1 — Provision 3 GPU servers

Each needs:
- NVIDIA GPU (≥8GB VRAM, RTX 30xx+/A4000+ recommended)
- Python 3.10+, CUDA-capable PyTorch
- Open port 9200 (or any port you'll set in `WORKER_PORT`) to the coordinator's IP

### Step 2 — Install on each GPU server

```bash
git clone <repo-url> linf_26 && cd linf_26
bash scripts/setup_common.sh miner    # installs torch + deps
cp scripts/worker.env.example scripts/worker.env
```

Edit `scripts/worker.env`:
```bash
export WORKER_STRATEGY="sigma_a"   # ← server 1 uses sigma_a
                                    # ← server 2 uses sigma_b
                                    # ← server 3 uses jsma
export WORKER_PORT="9200"
export WORKER_HOST="0.0.0.0"
export WORKER_DEADLINE_S="9.0"
```

Each server must use a **different** `WORKER_STRATEGY` value. They are:
- `sigma_a` — σ-zero zeros-init, B=2 batched, N=300
- `sigma_b` — σ-zero random-init, B=4 batched, N=240, more target classes
- `jsma`   — JSMA-greedy cluster_kernel=3 (different algorithm family)

Then start the worker (PM2 auto-restart):
```bash
bash scripts/run_worker.sh
```

Verify:
```bash
curl http://localhost:9200/health
# {"status":"ok","strategy":"sigma_a","device":"cuda","served":0,"flipped":0,"tf32":true}
```

### Step 3 — Configure coordinator (wallet server)

On the server that runs the bittensor wallet/miner, edit `scripts/miner.env`:

```bash
# Point at the 3 worker servers (use private IPs if same VPC)
export PERTURB_WORKER_URLS="http://10.0.1.1:9200,http://10.0.1.2:9200,http://10.0.1.3:9200"

# Coordinator's per-worker timeout. Should be ~2s above worker's own deadline.
export PERTURB_WORKER_TIMEOUT_S="11.0"
```

Install httpx (only needed on coordinator):
```bash
.venv/bin/pip install httpx
```

Restart the coordinator miner:
```bash
pm2 restart perturb-miner
pm2 logs perturb-miner --lines 30
```

You should see request logs that include `worker_strategy=...` indicating distributed dispatch is active.

## Verification

### Per-request log (distributed mode)

```
workers: 3/3 ok, scored=3, best_strategy=jsma, best_score=0.9473, dispatch=10234ms
task=... pipeline_score=0.9473 shipped_score=0.9466 shipped_reason=success ...
```

If you see `falling back to local pipeline`, one or more workers timed out or aren't reachable.

### Per-worker health

```bash
curl http://10.0.1.1:9200/health
# Watch served / flipped counts grow as the validator queries the miner.
```

## Network requirements

For geographically-distributed workers (your case):

| component | budget | typical |
|---|---|---|
| validator → coordinator | network | ~50-150ms |
| coordinator → worker (parallel) | network | ~50-300ms each |
| worker attack execution | wall | 9s |
| worker → coordinator (return) | network | ~50-300ms each |
| coordinator scoring | wall | ~200ms |
| coordinator boost | wall | ~500ms |
| coordinator → validator | network | ~50-150ms |
| **total budget** | **15s** | **~11-12s** |

If round-trip latency from coordinator → worker exceeds **800ms**, the attack budget gets squeezed. Mitigations:
- **Same cloud region / VPC** — strongly preferred
- **HTTP keep-alive** — already used by httpx async client
- **Pre-warm by sending a dummy /attack at startup**
- Lower `WORKER_DEADLINE_S` to compensate for slow networks

## Failure modes & graceful degradation

| scenario | behavior |
|---|---|
| 0 workers reachable | Coordinator falls back to **local pipeline** (current single-GPU code) |
| 1-2 workers reachable, others timeout | Coordinator uses what it got, scores them, picks best |
| All workers return non-flipping advs | Coordinator falls back to local pipeline |
| Coordinator out of memory | Bittensor axon errors → validator picks another miner |
| Validator times out (>15s) | Score 0 — the hard ceiling we can't cross |

Everything fails-safe: the worker mode is purely additive. Empty `PERTURB_WORKER_URLS` = exact current single-GPU behavior.

## Tuning per strategy

If a strategy underperforms (check via `/health` ratio of `flipped/served`):

```bash
# On the underperforming worker:
export WORKER_DEADLINE_S="10.0"        # give it more time
# Or for sigma_a, edit tools/worker_service.py:_strategy_sigma_a
# to use n_iterations=350, etc.
```

## Cost vs benefit (estimated)

| setup | gap to top miner | mean score | cost |
|---|---|---|---|
| Single GPU (current) | +0.0015-0.0025 | 0.9445 | 1× |
| 3-server parallel | **≈ 0 to +0.0005** | **0.9465-0.9475** | 3× |
| 5-server (add Sparse-RS, attack_strong) | possibly negative gap (we beat top) | 0.9470-0.9480 | 5× |

The win is concentrated on **hard images** where σ-zero alone gets stuck. JSMA on GPU-3 specifically targets the cheeseburger/vase/parachute-style cases that were +0.01-0.02 losses in our single-GPU benchmarks.

## Next steps after deployment

1. Watch `received/` JSON files for 100+ requests
2. Compute average score, compare to top miner
3. If still losing, the candidates are:
   - Add a 4th worker (Sparse-RS or attack_binary_search)
   - Tune individual worker strategies based on which one wins most often
   - Pre-compute adversarials for popular ImageNet images (cache by hash)
