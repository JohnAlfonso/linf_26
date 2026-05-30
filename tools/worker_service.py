"""Worker service for distributed multi-GPU mining.

Each GPU server runs this as a standalone FastAPI process. The
coordinator (neurons/miner.py) dispatches POST /attack to N workers in
parallel, gathers the results, scores each via the FP32-strict
verify_and_score, and ships the best.

Workers use TF32 for SPEED — the coordinator does final flip verification
in FP32 to keep validator-side correctness. This split lets workers run
~25% more σ-zero iterations than the coordinator could in the same time.

Each worker is configured via WORKER_STRATEGY env var to run a different
attack — diversity is what produces the Pareto win, not just N×speed.

Strategies:
  sigma_a   — σ-zero zeros-init, B=2 batched, N=300, T=2
  sigma_b   — σ-zero random-init, B=4 batched, N=240, T=4 (more targets)
  jsma      — JSMA-greedy with cluster_kernel=3 (different algorithm
              family, escapes σ-zero local minima on hard images)

Endpoints:
  GET  /health             — readiness check
  POST /attack             — run the configured attack
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from perturbnet.attacks import (
    _predict_idx_roundtrip,
    _shrink_support,
    _sigma_zero_batched,
    _sparse_rs_run,
    _top_runner_ups,
    attack_jsma_greedy,
    attack_square_linf,
)
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import (
    load_efficientnet_v2_l,
    logits_for_images,
    normalize_prediction_label,
    resolve_target_index,
)


logger = logging.getLogger("worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

WORKER_STRATEGY = os.getenv("WORKER_STRATEGY", "sigma_a").lower()
WORKER_PORT = int(os.getenv("WORKER_PORT", "9200"))
WORKER_HOST = os.getenv("WORKER_HOST", "0.0.0.0")
WORKER_DEADLINE_S = float(os.getenv("WORKER_DEADLINE_S", "9.0"))
# Workers use TF32 by default for ~25% faster forwards. Coordinator
# re-checks flip via FP32 verify_and_score, so this is safe.
WORKER_DISABLE_TF32 = os.getenv("WORKER_DISABLE_TF32", "").lower() in {
    "1", "true", "yes", "on",
}


class WorkerState:
    model: torch.nn.Module | None = None
    device: torch.device | None = None
    lock = threading.Lock()
    served: int = 0
    flipped: int = 0


_state = WorkerState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Loading EfficientNetV2-L on {_state.device} "
        f"strategy={WORKER_STRATEGY} tf32={'off' if WORKER_DISABLE_TF32 else 'on'}"
    )
    if _state.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if WORKER_DISABLE_TF32:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        else:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    t0 = time.time()
    _state.model = load_efficientnet_v2_l(_state.device)
    # Warmup — populate cuDNN cache before first request.
    with torch.no_grad():
        for _ in range(3):
            _ = logits_for_images(
                model=_state.model,
                image_bchw=torch.rand(1, 3, 480, 480, device=_state.device),
            )
    gx = torch.rand(1, 3, 480, 480, device=_state.device, requires_grad=True)
    logits_for_images(model=_state.model, image_bchw=gx).sum().backward()
    if _state.device.type == "cuda":
        torch.cuda.synchronize()
    logger.info(
        f"Worker ready in {time.time()-t0:.1f}s — listening on {WORKER_HOST}:{WORKER_PORT}"
    )
    yield
    _state.model = None


app = FastAPI(title=f"Worker-{WORKER_STRATEGY}", version="0.1.0", lifespan=lifespan)


class AttackRequest(BaseModel):
    clean_image_b64: str = Field(..., min_length=1)
    true_label: str = Field(..., min_length=1)
    epsilon: float = Field(..., gt=0.0, le=1.0)
    deadline_s: float = Field(WORKER_DEADLINE_S, gt=0.0, le=14.0)


class AttackResponse(BaseModel):
    adv_image_b64: str
    k: int
    norm: float
    rmse: float
    margin: float
    flipped: bool
    strategy: str
    elapsed_ms: int


def _margin_at(
    model: torch.nn.Module, adv: torch.Tensor, true_idx: int,
) -> tuple[float, int]:
    """Returns (margin = runner_up_logit − true_logit, runner_up_idx).
    Positive margin means the adversarial flip exists."""
    with torch.no_grad():
        lg = logits_for_images(model=model, image_bchw=adv.unsqueeze(0))[0]
    masked = lg.clone()
    masked[true_idx] = float("-inf")
    r = int(masked.argmax().item())
    return float((lg[r] - lg[true_idx]).item()), r


def _strategy_sigma_a(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero zeros-init, B=2 batched (untargeted + 1 targeted)."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=2,
    )
    targets: list[int | None] = [None]
    for rup in runner_ups[1:]:
        targets.append(int(rup))
    B = len(targets)
    d = clean.numel()
    init_u_batch = torch.zeros((B, d), device=device)
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=300,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targets,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sigma_b(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero with random init, B=4 batched (2 untargeted + 2 targeted at
    runner-ups #1, #2, #3). Random init diversifies the convergence —
    different starting points often find different (K, margin) tradeoffs."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=4,
    )
    targets: list[int | None] = [None]
    for rup in runner_ups[1:4]:
        targets.append(int(rup))
    B = len(targets)
    d = clean.numel()
    gen = torch.Generator(device=device).manual_seed(1)
    init_u_batch = (
        (torch.rand((B, d), generator=gen, device=device) * 2.0 - 1.0) * 0.5
    )
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=240,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targets,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_jsma(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """JSMA-greedy with cluster_kernel=3 — different algorithm family.
    Picks pixels by saliency one batch at a time. Doesn't have σ-zero's
    soft-L0 relaxation, so it escapes σ-zero local minima on hard
    images. Returns the LAST yielded candidate (greedy-grown K).

    Earlier runs under MPS contention with the old config (5 runner_ups,
    n_batches_cap=250) took 35s — the inner deadline check only fired
    between yields, so cases that never flipped on the first runner_up
    ran the full 250 batches. Now we pass `deadline` into the inner loop
    and shrink the search (3 runner_ups, n_batches_cap=50). Still gives
    JSMA the algorithmic diversity benefit on hard images.
    """
    last_adv = clean.clone()
    try:
        for adv in attack_jsma_greedy(
            model=model, clean=clean, target_idx=target_idx, device=device,
            magnitude=1.0 / 255.0,
            batch_pixels=16, max_k_fraction=0.05,
            num_runner_ups=3,
            # backward_eliminate=True is O(K) forward passes, which under
            # MPS contention turned out to dominate JSMA's wall time
            # (k=463 → 30+s). JSMA's contribution here is algorithmic
            # diversity, not K minimization (σ-zero workers handle that);
            # the coordinator picks the best of all 3 anyway.
            backward_eliminate=False,
            cluster_kernel=3,
            yield_after_flip_batches=2,
            n_batches_cap=50,
            deadline=deadline,
        ):
            last_adv = adv
            if time.time() >= deadline:
                break
    except Exception as exc:
        logger.warning(f"jsma failed: {exc}")
    return last_adv


def _strategy_square(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """Square Attack — kept available but UNUSED in default deployment.
    Smoke testing showed k=28k+ on flips that don't survive PNG roundtrip,
    far worse than σ-zero's k=276. Square Attack's patch primitive is
    L∞-budget-optimized, not low-K-optimized. Left in the strategy
    registry for ad-hoc experiments only."""
    try:
        best_adv, _margin, _k = attack_square_linf(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=1.0 / 255.0,
            n_queries=400, p_init=0.001, h_min=1, rng_seed=42,
            deadline=deadline,
        )
        return best_adv
    except Exception as exc:
        logger.warning(f"square failed: {exc}")
        return clean.clone()


def _strategy_sparse_rs(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero warm-start → Sparse-RS refinement.

    Phase 1 (~3s): brief σ-zero (B=1, n_iter=80) produces a seed delta.
    Phase 2 (~6s): _sparse_rs_run (random local search) refines K — adds
    pixels until flip, then drops pixels while preserving the flip.

    Key property: phase 2 is FORWARD-ONLY (no backward), so under MPS
    contention each iteration is ~30ms (vs σ-zero's ~70ms with backward).
    Lower GPU pressure than sigma_a/sigma_b during the RS phase.

    Diversity vs other workers:
    - sigma_a/sigma_b: pure σ-zero, no RS refinement
    - jsma: gradient-greedy (no random exploration)
    - sparse_rs: σ-zero seed + stochastic K-minimization via random search
    """
    t_start = time.time()
    # Phase 1: σ-zero seed (B=1, fast). Budget: ~35% of total deadline.
    seed_deadline = t_start + min(3.5, (deadline - t_start) * 0.35)
    runner_ups = _top_runner_ups(model=model, clean=clean, true_idx=target_idx, n=1)
    targets: list[int | None] = [None]
    d = clean.numel()
    init_u_batch = torch.zeros((1, d), device=device)
    try:
        seed_adv, _k = _sigma_zero_batched(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=1.0 / 255.0, n_iterations=80,
            init_u_batch=init_u_batch,
            targeted_idx_batch=targets,
            deadline=seed_deadline,
        )
    except Exception as exc:
        logger.warning(f"sparse_rs seed phase (σ-zero) failed: {exc}")
        seed_adv = None

    if seed_adv is None:
        seed_delta = torch.zeros_like(clean)
    else:
        seed_delta = (seed_adv - clean).detach()

    # Phase 2: Sparse-RS refinement. Forward-only, lighter GPU.
    try:
        rs_adv, _rs_k = _sparse_rs_run(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=1.0 / 255.0,
            n_queries=300,
            seed_delta=seed_delta,
            rng_seed=7,
            max_swap=3,
            deadline=deadline,
        )
        if rs_adv is not None:
            return rs_adv
    except Exception as exc:
        logger.warning(f"sparse_rs refinement failed: {exc}")

    # Fallback: ship the σ-zero seed itself (no refinement).
    return (clean + seed_delta).clamp(0.0, 1.0)


def _strategy_sigma_c(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero zeros-init, B=8 batched (1 untargeted + 7 targeted at top
    runner-ups). Largest batch = most parallel Pareto-K candidates per
    forward pass — better K convergence. Designed for GPU2 (RTX PRO 6000)
    which has the VRAM headroom for B=8.

    Diversity: vs sigma_a (B=2), this samples 4× more target classes per
    iteration → finds optimal target much faster on hard images."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=8,
    )
    targets: list[int | None] = [None]
    for rup in runner_ups[1:8]:
        targets.append(int(rup))
    B = len(targets)
    d = clean.numel()
    init_u_batch = torch.zeros((B, d), device=device)
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=200,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targets,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sigma_d(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero random-init, B=4, RNG seed=17. Different basin from
    sigma_b (seed=1) — increases probability of finding a low-K
    solution on hard images via stochastic restart."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=4,
    )
    targets: list[int | None] = [None]
    for rup in runner_ups[1:4]:
        targets.append(int(rup))
    B = len(targets)
    d = clean.numel()
    gen = torch.Generator(device=device).manual_seed(17)
    init_u_batch = (
        (torch.rand((B, d), generator=gen, device=device) * 2.0 - 1.0) * 0.5
    )
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=240,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targets,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sigma_e(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero random-init, B=4, RNG seed=42. Third restart basin —
    completes a Markov-style multi-restart cluster (seeds 1, 17, 42)
    across sigma_b, sigma_d, sigma_e. Coordinator picks the lowest-K
    result from all three."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=4,
    )
    targets: list[int | None] = [None]
    for rup in runner_ups[1:4]:
        targets.append(int(rup))
    B = len(targets)
    d = clean.numel()
    gen = torch.Generator(device=device).manual_seed(42)
    init_u_batch = (
        (torch.rand((B, d), generator=gen, device=device) * 2.0 - 1.0) * 0.5
    )
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=240,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targets,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sparse_rs_v2(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """Sparse-RS warmstart with RNG seed=99 (vs sparse_rs's seed=7).
    Same algorithm as sparse_rs but different random trajectory →
    different (K, margin) tradeoff on a given image. Together with
    sparse_rs, this acts as a 2-restart Sparse-RS ensemble."""
    t_start = time.time()
    seed_deadline = t_start + min(3.5, (deadline - t_start) * 0.35)
    runner_ups = _top_runner_ups(model=model, clean=clean, true_idx=target_idx, n=1)
    targets: list[int | None] = [None]
    d = clean.numel()
    init_u_batch = torch.zeros((1, d), device=device)
    try:
        seed_adv, _k = _sigma_zero_batched(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=1.0 / 255.0, n_iterations=80,
            init_u_batch=init_u_batch,
            targeted_idx_batch=targets,
            deadline=seed_deadline,
        )
    except Exception as exc:
        logger.warning(f"sparse_rs_v2 seed phase (σ-zero) failed: {exc}")
        seed_adv = None
    if seed_adv is None:
        seed_delta = torch.zeros_like(clean)
    else:
        seed_delta = (seed_adv - clean).detach()
    try:
        rs_adv, _rs_k = _sparse_rs_run(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=1.0 / 255.0,
            n_queries=300,
            seed_delta=seed_delta,
            rng_seed=99,   # different from sparse_rs's seed=7
            max_swap=3,
            deadline=deadline,
        )
        if rs_adv is not None:
            return rs_adv
    except Exception as exc:
        logger.warning(f"sparse_rs_v2 refinement failed: {exc}")
    return (clean + seed_delta).clamp(0.0, 1.0)


_STRATEGIES = {
    "sigma_a": _strategy_sigma_a,
    "sigma_b": _strategy_sigma_b,
    "jsma": _strategy_jsma,
    "square": _strategy_square,
    "sparse_rs": _strategy_sparse_rs,
    # New strategies for GPU2 (workers 5-8):
    "sigma_c": _strategy_sigma_c,
    "sigma_d": _strategy_sigma_d,
    "sigma_e": _strategy_sigma_e,
    "sparse_rs_v2": _strategy_sparse_rs_v2,
}


def _run_attack(req: AttackRequest) -> AttackResponse:
    if _state.model is None or _state.device is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    model = _state.model
    device = _state.device

    try:
        clean = decode_image_b64(req.clean_image_b64).to(device)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc

    true_label = normalize_prediction_label(req.true_label)
    target_idx = resolve_target_index(true_label)
    if target_idx is None:
        raise HTTPException(
            status_code=400,
            detail=f"unresolvable true_label: {req.true_label!r}",
        )

    fn = _STRATEGIES.get(WORKER_STRATEGY)
    if fn is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"unknown WORKER_STRATEGY={WORKER_STRATEGY!r}; "
                f"available: {list(_STRATEGIES)}"
            ),
        )

    t_start = time.time()
    deadline_abs = t_start + min(float(req.deadline_s), WORKER_DEADLINE_S)
    with _state.lock:
        adv = fn(model, clean, target_idx, device, deadline_abs)

    # PNG roundtrip to validator's view + measurements.
    adv_b64 = encode_image_b64(adv.detach().cpu())
    adv_rt = decode_image_b64(adv_b64).to(device)

    delta = adv_rt - clean
    norm = float(delta.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(delta ** 2)).item())
    k = int((delta.abs() > 1e-9).sum().item())

    margin, _ = _margin_at(model, adv_rt, target_idx)
    flipped = _predict_idx_roundtrip(model, adv_rt) != target_idx

    _state.served += 1
    if flipped:
        _state.flipped += 1

    return AttackResponse(
        adv_image_b64=adv_b64,
        k=k, norm=norm, rmse=rmse, margin=margin, flipped=flipped,
        strategy=WORKER_STRATEGY,
        elapsed_ms=int((time.time() - t_start) * 1000),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _state.model is not None else "loading",
        "strategy": WORKER_STRATEGY,
        "device": str(_state.device) if _state.device else None,
        "served": _state.served,
        "flipped": _state.flipped,
        "tf32": (
            torch.backends.cuda.matmul.allow_tf32
            if _state.device and _state.device.type == "cuda"
            else None
        ),
    }


@app.post("/attack", response_model=AttackResponse)
def attack(req: AttackRequest) -> AttackResponse:
    return _run_attack(req)


def main() -> None:
    import uvicorn
    uvicorn.run(
        "tools.worker_service:app",
        host=WORKER_HOST, port=WORKER_PORT, reload=False, workers=1,
    )


if __name__ == "__main__":
    main()
