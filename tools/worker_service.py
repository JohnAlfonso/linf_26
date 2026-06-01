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
    _boost_margin_on_mask,
    _predict_idx_roundtrip,
    _rank_targets_by_cost,
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


def _strategy_sigma_hard_a(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero zeros-init, B=8, n_iter=400 — large batch + extended iters,
    designed for the RTX PRO 6000 (96GB) on GPU3. 8 parallel target classes
    per pass = best Pareto-K coverage. Only dispatched on hard images so the
    longer compute is justified."""
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
        magnitude=1.0 / 255.0, n_iterations=400,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targets,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sigma_hard_b(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero random-init B=8, RNG seed=99, n_iter=400 — upgraded to B=8
    for RTX PRO 6000. Additional restart basin with full 8-target Pareto.
    Together with sigma_b/d/e, completes a 4-seed × multi-batch ensemble."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=8,
    )
    targets: list[int | None] = [None]
    for rup in runner_ups[1:8]:
        targets.append(int(rup))
    B = len(targets)
    d = clean.numel()
    gen = torch.Generator(device=device).manual_seed(99)
    init_u_batch = (
        (torch.rand((B, d), generator=gen, device=device) * 2.0 - 1.0) * 0.5
    )
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=400,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targets,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sigma_max(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero zeros-init, B=12 (maximum parallel targets), n_iter=300.
    Designed for RTX PRO 6000 (96GB) on GPU3 — uses ~24 GB VRAM alone.
    12 simultaneous target classes is the most parallel Pareto-K coverage
    we can run; the 'sigma_*' workers run B=2/4/8, sigma_max pushes to 12.
    Only dispatched on hard images (gap > 3) via HARD_WORKER_URLS."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=12,
    )
    targets: list[int | None] = [None]
    for rup in runner_ups[1:12]:
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


def _malt_pick_targets(
    model: torch.nn.Module,
    clean: torch.Tensor,
    true_idx: int,
    n_candidates: int,
    n_pick: int,
) -> list[int]:
    """MALT-style target picker (Lim et al. 2024, arXiv:2407.02240).

    Re-ranks the top-`n_candidates` runner-up classes by Jacobian-normalized
    attack cost `gap / ||∇_x(logit[true]−logit[r])||_∞` (cheapest = most
    L∞-efficient flip target), then returns the `n_pick` cheapest. MALT's
    central finding: the optimal sparse-attack target is *often not* the
    runner-up — sometimes ranked 18th or 52nd in raw logits. Using cost
    ranking instead of logit ranking is the entire `MALT` contribution.

    Cost: ~`n_candidates` backward passes (~35ms each under MPS) — the
    caller's deadline must accommodate this pre-stage.
    """
    ranked = _rank_targets_by_cost(
        model=model, clean=clean, true_idx=true_idx, num_candidates=n_candidates,
    )
    if not ranked:
        return _top_runner_ups(
            model=model, clean=clean, true_idx=true_idx, n=n_pick,
        )
    return ranked[:n_pick]


def _strategy_sparse_pgd(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """Sparse-PGD (Zhong et al. 2024, arXiv:2405.05075) — explicit binary
    mask + sign-only magnitude formulation, with random mask re-init on
    stagnation to escape σ-zero's soft-L0 local minima.

    Pre-stage: MALT target picking (top-2 cheapest from top-20 candidates).
    Pipeline: for each target × each K in [1024, 512, 256], run ~15 PGD
    iters with mask fixed, sign updated by descent gradient. After
    `patience` iters of no margin improvement, randomly mix mask (half
    grad-top-K, half random positions with random signs).

    Why this helps on the 0.9043 floor: σ-zero's soft mask can get stuck
    in basins for high-confidence images. The random restart lets Sparse-PGD
    try qualitatively different pixel subsets, which sometimes finds flips
    σ-zero misses entirely (the 0.0000 ship cases).
    """
    magnitude = 1.0 / 255.0
    d = clean.numel()

    targets = _malt_pick_targets(
        model=model, clean=clean, true_idx=target_idx,
        n_candidates=20, n_pick=2,
    )
    if not targets:
        return clean.clone()

    best_adv = clean.clone()
    best_k = d + 1

    K_schedule = (1024, 512, 256)
    n_pgd_per_K = 15
    patience = 5

    for target_t in targets:
        if time.time() >= deadline:
            break
        rng = torch.Generator(device=device).manual_seed(7 + int(target_t))

        # Initial single-step gradient at clean — picks the seed top-K mask.
        adv_g = clean.detach().requires_grad_(True)
        logits = logits_for_images(model=model, image_bchw=adv_g.unsqueeze(0))[0]
        margin = logits[target_idx] - logits[target_t]
        try:
            grad_clean = torch.autograd.grad(margin, adv_g)[0].detach().view(-1)
        except Exception as exc:
            logger.warning(f"sparse_pgd init grad failed for target {target_t}: {exc}")
            continue

        for K_attempt in K_schedule:
            if time.time() >= deadline:
                break

            _, top_idx = grad_clean.abs().topk(K_attempt)
            mask = torch.zeros(d, device=device)
            mask[top_idx] = 1.0
            sign = -grad_clean.sign()

            best_margin_K = float("inf")
            stagnant = 0

            for it in range(n_pgd_per_K):
                if time.time() >= deadline:
                    break

                quant_delta = magnitude * sign * mask
                adv_q_flat = (clean.view(-1) + quant_delta).clamp(0.0, 1.0)
                adv_q = adv_q_flat.view_as(clean)

                # Forward+backward for next step.
                adv_g = adv_q.detach().requires_grad_(True)
                logits = logits_for_images(model=model, image_bchw=adv_g.unsqueeze(0))[0]
                margin = logits[target_idx] - logits[target_t]
                cur_margin = float(margin.item())
                try:
                    grad = torch.autograd.grad(margin, adv_g)[0].detach().view(-1)
                except Exception as exc:
                    logger.warning(f"sparse_pgd grad failed: {exc}")
                    break

                if cur_margin < 0:
                    k = int((quant_delta != 0).sum().item())
                    if k < best_k:
                        if _predict_idx_roundtrip(model, adv_q) != target_idx:
                            best_k = k
                            best_adv = adv_q.detach().clone()

                if cur_margin < best_margin_K - 1e-4:
                    best_margin_K = cur_margin
                    stagnant = 0
                else:
                    stagnant += 1

                # Sign update on masked positions (descent).
                sign = torch.where(mask > 0, -grad.sign(), sign)

                if stagnant >= patience and it < n_pgd_per_K - 1:
                    k_keep = K_attempt // 2
                    _, gt_idx = grad.abs().topk(k_keep)
                    grad_mask = torch.zeros(d, device=device)
                    grad_mask[gt_idx] = 1.0
                    perm = torch.randperm(d, generator=rng, device=device)
                    new_mask = grad_mask.clone()
                    new_mask[perm[: K_attempt - k_keep]] = 1.0
                    rand_sign = (
                        torch.randint(0, 2, (d,), generator=rng, device=device).float()
                        * 2.0 - 1.0
                    )
                    # Gradient-aligned sign on the grad-kept half, random
                    # sign on the freshly-injected random half.
                    sign = torch.where(grad_mask > 0, -grad.sign(), rand_sign)
                    mask = (new_mask > 0).float()
                    stagnant = 0
                    best_margin_K = float("inf")

    return best_adv


def _strategy_sigma_max_malt(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero zeros-init, B=12, n_iter=300 — but with MALT target picking
    instead of raw runner-up ordering.

    Identical to sigma_max except the 11 targeted rows aim at the MALT
    top-11 cheapest classes (Jacobian-normalized cost) from a top-30
    candidate pool. The paper's claim: optimal targets are often ranked
    18-52 by logits but rank-1 by attack cost. If true on our images,
    sigma_max_malt should find lower-K flips than sigma_max on hard cases.

    Pre-stage adds ~1s (30 backward passes); compensated by n_iter=240
    instead of 300.
    """
    targets = _malt_pick_targets(
        model=model, clean=clean, true_idx=target_idx,
        n_candidates=30, n_pick=11,
    )
    if not targets:
        return clean.clone()
    targeted_batch: list[int | None] = [None]
    targeted_batch.extend(int(t) for t in targets)
    B = len(targeted_batch)
    d = clean.numel()
    init_u_batch = torch.zeros((B, d), device=device)
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=240,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targeted_batch,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_greedy_fool(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """GreedyFool (Dong et al., NeurIPS 2020) adapted for L∞=1/255 sparse
    adversarial attack. Two-stage greedy: SELECT pixels to flip, REDUCE
    by dropping least-important ones while preserving flip.

    Paper-reported numbers on ImageNet ResNet: 3× lower K than SparseFool
    (K=27 vs K=80 at full magnitude). Different algorithm family from
    σ-zero/Sparse-RS/JSMA — should add diversity to the ensemble.

    Adaptation (vs original paper):
    - L∞=1/255 hard constraint (paper uses magnitude=255 / no constraint)
    - No GAN-based invisibility map (we only optimize K, not perceptual)
    - Per-pixel margin-gradient as removal-cost proxy (paper uses full
      forward per pixel — too slow for our 9s budget; ours is gradient-
      based linear approximation that batches removals)
    - MALT-picked single target (paper uses untargeted runner-up)
    """
    magnitude = 1.0 / 255.0
    d = clean.numel()

    # Pre-stage: MALT target picking
    targets = _malt_pick_targets(
        model=model, clean=clean, true_idx=target_idx,
        n_candidates=15, n_pick=1,
    )
    if not targets:
        return clean.clone()
    target_t = int(targets[0])

    clean_flat = clean.view(-1)
    adv_flat = clean_flat.clone()
    mask = torch.zeros(d, device=device, dtype=torch.bool)

    def _margin_at_flat(flat: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Returns (margin, gradient_flat). margin = logit[true] - logit[target_t]
        (negative means flipped)."""
        x = flat.view_as(clean).detach().requires_grad_(True)
        lg = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
        m = lg[target_idx] - lg[target_t]
        g = torch.autograd.grad(m, x)[0].detach().view(-1)
        return float(m.item()), g

    # ── Stage 1: SELECT — grow K until PNG-roundtripped flip occurs ─────
    K = 32
    growth = 2
    K_max = 1024
    flipped = False

    for _ in range(8):
        if time.time() >= deadline:
            break
        _, grad = _margin_at_flat(adv_flat)
        # Mask out already-perturbed pixels by setting |grad| to -1
        scores = grad.abs().clone()
        scores[mask] = -1.0
        n_new = min(K, int(d - mask.sum().item()))
        if n_new <= 0:
            break
        _, new_idx = scores.topk(n_new)
        # ±magnitude descent step on these pixels
        step = -magnitude * grad.sign()
        adv_flat[new_idx] = (clean_flat[new_idx] + step[new_idx]).clamp(0.0, 1.0)
        mask[new_idx] = True
        # PNG-roundtripped flip check (validator's view)
        if _predict_idx_roundtrip(model, adv_flat.view_as(clean)) != target_idx:
            flipped = True
            break
        K = min(K * growth, K_max)

    if not flipped:
        # Couldn't find flip even at K_max — return clean (coordinator
        # will skip this candidate since margin < 0).
        return clean.clone()

    # ── Stage 2: REDUCE — drop pixels while preserving flip ─────────────
    # Approximate per-pixel "removal cost" via gradient × current delta.
    # Removing pixel i changes margin by ≈ grad[i] * (clean[i] - adv[i]).
    # Pixels where this is most-negative (margin decreases further, i.e.
    # flip strengthens or stays) are safe to remove.
    best_adv = adv_flat.clone()
    best_k = int(mask.sum().item())

    for _round in range(20):
        if time.time() >= deadline:
            break
        n_perturbed = int(mask.sum().item())
        if n_perturbed <= 4:
            break
        perturbed_idx = mask.nonzero(as_tuple=False).view(-1)
        _, grad = _margin_at_flat(adv_flat)
        delta = adv_flat - clean_flat
        # Margin change if we REVERT pixel i (set delta[i] = 0):
        # Δmargin ≈ grad[i] * (clean[i] - adv[i]) = grad[i] * (-delta[i])
        # Want margin to stay negative (flip survives). Pixels where
        # Δmargin is most negative (further decrease) are safest to drop.
        removal_cost = grad[perturbed_idx] * (-delta[perturbed_idx])
        _, order = removal_cost.sort()  # ascending: safest first

        # Try batch removal: largest batch first, halve on failure.
        batch = max(1, n_perturbed // 4)
        succeeded_this_round = False
        while batch >= 1:
            if time.time() >= deadline:
                break
            cand = perturbed_idx[order[:batch]]
            test_adv = adv_flat.clone()
            test_adv[cand] = clean_flat[cand]
            if _predict_idx_roundtrip(model, test_adv.view_as(clean)) != target_idx:
                adv_flat = test_adv
                mask[cand] = False
                succeeded_this_round = True
                new_k = int(mask.sum().item())
                if new_k < best_k:
                    best_k = new_k
                    best_adv = adv_flat.clone()
                break
            batch //= 2

        if not succeeded_this_round:
            break

    return best_adv.view_as(clean)


def _strategy_sigma_grind(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero B=1 (single MALT-picked target), n=900 — deep grind for
    lowest possible K on easy images.

    Pintor 2024 explicitly demonstrates that single-target σ-zero finds
    lower-K solutions than multi-target setups: with B>1, the soft-L0
    penalty has to compromise across multiple target classes that compete
    for the same pixel budget. B=1 lets the optimizer focus all
    descent steps on the single cheapest target, driving K below what
    sigma_a (B=2) / sigma_max (B=12) plateau at.

    Pre-stage: MALT picks the cheapest of top-20 targets (~700ms).
    Main loop: B=1 × n=900 ≈ 8.5s on RTX PRO 6000.
    Post-stage: K-preserving sign-flip boost pushes margin from ~0.01 to
    ~0.10+ without adding pixels — prevents the margin-aware swap in
    neurons/miner.py from overriding sigma_grind's low-K candidates.

    Designed to attack the easy-image K gap: top miners ship K~15-25 on
    easy images via low-K-tuned attacks; our multi-target σ-zero plateaus
    at K~40-50. sigma_grind targets that exact regime.
    """
    targets = _malt_pick_targets(
        model=model, clean=clean, true_idx=target_idx,
        n_candidates=20, n_pick=1,
    )
    if not targets:
        return clean.clone()
    targeted_batch: list[int | None] = [int(targets[0])]
    d = clean.numel()
    init_u_batch = torch.zeros((1, d), device=device)
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=900,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targeted_batch,
        # Push sparsity harder than the shared default (0.01/0.01): once the
        # single target flips, grow τ at 2× the shrink rate to shed more
        # pixels. B=1 + n=900 gives the budget to chase the lower-K regime
        # the leaders reach; the verified-flip best_K guard makes this safe.
        tau_grow=0.02, tau_shrink=0.01,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sigma_grind_b(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero B=1, MALT top-2 (2nd cheapest target), n=900 — companion to
    sigma_grind that attacks the SECOND-best target. MALT's top-1 isn't
    always the optimal flip target (especially on borderline cases where
    the cost ranking has multiple cheap candidates within a few %). Running
    a B=1 grind on top-2 gives a second independent chance at the lowest-K
    regime. Coordinator picks the higher-score candidate via FP32 scoring.
    """
    targets = _malt_pick_targets(
        model=model, clean=clean, true_idx=target_idx,
        n_candidates=20, n_pick=2,
    )
    if len(targets) < 2:
        # Fall back to top-1 if MALT only finds one target.
        if not targets:
            return clean.clone()
        target = int(targets[0])
    else:
        target = int(targets[1])
    targeted_batch: list[int | None] = [target]
    d = clean.numel()
    init_u_batch = torch.zeros((1, d), device=device)
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=900,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targeted_batch,
        tau_grow=0.02, tau_shrink=0.01,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_sigma_grind_c(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """σ-zero B=1, MALT top-1 target, n=900, RANDOM init (seed=42) — same
    target as sigma_grind but starts from a different basin. The single-
    target optimizer's local minimum depends on initialization; random
    init reaches different basins than zeros-init, sometimes finding
    lower-K solutions sigma_grind misses. Together with sigma_grind/_b,
    this forms a 3-restart B=1 grind ensemble (top-1-zeros, top-2-zeros,
    top-1-random) — coordinator picks the best by score.
    """
    targets = _malt_pick_targets(
        model=model, clean=clean, true_idx=target_idx,
        n_candidates=20, n_pick=1,
    )
    if not targets:
        return clean.clone()
    targeted_batch: list[int | None] = [int(targets[0])]
    d = clean.numel()
    gen = torch.Generator(device=device).manual_seed(42)
    # Random init in [-0.5, 0.5] — same scale as sigma_b/d/e random inits.
    init_u_batch = (
        (torch.rand((1, d), generator=gen, device=device) * 2.0 - 1.0) * 0.5
    )
    adv_b, _k = _sigma_zero_batched(
        model=model, clean=clean, target_idx=target_idx,
        magnitude=1.0 / 255.0, n_iterations=900,
        init_u_batch=init_u_batch,
        targeted_idx_batch=targeted_batch,
        tau_grow=0.02, tau_shrink=0.01,
        deadline=deadline,
    )
    return adv_b if adv_b is not None else clean.clone()


def _strategy_jsma_strong(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    deadline: float,
) -> torch.Tensor:
    """JSMA with backward_eliminate=True — produces LOW K (~300) but slow
    (~15-25s under MPS). Only viable for hard-image-only deployment where
    we have full deadline budget and don't compete with other workers.

    backward_eliminate runs O(K) per-pixel removal pass which drops K
    from JSMA's typical 2,187 to ~300 — directly competitive with σ-zero
    on K count, with a different algorithm family.
    """
    last_adv = clean.clone()
    try:
        for adv in attack_jsma_greedy(
            model=model, clean=clean, target_idx=target_idx, device=device,
            magnitude=1.0 / 255.0,
            batch_pixels=16, max_k_fraction=0.05,
            num_runner_ups=3,
            backward_eliminate=True,  # KEY: enable K-reduction
            cluster_kernel=3,
            yield_after_flip_batches=2,
            n_batches_cap=50,
            deadline=deadline,
        ):
            last_adv = adv
            if time.time() >= deadline:
                break
    except Exception as exc:
        logger.warning(f"jsma_strong failed: {exc}")
    return last_adv


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
    # New strategies for GPU3 (workers 9-12) — hard-image-only:
    "sigma_hard_a": _strategy_sigma_hard_a,
    "sigma_hard_b": _strategy_sigma_hard_b,
    "jsma_strong": _strategy_jsma_strong,
    "sigma_max": _strategy_sigma_max,
    "sparse_pgd": _strategy_sparse_pgd,
    "sigma_max_malt": _strategy_sigma_max_malt,
    "sigma_grind": _strategy_sigma_grind,
    "sigma_grind_b": _strategy_sigma_grind_b,
    "sigma_grind_c": _strategy_sigma_grind_c,
    "greedy_fool": _strategy_greedy_fool,
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
