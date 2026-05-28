from __future__ import annotations

import logging
import time
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import LABELS, logits_for_images, normalize_prediction_label

logger = logging.getLogger("perturb.attacks")


def _xent_grad_at(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_idx: int,
    device: torch.device,
) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.tensor([target_idx], device=device))
    return torch.autograd.grad(loss, x)[0].detach()


def _find_runner_up(model: torch.nn.Module, clean: torch.Tensor, true_idx: int) -> int:
    """Pick the second-best logit class so we can drive a tight margin attack."""
    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=clean.unsqueeze(0))[0]
    top = logits.topk(2)
    a, b = int(top.indices[0].item()), int(top.indices[1].item())
    return b if a == true_idx else a


def _top_runner_ups(
    model: torch.nn.Module, clean: torch.Tensor, true_idx: int, n: int
) -> list[int]:
    """Return the top-n non-true logit class indices, ordered by logit."""
    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=clean.unsqueeze(0))[0]
    order = logits.argsort(descending=True).tolist()
    return [int(c) for c in order if int(c) != true_idx][:n]


def _margin_grad_at(
    model: torch.nn.Module,
    x: torch.Tensor,
    true_idx: int,
    runner_up_idx: int,
) -> torch.Tensor:
    """Gradient of (logit[true] − logit[runner_up]); minimizing this flips the label
    by exactly the cheapest pair-wise margin."""
    x = x.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    margin = logits[true_idx] - logits[runner_up_idx]
    return torch.autograd.grad(margin, x)[0].detach()


def _rank_targets_by_cost(
    model: torch.nn.Module,
    clean: torch.Tensor,
    true_idx: int,
    num_candidates: int,
) -> list[int]:
    """Order the top runner-ups by estimated attack cost, cheapest first.

    Linear approximation of K for a sparse ±magnitude attack flipping
    `true_idx → r`:

        K_estimate(r) ≈ (logit[true] − logit[r]) / max_i |∇_x(logit[true] − logit[r])_i|

    Reading: distance to the r-vs-true decision boundary, divided by the
    most-responsive pixel direction toward it. Cheaper r flips with fewer
    pixels. Costs ~(num_candidates) forward+backward passes (50-100ms total)
    — much cheaper than running σ-zero per candidate to find out empirically.

    Returns the runner-up class indices ordered cheapest-first."""
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=true_idx, n=int(num_candidates),
    )
    if not runner_ups:
        return []
    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=clean.unsqueeze(0))[0]
    ranked: list[tuple[float, int]] = []
    for r in runner_ups:
        gap = float((logits[true_idx] - logits[r]).item())
        try:
            grad = _margin_grad_at(
                model=model, x=clean, true_idx=true_idx, runner_up_idx=r,
            )
            max_g = float(grad.abs().max().item())
        except Exception:
            max_g = 0.0
        cost = gap / max(1e-12, max_g)
        ranked.append((cost, r))
    ranked.sort(key=lambda t: t[0])
    return [r for _, r in ranked]


def _sparse_topk_oneshot(
    grad: torch.Tensor,
    clean: torch.Tensor,
    k: int,
    magnitude: float,
    descent: bool = False,
) -> torch.Tensor:
    """Perturb top-k positions by |grad| using ±magnitude * sign(grad).
    If descent=True, move opposite the gradient (used for margin-minimization)."""
    flat_abs = grad.view(-1).abs()
    k = max(1, min(k, flat_abs.numel()))
    _, idx = flat_abs.topk(k)
    mask_flat = torch.zeros_like(flat_abs)
    mask_flat[idx] = 1.0
    mask = mask_flat.view_as(grad)
    direction = -1.0 if descent else 1.0
    delta = direction * magnitude * grad.sign() * mask
    return (clean + delta).clamp(0.0, 1.0).detach()


def _predict_idx(model: torch.nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))
        return int(logits.argmax(dim=1).item())


def _predict_idx_roundtrip(model: torch.nn.Module, x: torch.Tensor) -> int:
    """Predict on the PNG-roundtripped tensor — matches the validator's view, so
    the binary search picks K values that survive 8-bit quantization.

    Note: we tried short-circuiting this with a pure-torch quantize
    `round(x*255)/255` on GPU. It is *almost* bit-equivalent to the PIL
    roundtrip but disagreed at the FP-ulp level for some images — enough to
    flip the model's prediction at the decision boundary. So we stay on the
    PIL path here for callers that need bit-exactness with the validator
    (binary_search, rmse_shrink). Performance-sensitive inner loops should
    use the cheaper `_predict_idx` instead and rely on the pipeline's own
    PIL roundtrip + score check to filter false-positive flips."""
    x_rt = decode_image_b64(encode_image_b64(x.detach().cpu())).to(x.device)
    return _predict_idx(model=model, x=x_rt)


def _label_norm(idx: int) -> str:
    """Validator-equivalent label string for an index — `LABELS[idx]` passed
    through `normalize_prediction_label`. The validator decides "flipped" by
    comparing this normalized string, not the raw index, so any internal
    flip-detection in the attack functions must match. ImageNet has *one*
    duplicate-normalized-label pair in IMAGENET1K_V1 (idx 264 vs 474, both
    normalize to 'cardigan'), so most images aren't affected — but the few
    that are would silently lose flips if we kept using index equality."""
    if 0 <= idx < len(LABELS):
        return normalize_prediction_label(LABELS[idx])
    return str(idx)


def _predict_label_norm(model: torch.nn.Module, x: torch.Tensor) -> str:
    return _label_norm(_predict_idx(model=model, x=x))


def _predict_label_norm_roundtrip(model: torch.nn.Module, x: torch.Tensor) -> str:
    return _label_norm(_predict_idx_roundtrip(model=model, x=x))


def _binary_search_min_k(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    sorted_idx: torch.Tensor,
    signs: torch.Tensor,
    magnitude: float,
    max_k: int,
) -> torch.Tensor | None:
    """Doubling + bisection for the smallest K (along the given sorted index
    order) such that perturbing those K positions flips the label. Returns the
    best adversarial tensor, or None if no K ≤ max_k flips."""
    total = clean.numel()

    def _build(k: int) -> torch.Tensor:
        new_delta_flat = torch.zeros(total, dtype=clean.dtype, device=clean.device)
        new_delta_flat[sorted_idx[:k]] = magnitude * signs[:k]
        return (clean + new_delta_flat.view_as(clean)).clamp(0.0, 1.0).detach()

    upper: int | None = None
    last_adv: torch.Tensor | None = None
    k = 32
    while True:
        k = min(k, max_k)
        adv = _build(k)
        last_adv = adv
        if _predict_idx_roundtrip(model, adv) != target_idx:
            upper = k
            break
        if k >= max_k:
            break
        k *= 2

    if upper is None:
        return None

    lo, hi = max(1, upper // 2 + 1), upper
    best_adv = last_adv
    while lo <= hi:
        mid = (lo + hi) // 2
        adv = _build(mid)
        if _predict_idx_roundtrip(model, adv) != target_idx:
            best_adv = adv
            hi = mid - 1
        else:
            lo = mid + 1
    return best_adv


def attack_binary_search(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    magnitude: float = 1.0 / 255.0,
    k_fraction_ceilings: tuple[float, ...] = (0.05, 0.15, 0.4),
    refine_iterations: int = 3,
    num_runner_ups: int = 5,
) -> Iterator[torch.Tensor]:
    """Binary-search for minimum K with iterative gradient refinement.

    Robust fallback chain to handle hard images:
      - try each of the top-N runner-up classes (the cheapest pairwise margin)
      - for each runner-up, escalate the K-ceiling through `k_fraction_ceilings`
      - within each (runner_up, ceiling), iteratively refine the gradient
      - yield each best-so-far candidate so the pipeline can pick the highest

    Per (runner_up, ceiling, refinement) attempt:
      1. Compute margin gradient at current perturbation (1 fwd+bwd).
      2. Sort pixels by |gradient|.
      3. Doubling-search + bisection for smallest flipping K (forwards only).

    This costs ~1 fwd+bwd + ~log2(N) forwards per attempt; most images flip on
    the first attempt with the cheapest runner-up.
    """
    total = clean.numel()
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=max(1, num_runner_ups)
    )

    yielded_any = False
    for runner_up in runner_ups:
        delta = torch.zeros_like(clean)
        for ceiling in k_fraction_ceilings:
            max_k = max(1, int(round(total * float(ceiling))))
            attempt_found = False
            for _ in range(max(1, refine_iterations)):
                x_curr = (clean + delta).clamp(0.0, 1.0)
                grad = _margin_grad_at(
                    model=model, x=x_curr, true_idx=target_idx, runner_up_idx=runner_up,
                )
                flat_grad = grad.view(-1)
                sorted_idx = flat_grad.abs().argsort(descending=True)
                signs = -flat_grad[sorted_idx].sign()
                # Drop pixels whose perturbation would saturate after PNG roundtrip
                # (already at 1.0 with +δ, or at 0.0 with −δ). These are dead weight
                # in the binary search — they stretch K without changing the image.
                clean_uint8 = (clean.view(-1) * 255.0).round()[sorted_idx]
                effective = (
                    ((signs > 0) & (clean_uint8 <= 254.0))
                    | ((signs < 0) & (clean_uint8 >= 1.0))
                )
                sorted_idx = sorted_idx[effective]
                signs = signs[effective]
                if sorted_idx.numel() == 0:
                    break
                best_adv = _binary_search_min_k(
                    model=model,
                    clean=clean,
                    target_idx=target_idx,
                    sorted_idx=sorted_idx,
                    signs=signs,
                    magnitude=magnitude,
                    max_k=max_k,
                )
                if best_adv is None:
                    break  # this refinement direction can't flip at this ceiling
                delta = (best_adv - clean).detach()
                yield best_adv.clone()
                yielded_any = True
                attempt_found = True
            if attempt_found:
                break  # found a flip at this ceiling; no need for wider ceiling
        # Continue to next runner-up. Different runner-up classes have very
        # different pairwise margins; a later one often needs much smaller K.
        # The pipeline picks the best across all yielded candidates and can
        # break early on its own score-threshold.

    if not yielded_any:
        # Final safety net: yield the clean image (will score 0, but never crashes).
        yield clean.detach().clone()


def _quantized_sparse_projection(
    delta: torch.Tensor, magnitude: float, k: int
) -> torch.Tensor:
    """Project delta onto: at most K nonzeros, each exactly ±magnitude.
    Keeps the K positions with largest |delta|, quantizes their sign."""
    flat = delta.view(-1)
    if k <= 0:
        return torch.zeros_like(delta)
    abs_flat = flat.abs()
    k = min(k, abs_flat.numel())
    _, top_idx = abs_flat.topk(k)
    out = torch.zeros_like(flat)
    out[top_idx] = flat[top_idx].sign() * magnitude
    return out.view_as(delta)


def _project_to_image_range(
    delta: torch.Tensor, clean: torch.Tensor
) -> torch.Tensor:
    """Project so clean + delta lies in [0, 1] pixel-wise."""
    return ((clean + delta).clamp(0.0, 1.0) - clean).detach()


def _rmse_shrink(
    model: torch.nn.Module,
    clean: torch.Tensor,
    delta: torch.Tensor,
    target_idx: int,
    min_k: int = 1,
) -> torch.Tensor:
    """Given a flipping delta, try to reduce its number of nonzeros (lower
    RMSE → higher score) while preserving the post-roundtrip flip. Binary
    search on K, dropping smallest-|delta| positions first."""
    flat = delta.view(-1)
    nonzero_count = int((flat != 0).sum().item())
    if nonzero_count <= min_k:
        return delta
    abs_flat = flat.abs()
    order = abs_flat.argsort(descending=True)
    signs = flat[order]
    magnitude = float(abs_flat.max().item())

    def _build(k: int) -> torch.Tensor:
        out = torch.zeros_like(flat)
        out[order[:k]] = signs[:k]
        return _project_to_image_range(out.view_as(clean), clean)

    lo, hi = min_k, nonzero_count
    best = delta
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_delta = _build(mid)
        adv = (clean + cand_delta).clamp(0.0, 1.0)
        if _predict_idx_roundtrip(model, adv) != target_idx:
            best = cand_delta
            hi = mid - 1
        else:
            lo = mid + 1
    _ = magnitude  # unused; retained for future weighted-shrink variants
    return best


def attack_strong(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    magnitude: float = 1.0 / 255.0,
    k_fractions: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40, 0.70),
    num_runner_ups: int = 10,
    num_restarts: int = 5,
    num_iterations: int = 60,
    step_size_rel: float = 1.0,
    shrink_after_flip: bool = True,
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """Strong sparse-PGD attack at fixed magnitude (default 1/255).

    For each (k_fraction, runner_up, restart):
      1. Initialize delta (zeros on restart 0; random sparse otherwise).
      2. Run up to num_iterations of:
           grad = ∂(logit[true] − logit[runner_up])/∂x at (clean + delta)
           delta ← delta − step · sign(grad)
           project delta onto {−mag, 0, +mag}^N with at most K nonzeros
           project (clean + delta) onto [0, 1]
           if post-roundtrip prediction != target_idx → flipped!
      3. On flip: optionally shrink K (RMSE minimization), then yield.

    Strictly stronger than attack_binary_search at the same Linf budget but
    much more expensive. Use when score>0.93 is required and time isn't.
    """
    total = clean.numel()
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=max(1, num_runner_ups)
    )
    gen = torch.Generator(device=device).manual_seed(int(seed))
    step = float(magnitude) * float(step_size_rel)

    for k_frac in k_fractions:
        k = max(1, int(round(total * float(k_frac))))
        for runner_up in runner_ups:
            for restart in range(max(1, num_restarts)):
                # ----- initialize delta -----
                if restart == 0:
                    delta = torch.zeros_like(clean)
                else:
                    flat = torch.zeros(total, device=device, dtype=clean.dtype)
                    perm = torch.randperm(total, generator=gen, device=device)[:k]
                    rsigns = (
                        torch.randint(0, 2, (k,), generator=gen, device=device).to(
                            clean.dtype
                        )
                        * 2.0
                        - 1.0
                    )
                    flat[perm] = magnitude * rsigns
                    delta = _project_to_image_range(flat.view_as(clean), clean)

                # ----- PGD iterations -----
                for _ in range(max(1, num_iterations)):
                    x = (clean + delta).clamp(0.0, 1.0).detach().requires_grad_(True)
                    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
                    margin = logits[target_idx] - logits[runner_up]
                    grad = torch.autograd.grad(margin, x)[0].detach()

                    # sign-step (descent on margin)
                    delta = (delta - step * grad.sign()).detach()
                    # L∞ ball projection
                    delta = delta.clamp(-magnitude, magnitude)
                    # sparse + quantized projection (top-K by |delta|, ±magnitude)
                    delta = _quantized_sparse_projection(delta, magnitude, k)
                    # pixel-range projection
                    delta = _project_to_image_range(delta, clean)

                    adv = (clean + delta).clamp(0.0, 1.0)
                    if _predict_idx_roundtrip(model, adv) != target_idx:
                        if shrink_after_flip:
                            delta = _rmse_shrink(
                                model=model,
                                clean=clean,
                                delta=delta,
                                target_idx=target_idx,
                            )
                            adv = (clean + delta).clamp(0.0, 1.0)
                        yield adv.detach().clone()
                        break  # next (runner_up, restart)


def attack_trivial_top_k_pixels(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    magnitude: float = 1.0 / 255.0,
    target_rmse_min: float = 0.0001,
    target_rmse_max: float = 0.00015,
    flip_required: bool = False,
    max_k_fraction: float = 0.05,
    batch_check: int = 4,
) -> Iterator[torch.Tensor]:
    """Trivial top-K pixel ±magnitude perturbation. Picks the K highest-|grad|
    non-saturated pixels along the descent direction of the (true − runner_up)
    margin. Two modes, controlled by `flip_required`:

    flip_required=False  →  RMSE-band priority (validator's flip gate is OFF).
      The rmse is deterministic in K:  rmse(K) = magnitude · √(K / N_pixels).
      The function stops at K_min, the smallest integer with rmse ≥ rmse_min —
      that's the highest-score point inside the target rmse band. One
      forward+backward total. Resulting rmse sits just above rmse_min.

    flip_required=True  →  SCORE priority (validator's flip gate is ON).
      Score is 0 unless the label flips, so the rmse band is *not* enforced
      here. The function grows K from K_min upward, checking the model's
      prediction every `batch_check` additions, until either the label flips
      or K hits the hard cap `max_k_fraction · N_pixels` (default 5%). The
      rmse band is honored as a *preference* — if a flip is found inside
      [K_min, K_max_band] the rmse is also in band, but if no flip is found
      there, K keeps growing past K_max_band and rmse will exceed rmse_max.

    For a 64×64×3 image (N=12288) at magnitude 1/255, default band
    [0.0001, 0.00015]:
      - K_min = 8  (rmse ≈ 0.000100)
      - K_max_band = 17  (rmse ≈ 0.000146)
      - hard cap K_max = round(0.05 · 12288) = 614  (rmse ≈ 0.000875)

    Cost in flip_required mode: at most (K_max − K_min) / batch_check ≈ 150
    model forwards on a hard image (~7-8 s on a GPU). Easy images flip near
    K_min and cost is ~50 ms.
    """
    import math
    n_pixels = clean.numel()
    k_min = max(1, math.ceil((target_rmse_min / magnitude) ** 2 * n_pixels))
    k_max_band = max(k_min, math.floor((target_rmse_max / magnitude) ** 2 * n_pixels))
    k_hard_cap = max(k_max_band, int(round(max_k_fraction * n_pixels)))

    runner_up = _find_runner_up(model=model, clean=clean, true_idx=target_idx)
    grad = _margin_grad_at(
        model=model, x=clean, true_idx=target_idx, runner_up_idx=runner_up,
    )
    flat_grad = grad.view(-1)
    clean_flat = clean.view(-1)
    order = flat_grad.abs().argsort(descending=True)
    clean_uint8 = (clean_flat * 255.0).round()

    delta_flat = torch.zeros_like(clean_flat)
    picked = 0

    if not flip_required:
        # RMSE-band priority: pick exactly K_min non-saturated pixels.
        for idx_t in order:
            if picked >= k_min:
                break
            idx = int(idx_t.item())
            sign = -float(flat_grad[idx].sign().item())
            pv = float(clean_uint8[idx].item())
            if sign > 0 and pv >= 255.0:
                continue
            if sign < 0 and pv <= 0.0:
                continue
            delta_flat[idx] = sign * magnitude
            picked += 1
        adv = (clean + delta_flat.view_as(clean)).clamp(0.0, 1.0)
        yield adv.detach()
        return

    # flip_required=True: SCORE priority. Grow K until label flips or hard cap.
    last_check_at = 0
    batch_check = max(1, int(batch_check))
    for idx_t in order:
        if picked >= k_hard_cap:
            break
        idx = int(idx_t.item())
        sign = -float(flat_grad[idx].sign().item())
        pv = float(clean_uint8[idx].item())
        if sign > 0 and pv >= 255.0:
            continue
        if sign < 0 and pv <= 0.0:
            continue
        delta_flat[idx] = sign * magnitude
        picked += 1

        # Don't check before K_min — no point yielding rmse-too-low candidates,
        # the pipeline's filter would reject them anyway (when applicable).
        if picked < k_min:
            continue
        # Batched flip check — saves model forwards.
        if picked - last_check_at < batch_check:
            continue
        last_check_at = picked
        adv = (clean + delta_flat.view_as(clean)).clamp(0.0, 1.0)
        if _predict_idx(model, adv) != target_idx:
            yield adv.detach()
            return

    # Hit the hard cap without flipping. Yield best effort (will score 0).
    adv = (clean + delta_flat.view_as(clean)).clamp(0.0, 1.0)
    yield adv.detach()


def _backward_eliminate(
    model: torch.nn.Module,
    clean: torch.Tensor,
    delta: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    """Per-pixel backward elimination: given a flipping delta, try removing
    each perturbed position (smallest |delta| first) and keep the removal iff
    the post-roundtrip flip survives. Strictly stronger than `_rmse_shrink`'s
    binary search — that one can only drop prefixes, this one can keep pixel
    N while dropping pixel N-1.

    Cost: O(K) forward passes (K = number of nonzero positions). With K in
    the 20-200 range this is ~50-500ms on GPU."""
    flat = delta.view(-1).clone()
    nonzero = (flat != 0).nonzero(as_tuple=False).view(-1)
    if nonzero.numel() == 0:
        return (clean + delta).clamp(0.0, 1.0).detach()
    # Drop weakest perturbations first; keep strongest.
    order = flat[nonzero].abs().argsort()
    positions = nonzero[order].tolist()

    for pos in positions:
        saved = float(flat[pos].item())
        flat[pos] = 0.0
        adv = (clean + flat.view_as(clean)).clamp(0.0, 1.0)
        if _predict_idx_roundtrip(model, adv) == target_idx:
            # Dropping this position broke the flip — restore.
            flat[pos] = saved
    return (clean + flat.view_as(clean)).clamp(0.0, 1.0).detach()


def _spatial_smooth_saliency(grad: torch.Tensor, kernel: int) -> torch.Tensor:
    """Box-smooth the per-pixel |gradient| so each pixel is scored by the
    saliency of its *neighborhood*, not just itself. Pixels inside a
    high-saliency region then outrank isolated spikes — biasing selection
    toward contiguous clusters, which survive EfficientNet's 32× input
    downsampling far better than scattered single pixels. `kernel <= 1` is a
    no-op (plain |gradient|)."""
    sal = grad.abs()
    if int(kernel) <= 1:
        return sal
    pad = int(kernel) // 2
    smoothed = F.avg_pool2d(
        sal.unsqueeze(0), kernel_size=int(kernel), stride=1, padding=pad
    )
    return smoothed[0]


def attack_jsma_greedy(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    magnitude: float = 1.0 / 255.0,
    batch_pixels: int = 8,
    max_k_fraction: float = 0.05,
    yield_after_flip_batches: int = 0,
    num_runner_ups: int = 5,
    backward_eliminate: bool = True,
    cluster_kernel: int = 1,
    n_batches_cap: int = 250,
) -> Iterator[torch.Tensor]:
    """JSMA-style greedy pixel growth across multiple runner-up classes.

    For each runner-up class, repeatedly:
      1. Compute the gradient of the (true − runner_up) margin at the *current*
         perturbed image.
      2. Pick the top-`batch_pixels` unused positions by saliency.
      3. Set delta at those positions to ±magnitude in the descent direction
         (the sign that decreases margin toward zero / past it).
      4. Mark them used. Check if label flipped under PNG roundtrip; if so,
         optionally run per-pixel backward elimination, then yield.

    `cluster_kernel` makes step 2 cluster-aware: with kernel > 1 the selection
    saliency is the box-smoothed |gradient| (a pixel scored by its
    neighborhood), so the chosen pixels form contiguous patches rather than
    scattered spikes. EfficientNet downsamples its 480² input 32× — clustered
    perturbations survive that pooling far better, so this can flip with a
    smaller K. The perturbation *sign* still comes from the true gradient.
    `cluster_kernel = 1` keeps the original scattered behavior.

    Unlike `attack_sparse_sweep` (which probes fixed K values), this grows the
    active set one batch at a time and stops at the actual minimum K that
    flips the label — typically much smaller K → much lower RMSE → higher score.
    Trying multiple runner-ups matters: different target classes need wildly
    different K to flip; the cheapest one wins.
    """
    runner_ups = _top_runner_ups(
        model=model, clean=clean, true_idx=target_idx, n=max(1, num_runner_ups)
    )
    total = clean.numel()
    max_k = max(1, int(round(total * float(max_k_fraction))))
    batch_pixels = max(1, int(batch_pixels))
    n_batches = max(1, max_k // batch_pixels)
    # Bound the iteration count. On a large image max_k is huge and a fixed
    # small batch_pixels makes n_batches explode (a 350×525 image → ~3400
    # batches per runner-up → minutes). Cap n_batches and grow batch_pixels
    # to still cover max_k, so cost is ~image-size-independent.
    if n_batches > n_batches_cap:
        n_batches = n_batches_cap
        batch_pixels = max(1, max_k // n_batches)

    yielded_any = False
    last_delta = torch.zeros_like(clean)
    for runner_up in runner_ups:
        delta = torch.zeros_like(clean)
        used = torch.zeros(total, dtype=torch.bool, device=clean.device)
        flipped_at = -1

        for batch_i in range(n_batches):
            x_curr = (clean + delta).clamp(0.0, 1.0)
            grad = _margin_grad_at(
                model=model, x=x_curr, true_idx=target_idx, runner_up_idx=runner_up,
            )
            flat_grad = grad.view(-1)
            n_avail = int((~used).sum().item())
            if n_avail <= 0:
                break
            # Selection saliency: cluster-aware (box-smoothed) when
            # cluster_kernel > 1, plain |gradient| otherwise.
            saliency = _spatial_smooth_saliency(grad, cluster_kernel).view(-1)
            scores = saliency.masked_fill(used, float("-inf"))
            k_this = min(batch_pixels, n_avail)
            idx = scores.topk(k_this).indices
            signs = -flat_grad[idx].sign()  # sign from the TRUE gradient
            new_vals = (magnitude * signs).to(delta.dtype)
            delta.view(-1).index_copy_(0, idx, new_vals)
            used[idx] = True

            adv = (clean + delta).clamp(0.0, 1.0).detach()
            # Recover any slot that got zeroed out by [0,1] clamp.
            eff_delta = (adv - clean).detach()
            zeroed = (eff_delta.view(-1) == 0) & used
            used[zeroed] = False
            delta = eff_delta

            # Use the cheap float-space check here (no PIL roundtrip per
            # iteration). The pipeline does a PIL roundtrip + score check on
            # every yielded candidate, so non-roundtrip-stable flips just get
            # score=0 and are filtered out — the best valid yield still wins.
            if _predict_idx(model, adv) != target_idx:
                if backward_eliminate:
                    adv = _backward_eliminate(
                        model=model, clean=clean, delta=delta, target_idx=target_idx,
                    )
                    delta = (adv - clean).detach()
                yield adv.clone()
                yielded_any = True
                if flipped_at < 0:
                    flipped_at = batch_i
                if batch_i - flipped_at >= yield_after_flip_batches:
                    break  # move to the next runner-up

        last_delta = delta

    if not yielded_any:
        # Safety net: yield the last attempt (likely scores 0, but never crashes).
        yield (clean + last_delta).clamp(0.0, 1.0).detach()


def _refine_sparse_mask(
    model: torch.nn.Module,
    clean: torch.Tensor,
    true_idx: int,
    runner_up_idx: int,
    k: int,
    magnitude: float,
    refine_steps: int,
    persist_after_flip: int,
) -> torch.Tensor:
    """Iteratively pick top-K |margin-grad| positions and set delta there.
    Returns a candidate (clean + delta). After the first iteration that flips,
    runs `persist_after_flip` extra iterations to let the sign pattern settle
    onto the best flipping configuration."""
    delta = torch.zeros_like(clean)
    flipped_for = 0
    for _ in range(max(1, refine_steps)):
        x_curr = (clean + delta).clamp(0.0, 1.0)
        grad = _margin_grad_at(model=model, x=x_curr, true_idx=true_idx, runner_up_idx=runner_up_idx)
        cand = _sparse_topk_oneshot(
            grad=grad, clean=clean, k=k, magnitude=magnitude, descent=True,
        )
        delta = (cand - clean).detach()
        if _predict_idx(model, (clean + delta).clamp(0.0, 1.0)) != true_idx:
            flipped_for += 1
            if flipped_for >= persist_after_flip:
                break
    return (clean + delta).clamp(0.0, 1.0).detach()


def attack_sparse_sweep(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    k_fractions: Iterable[float] = (
        0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.012, 0.016, 0.02,
    ),
    magnitudes: Iterable[float] = (1.0 / 255.0,),
    refine_steps: int = 20,
    persist_after_flip: int = 3,
    num_runner_ups: int = 1,
) -> Iterator[torch.Tensor]:
    """Targeted sparse-sweep with iterative refinement.

    For each (K, magnitude) combo:
      - Repeatedly recompute the gradient of the (true − runner_up) margin at
        the *current* perturbed image, repick the top-K positions by |grad|,
        and REPLACE delta with ±magnitude·sign(grad) at those positions.
      - Replacing (not accumulating) keeps the active set bounded at K, so
        RMSE stays small while signs adapt to the local loss surface.
      - Run up to `refine_steps` iterations but exit early once the label has
        flipped for `persist_after_flip` consecutive iterations (sign pattern
        has settled).

    Smaller K → lower RMSE → higher score, but the label must still flip.
    """
    total = clean.numel()
    runner_ups = _top_runner_ups(model=model, clean=clean, true_idx=target_idx, n=max(1, num_runner_ups))

    for mag in magnitudes:
        for frac in k_fractions:
            k = max(1, int(round(total * float(frac))))
            for runner_up in runner_ups:
                yield _refine_sparse_mask(
                    model=model,
                    clean=clean,
                    true_idx=target_idx,
                    runner_up_idx=runner_up,
                    k=k,
                    magnitude=float(mag),
                    refine_steps=refine_steps,
                    persist_after_flip=persist_after_flip,
                )


def _sigma_zero_run(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    magnitude: float,
    n_iterations: int,
    init_u: torch.Tensor,
    sigma: float = 1e-3,
    eta0: float = 1.0,
    tau0: float = 0.3,
    targeted_idx: int | None = None,
    deadline: float | None = None,
) -> tuple[torch.Tensor | None, int, torch.Tensor, float]:
    """One σ-zero run (Pintor et al., ICLR 2025) adapted to a fixed per-pixel
    L∞. Optimizes a normalized perturbation u ∈ [−1,1]^d (real perturbation =
    u·magnitude); in these units σ-zero's published constants (σ=1e-3, η0=1,
    τ0=0.3, τ-step 0.01, cosine-annealed η) apply unchanged.

    Untargeted (default, `targeted_idx=None`) minimizes
        loss = relu(f_y − max_{k≠y} f_k) + (1/d) · Σ u_i²/(u_i²+σ)
    — flip to *any* other class. Targeted (`targeted_idx=t`) minimizes
        loss = relu(f_y − f_t) + (1/d) · Σ u_i²/(u_i²+σ)
    — push class `t` over `y`. Useful when the natural runner-up is hard to
    flip to but a different class is cheaper.

    The L∞-normalized gradient steps u; u is clamped to [−1,1] and the image
    box, then components with |u_i| < τ are projected to 0. τ grows when the
    soft perturbation flips, shrinks otherwise.

    Returns (best_adv, best_k, near_miss_u, near_miss_margin):
      - best_adv : sparsest quantized flipping candidate, or None
      - best_k   : its non-zero count (d+1 if no flip)
      - near_miss_u : the post-projection u at the smallest-margin iteration
      - near_miss_margin : that smallest margin (used to seed the fallback)."""
    import math

    d = clean.numel()
    clean_flat = clean.view(-1)
    u = init_u.detach().clone()
    # The validator scores "flip" by comparing *normalized label strings*, not
    # indices. ImageNet has a duplicate-normalized-label pair at idx 264/474
    # ('cardigan' corgi vs sweater); without this string baseline we'd accept
    # a 264→474 transition as a flip even though the validator would score 0.
    target_label_norm = _label_norm(target_idx)
    best_adv: torch.Tensor | None = None
    best_k = d + 1
    near_miss_u = u.clone()
    near_miss_margin = float("inf")
    tau = float(tau0)
    eta = float(eta0)

    for i in range(1, int(n_iterations) + 1):
        if deadline is not None and time.time() >= deadline:
            break
        u_leaf = u.detach().clone().requires_grad_(True)
        x_adv = (clean + (u_leaf * magnitude).view_as(clean)).clamp(0.0, 1.0)
        logits = logits_for_images(model=model, image_bchw=x_adv.unsqueeze(0))[0]
        if targeted_idx is not None:
            margin = logits[target_idx] - logits[targeted_idx]
        else:
            other_mask = torch.zeros_like(logits)
            other_mask[target_idx] = float("-inf")
            margin = logits[target_idx] - (logits + other_mask).max()
        l_class = torch.clamp(margin, min=0.0)
        soft_l0 = (u_leaf * u_leaf / (u_leaf * u_leaf + sigma)).sum()
        loss = l_class + soft_l0 / d
        grad = torch.autograd.grad(loss, u_leaf)[0].detach()

        inf_norm = grad.abs().max()
        if inf_norm > 0:
            grad = grad / inf_norm

        u = (u - eta * grad).clamp(-1.0, 1.0)
        u = ((clean_flat + u * magnitude).clamp(0.0, 1.0) - clean_flat) / magnitude
        u = torch.where(u.abs() >= tau, u, torch.zeros_like(u))

        eta = float(eta0) * (1.0 + math.cos(math.pi * i / max(1, n_iterations))) / 2.0

        m = float(margin.item())
        if m < near_miss_margin:
            near_miss_margin = m
            near_miss_u = u.clone()

        if m < 0.0:
            tau = min(1.0, tau + 0.01 * eta)
            quant = torch.where(u != 0, u.sign() * magnitude, torch.zeros_like(u))
            adv_q = (clean_flat + quant).clamp(0.0, 1.0).view_as(clean)
            # Float-space flip is the cheap gate; only a candidate that also
            # survives the PNG roundtrip (the validator's actual view) and is
            # sparser than the current best is adopted.
            if _predict_label_norm(model, adv_q) != target_label_norm:
                k = int((quant != 0).sum().item())
                if k < best_k and _predict_label_norm_roundtrip(model, adv_q) != target_label_norm:
                    best_k = k
                    best_adv = adv_q.detach().clone()
        else:
            tau = max(0.0, tau - 0.01 * eta)

    return best_adv, best_k, near_miss_u, near_miss_margin


def _sigma_zero_batched(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    magnitude: float,
    n_iterations: int,
    init_u_batch: torch.Tensor,
    targeted_idx_batch: list[int | None] | None = None,
    sigma: float = 1e-3,
    eta0: float = 1.0,
    tau0: float = 0.3,
    deadline: float | None = None,
) -> tuple[torch.Tensor | None, int]:
    """Batched σ-zero: runs `B = init_u_batch.shape[0]` independent attacks in
    a single batched forward pass, sharing the model's GPU compute. Each row
    has its own delta, per-row threshold τ, and optionally a different
    targeted class.

    Same per-attack algorithm as `_sigma_zero_run` (Pintor et al., ICLR 2025)
    — soft-L0 penalty + margin descent + adaptive threshold — but the forward
    pass costs roughly 1.3-1.5× a single sample instead of B× separate calls
    on V2-class models. Net wall-time win for multi-restart / multi-target
    setups.

    `targeted_idx_batch[b] = t` runs row `b` as a targeted attack flipping
    `target_idx → t`; `None` runs row `b` untargeted (any non-true class).
    `targeted_idx_batch=None` defaults to all-untargeted.

    Returns (best_adv, best_k) — the sparsest *post-PNG-roundtrip* flipping
    candidate across all B rows."""
    import math

    B, d = init_u_batch.shape
    assert d == clean.numel(), f"init_u_batch dim mismatch: {d} vs {clean.numel()}"
    if targeted_idx_batch is None:
        targeted_idx_batch = [None] * B
    assert len(targeted_idx_batch) == B

    # Validator-matching flip check (see `_label_norm` docstring).
    target_label_norm = _label_norm(target_idx)
    device = init_u_batch.device
    clean_flat = clean.view(-1)
    u = init_u_batch.detach().clone()  # (B, d)

    best_adv: torch.Tensor | None = None
    best_k = clean.numel() + 1

    # Per-row threshold (tau adapts independently per row).
    tau = torch.full((B,), float(tau0), device=device)
    eta = float(eta0)

    for i in range(1, int(n_iterations) + 1):
        if deadline is not None and time.time() >= deadline:
            break

        u_leaf = u.detach().clone().requires_grad_(True)
        # delta: (B, d) → (B, C, H, W) → broadcast with clean: (1, C, H, W).
        delta = (u_leaf * magnitude).view(B, *clean.shape)
        x_adv = (clean.unsqueeze(0) + delta).clamp(0.0, 1.0)  # (B, C, H, W)
        logits = logits_for_images(model=model, image_bchw=x_adv)  # (B, num_classes)

        # Per-row margin: logit[true] − {logit[target_b] if targeted else max-other}.
        margins = []
        for b in range(B):
            t_b = targeted_idx_batch[b]
            if t_b is not None:
                margins.append(logits[b, target_idx] - logits[b, t_b])
            else:
                masked = logits[b].clone()
                masked[target_idx] = float("-inf")
                margins.append(logits[b, target_idx] - masked.max())
        margins_t = torch.stack(margins)  # (B,)

        l_class = torch.clamp(margins_t, min=0.0)  # (B,)
        soft_l0 = (u_leaf * u_leaf / (u_leaf * u_leaf + sigma)).sum(dim=1)  # (B,)
        loss = (l_class + soft_l0 / d).sum()  # scalar (one backward call)

        grad = torch.autograd.grad(loss, u_leaf)[0].detach()  # (B, d)

        # L∞-normalize each row independently — so each row's step is in [-eta, eta].
        inf_norm = grad.abs().max(dim=1, keepdim=True).values  # (B, 1)
        grad = grad / inf_norm.clamp(min=1e-12)

        u = (u - eta * grad).clamp(-1.0, 1.0)
        # Box constraint per row: keep `clean + u*mag` in [0,1].
        delta_real = u * magnitude  # (B, d)
        u = (
            (clean_flat.unsqueeze(0) + delta_real).clamp(0.0, 1.0)
            - clean_flat.unsqueeze(0)
        ) / magnitude
        # Threshold projection per row.
        u = torch.where(u.abs() >= tau.unsqueeze(1), u, torch.zeros_like(u))

        eta = float(eta0) * (1.0 + math.cos(math.pi * i / max(1, n_iterations))) / 2.0

        # Per-row τ update: grow when row flipped (sparser), shrink when not.
        with torch.no_grad():
            flipped_rows = (margins_t < 0.0)
            tau_delta = torch.where(
                flipped_rows,
                torch.full_like(tau, 0.01 * eta),
                torch.full_like(tau, -0.01 * eta),
            )
            tau = (tau + tau_delta).clamp(0.0, 1.0)

        # Per-row best-K tracking. Only check rows that flipped in float space
        # and whose quantized K could beat the current best.
        flipped_idx = flipped_rows.nonzero(as_tuple=False).view(-1).tolist()
        for b in flipped_idx:
            quant_b = torch.where(
                u[b] != 0, u[b].sign() * magnitude, torch.zeros_like(u[b]),
            )
            k = int((quant_b != 0).sum().item())
            if k >= best_k:
                continue
            adv_q = (clean_flat + quant_b).clamp(0.0, 1.0).view_as(clean)
            if _predict_label_norm(model, adv_q) != target_label_norm:
                if _predict_label_norm_roundtrip(model, adv_q) != target_label_norm:
                    best_k = k
                    best_adv = adv_q.detach().clone()

    return best_adv, best_k


def _sparse_rs_run(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    magnitude: float,
    n_queries: int,
    seed_delta: torch.Tensor,
    rng_seed: int = 0,
    max_swap: int = 3,
) -> tuple[torch.Tensor | None, int]:
    """Gradient-free random local search (Sparse-RS style) for a sparse
    ±magnitude flip, seeded from `seed_delta` (the quantized σ-zero near-miss).

    Each query randomly mutates the perturbed-pixel set — adds/swaps pixels
    while not flipped, drops pixels once flipped — and keeps the mutation iff
    it reduces the margin (or preserves the flip with fewer pixels). Uses only
    forward passes, no gradients, so it explores the discrete perturbation
    space directly and can land flips a gradient relaxation missed.

    Returns (best_adv, best_k)."""
    clean_flat = clean.view(-1)
    d = clean.numel()
    gen = torch.Generator(device=clean.device).manual_seed(int(rng_seed))

    def _margin(adv_flat: torch.Tensor) -> float:
        with torch.no_grad():
            logits = logits_for_images(
                model=model, image_bchw=adv_flat.view_as(clean).unsqueeze(0)
            )[0]
        masked = logits.clone()
        masked[target_idx] = float("-inf")
        return float((logits[target_idx] - masked.max()).item())

    def _randperm(n: int) -> torch.Tensor:
        return torch.randperm(n, generator=gen, device=clean.device)

    delta = seed_delta.detach().view(-1).clone()
    cur_adv = (clean_flat + delta).clamp(0.0, 1.0)
    cur_margin = _margin(cur_adv)

    best_adv: torch.Tensor | None = None
    best_k = d + 1

    def _record(adv_flat: torch.Tensor) -> None:
        nonlocal best_adv, best_k
        k = int(((adv_flat - clean_flat).abs() > 1e-9).sum().item())
        if k < best_k:
            best_k = k
            best_adv = adv_flat.view_as(clean).detach().clone()

    if cur_margin < 0.0:
        _record(cur_adv)

    for _ in range(int(n_queries)):
        cand = delta.clone()
        active = (cand != 0).nonzero(as_tuple=False).view(-1)
        inactive = (cand == 0).nonzero(as_tuple=False).view(-1)
        n = 1 + int(torch.randint(
            0, max(1, max_swap), (1,), generator=gen, device=clean.device
        ).item())

        if cur_margin >= 0.0:
            # Not flipped — add pixels (random sign), occasionally swap some out.
            if inactive.numel() > 0:
                add = inactive[_randperm(inactive.numel())[:n]]
                signs = torch.randint(
                    0, 2, (add.numel(),), generator=gen, device=clean.device
                ).to(clean.dtype) * 2.0 - 1.0
                cand[add] = signs * magnitude
            if active.numel() > n and float(
                torch.rand(1, generator=gen, device=clean.device).item()
            ) < 0.3:
                cand[active[_randperm(active.numel())[:n]]] = 0.0
        else:
            # Flipped — drop one random active pixel, and 25% of the time also
            # add one elsewhere (a swap). The swap keeps K unchanged but moves
            # the support, which escapes the greedy local minimum where no
            # single pixel can be dropped on its own.
            if active.numel() > 1:
                cand[active[_randperm(active.numel())[:1]]] = 0.0
            swap = float(
                torch.rand(1, generator=gen, device=clean.device).item()
            ) < 0.25
            if swap and inactive.numel() > 0:
                add = inactive[_randperm(inactive.numel())[:1]]
                sgn = torch.randint(
                    0, 2, (1,), generator=gen, device=clean.device
                ).to(clean.dtype) * 2.0 - 1.0
                cand[add] = sgn * magnitude

        cand_adv = (clean_flat + cand).clamp(0.0, 1.0)
        cand_margin = _margin(cand_adv)

        accept = (
            cand_margin < cur_margin if cur_margin >= 0.0 else cand_margin < 0.0
        )
        if accept:
            delta = cand_adv - clean_flat
            cur_margin = cand_margin
            if cur_margin < 0.0:
                _record(cand_adv)

    return best_adv, best_k


def _boost_margin_on_mask(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    target_idx: int,
    magnitude: float,
    n_iterations: int = 12,
    deadline: float | None = None,
) -> torch.Tensor:
    """Sign-PGD restricted to the current non-zero support — preserves K,
    grows the (runner_up − true) margin so a subsequent `_shrink_support`
    pass can drop more pixels.

    The pixel set is frozen (no add/drop); only the ±magnitude *sign* on each
    surviving cell can change. Each iteration recomputes the margin gradient
    at the current adv and assigns each masked pixel the sign that locally
    minimizes (true − runner_up) — i.e. flips signs that were "wrong" for
    pushing the model away from the true class. Periodically re-picks the
    runner-up since the closest competitor can shift during sign-flips.

    Returns the constellation (same support, possibly different signs) that
    achieved the deepest margin during iteration. Caller is responsible for
    re-checking the PNG-roundtrip flip — sign-flips can rarely degrade the
    flip post-quantization, but the helper never narrows the support."""
    clean_flat = clean.view(-1)
    delta = (adv - clean).view(-1).detach().clone()
    mask = (delta != 0)
    if not mask.any():
        return adv.detach()

    runner_up = _find_runner_up(model=model, clean=adv, true_idx=target_idx)
    best_adv = adv.detach().clone()
    best_margin = -float("inf")  # margin = logit[runner_up] − logit[target]

    def _margin_at(x: torch.Tensor) -> float:
        with torch.no_grad():
            logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
        masked_logits = logits.clone()
        masked_logits[target_idx] = float("-inf")
        return float((masked_logits.max() - logits[target_idx]).item())

    cur_margin = _margin_at(adv)
    if cur_margin > best_margin:
        best_margin = cur_margin
        best_adv = adv.detach().clone()

    for it in range(int(n_iterations)):
        if deadline is not None and time.time() >= deadline:
            break
        cur = (clean_flat + delta).clamp(0.0, 1.0).view_as(clean)
        if it % 4 == 0:
            runner_up = _find_runner_up(
                model=model, clean=cur, true_idx=target_idx,
            )
        grad = _margin_grad_at(
            model=model, x=cur, true_idx=target_idx, runner_up_idx=runner_up,
        ).view(-1)
        # Step is sign(-grad) (descend the margin); restrict to mask; project
        # to ±magnitude so K stays exactly the same.
        new_signs = (-grad).sign()
        new_signs[new_signs == 0] = delta[new_signs == 0].sign()
        delta = torch.where(
            mask, magnitude * new_signs, torch.zeros_like(delta),
        )
        new_adv = (clean_flat + delta).clamp(0.0, 1.0).view_as(clean)
        m = _margin_at(new_adv)
        if m > best_margin:
            best_margin = m
            best_adv = new_adv.detach().clone()

    return best_adv


def _shrink_support(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    target_idx: int,
    magnitude: float,
    deadline: float | None = None,
) -> torch.Tensor:
    """Minimize K of a flipping adversarial image — drop perturbed pixels
    while the post-PNG-roundtrip flip survives.

      1. Rank the perturbed pixels by |margin gradient| ascending — the
         least-sensitive pixels first.
      2. Binary-search the largest prefix of that ranking that can be dropped
         together while keeping the flip. Dropping more pixels only weakens
         the perturbation, so feasibility is monotone — O(log K) forwards.
      3. Per-pixel cleanup pass over the survivors — O(K_survivors) forwards.

    `deadline` is an epoch-seconds cutoff (e.g. `time.time() + budget`). Each
    phase checks it and bails early returning the best `delta` found so far —
    we only ever zero out pixels after confirming the flip survives, so any
    partial result is still a valid flipping adversarial.

    Returns the shrunk adv (still flipping, K reduced or unchanged)."""
    clean_flat = clean.view(-1)
    delta = (adv - clean).view(-1).clone()
    support = (delta != 0).nonzero(as_tuple=False).view(-1)
    if support.numel() == 0:
        return adv.detach()

    # Validator-matching flip check (see `_label_norm` docstring).
    target_label_norm = _label_norm(target_idx)

    def _expired() -> bool:
        return deadline is not None and time.time() >= deadline

    def _current() -> torch.Tensor:
        return (clean_flat + delta).clamp(0.0, 1.0).view_as(clean).detach()

    # 1. importance ranking via the (target − runner-up) margin gradient.
    if _expired():
        return _current()
    runner_up = _find_runner_up(model=model, clean=adv, true_idx=target_idx)
    grad = _margin_grad_at(
        model=model, x=adv, true_idx=target_idx, runner_up_idx=runner_up,
    ).view(-1)
    ranked = support[grad[support].abs().argsort()]  # least sensitive first

    def _adv_dropping(dropped: torch.Tensor) -> torch.Tensor:
        d2 = delta.clone()
        d2[dropped] = 0.0
        return (clean_flat + d2).clamp(0.0, 1.0).view_as(clean)

    # 2. binary search the largest droppable prefix.
    lo, hi, best_m = 0, int(support.numel()), 0
    while lo <= hi:
        if _expired():
            break
        mid = (lo + hi) // 2
        if _predict_label_norm_roundtrip(model, _adv_dropping(ranked[:mid])) != target_label_norm:
            best_m = mid
            lo = mid + 1
        else:
            hi = mid - 1
    delta[ranked[:best_m]] = 0.0

    # 3. per-pixel cleanup over remaining pixels (least-sensitive first).
    if _expired():
        return _current()
    survivors = (delta != 0).nonzero(as_tuple=False).view(-1)
    survivors = survivors[grad[survivors].abs().argsort()]
    for pos in survivors.tolist():
        if _expired():
            break
        saved = float(delta[pos].item())
        delta[pos] = 0.0
        adv_try = (clean_flat + delta).clamp(0.0, 1.0).view_as(clean)
        if _predict_label_norm_roundtrip(model, adv_try) == target_label_norm:
            delta[pos] = saved  # this pixel is load-bearing — keep it

    return _current()


def attack_sigma_zero(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    magnitude: float = 1.0 / 255.0,
    n_iterations: int = 200,
    n_restarts: int = 1,
    num_targets: int = 1,
    n_iterations_targeted: int | None = None,
    deadline: float | None = None,
    target_mode: str = "natural",
    use_batched: bool = False,
) -> Iterator[torch.Tensor]:
    """σ-zero (Pintor et al., ICLR 2025) tier wrapper with multi-target.

    Phase 1 — untargeted from zero init (paper's default setup). Flips to
    whichever class the model is naturally closest to.

    Phase 2 — for each of the next (num_targets − 1) runner-ups, run σ-zero
    targeted at that *specific* class. On hard images the natural runner-up
    is sometimes structurally expensive to flip to (e.g. guinea pig →
    hamster needs ~3500 pixels) while a different close class is far
    cheaper (e.g. guinea pig → squirrel). Targeted runs let us discover
    that without touching the untargeted flip.

    Phase 3 — additional zero/random-init restarts of untargeted (only when
    `n_restarts > 1`).

    Each successful flip is yielded as it lands; the pipeline keeps the
    smallest-K candidate. `deadline` (epoch seconds) gates the inter-run
    loop so a long tail of expensive targeted runs can't blow past the
    budget.

    Empirically (this code-base, EfficientNetV2-M @ 480², N=200 iterations):
      - lacewing: zero-init lands K≈87 in ~4.7s; the in-house tier ladder
        (sparse_sweep + jsma + binary_search) followed by polish only
        reaches K≈139. Standalone σ-zero from zero init beats the whole
        ladder + polish chain on easy images.
      - fiddler crab / guinea pig: σ-zero plateaus at K≈400 / K≈3500
        respectively when flipping to the natural runner-up. Multi-target
        is the lever for these — at least one alternate class is often
        much cheaper to reach."""
    d = clean.numel()
    n_iter_t = (
        int(n_iterations_targeted)
        if n_iterations_targeted is not None
        else int(n_iterations)
    )

    def _expired() -> bool:
        return deadline is not None and time.time() >= deadline

    # All phases run *before* the first yield. The pipeline's outer loop
    # narrows its deadline (`_tier_phase_deadline()`) the moment `best_score
    # > 0`, and that check fires between yields — so if we yielded the
    # untargeted flip first, the loop would break before any targeted run
    # got a turn. Buffering candidates means the outer loop only sees the
    # final selection.
    candidates: list[torch.Tensor] = []

    # Batched fast path. Stack untargeted + targeted runs into a single
    # batched forward pass via `_sigma_zero_batched`. On V2-L a B=3 batched
    # forward is ~1.4× a B=1 forward (vs 3× for serial), so we get
    # multi-target essentially for free in wall time.
    if use_batched and target_mode == "natural" and n_restarts == 1:
        try:
            runner_ups = _top_runner_ups(
                model=model, clean=clean, true_idx=target_idx,
                n=max(1, int(num_targets)),
            )
        except Exception:
            runner_ups = []
        # Row 0: untargeted. Rows 1..: targeted at runner-ups[1..].
        targets: list[int | None] = [None]
        for rup in list(runner_ups)[1:]:
            targets.append(int(rup))
        B = len(targets)
        init_u_batch = torch.zeros((B, d), device=device)
        adv_b, _k = _sigma_zero_batched(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=magnitude, n_iterations=int(n_iterations),
            init_u_batch=init_u_batch,
            targeted_idx_batch=targets,
            deadline=deadline,
        )
        if adv_b is not None:
            yield adv_b
        return  # batched branch handles all configured runs at once

    # Pick which runner-ups to target. Three modes:
    #   "natural"   — untargeted first, then targeted at the rest (current
    #                 default; matches the previous behavior).
    #   "cheapest"  — rank top runner-ups by `_rank_targets_by_cost` and
    #                 attack ONLY the cheapest with the full N=`n_iterations`
    #                 budget. ~150ms of diagnostic forwards/backwards, then
    #                 a single concentrated attack.
    #   "skip-natural" — skip the natural runner-up (#1) entirely and
    #                 target only #2..#num_targets. Useful on hard images
    #                 where the natural target is structurally expensive.
    mode = str(target_mode).lower()

    if mode == "cheapest":
        ranked = _rank_targets_by_cost(
            model=model, clean=clean, true_idx=target_idx,
            num_candidates=max(2, int(num_targets)),
        )
        if not _expired() and ranked:
            adv_t, _k, _nm_u, _nm_m = _sigma_zero_run(
                model=model, clean=clean, target_idx=target_idx,
                magnitude=magnitude, n_iterations=int(n_iterations),
                init_u=torch.zeros(d, device=device),
                targeted_idx=int(ranked[0]),
                deadline=deadline,
            )
            if adv_t is not None:
                candidates.append(adv_t)
    else:
        # Phase 1 — untargeted from zero init.
        if mode != "skip-natural" and not _expired():
            adv, _k, _nm_u, _nm_m = _sigma_zero_run(
                model=model, clean=clean, target_idx=target_idx,
                magnitude=magnitude, n_iterations=int(n_iterations),
                init_u=torch.zeros(d, device=device),
                deadline=deadline,
            )
            if adv is not None:
                candidates.append(adv)

        # Phase 2 — targeted at the next runner-ups.
        if num_targets > 1 and not _expired():
            try:
                runner_ups = _top_runner_ups(
                    model=model, clean=clean, true_idx=target_idx,
                    n=int(num_targets),
                )
            except Exception:
                runner_ups = []
            # In "natural" mode we already tried #1 untargeted; skip it
            # here. In "skip-natural" mode we target *every* runner-up,
            # including #1.
            start_idx = 1 if mode == "natural" else 0
            for rup in list(runner_ups)[start_idx:]:
                if _expired():
                    break
                adv_t, _k, _nm_u, _nm_m = _sigma_zero_run(
                    model=model, clean=clean, target_idx=target_idx,
                    magnitude=magnitude, n_iterations=n_iter_t,
                    init_u=torch.zeros(d, device=device),
                    targeted_idx=int(rup),
                    deadline=deadline,
                )
                if adv_t is not None:
                    candidates.append(adv_t)

    # Phase 3 — additional restarts (random init), untargeted.
    for restart in range(1, int(n_restarts)):
        if _expired():
            break
        gen = torch.Generator(device=device).manual_seed(restart)
        init_u = (
            (torch.rand(d, generator=gen, device=device) * 2.0 - 1.0) * 0.5
        )
        adv, _k, _nm_u, _nm_m = _sigma_zero_run(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=magnitude, n_iterations=int(n_iterations),
            init_u=init_u,
            deadline=deadline,
        )
        if adv is not None:
            candidates.append(adv)

    # Yield ascending by K so the pipeline scores the best candidate first.
    clean_flat = clean.view(-1)

    def _k(adv_tensor: torch.Tensor) -> int:
        return int(((adv_tensor.view(-1) - clean_flat).abs() > 1e-9).sum().item())

    candidates.sort(key=_k)
    for c in candidates:
        yield c


def attack_adaptive(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_idx: int,
    device: torch.device,
    magnitude: float = 1.0 / 255.0,
    n_restarts: int = 4,
    n_iterations: int = 200,
    shrink_rounds: int = 3,
    sparse_rs_queries: int = 400,
) -> Iterator[torch.Tensor]:
    """Maximum-effort K-minimization at fixed 1/255 — no magnitude escalation.

    The score, with a 100% flip rate, is a monotone function of K (perturbed
    pixel count) alone: fewer pixels → lower RMSE → higher score. So this is a
    pure K-minimization grinder.

    Phase 1 — portfolio. Collect a flipping perturbation from a diverse set of
      attacks: σ-zero with restarts (gradient/continuous) and JSMA-greedy
      (gradient/greedy). If none flips, fall back to a gradient-free Sparse-RS
      search seeded from σ-zero's closest near-miss. Keep the lowest-K flip.

    Phase 2 — iterated shrink. Relentlessly reduce K of the best flip:
      (a) `_shrink_support` — gradient-ordered backward elimination.
      (b) re-optimize — re-run σ-zero seeded from the pruned support; a fresh
          descent from there often settles on a strictly smaller one.
      (c) Sparse-RS swap-polish — gradient-free drop/swap moves that escape
          the greedy local minimum.
    Repeat until K stops improving. Every improvement is yielded so the
    pipeline always holds the best."""
    d = clean.numel()
    clean_flat = clean.view(-1)
    gen = torch.Generator(device=device).manual_seed(0)

    best_adv: torch.Tensor | None = None
    best_k = d + 1

    def _k(adv: torch.Tensor) -> int:
        return int(((adv - clean).abs() > 1e-9).sum().item())

    def _consider(adv: torch.Tensor | None) -> bool:
        """Adopt `adv` as the new best iff it flips (post-roundtrip) with
        strictly fewer pixels. Returns True on improvement."""
        nonlocal best_adv, best_k
        if adv is None:
            return False
        if _predict_idx_roundtrip(model, adv) == target_idx:
            return False  # not a real flip
        k = _k(adv)
        if k < best_k:
            best_k = k
            best_adv = adv.detach().clone()
            return True
        return False

    logger.info(
        f"[adaptive] start: image_pixels={d} (≈{d // 3}px) "
        f"restarts={n_restarts} iters={n_iterations}"
    )

    # ── Phase 1: portfolio ──────────────────────────────────────────
    seed_u = torch.zeros(d, device=device, dtype=clean.dtype)
    seed_margin = float("inf")
    sigma_flips = 0

    for restart in range(max(1, n_restarts)):
        if restart == 0:
            init_u = torch.zeros(d, device=device, dtype=clean.dtype)
        else:
            init_u = torch.empty(
                d, device=device, dtype=clean.dtype
            ).uniform_(-1.0, 1.0, generator=gen) * 0.5
        adv_r, _k_r, near_miss_u, near_miss_margin = _sigma_zero_run(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=magnitude, n_iterations=n_iterations, init_u=init_u,
        )
        if adv_r is not None:
            sigma_flips += 1
        if _consider(adv_r):
            yield best_adv
        if near_miss_margin < seed_margin:
            seed_margin = near_miss_margin
            seed_u = near_miss_u
    logger.info(
        f"[adaptive] sigma-zero: {sigma_flips}/{n_restarts} restarts flipped, "
        f"best_k={best_k if best_adv is not None else 'none'} "
        f"closest_near_miss_margin={seed_margin:.5f}"
    )

    jsma_before = best_k
    for adv_j in attack_jsma_greedy(
        model=model, clean=clean, target_idx=target_idx, device=device,
        magnitude=magnitude, num_runner_ups=5, backward_eliminate=False,
    ):
        if _consider(adv_j):
            yield best_adv
    logger.info(
        f"[adaptive] jsma: best_k="
        f"{best_k if best_adv is not None else 'none'} "
        f"(improved={best_adv is not None and best_k < jsma_before})"
    )

    if best_adv is None:
        # No gradient attack flipped — gradient-free search from the near-miss.
        seed_delta = torch.where(
            seed_u != 0, seed_u.sign() * magnitude, torch.zeros_like(seed_u)
        )
        rs_adv, _rs_k = _sparse_rs_run(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=magnitude, n_queries=sparse_rs_queries,
            seed_delta=seed_delta,
        )
        if _consider(rs_adv):
            yield best_adv
        logger.info(
            f"[adaptive] sparse_rs fallback: "
            f"flipped={best_adv is not None}"
        )

    if best_adv is None:
        # Nothing flipped at all — best-effort yield (will score 0).
        logger.warning(
            f"[adaptive] FAILED — no attack flipped the label at 1/255 "
            f"(closest near-miss margin={seed_margin:.5f}; "
            f"margin>0 means genuinely not flipped)"
        )
        seed_delta = torch.where(
            seed_u != 0, seed_u.sign() * magnitude, torch.zeros_like(seed_u)
        )
        yield (clean_flat + seed_delta).clamp(0.0, 1.0).view_as(clean).detach()
        return

    # ── Phase 2: iterated shrink ────────────────────────────────────
    for _round in range(max(1, shrink_rounds)):
        k_before = best_k

        # (a) gradient-ordered backward elimination
        if _consider(_shrink_support(
            model=model, clean=clean, adv=best_adv,
            target_idx=target_idx, magnitude=magnitude,
        )):
            yield best_adv

        # (b) re-optimize: σ-zero seeded from the current support
        init_u = ((best_adv - clean).view(-1) / magnitude).clamp(-1.0, 1.0)
        adv_re, _k_re, _nm_u, _nm_m = _sigma_zero_run(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=magnitude, n_iterations=n_iterations, init_u=init_u,
        )
        if _consider(adv_re):
            yield best_adv

        # (c) Sparse-RS swap-polish, seeded from the current best
        rs_adv, _rs_k = _sparse_rs_run(
            model=model, clean=clean, target_idx=target_idx,
            magnitude=magnitude, n_queries=sparse_rs_queries,
            seed_delta=(best_adv - clean).view(-1),
        )
        if _consider(rs_adv):
            yield best_adv

        logger.info(
            f"[adaptive] shrink round {_round + 1}: best_k={best_k} "
            f"(was {k_before})"
        )
        if best_k >= k_before:
            break  # converged — no round improved K

    yield best_adv
