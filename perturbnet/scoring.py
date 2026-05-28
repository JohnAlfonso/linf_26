from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from perturbnet import constants as C
from perturbnet.model import normalize_prediction_label, predict_label


@dataclass
class EvaluationResult:
    score: float
    reason: str
    model_prediction: str = ""
    response_time_ms: int = 0
    norm: float = 0.0
    rmse: float = 0.0
    epsilon: float = 0.0
    ssim: float = 0.0
    psnr_db: float = 0.0


def _compute_ssim(
    x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11
) -> float:
    if x_clean.ndim != 3 or x_adv.ndim != 3:
        return 0.0
    if x_clean.shape != x_adv.shape:
        return 0.0
    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = (
        F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding)
        - mu_x * mu_x
    )
    sigma_y = (
        F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding)
        - mu_y * mu_y
    )
    sigma_xy = (
        F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding)
        - mu_x * mu_y
    )

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def verify_and_score(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    true_label: str,
    epsilon: float,
    skip_flip_check: bool = False,
    response_time_ms: int = 0,
) -> EvaluationResult:
    """Mirror of `PerturbValidator.verify_and_score` operating on already-
    decoded CHW tensors. Same gates and same scoring math; constants come
    from `perturbnet.constants` so a single env-var change tracks both sides.

    `skip_flip_check=True` bypasses the label-flip gate (matches the
    pipeline's `PERTURB_SKIP_FLIP_CHECK` mode). Speed_score is omitted —
    the miner-side pipeline uses this for tier selection, not for the
    final on-chain score, and `SPEED_WEIGHT` defaults to 0 anyway.
    """
    if x_adv.shape != x_clean.shape:
        return EvaluationResult(
            score=0.0, reason="shape_mismatch", response_time_ms=response_time_ms,
            epsilon=float(epsilon),
        )
    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        return EvaluationResult(
            score=0.0, reason="value_out_of_range",
            response_time_ms=response_time_ms, epsilon=float(epsilon),
        )

    try:
        prediction = predict_label(model=model, image_chw=x_adv)
    except Exception as exc:
        return EvaluationResult(
            score=0.0, reason=f"model_inference_failed:{exc}",
            response_time_ms=response_time_ms, epsilon=float(epsilon),
        )

    norm = float((x_adv - x_clean).abs().max().item())

    if norm < C.MIN_LINF_DELTA:
        return EvaluationResult(
            score=0.0, reason="below_min_delta", model_prediction=prediction,
            response_time_ms=response_time_ms, norm=norm, epsilon=float(epsilon),
        )

    effective_max_delta = min(float(epsilon), float(C.MAX_LINF_DELTA))
    rmse = float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item())

    if norm > effective_max_delta:
        return EvaluationResult(
            score=0.0, reason="above_max_delta", model_prediction=prediction,
            response_time_ms=response_time_ms, norm=norm, rmse=rmse,
            epsilon=float(epsilon),
        )

    normalized_prediction = normalize_prediction_label(prediction)
    if not skip_flip_check and normalized_prediction == true_label:
        return EvaluationResult(
            score=0.0, reason="label_match_with_original",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms, norm=norm, rmse=rmse,
            epsilon=float(epsilon),
        )

    ssim = _compute_ssim(x_clean=x_clean, x_adv=x_adv)
    if ssim < C.MIN_SSIM:
        return EvaluationResult(
            score=0.0, reason="below_min_ssim",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms, norm=norm, rmse=rmse,
            epsilon=float(epsilon), ssim=ssim,
        )

    psnr_db = _compute_psnr_db(x_clean=x_clean, x_adv=x_adv)
    if C.MIN_PSNR_DB > 0.0 and psnr_db < C.MIN_PSNR_DB:
        return EvaluationResult(
            score=0.0, reason="below_min_psnr_db",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms, norm=norm, rmse=rmse,
            epsilon=float(epsilon), ssim=ssim, psnr_db=psnr_db,
        )

    denom = max(1e-12, effective_max_delta - float(C.MIN_LINF_DELTA))
    linf_ratio = min(max((norm - float(C.MIN_LINF_DELTA)) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2

    rmse_ratio = min(max(rmse / max(1e-12, effective_max_delta), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2

    total_weight = max(
        1e-12, float(C.LINF_COMPONENT_WEIGHT) + float(C.RMSE_COMPONENT_WEIGHT)
    )
    perturbation_score = (
        float(C.LINF_COMPONENT_WEIGHT) * linf_score
        + float(C.RMSE_COMPONENT_WEIGHT) * rmse_score
    ) / total_weight

    # SPEED_WEIGHT is 0 by default and we don't have a timeout to ratio against
    # here, so the miner-side score is the pure perturbation_score. This is
    # only used for tier selection on the miner; the validator computes its
    # own authoritative score on chain.
    score = float(C.PERTURBATION_WEIGHT) * perturbation_score

    return EvaluationResult(
        score=float(score), reason="success",
        model_prediction=normalized_prediction,
        response_time_ms=response_time_ms, norm=norm, rmse=rmse,
        epsilon=float(epsilon), ssim=ssim, psnr_db=psnr_db,
    )
