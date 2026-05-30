"""Generate per-class Universal Adversarial Perturbations (UAPs).

For each ImageNet class with enough sample images in `received/`, compute
a single ±1/255 perturbation pattern that flips most of those samples.
Store the UAP to `uap_cache/<class_idx>.pt`.

Algorithm (simplified Moosavi-Dezfooli 2017, adapted for L∞ = 1/255):
  For each class c:
    delta = zeros((3, H, W))
    for iteration in range(N_ITERS):
      for each sample image x of class c:
        adv = (x + delta).clamp(0,1)
        if predict(adv) == c:  # not flipped
          # compute single-step perturbation r toward runner_up
          r = sign(grad_of_loss(adv)) * STEP_SIZE
          delta = delta + r
          delta = clamp(delta, -1/255, 1/255)
    save delta to cache

The UAP is class-conditional: at inference time, look up UAP[true_class],
add to clean image, and verify the flip survives PNG round-trip.

Run with:
  python -m tools.uap_generate --min-samples 5
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from collections import defaultdict

import torch

from perturbnet.attacks import _predict_idx, _predict_idx_roundtrip, _top_runner_ups
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import (
    LABELS,
    LABEL_TO_INDEX,
    load_efficientnet_v2_l,
    logits_for_images,
    normalize_prediction_label,
    resolve_target_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("uap_gen")

# Paths
RECEIVED_DIR = Path("/root/linf_26/received")
UAP_CACHE_DIR = Path("/root/linf_26/uap_cache")
UAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# UAP hyperparameters
N_ITERS = 8                 # passes over samples per class
STEP_SIZE = 1.0 / 255.0     # exactly 1 quantization level per step
MAGNITUDE = 1.0 / 255.0     # L∞ cap = exactly 1/255

# Loss: margin-based
# Loss = logit[true_class] - max(logit[others])
# Lower (more negative) = closer to flip


def _load_samples_by_class(min_samples: int = 5) -> dict[int, list[Path]]:
    """Group received/<task>.json files by their true_label's class idx."""
    by_class: dict[int, list[Path]] = defaultdict(list)
    json_files = sorted(RECEIVED_DIR.glob("*.json"))
    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
            true_label = data.get("true_label")
            if not true_label:
                continue
            normed = normalize_prediction_label(true_label)
            idx = resolve_target_index(normed)
            if idx is None:
                continue
            png = jf.with_suffix(".png")
            if png.exists():
                by_class[idx].append(png)
        except Exception as exc:
            logger.warning(f"failed to read {jf}: {exc}")
    # Filter classes by min_samples
    return {c: pngs for c, pngs in by_class.items() if len(pngs) >= min_samples}


def _margin_grad(
    model: torch.nn.Module,
    x: torch.Tensor,
    true_idx: int,
) -> torch.Tensor:
    """Gradient of (margin = logit[true] - max(logit[others])) wrt x."""
    x = x.detach().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    masked = logits.clone()
    masked[true_idx] = float("-inf")
    margin = logits[true_idx] - masked.max()
    g = torch.autograd.grad(margin, x)[0]
    return g.detach()


def _generate_uap_for_class(
    model: torch.nn.Module,
    device: torch.device,
    class_idx: int,
    samples: list[Path],
) -> tuple[torch.Tensor, dict]:
    """Generate a UAP for one class. Returns (delta, stats)."""
    logger.info(
        f"class={class_idx} ({LABELS[class_idx]}) "
        f"n_samples={len(samples)} starting UAP generation"
    )
    # Load all samples once
    import numpy as np
    from PIL import Image
    images: list[torch.Tensor] = []
    for p in samples:
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
            images.append(x)
        except Exception as exc:
            logger.warning(f"  failed to load {p.name}: {exc}")
    if not images:
        return torch.zeros(3, 480, 480, device=device), {"error": "no images loaded"}

    H, W = images[0].shape[-2], images[0].shape[-1]
    C = images[0].shape[0]
    # Initialize delta (try matching first image shape; all imagenet vary slightly)
    delta = torch.zeros((C, H, W), device=device)

    t_start = time.time()
    n_flipped_per_iter = []
    for it in range(N_ITERS):
        flipped = 0
        for x in images:
            # Resize delta to match x if shapes differ (rare on ImageNet)
            d = delta
            if d.shape != x.shape:
                d = torch.nn.functional.interpolate(
                    delta.unsqueeze(0), size=x.shape[-2:], mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            adv = (x + d).clamp(0.0, 1.0)
            pred = _predict_idx(model, adv)
            if pred != class_idx:
                flipped += 1
                continue
            # Compute single-step perturbation toward runner_up
            g = _margin_grad(model, x + d, true_idx=class_idx)
            step = -STEP_SIZE * g.sign()
            d_new = (d + step).clamp(-MAGNITUDE, MAGNITUDE)
            # Update delta (only the regions matching x's shape — pad/crop if needed)
            if d_new.shape == delta.shape:
                delta = d_new
            else:
                # Resize d_new back to delta's shape
                delta = torch.nn.functional.interpolate(
                    d_new.unsqueeze(0), size=delta.shape[-2:], mode="bilinear",
                    align_corners=False,
                ).squeeze(0).clamp(-MAGNITUDE, MAGNITUDE)
        n_flipped_per_iter.append(flipped)
        logger.info(
            f"  iter {it+1}/{N_ITERS}: {flipped}/{len(images)} flipped"
        )
        if flipped == len(images):
            logger.info(f"  all samples flipped — early exit")
            break

    # Final evaluation (with PNG round-trip — what validator sees)
    flipped_rt = 0
    for x in images:
        d = delta
        if d.shape != x.shape:
            d = torch.nn.functional.interpolate(
                delta.unsqueeze(0), size=x.shape[-2:], mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        adv = (x + d).clamp(0.0, 1.0)
        if _predict_idx_roundtrip(model, adv) != class_idx:
            flipped_rt += 1
    elapsed = time.time() - t_start

    # Quantize delta to {-1/255, 0, +1/255} for compact storage
    delta_q = (delta / MAGNITUDE).round().clamp(-1, 1) * MAGNITUDE

    k = int((delta_q.abs() > 1e-9).sum().item())
    stats = {
        "class_idx": class_idx,
        "class_label": LABELS[class_idx],
        "n_samples": len(images),
        "flipped_per_iter": n_flipped_per_iter,
        "flipped_roundtrip": flipped_rt,
        "k_pixels": k,
        "elapsed_s": round(elapsed, 2),
    }
    return delta_q, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--max-classes", type=int, default=100)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate UAPs that already exist in cache",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading EfficientNetV2-L on {device}")
    model = load_efficientnet_v2_l(device).eval()
    # FP32 strict (match coordinator)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    samples_by_class = _load_samples_by_class(min_samples=args.min_samples)
    logger.info(
        f"Found {len(samples_by_class)} classes with >= {args.min_samples} samples"
    )

    # Sort by sample count descending — process best-covered classes first
    sorted_classes = sorted(
        samples_by_class.items(), key=lambda kv: -len(kv[1])
    )[: args.max_classes]

    summary = []
    for class_idx, samples in sorted_classes:
        out_path = UAP_CACHE_DIR / f"class_{class_idx:04d}.pt"
        if out_path.exists() and not args.overwrite:
            logger.info(f"class={class_idx} ({LABELS[class_idx]}): cache hit, skipping")
            continue
        try:
            delta, stats = _generate_uap_for_class(
                model=model, device=device,
                class_idx=class_idx, samples=samples,
            )
            torch.save(delta.cpu(), out_path)
            (UAP_CACHE_DIR / f"class_{class_idx:04d}.json").write_text(
                json.dumps(stats, indent=2)
            )
            summary.append(stats)
            logger.info(
                f"class={class_idx} done: flipped_rt={stats['flipped_roundtrip']}/{stats['n_samples']} "
                f"K={stats['k_pixels']} saved to {out_path.name}"
            )
        except Exception as exc:
            logger.error(f"class={class_idx} failed: {exc}")

    # Save summary
    summary_path = UAP_CACHE_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
