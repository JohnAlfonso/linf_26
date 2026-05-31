import argparse
import asyncio
import base64
import json
import logging as pylogging
import os
import time
import typing
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

import bittensor as bt
import torch
from fastapi import HTTPException

from perturbnet.attacks import _boost_margin_on_mask, _shrink_support
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import (
    load_efficientnet_v2_l,
    logits_for_images,
    normalize_prediction_label,
    resolve_target_index,
)
from perturbnet.protocol import AttackChallenge
from perturbnet.scoring import verify_and_score
import tools.perturb_pipeline_service as _pps_module
from tools.perturb_pipeline_service import (
    PerturbRequest,
    _run_pipeline,
)
from tools.perturb_pipeline_service import _state as _pipeline_state

logger = pylogging.getLogger(__name__)


# σ-zero produces minimum-K perturbations sitting at the decision boundary.
# The miner's GPU may report a flip that the validator's GPU rejects due to
# cuDNN/cuBLAS nondeterminism — a logit shift on the order of 1e-6 is enough
# to flip the predicted class back. We require a real logit gap before
# shipping; if the pipeline's adversarial sits below the gap, top-K
# margin-descent PGD pushes it past the boundary by perturbing only the
# most salient pixels per step (trades a few hundred extra pixels of RMSE
# for robustness, much cheaper than a full sign-step on every pixel).
#
# Margin floor. CHOSEN FOR STABILITY: 0.3 gives ~3× the FP-noise buffer
# of 0.1 — fewer occasional `label_match_with_original` failures on
# validator side. Bench winner was 0.1 but that's too aggressive
# for production; the bench can't simulate cross-machine cuDNN/BLAS
# algorithm divergence which can produce 0.01-0.1 logit noise. 0.3
# is the safe floor with TF32 off. Bump to 0.5 if you still see
# label_match failures. With TF32 ON, use 1.0+ (noise is larger).
MIN_MARGIN_LOGITS = float(os.getenv("PERTURB_MIN_MARGIN_LOGITS", "0.3"))
MARGIN_BOOST_MAX_STEPS = int(os.getenv("PERTURB_MARGIN_BOOST_MAX_STEPS", "12"))
# Pixels touched per boost step — ABSOLUTE count, image-size-independent.
# Scaling by image dimension (the old `fraction` knob) over-perturbed
# production-size images by ~10× — every boost step on a 500K-pixel
# image was adding ~2500 pixels when ~250 suffices to clear the 0.3
# margin threshold. 256 absolute is tuned for the typical ~500K-pixel
# validator images we observed (`rmse ~= 0.0001` post-boost, leaving
# `perturbation_score >= 0.95`); tune up for huge images (4+ MP), tune
# down for tiny ones (64×64).
# 384 absolute: bench-tuned sweet spot on 145 saved challenges
# (m01_k384 won +0.0005 vs k256, k512/k1024 all hurt).
MARGIN_BOOST_TOPK_PIXELS = int(os.getenv("PERTURB_MARGIN_BOOST_TOPK_PIXELS", "384"))
MARGIN_BOOST_STEP_SIZE = 1.0 / 255.0

# ── Phase D: adaptive top-K growth ───────────────────────────────────
# Start the per-step pixel budget LOW (32) and grow geometrically
# (×2 default) up to MARGIN_BOOST_TOPK_PIXELS. Trade-off:
#  - Easy images: succeed in 1-3 steps with ~32-128 added pixels (rmse
#    contribution ~5e-5 instead of ~1.6e-4 with fixed-256). Score wins
#    +0.001-0.005 vs Phase A+B+C alone.
#  - Hard images: hit the cap by step 4-5 anyway, same total cost as
#    fixed K=256, just with slightly different pixel selection over the
#    earlier (small-K) steps.
# Growth factor is configurable; 2.0 (doubling) hits the cap in
# log2(256/32) = 3 steps. Higher growth converges faster on hard
# images, lower growth is more RMSE-efficient on borderline cases.
MARGIN_BOOST_TOPK_INITIAL = int(
    os.getenv("PERTURB_MARGIN_BOOST_TOPK_INITIAL", "32")
)
MARGIN_BOOST_TOPK_GROWTH = float(
    os.getenv("PERTURB_MARGIN_BOOST_TOPK_GROWTH", "2.0")
)
# Cap the boost's L∞ budget at exactly the σ-zero magnitude (1/255). Two
# reasons: (1) every pixel value is already a clean multiple of 1/255 after
# σ-zero's PNG roundtrip, so adding ±1/255 then clamping stays bit-exact —
# no roundtrip headroom needed. (2) holding `max|delta| = 1/255` maximizes
# `linf_score = (1 − (norm − 0.003)/0.027)² ≈ 0.933`; letting max|delta|
# grow to 2/255 drops it to ~0.676. Score-wise, RMSE growth from extra
# perturbed pixels costs far less than linf_score growth from larger steps.
MARGIN_BOOST_LINF_BUDGET = 1.0 / 255.0
VALIDATOR_MAX_LINF = 0.03

# Hard wall-clock cap for the ENTIRE forward() call. Must stay below the
# validator's `TIMEOUT_SECONDS = 15` minus network roundtrip (50-200 ms)
# and a safety cushion. If `forward()` returns later than this, the
# dendrite call times out and the request scores 0 regardless of
# perturbation quality. 14.0 s gives ~1 s of cushion for network +
# bittensor framing overhead.
MINER_HARD_TIMEOUT_S = float(os.getenv("PERTURB_MINER_HARD_TIMEOUT_S", "14.0"))

# ── Distributed multi-GPU workers ───────────────────────────────────
# When non-empty, the miner becomes a COORDINATOR that dispatches each
# request to N GPU worker servers in parallel. Each worker runs a
# different attack strategy (set via WORKER_STRATEGY on the worker
# server). The coordinator gathers all responses, scores each via the
# FP32-strict verify_and_score, picks the best, and runs miner-side
# boost before shipping.
#
# Format: comma-separated base URLs. Example:
#   PERTURB_WORKER_URLS=http://10.0.1.1:9200,http://10.0.1.2:9200,http://10.0.1.3:9200
#
# Empty (default) → fall back to local single-GPU pipeline.
_WORKER_URLS_RAW = os.getenv("PERTURB_WORKER_URLS", "").strip()
WORKER_URLS: list[str] = [u.strip() for u in _WORKER_URLS_RAW.split(",") if u.strip()]
# Hard-image-only workers: only dispatched when phase_e gap > 3 (hard image).
# Saves their compute on easy cases. Typically deployed on a dedicated GPU
# (e.g., GPU3 RTX 3090) with specialized hard-image strategies.
_HARD_WORKER_URLS_RAW = os.getenv("PERTURB_HARD_WORKER_URLS", "").strip()
HARD_WORKER_URLS: list[str] = [u.strip() for u in _HARD_WORKER_URLS_RAW.split(",") if u.strip()]
# Per-worker request timeout. Should be larger than the worker's own
# attack deadline (default 9s on the worker side) to leave room for
# network roundtrip. 11s = 9s worker + 2s network overhead.
WORKER_REQUEST_TIMEOUT_S = float(os.getenv("PERTURB_WORKER_TIMEOUT_S", "11.0"))

# ── Async request/result logging ─────────────────────────────────────
# Saves the clean image (PNG) + request metadata at the start, then
# updates the JSON with result fields + perturbed image at the end. Both
# happen on a background thread pool so they don't add to the wall-clock
# time the validator measures. Format mirrors `test_data/` for easy
# benchmark replay.
#   - Set MINER_RECEIVED_DIR (or PERTURB_MINER_RECEIVED_DIR) to enable
#   - Set PERTURB_DISABLE_REQUEST_LOG=1 to disable explicitly
SAVE_REQUEST_DIR = (
    os.getenv("PERTURB_MINER_RECEIVED_DIR", "")
    or os.getenv("MINER_RECEIVED_DIR", "")
)
SAVE_REQUEST_ENABLED = bool(SAVE_REQUEST_DIR) and (
    os.getenv("PERTURB_DISABLE_REQUEST_LOG", "").strip().lower()
    not in {"1", "true", "yes", "on"}
)


def _do_save_request(
    save_dir: str, request_meta: dict, clean_image_b64: str,
) -> typing.Optional[str]:
    """Background-thread: persist `<ts>_<task_id>.png` + `.json`.
    Returns the JSON path on success, None on failure. Never raises —
    a save failure must not affect the validator response."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        stem = f"{request_meta['received_at']}_{request_meta['task_id']}"
        png_path = os.path.join(save_dir, f"{stem}.png")
        json_path = os.path.join(save_dir, f"{stem}.json")
        with open(png_path, "wb") as fh:
            fh.write(base64.b64decode(clean_image_b64))
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(request_meta, fh, indent=2, default=str)
        return json_path
    except Exception as exc:
        logger.warning(f"save_request failed: {exc}")
        return None


def _do_save_result(
    request_future: Future,
    save_dir: str,
    task_id: str,
    result_data: dict,
    perturbed_image_b64: typing.Optional[str],
) -> None:
    """Background-thread: waits for the request save to finish, then
    updates the JSON in-place with `result_data` and writes
    `<stem>_adv.png`. Idempotent on failure — partial state is fine."""
    try:
        # Wait for the request-save to land; if it failed (None), fall
        # back to writing a fresh JSON keyed on task_id.
        json_path: typing.Optional[str] = None
        try:
            json_path = request_future.result(timeout=10.0)
        except Exception as exc:
            logger.warning(f"save_result: request-save wait failed: {exc}")
        if json_path is None or not os.path.exists(json_path):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            stem = f"{ts}_{task_id}"
            json_path = os.path.join(save_dir, f"{stem}.json")
            existing: dict = {"task_id": task_id}
        else:
            with open(json_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        existing["result"] = result_data
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, default=str)
        if perturbed_image_b64:
            adv_path = os.path.splitext(json_path)[0] + "_adv.png"
            try:
                with open(adv_path, "wb") as fh:
                    fh.write(base64.b64decode(perturbed_image_b64))
            except Exception as exc:
                logger.warning(f"save_result adv-png failed: {exc}")
    except Exception as exc:
        logger.warning(f"save_result failed: {exc}")

# When the pipeline returns a "dense" result (rmse above this threshold),
# we suspect σ-zero failed and the pipeline's PGD fallback fired —
# producing a high-margin but pixel-heavy flip. The 56-challenge benchmark
# showed one such case (wood rabbit) cost 0.272 in score. Post-pipeline
# `_shrink_support` reduces K while preserving the flip, recovering most
# of that loss. Threshold 0.0008 corresponds to ~K=4000 on a 500K-pixel
# image — well above σ-zero's typical K=89-3800 range.
DENSE_RMSE_THRESHOLD = float(os.getenv("PERTURB_DENSE_RMSE_THRESHOLD", "0.0008"))
# Hard deadline for the post-pipeline shrink. Reserve ~2s for boost + score.
DENSE_SHRINK_BUDGET_S = float(os.getenv("PERTURB_DENSE_SHRINK_BUDGET_S", "1.5"))

# ── Phase B: σ-zero tuning overrides ─────────────────────────────────
# The pipeline's tier 1 dict bakes `n_iterations=180, num_targets=2`. We
# can't edit the dict without modifying method/perturb_pipeline_service.py,
# so we monkey-patch `attack_sigma_zero` itself to raise the floor on
# both parameters at request time.
#
# `num_targets=3` adds one more runner-up class to the batched pass
# (B=3 → ~1.2× single-image forward cost on RTX PRO 6000 vs 1.0× for
# B=2 in the unpatched pipeline). The extra row attacks a *different*
# class; different runner-ups often need wildly different K, and the
# pipeline keeps the lowest-K candidate, so this trades ~10% more
# compute for a meaningful chance at smaller K on hard images.
#
# `n_iterations=200` is a slight bump (+11%) inside σ-zero's measured
# convergence regime — most of the K-shrinkage happens before 200
# iterations, but the tail still adds diminishing returns. With TF32
# (Phase A) σ-zero @ N=200, B=3 fits in ~10.5s, leaving ~2.5s for
# polish + boost inside the 13s wall.
# CHOSEN FOR STABILITY: 220 gives more deadline headroom than 300.
# At 300 with TF32 off, σ-zero can take 10-12s and gets truncated mid-
# converge; at 220 it finishes cleanly in ~9-10s. Stable runs > theoretical
# peak — small score loss vs N=300, but no awkward mid-cosine truncation.
SIGMA_ZERO_MIN_ITERATIONS = int(os.getenv("PERTURB_SIGMA_ITER", "220"))
SIGMA_ZERO_MIN_TARGETS = int(os.getenv("PERTURB_SIGMA_TARGETS", "3"))

# ── Phase E: per-image difficulty routing ────────────────────────────
# Image "difficulty" is well-predicted by the clean-image logit gap
# (= logit[true] − logit[runner_up]). One forward pass gives it in
# ~30ms. Larger gap means the model is more confident about the true
# class → flipping it needs more perturbation → σ-zero needs more
# diversity (more target classes) to find one that's cheap to flip to.
# Smaller gap means a flip is naturally close → fewer iterations
# suffice and we can spend the slack on polish or restarts.
#
# The thresholds + (N, T) values below are conservative; bump
# PERTURB_PHASE_E_AGGRESSIVE=1 to use a wider envelope (higher T on
# hard images, fewer iterations on easy ones).
PHASE_E_AGGRESSIVE = os.getenv("PERTURB_PHASE_E_AGGRESSIVE", "").strip().lower() in {
    "1", "true", "yes", "on"
}
# (gap_threshold, n_iterations, num_targets)
if PHASE_E_AGGRESSIVE:
    PHASE_E_BANDS: tuple[tuple[float, int, int], ...] = (
        (0.5,   180, 2),  # very easy: small gap, σ-zero converges quickly
        (3.0,   200, 3),  # easy/medium: default-ish
        (10.0,  200, 4),  # hard: more target diversity
        (float("inf"), 220, 5),  # very hard: max effort
    )
else:
    PHASE_E_BANDS = (
        (1.0,   200, 3),  # easy: default
        (5.0,   200, 4),  # medium: one more target
        (float("inf"), 200, 4),  # hard: same as medium (conservative cap)
    )

# Per-request σ-zero overrides set by forward() before each
# `_run_pipeline` call, read by the monkey-patched attack_sigma_zero.
# Empty dict → fall back to module-level SIGMA_ZERO_MIN_* floors.
_PER_REQUEST_SIGMA: dict[str, int] = {}

# ── Phase C: K-preserving margin pre-boost ───────────────────────────
# σ-zero deliberately lands at the decision boundary (margin ≈ 0). Before
# we burn RMSE by adding new pixels in `_boost_margin_if_low`, try the
# free path: sign-flip PGD restricted to σ-zero's existing perturbed
# support. K stays exactly the same, but the (runner_up − true) margin
# can grow several logits if σ-zero's sign assignment was sparsity-
# optimal but not margin-optimal (which is common — σ-zero only
# minimizes l_class when margin > 0, so once it dips negative it freezes
# margin and only optimizes sparsity).
#
# The pipeline's polish phase already calls `_boost_margin_on_mask` but
# only adopts the result if K shrinks — which it CAN'T because the helper
# preserves K by design. So polish discards every iteration. Here we run
# it again and keep whatever margin gain we get, for free.
#
# 6 iterations + 0.4s deadline: cheap-failure budget for cases where
# σ-zero already saturated K and signs can't improve (hard images). The
# helper re-picks runner_up every 4 iterations, so 6 covers 2 re-picks
# — enough on easy images, bails fast on hard.
SIGN_BOOST_ITERATIONS = int(os.getenv("PERTURB_SIGN_BOOST_ITERS", "6"))
SIGN_BOOST_BUDGET_S = float(os.getenv("PERTURB_SIGN_BOOST_BUDGET_S", "0.4"))


def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_axon(wallet, config):
    resolved_config = config() if callable(config) else config
    axon_config = getattr(resolved_config, "axon", None)
    axon_kwargs = {"wallet": wallet}
    if axon_config is not None:
        for key in ("port", "ip", "external_port", "external_ip", "max_workers"):
            value = getattr(axon_config, key, None)
            if value is not None:
                axon_kwargs[key] = value

    if hasattr(bt, "axon"):
        try:
            return bt.axon(**axon_kwargs)
        except TypeError:
            return bt.axon(wallet=wallet, config=resolved_config)
    axon_cls = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("No axon constructor found in bittensor.")
    try:
        return axon_cls(**axon_kwargs)
    except TypeError:
        return axon_cls(wallet=wallet, config=resolved_config)


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_v2_l(self.device)

        # Knob for disabling TF32 — TF32's 10-bit matmul mantissa can
        # cause cross-precision disagreement with FP32 validators on
        # borderline adversarials, producing `label_match_with_original`
        # failures (score=0). Set PERTURB_DISABLE_TF32=1 to force FP32
        # everywhere (~23% slower forwards, but eliminates the noise
        # source). Combined with MIN_MARGIN_LOGITS=1.0 this is belt-
        # and-suspenders against the failure mode.
        _disable_tf32 = os.getenv("PERTURB_DISABLE_TF32", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        # ── Phase A: GPU-side throughput wins ─────────────────────────
        # Trades one-time startup cost for faster per-request inference,
        # which lets σ-zero run more iterations inside the 13s deadline
        # → smaller K → higher rmse_score → higher final score.
        #
        # 1) cuDNN autotuning + TF32. Measured speedups on RTX PRO 6000:
        #    benchmark + TF32 give ~1.2× over default torch settings;
        #    `set_float32_matmul_precision("high")` switches matmuls to
        #    TF32 (≈19 bits of mantissa instead of 23 — negligible vs.
        #    FP rounding noise the attacks already tolerate).
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if _disable_tf32:
                # Force strict FP32 everywhere. cuDNN benchmark stays on
                # because algorithm-selection is independent of precision.
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
            else:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")

        # 2) torch.compile: OFF by default. Benchmarked on RTX PRO 6000:
        #    TF32 alone already gives 1.23× forward+backward speedup;
        #    adding `torch.compile(mode="default")` doesn't measurably
        #    improve backward (where σ-zero spends ~90% of its budget)
        #    while costing ~2 minutes of one-time compilation at startup.
        #    Net: the compile tax doesn't pay back on σ-zero.
        #    Opt-in with PERTURB_ENABLE_TORCH_COMPILE=1 if you have
        #    a workload that's forward-heavy (rare for sparse attacks).
        compile_ok = False
        enable_compile = os.getenv(
            "PERTURB_ENABLE_TORCH_COMPILE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if enable_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="default")
                compile_ok = True
            except Exception as exc:
                logger.warning(f"torch.compile failed ({exc}); using eager model")

        # 3) Warmup: triggers torch.compile codegen AND populates cuDNN's
        #    algorithm cache so the first validator request doesn't pay
        #    the JIT/autotune tax. Warm both inference and backward
        #    because every attack tier needs both paths.
        try:
            t_warm = time.time()
            with torch.no_grad():
                for _ in range(3):
                    _ = logits_for_images(
                        model=self.model,
                        image_bchw=torch.rand(1, 3, 480, 480, device=self.device),
                    )
            grad_x = torch.rand(1, 3, 480, 480, device=self.device, requires_grad=True)
            logits_for_images(
                model=self.model, image_bchw=grad_x,
            ).sum().backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            logger.info(
                f"[PHASE_A] warmup={int((time.time() - t_warm) * 1000)}ms "
                f"compile={compile_ok} "
                f"cudnn_benchmark={torch.backends.cudnn.benchmark} "
                f"tf32={torch.backends.cuda.matmul.allow_tf32}"
            )
        except Exception as exc:
            logger.warning(
                f"[PHASE_A] warmup failed ({exc}); first request may be slow"
            )

        # Reuse the miner's loaded model inside the pipeline so we don't pay
        # a second model load (the pipeline's FastAPI lifespan would normally
        # load its own). Populating `_state` makes `_run_pipeline` use ours.
        # Critically, this happens AFTER torch.compile so the pipeline uses
        # the compiled module too.
        _pipeline_state.model = self.model
        _pipeline_state.device = self.device

        # ── Phase B: σ-zero parameter overrides ───────────────────────
        # Monkey-patch `attack_sigma_zero` so the pipeline's hard-coded
        # `n_iterations=180, num_targets=2` are floors, not hard limits.
        # Setting env vars to a lower value than the pipeline's defaults
        # leaves the pipeline values intact (we only raise, never lower).
        _orig_attack_sigma_zero = _pps_module.attack_sigma_zero

        def _patched_attack_sigma_zero(
            model, clean, target_idx, device,
            magnitude=1.0 / 255.0,
            n_iterations=200,
            n_restarts=1,
            num_targets=1,
            n_iterations_targeted=None,
            deadline=None,
            target_mode="natural",
            use_batched=False,
        ):
            # Phase E reads per-request overrides if set by forward(),
            # otherwise falls back to Phase B floor constants.
            iter_floor = _PER_REQUEST_SIGMA.get("n_iter", SIGMA_ZERO_MIN_ITERATIONS)
            targets_floor = _PER_REQUEST_SIGMA.get("n_targets", SIGMA_ZERO_MIN_TARGETS)
            return _orig_attack_sigma_zero(
                model=model, clean=clean, target_idx=target_idx, device=device,
                magnitude=magnitude,
                n_iterations=max(int(n_iterations), iter_floor),
                n_restarts=n_restarts,
                num_targets=max(int(num_targets), targets_floor),
                n_iterations_targeted=n_iterations_targeted,
                deadline=deadline,
                target_mode=target_mode,
                use_batched=use_batched,
            )

        _pps_module.attack_sigma_zero = _patched_attack_sigma_zero
        logger.info(
            f"[PHASE_B] σ-zero overrides: "
            f"min_iter={SIGMA_ZERO_MIN_ITERATIONS} "
            f"min_targets={SIGMA_ZERO_MIN_TARGETS}"
        )

        # Async save executor: writes request + result data to disk on a
        # background thread so file I/O never blocks the validator
        # response. 2 workers handles request+result for the current
        # call plus light backlog from any previous call still draining.
        if SAVE_REQUEST_ENABLED:
            self._save_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="miner-save"
            )
            self._save_dir = SAVE_REQUEST_DIR
            logger.info(
                f"[SAVE] request+result logging enabled → {self._save_dir}"
            )
        else:
            self._save_executor = None
            self._save_dir = ""
            logger.info(
                "[SAVE] request logging disabled "
                "(set MINER_RECEIVED_DIR in miner.env to enable)"
            )

        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def _log_step_start(self, step_name: str, **context: typing.Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.info(f"[STEP_START] {step_name} {rendered}")
        else:
            logger.info(f"[STEP_START] {step_name}")

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    async def _dispatch_workers(
        self,
        synapse: AttackChallenge,
        true_label: str,
        epsilon: float,
        hard_deadline: float,
        is_hard_image: bool = False,
    ) -> typing.Optional[typing.Any]:
        """Distributed-mode: send the challenge to N worker servers in
        parallel, gather their adversarials, score each via FP32-strict
        `verify_and_score`, and return a pipeline-shaped response
        wrapping the best candidate. Returns None if no worker
        produced a flipping adversarial within budget.

        The coordinator deliberately runs FP32 strict (TF32 disabled)
        while workers may use TF32 — this catches cross-precision
        flip disagreement here, before shipping. Workers that return
        non-flipping adversarials (when checked under coordinator's
        FP32) are silently discarded.
        """
        import types

        # Lazy import: httpx may not be installed in older environments.
        try:
            import httpx
        except ImportError as exc:
            logger.warning(
                f"PERTURB_WORKER_URLS is set but httpx not installed: {exc}; "
                f"falling back to local pipeline"
            )
            return None

        # Worker deadline = wall budget minus boost+ship reserve.
        # On HARD images (phase_e gap > 3), give boost more time by cutting
        # worker time. Hard images need more boost iterations to climb out
        # of the negative-margin valley; workers already plateau by 7s on
        # hard cases anyway.
        reserve_s = 4.5 if is_hard_image else 2.5
        worker_deadline_s = max(2.0, hard_deadline - time.time() - reserve_s)
        payload = {
            "clean_image_b64": synapse.clean_image_b64,
            "true_label": synapse.true_label or "",
            "epsilon": epsilon,
            "deadline_s": worker_deadline_s,
        }

        # Per-call timeout has to accommodate slow geo-distant workers.
        per_call_timeout = httpx.Timeout(
            timeout=WORKER_REQUEST_TIMEOUT_S, connect=2.0,
        )

        async def _call_worker(url: str) -> typing.Optional[dict]:
            try:
                async with httpx.AsyncClient(timeout=per_call_timeout) as client:
                    r = await client.post(f"{url}/attack", json=payload)
                    r.raise_for_status()
                    return r.json()
            except Exception as exc:
                logger.warning(f"worker {url} failed: {exc}")
                return None

        # Dispatch list: always-on WORKER_URLS plus HARD_WORKER_URLS only
        # when this is a hard image (gap > 3). Hard workers run extended
        # iterations / aggressive K-reduction that's only worth the GPU
        # cost when σ-zero is expected to struggle.
        dispatch_urls = list(WORKER_URLS)
        if is_hard_image and HARD_WORKER_URLS:
            dispatch_urls.extend(HARD_WORKER_URLS)

        t_dispatch = time.time()
        try:
            results = await asyncio.gather(*[_call_worker(u) for u in dispatch_urls])
        except Exception as exc:
            logger.warning(f"asyncio.gather failed: {exc}")
            return None
        dispatch_ms = int((time.time() - t_dispatch) * 1000)

        candidates = [r for r in results if r is not None]
        if not candidates:
            logger.warning(
                f"all {len(dispatch_urls)} workers failed in {dispatch_ms}ms; "
                f"falling back to local pipeline"
            )
            return None

        # Coordinator-side FP32 scoring. The worker reports `flipped`
        # under TF32; we re-verify under FP32 strict here.
        # Selection: highest FP32 score wins. Earlier experiment with
        # "boost-aware" tie-break by margin/K was rolled back — real
        # validator scores dropped because jsma's high-margin/no-boost
        # ship-score still trailed sigma's boosted ship-score on most
        # images. FP32 score is the most reliable signal we have.
        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        best_b64: typing.Optional[str] = None
        best_score = -1.0
        best_result = None
        best_strategy = "?"
        scored_count = 0
        # Track all evaluated candidates so we can fall back intelligently
        # when nobody flipped under FP32 (avoid shipping clean image).
        all_evaluated: list[tuple[float, dict, typing.Any]] = []
        for c in candidates:
            try:
                adv = decode_image_b64(c["adv_image_b64"]).to(self.device)
                r = verify_and_score(
                    model=self.model, x_clean=clean, x_adv=adv,
                    true_label=true_label, epsilon=epsilon,
                )
                scored_count += 1
                all_evaluated.append((float(r.score), c, r))
                if r.score > best_score:
                    best_score = float(r.score)
                    best_b64 = c["adv_image_b64"]
                    best_result = r
                    best_strategy = c.get("strategy", "?")
            except Exception as exc:
                logger.warning(f"score worker result failed: {exc}")

        # When NO worker flipped under FP32 (best_score == 0), the first
        # candidate wins by tie-breaking, but it may have effectively zero
        # delta — e.g. σ-zero deadline-out → returned clean. Shipping that
        # gives validator score=0 with reason=below_min_delta. Instead,
        # pick the candidate with the highest k>0 + TF32 margin so boost
        # has a real perturbation to grow from.
        if best_score <= 0.0 and all_evaluated:
            usable = [
                (sc, c, r) for sc, c, r in all_evaluated
                if int(c.get("k", 0)) > 0
            ]
            if usable:
                pick = max(
                    usable,
                    key=lambda t: (float(t[1].get("margin", -1e9)), int(t[1].get("k", 0))),
                )
                best_score, c, r = pick
                best_b64 = c["adv_image_b64"]
                best_result = r
                best_strategy = c.get("strategy", "?")

        if best_b64 is None or best_result is None:
            logger.warning(
                f"workers returned {len(candidates)} candidates but none "
                f"flipped under FP32 (dispatch={dispatch_ms}ms); "
                f"falling back to local pipeline"
            )
            return None

        logger.info(
            f"workers: {len(candidates)}/{len(dispatch_urls)} ok "
            f"(hard={is_hard_image}), scored={scored_count}, "
            f"best_strategy={best_strategy}, "
            f"best_score={best_score:.4f}, dispatch={dispatch_ms}ms"
        )

        # Wrap in a pipeline-shaped response so the rest of forward()
        # (dense-shrink, boost, scoring) works unchanged.
        fake_resp = types.SimpleNamespace(
            perturbed_image_b64=best_b64,
            best_result={
                "score": float(best_score),
                "reason": str(best_result.reason),
                "model_prediction": str(best_result.model_prediction),
                "norm": float(best_result.norm),
                "rmse": float(best_result.rmse),
                "epsilon": float(best_result.epsilon),
                "ssim": float(best_result.ssim),
                "psnr_db": float(best_result.psnr_db),
                "worker_strategy": best_strategy,
                "worker_count": len(candidates),
            },
            best_step=-1,
            elapsed_seconds=0.0,
        )
        return fake_resp

    def _phase_e_choose_band(
        self, clean: torch.Tensor, target_idx: int,
    ) -> typing.Tuple[float, int, int]:
        """Phase E: pick σ-zero (n_iterations, num_targets) from the clean-
        image logit gap. One forward pass (~30ms on TF32 RTX PRO 6000).

        Returns (gap, n_iter, n_targets).
        """
        with torch.no_grad():
            lg = logits_for_images(
                model=self.model, image_bchw=clean.unsqueeze(0),
            )[0]
        masked = lg.clone()
        masked[target_idx] = float("-inf")
        runner_up_logit = float(masked.max().item())
        true_logit = float(lg[target_idx].item())
        gap = true_logit - runner_up_logit
        for thresh, n_iter, n_tgts in PHASE_E_BANDS:
            if gap < thresh:
                return gap, n_iter, n_tgts
        # Shouldn't reach (last band has gap_threshold=inf), but be safe.
        _, n_iter, n_tgts = PHASE_E_BANDS[-1]
        return gap, n_iter, n_tgts

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        task_id = getattr(synapse, "task_id", "unknown")
        self._log_step_start(
            "miner_forward",
            task_id=task_id,
            norm_type=getattr(synapse, "norm_type", "unknown"),
            epsilon=getattr(synapse, "epsilon", "unknown"),
        )
        if synapse.norm_type != "Linf":
            logger.info(f"Skipping task={task_id}: unsupported norm_type={synapse.norm_type}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        # Resolve the target index from the validator's `true_label` rather
        # than letting the pipeline call its own `predict_index(model, clean)`.
        # If miner and validator GPUs disagree at the decision boundary, the
        # pipeline's argmax may differ from the validator's, and the attack
        # ends up flipping toward the validator's true class (= no flip on
        # the validator side). Use the same normalization the validator's
        # `verify_and_score` uses for label comparison.
        true_label = normalize_prediction_label(synapse.true_label or "")
        desired_target_idx = (
            resolve_target_index(true_label) if true_label else None
        )
        if desired_target_idx is None:
            logger.warning(
                f"task={task_id} could not resolve true_label='{synapse.true_label}'; "
                f"falling back to pipeline's local prediction"
            )

        # Async save phase 1: queue the clean image + request metadata
        # write. ThreadPoolExecutor.submit() returns instantly; the
        # actual disk write happens on a worker thread. Captured here
        # so the result-save phase can wait for the file to exist.
        save_request_future: typing.Optional[Future] = None
        request_meta: typing.Optional[dict] = None
        if self._save_executor is not None:
            request_meta = {
                "task_id": str(task_id),
                "norm_type": str(synapse.norm_type),
                "epsilon": float(synapse.epsilon),
                "min_delta": float(getattr(synapse, "min_delta", 0.0)),
                "true_label": str(synapse.true_label or ""),
                "caller_hotkey": str(
                    getattr(getattr(synapse, "dendrite", None), "hotkey", "")
                ),
                "received_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "received_unix": time.time(),
                "timeout_seconds": int(
                    getattr(synapse, "timeout_seconds", 0)
                ),
            }
            try:
                save_request_future = self._save_executor.submit(
                    _do_save_request,
                    self._save_dir,
                    request_meta,
                    synapse.clean_image_b64,
                )
            except Exception as exc:
                logger.warning(f"save_request submit failed: {exc}")

        req = PerturbRequest(
            image_b64=synapse.clean_image_b64,
            image_name=str(task_id),
            epsilon=float(synapse.epsilon),
            save=False,
        )
        t_start = time.time()
        # Hard wall-clock deadline for the entire forward() call. The
        # validator's dendrite times out at 15s and scores 0 on timeout,
        # so we MUST return before this — every boost step checks it.
        hard_deadline = t_start + MINER_HARD_TIMEOUT_S
        phase_e_info = "skipped"
        try:
            # The pipeline's `_state.lock` serializes the GPU; only one
            # attack runs at a time even if Axon dispatches concurrently.
            # The monkey-patch on `predict_index` is also done inside the
            # lock so concurrent calls can't see each other's overrides.
            with _pipeline_state.lock:
                # Phase E: choose σ-zero parameters from clean-image gap.
                # Only when we have a validator-supplied true_label —
                # otherwise we don't know which class to measure gap to.
                if desired_target_idx is not None:
                    gap = 0.0
                    try:
                        clean_for_gap = decode_image_b64(
                            synapse.clean_image_b64
                        ).to(self.device)
                        gap, n_iter, n_tgts = self._phase_e_choose_band(
                            clean_for_gap, desired_target_idx,
                        )
                        _PER_REQUEST_SIGMA["n_iter"] = n_iter
                        _PER_REQUEST_SIGMA["n_targets"] = n_tgts
                        phase_e_info = f"gap={gap:.2f},N={n_iter},T={n_tgts}"
                    except Exception as exc:
                        logger.warning(f"phase_e gap computation failed: {exc}")
                        _PER_REQUEST_SIGMA.clear()

                # Hard-image detection: validator stats show our min_score
                # (0.9036) is what drops avg, vs top miner min (0.9299).
                # On gap > 3 images, σ-zero workers plateau early and the
                # bottleneck is boost time. Reallocate budget: less worker
                # time, more boost time on hard images.
                is_hard_image = gap > 3.0

                # Distributed multi-GPU path. If WORKER_URLS is set, send
                # the request to N worker servers in parallel, score
                # each via FP32-strict verify_and_score, and use the
                # best as `resp`. If all workers fail, fall through to
                # the local pipeline below.
                used_workers = False
                resp = None
                if WORKER_URLS and desired_target_idx is not None:
                    try:
                        worker_resp = await self._dispatch_workers(
                            synapse=synapse,
                            true_label=true_label,
                            epsilon=float(synapse.epsilon),
                            hard_deadline=hard_deadline,
                            is_hard_image=is_hard_image,
                        )
                        if worker_resp is not None:
                            resp = worker_resp
                            used_workers = True
                    except Exception as exc:
                        logger.warning(
                            f"task={task_id} worker dispatch failed: {exc}; "
                            f"falling back to local pipeline"
                        )

                if not used_workers:
                    orig_predict_index = _pps_module.predict_index
                    if desired_target_idx is not None:
                        _pps_module.predict_index = (
                            lambda model, image_chw, _idx=desired_target_idx: _idx
                        )
                    try:
                        resp = _run_pipeline(req)
                    finally:
                        _pps_module.predict_index = orig_predict_index
                        _PER_REQUEST_SIGMA.clear()

                # Pre-boost: if the pipeline returned a dense result
                # (rmse > DENSE_RMSE_THRESHOLD), it almost certainly came
                # from the pipeline's PGD fallback after σ-zero failed.
                # Run `_shrink_support` to drop pixels while preserving
                # the flip, recovering most of the score loss. Tested on
                # the wood rabbit case (rmse=0.02 → rmse=0.0005 → score
                # 0.68 → 0.94).
                pipeline_b64_for_boost = resp.perturbed_image_b64
                dense_shrink_info = "skipped"
                # Gate reduced 2.5→1.8s to let shrink fire even when 12-worker
                # dispatch eats 11.7s. Shrink's actual budget is still bounded
                # by hard_deadline - 1.5 (line below), so it self-limits.
                # Trade: less boost time when shrink runs, but shrink rescues
                # more (+0.005-0.020 observed on hard images).
                if desired_target_idx is not None and time.time() < hard_deadline - 1.8:
                    try:
                        adv_check = decode_image_b64(
                            pipeline_b64_for_boost
                        ).to(self.device)
                        clean_check = decode_image_b64(
                            synapse.clean_image_b64
                        ).to(self.device)
                        rmse_check = float(torch.sqrt(
                            torch.mean((adv_check - clean_check) ** 2)
                        ).item())
                        if rmse_check > DENSE_RMSE_THRESHOLD:
                            # Tightened from -1.5 → -1.2: gives shrink a touch
                            # more time on tight cases (boost gets less but
                            # margin is usually already passable).
                            shrink_dl = min(
                                time.time() + DENSE_SHRINK_BUDGET_S,
                                hard_deadline - 1.2,
                            )
                            shrunk = _shrink_support(
                                model=self.model,
                                clean=clean_check,
                                adv=adv_check,
                                target_idx=desired_target_idx,
                                magnitude=1.0 / 255.0,
                                deadline=shrink_dl,
                            )
                            # Re-encode and verify it still flips.
                            shrunk_b64 = encode_image_b64(shrunk.detach().cpu())
                            rt_check = decode_image_b64(shrunk_b64).to(self.device)
                            with torch.no_grad():
                                lg = logits_for_images(
                                    model=self.model,
                                    image_bchw=rt_check.unsqueeze(0),
                                )[0]
                            masked = lg.clone()
                            masked[desired_target_idx] = float("-inf")
                            flip_ok = (
                                masked.max().item()
                                > lg[desired_target_idx].item()
                            )
                            if flip_ok:
                                rmse_after = float(torch.sqrt(
                                    torch.mean((rt_check - clean_check) ** 2)
                                ).item())
                                pipeline_b64_for_boost = shrunk_b64
                                dense_shrink_info = (
                                    f"rmse:{rmse_check:.5f}→{rmse_after:.5f}"
                                )
                            else:
                                dense_shrink_info = (
                                    f"discarded(broke_flip,rmse={rmse_check:.5f})"
                                )
                        else:
                            dense_shrink_info = f"not_dense(rmse={rmse_check:.5f})"
                    except Exception as exc:
                        dense_shrink_info = f"failed:{type(exc).__name__}"
                        logger.warning(f"dense shrink failed: {exc}")

                # Margin boost happens under the same lock — it uses the
                # GPU via the shared model. Pass the hard deadline so
                # each step bails before we'd cross the validator's 15s
                # timeout. Reserve 0.3s for encode + return.
                # On HARD images, boost typically hits its K cap with
                # margin still negative; raise the cap so it has room
                # to grow further (we already gave it more time via
                # reduced worker reserve).
                if desired_target_idx is not None:
                    topk_cap_override = 640 if is_hard_image else None
                    boosted_b64, margin_info = self._boost_margin_if_low(
                        adv_b64=pipeline_b64_for_boost,
                        clean_b64=synapse.clean_image_b64,
                        true_idx=desired_target_idx,
                        epsilon=float(synapse.epsilon),
                        deadline=hard_deadline - 0.3,
                        topk_cap_override=topk_cap_override,
                    )
                    synapse.perturbed_image_b64 = boosted_b64
                else:
                    margin_info = "skipped(no_true_label)"
                    synapse.perturbed_image_b64 = resp.perturbed_image_b64

                # ── Final L∞ projection ──────────────────────────────
                # The pipeline's PGD fallback can produce adversarials
                # with max|delta| > 1/255 (it uses safe_eps = ε − 1.5/255
                # which is much larger). Our boost only constrains the
                # top-K pixels it touches per step, so non-touched PGD-
                # fallback pixels stay at high magnitudes — tanking
                # `linf_score` (e.g. jellyfish gap 0.42, wood rabbit 0.27).
                #
                # Project ALL pixels to clean ± 1/255 and check if the
                # flip survives. If yes, ship the projected version
                # (preserves `linf_score ≈ 0.933`). If no, fall back to
                # the unprojected adv (better score=0.5-0.7 than score=0
                # from a broken flip).
                projection_info = "skipped"
                if (
                    desired_target_idx is not None
                    and time.time() < hard_deadline - 0.2
                ):
                    try:
                        shipped = decode_image_b64(
                            synapse.perturbed_image_b64
                        ).to(self.device)
                        clean_t = decode_image_b64(
                            synapse.clean_image_b64
                        ).to(self.device)
                        norm_before = float(
                            (shipped - clean_t).abs().max().item()
                        )
                        if norm_before > 1.0 / 255.0 + 1e-6:
                            delta = (shipped - clean_t).clamp(-1.0 / 255.0, 1.0 / 255.0)
                            projected = (clean_t + delta).clamp(0.0, 1.0)
                            # Re-encode + decode (PNG roundtrip = validator's view)
                            proj_b64 = encode_image_b64(projected.detach().cpu())
                            rt = decode_image_b64(proj_b64).to(self.device)
                            with torch.no_grad():
                                lg = logits_for_images(
                                    model=self.model,
                                    image_bchw=rt.unsqueeze(0),
                                )[0]
                            masked = lg.clone()
                            masked[desired_target_idx] = float("-inf")
                            flip_survives = (
                                masked.max().item()
                                > lg[desired_target_idx].item()
                            )
                            if flip_survives:
                                synapse.perturbed_image_b64 = proj_b64
                                norm_after = float(
                                    (rt - clean_t).abs().max().item()
                                )
                                projection_info = (
                                    f"norm:{norm_before:.5f}→{norm_after:.5f}"
                                )
                            else:
                                projection_info = (
                                    f"discarded(broke_flip,"
                                    f"norm={norm_before:.5f})"
                                )
                        else:
                            projection_info = f"in_budget({norm_before:.5f})"
                    except Exception as exc:
                        projection_info = f"failed:{type(exc).__name__}"
                        logger.warning(f"L∞ projection failed: {exc}")

                # Re-score the shipped adversarial via the validator-shaped
                # `verify_and_score`. The pipeline's `resp.best_result.score`
                # reflects the pre-boost adversarial; the validator sees the
                # post-boost one (boost adds pixels → higher rmse → slightly
                # lower score). Logging both makes regressions diagnosable.
                # Skip if past the hard deadline — better to ship a few
                # ms early than blow the validator's 15s timeout for a
                # log line.
                pipeline_score = float(resp.best_result.get("score", 0.0))
                shipped_score: float
                shipped_reason: str
                if time.time() >= hard_deadline - 0.2:
                    shipped_score = -1.0
                    shipped_reason = "skipped(deadline)"
                else:
                    try:
                        shipped_adv = decode_image_b64(
                            synapse.perturbed_image_b64
                        ).to(self.device)
                        shipped_result = verify_and_score(
                            model=self.model,
                            x_clean=decode_image_b64(synapse.clean_image_b64).to(self.device),
                            x_adv=shipped_adv,
                            true_label=true_label,
                            epsilon=float(synapse.epsilon),
                        )
                        shipped_score = float(shipped_result.score)
                        shipped_reason = shipped_result.reason
                    except Exception as exc:
                        shipped_score = -1.0
                        shipped_reason = f"rescore_failed:{exc}"

            elapsed_ms = int((time.time() - t_start) * 1000)
            reason = resp.best_result.get("reason", "?")
            logger.info(
                f"task={task_id} pipeline_score={pipeline_score:.4f} "
                f"shipped_score={shipped_score:.4f} shipped_reason={shipped_reason} "
                f"pipeline_reason={reason} margin={margin_info} "
                f"dense_shrink={dense_shrink_info} linf_proj={projection_info} "
                f"phase_e={phase_e_info} elapsed={elapsed_ms}ms"
            )
        except HTTPException as exc:
            # Pipeline raises 422 only when even the PGD fallback couldn't
            # produce a flipping candidate. Returning the clean image yields
            # score=0 on the validator side, but is non-destructive.
            elapsed_ms = int((time.time() - t_start) * 1000)
            logger.warning(
                f"task={task_id} pipeline_failed status={exc.status_code} "
                f"detail={exc.detail} elapsed={elapsed_ms}ms; returning clean"
            )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            pipeline_score = 0.0
            shipped_score = 0.0
            shipped_reason = f"pipeline_failed:{exc.status_code}"
            margin_info = "skipped"
            dense_shrink_info = "skipped"
            projection_info = "skipped"
            reason = "exception"
        except Exception as exc:
            elapsed_ms = int((time.time() - t_start) * 1000)
            logger.exception(
                f"task={task_id} pipeline raised unexpectedly "
                f"elapsed={elapsed_ms}ms; returning clean"
            )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            pipeline_score = 0.0
            shipped_score = 0.0
            shipped_reason = f"exception:{type(exc).__name__}"
            margin_info = "skipped"
            dense_shrink_info = "skipped"
            projection_info = "skipped"
            reason = "exception"

        # Async save phase 2: queue result write. Fire-and-forget — the
        # synapse response goes back to the validator immediately while
        # the disk write happens on the background thread pool.
        if self._save_executor is not None and save_request_future is not None:
            result_data = {
                "elapsed_ms": int(elapsed_ms),
                "pipeline_score": float(pipeline_score),
                "shipped_score": float(shipped_score),
                "shipped_reason": str(shipped_reason),
                "pipeline_reason": str(reason),
                "margin_info": str(margin_info),
                "dense_shrink_info": str(dense_shrink_info),
                "linf_proj_info": str(projection_info),
                "phase_e_info": str(phase_e_info),
                "completed_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "completed_unix": time.time(),
            }
            try:
                self._save_executor.submit(
                    _do_save_result,
                    save_request_future,
                    self._save_dir,
                    str(task_id),
                    result_data,
                    synapse.perturbed_image_b64,
                )
            except Exception as exc:
                logger.warning(f"save_result submit failed: {exc}")

        return synapse

    def _boost_margin_if_low(
        self,
        adv_b64: str,
        clean_b64: str,
        true_idx: int,
        epsilon: float,
        deadline: float | None = None,
        topk_cap_override: int | None = None,
    ) -> typing.Tuple[str, str]:
        """If the pipeline's adversarial sits within `MIN_MARGIN_LOGITS` of
        the decision boundary, push it past via short margin-descent PGD
        constrained to the validator's L∞ budget (with PNG-roundtrip
        headroom). Trades a small RMSE penalty for robustness against
        miner↔validator hardware nondeterminism.

        Returns `(b64, info)` where `info` is a short status string for
        the caller's log line.
        """
        adv = decode_image_b64(adv_b64).to(self.device)
        clean = decode_image_b64(clean_b64).to(self.device)

        def _margin_and_runner(x: torch.Tensor) -> typing.Tuple[float, int]:
            with torch.no_grad():
                lg = logits_for_images(
                    model=self.model, image_bchw=x.unsqueeze(0),
                )[0]
            masked = lg.clone()
            masked[true_idx] = float("-inf")
            r = int(masked.argmax().item())
            m = float((lg[r] - lg[true_idx]).item())
            return m, r

        margin, runner_up = _margin_and_runner(adv)
        if margin >= MIN_MARGIN_LOGITS:
            return adv_b64, f"{margin:.3f}(ok)"

        # Skip the rest if we're past the hard deadline (better to ship
        # the unboosted σ-zero result than time out and score 0).
        if deadline is not None and time.time() >= deadline:
            return adv_b64, f"{margin:.3f}(deadline-skip)"

        # ── Phase C: try the K-preserving sign-flip boost first ──────
        # If σ-zero's signs aren't margin-optimal, sign-PGD on the
        # existing support can lift margin several logits *without
        # adding a single new pixel*. If this lands us above the
        # threshold, we ship a result with σ-zero's original K and pay
        # ZERO RMSE cost. If it improves margin but not enough, we use
        # the new adv as the starting point for the K-growing fallback.
        # Skip entirely if the hard deadline doesn't have room for it.
        sign_boost_dl = (
            min(time.time() + SIGN_BOOST_BUDGET_S, deadline - 0.2)
            if deadline is not None
            else time.time() + SIGN_BOOST_BUDGET_S
        )
        try:
            if deadline is None or sign_boost_dl > time.time() + 0.05:
                sign_boosted = _boost_margin_on_mask(
                    model=self.model,
                    clean=clean,
                    adv=adv,
                    target_idx=true_idx,
                    magnitude=MARGIN_BOOST_STEP_SIZE,
                    n_iterations=SIGN_BOOST_ITERATIONS,
                    deadline=sign_boost_dl,
                )
            else:
                sign_boosted = adv  # not enough time, fall through
            # Sign-flips can rarely break the flip post-quantization; verify
            # via PNG roundtrip (the validator's actual view) before adopting.
            rt = decode_image_b64(
                encode_image_b64(sign_boosted.detach().cpu())
            ).to(self.device)
            m_sign, ru_sign = _margin_and_runner(rt)
            if m_sign > 0.0:
                if m_sign >= MIN_MARGIN_LOGITS:
                    # K-preserving boost was enough — ship with no RMSE cost.
                    return (
                        encode_image_b64(rt.detach().cpu()),
                        f"{m_sign:.3f}(sign-flipped)",
                    )
                # Improved but not enough; use as starting point for K-growing.
                if m_sign > margin:
                    adv = rt
                    margin = m_sign
                    runner_up = ru_sign
        except Exception as exc:
            logger.warning(f"phase_c sign-flip boost failed: {exc}")

        # Bound the boost's L∞ to exactly the σ-zero magnitude so the
        # perturbation never exceeds `1/255` in any single pixel (preserves
        # linf_score ≈ 0.933). The validator's gate at `min(epsilon, 0.03)`
        # is a much looser ceiling that we never approach.
        safe_eps = min(MARGIN_BOOST_LINF_BUDGET, min(float(epsilon), VALIDATOR_MAX_LINF))
        if safe_eps <= 0.0:
            return adv_b64, f"{margin:.3f}(eps_too_small)"

        # Phase D: adaptive top-K starts low and grows toward the cap.
        topk_cap_base = topk_cap_override if topk_cap_override is not None else MARGIN_BOOST_TOPK_PIXELS
        topk_cap = max(1, min(topk_cap_base, adv.numel()))
        topk = max(1, min(MARGIN_BOOST_TOPK_INITIAL, topk_cap))
        clean_flat = clean.view(-1)
        # SAFETY: track the best-flipping candidate we've seen during the
        # loop. Boost iterations can make the margin worse (the
        # retargeting + saturation interaction occasionally lands on a
        # config that flips a different way). If the loop exhausts
        # MAX_STEPS or hits the deadline mid-step, we ship the
        # best-margin candidate, not the last one. If no positive-margin
        # candidate was ever seen during the boost, we fall back to the
        # ORIGINAL pipeline result rather than a guaranteed score-0 ship.
        original_b64 = adv_b64
        best_post_b64 = encode_image_b64(adv.detach().cpu())
        best_post_margin = margin
        for step_i in range(MARGIN_BOOST_MAX_STEPS):
            # Bail before the hard deadline. Each step costs ~150ms
            # (1 fwd+bwd + 1 PNG roundtrip + 1 margin check) so we need
            # ≥200ms of headroom to safely take another step.
            if deadline is not None and time.time() >= deadline - 0.2:
                # Ship the best-flipping candidate seen so far.
                if best_post_margin > 0.0:
                    return best_post_b64, f"{best_post_margin:.3f}(deadline@{step_i})"
                return original_b64, f"{best_post_margin:.3f}(deadline-revert@{step_i})"
            adv = adv.detach().requires_grad_(True)
            lg = logits_for_images(
                model=self.model, image_bchw=adv.unsqueeze(0),
            )[0]
            # Minimize (logit[true] − logit[runner_up]) to maximize the
            # positive margin (runner_up − true).
            obj = lg[true_idx] - lg[runner_up]
            grad = torch.autograd.grad(obj, adv)[0]
            adv_flat = adv.view(-1).detach()
            flat_grad = grad.view(-1)
            # Per-pixel effective step = clamped(adv − (1/255)·sign(grad)) − adv.
            # For pixels saturated against the step direction (either at the
            # L∞ cap or at [0,1] cap), eff_step = 0 → no-op. Plain top-|grad|
            # ranking would keep picking these saturated cells (since σ-zero
            # already perturbed the largest-|grad| pixels), wasting steps.
            # Ranking by (−grad · eff_step) = effective margin gain pushes
            # the boost onto pixels that can ACTUALLY move.
            attempted = adv_flat - MARGIN_BOOST_STEP_SIZE * flat_grad.sign()
            post_adv = torch.max(
                torch.min(attempted, clean_flat + safe_eps),
                clean_flat - safe_eps,
            ).clamp(0.0, 1.0)
            eff_step = post_adv - adv_flat
            margin_gain = -flat_grad * eff_step  # ≥ 0 when step descends obj
            _, top_idx = margin_gain.topk(min(topk, margin_gain.numel()))
            delta_apply = torch.zeros_like(flat_grad)
            delta_apply[top_idx] = eff_step[top_idx]
            adv = (adv_flat + delta_apply).view_as(adv)
            # No further projection: eff_step already respects L∞ and [0,1].
            # Re-check via the PNG-roundtripped tensor — the validator's view.
            rt = decode_image_b64(
                encode_image_b64(adv.detach().cpu())
            ).to(self.device)
            new_margin, new_runner = _margin_and_runner(rt)
            if new_margin >= MIN_MARGIN_LOGITS:
                adv = rt
                margin = new_margin
                return (
                    encode_image_b64(adv.detach().cpu()),
                    f"{margin:.3f}(boosted@{step_i+1},k={topk})",
                )
            # Track best-flipping candidate (positive margin).
            if new_margin > best_post_margin:
                best_post_margin = new_margin
                best_post_b64 = encode_image_b64(rt.detach().cpu())
            # If the runner-up shifted (boundary moved), retarget.
            runner_up = new_runner
            # Phase D: grow top-K for next iteration if we haven't capped.
            topk = min(int(topk * MARGIN_BOOST_TOPK_GROWTH), topk_cap)

        # Boost loop exhausted MAX_STEPS without crossing MIN_MARGIN_LOGITS.
        # Ship the best-margin candidate we tracked, NOT the latest one
        # (boost can make things worse). If even the best is non-flipping
        # (margin ≤ 0), revert to the original pipeline result, which at
        # least flipped according to the pipeline's own scoring.
        if best_post_margin > 0.0:
            return best_post_b64, f"{best_post_margin:.3f}(partial,k={topk})"
        return original_b64, f"{best_post_margin:.3f}(revert,k={topk})"

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        self._log_step_start(
            "miner_blacklist",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Blacklist reject: missing caller hotkey")
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            logger.warning(f"Blacklist reject: unregistered caller hotkey={hotkey}")
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            logger.warning(f"Blacklist reject: caller uid={uid} lacks validator permit")
            return True, "Caller is not validator"

        logger.info(f"Blacklist allow: caller uid={uid} hotkey={hotkey}")
        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        self._log_step_start(
            "miner_priority",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.info("Priority=0.0: missing caller hotkey")
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.info(f"Priority=0.0: unknown hotkey={synapse.dendrite.hotkey}")
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[uid])
        logger.info(f"Priority computed: uid={uid} priority={priority:.6f}")
        return priority

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        logger.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))),
    )
    parser.add_argument(
        "--axon.external_port",
        dest="axon_external_port",
        type=int,
        default=int(os.getenv("MINER_EXTERNAL_PORT", os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000")))),
    )

    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = int(getattr(config.axon, "port", getattr(config, "axon_port", 9000)))

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))
    config.axon.external_port = int(
        getattr(config.axon, "external_port", getattr(config, "axon_external_port", config.axon.port))
    )

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()
