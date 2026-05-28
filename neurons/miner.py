import argparse
import logging as pylogging
import os
import time
import typing

import bittensor as bt
import torch
from fastapi import HTTPException

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
# 0.3 logits is ~5 orders of magnitude above FP noise (~1e-6) — empirically
# enough to survive cross-GPU prediction disagreement without burning RMSE
# on overshooting margins.
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
MARGIN_BOOST_TOPK_PIXELS = int(os.getenv("PERTURB_MARGIN_BOOST_TOPK_PIXELS", "256"))
MARGIN_BOOST_STEP_SIZE = 1.0 / 255.0
# Cap the boost's L∞ budget at exactly the σ-zero magnitude (1/255). Two
# reasons: (1) every pixel value is already a clean multiple of 1/255 after
# σ-zero's PNG roundtrip, so adding ±1/255 then clamping stays bit-exact —
# no roundtrip headroom needed. (2) holding `max|delta| = 1/255` maximizes
# `linf_score = (1 − (norm − 0.003)/0.027)² ≈ 0.933`; letting max|delta|
# grow to 2/255 drops it to ~0.676. Score-wise, RMSE growth from extra
# perturbed pixels costs far less than linf_score growth from larger steps.
MARGIN_BOOST_LINF_BUDGET = 1.0 / 255.0
VALIDATOR_MAX_LINF = 0.03


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
        # Reuse the miner's loaded model inside the pipeline so we don't pay
        # a second model load (the pipeline's FastAPI lifespan would normally
        # load its own). Populating `_state` makes `_run_pipeline` use ours.
        _pipeline_state.model = self.model
        _pipeline_state.device = self.device

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

        req = PerturbRequest(
            image_b64=synapse.clean_image_b64,
            image_name=str(task_id),
            epsilon=float(synapse.epsilon),
            save=False,
        )
        t_start = time.time()
        try:
            # The pipeline's `_state.lock` serializes the GPU; only one
            # attack runs at a time even if Axon dispatches concurrently.
            # The monkey-patch on `predict_index` is also done inside the
            # lock so concurrent calls can't see each other's overrides.
            with _pipeline_state.lock:
                orig_predict_index = _pps_module.predict_index
                if desired_target_idx is not None:
                    _pps_module.predict_index = (
                        lambda model, image_chw, _idx=desired_target_idx: _idx
                    )
                try:
                    resp = _run_pipeline(req)
                finally:
                    _pps_module.predict_index = orig_predict_index

                # Margin boost happens under the same lock — it uses the
                # GPU via the shared model.
                if desired_target_idx is not None:
                    boosted_b64, margin_info = self._boost_margin_if_low(
                        adv_b64=resp.perturbed_image_b64,
                        clean_b64=synapse.clean_image_b64,
                        true_idx=desired_target_idx,
                        epsilon=float(synapse.epsilon),
                    )
                    synapse.perturbed_image_b64 = boosted_b64
                else:
                    margin_info = "skipped(no_true_label)"
                    synapse.perturbed_image_b64 = resp.perturbed_image_b64

                # Re-score the shipped adversarial via the validator-shaped
                # `verify_and_score`. The pipeline's `resp.best_result.score`
                # reflects the pre-boost adversarial; the validator sees the
                # post-boost one (boost adds pixels → higher rmse → slightly
                # lower score). Logging both makes regressions diagnosable.
                pipeline_score = float(resp.best_result.get("score", 0.0))
                shipped_score: float
                shipped_reason: str
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
                f"elapsed={elapsed_ms}ms"
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
        except Exception:
            elapsed_ms = int((time.time() - t_start) * 1000)
            logger.exception(
                f"task={task_id} pipeline raised unexpectedly "
                f"elapsed={elapsed_ms}ms; returning clean"
            )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
        return synapse

    def _boost_margin_if_low(
        self,
        adv_b64: str,
        clean_b64: str,
        true_idx: int,
        epsilon: float,
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

        # Bound the boost's L∞ to exactly the σ-zero magnitude so the
        # perturbation never exceeds `1/255` in any single pixel (preserves
        # linf_score ≈ 0.933). The validator's gate at `min(epsilon, 0.03)`
        # is a much looser ceiling that we never approach.
        safe_eps = min(MARGIN_BOOST_LINF_BUDGET, min(float(epsilon), VALIDATOR_MAX_LINF))
        if safe_eps <= 0.0:
            return adv_b64, f"{margin:.3f}(eps_too_small)"

        topk = max(1, min(MARGIN_BOOST_TOPK_PIXELS, adv.numel()))
        clean_flat = clean.view(-1)
        for step_i in range(MARGIN_BOOST_MAX_STEPS):
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
                    f"{margin:.3f}(boosted@{step_i+1})",
                )
            # If the runner-up shifted (boundary moved), retarget.
            runner_up = new_runner

        # Boost didn't reach the threshold within the step budget. Ship the
        # best adv we have — still better than the original if margin grew.
        return (
            encode_image_b64(adv.detach().cpu()),
            f"{margin:.3f}(partial)",
        )

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
