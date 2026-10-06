"""
ML Engine for AutoScale IQ — Surge Prediction Pipeline.

Loads the locked 5M-row HistGradientBoostingClassifier + Isotonic calibrator
and computes surge probability from a rolling window of invocation observations.

Pipeline:
    Rolling window → Feature generation → Model inference →
    Isotonic calibration → Risk classification (NORMAL / ELEVATED / CRITICAL)

This module is READ-ONLY with respect to the safety controller.
It never directly executes optimizations.

Locked artifacts:
    model:       ml/artifacts/model_5m_config6.joblib
    calibrator:  ml/artifacts/calibrator_5m_config6_isotonic.joblib
    thresholds:  ml/artifacts/thresholds_5m_config6.json
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import joblib
import numpy as np

from backend.schemas import RiskSignal

logger = logging.getLogger("autoscale_iq.ml_engine")

# ---------------------------------------------------------------------------
# Locked feature column order (must match training schema exactly)
# Source: ml/data/processed/train/day_01.parquet column order
# ---------------------------------------------------------------------------
_FEATURE_NAMES: List[str] = [
    # Workload (1–6)
    "invocations_t",
    "rolling_mean_5m",
    "rolling_mean_15m",
    "rolling_mean_60m",
    "rolling_max_15m",
    "rolling_max_60m",
    # Trend (7–8)
    "rate_delta_1m",
    "rate_delta_5m",
    # Volatility (9)
    "rolling_std_15m",
    # Time (10–12)
    "minute_of_day",
    "minute_sin",
    "minute_cos",
    # Trigger one-hot (13–19) — exactly 7 categories from training
    "trigger_timer",
    "trigger_http",
    "trigger_queue",
    "trigger_orchestration",
    "trigger_event",
    "trigger_storage",
    "trigger_others",
]

# Number of 1-minute observations needed to fully populate all rolling windows
# (60-minute max window + 1 for current)
_MIN_WINDOW_SIZE = 61
_MAX_WINDOW_SIZE = 61  # keep only what we need

# Rolling window sizes (in 1-minute ticks)
_WIN_5 = 5
_WIN_15 = 15
_WIN_60 = 60


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ObservationTick:
    """
    A single 1-minute metric observation contributed to the feature window.

    Only fields that are part of the locked 19-feature training schema are
    included.  Resource/latency metrics (CPU %, memory %, response time,
    DB query time) are NOT model inputs and should not be set here.
    """
    invocations: float              # requests/invocations in this minute
    # Trigger one-hot flags — exactly 7 categories from training
    trigger_timer: bool = False
    trigger_http: bool = True
    trigger_queue: bool = False
    trigger_orchestration: bool = False
    trigger_event: bool = False
    trigger_storage: bool = False
    trigger_others: bool = False


@dataclass
class MLPredictionResult:
    """Structured result from a single ML inference call."""
    surge_probability: float           # Calibrated surge probability [0, 1]
    raw_probability: float             # Uncalibrated model output
    risk_signal: RiskSignal            # Derived risk classification
    watch_threshold: float             # tau_watch used for classification
    critical_threshold: float          # tau_crit used for classification
    feature_vector: List[float]        # 19 features used for this prediction
    window_size: int                   # Number of ticks in the window
    insufficient_data: bool            # True if window < _MIN_WINDOW_SIZE
    inference_time_ms: float           # Inference latency


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _compute_features(
    window: Deque[ObservationTick],
    current_tick: ObservationTick,
    minute_of_day: int,
) -> np.ndarray:
    """
    Compute the locked 19-dimensional feature vector from the rolling window.

    Feature order matches the training parquet schema exactly:
        1–6   workload rolling stats
        7–8   rate deltas
        9     rolling std
        10–12 time encoding
        13–19 trigger one-hot (7 categories)

    The window contains previous observations (oldest at left).
    current_tick is the observation being predicted for.
    """
    # Build list with current tick at end
    ticks = list(window) + [current_tick]
    invocations_seq = np.array([t.invocations for t in ticks], dtype=np.float64)

    n = len(invocations_seq)
    inv_t = invocations_seq[-1]

    # Bounded slices (handle shorter windows gracefully)
    last_5 = invocations_seq[max(0, n - _WIN_5):]
    last_15 = invocations_seq[max(0, n - _WIN_15):]
    last_60 = invocations_seq[max(0, n - _WIN_60):]

    rolling_mean_5 = float(np.mean(last_5)) if len(last_5) > 0 else 0.0
    rolling_mean_15 = float(np.mean(last_15)) if len(last_15) > 0 else 0.0
    rolling_mean_60 = float(np.mean(last_60)) if len(last_60) > 0 else 0.0
    rolling_max_15 = float(np.max(last_15)) if len(last_15) > 0 else 0.0
    rolling_max_60 = float(np.max(last_60)) if len(last_60) > 0 else 0.0
    rolling_std_15 = float(np.std(last_15)) if len(last_15) > 1 else 0.0

    # Rate deltas: difference in invocations over 1-min and 5-min windows
    rate_delta_1m = float(invocations_seq[-1] - invocations_seq[-2]) if n >= 2 else 0.0
    rate_delta_5m = (
        float(invocations_seq[-1] - invocations_seq[-6]) if n >= 6
        else float(invocations_seq[-1] - invocations_seq[0]) if n >= 2
        else 0.0
    )

    # Time features
    minute_sin = math.sin(2 * math.pi * minute_of_day / 1440)
    minute_cos = math.cos(2 * math.pi * minute_of_day / 1440)

    # Trigger one-hot (7 categories — locked training schema features 13–19)
    t = current_tick
    return np.array([
        # 1–6  workload
        inv_t,
        rolling_mean_5,
        rolling_mean_15,
        rolling_mean_60,
        rolling_max_15,
        rolling_max_60,
        # 7–8  trend
        rate_delta_1m,
        rate_delta_5m,
        # 9    volatility
        rolling_std_15,
        # 10–12 time
        float(minute_of_day),
        minute_sin,
        minute_cos,
        # 13–19 trigger one-hot
        1.0 if t.trigger_timer else 0.0,
        1.0 if t.trigger_http else 0.0,
        1.0 if t.trigger_queue else 0.0,
        1.0 if t.trigger_orchestration else 0.0,
        1.0 if t.trigger_event else 0.0,
        1.0 if t.trigger_storage else 0.0,
        1.0 if t.trigger_others else 0.0,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Surge Predictor
# ---------------------------------------------------------------------------


class SurgePredictor:
    """
    Singleton ML inference engine for surge prediction.

    Loads the locked model + calibrator once at startup and provides
    thread-safe probability inference from a rolling observation window.

    Usage:
        predictor = SurgePredictor.get_instance()
        result = predictor.predict(current_tick, minute_of_day)
    """

    _instance: Optional["SurgePredictor"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: Path,
        calibrator_path: Path,
        tau_watch: float,
        tau_crit: float,
        max_window: int = _MAX_WINDOW_SIZE,
    ):
        logger.info("Loading ML model from %s", model_path)
        model_artifact = joblib.load(model_path)
        # Artifacts are stored as dict wrappers; unwrap to the actual estimator
        if isinstance(model_artifact, dict):
            self._model = model_artifact["model"]
        else:
            self._model = model_artifact
        if not hasattr(self._model, "predict_proba"):
            raise ValueError(
                f"Loaded model object has no predict_proba — "
                f"got {type(self._model)}. Check artifact format."
            )

        logger.info("Loading calibrator from %s", calibrator_path)
        cal_artifact = joblib.load(calibrator_path)
        if isinstance(cal_artifact, dict):
            self._calibrator = cal_artifact["calibrator"]
        else:
            self._calibrator = cal_artifact

        self._tau_watch = tau_watch
        self._tau_crit = tau_crit
        self._max_window = max_window
        self._window: Deque[ObservationTick] = deque(maxlen=max_window)
        self._inference_lock = threading.Lock()

        logger.info(
            "SurgePredictor ready — tau_watch=%.3f, tau_crit=%.3f",
            tau_watch,
            tau_crit,
        )

    @classmethod
    def get_instance(cls) -> "SurgePredictor":
        """
        Return the singleton instance. Must call initialize() first.
        Raises RuntimeError if not yet initialized.
        """
        if cls._instance is None:
            raise RuntimeError(
                "SurgePredictor not initialized. Call SurgePredictor.initialize() first."
            )
        return cls._instance

    @classmethod
    def initialize(
        cls,
        model_path: Path,
        calibrator_path: Path,
        thresholds_path: Path,
    ) -> "SurgePredictor":
        """
        Create (or return existing) singleton instance.
        Thread-safe.
        """
        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            with open(thresholds_path) as f:
                thresholds = json.load(f)

            tau_watch = float(thresholds["watch_threshold"])
            tau_crit = float(thresholds["critical_threshold"])

            cls._instance = cls(
                model_path=model_path,
                calibrator_path=calibrator_path,
                tau_watch=tau_watch,
                tau_crit=tau_crit,
            )
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for testing only."""
        with cls._lock:
            cls._instance = None

    def push_tick(self, tick: ObservationTick) -> None:
        """Add an observation to the rolling window (without running inference)."""
        with self._inference_lock:
            self._window.append(tick)

    def predict(
        self,
        current_tick: ObservationTick,
        minute_of_day: int,
    ) -> MLPredictionResult:
        """
        Run full feature extraction + model inference + calibration.

        The current_tick is used for features but NOT pushed into the rolling window.
        Call push_tick() separately if you want it retained for future predictions.
        """
        t0 = time.perf_counter()

        with self._inference_lock:
            window_snapshot = list(self._window)

        window_size = len(window_snapshot) + 1  # include current tick
        insufficient = window_size < _MIN_WINDOW_SIZE

        if insufficient:
            logger.debug(
                "Insufficient window (%d/%d ticks) — defaulting to NORMAL",
                window_size,
                _MIN_WINDOW_SIZE,
            )

        # Compute features (always — even with short window, graceful degradation)
        feature_vec = _compute_features(
            deque(window_snapshot, maxlen=_MAX_WINDOW_SIZE),
            current_tick,
            minute_of_day,
        )

        X = feature_vec.reshape(1, -1)

        # Raw model probability (positive class)
        raw_prob = float(self._model.predict_proba(X)[0, 1])

        # Isotonic calibration — IsotonicCalibrator.predict_proba returns 1-d array
        cal_output = self._calibrator.predict_proba(np.array([raw_prob]))
        calibrated_prob = float(np.asarray(cal_output).flat[0])
        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))


        # Risk classification using locked thresholds
        if calibrated_prob >= self._tau_crit:
            risk = RiskSignal.CRITICAL
        elif calibrated_prob >= self._tau_watch:
            risk = RiskSignal.ELEVATED
        else:
            risk = RiskSignal.NORMAL

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return MLPredictionResult(
            surge_probability=calibrated_prob,
            raw_probability=raw_prob,
            risk_signal=risk,
            watch_threshold=self._tau_watch,
            critical_threshold=self._tau_crit,
            feature_vector=feature_vec.tolist(),
            window_size=window_size,
            insufficient_data=insufficient,
            inference_time_ms=round(inference_time_ms, 3),
        )

    @property
    def window_length(self) -> int:
        """Current number of retained ticks in the rolling window."""
        return len(self._window)

    @property
    def thresholds(self) -> Tuple[float, float]:
        """Return (tau_watch, tau_crit) tuple."""
        return self._tau_watch, self._tau_crit
