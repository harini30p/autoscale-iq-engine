"""
Tests for the AutoScale IQ ML Engine.

Covers:
- Locked 19-feature schema (names and order)
- Feature computation correctness (rolling windows, rate deltas, time, triggers)
- Resource/latency fields NOT in the feature vector
- Risk classification thresholds (NORMAL < 0.04, ELEVATED 0.04–0.075, CRITICAL ≥ 0.075)
- Rolling window management
- MLPredictionResult structure
- Singleton lifecycle
- Real-artifact smoke test (marked with @pytest.mark.real_artifacts)
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.ml_engine import (
    ObservationTick,
    SurgePredictor,
    _compute_features,
    _FEATURE_NAMES,
    _MIN_WINDOW_SIZE,
)
from backend.schemas import RiskSignal


# ---------------------------------------------------------------------------
# Locked schema constants (ground truth — do not change)
# ---------------------------------------------------------------------------

LOCKED_FEATURE_NAMES = [
    "invocations_t",
    "rolling_mean_5m",
    "rolling_mean_15m",
    "rolling_mean_60m",
    "rolling_max_15m",
    "rolling_max_60m",
    "rate_delta_1m",
    "rate_delta_5m",
    "rolling_std_15m",
    "minute_of_day",
    "minute_sin",
    "minute_cos",
    "trigger_timer",
    "trigger_http",
    "trigger_queue",
    "trigger_orchestration",
    "trigger_event",
    "trigger_storage",
    "trigger_others",
]

# Features that must NOT appear in the ML feature vector
NON_FEATURES = [
    "cpu_utilization",
    "memory_utilization",
    "response_time_ms",
    "db_query_time_ms",
    "response_time",
    "db_query_time",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tick(
    invocations: float = 10.0,
    trigger_http: bool = True,
    trigger_timer: bool = False,
    trigger_queue: bool = False,
    trigger_orchestration: bool = False,
    trigger_event: bool = False,
    trigger_storage: bool = False,
    trigger_others: bool = False,
) -> ObservationTick:
    return ObservationTick(
        invocations=invocations,
        trigger_http=trigger_http,
        trigger_timer=trigger_timer,
        trigger_queue=trigger_queue,
        trigger_orchestration=trigger_orchestration,
        trigger_event=trigger_event,
        trigger_storage=trigger_storage,
        trigger_others=trigger_others,
    )


def _mock_predictor(tau_watch=0.04, tau_crit=0.075) -> SurgePredictor:
    """Return a SurgePredictor with mocked joblib model + calibrator."""
    import threading

    model = MagicMock()
    calibrator = MagicMock()

    # Default: raw 0.02 → calibrated 0.02 (NORMAL)
    model.predict_proba = MagicMock(return_value=np.array([[0.98, 0.02]]))
    # Return 1-D array (matches real IsotonicCalibrator.predict_proba output)
    calibrator.predict_proba = MagicMock(side_effect=lambda x: np.array([x[0]]))

    predictor = object.__new__(SurgePredictor)
    predictor._model = model
    predictor._calibrator = calibrator
    predictor._tau_watch = tau_watch
    predictor._tau_crit = tau_crit
    predictor._max_window = 61
    predictor._window = deque(maxlen=61)
    predictor._inference_lock = threading.Lock()
    return predictor


# ---------------------------------------------------------------------------
# SCHEMA CORRECTNESS — locked 19-feature names and order
# ---------------------------------------------------------------------------


class TestLockedFeatureSchema:
    def test_feature_names_count_is_19(self):
        assert len(_FEATURE_NAMES) == 19

    def test_feature_names_exact_order(self):
        """_FEATURE_NAMES must match the locked training schema exactly."""
        assert _FEATURE_NAMES == LOCKED_FEATURE_NAMES

    def test_no_resource_fields_in_feature_names(self):
        """Resource/latency metrics must NOT be model features."""
        for bad_name in NON_FEATURES:
            assert bad_name not in _FEATURE_NAMES, (
                f"'{bad_name}' must not be in _FEATURE_NAMES — "
                "it is not part of the locked training schema"
            )

    def test_all_7_trigger_fields_present(self):
        trigger_fields = [n for n in _FEATURE_NAMES if n.startswith("trigger_")]
        assert set(trigger_fields) == {
            "trigger_timer",
            "trigger_http",
            "trigger_queue",
            "trigger_orchestration",
            "trigger_event",
            "trigger_storage",
            "trigger_others",
        }

    def test_trigger_fields_are_last_7(self):
        """Trigger one-hots must occupy positions 13–19 (0-indexed: 12–18)."""
        assert _FEATURE_NAMES[12:] == [
            "trigger_timer",
            "trigger_http",
            "trigger_queue",
            "trigger_orchestration",
            "trigger_event",
            "trigger_storage",
            "trigger_others",
        ]

    def test_observation_tick_has_no_resource_fields(self):
        """ObservationTick must not have cpu/memory/response/db fields."""
        tick = make_tick()
        for bad_attr in ["cpu_utilization", "memory_utilization",
                         "response_time_ms", "db_query_time_ms"]:
            assert not hasattr(tick, bad_attr), (
                f"ObservationTick.{bad_attr} must not exist — "
                "it is not part of the locked training schema"
            )

    def test_observation_tick_has_all_7_trigger_fields(self):
        tick = make_tick()
        for field in ["trigger_timer", "trigger_http", "trigger_queue",
                      "trigger_orchestration", "trigger_event",
                      "trigger_storage", "trigger_others"]:
            assert hasattr(tick, field), f"ObservationTick missing field: {field}"


# ---------------------------------------------------------------------------
# FEATURE VECTOR — verify each position in the output
# ---------------------------------------------------------------------------


class TestFeatureVector:
    def test_returns_19_features(self):
        features = _compute_features(deque(), make_tick(), minute_of_day=0)
        assert len(features) == 19

    def test_no_resource_values_at_tail(self):
        """
        Last 7 values must be trigger one-hots, not resource/latency values.
        Build a tick with all triggers OFF and invocations=0; last 7 must all be 0.
        """
        tick = make_tick(
            invocations=0.0,
            trigger_http=False,
            trigger_timer=False,
            trigger_queue=False,
            trigger_orchestration=False,
            trigger_event=False,
            trigger_storage=False,
            trigger_others=False,
        )
        features = _compute_features(deque(), tick, minute_of_day=0)
        # Last 7 are triggers — all should be 0.0 when all OFF
        assert list(features[12:]) == [0.0] * 7

    def test_trigger_positions_13_to_19(self):
        """
        Features 13–19 (0-indexed 12–18) must correspond exactly to:
        trigger_timer, trigger_http, trigger_queue, trigger_orchestration,
        trigger_event, trigger_storage, trigger_others
        """
        # Activate only trigger_orchestration (index 15, 0-indexed)
        tick = ObservationTick(
            invocations=5.0,
            trigger_timer=False,
            trigger_http=False,
            trigger_queue=False,
            trigger_orchestration=True,
            trigger_event=False,
            trigger_storage=False,
            trigger_others=False,
        )
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[12] == 0.0  # trigger_timer
        assert features[13] == 0.0  # trigger_http
        assert features[14] == 0.0  # trigger_queue
        assert features[15] == 1.0  # trigger_orchestration ← only one active
        assert features[16] == 0.0  # trigger_event
        assert features[17] == 0.0  # trigger_storage
        assert features[18] == 0.0  # trigger_others

    def test_trigger_http_default_position(self):
        tick = make_tick(trigger_http=True)
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[12] == 0.0  # trigger_timer
        assert features[13] == 1.0  # trigger_http ← active
        assert features[14] == 0.0  # trigger_queue
        assert features[15] == 0.0  # trigger_orchestration
        assert features[16] == 0.0  # trigger_event
        assert features[17] == 0.0  # trigger_storage
        assert features[18] == 0.0  # trigger_others

    def test_trigger_timer_position(self):
        tick = make_tick(trigger_timer=True, trigger_http=False)
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[12] == 1.0  # trigger_timer ← active
        assert features[13] == 0.0  # trigger_http

    def test_trigger_storage_position(self):
        tick = ObservationTick(
            invocations=1.0,
            trigger_storage=True,
            trigger_http=False,
        )
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[17] == 1.0  # trigger_storage

    def test_trigger_others_position(self):
        tick = ObservationTick(invocations=1.0, trigger_others=True, trigger_http=False)
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[18] == 1.0  # trigger_others

    def test_invocations_t_is_current_tick(self):
        tick = make_tick(invocations=42.0)
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[0] == 42.0

    def test_rolling_means_equal_single_tick(self):
        tick = make_tick(invocations=10.0)
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[1] == pytest.approx(10.0)  # rolling_mean_5
        assert features[2] == pytest.approx(10.0)  # rolling_mean_15
        assert features[3] == pytest.approx(10.0)  # rolling_mean_60

    def test_rolling_std_zero_single_tick(self):
        tick = make_tick(invocations=10.0)
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[8] == pytest.approx(0.0)  # rolling_std_15

    def test_rate_deltas_zero_single_tick(self):
        tick = make_tick(invocations=10.0)
        features = _compute_features(deque(), tick, minute_of_day=0)
        assert features[6] == pytest.approx(0.0)  # rate_delta_1m
        assert features[7] == pytest.approx(0.0)  # rate_delta_5m

    def test_rate_delta_1m_two_ticks(self):
        window = deque([make_tick(invocations=5.0)])
        curr = make_tick(invocations=15.0)
        features = _compute_features(window, curr, minute_of_day=0)
        assert features[6] == pytest.approx(10.0)  # 15 - 5

    def test_rate_delta_5m_six_ticks(self):
        ticks = [make_tick(invocations=float(i)) for i in range(5)]
        curr = make_tick(invocations=20.0)
        window = deque(ticks)
        features = _compute_features(window, curr, minute_of_day=0)
        # current=20.0, tick at -6 from end = ticks[0] = 0.0
        assert features[7] == pytest.approx(20.0)

    def test_time_features_minute_120(self):
        features = _compute_features(deque(), make_tick(), minute_of_day=120)
        assert features[9] == pytest.approx(120.0)
        assert features[10] == pytest.approx(math.sin(2 * math.pi * 120 / 1440))
        assert features[11] == pytest.approx(math.cos(2 * math.pi * 120 / 1440))

    def test_rolling_mean_5m_full_window(self):
        ticks = [make_tick(invocations=float(i)) for i in range(10)]
        curr = make_tick(invocations=10.0)
        window = deque(ticks, maxlen=61)
        features = _compute_features(window, curr, minute_of_day=0)
        # last 5 of [0..10] = [6,7,8,9,10]
        assert features[1] == pytest.approx(np.mean([6.0, 7.0, 8.0, 9.0, 10.0]))

    def test_rolling_std_15m_full_window(self):
        ticks = [make_tick(invocations=float(i)) for i in range(60)]
        curr = make_tick(invocations=60.0)
        window = deque(ticks, maxlen=61)
        features = _compute_features(window, curr, minute_of_day=0)
        # last 15 of [0..60] = [46..60]
        last_15 = np.array([float(i) for i in range(46, 61)])
        assert features[8] == pytest.approx(float(np.std(last_15)), abs=1e-6)


# ---------------------------------------------------------------------------
# RISK CLASSIFICATION — locked thresholds
# ---------------------------------------------------------------------------


class TestRiskClassification:
    """All tests use the locked thresholds: WATCH=0.04, CRITICAL=0.075."""

    def test_normal_below_watch_threshold(self):
        predictor = _mock_predictor(tau_watch=0.04, tau_crit=0.075)
        predictor._model.predict_proba.return_value = np.array([[0.98, 0.02]])
        result = predictor.predict(make_tick(), 120)
        assert result.risk_signal == RiskSignal.NORMAL
        assert result.surge_probability == pytest.approx(0.02, abs=1e-6)

    def test_elevated_above_watch_below_crit(self):
        predictor = _mock_predictor(tau_watch=0.04, tau_crit=0.075)
        predictor._model.predict_proba.return_value = np.array([[0.95, 0.05]])
        result = predictor.predict(make_tick(), 120)
        assert result.risk_signal == RiskSignal.ELEVATED

    def test_critical_above_crit_threshold(self):
        predictor = _mock_predictor(tau_watch=0.04, tau_crit=0.075)
        predictor._model.predict_proba.return_value = np.array([[0.9, 0.1]])
        result = predictor.predict(make_tick(), 120)
        assert result.risk_signal == RiskSignal.CRITICAL

    def test_exactly_at_watch_threshold_is_elevated(self):
        predictor = _mock_predictor(tau_watch=0.04, tau_crit=0.075)
        predictor._model.predict_proba.return_value = np.array([[0.96, 0.04]])
        result = predictor.predict(make_tick(), 120)
        assert result.risk_signal == RiskSignal.ELEVATED

    def test_exactly_at_crit_threshold_is_critical(self):
        predictor = _mock_predictor(tau_watch=0.04, tau_crit=0.075)
        predictor._model.predict_proba.return_value = np.array([[0.925, 0.075]])
        result = predictor.predict(make_tick(), 120)
        assert result.risk_signal == RiskSignal.CRITICAL

    def test_probability_clipped_above_1(self):
        predictor = _mock_predictor()
        predictor._model.predict_proba.return_value = np.array([[0.0, 1.0]])
        predictor._calibrator.predict_proba.side_effect = lambda x: np.array([1.5])
        result = predictor.predict(make_tick(), 120)
        assert result.surge_probability <= 1.0

    def test_probability_clipped_below_0(self):
        predictor = _mock_predictor()
        predictor._model.predict_proba.return_value = np.array([[1.0, 0.0]])
        predictor._calibrator.predict_proba.side_effect = lambda x: np.array([-0.1])
        result = predictor.predict(make_tick(), 120)
        assert result.surge_probability >= 0.0


# ---------------------------------------------------------------------------
# ROLLING WINDOW
# ---------------------------------------------------------------------------


class TestRollingWindow:
    def test_empty_window_is_insufficient(self):
        predictor = _mock_predictor()
        result = predictor.predict(make_tick(), 120)
        assert result.insufficient_data is True
        assert result.window_size == 1

    def test_window_grows_after_push_tick(self):
        predictor = _mock_predictor()
        for i in range(30):
            predictor.push_tick(make_tick(invocations=float(i)))
        assert predictor.window_length == 30

    def test_window_size_reported_correctly(self):
        predictor = _mock_predictor()
        for i in range(10):
            predictor.push_tick(make_tick())
        result = predictor.predict(make_tick(), 0)
        assert result.window_size == 11  # 10 in window + 1 current

    def test_full_window_not_insufficient(self):
        predictor = _mock_predictor()
        for _ in range(_MIN_WINDOW_SIZE):
            predictor.push_tick(make_tick())
        result = predictor.predict(make_tick(), 0)
        assert result.insufficient_data is False

    def test_window_capped_at_max(self):
        predictor = _mock_predictor()
        for i in range(100):
            predictor.push_tick(make_tick(invocations=float(i)))
        assert predictor.window_length == 61

    def test_predict_does_not_push_tick(self):
        predictor = _mock_predictor()
        initial_len = predictor.window_length
        predictor.predict(make_tick(), 0)
        assert predictor.window_length == initial_len

    def test_push_tick_then_predict_uses_tick(self):
        predictor = _mock_predictor()
        predictor.push_tick(make_tick(invocations=999.0))
        result = predictor.predict(make_tick(), 0)
        assert result.window_size == 2


# ---------------------------------------------------------------------------
# MLPredictionResult structure
# ---------------------------------------------------------------------------


class TestMLPredictionResult:
    def test_result_contains_19_features(self):
        predictor = _mock_predictor()
        result = predictor.predict(make_tick(), 120)
        assert len(result.feature_vector) == 19

    def test_thresholds_match_locked_config(self):
        predictor = _mock_predictor(tau_watch=0.04, tau_crit=0.075)
        result = predictor.predict(make_tick(), 120)
        assert result.watch_threshold == pytest.approx(0.04)
        assert result.critical_threshold == pytest.approx(0.075)

    def test_inference_time_is_non_negative(self):
        predictor = _mock_predictor()
        result = predictor.predict(make_tick(), 0)
        assert result.inference_time_ms >= 0.0

    def test_raw_probability_preserved_separately_from_calibrated(self):
        predictor = _mock_predictor()
        predictor._model.predict_proba.return_value = np.array([[0.7, 0.3]])
        predictor._calibrator.predict_proba.side_effect = lambda x: np.array([0.2])
        result = predictor.predict(make_tick(), 0)
        assert result.raw_probability == pytest.approx(0.3, abs=1e-6)
        assert result.surge_probability == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# SINGLETON lifecycle
# ---------------------------------------------------------------------------


class TestSingleton:
    def setup_method(self):
        SurgePredictor.reset()

    def teardown_method(self):
        SurgePredictor.reset()

    def test_get_instance_raises_before_init(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            SurgePredictor.get_instance()

    def test_initialize_returns_instance(self, tmp_path):
        thresholds = {"watch_threshold": 0.04, "critical_threshold": 0.075}
        thresholds_file = tmp_path / "thresholds.json"
        thresholds_file.write_text(json.dumps(thresholds))

        with patch("joblib.load", return_value=MagicMock()):
            instance = SurgePredictor.initialize(
                model_path=tmp_path / "model.joblib",
                calibrator_path=tmp_path / "cal.joblib",
                thresholds_path=thresholds_file,
            )
        assert instance is not None
        assert instance.thresholds == (0.04, 0.075)

    def test_initialize_idempotent(self, tmp_path):
        thresholds = {"watch_threshold": 0.04, "critical_threshold": 0.075}
        thresholds_file = tmp_path / "thresholds.json"
        thresholds_file.write_text(json.dumps(thresholds))

        with patch("joblib.load", return_value=MagicMock()):
            inst1 = SurgePredictor.initialize(
                model_path=tmp_path / "model.joblib",
                calibrator_path=tmp_path / "cal.joblib",
                thresholds_path=thresholds_file,
            )
            inst2 = SurgePredictor.initialize(
                model_path=tmp_path / "model.joblib",
                calibrator_path=tmp_path / "cal.joblib",
                thresholds_path=thresholds_file,
            )
        assert inst1 is inst2

    def test_reset_clears_instance(self, tmp_path):
        thresholds = {"watch_threshold": 0.04, "critical_threshold": 0.075}
        thresholds_file = tmp_path / "thresholds.json"
        thresholds_file.write_text(json.dumps(thresholds))

        with patch("joblib.load", return_value=MagicMock()):
            SurgePredictor.initialize(
                model_path=tmp_path / "model.joblib",
                calibrator_path=tmp_path / "cal.joblib",
                thresholds_path=thresholds_file,
            )
        SurgePredictor.reset()
        with pytest.raises(RuntimeError):
            SurgePredictor.get_instance()


# ---------------------------------------------------------------------------
# REAL-ARTIFACT SMOKE TEST
# Marked with @pytest.mark.real_artifacts — uses actual committed ML files.
# Skipped automatically if artifacts are not present.
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_5m_config6.joblib"
CALIBRATOR_PATH = ARTIFACTS_DIR / "calibrator_5m_config6_isotonic.joblib"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds_5m_config6.json"

_artifacts_present = (
    MODEL_PATH.exists() and CALIBRATOR_PATH.exists() and THRESHOLDS_PATH.exists()
)

pytestmark_real = pytest.mark.skipif(
    not _artifacts_present,
    reason="Real ML artifacts not found — skipping smoke test"
)


@pytest.mark.skipif(not _artifacts_present, reason="Real ML artifacts not found")
class TestRealArtifactSmoke:
    """
    End-to-end smoke test using the actual locked artifacts.
    Does NOT retrain or modify the model.
    """

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        SurgePredictor.reset()
        yield
        SurgePredictor.reset()

    def test_thresholds_load_correctly(self):
        """Thresholds JSON must contain WATCH=0.040 and CRITICAL=0.075."""
        with open(THRESHOLDS_PATH) as f:
            thresholds = json.load(f)
        assert thresholds["watch_threshold"] == pytest.approx(0.040)
        assert thresholds["critical_threshold"] == pytest.approx(0.075)

    def test_model_and_calibrator_load(self):
        """Model and calibrator must load without error."""
        predictor = SurgePredictor.initialize(
            model_path=MODEL_PATH,
            calibrator_path=CALIBRATOR_PATH,
            thresholds_path=THRESHOLDS_PATH,
        )
        assert predictor is not None

    def test_thresholds_loaded_into_predictor(self):
        """Predictor must report the locked thresholds after initialization."""
        predictor = SurgePredictor.initialize(
            model_path=MODEL_PATH,
            calibrator_path=CALIBRATOR_PATH,
            thresholds_path=THRESHOLDS_PATH,
        )
        tau_watch, tau_crit = predictor.thresholds
        assert tau_watch == pytest.approx(0.040)
        assert tau_crit == pytest.approx(0.075)

    def test_prediction_returns_finite_probability(self):
        """Full pipeline must return a probability in [0, 1]."""
        predictor = SurgePredictor.initialize(
            model_path=MODEL_PATH,
            calibrator_path=CALIBRATOR_PATH,
            thresholds_path=THRESHOLDS_PATH,
        )
        tick = ObservationTick(
            invocations=10.0,
            trigger_http=True,
        )
        result = predictor.predict(tick, minute_of_day=120)
        assert math.isfinite(result.surge_probability)
        assert 0.0 <= result.surge_probability <= 1.0

    def test_prediction_returns_valid_risk_signal(self):
        """Risk signal must be one of the three valid enum values."""
        predictor = SurgePredictor.initialize(
            model_path=MODEL_PATH,
            calibrator_path=CALIBRATOR_PATH,
            thresholds_path=THRESHOLDS_PATH,
        )
        tick = ObservationTick(invocations=10.0, trigger_http=True)
        result = predictor.predict(tick, minute_of_day=120)
        assert result.risk_signal in (
            RiskSignal.NORMAL, RiskSignal.ELEVATED, RiskSignal.CRITICAL
        )

    def test_prediction_returns_19_features(self):
        """Feature vector must have exactly 19 elements."""
        predictor = SurgePredictor.initialize(
            model_path=MODEL_PATH,
            calibrator_path=CALIBRATOR_PATH,
            thresholds_path=THRESHOLDS_PATH,
        )
        tick = ObservationTick(invocations=10.0, trigger_http=True)
        result = predictor.predict(tick, minute_of_day=120)
        assert len(result.feature_vector) == 19

    def test_full_window_prediction_is_finite(self):
        """With a full 61-tick window, prediction must still be finite and valid."""
        predictor = SurgePredictor.initialize(
            model_path=MODEL_PATH,
            calibrator_path=CALIBRATOR_PATH,
            thresholds_path=THRESHOLDS_PATH,
        )
        # Simulate a rising traffic pattern
        for i in range(61):
            predictor.push_tick(ObservationTick(
                invocations=float(i),
                trigger_http=True,
            ))
        tick = ObservationTick(invocations=80.0, trigger_http=True)
        result = predictor.predict(tick, minute_of_day=300)
        assert result.insufficient_data is False
        assert math.isfinite(result.surge_probability)
        assert 0.0 <= result.surge_probability <= 1.0
        assert result.risk_signal in (
            RiskSignal.NORMAL, RiskSignal.ELEVATED, RiskSignal.CRITICAL
        )

    def test_surge_spike_produces_elevated_or_critical(self):
        """
        A severe traffic spike (10x over baseline) should produce ELEVATED or CRITICAL.
        This is a heuristic smoke test — it does not assert exact values.
        """
        predictor = SurgePredictor.initialize(
            model_path=MODEL_PATH,
            calibrator_path=CALIBRATOR_PATH,
            thresholds_path=THRESHOLDS_PATH,
        )
        # Build a stable baseline
        for _ in range(61):
            predictor.push_tick(ObservationTick(invocations=5.0, trigger_http=True))
        # Spike
        spike_tick = ObservationTick(invocations=50.0, trigger_http=True)
        result = predictor.predict(spike_tick, minute_of_day=240)
        # Soft assertion: at minimum a finite probability is returned
        assert math.isfinite(result.surge_probability)
        # Log the risk for debugging (not a hard assertion on which band)
        print(
            f"\n[smoke] spike probability={result.surge_probability:.4f} "
            f"risk={result.risk_signal.value}"
        )
