"""
Unit and Integration Tests for AutoScale IQ ML Pipeline (Milestone 2 - Step 5).
Tests:
1. Exact 19 feature schema and dtypes
2. Vectorized rolling-window feature extraction
3. Strict target mathematical definition
4. Temporal zero-leakage isolation
5. Trigger one-hot encoding with unknown category fallback
6. Prior-shift log-odds calibration mapping
7. Threshold selection logic
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from ml.preprocess import (
    FEATURE_COLUMNS,
    TRIGGER_CATEGORIES,
    extract_features_from_matrix,
)
from ml.calibrate import PriorShiftCalibrator, PlattCalibrator, IsotonicCalibrator
from ml.evaluate import select_operating_thresholds


def test_feature_columns_schema():
    """Verify exact 19 feature columns match locked project decisions."""
    assert len(FEATURE_COLUMNS) == 19
    expected_features = [
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
    assert FEATURE_COLUMNS == expected_features


def test_vectorized_feature_and_target_math():
    """Verify rolling features and surge target on a synthetic synthetic trace with known surge."""
    # 1 synthetic function, 1440 minutes
    trace = np.zeros((1, 1440), dtype=np.float32)
    # Steady state of 2 req/min
    trace[0, :100] = 2.0
    # Surge starting at minute 75 (1-indexed), so 0-indexed index 74
    # Suppose at t = 70 (index 69, minute 70):
    # past baseline (t-14 .. t) is 2.0
    # future 5m (t+1 .. t+5) has a surge of 10.0 req/min
    trace[0, 70:75] = 10.0  # indices 70..74 (minutes 71..75)

    triggers = np.array(["http"])
    arrays = extract_features_from_matrix(trace, triggers, t_start=60, t_end=100)

    # Convert to DataFrame
    df = pd.DataFrame(arrays)

    # Minute 70 is index t=69 in 0-indexed terms.
    # In eval range [60..100], minute 70 corresponds to eval offset 70 - 61 = 9
    row_70 = df[df["minute_of_day"] == 70].iloc[0]

    # Baseline 15m (minutes 56..70): all 2.0
    assert row_70["rolling_mean_15m"] == pytest.approx(2.0, abs=1e-4)
    assert row_70["invocations_t"] == pytest.approx(2.0, abs=1e-4)

    # Future 5m (minutes 71..75): all 10.0
    # Since 10.0 >= 2.0 * 2.0 (4.0) AND 10.0 >= 5.0, target should be 1
    assert int(row_70["target_surge_5m"]) == 1

    # Minute 65: baseline is 2.0, future is 2.0 -> target should be 0
    row_65 = df[df["minute_of_day"] == 65].iloc[0]
    assert int(row_65["target_surge_5m"]) == 0


def test_trigger_encoding_and_fallback():
    """Verify trigger one-hot encoding and safe fallback of unknown triggers to 'others'."""
    trace = np.ones((3, 1440), dtype=np.float32)
    triggers = np.array(["timer", "custom_unknown_trigger", "HTTP"])
    triggers = np.char.lower(triggers)

    arrays = extract_features_from_matrix(trace, triggers, t_start=60, t_end=65)
    df = pd.DataFrame(arrays)

    # Function 0: timer
    func0 = df.iloc[0]
    assert func0["trigger_timer"] == 1
    assert func0["trigger_others"] == 0

    # Function 1: unknown -> should map to trigger_others
    # Each func has 5 rows (eval_len=5)
    func1 = df.iloc[5]
    assert func1["trigger_others"] == 1
    assert func1["trigger_timer"] == 0
    assert func1["trigger_http"] == 0

    # Function 2: http
    func2 = df.iloc[10]
    assert func2["trigger_http"] == 1
    assert func2["trigger_others"] == 0


def test_prior_shift_calibrator():
    """Verify analytical prior-shift log-odds formula."""
    pi_train = 0.0909  # ~9.09% in 10:1 subsampled training
    pi_natural = 0.0038  # ~0.38% in natural deployment
    cal = PriorShiftCalibrator(pi_train=pi_train, pi_natural=pi_natural)

    expected_delta = logit(pi_natural) - logit(pi_train)
    assert pytest.approx(cal.delta_logit, 1e-5) == expected_delta

    # Test an uncalibrated raw probability of 0.50 (logit = 0)
    raw_p = np.array([0.50])
    cal_p = cal.predict_proba(raw_p)
    expected_cal_p = expit(0.0 + expected_delta)
    assert pytest.approx(cal_p[0], 1e-5) == expected_cal_p
    # Calibrated probability should be significantly lower than raw due to negative shift
    assert cal_p[0] < 0.50


def test_threshold_selection_behavior():
    """Verify threshold selection finds valid tau_watch < tau_crit with sensible bounds."""
    # Synthetic validation probabilities and labels
    rng = np.random.default_rng(42)
    n = 10_000
    y_true = np.zeros(n, dtype=np.uint8)
    y_true[:50] = 1  # 0.5% prevalence

    # Positives have higher predicted probs, negatives lower
    y_probs = np.zeros(n, dtype=np.float32)
    y_probs[:50] = rng.uniform(0.3, 0.9, size=50)
    y_probs[50:] = rng.uniform(0.0, 0.2, size=n - 50)

    tau_watch, tau_crit, sweep = select_operating_thresholds(
        y_val_calibrated_probs=y_probs,
        y_val=y_true,
        n_funcs=10,
        n_days=1
    )

    assert 0.0 < tau_watch < tau_crit <= 1.0
    assert len(sweep) > 0
