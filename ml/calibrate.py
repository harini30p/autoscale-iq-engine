"""
Probability Calibration and Reliability Analysis for AutoScale IQ (Milestone 2 - Step 5).
Addresses training vs natural deployment prevalence prior-shift (~9.09% vs ~0.38%).
Compares Prior-Shift Log-Odds Correction, Platt/Sigmoid Calibration, and Isotonic Calibration.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.special import logit, expit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, average_precision_score


class PriorShiftCalibrator:
    """
    Exact analytical log-odds correction for prior shift.
    logit(P_calibrated) = logit(P_raw) - logit(pi_train) + logit(pi_natural)
    """

    def __init__(self, pi_train: float, pi_natural: float, clip_eps: float = 1e-7):
        self.pi_train = float(pi_train)
        self.pi_natural = float(pi_natural)
        self.clip_eps = clip_eps
        self.delta_logit = logit(self.pi_natural) - logit(self.pi_train)

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(raw_probs, self.clip_eps, 1.0 - self.clip_eps)
        raw_logits = logit(p_clipped)
        adj_logits = raw_logits + self.delta_logit
        return expit(adj_logits)


class PlattCalibrator:
    """Sigmoid / Platt scaling fit on validation logits."""

    def __init__(self, clip_eps: float = 1e-7):
        self.lr = LogisticRegression(C=1.0, solver="lbfgs")
        self.clip_eps = clip_eps

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        p_clipped = np.clip(raw_probs, self.clip_eps, 1.0 - self.clip_eps)
        logits = logit(p_clipped).reshape(-1, 1)
        self.lr.fit(logits, y_true)
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(raw_probs, self.clip_eps, 1.0 - self.clip_eps)
        logits = logit(p_clipped).reshape(-1, 1)
        return self.lr.predict_proba(logits)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression fit on validation probabilities."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        self.iso.fit(raw_probs, y_true)
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        return self.iso.predict(raw_probs)


def compare_calibration_methods(
    y_raw_probs: np.ndarray,
    y_val: np.ndarray,
    pi_train: float,
    pi_val: float
) -> Tuple[Any, str, Dict[str, Dict[str, float]]]:
    """
    Compare raw model vs Prior-Shift vs Platt vs Isotonic on the natural validation set.
    """
    print("=" * 80)
    print("PHASE 9: PROBABILITY CALIBRATION & RELIABILITY COMPARISON")
    print("=" * 80)
    print(f"  Training Sample Prevalence:    {pi_train * 100:.2f}% (pi_train = {pi_train:.5f})")
    print(f"  Validation Natural Prevalence: {pi_val * 100:.3f}% (pi_val = {pi_val:.5f})")

    # 1. Uncalibrated / Raw
    brier_raw = brier_score_loss(y_val, y_raw_probs)
    pr_raw = average_precision_score(y_val, y_raw_probs)

    # 2. Method A: Prior-Shift
    cal_prior = PriorShiftCalibrator(pi_train=pi_train, pi_natural=pi_val)
    p_prior = cal_prior.predict_proba(y_raw_probs)
    brier_prior = brier_score_loss(y_val, p_prior)
    pr_prior = average_precision_score(y_val, p_prior)

    # 3. Method B: Platt / Sigmoid
    cal_platt = PlattCalibrator().fit(y_raw_probs, y_val)
    p_platt = cal_platt.predict_proba(y_raw_probs)
    brier_platt = brier_score_loss(y_val, p_platt)
    pr_platt = average_precision_score(y_val, p_platt)

    # 4. Method C: Isotonic
    cal_iso = IsotonicCalibrator().fit(y_raw_probs, y_val)
    p_iso = cal_iso.predict_proba(y_raw_probs)
    brier_iso = brier_score_loss(y_val, p_iso)
    pr_iso = average_precision_score(y_val, p_iso)

    n_cal_samples = len(y_val)
    n_cal_pos = int((y_val == 1).sum())
    n_cal_neg = int((y_val == 0).sum())
    cal_prev_pct = (n_cal_pos / n_cal_samples) * 100

    report = {
        "calibration_sample_size": n_cal_samples,
        "calibration_positives": n_cal_pos,
        "calibration_negatives": n_cal_neg,
        "calibration_pos_prevalence_pct": cal_prev_pct,
        "raw_uncalibrated": {"brier": float(brier_raw), "pr_auc": float(pr_raw), "mean_pred": float(np.mean(y_raw_probs))},
        "prior_shift_correction": {"brier": float(brier_prior), "pr_auc": float(pr_prior), "mean_pred": float(np.mean(p_prior)), "delta_logit": float(cal_prior.delta_logit)},
        "platt_sigmoid": {"brier": float(brier_platt), "pr_auc": float(pr_platt), "mean_pred": float(np.mean(p_platt))},
        "isotonic": {"brier": float(brier_iso), "pr_auc": float(pr_iso), "mean_pred": float(np.mean(p_iso))},
    }

    print("\n--- Calibration Method Comparison ---")
    print(f"  Raw (Uncalibrated):      Brier = {brier_raw:.6f} | Mean Prob = {np.mean(y_raw_probs):.4f} | PR-AUC = {pr_raw:.4f}")
    print(f"  Prior-Shift Correction:  Brier = {brier_prior:.6f} | Mean Prob = {np.mean(p_prior):.4f} | PR-AUC = {pr_prior:.4f} (delta_logit = {cal_prior.delta_logit:.4f})")
    print(f"  Platt / Sigmoid:         Brier = {brier_platt:.6f} | Mean Prob = {np.mean(p_platt):.4f} | PR-AUC = {pr_platt:.4f}")
    print(f"  Isotonic Regression:     Brier = {brier_iso:.6f} | Mean Prob = {np.mean(p_iso):.4f} | PR-AUC = {pr_iso:.4f}")

    # Select best calibration method by lowest Brier score
    best_method = min(["prior_shift_correction", "platt_sigmoid", "isotonic"], key=lambda k: report[k]["brier"])
    if best_method == "prior_shift_correction":
        best_calibrator = cal_prior
    elif best_method == "platt_sigmoid":
        best_calibrator = cal_platt
    else:
        best_calibrator = cal_iso

    print(f"\n[+] Selected Calibration Method: {best_method.upper()} (Lowest Validation Brier Score = {report[best_method]['brier']:.6f})")
    return best_calibrator, best_method, report
