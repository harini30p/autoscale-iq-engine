"""
Evaluation, Threshold Selection, Explainability, and Test Set Evaluation for AutoScale IQ.
Evaluates final tuned & calibrated surge model on validation and final unseen test sets (Days 12–14).
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss,
)

from ml.train import FEATURE_COLUMNS, predict_proba_streaming

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_DIR = BASE_DIR / "ml" / "data" / "processed" / "val"
TEST_DIR = BASE_DIR / "ml" / "data" / "processed" / "test"
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"


def select_operating_thresholds(
    y_val_calibrated_probs: np.ndarray,
    y_val: np.ndarray,
    n_funcs: Optional[int] = None,
    n_days: Optional[int] = None
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate threshold sweep on calibrated validation probabilities.
    Selects tau_watch and tau_crit for Safety Controller integration.
    """
    print("=" * 80)
    print("PHASE 10: THRESHOLD SELECTION ON NATURAL VALIDATION SET")
    print("=" * 80)

    if n_funcs is not None and n_days is not None:
        total_func_days = float(n_funcs * n_days)
    else:
        total_func_days = float(len(y_val) / 1375.0)
    thresholds = np.linspace(0.01, 0.80, 80)
    sweep_results = []

    for tau in thresholds:
        y_pred = (y_val_calibrated_probs >= tau).astype(np.uint8)
        tp = int(((y_pred == 1) & (y_val == 1)).sum())
        fp = int(((y_pred == 1) & (y_val == 0)).sum())
        fn = int(((y_pred == 0) & (y_val == 1)).sum())
        tn = int(((y_pred == 0) & (y_val == 0)).sum())

        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)

        fa_per_func_day = fp / total_func_days if total_func_days > 0 else 0.0

        sweep_results.append({
            "threshold": float(tau),
            "precision": float(prec),
            "recall": float(rec),
            "f2": float(f2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_alerts": int(tp + fp),
            "fa_per_func_day": float(fa_per_func_day),
            "fa_per_1k_func_days": float(fa_per_func_day * 1000.0),
        })

    # Find best tau_crit (optimal F2 score with high precision)
    f2_sorted = sorted(sweep_results, key=lambda x: x["f2"], reverse=True)
    best_f2_entry = f2_sorted[0]
    tau_crit = best_f2_entry["threshold"]

    # For tau_watch: lower threshold with higher recall (~75-80%)
    watch_candidates = [r for r in sweep_results if r["recall"] >= 0.75 and r["precision"] >= 0.20]
    if watch_candidates:
        tau_watch = min(watch_candidates, key=lambda x: x["threshold"])["threshold"]
    else:
        tau_watch = max(0.02, tau_crit * 0.5)

    if tau_watch >= tau_crit:
        tau_watch = round(tau_crit * 0.6, 3)

    watch_entry = min(sweep_results, key=lambda x: abs(x["threshold"] - tau_watch))
    crit_entry = min(sweep_results, key=lambda x: abs(x["threshold"] - tau_crit))

    print("\n--- Selected Thresholds for Safety Controller ---")
    print(f"  tau_watch = {tau_watch:.3f} | Recall: {watch_entry['recall']*100:.1f}% | Precision: {watch_entry['precision']*100:.1f}% | FA/func-day: {watch_entry['fa_per_func_day']:.4f}")
    print(f"  tau_crit  = {tau_crit:.3f}  | Recall: {crit_entry['recall']*100:.1f}% | Precision: {crit_entry['precision']*100:.1f}% | F2: {crit_entry['f2']:.4f} | FA/func-day: {crit_entry['fa_per_func_day']:.4f}")

    return tau_watch, tau_crit, sweep_results


def compute_feature_importances(
    model: Any,
    val_file: Path,
    sample_size: int = 50_000
) -> List[Dict[str, Any]]:
    """Compute permutation feature importance on validation sample."""
    print("\n" + "=" * 80)
    print("PHASE 11: EXPLAINABILITY & FEATURE IMPORTANCE")
    print("=" * 80)

    pf = pq.ParquetFile(val_file)
    # Read first 100k rows to sample
    batch = next(pf.iter_batches(batch_size=100_000, columns=FEATURE_COLUMNS + ["target_surge_5m"]))
    df_sample = batch.to_pandas()

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(df_sample), size=min(len(df_sample), sample_size), replace=False)
    X_sub = df_sample[FEATURE_COLUMNS].iloc[sample_idx].to_numpy(dtype=np.float32)
    y_sub = df_sample["target_surge_5m"].iloc[sample_idx].to_numpy(dtype=np.uint8)

    perm_res = permutation_importance(
        model,
        X_sub,
        y_sub,
        n_repeats=5,
        scoring="average_precision",
        random_state=42
    )

    importances = []
    for idx, col in enumerate(FEATURE_COLUMNS):
        importances.append({
            "feature": col,
            "importance_mean": float(perm_res.importances_mean[idx]),
            "importance_std": float(perm_res.importances_std[idx]),
        })

    importances.sort(key=lambda x: x["importance_mean"], reverse=True)

    print("\nTop Features by Permutation PR-AUC Importance:")
    for rank, item in enumerate(importances[:10], 1):
        print(f"  {rank:02d}. {item['feature']:<25} : {item['importance_mean']:.5f} (+/- {item['importance_std']:.5f})")

    return importances


def evaluate_final_test_set(
    model: Any,
    calibrator: Any,
    tau_watch: float,
    tau_crit: float,
    test_files: List[Path]
) -> Dict[str, Any]:
    """
    Perform single final evaluation on the locked chronological Test Set (Days 12–14).
    """
    print("\n" + "=" * 80)
    print("PHASE 12: FINAL CHRONOLOGICAL TEST SET EVALUATION (DAYS 12–14)")
    print("=" * 80)

    y_test, raw_probs, infer_time = predict_proba_streaming(model, test_files)
    cal_probs = calibrator.predict_proba(raw_probs)

    n_samples = len(y_test)
    n_pos = int((y_test == 1).sum())
    n_days = len(test_files)
    n_funcs = n_samples // (n_days * 1375)
    total_func_days = n_funcs * n_days

    latency_per_10k = (infer_time / n_samples) * 10_000 * 1000.0  # in ms

    pr_auc = float(average_precision_score(y_test, cal_probs))
    roc_auc = float(roc_auc_score(y_test, cal_probs))
    brier = float(brier_score_loss(y_test, cal_probs))

    # Evaluate at tau_crit
    y_pred_crit = (cal_probs >= tau_crit).astype(np.uint8)
    prec_crit = float(precision_score(y_test, y_pred_crit, zero_division=0))
    rec_crit = float(recall_score(y_test, y_pred_crit, zero_division=0))
    f2_crit = float(fbeta_score(y_test, y_pred_crit, beta=2, zero_division=0))

    tp = int(((y_pred_crit == 1) & (y_test == 1)).sum())
    fp = int(((y_pred_crit == 1) & (y_test == 0)).sum())
    fn = int(((y_pred_crit == 0) & (y_test == 1)).sum())
    tn = int(((y_pred_crit == 0) & (y_test == 0)).sum())

    fa_per_func_day = fp / total_func_days if total_func_days > 0 else 0.0
    fa_per_1k_func_days = fa_per_func_day * 1000.0

    # Evaluate at tau_watch
    y_pred_watch = (cal_probs >= tau_watch).astype(np.uint8)
    prec_watch = float(precision_score(y_test, y_pred_watch, zero_division=0))
    rec_watch = float(recall_score(y_test, y_pred_watch, zero_division=0))

    test_report = {
        "test_days": [f.name for f in test_files],
        "total_test_samples": n_samples,
        "positive_samples": n_pos,
        "positive_prevalence_pct": (n_pos / n_samples) * 100,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "inference_time_sec": infer_time,
        "inference_latency_per_10k_ms": latency_per_10k,
        "tau_watch": tau_watch,
        "tau_crit": tau_crit,
        "crit_threshold_metrics": {
            "precision": prec_crit,
            "recall": rec_crit,
            "f2": f2_crit,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "predicted_alerts": tp + fp,
            "surge_capture_rate_pct": rec_crit * 100,
            "fa_per_func_day": fa_per_func_day,
            "fa_per_1k_func_days": fa_per_1k_func_days,
        },
        "watch_threshold_metrics": {
            "precision": prec_watch,
            "recall": rec_watch,
            "predicted_alerts": int((y_pred_watch == 1).sum()),
        }
    }

    print("\n--- Final Test Set (Days 12–14) Performance ---")
    print(f"  Total Test Samples:       {n_samples:,} ({n_pos:,} surges = {n_pos/n_samples*100:.3f}% natural prevalence)")
    print(f"  PR-AUC (Avg Precision):   {pr_auc:.4f}")
    print(f"  ROC-AUC:                  {roc_auc:.4f}")
    print(f"  Brier Score:              {brier:.6f}")
    print(f"  Inference Latency:        {latency_per_10k:.2f} ms per 10,000 predictions ({n_samples/infer_time:,.0f} samples/sec)")
    print(f"\n--- Operational Metrics at tau_crit = {tau_crit:.3f} ---")
    print(f"  Precision:                {prec_crit:.4f}")
    print(f"  Recall (Capture Rate):    {rec_crit*100:.2f}% ({tp:,} / {n_pos:,} surges captured)")
    print(f"  F2 Score:                 {f2_crit:.4f}")
    print(f"  Predicted Alerts:         {tp+fp:,} (TP={tp:,}, FP={fp:,})")
    print(f"  False Alarms / Func-Day:  {fa_per_func_day:.4f} ({fa_per_1k_func_days:.1f} false alarms per 1k func-days)")

    return test_report
