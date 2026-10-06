"""
Non-ML Heuristic Baselines for AutoScale IQ (Milestone 2 - Step 5).
Evaluates rule-based workload surge detectors on the natural validation set using streaming batches.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_DIR = BASE_DIR / "ml" / "data" / "processed" / "val"


def evaluate_rule_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_funcs: int,
    n_days: int
) -> Dict[str, Any]:
    """
    Compute full classification and operational metrics for a binary detector.
    """
    total_samples = len(y_true)
    total_pos = int((y_true == 1).sum())
    total_alerts = int((y_pred == 1).sum())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_pred)

    total_func_days = n_funcs * n_days
    fa_per_func_day = fp / total_func_days if total_func_days > 0 else 0.0
    fa_per_1k_func_days = fa_per_func_day * 1000.0

    return {
        "total_samples": total_samples,
        "positives": total_pos,
        "predicted_alerts": total_alerts,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fa_per_func_day": float(fa_per_func_day),
        "fa_per_1k_func_days": float(fa_per_1k_func_days),
    }


def run_baseline_evaluation(val_files: List[Path], batch_size: int = 5_000_000) -> Dict[str, Any]:
    """
    Run Baseline 1 (Momentum) and Baseline 2 (Acceleration) across validation partitions using streaming batches.
    """
    print("=" * 80)
    print("PHASE 6: NON-ML BASELINE EVALUATION (ON NATURAL VALIDATION SET)")
    print("=" * 80)

    all_y_true = []
    all_pred_m1 = []
    all_pred_m2 = []

    cols = ["invocations_t", "rolling_mean_15m", "rate_delta_1m", "rate_delta_5m", "target_surge_5m"]

    for f in val_files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            y_b = batch.column("target_surge_5m").to_numpy()
            inv_t = batch.column("invocations_t").to_numpy()
            roll15 = batch.column("rolling_mean_15m").to_numpy()
            delta1 = batch.column("rate_delta_1m").to_numpy()
            delta5 = batch.column("rate_delta_5m").to_numpy()

            # Baseline 1: Momentum Rule
            pred_b1 = ((inv_t >= 2.0 * roll15) & (inv_t >= 5.0)).astype(np.uint8)

            # Baseline 2: Acceleration Rule
            pred_b2 = ((delta1 > 0.0) & (delta5 >= 1.5 * roll15) & (inv_t >= 5.0)).astype(np.uint8)

            all_y_true.append(y_b)
            all_pred_m1.append(pred_b1)
            all_pred_m2.append(pred_b2)

    y_true = np.concatenate(all_y_true)
    pred_b1 = np.concatenate(all_pred_m1)
    pred_b2 = np.concatenate(all_pred_m2)

    n_days = len(val_files)
    n_funcs = len(y_true) // (n_days * 1375)

    res_b1 = evaluate_rule_predictions(y_true, pred_b1, n_funcs=n_funcs, n_days=n_days)
    res_b2 = evaluate_rule_predictions(y_true, pred_b2, n_funcs=n_funcs, n_days=n_days)

    print(f"\n[+] Total Validation Samples: {len(y_true):,} ({res_b1['positives']:,} positives = {res_b1['positives']/len(y_true)*100:.3f}% prevalence)")
    print(f"\n--- Baseline 1: Heuristic Momentum Rule ---")
    print(f"  PR-AUC / Avg Precision:   {res_b1['pr_auc']:.4f}")
    print(f"  Precision:                {res_b1['precision']:.4f}")
    print(f"  Recall:                   {res_b1['recall']:.4f}")
    print(f"  F2 Score:                 {res_b1['f2']:.4f}")
    print(f"  ROC-AUC:                  {res_b1['roc_auc']:.4f}")
    print(f"  Predicted Alerts:         {res_b1['predicted_alerts']:,} (TP={res_b1['tp']:,}, FP={res_b1['fp']:,})")
    print(f"  False Alarms / Func-Day:  {res_b1['fa_per_func_day']:.4f} ({res_b1['fa_per_1k_func_days']:.1f} per 1k func-days)")

    print(f"\n--- Baseline 2: Heuristic Acceleration Rule ---")
    print(f"  PR-AUC / Avg Precision:   {res_b2['pr_auc']:.4f}")
    print(f"  Precision:                {res_b2['precision']:.4f}")
    print(f"  Recall:                   {res_b2['recall']:.4f}")
    print(f"  F2 Score:                 {res_b2['f2']:.4f}")
    print(f"  ROC-AUC:                  {res_b2['roc_auc']:.4f}")
    print(f"  Predicted Alerts:         {res_b2['predicted_alerts']:,} (TP={res_b2['tp']:,}, FP={res_b2['fp']:,})")
    print(f"  False Alarms / Func-Day:  {res_b2['fa_per_func_day']:.4f} ({res_b2['fa_per_1k_func_days']:.1f} per 1k func-days)")

    return {
        "baseline_1_momentum": res_b1,
        "baseline_2_acceleration": res_b2,
    }


if __name__ == "__main__":
    val_files = sorted(VAL_DIR.glob("day_*.parquet"))
    if val_files:
        run_baseline_evaluation(val_files)
    else:
        print("No validation files found in ml/data/processed/val")
