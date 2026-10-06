"""
Standalone 5M-Row Progressive Training Experiment for AutoScale IQ (Milestone 2 - Step 5).
Strictly measures:
1. Training load & fit time, peak RAM, positive/negative counts
2. Streaming validation inference on Day 09 (65,531,125 rows)
3. Direct comparison with 200k and 1M benchmarks
4. Calibration comparison (Raw, Prior-Shift, Platt, Isotonic) on 2M natural validation sample
5. Threshold evaluation (tau_watch, tau_crit)
"""

import json
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.train import load_training_data, evaluate_model_on_validation, FEATURE_COLUMNS
from ml.calibrate import compare_calibration_methods
from ml.evaluate import select_operating_thresholds

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "ml" / "data" / "processed" / "train"
VAL_DIR = BASE_DIR / "ml" / "data" / "processed" / "val"


def run_5m_experiment():
    print("=" * 80)
    print("AUTOSCALE IQ — 5M PROGRESSIVE TRAINING EXPERIMENT")
    print("=" * 80)

    train_files = sorted(TRAIN_DIR.glob("day_*.parquet"))
    val_files = sorted(VAL_DIR.glob("day_*.parquet"))

    print(f"\n[Step 1] Checkpoint Verification:")
    print(f"  Discovered {len(train_files)} Train Parquet Files: {[f.name for f in train_files]}")
    print(f"  Discovered {len(val_files)} Val Parquet Files: {[f.name for f in val_files]}")
    assert len(train_files) == 8, f"Expected 8 train parquet files, found {len(train_files)}"
    assert len(val_files) >= 1, f"Expected at least 1 val parquet file, found {len(val_files)}"

    # 1. Load 5,000,000 training samples
    print("\n[*] Loading exactly 5,000,000 Training Samples...")
    t0_load = time.perf_counter()
    X_train, y_train = load_training_data(train_files, max_rows=5_000_000, random_seed=42)
    load_time = time.perf_counter() - t0_load

    n_train_total = len(y_train)
    n_train_pos = int((y_train == 1).sum())
    n_train_neg = int((y_train == 0).sum())
    train_pos_prev = (n_train_pos / n_train_total) * 100

    print(f"  Training Samples: {n_train_total:,} (Positives: {n_train_pos:,} = {train_pos_prev:.4f}%, Negatives: {n_train_neg:,})")
    print(f"  Training Features Shape: {X_train.shape}, Target Shape: {y_train.shape}")
    print(f"  Load Time: {load_time:.2f}s")

    # 2. Train HistGradientBoostingClassifier (same configuration as 200k and 1M)
    print("\n[*] Training HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)...")
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )

    tracemalloc.start()
    t0_fit = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0_fit
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_mem_bytes / (1024 * 1024)

    print(f"  Model Fitted: {clf.n_iter_} trees in {fit_time:.2f}s | Peak RAM: {peak_ram_mb:.2f} MB")

    # 3. Stream validation on Day 09 natural partition (65,531,125 rows)
    print("\n[*] Streaming Inference on Day 09 Natural Validation Partition (65,531,125 rows)...")
    t0_val = time.perf_counter()
    val_metrics = evaluate_model_on_validation(clf, [val_files[0]], threshold=0.5, batch_size=2_000_000)
    val_time = time.perf_counter() - t0_val

    print("\n--- Day 09 Validation Results (5M Model) ---")
    print(f"  Validation Total Rows:    {val_metrics['total_val_samples']:,}")
    print(f"  Validation Positives:     {val_metrics['total_val_positives']:,} ({val_metrics['total_val_positives']/val_metrics['total_val_samples']*100:.4f}% natural prevalence)")
    print(f"  Validation PR-AUC:        {val_metrics['pr_auc']:.4f}")
    print(f"  Validation ROC-AUC:       {val_metrics['roc_auc']:.4f}")
    print(f"  Validation Recall:        {val_metrics['recall']:.4f}")
    print(f"  Validation Precision:     {val_metrics['precision']:.4f}")
    print(f"  Validation F2 Score:      {val_metrics['f2']:.4f}")
    print(f"  Validation Brier Score:   {val_metrics['brier']:.6f}")
    print(f"  Validation Infer Time:    {val_metrics['infer_time_sec']:.2f}s ({val_metrics['samples_per_sec']:,.0f} samples/sec)")

    # 4. Calibration on exact 2M validation sample
    print("\n[*] Running Probability Calibration on 2M Validation Sample...")
    pf_val = pq.ParquetFile(val_files[0])
    cal_batch = next(pf_val.iter_batches(batch_size=2_000_000, columns=FEATURE_COLUMNS + ["target_surge_5m"]))
    y_cal = cal_batch.column("target_surge_5m").to_numpy()
    X_cal = np.column_stack([cal_batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in FEATURE_COLUMNS])

    raw_probs = clf.predict_proba(X_cal)[:, 1].astype(np.float32)

    pi_train = n_train_pos / n_train_total
    pi_val = 0.0038051856

    best_calibrator, best_method, cal_report = compare_calibration_methods(
        y_raw_probs=raw_probs,
        y_val=y_cal,
        pi_train=pi_train,
        pi_val=pi_val
    )

    # 5. Threshold Selection on Calibrated Probabilities
    cal_probs = best_calibrator.predict_proba(raw_probs)
    tau_watch, tau_crit, sweep_results = select_operating_thresholds(
        y_val_calibrated_probs=cal_probs,
        y_val=y_cal
    )

    watch_entry = min(sweep_results, key=lambda x: abs(x["threshold"] - tau_watch))
    crit_entry = min(sweep_results, key=lambda x: abs(x["threshold"] - tau_crit))

    print(f"\n--- Selected Operating Thresholds (Calibrated 5M Model) ---")
    print(f"  tau_watch = {tau_watch:.3f} | Recall: {watch_entry['recall']*100:.1f}% | Precision: {watch_entry['precision']*100:.1f}% | FA/func-day: {watch_entry['fa_per_func_day']:.4f}")
    print(f"  tau_crit  = {tau_crit:.3f}  | Recall: {crit_entry['recall']*100:.1f}% | Precision: {crit_entry['precision']*100:.1f}% | F2: {crit_entry['f2']:.4f} | FA/func-day: {crit_entry['fa_per_func_day']:.4f}")

    # 6. Direct comparison dictionary
    res_5m = {
        "train_rows": n_train_total,
        "train_pos": n_train_pos,
        "train_neg": n_train_neg,
        "train_pos_prev_pct": train_pos_prev,
        "load_time_sec": load_time,
        "fit_time_sec": fit_time,
        "peak_ram_mb": peak_ram_mb,
        "n_trees": clf.n_iter_,
        "val_rows": val_metrics["total_val_samples"],
        "val_pos": val_metrics["total_val_positives"],
        "pr_auc": val_metrics["pr_auc"],
        "roc_auc": val_metrics["roc_auc"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "f2": val_metrics["f2"],
        "brier": val_metrics["brier"],
        "infer_time_sec": val_metrics["infer_time_sec"],
        "throughput_samples_sec": val_metrics["samples_per_sec"],
        "calibration_report": cal_report,
        "selected_calibrator": best_method,
        "tau_watch": tau_watch,
        "tau_crit": tau_crit,
        "watch_metrics": watch_entry,
        "crit_metrics": crit_entry,
    }

    # Save 5M benchmark log
    out_file = BASE_DIR / "ml" / "artifacts" / "benchmark_5m_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res_5m, f, indent=2)
    print(f"\n[+] Saved 5M Benchmark Results to {out_file}")

    return res_5m


if __name__ == "__main__":
    run_5m_experiment()
