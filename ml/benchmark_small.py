import time
import tracemalloc
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import pyarrow.parquet as pq
import numpy as np

from ml.train import load_training_data, evaluate_model_on_validation, FEATURE_COLUMNS
from ml.calibrate import compare_calibration_methods
from ml.evaluate import select_operating_thresholds, compute_feature_importances

def run_bench():
    train_files = sorted(Path('ml/data/processed/train').glob('day_*.parquet'))
    val_files = sorted(Path('ml/data/processed/val').glob('day_*.parquet'))

    print("=" * 60)
    print("SMALL BENCHMARK: 200k Training Samples + Val Stream")
    print("=" * 60)
    tracemalloc.start()
    t0 = time.perf_counter()
    X_train, y_train = load_training_data(train_files, max_rows=200_000)
    load_time = time.perf_counter() - t0
    n_pos = int((y_train == 1).sum())
    print(f"Loaded X: {X_train.shape}, y: {y_train.shape} ({n_pos:,} pos = {n_pos/len(y_train)*100:.2f}%) in {load_time:.2f}s")

    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
    t_fit = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t_fit
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Trained {clf.n_iter_} trees in {fit_time:.2f}s | Peak RAM: {peak_mem / (1024 * 1024):.2f} MB")

    print("\n--- Validation Stream Evaluation on Day 09 (65.5M rows) ---")
    t_eval = time.perf_counter()
    val_metrics = evaluate_model_on_validation(clf, [val_files[0]])
    eval_time = time.perf_counter() - t_eval
    print(f"Val PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Val Recall: {val_metrics['recall']:.4f}")
    print(f"Val Precision: {val_metrics['precision']:.4f}")
    print(f"Val F2: {val_metrics['f2']:.4f}")
    print(f"Val Brier: {val_metrics['brier']:.6f}")
    print(f"Inference Time: {eval_time:.2f}s ({val_metrics['samples_per_sec']:,.0f} samples/sec)")

    print("\n--- Calibration Test (2M Validation Sample) ---")
    pf_val = pq.ParquetFile(val_files[0])
    cal_batch = next(pf_val.iter_batches(batch_size=2_000_000, columns=FEATURE_COLUMNS + ["target_surge_5m"]))
    y_cal = cal_batch.column("target_surge_5m").to_numpy()
    X_cal = np.column_stack([cal_batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in FEATURE_COLUMNS])
    raw_probs = clf.predict_proba(X_cal)[:, 1].astype(np.float32)

    best_cal, best_method, report = compare_calibration_methods(
        y_raw_probs=raw_probs,
        y_val=y_cal,
        pi_train=n_pos / len(y_train),
        pi_val=0.003805
    )

    print("\n--- Threshold Sweep Test ---")
    cal_probs = best_cal.predict_proba(raw_probs)
    tau_watch, tau_crit, sweep = select_operating_thresholds(cal_probs, y_cal)
    print(f"Selected tau_watch: {tau_watch:.3f}, tau_crit: {tau_crit:.3f}")

if __name__ == "__main__":
    run_bench()
