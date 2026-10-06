"""
Master Pipeline Orchestration Script for AutoScale IQ (Milestone 2 - Step 5).
Executes Phases 6 through 12 sequentially, recording all measured benchmarks, metrics, and parameters.
Memory-safe implementation with streaming parquet batch processing.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pyarrow.parquet as pq

from ml.baseline import run_baseline_evaluation
from ml.train import (
    load_training_data,
    run_progressive_experiments,
    run_hyperparameter_search,
    FEATURE_COLUMNS,
)
from ml.calibrate import compare_calibration_methods
from ml.evaluate import (
    select_operating_thresholds,
    compute_feature_importances,
    evaluate_final_test_set,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "ml" / "data" / "processed" / "train"
VAL_DIR = BASE_DIR / "ml" / "data" / "processed" / "val"
TEST_DIR = BASE_DIR / "ml" / "data" / "processed" / "test"
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"


def main():
    t_start_all = time.perf_counter()
    print("=" * 80)
    print("AUTOSCALE IQ — ML SURGE PREDICTION PIPELINE EXECUTION (STEP 5)")
    print("=" * 80)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_files = sorted(TRAIN_DIR.glob("day_*.parquet"))
    val_files = sorted(VAL_DIR.glob("day_*.parquet"))
    test_files = sorted(TEST_DIR.glob("day_*.parquet"))

    print(f"\nDiscovered Processed Parquet Partitions:")
    print(f"  Train Partitions ({len(train_files)} days): {[f.name for f in train_files]}")
    print(f"  Val Partitions   ({len(val_files)} days): {[f.name for f in val_files]}")
    print(f"  Test Partitions  ({len(test_files)} days): {[f.name for f in test_files]}")

    # Compute validation set natural prevalence via fast target stream
    print("\n[*] Auditing Natural Validation Distribution (Days 09–11)...")
    total_val_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in val_files)
    total_val_pos = sum(
        int(np.sum(b.column("target_surge_5m").to_numpy()))
        for f in val_files
        for b in pq.ParquetFile(f).iter_batches(batch_size=5_000_000, columns=["target_surge_5m"])
    )
    pi_val = total_val_pos / total_val_rows
    print(f"  Validation Total Rows: {total_val_rows:,} ({total_val_pos:,} pos = {pi_val * 100:.4f}% prevalence)")

    # -------------------------------------------------------------
    # Phase 6: Non-ML Baselines
    # -------------------------------------------------------------
    baseline_results = run_baseline_evaluation(val_files)

    # -------------------------------------------------------------
    # Phase 7: Progressive ML Experiments (A, B, C)
    # -------------------------------------------------------------
    progressive_results = run_progressive_experiments(train_files, val_files)

    # -------------------------------------------------------------
    # Phase 8: Compact Hyperparameter Search on Best Sample Size
    # -------------------------------------------------------------
    print("\n[*] Loading 1,000,000 Training Samples for Hyperparameter Search...")
    X_train_tune, y_train_tune = load_training_data(train_files, max_rows=1_000_000)
    pi_train = float((y_train_tune == 1).sum() / len(y_train_tune))

    best_model, best_config, tuning_log = run_hyperparameter_search(
        X_train=X_train_tune,
        y_train=y_train_tune,
        val_files=val_files
    )

    # -------------------------------------------------------------
    # Phase 9: Probability Calibration on Representative Validation Sample
    # -------------------------------------------------------------
    print("\n[*] Constructing Representative Calibration Sample from Natural Validation Data...")
    pf_val = pq.ParquetFile(val_files[0])
    cal_batch = next(pf_val.iter_batches(batch_size=2_000_000, columns=FEATURE_COLUMNS + ["target_surge_5m"]))
    y_cal_sample = cal_batch.column("target_surge_5m").to_numpy()
    X_cal_sample = np.column_stack([cal_batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in FEATURE_COLUMNS])

    cal_raw_probs = best_model.predict_proba(X_cal_sample)[:, 1].astype(np.float32)

    best_calibrator, best_cal_method, cal_report = compare_calibration_methods(
        y_raw_probs=cal_raw_probs,
        y_val=y_cal_sample,
        pi_train=pi_train,
        pi_val=pi_val
    )

    # -------------------------------------------------------------
    # Phase 10: Threshold Selection on Calibrated Validation Data
    # -------------------------------------------------------------
    cal_probs_sample = best_calibrator.predict_proba(cal_raw_probs)
    tau_watch, tau_crit, sweep_results = select_operating_thresholds(
        y_val_calibrated_probs=cal_probs_sample,
        y_val=y_cal_sample
    )

    # -------------------------------------------------------------
    # Phase 11: Feature Importance & Explainability
    # -------------------------------------------------------------
    importances = compute_feature_importances(best_model, val_files[0], sample_size=50_000)

    # -------------------------------------------------------------
    # Phase 12: Final Chronological Test Set Evaluation (Days 12–14)
    # -------------------------------------------------------------
    test_results = evaluate_final_test_set(
        model=best_model,
        calibrator=best_calibrator,
        tau_watch=tau_watch,
        tau_crit=tau_crit,
        test_files=test_files
    )

    # -------------------------------------------------------------
    # Export Model Artifact Bundle
    # -------------------------------------------------------------
    artifact_payload = {
        "model": best_model,
        "calibrator": best_calibrator,
        "calibrator_method": best_cal_method,
        "tau_watch": tau_watch,
        "tau_crit": tau_crit,
        "feature_names": FEATURE_COLUMNS,
        "pi_train": pi_train,
        "pi_val": pi_val,
        "best_config": best_config,
    }

    model_path = ARTIFACTS_DIR / "surge_model.joblib"
    joblib.dump(artifact_payload, model_path)
    model_size_kb = model_path.stat().st_size / 1024
    print(f"\n[+] Saved Serialized Model Bundle to {model_path} ({model_size_kb:.2f} KB)")

    # Save summary report JSON
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_results": baseline_results,
        "progressive_experiments": progressive_results,
        "best_hyperparameters": best_config,
        "hyperparameter_tuning_log": tuning_log,
        "calibration_comparison": cal_report,
        "selected_thresholds": {"tau_watch": tau_watch, "tau_crit": tau_crit},
        "top_feature_importances": importances[:10],
        "test_results": test_results,
        "model_artifact_size_kb": model_size_kb,
        "total_pipeline_time_sec": time.perf_counter() - t_start_all,
    }

    report_path = ARTIFACTS_DIR / "step5_training_report.json"
    with open(report_path, "w") as fp:
        json.dump(summary_report, fp, indent=2)
    print(f"[+] Saved Empirical Report JSON to {report_path}")

    print("\n" + "=" * 80)
    print(f"[+] FULL ML PIPELINE COMPLETED IN {time.perf_counter() - t_start_all:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
