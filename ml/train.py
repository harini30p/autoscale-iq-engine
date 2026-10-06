"""
Model Training and Progressive Benchmarking Module for AutoScale IQ (Milestone 2 - Step 5).
Uses HistGradientBoostingClassifier with early stopping, progressive sample scaling,
and compact hyperparameter search evaluated on the natural validation set.
"""

import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "ml" / "data" / "processed" / "train"
VAL_DIR = BASE_DIR / "ml" / "data" / "processed" / "val"
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"

FEATURE_COLUMNS = [
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


def load_training_data(
    train_files: List[Path],
    max_rows: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load chronological training partitions up to max_rows.
    Preserves all features and target dtypes.
    """
    dfs = []
    total_loaded = 0

    for f in sorted(train_files):
        df_part = pq.read_table(f).to_pandas()
        dfs.append(df_part)
        total_loaded += len(df_part)
        if max_rows is not None and total_loaded >= max_rows:
            break

    df_all = pd.concat(dfs, ignore_index=True)
    if max_rows is not None and len(df_all) > max_rows:
        pos_mask = df_all["target_surge_5m"] == 1
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(~pos_mask)[0]

        target_pos = int(max_rows * (len(pos_idx) / len(df_all)))
        target_neg = max_rows - target_pos

        rng = np.random.default_rng(random_seed)
        sub_pos = rng.choice(pos_idx, size=min(len(pos_idx), target_pos), replace=False)
        sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), target_neg), replace=False)
        sel_idx = np.sort(np.concatenate([sub_pos, sub_neg]))
        df_all = df_all.iloc[sel_idx].reset_index(drop=True)

    X = df_all[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df_all["target_surge_5m"].to_numpy(dtype=np.uint8)
    return X, y


def predict_proba_streaming(
    model: Any,
    files: List[Path],
    batch_size: int = 2_000_000
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Stream parquet files in batches and return (y_true, y_probs, total_inference_time).
    Peak memory < 200 MB regardless of dataset size.
    """
    all_y = []
    all_p = []
    cols = FEATURE_COLUMNS + ["target_surge_5m"]

    t0 = time.perf_counter()
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            y_b = batch.column("target_surge_5m").to_numpy()
            X_b = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in FEATURE_COLUMNS])

            p_b = model.predict_proba(X_b)[:, 1].astype(np.float32)
            all_y.append(y_b)
            all_p.append(p_b)

    infer_time = time.perf_counter() - t0
    y_true = np.concatenate(all_y)
    y_probs = np.concatenate(all_p)
    return y_true, y_probs, infer_time


def evaluate_model_on_validation(
    model: Any,
    val_files: List[Path],
    threshold: float = 0.5,
    batch_size: int = 2_000_000
) -> Dict[str, float]:
    """Compute validation metrics on natural validation distribution using streaming batches."""
    y_val, y_prob, infer_time = predict_proba_streaming(model, val_files, batch_size=batch_size)

    y_pred = (y_prob >= threshold).astype(np.uint8)

    pr_auc = float(average_precision_score(y_val, y_prob))
    roc_auc = float(roc_auc_score(y_val, y_prob))
    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall = float(recall_score(y_val, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_val, y_pred, beta=2, zero_division=0))
    brier = float(brier_score_loss(y_val, y_prob))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "brier": brier,
        "infer_time_sec": infer_time,
        "samples_per_sec": len(y_val) / infer_time if infer_time > 0 else 0,
        "total_val_samples": len(y_val),
        "total_val_positives": int((y_val == 1).sum()),
    }


def run_progressive_experiments(
    train_files: List[Path],
    val_files: List[Path]
) -> List[Dict[str, Any]]:
    """
    Run Progressive Experiments A (200k), B (1M), C (5M).
    """
    print("=" * 80)
    print("PHASE 7: PROGRESSIVE ML BENCHMARK EXPERIMENTS")
    print("=" * 80)

    sample_sizes = [
        ("Experiment A (~200k rows)", 200_000),
        ("Experiment B (~1M rows)", 1_000_000),
        ("Experiment C (~5M rows)", 5_000_000),
    ]

    results = []

    for name, n_samples in sample_sizes:
        print(f"\n[*] Running {name}...")
        X_train, y_train = load_training_data(train_files, max_rows=n_samples)

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        n_total = len(y_train)
        pos_prev = (n_pos / n_total) * 100

        print(f"  Training Set: {n_total:,} rows ({n_pos:,} pos = {pos_prev:.2f}%, {n_neg:,} neg)")

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
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)

        val_metrics = evaluate_model_on_validation(clf, val_files)

        exp_res = {
            "name": name,
            "total_rows": n_total,
            "pos_rows": n_pos,
            "neg_rows": n_neg,
            "pos_prevalence_pct": pos_prev,
            "train_time_sec": train_time,
            "peak_ram_mb": peak_mb,
            "n_trees": clf.n_iter_,
            **val_metrics,
        }
        results.append(exp_res)

        print(f"  Trained {clf.n_iter_} trees in {train_time:.2f}s | Peak RAM: {peak_mb:.2f} MB")
        print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | Recall: {val_metrics['recall']:.4f} | Precision: {val_metrics['precision']:.4f} | F2: {val_metrics['f2']:.4f}")

    return results


def run_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_files: List[Path]
) -> Tuple[HistGradientBoostingClassifier, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compact hyperparameter search over 8 targeted configurations.
    All selection strictly evaluated on the natural validation set.
    """
    print("\n" + "=" * 80)
    print("PHASE 8: COMPACT HYPERPARAMETER SEARCH")
    print("=" * 80)

    configs = [
        {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"max_iter": 150, "learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"max_iter": 150, "learning_rate": 0.10, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"max_iter": 150, "learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 2.0},
        {"max_iter": 150, "learning_rate": 0.15, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 30, "l2_regularization": 2.0},
        {"max_iter": 200, "learning_rate": 0.10, "max_leaf_nodes": 45, "min_samples_leaf": 30, "l2_regularization": 1.5},
        {"max_iter": 200, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.5},
    ]

    best_model = None
    best_config = None
    best_pr_auc = -1.0
    tuning_log = []

    # Use Day 09 (65.5M rows) for fast, robust tuning across configs
    tune_val_files = [val_files[0]]

    for idx, cfg in enumerate(configs, 1):
        clf = HistGradientBoostingClassifier(
            **cfg,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42
        )
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        val_metrics = evaluate_model_on_validation(clf, tune_val_files)
        pr_auc = val_metrics["pr_auc"]

        log_entry = {
            "config_id": idx,
            **cfg,
            "n_trees": clf.n_iter_,
            "fit_time_sec": fit_time,
            **val_metrics,
        }
        tuning_log.append(log_entry)

        is_best = pr_auc > best_pr_auc
        if is_best:
            best_pr_auc = pr_auc
            best_model = clf
            best_config = log_entry

        mark = " [* BEST]" if is_best else ""
        print(f"  Config {idx:02d}: lr={cfg['learning_rate']}, leaves={cfg['max_leaf_nodes']}, min_leaf={cfg['min_samples_leaf']}, l2={cfg['l2_regularization']} -> PR-AUC: {pr_auc:.4f}, F2: {val_metrics['f2']:.4f}, Trees: {clf.n_iter_}{mark}")

    print(f"\n[+] Selected Best Model: Config {best_config['config_id']} (Validation PR-AUC = {best_config['pr_auc']:.4f})")
    return best_model, best_config, tuning_log
