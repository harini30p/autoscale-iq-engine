"""
Fast Vectorized Empirical Target & Class Balance Analysis for AutoScale IQ.
Calculates exact positive/negative ratios for candidate targets on Azure Functions 2019 data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "ml" / "data" / "raw"


def analyze_target_distributions():
    print("=" * 80)
    print("VECTORIZED TARGET DESIGN & CLASS BALANCE ANALYSIS (AZURE FUNCTIONS 2019)")
    print("=" * 80)

    inv_file_d1 = RAW_DIR / "invocations_per_function_md.anon.d01.csv"
    if not inv_file_d1.exists():
        print(f"[-] File not found: {inv_file_d1}")
        return

    print(f"[*] Loading {inv_file_d1.name} ...")
    df = pd.read_csv(inv_file_d1)
    minute_cols = [str(i) for i in range(1, 1441)]
    mat = df[minute_cols].values.astype(np.float32)  # shape: (N_functions, 1440)
    n_funcs, n_mins = mat.shape
    print(f"[+] Loaded {n_funcs:,} functions across {n_mins} minutes.")

    # Filter functions with active invocations (>= 10 calls during the day)
    func_totals = mat.sum(axis=1)
    active_mask = func_totals >= 10
    active_mat = mat[active_mask]  # shape: (N_active, 1440)
    n_active = len(active_mat)
    print(f"[+] Active functions evaluated: {n_active:,} ({n_active/n_funcs*100:.1f}%)")

    # Time boundaries:
    # We evaluate t in [60, 1435] (1-indexed minute 61 to 1435; in 0-indexed terms: t from 60 to 1434)
    # At index t:
    # - current: active_mat[:, t]
    # - baseline_15m: mean of active_mat[:, t-14 : t+1]
    # - baseline_60m: mean of active_mat[:, t-59 : t+1]
    # - future_5m_avg: mean of active_mat[:, t+1 : t+6]
    t_start, t_end = 60, 1435
    eval_len = t_end - t_start  # 1375 evaluation points per function

    print(f"[+] Computing sliding window metrics across {n_active:,} active functions...")

    # Vectorized compute using uniform 1D filter / cumulative sum
    # Cumulative sum along time axis: shape (N, 1441) with prepended 0
    padded_cumsum = np.pad(np.cumsum(active_mat, axis=1), ((0, 0), (1, 0)), mode="constant")

    # future_5m_avg at index t: (cumsum[t+6] - cumsum[t+1]) / 5
    # For t in range(t_start, t_end):
    # t_indices = np.arange(t_start, t_end)
    t_idx = np.arange(t_start, t_end)  # 60 .. 1434

    # future_5m_avg matrix of shape (n_active, 1375)
    future_5m_avg = (padded_cumsum[:, t_idx + 6] - padded_cumsum[:, t_idx + 1]) / 5.0
    baseline_15m = (padded_cumsum[:, t_idx + 1] - padded_cumsum[:, t_idx - 14]) / 15.0
    baseline_60m = (padded_cumsum[:, t_idx + 1] - padded_cumsum[:, t_idx - 59]) / 60.0
    current_val = active_mat[:, t_idx]

    # Historical function-level percentiles
    func_p90 = np.percentile(active_mat, 90, axis=1, keepdims=True)  # shape: (n_active, 1)
    func_p95 = np.percentile(active_mat, 95, axis=1, keepdims=True)  # shape: (n_active, 1)

    total_eval_points = n_active * eval_len
    print(f"[+] Total evaluated observation points: {total_eval_points:,}\n")

    print("=" * 80)
    print("1. RELATIVE SURGE MULTIPLIER K (future_5m_avg >= K * baseline_15m & future_5m_avg >= 5)")
    print("=" * 80)
    for K in [1.5, 2.0, 2.5, 3.0, 4.0]:
        mask = (future_5m_avg >= K * baseline_15m) & (future_5m_avg >= 5.0)
        pos_count = np.sum(mask)
        pos_pct = pos_count / total_eval_points * 100
        print(f"  Surge K={K:.1f}x:  Positive = {pos_pct:6.2f}% ({pos_count:,} / {total_eval_points:,})")

    print("\n" + "=" * 80)
    print("2. HISTORICAL PERCENTILE THRESHOLD (future_5m_avg >= func_pXX & future_5m_avg >= 5)")
    print("=" * 80)
    for p_name, p_arr in [("P90", func_p90), ("P95", func_p95)]:
        mask = (future_5m_avg >= p_arr) & (future_5m_avg >= 5.0)
        pos_count = np.sum(mask)
        pos_pct = pos_count / total_eval_points * 100
        print(f"  Threshold >= {p_name}: Positive = {pos_pct:6.2f}% ({pos_count:,} / {total_eval_points:,})")

    print("\n" + "=" * 80)
    print("3. RECOMMENDED CANONICAL TARGET FORMULATIONS")
    print("=" * 80)
    
    # Target Option 1: Significant Surge (K=2.0x baseline + min volume >= 5 calls/min)
    t1_mask = (future_5m_avg >= 2.0 * baseline_15m) & (future_5m_avg >= 5.0)
    t1_pos = np.sum(t1_mask)
    print(f"Option 1 (Relative Surge K=2.0x, min 5 req/min):")
    print(f"  Positive Class (High-Load Surge): {t1_pos / total_eval_points * 100:.2f}% ({t1_pos:,})")
    print(f"  Negative Class (Normal/Sub-surge): {100 - t1_pos / total_eval_points * 100:.2f}%")

    # Target Option 2: High-Severity Surge (K=3.0x baseline + min volume >= 10 calls/min)
    t2_mask = (future_5m_avg >= 3.0 * baseline_15m) & (future_5m_avg >= 10.0)
    t2_pos = np.sum(t2_mask)
    print(f"\nOption 2 (Severe Surge K=3.0x, min 10 req/min):")
    print(f"  Positive Class (Severe Surge):   {t2_pos / total_eval_points * 100:.2f}% ({t2_pos:,})")
    print(f"  Negative Class (Normal/Sub-surge): {100 - t2_pos / total_eval_points * 100:.2f}%")

    # Target Option 3: Dual Adaptive Target (Surge OR P95 Tail Exceedance with volume >= 5)
    t3_mask = ((future_5m_avg >= 2.0 * baseline_15m) | (future_5m_avg >= func_p95)) & (future_5m_avg >= 5.0)
    t3_pos = np.sum(t3_mask)
    print(f"\nOption 3 (Adaptive Dual Surge: >=2x baseline OR >=P95, min 5 req/min):")
    print(f"  Positive Class (Surge / P95 Tail): {t3_pos / total_eval_points * 100:.2f}% ({t3_pos:,})")
    print(f"  Negative Class (Normal):           {100 - t3_pos / total_eval_points * 100:.2f}%")

    print("\n" + "=" * 80)
    print("[+] TARGET DESIGN ANALYSIS COMPLETED.")
    print("=" * 80)


if __name__ == "__main__":
    analyze_target_distributions()
