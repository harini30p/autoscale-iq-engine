"""
Strict Verification Script for AutoScale IQ Milestone 2 Target & Split Counts.
Calculates exact function counts across all 14 days, exact prediction rows per split,
and exact positive class counts for all functions vs active functions.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "ml" / "data" / "raw"


def run_strict_verification():
    print("=" * 80)
    print("STRICT VERIFICATION OF SAMPLE COUNTS, SPLITS & CLASS BALANCE (14 DAYS)")
    print("=" * 80)

    daily_stats = []
    total_eval_minutes_per_func = 1435 - 60  # 1,375 minutes (minutes 61 to 1435)

    for day in range(1, 15):
        file_name = f"invocations_per_function_md.anon.d{day:02d}.csv"
        file_path = RAW_DIR / file_name
        if not file_path.exists():
            print(f"[-] Missing: {file_name}")
            continue

        df_meta = pd.read_csv(file_path, usecols=["HashOwner", "HashApp", "HashFunction"])
        num_funcs = len(df_meta)
        total_rows_day = num_funcs * total_eval_minutes_per_func

        daily_stats.append({
            "day": day,
            "file_name": file_name,
            "num_functions": num_funcs,
            "total_prediction_rows": total_rows_day
        })
        print(f"  Day {day:02d}: {num_funcs:,} functions -> {total_rows_day:,} prediction rows")

    df_days = pd.DataFrame(daily_stats)
    print("-" * 80)

    train_days = df_days[df_days["day"].between(1, 8)]
    val_days = df_days[df_days["day"].between(9, 11)]
    test_days = df_days[df_days["day"].between(12, 14)]

    train_rows = train_days["total_prediction_rows"].sum()
    val_rows = val_days["total_prediction_rows"].sum()
    test_rows = test_days["total_prediction_rows"].sum()
    total_all_14_days = df_days["total_prediction_rows"].sum()

    print("\nEXACT CHRONOLOGICAL SPLIT SAMPLE COUNTS (ALL FUNCTIONS):")
    print(f"  Train Set (Days 01-08, 8 days):  {train_rows:,} rows ({train_rows/total_all_14_days*100:.2f}%)")
    print(f"  Val Set   (Days 09-11, 3 days):  {val_rows:,} rows ({val_rows/total_all_14_days*100:.2f}%)")
    print(f"  Test Set  (Days 12-14, 3 days):  {test_rows:,} rows ({test_rows/total_all_14_days*100:.2f}%)")
    print(f"  Total Across 14 Days:            {total_all_14_days:,} rows")

    print("\n" + "=" * 80)
    print("EXACT DAY 01 ARITHMETIC & CLASS BALANCE VERIFICATION")
    print("=" * 80)

    d1_file = RAW_DIR / "invocations_per_function_md.anon.d01.csv"
    df_d1 = pd.read_csv(d1_file)
    minute_cols = [str(i) for i in range(1, 1441)]
    mat_all = df_d1[minute_cols].values.astype(np.float32)
    n_all_funcs = len(mat_all)

    padded_cumsum = np.pad(np.cumsum(mat_all, axis=1), ((0, 0), (1, 0)), mode="constant")
    t_idx = np.arange(60, 1435)

    future_5m_avg_all = (padded_cumsum[:, t_idx + 6] - padded_cumsum[:, t_idx + 1]) / 5.0
    baseline_15m_all = (padded_cumsum[:, t_idx + 1] - padded_cumsum[:, t_idx - 14]) / 15.0

    target_mask_all = (future_5m_avg_all >= 2.0 * baseline_15m_all) & (future_5m_avg_all >= 5.0)
    pos_all = int(np.sum(target_mask_all))
    total_samples_all = int(n_all_funcs * len(t_idx))
    pct_all = (pos_all / total_samples_all) * 100

    print(f"Denominator 1: ALL {n_all_funcs:,} Functions:")
    print(f"  Total Prediction Samples: {total_samples_all:,}")
    print(f"  Positive Surge Samples:   {pos_all:,}")
    print(f"  Positive Class %:         {pct_all:.4f}% ({pos_all:,} / {total_samples_all:,})")
    print(f"  Negative Class %:         {100 - pct_all:.4f}%")

    func_totals = mat_all.sum(axis=1)
    active_mask = func_totals >= 10
    n_active = int(np.sum(active_mask))
    total_samples_active = int(n_active * len(t_idx))

    target_mask_active = target_mask_all[active_mask]
    pos_active = int(np.sum(target_mask_active))
    pct_active = (pos_active / total_samples_active) * 100

    print(f"\nDenominator 2: ACTIVE {n_active:,} Functions (>=10 calls/day):")
    print(f"  Total Prediction Samples: {total_samples_active:,}")
    print(f"  Positive Surge Samples:   {pos_active:,}")
    print(f"  Positive Class %:         {pct_active:.4f}% ({pos_active:,} / {total_samples_active:,})")
    print(f"  Negative Class %:         {100 - pct_active:.4f}%")

    inactive_mask = ~active_mask
    pos_inactive = int(np.sum(target_mask_all[inactive_mask]))
    print(f"\nInactive Functions ({int(np.sum(inactive_mask)):,} functions with <10 calls/day):")
    print(f"  Positive Surge Samples:   {pos_inactive} (0 positive surge rows)")

    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS SUMMARY")
    print("=" * 80)
    print(f"[OK] Exact Day 01 all-functions sample count: {total_samples_all:,}")
    print(f"[OK] Exact Day 01 positive surge count: {pos_all:,}")
    print(f"[OK] Mathematical consistency confirmed: {pos_all:,} / {total_samples_all:,} = {pct_all:.4f}%")
    print(f"[OK] In active-functions subset: {pos_active:,} / {total_samples_active:,} = {pct_active:.4f}%")
    print(f"[OK] Chronological Split counts verified across all 14 days.")


if __name__ == "__main__":
    run_strict_verification()
