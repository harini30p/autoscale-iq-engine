"""
Azure Functions 2019 Dataset Acquisition and Inspection Script for AutoScale IQ (Milestone 2 - Step 1).
Downloads official Azure Functions 2019 Trace, extracts archive, and performs thorough schema,
statistical, temporal, and consistency inspection across invocation, duration, and memory records.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ml" / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
)
ARCHIVE_PATH = DATA_DIR / "azurefunctions_dataset2019.tar.xz"


def download_dataset():
    """Download the official Azure Functions 2019 dataset archive if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE_PATH.exists():
        print(f"[*] Downloading Azure Functions 2019 dataset from:\n    {ARCHIVE_URL}")
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded / total_size * 100
                sys.stdout.write(f"\r    Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r    Downloaded: {downloaded / (1024*1024):.1f} MB")
                sys.stdout.flush()

        urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE_PATH, reporthook=report_progress)
        print("\n[+] Download complete.")
    else:
        print(f"[+] Archive already exists at: {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size / (1024*1024):.2f} MB)")


def extract_dataset():
    """Extract archive into raw data directory."""
    print(f"[*] Checking archive contents...")
    with tarfile.open(ARCHIVE_PATH, "r:xz") as tar:
        members = tar.getmembers()
        print(f"[+] Total files in archive: {len(members)}")
        extracted_files = list(RAW_DIR.glob("*.csv"))
        if len(extracted_files) < 30:  # Expected ~40 CSV files (14 invocations + 14 durations + 12 memory)
            print(f"[*] Extracting archive into {RAW_DIR} ...")
            tar.extractall(path=RAW_DIR)
            print("[+] Extraction complete.")
        else:
            print(f"[+] Dataset already extracted ({len(extracted_files)} CSV files found in {RAW_DIR}).")


def inspect_archive_structure() -> Dict[str, Any]:
    """Inspect archive size, extracted files, sizes, and extensions."""
    archive_size_mb = ARCHIVE_PATH.stat().st_size / (1024 * 1024)
    all_files = list(RAW_DIR.glob("**/*"))
    csv_files = [f for f in all_files if f.is_file() and f.suffix == ".csv"]
    
    total_extracted_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    
    invocation_files = sorted([f for f in csv_files if "invocations_per_function" in f.name])
    duration_files = sorted([f for f in csv_files if "function_durations" in f.name])
    memory_files = sorted([f for f in csv_files if "app_memory" in f.name])

    print("\n" + "="*80)
    print("ARCHIVE & FILE STRUCTURE INSPECTION")
    print("="*80)
    print(f"Archive file: {ARCHIVE_PATH.name}")
    print(f"Archive size: {archive_size_mb:.2f} MB")
    print(f"Extracted CSV files count: {len(csv_files)}")
    print(f"Total extracted size: {total_extracted_size_mb:.2f} MB")
    print(f"  - Invocations files ({len(invocation_files)} files, days 01-14): {[f.name for f in invocation_files]}")
    print(f"  - Duration files    ({len(duration_files)} files, days 01-14): {[f.name for f in duration_files]}")
    print(f"  - Memory files      ({len(memory_files)} files, days 01-12): {[f.name for f in memory_files]}")

    return {
        "archive_size_mb": archive_size_mb,
        "csv_files_count": len(csv_files),
        "total_extracted_size_mb": total_extracted_size_mb,
        "invocation_files": invocation_files,
        "duration_files": duration_files,
        "memory_files": memory_files,
    }


def inspect_invocations(files: List[Path]) -> Dict[str, Any]:
    """Inspect invocation schema, nulls, duplicates, triggers, and workload statistics."""
    print("\n" + "="*80)
    print("1. INVOCATION DATASET INSPECTION")
    print("="*80)
    
    # Load day 01 as representative
    df_d1 = pd.read_csv(files[0])
    cols = list(df_d1.columns)
    
    print(f"Representative file: {files[0].name}")
    print(f"Row count: {len(df_d1):,}")
    print(f"Column count: {len(cols):,}")
    print(f"Metadata columns: {cols[:4]}")
    print(f"Minute columns sample: {cols[4:9]} ... {cols[-3:]}")
    print(f"Missing values count per metadata column:\n{df_d1[cols[:4]].isnull().sum().to_dict()}")
    print(f"Duplicate rows: {df_d1.duplicated(subset=['HashOwner', 'HashApp', 'HashFunction']).sum()}")
    print(f"Unique Owners: {df_d1['HashOwner'].nunique():,}")
    print(f"Unique Apps: {df_d1['HashApp'].nunique():,}")
    print(f"Unique Functions: {df_d1['HashFunction'].nunique():,}")
    print(f"Trigger categories distribution:\n{df_d1['Trigger'].value_counts().to_dict()}")

    # Compute aggregate statistics over minute columns across Day 01
    minute_cols = cols[4:]
    minute_matrix = df_d1[minute_cols].values
    
    total_invocations_d1 = minute_matrix.sum()
    zero_count = (minute_matrix == 0).sum()
    total_cells = minute_matrix.size
    pct_zeros = (zero_count / total_cells) * 100
    
    flat_nonzero = minute_matrix[minute_matrix > 0]
    
    print("\nWorkload Statistics (Day 01 - Function/Minute granularity):")
    print(f"  Total Invocations: {total_invocations_d1:,}")
    print(f"  Total Function-Minute observations: {total_cells:,}")
    print(f"  Zero-invocation observations: {zero_count:,} ({pct_zeros:.2f}%)")
    print(f"  Non-zero observations: {len(flat_nonzero):,} ({100 - pct_zeros:.2f}%)")
    print(f"  Mean invocations/min (all): {minute_matrix.mean():.4f}")
    print(f"  Mean invocations/min (non-zero): {flat_nonzero.mean():.4f}")
    print(f"  Median (non-zero): {np.median(flat_nonzero):.1f}")
    print(f"  Std Dev (all): {minute_matrix.std():.4f}")
    print(f"  Max invocations/min: {minute_matrix.max():,}")
    print(f"  95th percentile (non-zero): {np.percentile(flat_nonzero, 95):.1f}")
    print(f"  99th percentile (non-zero): {np.percentile(flat_nonzero, 99):.1f}")
    print(f"  99.9th percentile (non-zero): {np.percentile(flat_nonzero, 99.9):.1f}")

    # Aggregated system-wide / app-wide time series over 1440 minutes
    app_minute_series = df_d1.groupby("HashApp")[minute_cols].sum()
    system_minute_series = df_d1[minute_cols].sum(axis=0)

    print("\nSystem-Wide Workload Profile (Day 01 aggregate per minute):")
    print(f"  System minute volume mean: {system_minute_series.mean():.2f} req/min")
    print(f"  System minute volume min:  {system_minute_series.min():,} req/min")
    print(f"  System minute volume max:  {system_minute_series.max():,} req/min")
    print(f"  System minute volume std:  {system_minute_series.std():.2f}")

    return {
        "df_d1_rows": len(df_d1),
        "columns": cols,
        "unique_apps": df_d1["HashApp"].nunique(),
        "unique_functions": df_d1["HashFunction"].nunique(),
        "total_invocations_d1": total_invocations_d1,
        "pct_zeros": pct_zeros,
    }


def inspect_durations(files: List[Path]) -> Dict[str, Any]:
    """Inspect duration dataset schema, nulls, range of average/min/max, and percentiles."""
    print("\n" + "="*80)
    print("2. EXECUTION DURATION DATASET INSPECTION")
    print("="*80)

    df_dur = pd.read_csv(files[0])
    cols = list(df_dur.columns)

    print(f"Representative file: {files[0].name}")
    print(f"Row count: {len(df_dur):,}")
    print(f"Columns ({len(cols)}): {cols}")
    print(f"Missing values:\n{df_dur.isnull().sum().to_dict()}")
    print(f"Duplicate (HashApp, HashFunction): {df_dur.duplicated(subset=['HashApp', 'HashFunction']).sum()}")
    
    print("\nDuration Descriptive Statistics (in milliseconds):")
    stats_cols = ["Average", "Count", "Minimum", "Maximum", "Percentile_Average_50", "Percentile_Average_95", "Percentile_Average_99"]
    avail_cols = [c for c in stats_cols if c in df_dur.columns]
    desc = df_dur[avail_cols].describe()
    print(desc.to_string())

    return {
        "rows": len(df_dur),
        "columns": cols,
        "stats": desc.to_dict(),
    }


def inspect_memory(files: List[Path]) -> Dict[str, Any]:
    """Inspect memory dataset schema, nulls, range of average allocated MB, and percentiles."""
    print("\n" + "="*80)
    print("3. APPLICATION MEMORY DATASET INSPECTION")
    print("="*80)

    df_mem = pd.read_csv(files[0])
    cols = list(df_mem.columns)

    print(f"Representative file: {files[0].name}")
    print(f"Row count: {len(df_mem):,}")
    print(f"Columns ({len(cols)}): {cols}")
    print(f"Missing values:\n{df_mem.isnull().sum().to_dict()}")
    print(f"Duplicate (HashApp): {df_mem.duplicated(subset=['HashApp']).sum()}")

    print("\nMemory Descriptive Statistics (in Megabytes):")
    stats_cols = ["AverageAllocatedMb", "SampleCount", "PercentileAllocatedMb_50", "PercentileAllocatedMb_95", "PercentileAllocatedMb_99"]
    avail_cols = [c for c in stats_cols if c in df_mem.columns]
    desc = df_mem[avail_cols].describe()
    print(desc.to_string())

    return {
        "rows": len(df_mem),
        "columns": cols,
        "stats": desc.to_dict(),
    }


def inspect_consistency_and_correlation(
    inv_files: List[Path],
    dur_files: List[Path],
    mem_files: List[Path]
):
    """Check entity identifier overlap and consistency across invocation, duration, and memory traces."""
    print("\n" + "="*80)
    print("4. CROSS-DATASET CONSISTENCY & CORRELATION ANALYSIS")
    print("="*80)

    df_inv = pd.read_csv(inv_files[0])
    df_dur = pd.read_csv(dur_files[0])
    df_mem = pd.read_csv(mem_files[0])

    inv_owners = set(df_inv["HashOwner"].unique())
    inv_apps = set(df_inv["HashApp"].unique())
    inv_funcs = set(zip(df_inv["HashApp"], df_inv["HashFunction"]))

    dur_owners = set(df_dur["HashOwner"].unique())
    dur_apps = set(df_dur["HashApp"].unique())
    dur_funcs = set(zip(df_dur["HashApp"], df_dur["HashFunction"]))

    mem_owners = set(df_mem["HashOwner"].unique())
    mem_apps = set(df_mem["HashApp"].unique())

    print("Owner Overlap (Day 01):")
    print(f"  Invocation Owners: {len(inv_owners):,}")
    print(f"  Duration Owners:   {len(dur_owners):,}")
    print(f"  Memory Owners:     {len(mem_owners):,}")
    print(f"  Common across Invocation & Duration: {len(inv_owners.intersection(dur_owners)):,}")
    print(f"  Common across Invocation & Memory:   {len(inv_owners.intersection(mem_owners)):,}")

    print("\nApplication Overlap (Day 01):")
    print(f"  Invocation Apps: {len(inv_apps):,}")
    print(f"  Duration Apps:   {len(dur_apps):,}")
    print(f"  Memory Apps:     {len(mem_apps):,}")
    print(f"  Inv & Dur App Overlap: {len(inv_apps.intersection(dur_apps)):,} ({len(inv_apps.intersection(dur_apps))/len(inv_apps)*100:.1f}%)")
    print(f"  Inv & Mem App Overlap: {len(inv_apps.intersection(mem_apps)):,} ({len(inv_apps.intersection(mem_apps))/len(inv_apps)*100:.1f}%)")

    print("\nFunction (App, Function) Overlap (Day 01):")
    print(f"  Invocation (App, Func) pairs: {len(inv_funcs):,}")
    print(f"  Duration (App, Func) pairs:   {len(dur_funcs):,}")
    print(f"  Inv & Dur Function Overlap:   {len(inv_funcs.intersection(dur_funcs)):,} ({len(inv_funcs.intersection(dur_funcs))/len(inv_funcs)*100:.1f}%)")

    # Day count across datasets
    print("\nDataset Span & Daily File Availability:")
    print(f"  Invocation files count: {len(inv_files)} days (days 01 to 14)")
    print(f"  Duration files count:   {len(dur_files)} days (days 01 to 14)")
    print(f"  Memory files count:     {len(mem_files)} days (days 01 to 12) -> Note: Days 13 and 14 are absent by design in official trace")


def main():
    download_dataset()
    extract_dataset()
    meta = inspect_archive_structure()
    inspect_invocations(meta["invocation_files"])
    inspect_durations(meta["duration_files"])
    inspect_memory(meta["memory_files"])
    inspect_consistency_and_correlation(
        meta["invocation_files"],
        meta["duration_files"],
        meta["memory_files"]
    )
    print("\n" + "="*80)
    print("[+] DATASET INSPECTION SCRIPT COMPLETED SUCCESSFULLY.")
    print("="*80)


if __name__ == "__main__":
    main()
