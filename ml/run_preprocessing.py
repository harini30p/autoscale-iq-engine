"""
Batch Preprocessing Script for AutoScale IQ (Milestone 2 - Step 5).
Extracts:
- Train: Days 01–08 (10:1 negative subsample)
- Val:   Days 09–11 (Natural distribution)
- Test:  Days 12–14 (Natural distribution)
"""

import time
from pathlib import Path
from ml.preprocess import process_dataset_split

def main():
    t_start = time.perf_counter()
    print("=" * 80)
    print("STARTING FULL PREPROCESSING PIPELINE")
    print("=" * 80)

    # 1. Train Split (Days 01–08)
    process_dataset_split(
        split_name="train",
        day_range=(1, 8),
        subsample_negatives_ratio=10.0,
        include_metadata=False
    )

    # 2. Validation Split (Days 09–11)
    # Check if Day 09 is already generated
    val_p9 = Path("ml/data/processed/val/day_09.parquet")
    if not val_p9.exists() or val_p9.stat().st_size < 100 * 1024 * 1024:
        process_dataset_split(
            split_name="val",
            day_range=(9, 11),
            subsample_negatives_ratio=None,
            include_metadata=False
        )
    else:
        print("\n[*] Processing VAL split (Days 10 to 11)... (Day 09 already present)")
        process_dataset_split(
            split_name="val",
            day_range=(10, 11),
            subsample_negatives_ratio=None,
            include_metadata=False
        )

    # 3. Test Split (Days 12–14)
    process_dataset_split(
        split_name="test",
        day_range=(12, 14),
        subsample_negatives_ratio=None,
        include_metadata=False
    )

    print("\n" + "=" * 80)
    print(f"[+] PREPROCESSING PIPELINE COMPLETED IN {time.perf_counter() - t_start:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
