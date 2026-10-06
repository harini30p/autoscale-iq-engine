"""
Memory-Safe Feature Extraction Pipeline for AutoScale IQ (Milestone 2 - Step 5).
Processes raw Azure Functions 2019 trace CSV files day-by-day, constructing the locked 19 features
and binary surge target with zero temporal leakage.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from scipy.ndimage import maximum_filter1d

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "ml" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"

# Locked 19 Feature Names
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

TRIGGER_CATEGORIES = [
    "timer",
    "http",
    "queue",
    "orchestration",
    "event",
    "storage",
    "others",
]

MINUTE_COLS = [str(i) for i in range(1, 1441)]


def extract_features_from_matrix(
    mat: np.ndarray,
    triggers: np.ndarray,
    t_start: int = 60,
    t_end: int = 1435
) -> Dict[str, np.ndarray]:
    """Extract features and target directly from a 2D float32 matrix (n_funcs, 1440)."""
    n_funcs = len(mat)
    eval_len = t_end - t_start
    t_indices = np.arange(t_start, t_end)

    padded_cumsum = np.pad(np.cumsum(mat, axis=1), ((0, 0), (1, 0)), mode="constant")
    padded_cumsum_sq = np.pad(np.cumsum(mat**2, axis=1), ((0, 0), (1, 0)), mode="constant")

    max15 = maximum_filter1d(mat, size=15, axis=1, origin=7)
    max60 = maximum_filter1d(mat, size=60, axis=1, origin=29)

    roll_mean_15m_all = (padded_cumsum[:, t_indices + 1] - padded_cumsum[:, t_indices - 14]) / 15.0
    future_5m_avg = (padded_cumsum[:, t_indices + 6] - padded_cumsum[:, t_indices + 1]) / 5.0
    target_mask = (future_5m_avg >= 2.0 * roll_mean_15m_all) & (future_5m_avg >= 5.0)

    sel_f = np.repeat(np.arange(n_funcs), eval_len)
    sel_t_mat = np.tile(t_indices, n_funcs)
    sel_t_eval = np.tile(np.arange(eval_len), n_funcs)

    inv_t = mat[sel_f, sel_t_mat]
    roll_mean_5m = (padded_cumsum[sel_f, sel_t_mat + 1] - padded_cumsum[sel_f, sel_t_mat - 4]) / 5.0
    roll_mean_15m = (padded_cumsum[sel_f, sel_t_mat + 1] - padded_cumsum[sel_f, sel_t_mat - 14]) / 15.0
    roll_mean_60m = (padded_cumsum[sel_f, sel_t_mat + 1] - padded_cumsum[sel_f, sel_t_mat - 59]) / 60.0

    sum_sq_15m = (padded_cumsum_sq[sel_f, sel_t_mat + 1] - padded_cumsum_sq[sel_f, sel_t_mat - 14]) / 15.0
    var_15m = np.maximum(0.0, sum_sq_15m - roll_mean_15m**2)
    roll_std_15m = np.sqrt(var_15m)

    roll_max_15m = max15[sel_f, sel_t_mat]
    roll_max_60m = max60[sel_f, sel_t_mat]

    rate_delta_1m = inv_t - mat[sel_f, sel_t_mat - 1]
    rate_delta_5m = inv_t - mat[sel_f, sel_t_mat - 5]

    minute_of_day = (sel_t_mat + 1).astype(np.int16)
    minute_sin = np.sin(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)
    minute_cos = np.cos(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)

    sel_triggers = triggers[sel_f]
    trig_dict = {}
    for cat in TRIGGER_CATEGORIES:
        if cat == "others":
            trig_dict[f"trigger_{cat}"] = (~np.isin(sel_triggers, TRIGGER_CATEGORIES[:-1])).astype(np.uint8)
        else:
            trig_dict[f"trigger_{cat}"] = (sel_triggers == cat).astype(np.uint8)

    target_vec = target_mask[sel_f, sel_t_eval].astype(np.uint8)

    return {
        "invocations_t": inv_t,
        "rolling_mean_5m": roll_mean_5m,
        "rolling_mean_15m": roll_mean_15m,
        "rolling_mean_60m": roll_mean_60m,
        "rolling_max_15m": roll_max_15m,
        "rolling_max_60m": roll_max_60m,
        "rate_delta_1m": rate_delta_1m,
        "rate_delta_5m": rate_delta_5m,
        "rolling_std_15m": roll_std_15m,
        "minute_of_day": minute_of_day,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        **trig_dict,
        "target_surge_5m": target_vec,
    }


def extract_day_arrays(
    day_num: int,
    subsample_negatives_ratio: Optional[float] = None,
    random_seed: int = 42,
    include_metadata: bool = False
) -> Dict[str, np.ndarray]:
    """
    Fast, memory-safe extraction of locked 19 features and target for a single day.
    Uses vectorization, SciPy C-filters, and index-level subsampling.
    """
    csv_path = RAW_DIR / f"invocations_per_function_md.anon.d{day_num:02d}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw trace file not found: {csv_path}")

    tbl = pv.read_csv(csv_path)
    mat = np.column_stack([tbl[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in MINUTE_COLS])
    triggers_raw = tbl["Trigger"].to_numpy(zero_copy_only=False).astype(str)
    triggers = np.char.lower(triggers_raw)

    n_funcs = len(mat)
    t_start, t_end = 60, 1435
    eval_len = t_end - t_start  # 1375 minutes
    t_indices = np.arange(t_start, t_end)

    # 1. Padded cumulative sums (tau <= t)
    padded_cumsum = np.pad(np.cumsum(mat, axis=1), ((0, 0), (1, 0)), mode="constant")
    padded_cumsum_sq = np.pad(np.cumsum(mat**2, axis=1), ((0, 0), (1, 0)), mode="constant")

    # 2. Vectorized 1D rolling filters
    max15 = maximum_filter1d(mat, size=15, axis=1, origin=7)
    max60 = maximum_filter1d(mat, size=60, axis=1, origin=29)

    # 3. Target calculation: future_5m >= 2.0 * roll15 AND future_5m >= 5.0
    roll_mean_15m_all = (padded_cumsum[:, t_indices + 1] - padded_cumsum[:, t_indices - 14]) / 15.0
    future_5m_avg = (padded_cumsum[:, t_indices + 6] - padded_cumsum[:, t_indices + 1]) / 5.0
    target_mask = (future_5m_avg >= 2.0 * roll_mean_15m_all) & (future_5m_avg >= 5.0)

    # 4. Selection of rows
    if subsample_negatives_ratio is not None and subsample_negatives_ratio > 0:
        pos_f, pos_t = np.where(target_mask)
        n_pos = len(pos_f)
        n_neg_target = int(n_pos * subsample_negatives_ratio)

        rng = np.random.default_rng(random_seed + day_num)
        cand_f = rng.integers(0, n_funcs, size=int(n_neg_target * 1.05) + 10)
        cand_t = rng.integers(0, eval_len, size=int(n_neg_target * 1.05) + 10)
        is_neg = ~target_mask[cand_f, cand_t]
        cand_f = cand_f[is_neg][:n_neg_target]
        cand_t = cand_t[is_neg][:n_neg_target]

        sel_f = np.concatenate([pos_f, cand_f])
        sel_t_eval = np.concatenate([pos_t, cand_t])
        sel_t_mat = t_indices[sel_t_eval]
    else:
        sel_f = np.repeat(np.arange(n_funcs), eval_len)
        sel_t_mat = np.tile(t_indices, n_funcs)
        sel_t_eval = np.tile(np.arange(eval_len), n_funcs)

    # 5. Extract features strictly on sampled indices (all tau <= t)
    inv_t = mat[sel_f, sel_t_mat]
    roll_mean_5m = (padded_cumsum[sel_f, sel_t_mat + 1] - padded_cumsum[sel_f, sel_t_mat - 4]) / 5.0
    roll_mean_15m = (padded_cumsum[sel_f, sel_t_mat + 1] - padded_cumsum[sel_f, sel_t_mat - 14]) / 15.0
    roll_mean_60m = (padded_cumsum[sel_f, sel_t_mat + 1] - padded_cumsum[sel_f, sel_t_mat - 59]) / 60.0

    sum_sq_15m = (padded_cumsum_sq[sel_f, sel_t_mat + 1] - padded_cumsum_sq[sel_f, sel_t_mat - 14]) / 15.0
    var_15m = np.maximum(0.0, sum_sq_15m - roll_mean_15m**2)
    roll_std_15m = np.sqrt(var_15m)

    roll_max_15m = max15[sel_f, sel_t_mat]
    roll_max_60m = max60[sel_f, sel_t_mat]

    rate_delta_1m = inv_t - mat[sel_f, sel_t_mat - 1]
    rate_delta_5m = inv_t - mat[sel_f, sel_t_mat - 5]

    minute_of_day = (sel_t_mat + 1).astype(np.int16)
    minute_sin = np.sin(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)
    minute_cos = np.cos(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)

    sel_triggers = triggers[sel_f]
    trig_dict = {}
    for cat in TRIGGER_CATEGORIES:
        if cat == "others":
            trig_dict[f"trigger_{cat}"] = (~np.isin(sel_triggers, TRIGGER_CATEGORIES[:-1])).astype(np.uint8)
        else:
            trig_dict[f"trigger_{cat}"] = (sel_triggers == cat).astype(np.uint8)

    target_vec = target_mask[sel_f, sel_t_eval].astype(np.uint8)

    feature_dict = {
        "invocations_t": inv_t,
        "rolling_mean_5m": roll_mean_5m,
        "rolling_mean_15m": roll_mean_15m,
        "rolling_mean_60m": roll_mean_60m,
        "rolling_max_15m": roll_max_15m,
        "rolling_max_60m": roll_max_60m,
        "rate_delta_1m": rate_delta_1m,
        "rate_delta_5m": rate_delta_5m,
        "rolling_std_15m": roll_std_15m,
        "minute_of_day": minute_of_day,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        **trig_dict,
        "target_surge_5m": target_vec,
    }

    if include_metadata:
        apps_raw = tbl["HashApp"].to_numpy(zero_copy_only=False).astype(str)
        funcs_raw = tbl["HashFunction"].to_numpy(zero_copy_only=False).astype(str)
        feature_dict["HashApp"] = apps_raw[sel_f]
        feature_dict["HashFunction"] = funcs_raw[sel_f]
        feature_dict["day"] = np.full(len(target_vec), day_num, dtype=np.int8)

    return feature_dict


def extract_day_features(
    day_num: int,
    subsample_negatives_ratio: Optional[float] = None,
    random_seed: int = 42,
    include_metadata: bool = False
) -> pd.DataFrame:
    """Extract features for a day and return as DataFrame."""
    arrays = extract_day_arrays(
        day_num=day_num,
        subsample_negatives_ratio=subsample_negatives_ratio,
        random_seed=random_seed,
        include_metadata=include_metadata
    )
    return pd.DataFrame(arrays)


def save_day_parquet(
    day_num: int,
    output_dir: Path,
    subsample_negatives_ratio: Optional[float] = None,
    chunk_funcs: int = 5000,
    random_seed: int = 42,
    include_metadata: bool = False
) -> Path:
    """Save extracted day features to compressed Parquet using streaming chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"day_{day_num:02d}.parquet"

    if subsample_negatives_ratio is not None and subsample_negatives_ratio > 0:
        arrays = extract_day_arrays(
            day_num=day_num,
            subsample_negatives_ratio=subsample_negatives_ratio,
            random_seed=random_seed,
            include_metadata=include_metadata
        )
        table = pa.Table.from_pydict(arrays)
        pq.write_table(table, out_file, compression="snappy")
        return out_file

    # Streaming chunks for natural distribution
    csv_path = RAW_DIR / f"invocations_per_function_md.anon.d{day_num:02d}.csv"
    tbl = pv.read_csv(csv_path)
    n_total = tbl.num_rows
    triggers_raw = tbl["Trigger"].to_numpy(zero_copy_only=False).astype(str)
    triggers = np.char.lower(triggers_raw)

    apps_raw = tbl["HashApp"].to_numpy(zero_copy_only=False).astype(str) if include_metadata else None
    funcs_raw = tbl["HashFunction"].to_numpy(zero_copy_only=False).astype(str) if include_metadata else None

    writer = None
    t_start, t_end = 60, 1435
    eval_len = t_end - t_start
    t_indices = np.arange(t_start, t_end)

    for start_idx in range(0, n_total, chunk_funcs):
        end_idx = min(start_idx + chunk_funcs, n_total)
        chunk_tbl = tbl.slice(start_idx, end_idx - start_idx)
        n_c = end_idx - start_idx
        mat = np.column_stack([chunk_tbl[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in MINUTE_COLS])

        padded_cumsum = np.pad(np.cumsum(mat, axis=1), ((0, 0), (1, 0)), mode="constant")
        padded_cumsum_sq = np.pad(np.cumsum(mat**2, axis=1), ((0, 0), (1, 0)), mode="constant")
        max15 = maximum_filter1d(mat, size=15, axis=1, origin=7)
        max60 = maximum_filter1d(mat, size=60, axis=1, origin=29)

        inv_t = mat[:, t_indices].ravel()
        roll_mean_5m = ((padded_cumsum[:, t_indices + 1] - padded_cumsum[:, t_indices - 4]) / 5.0).ravel()
        roll_mean_15m = ((padded_cumsum[:, t_indices + 1] - padded_cumsum[:, t_indices - 14]) / 15.0).ravel()
        roll_mean_60m = ((padded_cumsum[:, t_indices + 1] - padded_cumsum[:, t_indices - 59]) / 60.0).ravel()

        sum_sq_15m = (padded_cumsum_sq[:, t_indices + 1] - padded_cumsum_sq[:, t_indices - 14]) / 15.0
        var_15m = np.maximum(0.0, sum_sq_15m - (roll_mean_15m.reshape(n_c, eval_len))**2)
        roll_std_15m = np.sqrt(var_15m).ravel()

        roll_max_15m = max15[:, t_indices].ravel()
        roll_max_60m = max60[:, t_indices].ravel()
        rate_delta_1m = (mat[:, t_indices] - mat[:, t_indices - 1]).ravel()
        rate_delta_5m = (mat[:, t_indices] - mat[:, t_indices - 5]).ravel()

        min_vals = (t_indices + 1).astype(np.int16)
        minute_of_day = np.tile(min_vals, n_c)
        minute_sin = np.tile(np.sin(2.0 * np.pi * min_vals / 1440.0).astype(np.float32), n_c)
        minute_cos = np.tile(np.cos(2.0 * np.pi * min_vals / 1440.0).astype(np.float32), n_c)

        c_trig = triggers[start_idx:end_idx]
        trig_dict = {}
        for cat in TRIGGER_CATEGORIES:
            if cat == "others":
                mask = ~np.isin(c_trig, TRIGGER_CATEGORIES[:-1])
            else:
                mask = (c_trig == cat)
            trig_dict[f"trigger_{cat}"] = np.repeat(mask.astype(np.uint8), eval_len)

        fut5 = (padded_cumsum[:, t_indices + 6] - padded_cumsum[:, t_indices + 1]) / 5.0
        tgt = ((fut5 >= 2.0 * roll_mean_15m.reshape(n_c, eval_len)) & (fut5 >= 5.0)).astype(np.uint8).ravel()

        chunk_dict = {
            "invocations_t": inv_t,
            "rolling_mean_5m": roll_mean_5m,
            "rolling_mean_15m": roll_mean_15m,
            "rolling_mean_60m": roll_mean_60m,
            "rolling_max_15m": roll_max_15m,
            "rolling_max_60m": roll_max_60m,
            "rate_delta_1m": rate_delta_1m,
            "rate_delta_5m": rate_delta_5m,
            "rolling_std_15m": roll_std_15m,
            "minute_of_day": minute_of_day,
            "minute_sin": minute_sin,
            "minute_cos": minute_cos,
            **trig_dict,
            "target_surge_5m": tgt,
        }

        if include_metadata and apps_raw is not None and funcs_raw is not None:
            chunk_dict["HashApp"] = np.repeat(apps_raw[start_idx:end_idx], eval_len)
            chunk_dict["HashFunction"] = np.repeat(funcs_raw[start_idx:end_idx], eval_len)
            chunk_dict["day"] = np.full(len(tgt), day_num, dtype=np.int8)

        chunk_tbl_pa = pa.Table.from_pydict(chunk_dict)
        if writer is None:
            writer = pq.ParquetWriter(out_file, chunk_tbl_pa.schema, compression="snappy")
        writer.write_table(chunk_tbl_pa)

    if writer is not None:
        writer.close()

    return out_file


def process_dataset_split(
    split_name: str,
    day_range: Tuple[int, int],
    subsample_negatives_ratio: Optional[float] = None,
    include_metadata: bool = False
) -> List[Path]:
    """Process a range of days into a specific split directory."""
    split_dir = PROCESSED_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    print(f"\n[*] Processing {split_name.upper()} split (Days {day_range[0]:02d} to {day_range[1]:02d})...")
    for day in range(day_range[0], day_range[1] + 1):
        t0 = time.perf_counter()
        out_path = save_day_parquet(
            day_num=day,
            output_dir=split_dir,
            subsample_negatives_ratio=subsample_negatives_ratio,
            include_metadata=include_metadata
        )
        elapsed = time.perf_counter() - t0
        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  [+] Saved {out_path.name} ({file_size_mb:.2f} MB in {elapsed:.2f}s)")
        generated_files.append(out_path)

    return generated_files
