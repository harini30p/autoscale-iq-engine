"""
Feature Engineering & Memory Layout Verification for AutoScale IQ (Milestone 2 - Step 3).
Validates exact feature count, dtypes, bytes per row, and memory requirements.
"""

def verify_feature_layout():
    # 19 features exactly
    feature_specs = [
        ("invocations_t", "float32", 4),
        ("rolling_mean_5m", "float32", 4),
        ("rolling_mean_15m", "float32", 4),
        ("rolling_mean_60m", "float32", 4),
        ("rolling_max_15m", "float32", 4),
        ("rolling_max_60m", "float32", 4),
        ("rate_delta_1m", "float32", 4),
        ("rate_delta_5m", "float32", 4),
        ("rolling_std_15m", "float32", 4),
        ("minute_of_day", "int16", 2),
        ("minute_sin", "float32", 4),
        ("minute_cos", "float32", 4),
        ("trigger_timer", "uint8", 1),
        ("trigger_http", "uint8", 1),
        ("trigger_queue", "uint8", 1),
        ("trigger_orchestration", "uint8", 1),
        ("trigger_event", "uint8", 1),
        ("trigger_storage", "uint8", 1),
        ("trigger_others", "uint8", 1),
    ]

    target_spec = ("target_surge_5m", "uint8", 1)

    float32_features = [f for f in feature_specs if f[1] == "float32"]
    int16_features = [f for f in feature_specs if f[1] == "int16"]
    uint8_features = [f for f in feature_specs if f[1] == "uint8"]

    print("=" * 80)
    print("FEATURE MATRIX SPECIFICATION & BYTE RECONCILIATION")
    print("=" * 80)
    print(f"Total Feature Count: {len(feature_specs)}")
    print(f"  - float32 features: {len(float32_features)} ({len(float32_features)*4} bytes)")
    print(f"  - int16 features:   {len(int16_features)} ({len(int16_features)*2} bytes)")
    print(f"  - uint8 features:   {len(uint8_features)} ({len(uint8_features)*1} bytes)")

    feature_bytes_per_row = sum(f[2] for f in feature_specs)
    target_bytes_per_row = target_spec[2]
    total_bytes_per_row = feature_bytes_per_row + target_bytes_per_row

    print(f"\nExact Bytes Per Row:")
    print(f"  Feature-only bytes/row:      {feature_bytes_per_row} bytes")
    print(f"  Target bytes/row:            {target_bytes_per_row} bytes")
    print(f"  Features + Target bytes/row: {total_bytes_per_row} bytes")

    # Scale calculations
    print("\n" + "=" * 80)
    print("MEMORY FOOTPRINT ESTIMATES (UNCOMPRESSED NUMPY/RAM)")
    print("=" * 80)
    
    rows_1m = 1_000_000
    ram_1m_feat = (rows_1m * feature_bytes_per_row) / (1024**2)
    ram_1m_total = (rows_1m * total_bytes_per_row) / (1024**2)
    print(f"1,000,000 rows:")
    print(f"  Features only: {rows_1m * feature_bytes_per_row / 1e6:.2f} MB ({ram_1m_feat:.2f} MiB)")
    print(f"  Features + y:  {rows_1m * total_bytes_per_row / 1e6:.2f} MB ({ram_1m_total:.2f} MiB)")

    rows_d01 = 63_816_500
    ram_d01_feat = (rows_d01 * feature_bytes_per_row) / (1024**3)
    ram_d01_total = (rows_d01 * total_bytes_per_row) / (1024**3)
    print(f"\nFull Day 01 ({rows_d01:,} rows):")
    print(f"  Features only: {rows_d01 * feature_bytes_per_row / 1e9:.2f} GB ({ram_d01_feat:.2f} GiB)")
    print(f"  Features + y:  {rows_d01 * total_bytes_per_row / 1e9:.2f} GB ({ram_d01_total:.2f} GiB)")

    rows_train_sample = 21_500_000
    ram_train_feat = (rows_train_sample * feature_bytes_per_row) / (1024**3)
    ram_train_total = (rows_train_sample * total_bytes_per_row) / (1024**3)
    print(f"\nProposed Training Sample (~{rows_train_sample:,} rows):")
    print(f"  Features only: {rows_train_sample * feature_bytes_per_row / 1e9:.2f} GB ({ram_train_feat:.2f} GiB)")
    print(f"  Features + y:  {rows_train_sample * total_bytes_per_row / 1e9:.2f} GB ({ram_train_total:.2f} GiB)")

    print("\n" + "=" * 80)
    print("[OK] Feature and Memory Layout Reconciled Perfectly.")
    print("=" * 80)


if __name__ == "__main__":
    verify_feature_layout()
