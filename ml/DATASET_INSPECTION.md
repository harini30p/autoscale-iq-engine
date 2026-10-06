# Azure Functions 2019 Dataset Inspection Report

**Milestone 2 — Step 1: Dataset Acquisition and Inspection**  
**Project:** AutoScale IQ — ML-Powered Self-Optimizing Web Application  
**Trace:** Microsoft / Azure Public Dataset — Azure Functions Trace 2019  

---

## 1. Dataset Source and Acquisition

- **Official Source:** [Azure Public Dataset (GitHub)](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)
- **Publication:** USENIX ATC 2020: *"Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider"* (Shahrad et al.)
- **Download URL:** `https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz`
- **Storage Location:** `ml/data/raw/` (excluded from git tracking via `.gitignore`)
- **License:** Creative Commons Attribution (CC-BY 4.0)

---

## 2. Archive and File Structure

- **Archive File:** `azurefunctions_dataset2019.tar.xz`
- **Compressed Archive Size:** `136.35 MB`
- **Total Extracted CSV Files:** `40 files`
- **Total Uncompressed Size:** `2,041.67 MB` (~2.04 GB)

### Directory and File Inventory:
```text
ml/data/raw/
├── invocations_per_function_md.anon.d01.csv ... d14.csv   (14 files, Days 01–14)
├── function_durations_percentiles.anon.d01.csv ... d14.csv (14 files, Days 01–14)
└── app_memory_percentiles.anon.d01.csv ... d12.csv        (12 files, Days 01–12)
```
*Note: As officially documented by Microsoft in the ATC'20 release, memory statistics are provided for Days 01–12; Days 13 and 14 memory data are omitted by design in the public release.*

---

## 3. Dataset Schemas and Structural Analysis

### A. Function Invocation Counts (`invocations_per_function_md.anon.d*.csv`)
- **Granularity:** Function-level invocation count per minute over a 24-hour period (1,440 minutes).
- **Representative File (Day 01):** 46,412 rows × 1,444 columns.
- **Columns:**
  - `HashOwner` (*string/hash*): Unique identifier for the application owner/subscription.
  - `HashApp` (*string/hash*): Unique identifier for the Function App.
  - `HashFunction` (*string/hash*): Unique identifier for the function within the app.
  - `Trigger` (*categorical string*): Invocation trigger type.
  - `1` through `1440` (*integer*): Invocations during minute $1 \dots 1440$.
- **Trigger Distribution (Day 01):**
  - `timer`: 17,405 (37.5%)
  - `http`: 15,933 (34.3%)
  - `queue`: 6,924 (14.9%)
  - `orchestration`: 3,287 (7.1%)
  - `event`: 1,245 (2.7%)
  - `storage`: 949 (2.0%)
  - `others`: 669 (1.4%)
- **Quality Checks:** 0 missing values, 0 duplicate rows.

### B. Execution Duration Distributions (`function_durations_percentiles.anon.d*.csv`)
- **Granularity:** Daily aggregated execution duration statistics per function.
- **Representative File (Day 01):** 49,728 rows × 14 columns.
- **Columns (14):**
  - `HashOwner`, `HashApp`, `HashFunction`
  - `Average` (*float, ms*): Mean execution duration.
  - `Count` (*integer*): Total function executions observed during the day.
  - `Minimum` (*float, ms*): Minimum execution duration.
  - `Maximum` (*float, ms*): Maximum execution duration.
  - Percentiles: `percentile_Average_0`, `percentile_Average_1`, `percentile_Average_25`, `percentile_Average_50`, `percentile_Average_75`, `percentile_Average_99`, `percentile_Average_100`.
- **Quality Checks:** 0 missing values, 16 duplicate `(HashApp, HashFunction)` records across different owners.

### C. Application Memory Allocation (`app_memory_percentiles.anon.d*.csv`)
- **Granularity:** Daily aggregated memory allocation statistics per application.
- **Representative File (Day 01):** 17,275 rows × 12 columns.
- **Columns (12):**
  - `HashOwner`, `HashApp`
  - `SampleCount` (*integer*): Number of memory samples taken.
  - `AverageAllocatedMb` (*float, MB*): Average allocated memory.
  - Percentiles: `AverageAllocatedMb_pct1`, `AverageAllocatedMb_pct5`, `AverageAllocatedMb_pct25`, `AverageAllocatedMb_pct50`, `AverageAllocatedMb_pct75`, `AverageAllocatedMb_pct95`, `AverageAllocatedMb_pct99`, `AverageAllocatedMb_pct100`.
- **Quality Checks:** 0 missing values, 10 duplicate `HashApp` records.

---

## 4. Workload Statistics and Distribution

### Invocation Workload Profile (Day 01):
- **Total Invocations:** 909,783,379 requests across 46,412 functions.
- **Total Function-Minute Cells:** 66,833,280 observations.
- **Sparsity:** 84.98% of function-minute observations are 0 (idle minutes).
- **Non-Zero Invocations:**
  - Mean: $90.64\text{ calls/min}$
  - Median: $1.0\text{ call/min}$
  - Standard Deviation: $583.6$
  - 95th Percentile: $124.0\text{ calls/min}$
  - 99th Percentile: $1,165.0\text{ calls/min}$
  - 99.9th Percentile: $14,608.0\text{ calls/min}$
  - Maximum observed: $151,835\text{ calls/min}$
- **System-Wide Aggregate Load:**
  - Mean: $631,794\text{ req/min}$
  - Min: $501,568\text{ req/min}$
  - Max: $801,572\text{ req/min}$
  - Std Dev: $61,178\text{ req/min}$

### Execution Duration Profile (in milliseconds):
- **Mean Average Duration:** $9,504\text{ ms}$ (std: $40,061\text{ ms}$)
- **Median Average Duration:** $656\text{ ms}$
- **75th Percentile Duration:** $3,012\text{ ms}$
- **Maximum Execution Time:** $5,771,903\text{ ms}$ (~96 minutes)

### Application Memory Profile (in Megabytes):
- **Mean Average Allocated Memory:** $160.12\text{ MB}$ (std: $65.19\text{ MB}$)
- **Median Memory:** $142.0\text{ MB}$
- **Minimum / Maximum:** $52.0\text{ MB}$ / $1,227.0\text{ MB}$

---

## 5. Cross-Dataset Entity Consistency & Anomaly Findings

### Entity Identifier Overlap:
- **Application Level (`HashApp`):**
  - Overlap between Invocations and Durations: **99.7%** (17,532 / 17,577 apps match).
  - Overlap between Invocations and Memory: **97.6%** (17,155 / 17,577 apps match).
- **Function Level (`HashApp`, `HashFunction`):**
  - Overlap between Invocations and Durations: **98.7%** (45,790 / 46,412 functions match).

### Detected Data Anomalies:
1. **Negative Duration Anomaly in Raw Trace:**
   - The raw duration file contains negative values for `Minimum` (down to $-1,027,768\text{ ms}$) and `Average` (down to $-41,692\text{ ms}$). This is a documented telemetry clock skew / timestamp ordering artifact in the Azure production collectors.
   - *Action required during ML preprocessing:* Filter out or clamp non-positive durations prior to feature calculation.
2. **Missing Days 13 & 14 in Memory Data:**
   - Invocation and Duration data cover Days 01–14 (14 days), whereas Memory data covers Days 01–12 (12 days).
   - *Action required during ML preprocessing:* Use Days 01–12 for joint multimodal features, or use Days 13–14 with forward-filled/mean-imputed memory baselines.
3. **Absence of Native Host Metrics:**
   - The dataset does **not** contain CPU utilization, active user counts, database query latency, or raw server load. These must **not** be fabricated.

---

## 6. Temporal Reconstruction Methodology

The dataset does not contain calendar timestamps (e.g. `2019-07-15 14:30:00`), but has exact relative minute offsets:
$$\text{Day Index} \in [1 \dots 14], \quad \text{Minute Index} \in [1 \dots 1440]$$

We reconstruct continuous time series via:
$$\text{Global Minute } t = (\text{Day Index} - 1) \times 1440 + \text{Minute Index}$$
- **Total Trace Timeline:** $14 \times 1440 = 20,160\text{ minutes}$ (continuous 14-day workload).
- **Cyclical Encoding:**
  $$\text{Minute\_Sin} = \sin\left(\frac{2\pi \times \text{Minute}}{1440}\right), \quad \text{Minute\_Cos} = \cos\left(\frac{2\pi \times \text{Minute}}{1440}\right)$$

---

## 7. Usable ML Candidate Features

Derived **strictly** from authentic telemetry available in the dataset:

| Category | Feature Name | Description | Formula / Source |
|---|---|---|---|
| **Volume** | `invocation_count_t` | Instantaneous invocations in current minute | Column $t$ |
| **Trend / Rolling** | `rolling_mean_5m`, `rolling_mean_15m` | Short-term rolling average invocations | $\frac{1}{k}\sum_{i=0}^{k-1} \text{inv}_{t-i}$ |
| **Trend / Rolling** | `rolling_max_15m`, `rolling_max_60m` | Peak traffic observed in recent window | $\max(\text{inv}_{t-k \dots t})$ |
| **Velocity** | `invocation_rate_delta` | 1-minute and 5-minute traffic acceleration | $\text{inv}_t - \text{inv}_{t-1}$ |
| **Volatility** | `rolling_std_15m` | Standard deviation of recent invocations | $\sigma(\text{inv}_{t-15 \dots t})$ |
| **Workload Type** | `trigger_http`, `trigger_timer`, `trigger_queue` | One-hot encoded trigger category | Trigger column |
| **Execution Profile**| `func_avg_duration_ms`, `func_p95_duration_ms` | Historical function execution duration | Duration dataset |
| **Memory Profile** | `app_avg_memory_mb`, `app_p95_memory_mb` | Baseline application memory footprint | Memory dataset |
| **Temporal** | `minute_of_day`, `minute_sin`, `minute_cos` | Cyclical time-of-day feature | Minute column index |

---

## 8. Defensible Candidate Target Definitions (For Milestone 2 Review)

To remain methodologically rigorous without fabricating nonexistent CPU or DB latency metrics, we propose three candidate definitions for high-load / risk prediction:

### Candidate 1: Future Workload Surge (`Target_WorkloadSurge`)
- **Definition:** Predict whether average traffic in the future prediction window $[t+1 \dots t+5]$ will exceed the 95th percentile threshold of the application or $>3\times$ the recent 15-minute moving baseline:
  $$Y_t = \mathbb{I}\left( \text{Mean}(\text{inv}_{t+1 \dots t+5}) \ge \max(P_{95}, 3 \times \overline{\text{inv}}_{t-15 \dots t}) \right)$$
- **Pros:** Directly computable from continuous 1-minute invocation time series; directly relevant to proactive capacity management.

### Candidate 2: Heavy-Function Concurrency Spike (`Target_HeavyExecutionSpike`)
- **Definition:** Predict when high-duration functions (duration $> 2,000\text{ ms}$) experience concurrent invocation surges in window $[t+1 \dots t+5]$:
  $$Y_t = \mathbb{I}\left( \sum_{f \in \text{Heavy}} \text{inv}_{f, t+1 \dots t+5} > \text{Threshold} \right)$$
- **Pros:** Captures true execution bottleneck risk by weighting invocations by execution duration profile.

### Candidate 3: Composite Workload & Memory Stress (`Target_CompositeStress`)
- **Definition:** Joint threshold of sudden invocation spike on an application whose memory allocation profile is in the upper quartile ($> 187\text{ MB}$).
- **Pros:** Incorporates both traffic dynamics and application resource intensity.

---

## 9. Reproducibility Script

The inspection is automated and reproducible via:
```bash
python ml/inspect_dataset.py
```
Outputs complete summary logs, dataset verification, schema inspection, and correlation tables without manual CSV manipulation.
