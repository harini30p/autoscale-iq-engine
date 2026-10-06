# Machine Learning Target & Preprocessing Design

**Milestone 2 — Step 2: Target & Preprocessing Design**  
**Project:** AutoScale IQ — ML-Powered Self-Optimizing Web Application  
**Objective:** Workload Surge Prediction on Azure Functions Trace 2019  

---

## 1. Prediction Objective

The primary objective of the ML subsystem is to provide an early-warning signal for the Safety Controller:
$$\text{Past Workload History } (t' \le t) \xrightarrow{\quad \text{ML Classifier} \quad} \widehat{P}(\text{Workload Surge in } [t+1, t+5]) \xrightarrow{\quad \text{Signal} \quad} \text{Safety Controller}$$

- **Horizon:** 5-minute lookahead window ($[t+1 \dots t+5]$).
- **Goal:** Predict whether a serverless function will experience a significant workload surge before the surge manifests, allowing the controller to accumulate confirmation signals or watch system stability proactively.
- **Architectural Boundary:** The ML model outputs a probability $\widehat{P}$ and categorical risk signal (`normal`, `elevated`, `critical`). It **never** triggers optimizations directly.

---

## 2. Canonical Training Sample Structure

Each sample corresponds to a single **(Function, Day, Minute)** prediction point:

$$\mathbf{x}_{(f, d, t)} = \left[ \text{day}, \text{minute}, \text{HashApp}, \text{HashFunction}, \text{Trigger}, \mathbf{\Phi}_{\text{historical}}(f, d, t) \right]$$

### Entity Granularity:
- **Unit of Observation:** Function $f$ in Application $a$ at relative trace minute $t \in [61 \dots 1435]$ of day $d \in [1 \dots 14]$.
- **Evaluation Bounds:** Exactly 1,375 prediction points per active function per day.
- **Independence:** Functions and applications maintain distinct time series; no cross-function pollution occurs.

---

## 3. Exact Target Formulation

Let $I_{(f, d, \tau)}$ denote the number of invocations of function $f$ on day $d$ at minute $\tau$.

### A. Future Workload Metric
$$\overline{\text{inv}}_{[t+1 \dots t+5]} = \frac{1}{5} \sum_{k=1}^{5} I_{(f, d, t+k)}$$

### B. Recent Baseline Metric
$$\overline{\text{inv}}_{[t-14 \dots t]} = \frac{1}{15} \sum_{k=0}^{14} I_{(f, d, t-k)}$$

### C. Recommended Target Definition (Option 1: Relative Surge with Noise Floor)
A positive class label ($Y=1$) represents a significant relative surge ($\ge 2.0\times$ recent 15-minute moving average) exceeding a minimum operational volume threshold:

$$Y_{(f, d, t)} = \mathbb{I}\left( \overline{\text{inv}}_{[t+1 \dots t+5]} \ge 2.0 \times \overline{\text{inv}}_{[t-14 \dots t]} \quad \text{AND} \quad \overline{\text{inv}}_{[t+1 \dots t+5]} \ge 5.0 \text{ req/min} \right)$$

### Rationale:
1. **Mathematical Grounding:** Requiring $K = 2.0\times$ detects genuine doubling of incoming load.
2. **Noise Floor Protection ($\ge 5\text{ req/min}$):** Prevents spurious positive labels when a function transitions from $0 \to 1$ or $1 \to 2$ calls/min, which is negligible for system capacity.
3. **Class Balance:** Produces an empirical positive class rate of **$0.61\%$** across all active functions (yielding $\approx 245,460$ positive surge events per day in the dataset), providing abundant signal for tree-based classifiers without extreme degeneracy.

---

## 4. Historical Feature Definitions (Calculated Strictly at $t' \le t$)

All features $\mathbf{\Phi}(f, d, t)$ are strictly backward-looking:

| Feature Name | Type | Mathematical Formula | Purpose |
|---|---|---|---|
| `invocations_t` | Numeric | $I_{(f, d, t)}$ | Current instantaneous invocation count |
| `rolling_mean_5m` | Numeric | $\frac{1}{5} \sum_{k=0}^{4} I_{(f, d, t-k)}$ | Immediate 5-minute traffic trend |
| `rolling_mean_15m` | Numeric | $\frac{1}{15} \sum_{k=0}^{14} I_{(f, d, t-k)}$ | 15-minute short-term moving baseline |
| `rolling_mean_60m` | Numeric | $\frac{1}{60} \sum_{k=0}^{59} I_{(f, d, t-k)}$ | 1-hour sustained workload baseline |
| `rolling_max_15m` | Numeric | $\max_{k \in [0, 14]} I_{(f, d, t-k)}$ | Recent peak burst in last 15 minutes |
| `rolling_max_60m` | Numeric | $\max_{k \in [0, 59]} I_{(f, d, t-k)}$ | Peak burst in last 1 hour |
| `rate_delta_1m` | Numeric | $I_{(f, d, t)} - I_{(f, d, t-1)}$ | 1-minute workload velocity / acceleration |
| `rate_delta_5m` | Numeric | $I_{(f, d, t)} - I_{(f, d, t-5)}$ | 5-minute workload momentum |
| `rolling_std_15m` | Numeric | $\sqrt{\frac{1}{15}\sum_{k=0}^{14} (I_{(f, d, t-k)} - \overline{\text{inv}}_{15m})^2}$ | Workload volatility / jitter |
| `minute_of_day` | Integer | $t \in [61 \dots 1435]$ | Linear time of day |
| `minute_sin` | Float | $\sin\left(\frac{2\pi \cdot t}{1440}\right)$ | Cyclical daily periodicity |
| `minute_cos` | Float | $\cos\left(\frac{2\pi \cdot t}{1440}\right)$ | Cyclical daily periodicity |
| `trigger_type` | Categorical | One-hot: `http`, `timer`, `queue`, `orchestration`, `event`, `storage`, `others` | Workload trigger mechanism |

---

## 5. Target Leakage Analysis & Safeguards

### Strict Horizon Isolation:
$$\underbrace{t-59, \dots, t-1, t}_{\text{Feature Window (Past \& Present)}} \quad \Bigg\vert \quad \underbrace{t+1, t+2, t+3, t+4, t+5}_{\text{Target Window (Strict Future)}}$$

- **Zero Overlap:** The current minute $t$ is included in the feature set; the target calculation strictly starts at $t+1$.
- **No Global Aggregations:** Normalization and percentiles are never computed using full-day or full-dataset future data.

---

## 6. Decision on Duration and Memory Datasets

### Finding:
The `function_durations_percentiles` and `app_memory_percentiles` files provide daily summary statistics aggregated over the **entire 24-hour period (minutes 1 to 1440)** of day $d$.

### Leakage Risk:
Attaching day $d$'s duration average or memory percentile to an observation at minute $t < 1440$ on day $d$ would incorporate execution events that have not yet occurred at minute $t$.

### Architectural Decision:
1. **Initial ML Model (Step 3):** Exclude intra-day duration and memory tables entirely. The model relies exclusively on leak-free minute invocation series and static trigger metadata.
2. **Future Enhancement:** If duration/memory are incorporated later, they will be strictly **lagged from day $d-1$** (prior day baseline), eliminating any contemporaneous leakage.

---

## 7. Cold-Start and Boundary Handling

### Intra-Day Rolling Window Boundaries:
- Calculating a 60-minute rolling window requires at least 60 minutes of prior history.
- Calculating a 5-minute forward target requires 5 minutes of future data.

### Standardized Evaluation Slice:
$$\text{Valid Prediction Points: } t \in [61, 1435]$$
- **Minutes $1 \dots 60$:** Dropped from training/inference as warm-up history for feature construction.
- **Minutes $1436 \dots 1440$:** Reserved strictly as the target lookahead window for $t=1435$.
- **Outcome:** Yields $1,435 - 60 = 1,375$ clean, leak-free, non-null evaluation points per function per day.

---

## 8. Empirical Class Balance Analysis

Based on vectorized evaluation across all prediction points on Day 01 ($t \in [61 \dots 1435]$):

| Cohort | Total Prediction Rows | Positive Surge Rows ($Y=1$) | Negative Rows ($Y=0$) | Positive % |
|---|---|---|---|---|
| **All Functions (Total Trace)** | **63,816,500** | **245,460** | **63,571,040** | **0.3846%** ($\approx 0.38\%$) |
| **Active Functions ($\ge 10\text{ calls/day}$)** | **40,440,125** | **245,460** | **40,194,665** | **0.6070%** ($\approx 0.61\%$) |
| **Inactive Functions ($< 10\text{ calls/day}$)** | **23,376,375** | **0** | **23,376,375** | **0.0000%** |

### Mathematical Reconcilement:
Because the target requires a future 5-minute average of $\ge 5.0\text{ req/min}$ ($25$ invocations in 5 min), functions with $< 10$ invocations in the entire 24-hour day produce exactly 0 positive surge points. Therefore, the absolute number of positive surge points is identically $245,460$ across both denominators:
$$\frac{245,460}{63,816,500} = \mathbf{0.3846\%} \quad \text{and} \quad \frac{245,460}{40,440,125} = \mathbf{0.6070\%}$$

---

## 9. Chronological Train / Validation / Test Partition

Because serverless workloads exhibit temporal auto-correlation and seasonal daily patterns, random cross-validation is strictly forbidden.

### Exact 14-Day Partition Breakdown (850,499,375 Total Rows):

```
+------------------------------------+-----------------------+-----------------------+
|          TRAIN SET (8 Days)        |   VAL SET (3 Days)    |   TEST SET (3 Days)   |
|            Days 01 to 08           |     Days 09 to 11     |     Days 12 to 14     |
|         57.23% (486.8M rows)       | 23.31% (198.2M rows)  | 19.46% (165.5M rows)  |
+------------------------------------+-----------------------+-----------------------+
Timeline --->  Day 01  --------------------->  Day 08 | Day 09 --------> Day 11 | Day 12 --------> Day 14
```

- **Training Split (Days 01–08, 8 days):** **486,783,000 rows** ($57.23\%$).
- **Validation Split (Days 09–11, 3 days):** **198,232,375 rows** ($23.31\%$).
- **Test Split (Days 12–14, 3 days):** **165,484,000 rows** ($19.46\%$).
- **Total Dataset Span (Days 01–14):** **850,499,375 prediction rows**.

---

## 10. Recommended Final Feature Set (14 Features)

```python
FINAL_FEATURE_COLUMNS = [
    # Instantaneous
    "invocations_t",
    # Rolling baselines
    "rolling_mean_5m",
    "rolling_mean_15m",
    "rolling_mean_60m",
    "rolling_max_15m",
    "rolling_max_60m",
    # Volatility and velocity
    "rate_delta_1m",
    "rate_delta_5m",
    "rolling_std_15m",
    # Temporal position
    "minute_of_day",
    "minute_sin",
    "minute_cos",
    # Trigger categories (encoded)
    "trigger_type",
]
```

---

## 11. Recommended Next Implementation Step

1. **Step 3 (Data Preprocessing Pipeline):** Build an efficient chunked extraction script to generate feature matrices $\mathbf{X}_{\text{train}}, \mathbf{X}_{\text{val}}, \mathbf{X}_{\text{test}}$ and target vectors $\mathbf{y}_{\text{train}}, \mathbf{y}_{\text{val}}, \mathbf{y}_{\text{test}}$ in lightweight Parquet format.
2. **Step 4 (Baseline & Classifier Training):** Train a lightweight, explainable tree-based classifier (e.g., Random Forest or Gradient Boosting) evaluating Precision-Recall AUC and latency.
3. **Step 5 (Inference Integration):** Connect model probability predictions to the Safety Controller risk signals (`normal`, `elevated`, `critical`).
