# Machine Learning Feature Engineering & Preprocessing Design

**Milestone 2 — Step 3: Feature Engineering & Preprocessing Design**  
**Project:** AutoScale IQ — ML-Powered Self-Optimizing Web Application  
**Objective:** Workload Surge Prediction on Azure Functions Trace 2019  
**Status:** Validated Design Specification (Pre-Implementation Phase)  

---

## 1. Approved Target Specification (Immutable Baseline)

For function $f$ on day $d$ evaluated at minute $t \in [61 \dots 1435]$:

$$Y_{(f, d, t)} = \mathbb{I}\left( \overline{\text{inv}}_{[t+1 \dots t+5]} \ge 2.0 \times \overline{\text{inv}}_{[t-14 \dots t]} \quad \text{AND} \quad \overline{\text{inv}}_{[t+1 \dots t+5]} \ge 5.0 \text{ req/min} \right)$$

where:
- **Future Lookahead Horizon:** $\overline{\text{inv}}_{[t+1 \dots t+5]} = \frac{1}{5} \sum_{k=1}^5 I_{(f, d, t+k)}$ (strictly minutes $t+1 \dots t+5$).
- **Moving Baseline:** $\overline{\text{inv}}_{[t-14 \dots t]} = \frac{1}{15} \sum_{k=0}^{14} I_{(f, d, t-k)}$ (strictly minutes $t-14 \dots t$).
- **Prediction Granularity:** Exactly $1,375$ valid prediction points per function per day ($t=61$ to $t=1435$).

---

## 2. Feature Set Design & Mathematical Formulations

Every feature $\phi \in \mathbf{\Phi}(f, d, t)$ is strictly backward-looking ($\tau \le t$):

```
+----------------------------------------------------------------------------------------+
|                                  TIMELINE AT MINUTE t                                  |
|                                                                                        |
|  ... [t-59] ............... [t-14] ....... [t-5] .... [t-1] [ t ] | [t+1] ... [t+5] .. |
|  |<-------- 60m Baseline -------->|                               | |<-- 5m Target -->| |
|                 |<--- 15m Baseline -->|                           |                     |
|                                 |<-- 5m Baseline -->|             |                     |
|                                                     |<- 1m Delta -|                     |
|                                                                                         |
|  [================ FEATURE INPUTS (PAST & PRESENT) ==============] | [== TARGET Y ==]  |
+----------------------------------------------------------------------------------------+
```

### A. Instantaneous Workload (1 Feature)
- **`invocations_t` ($I_t$):** Instantaneous invocation count at current minute $t$. Captures the current volume baseline.

### B. Multi-Scale Rolling Baselines (5 Features)
- **`rolling_mean_5m`:** Short-term moving average:
  $$\overline{\text{inv}}_{5m}(t) = \frac{1}{5} \sum_{i=0}^4 I_{t-i}$$
- **`rolling_mean_15m`:** Medium-term moving average (identical to target baseline denominator):
  $$\overline{\text{inv}}_{15m}(t) = \frac{1}{15} \sum_{i=0}^{14} I_{t-i}$$
- **`rolling_mean_60m`:** Long-term 1-hour sustained moving average:
  $$\overline{\text{inv}}_{60m}(t) = \frac{1}{60} \sum_{i=0}^{59} I_{t-i}$$
- **`rolling_max_15m`:** Peak instantaneous burst in previous 15 minutes:
  $$\max_{15m}(t) = \max_{i \in [0, 14]} I_{t-i}$$
- **`rolling_max_60m`:** Peak instantaneous burst in previous 1 hour:
  $$\max_{60m}(t) = \max_{i \in [0, 59]} I_{t-i}$$

### C. Trend, Velocity & Acceleration (2 Features)
- **`rate_delta_1m`:** 1-minute velocity (rate of change):
  $$\Delta_{1m}(t) = I_t - I_{t-1}$$
  *Positive values indicate surging traffic; negative values indicate receding traffic.*
- **`rate_delta_5m`:** 5-minute momentum (sustained acceleration):
  $$\Delta_{5m}(t) = I_t - I_{t-5}$$
  *Measures multi-minute acceleration into the prediction point.*

### D. Workload Volatility & Jitter (1 Feature)
- **`rolling_std_15m`:** Standard deviation over previous 15 minutes:
  $$\sigma_{15m}(t) = \sqrt{\frac{1}{15} \sum_{i=0}^{14} \left(I_{t-i} - \overline{\text{inv}}_{15m}(t)\right)^2}$$
  *Distinguishes steady predictable streams from volatile, bursty workloads.*

### E. Temporal Diurnal Signals (3 Features)
- **`minute_of_day`:** Linear minute index $t \in [61 \dots 1435]$.
- **`minute_sin` & `minute_cos`:** Continuous 24-hour cyclical harmonic encodings:
  $$\text{minute\_sin}(t) = \sin\left(\frac{2\pi \cdot t}{1440}\right), \quad \text{minute\_cos}(t) = \cos\left(\frac{2\pi \cdot t}{1440}\right)$$
  *Ensures smooth continuity across diurnal cycles.*

### F. Trigger Mechanism Metadata (7 Features)
- **`Trigger`:** Categorical trigger type (`timer`, `http`, `queue`, `orchestration`, `event`, `storage`, `others`).
  - One-hot encoded into 7 binary flags: `trigger_timer`, `trigger_http`, `trigger_queue`, `trigger_orchestration`, `trigger_event`, `trigger_storage`, `trigger_others`.

---

## 3. High-Cardinality Identifier Decision (`HashApp` & `HashFunction`)

### Decision: Exclude `HashApp` and `HashFunction` from Model Inputs $\mathbf{X}$

### Rationale:
1. **Extreme Cardinality & Sparsity:** Day 01 contains 46,412 unique functions and 17,577 apps. One-hot encoding would inject $>60,000$ sparse columns, rendering training computationally infeasible and prone to severe overfitting.
2. **Leakage & Generalization Failure:** Target encoding (e.g. mean historical surge rate per function) risks lookahead leakage and fails when new, unseen functions appear in validation/test days (or production).
3. **Domain Invariance:** The core physics of a workload surge (traffic acceleration, volatility, baseline exceedance) are function-agnostic. A tree model trained on dynamic workload features generalizes across any function regardless of its anonymized hash.
4. **Metadata Preservation:** `HashOwner`, `HashApp`, `HashFunction`, `day`, and `minute` are retained as metadata columns in output tables for tracing, filtering, and evaluation, but are omitted from the numerical feature matrix $\mathbf{X}$.

---

## 4. Critical Leakage Audit Matrix

| Feature | Source Data | History Window | Future Info? | Day Boundary Cross? | Safe for Production Inference? | Keep? |
|---|---|---|---|---|---|---|
| `invocations_t` | Invocations CSV | $[t]$ (current) | **No** | No | Yes (live telemetry) | **YES** |
| `rolling_mean_5m` | Invocations CSV | $[t-4 \dots t]$ | **No** | No | Yes (rolling buffer) | **YES** |
| `rolling_mean_15m` | Invocations CSV | $[t-14 \dots t]$ | **No** | No | Yes (rolling buffer) | **YES** |
| `rolling_mean_60m` | Invocations CSV | $[t-59 \dots t]$ | **No** | No | Yes (rolling buffer) | **YES** |
| `rolling_max_15m` | Invocations CSV | $[t-14 \dots t]$ | **No** | No | Yes (rolling buffer) | **YES** |
| `rolling_max_60m` | Invocations CSV | $[t-59 \dots t]$ | **No** | No | Yes (rolling buffer) | **YES** |
| `rate_delta_1m` | Invocations CSV | $[t-1, t]$ | **No** | No | Yes (1m difference) | **YES** |
| `rate_delta_5m` | Invocations CSV | $[t-5, t]$ | **No** | No | Yes (5m difference) | **YES** |
| `rolling_std_15m` | Invocations CSV | $[t-14 \dots t]$ | **No** | No | Yes (rolling variance) | **YES** |
| `minute_of_day` | Trace Timestamp | $[t]$ (current) | **No** | No | Yes (current clock) | **YES** |
| `minute_sin` | Trace Timestamp | $[t]$ (current) | **No** | No | Yes (cyclical math) | **YES** |
| `minute_cos` | Trace Timestamp | $[t]$ (current) | **No** | No | Yes (cyclical math) | **YES** |
| `trigger_timer` | Metadata Column | Static | **No** | No | Yes (function config) | **YES** |
| `trigger_http` | Metadata Column | Static | **No** | No | Yes (function config) | **YES** |
| `trigger_queue` | Metadata Column | Static | **No** | No | Yes (function config) | **YES** |
| `trigger_orchestration` | Metadata Column | Static | **No** | No | Yes (function config) | **YES** |
| `trigger_event` | Metadata Column | Static | **No** | No | Yes (function config) | **YES** |
| `trigger_storage` | Metadata Column | Static | **No** | No | Yes (function config) | **YES** |
| `trigger_others` | Metadata Column | Static | **No** | No | Yes (function config) | **YES** |
| *Daily Duration Table* | Duration CSV | Full Day (1–1440) | **YES (Leaked)** | Crosses day | **NO** (Aggregated over future minutes) | **REJECTED** |
| *Daily Memory Table* | Memory CSV | Full Day (1–1440) | **YES (Leaked)** | Crosses day | **NO** (Aggregated over future minutes) | **REJECTED** |
| *Global Static P95* | Invocations CSV | Multi-day aggregate | **YES (Leaked)** | Crosses day | **NO** (Contains future days) | **REJECTED** |
| *HashFunction ID* | Metadata Column | Static | High Cardinality | N/A | Overfits on unseen IDs | **REJECTED** |

---

## 5. Day-Boundary & Cold-Start Strategy

### Strategy: Option A (Intra-Day Warm-up, $t \in [61 \dots 1435]$)
- **Feature Warm-Up:** Minutes $1 \dots 60$ provide historical warm-up for 60-minute rolling windows.
- **Lookahead Reservation:** Minutes $1436 \dots 1440$ provide the 5-minute future target lookahead window for minute $t=1435$.
- **Benefits:** Eliminates multi-day cross-file stitching discontinuities and missing-file dependencies.
- **Exact Row Count:** Yields exactly $1,375$ clean rows per function per day ($63,816,500$ rows on Day 01).

---

## 6. Missing & Zero Workload Handling

1. **Zero Values:** Invocations of `0` are valid numerical observations representing idle serverless instances ($0.0$, not NaN).
2. **Missing Data in Trace:** The raw Azure Functions 2019 trace has **zero missing values** across all 1,440 minute columns.
3. **Rolling Metrics over Zeroes:** If a function is idle for 60 minutes, rolling mean, max, delta, and std dev are all naturally `0.0` (mathematically well-defined and finite).
4. **No Imputation Required:** History is guaranteed for $t \ge 61$; no synthetic imputation is needed.

---

## 7. Categorical Preprocessing

- **Feature:** `Trigger` (7 distinct categories).
- **Method:** **One-Hot Encoding** fitted strictly on the **Training Set (Days 01–08)** with `handle_unknown='ignore'`.
- **Output:** 7 binary `uint8` columns.

---

## 8. Exact Feature Matrix & Memory Layout Specification

### Feature Matrix $\mathbf{X}$ (Exactly 19 Features):

| # | Column Name | Category | Dtype | Range | Storage Size |
|---|---|---|---|---|---|
| 1 | `invocations_t` | Workload | `float32` | $[0.0, \infty)$ | 4 bytes |
| 2 | `rolling_mean_5m` | Workload | `float32` | $[0.0, \infty)$ | 4 bytes |
| 3 | `rolling_mean_15m` | Workload | `float32` | $[0.0, \infty)$ | 4 bytes |
| 4 | `rolling_mean_60m` | Workload | `float32` | $[0.0, \infty)$ | 4 bytes |
| 5 | `rolling_max_15m` | Workload | `float32` | $[0.0, \infty)$ | 4 bytes |
| 6 | `rolling_max_60m` | Workload | `float32` | $[0.0, \infty)$ | 4 bytes |
| 7 | `rate_delta_1m` | Trend | `float32` | $(-\infty, \infty)$ | 4 bytes |
| 8 | `rate_delta_5m` | Trend | `float32` | $(-\infty, \infty)$ | 4 bytes |
| 9 | `rolling_std_15m` | Volatility | `float32` | $[0.0, \infty)$ | 4 bytes |
| 10 | `minute_of_day` | Time | `int16` | $[61, 1435]$ | 2 bytes |
| 11 | `minute_sin` | Time | `float32` | $[-1.0, 1.0]$ | 4 bytes |
| 12 | `minute_cos` | Time | `float32` | $[-1.0, 1.0]$ | 4 bytes |
| 13 | `trigger_timer` | Trigger | `uint8` | $\{0, 1\}$ | 1 byte |
| 14 | `trigger_http` | Trigger | `uint8` | $\{0, 1\}$ | 1 byte |
| 15 | `trigger_queue` | Trigger | `uint8` | $\{0, 1\}$ | 1 byte |
| 16 | `trigger_orchestration` | Trigger | `uint8` | $\{0, 1\}$ | 1 byte |
| 17 | `trigger_event` | Trigger | `uint8` | $\{0, 1\}$ | 1 byte |
| 18 | `trigger_storage` | Trigger | `uint8` | $\{0, 1\}$ | 1 byte |
| 19 | `trigger_others` | Trigger | `uint8` | $\{0, 1\}$ | 1 byte |

### Target Column $y$:
- **`target_surge_5m`:** `uint8` ($\{0, 1\}$) — 1 byte.

### Exact Bytes Per Row:
- **Feature-Only Bytes:** $11 \times 4\text{ (float32)} + 1 \times 2\text{ (int16)} + 7 \times 1\text{ (uint8)} = 44 + 2 + 7 = \mathbf{53\text{ bytes per row}}$.
- **Features + Target Bytes:** $53 + 1 = \mathbf{54\text{ bytes per row}}$.

---

## 9. Feature Scaling Strategy

- **Decision:** **No Feature Scaling (Raw Natural Scale)**
- **Reasoning:** Tree-based models make axis-aligned split decisions ($\text{feature}_j \le \theta$) and are invariant to monotonic scaling. Keeping features in natural units (`requests/minute`, $\Delta\text{ req}$) enables direct interpretability without `StandardScaler` state serialization overhead.

---

## 10. Massive-Dataset Management & Memory Footprint

### Memory Footprint Breakdown:

| Dataset Scope | Total Rows | Feature-Only RAM (53 bytes/row) | Features + Target RAM (54 bytes/row) | Disk Storage (Compressed Parquet) |
|---|---|---|---|---|
| **Per 1,000,000 Rows** | 1.0M rows | $53.00\text{ MB}$ ($50.54\text{ MiB}$) | $54.00\text{ MB}$ ($51.50\text{ MiB}$) | $\approx 6\text{ MB}$ |
| **Full Day 01** | 63,816,500 rows | $3.38\text{ GB}$ ($3.15\text{ GiB}$) | $3.45\text{ GB}$ ($3.21\text{ GiB}$) | $\approx 350\text{ MB}$ |
| **Training Sample (Days 01–08)** | $\approx 21.5\text{M}$ rows | $1.14\text{ GB}$ ($1.06\text{ GiB}$) | $1.16\text{ GB}$ ($1.08\text{ GiB}$) | $\approx 180\text{ MB}$ |
| **Validation Set (Days 09–11)** | 198,232,375 rows | $10.51\text{ GB}$ ($9.79\text{ GiB}$) | $10.70\text{ GB}$ ($9.97\text{ GiB}$) | $\approx 1.1\text{ GB}$ |
| **Test Set (Days 12–14)** | 165,484,000 rows | $8.77\text{ GB}$ ($8.17\text{ GiB}$) | $8.94\text{ GB}$ ($8.32\text{ GiB}$) | $\approx 0.9\text{ GB}$ |

### Preprocessing Throughput Caveat:
*Note on Extraction Performance:* While vectorized NumPy array operations execute rapidly in RAM, end-to-end extraction involves CSV parsing, rolling window convolutions, categorical encoding, and Parquet disk serialization. The processing throughput must be empirically benchmarked on a representative subset before full batch execution.

---

## 11. Model Training Scale & Subsampling Strategy

### Scale & Architecture Considerations:
- A full 486.8M-row training set contains $>99.6\%$ idle zero observations and would make standard unconstrained Random Forest or GBDT training computationally prohibitive.
- In Step 4, we will evaluate computationally practical tree architectures (e.g. Histogram-based Gradient Boosting / LightGBM, depth-constrained ensembles, or tuned subsample budgets). The feature pipeline remains model-agnostic.

### Training Subsampling Strategy (Days 01–08):
1. **100% Positive Surges:** Retain all rare surge events ($Y=1$, $\approx 1.96\text{M}$ rows across 8 days).
2. **10:1 Negative-to-Positive Ratio:** Sample non-surge observations ($Y=0$, $\approx 19.6\text{M}$ rows).
3. **Total Training Sample Size:** $\approx 21.5\text{M}$ rows ($\approx 1.16\text{ GB}$ in RAM), fitting comfortably in memory.

### Critical Calibration & Probability Threshold Caveat:
- Subsampling at 10:1 artificially alters the training class prevalence to $\approx 9.09\%$ ($1 \text{ in } 11$), compared to the natural trace prevalence of $\approx 0.38\%$ ($0.61\%$ on active functions).
- **Action for Step 4:** Raw model probability outputs will be biased upward. Decision thresholds and probability calibration (e.g., isotonic calibration, Platt scaling, or log-odds prior correction) must be fitted strictly on the **unaltered, natural chronological Validation Set (Days 09–11)** to produce realistic probabilities for the safety controller.

---

## 12. Preprocessing Quality Gate Checklist

Before passing data into model training, the pipeline must enforce:
- [ ] **Exact 19 Features:** 11 float32, 1 int16, 7 uint8.
- [ ] **Exact Dtypes & Storage:** 53 bytes feature-only, 54 bytes with target.
- [ ] **Zero NaNs or Inf:** Strictly finite numerical values across all columns.
- [ ] **Temporal Isolation:** Feature timestamps $\tau \le t$; target timestamps $\tau \in [t+1 \dots t+5]$.
- [ ] **Split Purity:** Days 01–08 (Train), Days 09–11 (Val), Days 12–14 (Test).
- [ ] **One-Hot Consistency:** All 7 trigger columns present across all splits.
