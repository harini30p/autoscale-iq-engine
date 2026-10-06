# Machine Learning Model Selection & Training Design

**Milestone 2 — Step 4: Model Selection & Training Design**  
**Project:** AutoScale IQ — ML-Powered Self-Optimizing Web Application  
**Objective:** Workload Surge Prediction on Azure Functions Trace 2019  
**Status:** Validated Design Specification (Pre-Implementation Phase)  

---

## 1. Locked Pipeline Specifications (From Steps 1–3)

| Component | Locked Specification |
|---|---|
| **Prediction Target** | Binary Workload Surge $Y_{(f, d, t)} \in \{0, 1\}$ over window $[t+1 \dots t+5]$:<br>$\bullet \quad \overline{\text{inv}}_{[t+1 \dots t+5]} \ge 2.0 \times \overline{\text{inv}}_{[t-14 \dots t]}$<br>$\bullet \quad \overline{\text{inv}}_{[t+1 \dots t+5]} \ge 5.0\text{ req/min}$ |
| **Feature Inputs** | Exactly **19 features** calculated strictly at $\tau \le t$:<br>$\bullet$ 6 Workload: `invocations_t`, `rolling_mean_5m`, `rolling_mean_15m`, `rolling_mean_60m`, `rolling_max_15m`, `rolling_max_60m`<br>$\bullet$ 2 Trend: `rate_delta_1m`, `rate_delta_5m`<br>$\bullet$ 1 Volatility: `rolling_std_15m`<br>$\bullet$ 3 Temporal: `minute_of_day`, `minute_sin`, `minute_cos`<br>$\bullet$ 7 Triggers (One-Hot): `trigger_timer`, `http`, `queue`, `orchestration`, `event`, `storage`, `others` |
| **Chronological Split** | $\bullet$ **Train:** Days 01–08 ($486.8\text{M}$ raw rows, $57.23\%$)<br>$\bullet$ **Validation:** Days 09–11 ($198.2\text{M}$ raw rows, $23.31\%$)<br>$\bullet$ **Test:** Days 12–14 ($165.5\text{M}$ raw rows, $19.46\%$) |
| **Training Sampling** | $\bullet$ **100% Positives** retained ($Y=1$, $\approx 1.96\text{M}$ rows across Days 01–08)<br>$\bullet$ **~10:1 Negative Subsample** ($Y=0$, $\approx 19.6\text{M}$ rows)<br>$\bullet$ **Total Training Pool:** $\approx 21.5\text{M}$ rows |
| **Validation / Test** | Evaluated at **100% natural chronological distribution** ($\approx 0.3846\%$ positive rate on full trace). |
| **Exclusions (Locked)** | $\bullet$ Exclude `HashApp` and `HashFunction` from feature matrix $\mathbf{X}$<br>$\bullet$ Exclude daily duration and memory aggregate tables |

---

## 2. Candidate Model Evaluation & Architecture Selection

We evaluate four candidate model architectures on ten operational criteria:

| Evaluation Criterion | Candidate A: HistGradientBoosting (`sklearn`) | Candidate B: LightGBM | Candidate C: Random Forest (`sklearn`) | Candidate D: Logistic / Rule Baseline |
|---|---|---|---|---|
| **1. Predictive Power on Tabular Time Series** | **High** (Boosted non-linear trees with interaction capture) | **High** (Fast leaf-wise GBDT) | **Moderate-High** (Bagged trees) | **Low-Moderate** (Linear / heuristic) |
| **2. Training Speed on Millions of Rows** | **Fast** (256-bin integer histogram binning) | **Fast** (Histogram binning + GOSS) | **Slow** (Continuous float sorting per tree: $O(M \cdot N \log N)$) | **Instantaneous** ($O(N)$) |
| **3. Memory Efficiency** | **High** (Compact uint8 bin representation) | **High** (Native histogram memory layout) | **Poor** (Stores full float arrays across all trees; heavy RAM) | **Minimal** |
| **4. Feature Suitability (19 Continuous & Binary)** | **Optimal** (Handles mixed continuous and one-hot features) | **Optimal** | **Good** | **Moderate** (Requires scaling) |
| **5. Class Imbalance Adaptability** | **Native** (Class weights + sample weights + early stopping) | **Native** (`is_unbalance`, `scale_pos_weight`) | **Moderate** (`class_weight='balanced'`) | **Poor** |
| **6. Probability Output Quality** | **Log-Loss Output** (Requires calibration from sampled training) | **Logit Output** (Requires calibration) | **Step-Wise Voting** (Coarse probabilities) | **Sigmoid Output** |
| **7. Model Interpretability** | **High** (Permutation importance + TreeSHAP) | **High** (SHAP values) | **Moderate** (Gini impurity importance) | **Very High** (Weights) |
| **8. Dependency & Build Complexity** | **Zero External Dependencies** (Built into standard `scikit-learn`) | **Moderate** (External C++ wheels on Windows) | **Zero External Dependencies** | **Zero** |
| **9. Local Windows Environment Reliability** | **100% Seamless** (No compiler or OpenMP conflicts) | **Good** (Occasional wheel mismatch) | **100% Seamless** | **100% Seamless** |
| **10. Demo & Technical Interview Defensibility** | **Gold Standard** for production tabular classification | **Strong** | **Acceptable** (but hard to justify $10\times$ slower runtime) | **Essential** baseline |

### Decision: Primary Model = Candidate A (`HistGradientBoostingClassifier`)
- **Justification:** Built directly into standard `scikit-learn`, zero compilation friction on Windows, native 256-bin integer histogram architecture capable of scaling to large tabular datasets, native monotonic constraints, and seamless integration with probability calibration tools.

---

## 3. Non-ML Reference Baselines

To prove that the ML model learns genuine predictive patterns rather than merely lagging behind current traffic, we define two deterministic reference baselines:

### Baseline 1: Moving-Average Ratio Rule (Heuristic Threshold)
$$\widehat{Y}_{\text{Rule1}}(t) = \mathbb{I}\left( I_t \ge 2.0 \times \overline{\text{inv}}_{15m}(t) \quad \text{AND} \quad I_t \ge 5.0\text{ req/min} \right)$$
- **Logic:** Predicts a future surge if the *current* minute has already doubled the 15-minute moving average.
- **Weakness:** Strictly reactive; it triggers only *after* traffic has already spiked, providing zero early-warning lead time.

### Baseline 2: Workload Momentum & Velocity Rule
$$\widehat{Y}_{\text{Rule2}}(t) = \mathbb{I}\left( (I_t - I_{t-1}) > 0 \quad \text{AND} \quad (I_t - I_{t-5}) \ge 1.5 \times \overline{\text{inv}}_{15m}(t) \right)$$
- **Logic:** Predicts a surge if traffic shows positive 1-minute velocity and 5-minute acceleration.

---

## 4. Class Imbalance Strategy

```
[ Natural Population Prevalence: ~0.3846% (1 in 260) ]
                      │
                      ▼ (Subsampling ~10:1 Negative:Positive on Train Split)
[ Sampled Training Distribution: ~9.09% (1 in 11) ]
                      │
                      ▼ (HistGradientBoosting with Log-Loss on Training Sample)
[ Raw Model Output: P_sampled (Reflects Sampled Training Distribution) ]
                      │
                      ▼ (Validation Calibration on Natural Days 09-11)
[ Calibrated Real-World Probability: P_cal (Aligned with Natural Distribution) ]
```

### Key Principles:
1. **Natural vs. Sampled Prevalence:** In the natural trace, positive surges represent $\approx 0.3846\%$ ($0.6070\%$ on active functions). The 10:1 negative subsampling on training data increases the training prevalence to $\frac{1}{1 + 10} = \mathbf{9.0909\%}$.
2. **Rejection of Synthetic Oversampling (SMOTE):** SMOTE synthesizes artificial points by interpolating between nearest neighbors in feature space. In high-dimensional discrete time-series features (e.g. rolling max and deltas), linear interpolation produces physically impossible workload trajectories and inflates false alarms.
3. **No Redundant Class Weights in Training:** A 10:1 ratio provides ample gradient signal for tree-based loss functions without needing extreme synthetic weighting.
4. **Accuracy Rejection:** A trivial model predicting constant $0$ achieves $99.62\%$ accuracy while missing $100\%$ of surges. Model selection is driven by Precision-Recall AUC (PR-AUC) and operational metrics.

---

## 5. Evaluation Metrics & Operational Multi-Scale Reporting

### Primary Evaluation Metric: Precision-Recall AUC (PR-AUC / Average Precision)
$$\text{PR-AUC} = \sum_{n} (R_n - R_{n-1}) P_n$$
- **Why PR-AUC:** Unlike ROC-AUC (which evaluates false positive rate against millions of true negatives and easily exceeds $0.99$ misleadingly), PR-AUC focuses strictly on the positive surge class ($Y=1$) and heavily penalizes false alarms.
- **Context:** PR-AUC must be interpreted relative to the natural positive prevalence baseline ($\pi \approx 0.0038$).

### Secondary Diagnostic Metrics:
- **Precision ($P$):** $\frac{\text{True Surges Detected}}{\text{Total Surge Alarms Raised}}$ (Minimizes false alarms).
- **Recall ($R$):** $\frac{\text{True Surges Detected}}{\text{Total Actual Surges}}$ (Minimizes missed surges).
- **$F_2$-Score:** $\frac{5 \cdot P \cdot R}{4P + R}$ (Weights recall $2\times$ higher than precision to prioritize system safety).
- **Brier Score:** $\frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$ (Measures probability calibration accuracy).
- **ROC-AUC:** Reported as a secondary diagnostic.

### Multi-Scale Operational Metrics for AutoScale IQ:

| Operational Metric | Interpretation | Importance in Technical Review |
|---|---|---|
| **Surge Capture Rate (Recall)** | $\%$ of genuine future workload surges caught | Proves system proactively protects application performance |
| **Precision at Critical Threshold** | $\%$ of high-risk alerts that represent true surges | Proves system does not trigger unnecessary optimization overhead |
| **False Alarms per Function-Day** | Mean false positive alerts per function per 24h | Operational stability metric at individual function level |
| **False Alarms per 1,000 Function-Days** | Aggregate false alarms scaled to a multi-tenant fleet | Normalizes fleet-wide false alarm noise for enterprise evaluation |
| **Predicted Alert Volume** | Total number of positive alarms generated per day | Validates operational load on the safety controller |
| **Early Warning Lead Time** | Minutes between first high-risk alert and surge peak | Proves controller has time to accumulate 3x confirmations before peak |
| **Unnecessary Optimization Activations** | Optimizations triggered where no surge occurred | Measures true end-to-end impact on application stability |

---

## 6. Probability Calibration Strategy

Because training uses ~10:1 subsampling ($\pi_{\text{train}} \approx 0.0909$), raw model predictions $P_{\text{sampled}}$ reflect the training distribution rather than the natural trace ($\pi_{\text{natural}} \approx 0.003846$).

### Candidate Calibration Approaches:

1. **Prior-Shift / Log-Odds Adjustment (Theoretical Transformation):**
   $$\text{logit}(P_{\text{cal}}) = \text{logit}(P_{\text{sampled}}) - \text{logit}(\pi_{\text{train}}) + \text{logit}(\pi_{\text{natural}})$$
   where:
   - $\text{logit}(\pi_{\text{train}}) = \ln\left(\frac{0.090909}{0.909091}\right) = -\ln(10) \approx -2.3026$
   - $\text{logit}(\pi_{\text{natural}}) = \ln\left(\frac{0.0038464}{0.9961536}\right) \approx -5.5569$
   - Resulting correction shift: $\Delta_{\text{logit}} = -5.5569 - (-2.3026) \approx \mathbf{-3.2543\text{ log-odds}}$.
   - *Note:* Prior-shift assumes conditionally independent features given the class label. While mathematically clean, it is an approximate transformation and must be empirically verified.

2. **Platt / Sigmoid Calibration on Validation Split:**
   - Fits a logistic mapping on the natural chronological Validation Set (Days 09–11).

3. **Isotonic Regression on Validation Split:**
   - Fits a non-parametric piecewise-constant isotonic mapping on Days 09–11.

### Calibration Protocol:
- **Fitting:** Calibration models (Platt or Isotonic) and prior-shift baselines are fitted **strictly using the natural chronological Validation Set (Days 09–11)**.
- **Untouched Test Set:** The test set (Days 12–14) is never used during calibration fitting.
- **Comparison:** In Step 5, we will compare uncalibrated, prior-shifted, and validation-fitted calibration curves (reliability diagrams and Brier scores) to select the best calibrated risk output.

---

## 7. Decision Threshold Selection Strategy

The ML model outputs a continuous calibrated surge risk probability $P_{\text{cal}} \in [0.0, 1.0]$. The default threshold of $0.5$ is completely arbitrary for highly imbalanced data.

### Validation-Based Threshold Selection:
Thresholds are determined empirically by sweeping operating points on the **natural Validation Set (Days 09–11)** to balance the operational trade-off:
- **Lower Threshold:** Higher recall (catches more surges) but higher false alarm rate.
- **Higher Threshold:** Lower false alarm rate but lower recall (misses subtle surges).

```
Calibrated Probability P_cal:
0.0 -------------- [ tau_watch ] -------------- [ tau_crit ] -------------- 1.0
        │                               │                              │
        ▼                               ▼                              ▼
 Risk: NORMAL                    Risk: ELEVATED                 Risk: CRITICAL
(Controller: NORMAL)          (Controller: WATCHING)        (Controller: Increments 3x Count)
```

### Three-Tier Risk Mapping to Safety Controller:
1. **$P_{\text{cal}} < \tau_{\text{watch}}$ $\longrightarrow$ `normal`:** System in standard steady-state.
2. **$\tau_{\text{watch}} \le P_{\text{cal}} < \tau_{\text{crit}}$ $\longrightarrow$ `elevated`:** Transitions controller to `WATCHING` state; observes system closely without modifying configuration.
3. **$P_{\text{cal}} \ge \tau_{\text{crit}}$ $\longrightarrow$ `critical`:** Increments the 3-consecutive high-risk confirmation counter.

### Architectural Invariant:
**ML prediction $\neq$ optimization action.** The ML model predicts *risk*. The Milestone 1 Safety Controller governs all actual optimization actions through:
- 3 consecutive high-risk confirmations
- Hard safety thresholds (CPU $\ge 90\%$, Memory $\ge 90\%$, Latency $\ge 1000\text{ ms}$)
- Post-optimization cooldown (30s)
- Recovery hysteresis (3 consecutive normal readings)
- Manual override switch
- Optimization failure handling

---

## 8. Training-Scale Strategy & Progressive Scaling Experiment

The full 8-day training pool contains $\approx 21.5\text{M}$ sampled rows ($1.96\text{M}$ positives, $19.6\text{M}$ negatives).

### Computational Reality:
Training unconstrained tree models on 21.5M rows can become computationally demanding on standard hardware. Therefore, the training size will be established through a **progressive scaling benchmark**:

```
Budget 1: 1M Sample  (100k pos / 900k neg)  ──> Benchmark Runtime & Memory ──> Fast Baseline
Budget 2: 5M Sample  (500k pos / 4.5M neg)  ──> Benchmark Runtime & Memory ──> Target Experiment
Budget 3: Full 21.5M Pool (1.96M pos / 19.6M neg) ──> Benchmark Runtime & Memory ──> Asymptotic Check
```

### Scaling Protocol:
1. Extract samples preserving the ~10:1 negative-to-positive ratio and maintaining chronological order across Days 01–08.
2. Empirically measure training runtime, peak memory usage, and PR-AUC on the Validation Set (Days 09–11).
3. If a 5M sample achieves asymptotic validation performance comparable to larger budgets, 5M rows is adopted as the practical training budget for hyperparameter search.
4. *Runtime Note:* Training times are not assumed in advance and must be measured experimentally on the host environment.

---

## 9. Hyperparameter Search Strategy

To avoid combinatorial explosion (e.g. 243-configuration brute-force grids), we define a **targeted search of 8 to 15 key configurations** evaluated on the natural **Validation Set (Days 09–11)**:

| Configuration Set | Focus / Objective | Parameters Evaluated |
|---|---|---|
| **Set 1: Baseline Architecture** | Standard GBDT baseline | `max_iter=100`, `learning_rate=0.10`, `max_leaf_nodes=31`, `min_samples_leaf=100`, `l2=1.0` |
| **Set 2: Deeper Interaction Search** | Capture multi-scale feature interactions | `max_leaf_nodes=[63, 127]`, `min_samples_leaf=[50, 100]`, `learning_rate=0.05` |
| **Set 3: Conservative Regularization** | Prevent overfitting on noisy bursts | `learning_rate=[0.03, 0.05]`, `max_iter=[150, 200]`, `l2_regularization=[5.0, 10.0]` |
| **Set 4: Early Stopping Tuning** | Convergence optimization | `n_iter_no_change=[10, 15]`, `tol=1e-4` |

### Rules:
- Model selection is based strictly on **Validation Set PR-AUC and Brier score**.
- Random cross-validation is strictly prohibited to preserve chronological integrity.
- Test Set (Days 12–14) is never touched during hyperparameter tuning.

---

## 10. Model Explainability Architecture (For NEXUS Technical Demo)

To ensure full transparency during technical evaluations:

```
Observation at minute t:
Features: rate_delta_1m=+45.0, rolling_mean_5m=38.2, rolling_std_15m=12.4
       │
       ▼ (Permutation Importance & TreeSHAP Attribution)
Top Predictive Drivers:
  1. [+] rate_delta_1m (+45.0 req/min) ──> Immediate acceleration
  2. [+] rolling_mean_5m (38.2 req/min) ──> Short-term sustained volume
  3. [+] rolling_std_15m (12.4 req/min) ──> Recent traffic volatility
       │
       ▼
NEXUS Demo Display:
"Surge Risk: ELEVATED (82%). Primary Driver: Rapid 1-minute acceleration (+45 req/min)."
```

### Components:
1. **Global Feature Importance:** Permutation importance on the Validation Set to identify global drivers.
2. **Local Prediction Explanation:** Fast TreeSHAP attribution to explain individual high-risk predictions in real time.

---

## 11. Acceptance & Success Criteria (Step 4 & Step 5)

A trained model is accepted for integration into AutoScale IQ when it demonstrates:

1. **Meaningful Baseline Beat:** The model's PR-AUC meaningfully outperforms the deterministic momentum baseline on the natural validation and test distributions.
2. **Useful Operational Recall & Precision:** Demonstrates a practical trade-off on the natural validation set (with $\ge 75\%$ recall and $\ge 50\%$ precision as aspirational demo targets, rather than arbitrary hard pass/fail gates).
3. **Controlled False Alarm Rate:** Fleet-wide false alarm rate remains operationally manageable (reported per function-day and per 1,000 function-days).
4. **Reliable Probability Calibration:** Calibrated probabilities reflect natural deployment prevalence with low Brier score.
5. **Actionable Early-Warning Lead Time:** Provides 1 to 5 minutes of predictive lead time before surge peaks.
6. **Low Inference Latency:** Feature extraction and model scoring execute in $< 1\text{ ms}$ per function observation.
7. **Computational Feasibility:** Preprocessing and training run reliably within host memory and CPU budgets.
8. **Out-of-Time Generalization:** Validated on the untouched chronological Test Set (Days 12–14).

---

## 12. End-to-End System Integration Architecture

```
[ Raw Azure Invocation Stream ]
               │
               ▼ (Feature Extraction: 19 Features strictly at tau <= t)
[ Feature Vector x_t (53 bytes) ]
               │
               ▼ (HistGradientBoostingClassifier Inference)
[ Raw Prediction P_sampled ]
               │
               ▼ (Validation-Fitted Calibration & Log-Odds Shift)
[ Calibrated Risk Probability P_cal ]
               │
               ▼ (Validation-Selected Thresholds: tau_watch, tau_crit)
[ Categorical Risk Signal: normal | elevated | critical ]
               │
               ▼
+-----------------------------------------------------------------------+
|                 Milestone 1 Safety Controller                         |
|   • 3 Consecutive High-Risk Confirmations                             |
|   • Hard Safety Thresholds (CPU >= 90%, Mem >= 90%, Latency >= 1000ms)|
|   • Post-Optimization Cooldown (30s)                                  |
|   • Recovery Hysteresis (3 Consecutive Normal Confirmations)          |
|   • Manual Override Switch                                            |
|   • Optimization Failure Handling                                     |
+-----------------------------------+-----------------------------------+
                                    │
                                    ▼
+-----------------------------------------------------------------------+
|                 Application Optimizer Levers                          |
|   • Caching: disabled <-> enabled                                     |
|   • Pagination: 50 <-> 20 items/page                                  |
|   • Heavy Components: enabled <-> disabled                            |
+-----------------------------------------------------------------------+
```

---

## 13. Summary of Computational Footprint

- **Feature Matrix Dtypes:** 11 float32 (44B) + 1 int16 (2B) + 7 uint8 (7B) = **53 bytes/row** (54 bytes including target $y$).
- **Per 1M Rows:** $53.00\text{ MB}$ ($50.54\text{ MiB}$) feature RAM.
- **Full Day 01 ($63.8\text{M}$ rows):** $3.38\text{ GB}$ ($3.15\text{ GiB}$) feature RAM.
- **Proposed Training Pool ($21.5\text{M}$ rows):** $1.14\text{ GB}$ ($1.06\text{ GiB}$) feature RAM.
- *Benchmark Principle:* Full training pipeline runtime and memory will be empirically benchmarked starting from small subsets ($1\text{M} \to 5\text{M} \to 21.5\text{M}$) rather than assuming static execution times.
