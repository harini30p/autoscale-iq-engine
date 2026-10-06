# AutoScale IQ — 5M Config 6 Benchmark & Validation Report

**Experiment:** Milestone 2 — Final Candidate Model Training (Config 6 on 5M Dataset)  
**Model Architecture:** `HistGradientBoostingClassifier(max_leaf_nodes=63, min_samples_leaf=50, l2=3.0, lr=0.1, max_iter=100, random_state=42)`  
**Training Dataset:** 5,000,000 rows (Days 01–08 chronological, 10:1 negative subsampling)  
**Validation Dataset:** 65,531,125 rows (Day 09 full natural streaming inference)  

---

## 1. Training Performance & Resource Consumption

| Dimension | Measured Value |
|:---|:---|
| **Exact Training Samples** | **5,000,000** rows |
| **Positive Count ($Y=1$)** | **454,545** ($9.0909\%$) |
| **Negative Count ($Y=0$)** | **4,545,455** ($90.9091\%$) |
| **Load Runtime** | **2.01s** |
| **Fit Runtime** | **174.82s** (~2.91 min) |
| **Peak RAM During Fit** | **1798.83 MB** (~1.76 GB) |
| **Actual Tree Count** | **100 trees** |
| **Saved Model Artifact** | `ml/artifacts/model_5m_config6.joblib` |

---

## 2. Complete Day 09 Natural Validation Comparison

| Metric | 5M Baseline (Config 1) | 1M Config 6 | 5M Config 6 (New) | $\Delta$ vs 5M Baseline | $\Delta$ vs 1M Config 6 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Hyperparameters** | `leaves=31, min=20, l2=1.0` | `leaves=63, min=50, l2=3.0` | `leaves=63, min=50, l2=3.0` | Optimized Hyperparameters | Scaled Data (1M $\to$ 5M) |
| **PR-AUC** | 0.3034 | 0.3137 | **0.3245** | **+0.0211 (+6.94\%)** | **+0.0108 (+3.43\%)** |
| **ROC-AUC** | 0.9731 | 0.9735 | **0.9740** | +0.0009 | +0.0005 |
| **Precision (raw 0.5)** | 0.1286 | 0.1312 | **0.1338** (13.38\%) | +0.0052 | +0.0026 |
| **Recall (raw 0.5)** | 0.7809 | 0.7817 | **0.7801** (78.01\%) | -0.0008 | -0.0016 |
| **F2 Score** | 0.3877 | 0.3925 | **0.3968** | +0.0091 | +0.0043 |
| **Brier Score** | 0.013731 | 0.013491 | **0.013286** | -0.000445 | -0.000205 |
| **Inference Time (65.5M)** | 273.67s | 595.25s | **389.67s** | - | - |
| **Throughput (samples/s)** | 239,451 s/s | 110,091 s/s | **168,169 s/s** | - | - |

---

## 3. Scaling & Readiness Assessment

* **Data Scaling Effect (1M $\to$ 5M on Config 6):** 5M dataset achieved additional PR-AUC gains over 1M Config 6.
* **Architecture Effect (5M Baseline $\to$ 5M Config 6):** Config 6 delivers **+0.0211 (+6.94\%)** PR-AUC gain over the baseline configuration on the full 5M training scale.
* **Test Suite & Safety:** Test suite is 100% passing (50/50 tests), zero out-of-core memory errors, and zero data leakage.
* **Readiness:** The 5M candidate model is now locked and ready for probability calibration and threshold evaluation.
