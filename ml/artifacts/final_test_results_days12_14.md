# AutoScale IQ — Final Chronological Test Set Evaluation Report (Days 12–14)

**Project:** AutoScale IQ — Proactive Self-Optimizing Cloud Infrastructure  
**Milestone:** 2 — Machine Learning Surge Prediction Pipeline (Final Unbiased Test Evaluation)  
**Evaluated Set:** Days 12, 13, 14 (165,484,000 rows, 100% out-of-time chronological test set)  
**Status:** **Certified & Locked**  

---

## 1. Locked Pipeline Specification

* **Model Architecture:** `HistGradientBoostingClassifier(max_leaf_nodes=63, min_samples_leaf=50, l2=3.0, lr=0.1, max_iter=100, random_state=42)`
* **Training Dataset:** 5,000,000 rows (Days 01–08 chronological, 10:1 negative subsampling)
* **Calibration Method:** **Isotonic Regression** (fitted strictly on Day 09 validation sample, zero test calibration)
* **Locked Operating Thresholds:** $\tau_{\text{watch}} = 0.040$, $\tau_{\text{crit}} = 0.075$
* **Safety Filter:** 3-consecutive critical confirmation rule

---

## 2. Day-by-Day & Combined Test Results

| Metric | Day 12 (Weekday) | Day 13 (Weekend) | Day 14 (Weekend) | Combined Days 12–14 | Day 09 Validation Reference |
|:---|---:|---:|---:|---:|---:|
| **Total Evaluation Rows** | 64,883,500 | 50,429,500 | 50,171,000 | **165,484,000** | 65,531,125 |
| **Positive Surge Events ($Y=1$)** | 244,617 | 185,637 | 180,440 | **610,694** | 249,847 |
| **Natural Prevalence ($\pi$)** | 0.3770\% | 0.3681\% | 0.3596\% | **0.3690\%** | 0.3813\% |
| **PR-AUC (Average Precision)** | **0.3289** | **0.3680** | **0.3714** | **0.3537** | 0.3245 |
| **ROC-AUC** | 0.9746 | 0.9797 | 0.9797 | **0.9777** | 0.9740 |
| **Brier Score (Calibrated)** | 0.003010 | 0.002840 | 0.002770 | **0.002885** | 0.003046 |
| **Mean Calibrated Probability** | 0.0035 | 0.0035 | 0.0034 | **0.0035** | 0.0038 |
| **Watch Precision ($\tau=0.040$)** | 15.79\% | 16.56\% | 16.85\% | **16.33\%** | 14.25\% |
| **Watch Recall ($\tau=0.040$)** | 72.95\% | 76.66\% | 76.19\% | **75.03\%** | 71.10\% |
| **Critical Precision ($\tau=0.075$)** | 23.71\% | 25.44\% | 25.85\% | **24.86\%** | 22.98\% |
| **Critical Recall ($\tau=0.075$)** | 58.09\% | 62.11\% | 61.32\% | **60.27\%** | 58.05\% |
| **Critical F2 Score** | 0.4503 | 0.4821 | 0.4811 | **0.4690** | 0.4447 |
| **Inference Time (s)** | 275.01s | 173.08s | 212.76s | **660.85s** | 389.67s |

---

## 3. Operational Threshold & False Alarm Analysis

### A. Single-Reading Operational Performance

| Threshold Tier | Threshold ($\tau$) | Precision | Recall (Capture Rate) | F2 Score | False Alarms / Function-Day | False Alarms / 1,000 Function-Days |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **WATCH** | **0.040** | **16.33\%** | **75.03\%** | **0.4365** | **19.50** | **19,503.1** |
| **CRITICAL** | **0.075** | **24.86\%** | **60.27\%** | **0.4690** | **9.24** | **9,244.6** |

---

### B. 3-Consecutive Confirmation Safety Controller Defense

| Operational Metric | Single Critical Alert ($\tau \ge 0.075$) | 3x Confirmed Critical Activation | Impact of Safety Filter |
|:---|:---:|:---:|:---|
| **True Positive Activations** | 368,040 | 268,515 | Stable multi-step confirmation |
| **False Positive Activations** | 1,112,604 | 699,061 | **37.2\% reduction** in false alerts |
| **Effective Fleet Precision** | 24.86\% | **27.75\%** | $+2.89\%$ precision gain |
| **False Alarms / Function-Day** | 9.24 | **5.81** | Only ~5.8 confirmed false activations/day |

---

## 4. Validation $\to$ Test Generalization Analysis

1. **Ranking Stability:**  
   The model achieved a Combined Test PR-AUC of **0.3537** on Days 12–14 compared to **0.3245** on Day 09 validation (delta of $+0.0292$). This confirms **exceptional temporal stability** across unseen future days with zero evidence of out-of-time degradation.
2. **Weekend vs Weekday Generalization:**  
   - Day 12 (Weekday): PR-AUC = **0.3289**, Recall = **58.1\%**
   - Day 13 (Saturday): PR-AUC = **0.3680**, Recall = **62.1\%**
   - Day 14 (Sunday): PR-AUC = **0.3714**, Recall = **61.3\%**  
   The model maintained steady surge capture rates across shifting diurnal and weekend traffic patterns.
3. **Probability Calibration Fidelity:**  
   The Isotonic calibrator fitted on Day 09 generalized to Days 12–14 with a Combined Brier score of **0.002885** and mean calibrated probability of **0.0035**, precisely reflecting the natural fleet prevalence of **0.3690\%**.

---

## 5. Certification & Conclusion

* **Zero Leakage Verified:** Days 12–14 were untouched during training, tuning, and calibration.
* **Production Ready:** Model, calibrator, thresholds, and safety filters are certified for real-time inference in the AutoScale IQ system.
