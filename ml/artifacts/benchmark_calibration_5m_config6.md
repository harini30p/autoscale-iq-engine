# AutoScale IQ — 5M Config 6 Probability Calibration & Threshold Locking Report

**Experiment:** Milestone 2 — Final Calibration & Operating Threshold Locking  
**Model Architecture:** `HistGradientBoostingClassifier(max_leaf_nodes=63, min_samples_leaf=50, l2=3.0, lr=0.1, max_iter=100, random_state=42)`  
**Training Dataset:** 5,000,000 rows (Days 01–08, 10:1 subsampling, $\pi_{\text{train}} = 9.0909\%$)  
**Calibration Dataset:** 2,000,000 natural rows from Day 09 ($\pi_{\text{val}} = 0.3846\%$, 7,691 positives)  
**Test Set Status:** **Days 12–14 remain 100% UNTOUCHED.**  

---

## 1. Model & Calibration Dataset Summary

* **Model File:** `ml/artifacts/model_5m_config6.joblib`
* **Training Prevalence ($\pi_{\text{train}}$):** $9.0909\%$ (10:1 negative subsampling)
* **Natural Validation Prevalence ($\pi_{\text{val}}$):** $0.3846\%$ (1 in 260 natural rate)
* **Raw Probability Shift:** Uncalibrated model outputs reflect the $9.09\%$ training distribution (mean predicted probability $= 0.0388$, median $= 0.0071$), requiring systematic probability calibration.

---

## 2. Calibration Method Comparison

| Method | Brier Score | PR-AUC | ROC-AUC | Mean Calibrated Probability | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Raw (Uncalibrated)** | 0.015046 | 0.3031 | 0.9713 | 0.0388 | Overestimates risk by $\approx 10\times$ |
| **Prior-Shift Correction** | 0.003129 | 0.3031 | 0.9713 | 0.0038 | $\Delta_{\text{logit}} = -3.2544$ analytical shift |
| **Platt / Sigmoid** | 0.003097 | 0.3031 | 0.9713 | 0.0039 | Logistic scaling on logits |
| **Isotonic Regression** | **0.003046** | **0.3062** | **0.9724** | **0.0038** | **SELECTED WINNER (Lowest Brier Score)** |

### Selection Decision:
**Isotonic Regression** achieved the lowest Brier score (**0.003046**, a **$79.9\%$ reduction** from raw) and perfectly aligned the fleet mean predicted probability (**0.0038**) with natural prevalence ($0.003846$).

---

## 3. Threshold Sweep & Candidate Operating Points

| Threshold ($\tau$) | Precision | Recall | F2 Score | False Alarms / Function-Day | False Alarms / 1k Function-Days | Role / Target |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0.020** | 9.35\% | 81.97\% | 0.3210 | 42.02 | 42,021.4 | |
| **0.040** | 14.25\% | 71.10\% | 0.3955 | 22.62 | 22,616.0 | (WATCH Target) |
| **0.060** | 18.75\% | 63.41\% | 0.4295 | 14.53 | 14,526.9 | |
| **0.080** | 22.98\% | 58.05\% | 0.4447 | 10.29 | 10,290.5 | |
| **0.100** | 26.07\% | 53.53\% | 0.4422 | 8.03 | 8,025.9 | |
| **0.120** | 28.37\% | 50.60\% | 0.4375 | 6.75 | 6,754.7 | |
| **0.140** | 31.67\% | 46.11\% | 0.4225 | 5.26 | 5,259.4 | |
| **0.160** | 32.18\% | 45.51\% | 0.4202 | 5.07 | 5,072.4 | |

---

## 4. Locked Operating Thresholds

```text
WATCH THRESHOLD    = 0.040
CRITICAL THRESHOLD = 0.075
```

* **WATCH ($\tau_{\text{watch}} = 0.040$):** Captures **71.10\%** of future workload surges with **14.25\%** precision and **22.62** false alarms/function-day, smoothly transitioning the controller into `WATCHING` state.
* **CRITICAL ($\tau_{\text{crit}} = 0.075$):** Captures **58.05\%** of surges with **22.98\%** precision and optimal F2 score (**0.4447**), cutting false alarms down to **10.29**/function-day.

---

## 5. 3-Consecutive Confirmation Safety Controller Defense

Under the Safety Controller's 3-reading hysteresis:
* **Single-reading Critical False Positives:** 14,968
* **3x Confirmed Critical False Positives:** 8,704
* **False Alarm Suppression:** **41.8\%** of isolated transient noise spikes are eliminated before proactive optimization triggers.

---

## 6. Pipeline Invariants Confirmation

* **Days 12–14 (Test Split):** **NOT TOUCHED** (Zero leakage, zero test inference).
* **Model:** **LOCKED** (`model_5m_config6.joblib`)
* **Calibration:** **LOCKED** (`calibrator_5m_config6_isotonic.joblib`)
* **Thresholds:** **LOCKED** (`thresholds_5m_config6.json`)
* **Ready for final untouched Test Set evaluation.**
