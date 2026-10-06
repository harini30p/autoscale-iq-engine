# AutoScale IQ — 1M Hyperparameter Search Report

**Experiment:** Milestone 2 — Compact Hyperparameter Search  
**Dataset:** 1,000,000 Chronological Training Rows (Days 01–08, 10:1 Subsampling)  
**Validation:** 2,000,000 Representative Natural Sample $\to$ Full Day 09 (65,531,125 rows) Verification  

---

## 1. 8-Configuration Search Results (2M Validation Sample)

| Rank | Config | Max Leaves | Min Samples Leaf | L2 Reg | PR-AUC | ROC-AUC | Precision | Recall | F2 Score | Fit Time (s) | Peak RAM (MB) | Trees |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Config 3** | 63 | 50 | 1.0 | **0.2977** | 0.9709 | 0.1168 | 0.7600 | 0.3616 | 21.43s | 382.9 MB | 100 |
| 2 | **Config 6** | 63 | 50 | 3.0 | **0.2942** | 0.9708 | 0.1177 | 0.7601 | 0.3634 | 47.08s | 378.7 MB | 100 |
| 3 | **Config 2** | 63 | 20 | 1.0 | **0.2886** | 0.9708 | 0.1163 | 0.7602 | 0.3607 | 44.66s | 380.3 MB | 100 |
| 4 | **Config 8** | 63 | 100 | 3.0 | **0.2806** | 0.9708 | 0.1162 | 0.7584 | 0.3602 | 36.72s | 379.1 MB | 100 |
| 5 | **Config 4** | 31 | 50 | 1.0 | **0.2778** | 0.9702 | 0.1130 | 0.7610 | 0.3544 | 15.32s | 383.9 MB | 100 |
| 6 | **Config 5** | 31 | 20 | 3.0 | **0.2754** | 0.9704 | 0.1118 | 0.7605 | 0.3520 | 41.12s | 379.1 MB | 100 |
| 7 | **Config 1** | 31 | 20 | 1.0 | **0.2652** | 0.9702 | 0.1119 | 0.7622 | 0.3525 | 42.86s | 380.4 MB | 100 |
| 8 | **Config 7** | 31 | 100 | 3.0 | **0.2651** | 0.9702 | 0.1133 | 0.7624 | 0.3552 | 10.65s | 383.0 MB | 100 |

---

## 2. Top 2 Finalists Full Day 09 Verification (65,531,125 rows)

| Metric | Finalist #1: Config 3 | Finalist #2: Config 6 | Delta / Comparison |
|:---|:---:|:---:|:---:|
| **Parameters** | `leaves=63, min_leaf=50, l2=1.0` | `leaves=63, min_leaf=50, l2=3.0` | Architecture variation |
| **Day 09 PR-AUC** | **0.3135** | **0.3137** | $\Delta = -0.0002$ |
| **Day 09 ROC-AUC** | 0.9735 | 0.9735 | $\Delta = -0.0000$ |
| **Day 09 Precision (raw 0.5)** | 0.1309 (13.09%) | 0.1312 (13.12%) | - |
| **Day 09 Recall (raw 0.5)** | 0.7815 (78.15%) | 0.7817 (78.17%) | - |
| **Day 09 F2 Score** | 0.3920 | 0.3925 | - |
| **Day 09 Brier Score** | 0.013560 | 0.013491 | - |
| **Fit Runtime** | 21.43s | 47.08s | - |
| **Peak RAM** | 382.9 MB | 378.7 MB | - |
| **Inference Time (65.5M rows)** | 348.09s | 595.25s | - |
| **Inference Throughput** | 188,261 samples/s | 110,091 samples/s | - |

---

## 3. Selected Winner

```text
WINNER: Config 6
max_leaf_nodes   = 63
min_samples_leaf = 50
l2_regularization= 3.0
```

### Rationale:
- **Decision Rule:** Highest full Day 09 PR-AUC (0.3137) with equal tree complexity.
- **Day 09 PR-AUC:** **0.3137**
- **1M Baseline Comparison:** Baseline PR-AUC was $0.2923$. Selected configuration achieves **0.3137** (+0.0214 improvement).
- **Readiness for 5M Retraining:** Verified leak-free, memory-safe, and ready for 5M final training.
