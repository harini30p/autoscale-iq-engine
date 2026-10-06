# AutoScale IQ — ML-Powered Self-Optimizing Web Application

**AutoScale IQ** is an autonomous web application performance management and self-optimization system designed to predict traffic surges using machine learning and safely apply application-level optimizations before resource exhaustion occurs.

> **Scope Note:**
> AutoScale IQ performs **application-level optimization** (caching toggles, query pagination tuning, and non-essential component shedding). It is **not** a Kubernetes or cloud infrastructure autoscaler (it does not spin up VMs or container pods).

---

## 1. Core Runtime Loop

AutoScale IQ operates in a continuous defense-in-depth loop:

$$\text{MONITOR} \longrightarrow \text{PREDICT} \longrightarrow \text{DECIDE} \longrightarrow \text{OPTIMIZE} \longrightarrow \text{MEASURE} \longrightarrow \text{RECOVER}$$

1. **MONITOR**: Ingests fresh 1-minute system telemetry observations (traffic volume, CPU, memory, latency, active users, DB query time). Stale metrics ($> 30\text{s}$) are rejected for safety.
2. **PREDICT**: Extracts a locked 19-dimensional feature vector and evaluates a trained `HistGradientBoostingClassifier` with isotonic probability calibration to estimate surge likelihood ($0 \le p \le 1$).
3. **DECIDE**: Maps probability into advisory risk signals (`NORMAL`, `ELEVATED`, `CRITICAL`). The `SafetyController` validates state transitions against confirmation counters (3 consecutive high-risk readings required), hard safety bounds, cooldown timers, and manual overrides.
4. **OPTIMIZE**: If confirmed, the `ApplicationOptimizer` safely and idempotently transitions application levers (enables caching, reduces pagination size to 20, disables heavy components).
5. **MEASURE**: Telemetry ingestion continues post-optimization to assess system responsiveness under adjusted configuration.
6. **RECOVER**: Once 3 consecutive `NORMAL` observations are confirmed in `RECOVERY` state, the optimizer safely restores default application configuration.

---

## 2. Architecture Overview

```
+-----------------------------------------------------------------------+
|                    Frontend (React 19 + Vite)                         |
|   • Observability UI  • Live State Polling  • ML Risk Panel           |
|   • Telemetry Cards   • Events Audit Trail  • Safety Override Control |
+-----------------------------------+-----------------------------------+
                                    | HTTP / JSON (CORS Enabled)
                                    v
+-----------------------------------------------------------------------+
|                         FastAPI Backend                               |
|   /health  |  /state  |  /events  |  /monitor  |  /predict  | /override   |
+-----------------------------------+-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  ML Feature Engine    |                       | Freshness & Hard      |
|  • 19-Feature Vector  |                       | Safety Check          |
|  • Rolling Window 61t |                       | • Stale Check (>30s)  |
|  • Trigger Flags      |                       | • Hard CPU/RAM Limits |
+-----------+-----------+                       +-----------+-----------+
            |                                               |
            v                                               |
+-----------------------+                                   |
| SurgePredictor (ML)   |                                   |
| • HistGradientBoost   |                                   |
| • Isotonic Calibrator |                                   |
| • tau_watch = 0.040   |                                   |
| • tau_crit  = 0.075   |                                   |
+-----------+-----------+                                   |
            |                                               |
            | (Advisory Risk Signal)                        | (Hard Thresholds)
            +-----------------------+-----------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                       Safety Controller                               |
|   State Machine: NORMAL <-> WATCHING -> OPTIMIZED <-> RECOVERY        |
|   • 3 Consecutive High-Risk Confirmations                             |
|   • Post-Optimization Cooldown (30s default)                          |
|   • Recovery Hysteresis (3 Consecutive Normal Confirmations)          |
|   • Manual Override Safety Gate                                       |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                      Application Optimizer                            |
|   • Caching: disabled <-> enabled                                     |
|   • Pagination: 50 <-> 20 items/page                                  |
|   • Heavy Components: enabled <-> disabled                            |
|   • Idempotent State Mutation & Safe Error Trapping                   |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                          SQLite Database                              |
|   • metrics  • ml_predictions  • optimization_events  • system_state  |
+-----------------------------------------------------------------------+
```

---

## 3. Machine Learning Surge Prediction System

- **Dataset**: Azure Functions 2019 Trace Dataset (chronologically partitioned).
- **Model Architecture**: `HistGradientBoostingClassifier` (5M-row training dataset, 10:1 negative-to-positive sampling ratio).
- **Locked 19-Feature Schema**:
  - *Workload (1–6)*: `invocations_t`, `rolling_mean_5m`, `rolling_mean_15m`, `rolling_mean_60m`, `rolling_max_15m`, `rolling_max_60m`
  - *Trend (7–8)*: `rate_delta_1m`, `rate_delta_5m`
  - *Volatility (9)*: `rolling_std_15m`
  - *Time Encoding (10–12)*: `minute_of_day`, `minute_sin`, `minute_cos`
  - *Trigger One-Hot (13–19)*: `trigger_timer`, `trigger_http`, `trigger_queue`, `trigger_orchestration`, `trigger_event`, `trigger_storage`, `trigger_others`
- **Calibration**: Isotonic Regression fitted on natural validation distribution.
- **Operational Thresholds**:
  - $\tau_{\text{watch}} = 0.040$ ($p \ge 0.040 \rightarrow \text{ELEVATED}$)
  - $\tau_{\text{crit}} = 0.075$ ($p \ge 0.075 \rightarrow \text{CRITICAL}$)
- **Advisory Role**: The ML engine outputs probability and risk classification; it **never** directly mutates application state or triggers optimization without controller confirmation.

---

## 4. Safety Mechanisms

The system implements multiple deterministic safety bounds to protect stability:

1. **Stale Metric Detection**: Ingestion verifies metric age relative to UTC. Readings older than 30 seconds are labeled stale, skip ML inference, and cannot advance confirmation counters.
2. **Consecutive High-Risk Confirmation**: Requires **3 consecutive** `CRITICAL` signals before transitioning to `OPTIMIZED`.
3. **Hard Safety Thresholds**: Deterministic emergency triggers that bypass ML confirmation:
   - CPU Utilization $\ge 90\%$
   - Memory Utilization $\ge 90\%$
   - Response Time $\ge 1000\text{ ms}$
   - System Load $\ge 4.0$
4. **Post-Optimization Cooldown**: Prevents rapid re-triggering loops after an optimization action (default 30 seconds).
5. **Recovery Hysteresis**: Requires **3 consecutive** `NORMAL` readings while in `RECOVERY` before restoring default configuration.
6. **Manual Override**: Immediate manual circuit-breaker blocking all automatic optimizer mutations.
7. **Graceful Failure Degradation**: Optimizer or DB logging failures trap exceptions safely without crashing the monitoring endpoint.

---

## 5. Application-Level Optimizations

When high risk is confirmed, the optimizer applies application-level configurations:

| Lever | Normal State | Optimized State | Impact |
|---|---|---|---|
| **Caching Tier** | `disabled` | `enabled` | Reduces database & computational load |
| **Query Pagination** | `50 items/page` | `20 items/page` | Reduces memory consumption & serialization overhead |
| **Heavy Components** | `enabled` | `disabled` | Sheds non-essential analytics & background rendering |

---

## 6. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and current UTC heartbeat timestamp |
| `GET` | `/state` | Returns current controller state, active levers, cooldown info, and confirmation counts |
| `GET` | `/events` | Returns recent optimization and recovery event audit logs |
| `POST` | `/override` | Enables or disables the manual safety override |
| `POST` | `/monitor` | Primary ingestion endpoint: runs freshness check, ML inference, prediction logging, and controller step |
| `POST` | `/predict` | Direct standalone ML inference endpoint (read-only; does not step controller or persist) |
| `POST` | `/metrics/validate` | Validates a metric payload against the Pydantic schema |

---

## 7. SQLite Persistence Layer

Database schema managed in `backend/database.py`:

- `metrics`: Log of all ingested telemetry observations.
- `ml_predictions`: Log of ML inference calls (calibrated probability, risk signal, thresholds, timestamp, 19-feature snapshot).
- `optimization_events`: Audit trail of every optimization and recovery action (timestamps, reason, previous state, new state, success status).
- `system_state`: Single-row persistent state storing controller state, active levers, manual override flag, and confirmation counters.

---

## 8. React Observability Dashboard

Built with **React 19** and **Vite** in `frontend/`:

- **Centralized API Client**: (`src/services/api.js`) Configurable via `VITE_API_BASE_URL`.
- **Backend Health Badge**: Real-time connection status and heartbeat timestamp indicator.
- **System Status Banner**: Displays active controller state (`NORMAL`, `WATCHING`, `OPTIMIZED`, `RECOVERY`), levers, cooldown countdown, and manual override toggle.
- **Telemetry Metric Cards**: Workload indicators (Traffic, Response Time, CPU, Memory, Active Users, DB Query Time) with clean waiting states.
- **ML Surge Prediction Panel**: Real-time calibrated surge probability bar with threshold markers ($\tau_{\text{watch}}=0.040, \tau_{\text{crit}}=0.075$), risk signal pill, and inference latency.
- **Optimization Events Table**: Live audit log of state mutations with before/after state transitions.
- **Live Polling**: 5-second automatic refresh with pause/resume controls.

---

## 9. Installation & Running

### Backend Setup

1. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### Frontend Setup

1. Install Node dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the Vite development server:
   ```bash
   npm run dev
   ```
   - Dashboard: [http://localhost:5173](http://localhost:5173)

3. Build production bundle:
   ```bash
   npm run build
   ```

---

## 10. Test Suite & Verification

Run the full pytest test suite (114 unit, ML artifact smoke, and runtime simulation tests):

```bash
python -m pytest -q
```

**Test Status:** `114 passed, 0 failed`

---

## 11. Current Status & Roadmap

### Completed Milestones
- [x] Backend safety foundation, state machine, and optimizer
- [x] Azure Functions 2019 dataset preprocessing and feature pipeline
- [x] 5M-row `HistGradientBoostingClassifier` training, isotonic calibration, and threshold locking
- [x] Backend ML integration and prediction persistence (`ml_predictions`)
- [x] Full runtime lifecycle integration (`ML -> Controller -> Optimizer -> Recovery`)
- [x] React observability dashboard foundation and API client

### Upcoming Milestones
- [ ] Historical metric/time-series API endpoints
- [ ] Rolling time-series charts in the dashboard
- [ ] Interactive traffic surge simulation scenarios
- [ ] Real-time demonstration flow polish
