# AutoScale IQ — ML-Powered Self-Optimizing Web Application

## Milestone 1: Backend Safety Foundation

**AutoScale IQ** is an autonomous web application performance management and self-optimization system designed to protect application responsiveness and stability.

> **Scope Note for Milestone 1:**
> This milestone establishes the **real, tested FastAPI backend safety foundation**.
> - **Machine Learning**: Not implemented yet (future milestone). The controller currently uses deterministic safety thresholds and generic risk signals (`normal`, `elevated`, `critical`).
> - **Frontend UI / React Dashboard**: Not implemented yet (future milestone).
> - **Simulation Engine**: Not implemented yet (future milestone).
> - **Infrastructure Autoscaling**: Not implemented; all optimizations are application-level configuration changes.

---

## 1. Architecture Overview

AutoScale IQ is built around a defense-in-depth safety architecture that strictly separates:
$$\text{Risk Detection} \longrightarrow \text{Safety Decision} \longrightarrow \text{Application Optimization}$$

```
+-----------------------------------------------------------------------+
|                             Client / API                              |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                             FastAPI App                               |
|   /health  |  /metrics/validate  |  /monitor  |  /state  |  /events   |
+-----------------------------------+-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  Metric Validation    |                       | Freshness & Safety    |
|   (Pydantic Schema)   |                       | (Stale Check & Hard   |
|   • Finite Bounds     |                       |  Safety Thresholds)   |
|   • UTC Normalization |                       +-----------+-----------+
+-----------+-----------+                                   |
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                       Safety Controller                               |
|   State Machine: NORMAL <-> WATCHING -> OPTIMIZED <-> RECOVERY        |
|   • 3 Consecutive High-Risk Confirmations                             |
|   • Post-Optimization Cooldown (30s default)                          |
|   • Recovery Hysteresis (3 Consecutive Normal Confirmations)          |
|   • Manual Override Gate                                              |
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
|   • metrics  • optimization_events  • system_state  • predictions     |
+-----------------------------------------------------------------------+
```

---

## 2. Backend Components

### A. Metric Schema & Validation (`backend/schemas.py`)
- Centralized Pydantic schemas validating:
  - `timestamp`: UTC ISO datetime
  - `traffic`: Non-negative request volume
  - `active_users`: Non-negative integer
  - `cpu_utilization`: $0 \le \text{CPU} \le 100$ %
  - `memory_utilization`: $0 \le \text{Memory} \le 100$ %
  - `response_time`: Non-negative response latency (ms)
  - `db_query_time`: Non-negative database query latency (ms)
  - `system_load`: Non-negative system load indicator
- Automatically rejects `NaN`, `Infinity`, negative numbers, missing fields, and malformed timestamps with HTTP 422.

### B. Freshness & Safety Thresholds (`backend/monitoring.py`)
- **Stale Metric Detection**: Configurable staleness window (`DEFAULT_STALE_THRESHOLD_SECONDS = 10.0`). Stale metrics are recorded in SQLite for auditability but are **strictly blocked** from advancing confirmation counters or triggering optimizations.
- **Hard Safety Thresholds**: Deterministic safety bounds that allow rapid emergency escalation:
  - CPU Utilization $\ge 90\%$
  - Memory Utilization $\ge 90\%$
  - Response Time $\ge 1000\text{ ms}$
  - System Load $\ge 5.0$

### C. Safety Controller (`backend/controller.py`)
Maintains a formal state machine:
- `NORMAL`: Standard operating conditions.
- `WATCHING`: Elevated risk detected or initial high-risk readings received; accumulating confirmations.
- `OPTIMIZED`: Application optimization levers active.
- `RECOVERY`: Sustained normal metrics being confirmed before restoring default configuration.

**Safety Controls:**
1. **High-Risk Confirmation**: Requires 3 consecutive high-risk readings before automatic optimization.
2. **Cooldown Period**: Post-optimization cooldown (default 30 seconds) prevents rapid re-triggering loops.
3. **Recovery Hysteresis**: Requires 3 consecutive normal readings before reverting optimizations to prevent flapping/oscillation.
4. **Manual Override**: Master switch allowing operators to pause automation immediately.

### D. Application Optimizer (`backend/optimizer.py`)
Applies and reverts application-level configurations:
- **Caching**: `disabled` (Normal) $\longleftrightarrow$ `enabled` (Optimized)
- **Pagination Size**: `50 items/page` (Normal) $\longleftrightarrow$ `20 items/page` (Optimized)
- **Heavy Components**: `enabled` (Normal) $\longleftrightarrow$ `disabled` (Optimized)
- All state transitions are idempotent and record structured audit logs into SQLite (`optimization_events`).

### E. SQLite Persistence Layer (`backend/database.py`)
- `metrics`: Time-series log of all ingested metric readings.
- `optimization_events`: Complete audit log of every optimization and recovery event.
- `system_state`: Single-source-of-truth table recording current configuration, controller state, and counters.
- `predictions`: Schema placeholder for future ML integration.

---

## 3. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and current UTC time |
| `POST` | `/metrics/validate` | Validates a metric payload without processing |
| `POST` | `/monitor` | Ingests metric, checks freshness, evaluates safety, and steps controller |
| `GET` | `/state` | Returns current controller state, configuration, and cooldown |
| `GET` | `/events` | Returns recent optimization and recovery event logs |
| `POST` | `/override` | Enables or disables manual safety override |

---

## 4. Installation and Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)

### Installation
1. Clone repository:
   ```bash
   git clone <repo-url>
   cd autoscale-iq-engine
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## 5. Running the Backend Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive OpenAPI Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interactive ReDoc documentation: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 6. Running Tests

Run the full pytest test suite:

```bash
python -m pytest -q
```

To run with verbose output and coverage:
```bash
python -m pytest -v
```

---

## 7. Current Limitations & Upcoming Milestones

- **No ML Model**: Milestone 1 uses deterministic safety thresholds and generic risk signals. Future milestones will introduce Azure telemetry datasets, feature engineering, and trained ML failure prediction models.
- **No Frontend**: The React dashboard will be developed in a future milestone.
- **Application-Level Levers Only**: Does not provision or scale cloud VMs/containers; instead, it optimizes the running application via caching, page sizing, and shedding non-essential components.
