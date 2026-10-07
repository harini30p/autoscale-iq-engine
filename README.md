# AutoScale IQ

AutoScale IQ is a student project for demonstrating telemetry-driven application optimization decisions. It ingests application metrics, uses a machine-learning model to estimate potential high-load risk, applies controller-gated configuration changes, records subsequent telemetry and event snapshots, and restores the default configuration when conditions recover.

It is a deployed demonstration system, not a cloud or infrastructure autoscaler. It does not deploy or scale virtual machines, containers, or cloud services.

## Deployment

- **Frontend:** Deployed on Vercel at [https://autoscale-iq-engine.vercel.app/](https://autoscale-iq-engine.vercel.app/).
- **FastAPI backend:** Deployed on Render at [https://autoscale-iq-backend.onrender.com](https://autoscale-iq-backend.onrender.com/).
- **Swagger API documentation:** [https://autoscale-iq-backend.onrender.com/docs](https://autoscale-iq-backend.onrender.com/docs).

This deployment is a demonstration of the self-optimizing web application workflow. Demo telemetry and scenarios are controlled simulated telemetry; the ML model was trained using the real Azure Functions 2019 dataset. The optimization layer changes simulated, persisted configuration only—it is not connected to actual cloud infrastructure autoscaling or a production application's serving behavior.

## Architecture

```text
MONITOR → PREDICT → DECIDE → OPTIMIZE → MEASURE → RECOVER
```

1. **Monitor:** The FastAPI backend validates telemetry and records it in SQLite. Fresh observations are passed to the prediction and controller paths.
2. **Predict:** The ML pipeline builds a rolling feature window and returns a calibrated surge probability and risk classification.
3. **Decide:** The safety controller considers the risk signal, telemetry freshness, deterministic hard thresholds, confirmation counts, cooldown, current state, and manual override.
4. **Optimize:** The optimizer records changes to the project's simulated application configuration.
5. **Measure:** Later telemetry is recorded; optimization events can include before/after metric snapshots and calculated differences.
6. **Recover:** When normal conditions are confirmed, the controller returns to `NORMAL` and the optimizer restores default configuration.

The **FastAPI backend** provides the runtime and HTTP API. **SQLite** stores telemetry, predictions, controller configuration, and optimization/recovery events. The **controller** is the source of truth for state transitions and safety gates; the **optimizer** applies and persists configuration changes. The **React dashboard** presents current state, telemetry, predictions, event history, and interactive scenarios. The **simulation engine** creates synthetic metric sequences; playback sends them to the same `/monitor` endpoint used by other telemetry.

## Machine-learning pipeline

The runtime loads a locked `HistGradientBoostingClassifier` and isotonic calibrator from `ml/artifacts/`. For each observation, it derives a 19-feature vector from a rolling window of invocation observations:

- Workload: current invocations, rolling means (5, 15, and 60 observations), and rolling maxima (15 and 60).
- Trend and volatility: 1- and 5-observation rate deltas and the 15-observation rolling standard deviation.
- Time: minute of day and its sine/cosine encoding.
- Trigger type: seven one-hot flags.

The current model artifacts use `0.040` as the watch threshold and `0.075` as the critical threshold. Probabilities below watch are `NORMAL`; probabilities at or above watch and below critical are `ELEVATED`; probabilities at or above critical are `CRITICAL`. These are model risk classifications, not optimization commands. The ML result is advisory to the controller; deterministic safety checks are evaluated separately and can trigger immediate optimization for a fresh observation when automation is not blocked by override or cooldown.

The model expects a full rolling window of 61 observations (60 prior observations plus the current one). Until then, it can produce predictions with insufficient data, so early results should be interpreted accordingly.

## Safety and controller

Current defaults from `backend/config.py`:

| Guard or setting | Value |
|---|---:|
| Stale telemetry threshold | 10 seconds |
| Consecutive critical confirmations | 3 |
| Consecutive normal recovery confirmations | 3 |
| Optimization cooldown | 30 seconds |
| CPU hard threshold | 90% |
| Memory hard threshold | 90% |
| Response-time hard threshold | 1000 ms |
| System-load hard threshold | 5.0 |

Stale telemetry is stored but does not run ML inference or advance automation; the controller blocks automatic action and leaves confirmation counters unchanged. For fresh telemetry, hard thresholds are deterministic checks independent of ML risk and bypass the normal three-critical-confirmation path. A manual override blocks automatic controller action, including hard-threshold-triggered optimization. Three consecutive critical readings are required for the normal ML-driven transition; three consecutive normal readings are required to restore defaults from optimized/recovery states. Elevated or critical readings interrupt recovery. The cooldown blocks repeated optimization during its active period.

## Optimization configuration

The optimizer changes these persisted configuration values:

| Lever | Default | Optimized |
|---|---|---|
| Caching | `disabled` | `enabled` |
| Pagination size | `50` | `20` |
| Heavy components | `enabled` | `disabled` |

These are simulated/persisted configuration settings in this project. They are **not** connected to a production application's cache, pagination implementation, component-shedding behavior, or deployed infrastructure.

## Interactive simulation

The available deterministic scenarios are:

| Scenario | Demonstrates |
|---|---|
| `baseline` | Low-load traffic and ML-window warmup |
| `gradual_surge` | Warmup, rising traffic, controller confirmations, optimization, then cooling telemetry |
| `sudden_spike` | Warmup, abrupt high traffic, then cooling telemetry |
| `hard_safety` | Warmup, CPU/memory hard-threshold readings, then cooling telemetry |
| `recovery` | Rising traffic to establish the optimized state, followed by normal readings for recovery |

The main demonstration sequence is **Start → Normal traffic → Surge → Optimization → Recovery**. Simulation playback assigns fresh timestamps and submits each tick through `POST /monitor`; it uses the same ML, controller, optimizer, and persistence path rather than a separate simulated decision path. The simulation reset restores controller and lever defaults and clears the in-memory ML window, but retains telemetry and event history.

## Limitations

- Simulation telemetry is synthetic and accelerated compared with real elapsed-time observations.
- Simulation results demonstrate the runtime decision flow. Before/after differences are observed values from the supplied simulated telemetry sequence.
- Those differences must **not** be presented as proof that an optimization lever causally improves a production application's performance.
- The optimizer changes only persisted configuration values. It does not modify a deployed application's serving behavior or infrastructure.
- The model's rolling-window predictions depend on accumulated observations; simulation warmup helps populate the window but does not make synthetic data equivalent to production telemetry.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service health and timestamp |
| `GET` | `/state` | Controller state, configuration, override, confirmation counts, and cooldown |
| `POST` | `/monitor` | Main telemetry ingestion, freshness check, ML prediction, controller decision, and possible optimization |
| `POST` | `/predict` | Direct prediction; does not step the controller or persist state |
| `GET` | `/metrics/history` | Recent telemetry history (`limit` query parameter) |
| `GET` | `/predictions/history` | Recent prediction history (`limit` query parameter) |
| `GET` | `/events` | Recent optimization and recovery events (`limit` query parameter) |
| `POST` | `/override` | Enable or disable the manual override using `{"manual_override": true}` or `{"manual_override": false}` |
| `POST` | `/metrics/validate` | Validate a telemetry payload |
| `GET` | `/simulation/scenarios` | List scenario metadata |
| `GET` | `/simulation/scenarios/{name}` | Get scenario metadata and untimestamped ticks |
| `POST` | `/simulation/reset` | Restore controller/lever defaults and clear the ML rolling window |

Interactive API documentation is available at `/docs` when the backend is running.

## Setup and run

The model, calibrator, and threshold artifacts are included under `ml/artifacts/`; running the application does not require running the offline ML training pipeline. ML runtime dependencies are explicitly declared in `backend/requirements.txt`.

From the repository root, create and activate a Python virtual environment, then install backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`. Start the backend from the repository root:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Install and start the frontend in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

The dashboard is served by Vite (usually at `http://localhost:5173`); the backend API and interactive documentation are at `http://localhost:8000` and `http://localhost:8000/docs`.

Build the frontend with:

```bash
cd frontend
npm run build
```

## Testing

Backend tests are under `backend/tests/`. Run them from the repository root:

```bash
python -m pytest backend/tests -q
```

The current backend-suite result is **149 passed**, with one existing Starlette/httpx deprecation warning. Offline ML-pipeline tests are located elsewhere under `ml/tests/` and are separate from the backend suite. Running `python -m pytest -q` from the repository root also discovers those tests, which may need additional data-science dependencies such as `pandas` and access to offline data.

## Project structure

```text
backend/              FastAPI runtime, controller, optimizer, SQLite layer, simulation
backend/tests/        Backend unit and runtime integration tests
frontend/             React dashboard and Vite tooling
ml/                   Offline ML pipeline, documentation, and locked runtime artifacts
```

## Project status

- Milestones 1–3: **Complete**
- Milestone 4 — Interactive Simulation: **Complete**
- Milestone 5 — Demo/Safety Polish: **Complete**
- Milestone 6 — Final NEXUS Preparation: **In progress**

The current milestone focuses on documentation, demo preparation, final validation, and repository readiness; it does not introduce a new ML model or cloud autoscaling architecture.

## Demo runbook

Start the backend and frontend, then use the dashboard's interactive simulation or the simulation API. Reset the controller and levers between flows when needed with `POST /simulation/reset`; this retains history.

1. **Traffic surge → optimization:** Run `gradual_surge`. Watch incoming telemetry, ML risk when available, controller confirmations/state, the optimization event and changed configuration, and subsequent telemetry snapshots.
2. **Recovery → `NORMAL`:** Run `recovery`, which establishes high load and then supplies recovery-shaped normal telemetry. Watch recovery confirmations, the return to `NORMAL`, and restoration of default lever values.
3. **Safety block:** Run `hard_safety` with manual override enabled to demonstrate automation being blocked, or send stale telemetry (a timestamp more than 10 seconds old) through `/monitor`. To demonstrate the deterministic hard-safety response rather than a block, run `hard_safety` with override disabled and observe the hard-threshold decision.

During each flow, distinguish the telemetry and ML risk signal from the controller decision. Observe the resulting state, optimization or blocked reason, persisted configuration, event snapshots, and recovery behavior. UI wording may vary by view; use the actual API response and event data as the evidence for what occurred.
