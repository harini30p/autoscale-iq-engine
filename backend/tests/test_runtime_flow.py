"""
Integration and Runtime Simulation Tests for AutoScale IQ:
Metrics -> Feature Extraction -> ML Prediction (SurgePredictor) ->
Isotonic Calibration -> Risk Signal -> Safety Controller ->
Optimization Decision -> Optimizer -> Recovery

Covers:
- Phase 3: Tests 1 through 8
- Phase 4: Full Runtime Simulation
"""

from datetime import datetime, timezone, timedelta
import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.config import (
    CACHE_NORMAL,
    CACHE_OPTIMIZED,
    CPU_CRITICAL_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    PAGINATION_NORMAL,
    PAGINATION_OPTIMIZED,
    HEAVY_COMPONENTS_NORMAL,
    HEAVY_COMPONENTS_OPTIMIZED,
    ML_MODEL_PATH,
    ML_CALIBRATOR_PATH,
    ML_THRESHOLDS_PATH,
)
from backend.controller import SafetyController
from backend.database import (
    init_db,
    get_system_state,
    get_recent_events,
    get_recent_predictions,
)
from backend.main import create_app
from backend.ml_engine import MLPredictionResult, ObservationTick, SurgePredictor
from backend.schemas import ControllerState, RiskSignal


def _metric_dict(
    timestamp: datetime = None,
    traffic: float = 100.0,
    cpu: float = 40.0,
    memory: float = 40.0,
    response_time: float = 100.0,
    load: float = 1.0,
) -> dict:
    ts = timestamp or datetime.now(timezone.utc)
    return {
        "timestamp": ts.isoformat(),
        "traffic": traffic,
        "active_users": 20,
        "cpu_utilization": cpu,
        "memory_utilization": memory,
        "response_time": response_time,
        "db_query_time": 20.0,
        "system_load": load,
    }


def _mock_ml_result(prob: float, risk: RiskSignal) -> MLPredictionResult:
    return MLPredictionResult(
        surge_probability=prob,
        raw_probability=prob * 0.8,
        risk_signal=risk,
        watch_threshold=0.040,
        critical_threshold=0.075,
        feature_vector=[100.0] + [0.0] * 18,
        window_size=10,
        insufficient_data=True,
        inference_time_ms=0.5,
    )


@pytest.fixture
def test_app_and_client(tmp_path):
    """Fixture providing a clean FastAPI app with ML enabled and TestClient."""
    db_file = str(tmp_path / "test_flow.db")
    init_db(db_file)

    # Initialize predictor singleton with real artifacts if available
    SurgePredictor.reset()
    app = create_app(
        db_path=db_file,
        ml_model_path=ML_MODEL_PATH,
        ml_calibrator_path=ML_CALIBRATOR_PATH,
        ml_thresholds_path=ML_THRESHOLDS_PATH,
        load_ml=True,
    )
    with TestClient(app) as client:
        yield app, client, db_file
    SurgePredictor.reset()


# =============================================================================
# PHASE 3 — Focused Runtime Integration Tests
# =============================================================================


def test_1_normal_ml_signal(test_app_and_client):
    """
    Test 1 — Normal ML signal:
    A fresh observation produces probability below 0.040 -> risk = NORMAL.
    Verify no optimization is triggered solely because ML ran.
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()

    normal_result = _mock_ml_result(0.020, RiskSignal.NORMAL)
    with patch.object(predictor, "predict", return_value=normal_result):
        payload = {"metric": _metric_dict(traffic=50.0)}
        response = client.post("/monitor", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["risk_signal"] == "normal"
    assert data["controller_state"] == "NORMAL"
    assert data["optimization_applied"] is False
    assert data["current_configuration"]["caching"] == CACHE_NORMAL
    assert data["ml_prediction"]["surge_probability"] == 0.020
    assert data["ml_prediction"]["risk_signal"] == "normal"


def test_2_elevated_ml_signal(test_app_and_client):
    """
    Test 2 — Elevated ML signal:
    A prediction in 0.040 <= prob < 0.075 produces ELEVATED.
    Verify controller transitions to WATCHING and receives signal correctly.
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()

    elevated_result = _mock_ml_result(0.055, RiskSignal.ELEVATED)
    with patch.object(predictor, "predict", return_value=elevated_result):
        payload = {"metric": _metric_dict(traffic=200.0)}
        response = client.post("/monitor", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["risk_signal"] == "elevated"
    assert data["controller_state"] == "WATCHING"
    assert data["optimization_applied"] is False
    assert data["current_configuration"]["caching"] == CACHE_NORMAL


def test_3_critical_ml_signal_is_advisory_not_immediate(test_app_and_client):
    """
    Test 3 — Critical ML signal:
    A prediction >= 0.075 produces CRITICAL.
    Verify that 1 CRITICAL signal goes through safety controller and does NOT
    immediately trigger optimization (enters WATCHING with consecutive_high_risk=1).
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()

    crit_result = _mock_ml_result(0.090, RiskSignal.CRITICAL)
    with patch.object(predictor, "predict", return_value=crit_result):
        payload = {"metric": _metric_dict(traffic=500.0)}
        response = client.post("/monitor", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["risk_signal"] == "critical"
    # Controller must require confirmations — should be WATCHING, not OPTIMIZED yet
    assert data["controller_state"] == "WATCHING"
    assert data["consecutive_high_risk"] == 1
    assert data["optimization_applied"] is False
    assert data["current_configuration"]["caching"] == CACHE_NORMAL


def test_4_consecutive_critical_confirmation_required(test_app_and_client):
    """
    Test 4 — Consecutive critical confirmation:
    Verify exactly 3 consecutive critical predictions are required before
    optimization can occur:
    - 1st CRITICAL -> WATCHING (count=1, no optimization)
    - 2nd CRITICAL -> WATCHING (count=2, no optimization)
    - 3rd CRITICAL -> OPTIMIZED (count=0, optimization applied)
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()

    crit_result = _mock_ml_result(0.085, RiskSignal.CRITICAL)

    with patch.object(predictor, "predict", return_value=crit_result):
        # 1st CRITICAL
        r1 = client.post("/monitor", json={"metric": _metric_dict(traffic=300.0)})
        assert r1.json()["controller_state"] == "WATCHING"
        assert r1.json()["consecutive_high_risk"] == 1
        assert r1.json()["optimization_applied"] is False

        # 2nd CRITICAL
        r2 = client.post("/monitor", json={"metric": _metric_dict(traffic=350.0)})
        assert r2.json()["controller_state"] == "WATCHING"
        assert r2.json()["consecutive_high_risk"] == 2
        assert r2.json()["optimization_applied"] is False

        # 3rd CRITICAL -> Confirmed -> Optimization applied
        r3 = client.post("/monitor", json={"metric": _metric_dict(traffic=400.0)})
        assert r3.json()["controller_state"] == "OPTIMIZED"
        assert r3.json()["optimization_applied"] is True
        assert r3.json()["current_configuration"]["caching"] == CACHE_OPTIMIZED
        assert r3.json()["current_configuration"]["pagination_size"] == PAGINATION_OPTIMIZED
        assert r3.json()["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_OPTIMIZED


def test_5_stale_metrics_skip_ml_and_logging(test_app_and_client):
    """
    Test 5 — Stale metrics:
    Verify stale metrics -> no ML prediction -> no ML prediction persisted,
    and stale-metric safety behavior remains intact.
    """
    app, client, db_file = test_app_and_client

    # Metric timestamp 1 hour in the past (> 30s threshold)
    stale_ts = datetime.now(timezone.utc) - timedelta(hours=1)
    stale_metric = _metric_dict(timestamp=stale_ts, traffic=500.0)

    response = client.post("/monitor", json={"metric": stale_metric, "risk_signal": "critical"})
    assert response.status_code == 200
    data = response.json()

    assert data["stale"] is True
    assert data["optimization_applied"] is False
    assert data["ml_prediction"] is None
    assert "stale" in (data["optimization_blocked_reason"] or "").lower()

    # Verify no prediction record was written to the database
    predictions = get_recent_predictions(db_path=db_file)
    assert len(predictions) == 0


def test_6_cooldown_blocks_repeated_optimization(test_app_and_client):
    """
    Test 6 — Cooldown:
    Verify that ML cannot cause repeated optimization while cooldown is active.
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()
    crit_result = _mock_ml_result(0.090, RiskSignal.CRITICAL)

    with patch.object(predictor, "predict", return_value=crit_result):
        # Trigger optimization with 3 consecutive critical signals
        for _ in range(3):
            client.post("/monitor", json={"metric": _metric_dict()})

    state = client.get("/state").json()
    assert state["controller_state"] == "OPTIMIZED"
    assert state["cooldown_active"] is True

    # Send 3 more critical readings while in cooldown
    with patch.object(predictor, "predict", return_value=crit_result):
        res = client.post("/monitor", json={"metric": _metric_dict()})

    data = res.json()
    assert data["optimization_applied"] is False
    assert data["cooldown_active"] is True
    assert "cooldown" in (data["optimization_blocked_reason"] or "").lower()


def test_7_ml_prediction_logging(test_app_and_client):
    """
    Test 7 — ML logging:
    Verify that a successful ML prediction creates a database record containing:
    probability, risk_signal, thresholds, timestamp, feature snapshot.
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()

    custom_result = MLPredictionResult(
        surge_probability=0.082,
        raw_probability=0.065,
        risk_signal=RiskSignal.CRITICAL,
        watch_threshold=0.040,
        critical_threshold=0.075,
        feature_vector=[123.45] + [1.0] * 18,
        window_size=15,
        insufficient_data=False,
        inference_time_ms=0.45,
    )

    t0 = datetime.now(timezone.utc)
    with patch.object(predictor, "predict", return_value=custom_result):
        response = client.post("/monitor", json={"metric": _metric_dict(timestamp=t0)})

    assert response.status_code == 200

    # Query logged predictions from SQLite
    preds = get_recent_predictions(db_path=db_file)
    assert len(preds) >= 1
    latest = preds[0]

    assert abs(latest["probability"] - 0.082) < 1e-5
    assert latest["risk_signal"] == "critical"
    assert abs(latest["watch_threshold"] - 0.040) < 1e-5
    assert abs(latest["critical_threshold"] - 0.075) < 1e-5
    assert len(latest["feature_snapshot"]) == 19
    assert abs(latest["feature_snapshot"][0] - 123.45) < 1e-5


def test_8_recovery_path_preserved(test_app_and_client):
    """
    Test 8 — Recovery:
    Verify that the ML integration does not break the existing recovery path:
    After system is in OPTIMIZED, 3 consecutive NORMAL predictions restore
    default configuration and transition controller to NORMAL.
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()

    crit_result = _mock_ml_result(0.085, RiskSignal.CRITICAL)
    norm_result = _mock_ml_result(0.015, RiskSignal.NORMAL)

    # 1. Drive system into OPTIMIZED
    with patch.object(predictor, "predict", return_value=crit_result):
        for _ in range(3):
            client.post("/monitor", json={"metric": _metric_dict(traffic=400.0)})

    assert client.get("/state").json()["controller_state"] == "OPTIMIZED"

    # 2. Send 3 consecutive NORMAL predictions
    with patch.object(predictor, "predict", return_value=norm_result):
        # 1st normal -> RECOVERY (count 1)
        r1 = client.post("/monitor", json={"metric": _metric_dict(traffic=50.0)})
        assert r1.json()["controller_state"] == "RECOVERY"
        assert r1.json()["consecutive_normal"] == 1
        assert r1.json()["current_configuration"]["caching"] == CACHE_OPTIMIZED

        # 2nd normal -> RECOVERY (count 2)
        r2 = client.post("/monitor", json={"metric": _metric_dict(traffic=50.0)})
        assert r2.json()["controller_state"] == "RECOVERY"
        assert r2.json()["consecutive_normal"] == 2
        assert r2.json()["current_configuration"]["caching"] == CACHE_OPTIMIZED

        # 3rd normal -> Sustained normal confirmed -> Defaults restored -> NORMAL
        r3 = client.post("/monitor", json={"metric": _metric_dict(traffic=50.0)})
        assert r3.json()["controller_state"] == "NORMAL"
        assert r3.json()["consecutive_normal"] == 0
        assert r3.json()["current_configuration"]["caching"] == CACHE_NORMAL
        assert r3.json()["current_configuration"]["pagination_size"] == PAGINATION_NORMAL
        assert r3.json()["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_NORMAL


# =============================================================================
# PHASE 4 — Full Runtime Simulation Test
# =============================================================================


def test_full_runtime_simulation_lifecycle(test_app_and_client):
    """
    Phase 4: End-to-end runtime simulation test simulating the complete lifecycle:
        NORMAL
          ↓ (rising traffic)
        ELEVATED
          ↓ (surge spike)
        CRITICAL (1/3)
          ↓
        CRITICAL (2/3)
          ↓
        CRITICAL (3/3 confirmed)
          ↓
        OPTIMIZED (caching enabled, pagination=20, heavy_components=disabled)
          ↓ (metrics improve)
        RECOVERY (1/3)
          ↓
        RECOVERY (2/3)
          ↓
        RECOVERY (3/3 confirmed)
          ↓
        NORMAL (defaults restored)
    """
    app, client, db_file = test_app_and_client
    predictor = SurgePredictor.get_instance()

    norm_result = _mock_ml_result(0.018, RiskSignal.NORMAL)
    elev_result = _mock_ml_result(0.052, RiskSignal.ELEVATED)
    crit_result = _mock_ml_result(0.088, RiskSignal.CRITICAL)

    # Step 1: Steady normal state
    with patch.object(predictor, "predict", return_value=norm_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=50.0)})
    assert res.json()["controller_state"] == "NORMAL"
    assert res.json()["optimization_applied"] is False

    # Step 2: Traffic rises, ML detects early warning (ELEVATED)
    with patch.object(predictor, "predict", return_value=elev_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=180.0)})
    assert res.json()["controller_state"] == "WATCHING"
    assert res.json()["risk_signal"] == "elevated"
    assert res.json()["optimization_applied"] is False

    # Step 3: Surge spike occurs -> 1st CRITICAL
    with patch.object(predictor, "predict", return_value=crit_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=350.0)})
    assert res.json()["controller_state"] == "WATCHING"
    assert res.json()["consecutive_high_risk"] == 1
    assert res.json()["optimization_applied"] is False

    # Step 4: Surge continues -> 2nd CRITICAL
    with patch.object(predictor, "predict", return_value=crit_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=420.0)})
    assert res.json()["controller_state"] == "WATCHING"
    assert res.json()["consecutive_high_risk"] == 2
    assert res.json()["optimization_applied"] is False

    # Step 5: Surge sustained -> 3rd CRITICAL -> Optimization triggered!
    with patch.object(predictor, "predict", return_value=crit_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=490.0)})
    assert res.json()["controller_state"] == "OPTIMIZED"
    assert res.json()["optimization_applied"] is True
    assert res.json()["current_configuration"]["caching"] == CACHE_OPTIMIZED
    assert res.json()["current_configuration"]["pagination_size"] == PAGINATION_OPTIMIZED
    assert res.json()["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_OPTIMIZED

    # Step 6: System metrics improve -> 1st recovery tick
    with patch.object(predictor, "predict", return_value=norm_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=60.0)})
    assert res.json()["controller_state"] == "RECOVERY"
    assert res.json()["consecutive_normal"] == 1
    assert res.json()["current_configuration"]["caching"] == CACHE_OPTIMIZED

    # Step 7: System healthy -> 2nd recovery tick
    with patch.object(predictor, "predict", return_value=norm_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=55.0)})
    assert res.json()["controller_state"] == "RECOVERY"
    assert res.json()["consecutive_normal"] == 2
    assert res.json()["current_configuration"]["caching"] == CACHE_OPTIMIZED

    # Step 8: System sustained healthy -> 3rd recovery tick -> RESTORE DEFAULTS
    with patch.object(predictor, "predict", return_value=norm_result):
        res = client.post("/monitor", json={"metric": _metric_dict(traffic=50.0)})
    assert res.json()["controller_state"] == "NORMAL"
    assert res.json()["consecutive_normal"] == 0
    assert res.json()["current_configuration"]["caching"] == CACHE_NORMAL
    assert res.json()["current_configuration"]["pagination_size"] == PAGINATION_NORMAL
    assert res.json()["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_NORMAL

    # Step 9: Verify optimization events log recorded both apply and restore
    events = get_recent_events(db_path=db_file)
    assert len(events) >= 2
    actions = [e["action"] for e in events]
    assert "apply_optimizations" in actions
    assert "restore_defaults" in actions


def _post_simulation_tick(client, tick):
    metric = {
        **tick["metric"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    response = client.post("/monitor", json={"metric": metric})
    assert response.status_code == 200, response.text
    return response.json(), metric


def test_milestone5_gradual_surge_optimizes_via_real_ml_and_monitor(test_app_and_client):
    _, client, db_file = test_app_and_client
    reset = client.post("/simulation/reset")
    assert reset.status_code == 200

    scenario = client.get("/simulation/scenarios/gradual_surge").json()
    assert scenario["phases"] == ["warmup", "ramp", "surge", "recovery"]
    assert scenario["warmup_tick_count"] == 61
    ramp_ticks = [tick for tick in scenario["ticks"] if tick["phase"] == "ramp"]
    traffic = [tick["metric"]["traffic"] for tick in ramp_ticks]
    assert traffic == sorted(traffic)
    assert traffic[-1] > traffic[0]

    observed = []
    optimized_response = None
    for tick in scenario["ticks"]:
        result, metric = _post_simulation_tick(client, tick)
        observed.append((tick, result, metric))
        if result["optimization_applied"]:
            optimized_response = result
            next_tick = scenario["ticks"][tick["index"] + 1]
            after_response, _ = _post_simulation_tick(client, next_tick)
            observed.append((next_tick, after_response, next_tick["metric"]))
            break

    assert optimized_response is not None
    first_warmup_result = observed[0][1]
    assert first_warmup_result["controller_state"] == "NORMAL"
    assert first_warmup_result["optimization_applied"] is False
    non_warmup = [result for tick, result, _ in observed if tick["phase"] != "warmup"]
    assert any(result["risk_signal"] == "elevated" for result in non_warmup)
    critical_counts = [
        result["consecutive_high_risk"]
        for result in non_warmup
        if result["risk_signal"] == "critical"
    ]
    assert 1 in critical_counts
    assert 2 in critical_counts
    assert optimized_response["risk_signal"] == "critical"
    assert optimized_response["controller_state"] == "OPTIMIZED"
    assert optimized_response["optimization_applied"] is True
    assert optimized_response["current_configuration"]["caching"] == CACHE_OPTIMIZED
    assert optimized_response["current_configuration"]["pagination_size"] == PAGINATION_OPTIMIZED
    assert optimized_response["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_OPTIMIZED
    assert all(result["ml_prediction"] is not None for _, result, _ in observed)

    apply_event = next(
        event
        for event in get_recent_events(db_path=db_file)
        if event["action"] == "apply_optimizations"
    )
    assert apply_event["before_metrics"] is not None
    assert apply_event["after_metrics"] is not None
    assert apply_event["impact"] is not None
    assert apply_event["impact"]["response_time"] is not None
    optimization_index = next(
        i for i, (_, result, _) in enumerate(observed)
        if result["optimization_applied"]
    )
    before_metric = observed[optimization_index][2]
    after_metric = observed[optimization_index + 1][2]
    assert apply_event["before_metrics"]["traffic"] == before_metric["traffic"]
    assert apply_event["after_metrics"]["traffic"] == after_metric["traffic"]
    assert apply_event["before_metrics"]["response_time"] != apply_event["after_metrics"]["response_time"]
    assert apply_event["impact"]["response_time"] != 0


def test_milestone5_recovery_scenario_restores_defaults_via_monitor(test_app_and_client):
    _, client, db_file = test_app_and_client
    reset = client.post("/simulation/reset")
    assert reset.status_code == 200

    scenario = client.get("/simulation/scenarios/recovery").json()
    observed = []
    for tick in scenario["ticks"]:
        result, metric = _post_simulation_tick(client, tick)
        observed.append((tick, result, metric))

    recovery_ticks = [
        (tick, result, metric)
        for tick, result, metric in observed
        if tick["phase"] == "recovery"
    ]
    assert recovery_ticks
    assert all(metric["traffic"] < 600 for _, _, metric in recovery_ticks)

    applied_indices = [
        i for i, (_, result, _) in enumerate(observed)
        if result["optimization_applied"]
    ]
    assert applied_indices
    assert any(
        result["controller_state"] == "OPTIMIZED"
        and result["current_configuration"]["caching"] == CACHE_OPTIMIZED
        for _, result, _ in observed
    )
    optimized_index = applied_indices[0]
    recovery_observations = observed[optimized_index + 1:]
    recovery_states = [result["controller_state"] for _, result, _ in recovery_observations]
    assert "RECOVERY" in recovery_states
    assert "NORMAL" in recovery_states
    assert any(
        result["consecutive_normal"] in (1, 2)
        for _, result, _ in recovery_observations
        if result["controller_state"] == "RECOVERY"
    )

    final_result = recovery_observations[-1][1]
    assert final_result["controller_state"] == "NORMAL"
    assert final_result["current_configuration"]["caching"] == CACHE_NORMAL
    assert final_result["current_configuration"]["pagination_size"] == PAGINATION_NORMAL
    assert final_result["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_NORMAL

    actions = [event["action"] for event in get_recent_events(db_path=db_file)]
    assert "apply_optimizations" in actions
    assert "restore_defaults" in actions


def test_milestone5_manual_override_blocks_hard_safety_scenario(test_app_and_client):
    _, client, _ = test_app_and_client
    reset = client.post("/simulation/reset")
    assert reset.status_code == 200

    override = client.post("/override", json={"manual_override": True})
    assert override.status_code == 200
    assert override.json()["manual_override"] is True

    scenario = client.get("/simulation/scenarios/hard_safety").json()
    hard_tick = next(tick for tick in scenario["ticks"] if tick["phase"] == "hard_safety")
    result, metric = _post_simulation_tick(client, hard_tick)

    assert metric["cpu_utilization"] >= CPU_CRITICAL_THRESHOLD
    assert metric["memory_utilization"] >= MEMORY_CRITICAL_THRESHOLD
    assert result["ml_prediction"] is not None
    assert result["stale"] is False
    assert result["manual_override"] is True
    assert result["controller_state"] == "NORMAL"
    assert result["optimization_applied"] is False
    assert "manual override" in result["optimization_blocked_reason"].lower()
    assert result["current_configuration"]["caching"] == CACHE_NORMAL
    assert result["current_configuration"]["pagination_size"] == PAGINATION_NORMAL
    assert result["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_NORMAL
