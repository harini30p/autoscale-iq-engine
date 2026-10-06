"""
Tests for AutoScale IQ FastAPI Endpoints.
Verifies /health, /metrics/validate, /monitor, /state, /events, /override,
/predict, and error handling.
"""

from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.database import init_db, get_recent_metrics, get_recent_events
from backend.schemas import RiskSignal, ControllerState
from backend.config import CACHE_NORMAL, CACHE_OPTIMIZED


@pytest.fixture
def client(tmp_path):
    db_file = str(tmp_path / "test_api.db")
    init_db(db_file)
    # load_ml=False to avoid requiring ML artifacts in unit tests
    app = create_app(db_path=db_file, load_ml=False)
    with TestClient(app) as test_client:
        yield test_client


def sample_metric_payload(timestamp: datetime = None, cpu: float = 45.0, response_time: float = 120.0):
    ts = timestamp or datetime.now(timezone.utc)
    return {
        "timestamp": ts.isoformat(),
        "traffic": 250.0,
        "active_users": 50,
        "cpu_utilization": cpu,
        "memory_utilization": 60.0,
        "response_time": response_time,
        "db_query_time": 30.0,
        "system_load": 1.2
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_validate_metric_valid(client):
    payload = sample_metric_payload()
    response = client.post("/metrics/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["metric"]["traffic"] == 250.0


def test_validate_metric_invalid_negative(client):
    payload = sample_metric_payload()
    payload["traffic"] = -10.0
    response = client.post("/metrics/validate", json=payload)
    assert response.status_code == 422


def test_validate_metric_invalid_cpu_bounds(client):
    payload = sample_metric_payload()
    payload["cpu_utilization"] = 150.0
    response = client.post("/metrics/validate", json=payload)
    assert response.status_code == 422


def test_monitor_fresh_metric_recorded(client):
    payload = {
        "metric": sample_metric_payload(),
        "risk_signal": RiskSignal.NORMAL.value
    }
    response = client.post("/monitor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metric_valid"] is True
    assert data["stale"] is False
    assert data["controller_state"] == ControllerState.NORMAL.value
    assert data["optimization_applied"] is False
    assert data["current_configuration"]["caching"] == CACHE_NORMAL


def test_monitor_stale_metric_recorded_without_optimization(client):
    # Stale metric from 30 seconds ago
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=30)
    payload = {
        "metric": sample_metric_payload(timestamp=stale_ts, cpu=99.0),
        "risk_signal": RiskSignal.CRITICAL.value
    }
    response = client.post("/monitor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["stale"] is True
    assert data["optimization_applied"] is False
    assert "stale" in data["optimization_blocked_reason"].lower()


def test_monitor_3_critical_triggers_optimization(client):
    for i in range(2):
        res = client.post("/monitor", json={
            "metric": sample_metric_payload(),
            "risk_signal": RiskSignal.CRITICAL.value
        })
        assert res.status_code == 200
        assert res.json()["controller_state"] == ControllerState.WATCHING.value
        assert res.json()["optimization_applied"] is False

    # 3rd reading
    res3 = client.post("/monitor", json={
        "metric": sample_metric_payload(),
        "risk_signal": RiskSignal.CRITICAL.value
    })
    assert res3.status_code == 200
    data = res3.json()
    assert data["controller_state"] == ControllerState.OPTIMIZED.value
    assert data["optimization_applied"] is True
    assert data["current_configuration"]["caching"] == CACHE_OPTIMIZED


def test_get_state(client):
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert "controller_state" in data
    assert "current_configuration" in data
    assert "manual_override" in data
    assert "cooldown_active" in data


def test_get_events(client):
    # Trigger an optimization to generate an event
    severe_metric = sample_metric_payload(cpu=98.0)
    client.post("/monitor", json={
        "metric": severe_metric,
        "risk_signal": RiskSignal.CRITICAL.value
    })

    response = client.get("/events?limit=10")
    assert response.status_code == 200
    events = response.json()
    assert len(events) >= 1
    assert events[0]["action"] == "apply_optimizations"
    assert events[0]["success"] is True


def test_override_endpoint(client):
    # Enable manual override
    res1 = client.post("/override", json={"manual_override": True})
    assert res1.status_code == 200
    assert res1.json()["manual_override"] is True

    # Check state reflects override
    state_res = client.get("/state")
    assert state_res.json()["manual_override"] is True

    # Verify automation blocked when override is active
    severe_metric = sample_metric_payload(cpu=98.0)
    mon_res = client.post("/monitor", json={
        "metric": severe_metric,
        "risk_signal": RiskSignal.CRITICAL.value
    })
    assert mon_res.json()["optimization_applied"] is False
    assert "override" in mon_res.json()["optimization_blocked_reason"].lower()

    # Disable manual override
    res2 = client.post("/override", json={"manual_override": False})
    assert res2.status_code == 200
    assert res2.json()["manual_override"] is False


def test_monitor_response_includes_ml_prediction_none_when_ml_disabled(client):
    """When load_ml=False, ml_prediction should be None."""
    payload = {
        "metric": sample_metric_payload(),
        "risk_signal": RiskSignal.NORMAL.value
    }
    response = client.post("/monitor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ml_prediction"] is None


def test_predict_endpoint_returns_503_when_ml_disabled(client):
    """When load_ml=False, /predict should return 503."""
    payload = {
        "invocations": 10.0,
        "minute_of_day": 120,
        "trigger_http": True,
        "trigger_timer": False,
        "trigger_queue": False,
        "trigger_orchestration": False,
        "trigger_event": False,
        "trigger_storage": False,
        "trigger_others": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


# =============================================================================
# History endpoints tests (/metrics/history and /predictions/history)
# =============================================================================


def test_metrics_history_empty(client):
    """Empty database returns empty list rather than error."""
    response = client.get("/metrics/history")
    assert response.status_code == 200
    assert response.json() == []


def test_metrics_history_returns_data_and_respects_limit(client):
    """Verify stored telemetry is returned and respects limit parameter."""
    db_path = client.app.state.db_path
    base_time = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    # Ingest 5 metrics via /monitor
    for i in range(5):
        payload = {
            "metric": sample_metric_payload(
                timestamp=base_time + timedelta(minutes=i),
                cpu=40.0 + i,
                response_time=100.0 + i * 10
            ),
            "risk_signal": "normal"
        }
        res = client.post("/monitor", json=payload)
        assert res.status_code == 200

    # Query with default limit (60)
    res_all = client.get("/metrics/history")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert len(data_all) == 5

    # Query with limit=3
    res_limit = client.get("/metrics/history?limit=3")
    assert res_limit.status_code == 200
    data_limit = res_limit.json()
    assert len(data_limit) == 3


def test_metrics_history_chronological_ordering(client):
    """Verify returned records are in chronological order (oldest to newest)."""
    base_time = datetime(2026, 10, 6, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(4):
        metric = sample_metric_payload(
            timestamp=base_time + timedelta(minutes=i * 2),
            cpu=30.0 + i * 5,
        )
        metric["traffic"] = 100.0 + i * 50
        client.post("/monitor", json={"metric": metric, "risk_signal": "normal"})

    res = client.get("/metrics/history?limit=4")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 4

    # Verify chronological ascending sequence
    traffic_seq = [item["traffic"] for item in items]
    assert traffic_seq == [100.0, 150.0, 200.0, 250.0]


def test_predictions_history_empty(client):
    """Empty ml_predictions table returns empty list."""
    response = client.get("/predictions/history")
    assert response.status_code == 200
    assert response.json() == []


def test_predictions_history_returns_data_and_respects_limit(client):
    """Verify stored predictions are returned and respect limit."""
    from backend.database import insert_ml_prediction

    db_path = client.app.state.db_path
    base_time = datetime(2026, 10, 6, 14, 0, 0, tzinfo=timezone.utc)

    for i in range(5):
        insert_ml_prediction(
            probability=0.02 + i * 0.015,
            risk_signal="normal" if i < 2 else "elevated",
            watch_threshold=0.040,
            critical_threshold=0.075,
            feature_snapshot=[10.0 + i] + [0.0] * 18,
            timestamp=base_time + timedelta(minutes=i),
            db_path=db_path
        )

    # Fetch with limit=3
    res = client.get("/predictions/history?limit=3")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 3


def test_predictions_history_fields_and_chronological_order(client):
    """Verify prediction fields (probability, risk signal, thresholds) and chronological ordering."""
    from backend.database import insert_ml_prediction

    db_path = client.app.state.db_path
    t0 = datetime(2026, 10, 6, 15, 0, 0, tzinfo=timezone.utc)

    insert_ml_prediction(
        probability=0.032,
        risk_signal="normal",
        watch_threshold=0.040,
        critical_threshold=0.075,
        feature_snapshot=[50.0] + [0.0] * 18,
        timestamp=t0,
        db_path=db_path
    )
    insert_ml_prediction(
        probability=0.055,
        risk_signal="elevated",
        watch_threshold=0.040,
        critical_threshold=0.075,
        feature_snapshot=[150.0] + [0.0] * 18,
        timestamp=t0 + timedelta(minutes=1),
        db_path=db_path
    )
    insert_ml_prediction(
        probability=0.088,
        risk_signal="critical",
        watch_threshold=0.040,
        critical_threshold=0.075,
        feature_snapshot=[450.0] + [0.0] * 18,
        timestamp=t0 + timedelta(minutes=2),
        db_path=db_path
    )

    res = client.get("/predictions/history?limit=10")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 3

    # Check chronological order
    assert abs(items[0]["probability"] - 0.032) < 1e-4
    assert items[0]["risk_signal"] == "normal"
    assert items[0]["watch_threshold"] == 0.040

    assert abs(items[1]["probability"] - 0.055) < 1e-4
    assert items[1]["risk_signal"] == "elevated"

    assert abs(items[2]["probability"] - 0.088) < 1e-4
    assert items[2]["risk_signal"] == "critical"
    assert items[2]["critical_threshold"] == 0.075


def test_history_endpoints_bound_validation(client):
    """Verify limit boundaries (1 <= limit <= 500) return 422 for invalid bounds."""
    # Under lower bound (< 1)
    res_low_metric = client.get("/metrics/history?limit=0")
    assert res_low_metric.status_code == 422

    res_low_pred = client.get("/predictions/history?limit=0")
    assert res_low_pred.status_code == 422

    # Over upper bound (> 500)
    res_high_metric = client.get("/metrics/history?limit=501")
    assert res_high_metric.status_code == 422

    res_high_pred = client.get("/predictions/history?limit=501")
    assert res_high_pred.status_code == 422

