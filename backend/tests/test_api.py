"""
Tests for AutoScale IQ FastAPI Endpoints.
Verifies /health, /metrics/validate, /monitor, /state, /events, /override, and error handling.
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
    app = create_app(db_path=db_file)
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
