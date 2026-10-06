"""
Tests for Optimization Before/After Metric Snapshots and Impact Calculation.

Verifies:
1. Optimization event stores before_metrics when optimization is triggered.
2. First subsequent valid metric populates after_metrics.
3. Improvement percentages are calculated correctly.
4. Zero/missing baseline values do not cause division errors.
5. Failed optimization does not create fake after_metrics.
6. Old optimization events without metric snapshots still deserialize correctly.
7. Recovery behavior remains unchanged (restore_defaults path).
8. Idempotency: after_metrics is only set once.
"""

import json
from datetime import datetime, timezone, timedelta
import pytest

from backend.database import (
    init_db,
    record_event,
    get_recent_events,
    attach_before_metrics_to_latest_event,
    attach_after_metrics_to_latest_event,
    _compute_impact,
)
from backend.main import create_app
from backend.schemas import RiskSignal, ControllerState
from fastapi.testclient import TestClient


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    db_file = str(tmp_path / "impact_test.db")
    init_db(db_file)
    return db_file


@pytest.fixture
def client(tmp_path):
    db_file = str(tmp_path / "impact_api.db")
    init_db(db_file)
    app = create_app(db_path=db_file, load_ml=False)
    with TestClient(app) as tc:
        yield tc


def fresh_metric(cpu=45.0, response_time=120.0, memory=50.0, db_qt=30.0, traffic=200.0):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traffic": traffic,
        "active_users": 40,
        "cpu_utilization": cpu,
        "memory_utilization": memory,
        "response_time": response_time,
        "db_query_time": db_qt,
        "system_load": 1.0,
    }


# ── _compute_impact unit tests ────────────────────────────────────────────────

def test_compute_impact_positive_improvement():
    before = {"response_time": 240.0, "cpu_utilization": 72.0, "memory_utilization": 65.0, "db_query_time": 45.0}
    after  = {"response_time":  85.0, "cpu_utilization": 28.0, "memory_utilization": 35.0, "db_query_time": 18.0}
    impact = _compute_impact(before, after)
    assert impact is not None
    # response_time: ((240 - 85) / 240) * 100 ≈ 64.58
    assert abs(impact["response_time"] - 64.58) < 0.1
    # cpu: ((72 - 28) / 72) * 100 ≈ 61.11
    assert abs(impact["cpu_utilization"] - 61.11) < 0.1


def test_compute_impact_degradation_is_negative():
    before = {"response_time": 100.0, "cpu_utilization": 30.0, "memory_utilization": 40.0, "db_query_time": 20.0}
    after  = {"response_time": 200.0, "cpu_utilization": 60.0, "memory_utilization": 80.0, "db_query_time": 40.0}
    impact = _compute_impact(before, after)
    assert impact["response_time"] < 0.0
    assert impact["cpu_utilization"] < 0.0


def test_compute_impact_zero_baseline_returns_none():
    """Division by zero must be handled gracefully — returns None for that field."""
    before = {"response_time": 0.0, "cpu_utilization": 50.0, "memory_utilization": 50.0, "db_query_time": 0.0}
    after  = {"response_time": 10.0, "cpu_utilization": 25.0, "memory_utilization": 25.0, "db_query_time":  5.0}
    impact = _compute_impact(before, after)
    assert impact["response_time"] is None   # was 0 → no div
    assert impact["db_query_time"] is None   # was 0 → no div
    # Non-zero baselines still computed
    assert impact["cpu_utilization"] is not None


def test_compute_impact_missing_before_returns_none():
    impact = _compute_impact(None, {"response_time": 80.0, "cpu_utilization": 25.0})
    assert impact is None


def test_compute_impact_missing_after_returns_none():
    impact = _compute_impact({"response_time": 200.0, "cpu_utilization": 60.0}, None)
    assert impact is None


# ── Database persistence tests ────────────────────────────────────────────────

def test_old_events_without_snapshots_deserialize_cleanly(db):
    """Legacy events (no metric columns) must still deserialize with before/after_metrics=None."""
    # Insert event directly without snapshot fields (simulating old schema behaviour)
    import sqlite3, json as js
    with sqlite3.connect(db) as conn:
        conn.execute("""
            INSERT INTO optimization_events
                (timestamp, action, reason, previous_state, new_state, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            "apply_optimizations",
            "legacy event",
            js.dumps({"caching": "disabled"}),
            js.dumps({"caching": "enabled"}),
            1,
            None,
        ))
        conn.commit()

    events = get_recent_events(db_path=db)
    assert len(events) == 1
    evt = events[0]
    assert evt["before_metrics"] is None
    assert evt["after_metrics"] is None
    assert evt["impact"] is None


def test_attach_before_metrics(db):
    """attach_before_metrics_to_latest_event stores snapshot correctly."""
    record_event(
        action="apply_optimizations",
        reason="3 consecutive criticals",
        previous_state={"caching": "disabled"},
        new_state={"caching": "enabled"},
        success=True,
        db_path=db,
    )
    before_snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traffic": 550.0,
        "response_time": 240.0,
        "db_query_time": 45.0,
        "cpu_utilization": 72.0,
        "memory_utilization": 65.0,
        "active_users": 120,
        "system_load": 1.8,
    }
    result = attach_before_metrics_to_latest_event(before_snap, db_path=db)
    assert result is True

    events = get_recent_events(db_path=db)
    assert events[0]["before_metrics"] is not None
    assert abs(events[0]["before_metrics"]["response_time"] - 240.0) < 0.01
    assert events[0]["after_metrics"] is None


def test_attach_after_metrics_and_impact(db):
    """attach_after_metrics stores snapshot and impact is computed."""
    record_event(
        action="apply_optimizations",
        reason="test",
        previous_state={"caching": "disabled"},
        new_state={"caching": "enabled"},
        success=True,
        db_path=db,
    )
    before_snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traffic": 500.0,
        "response_time": 200.0,
        "db_query_time": 40.0,
        "cpu_utilization": 80.0,
        "memory_utilization": 70.0,
        "active_users": 100,
        "system_load": 1.5,
    }
    after_snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traffic": 490.0,
        "response_time": 80.0,
        "db_query_time": 20.0,
        "cpu_utilization": 30.0,
        "memory_utilization": 35.0,
        "active_users": 95,
        "system_load": 0.8,
    }
    attach_before_metrics_to_latest_event(before_snap, db_path=db)
    attach_after_metrics_to_latest_event(after_snap, db_path=db)

    events = get_recent_events(db_path=db)
    evt = events[0]
    assert evt["before_metrics"]["response_time"] == 200.0
    assert evt["after_metrics"]["response_time"] == 80.0
    # impact: ((200 - 80) / 200) * 100 = 60.0
    assert evt["impact"] is not None
    assert abs(evt["impact"]["response_time"] - 60.0) < 0.01


def test_after_metrics_idempotent(db):
    """after_metrics is set only once — subsequent calls are no-ops."""
    record_event(
        action="apply_optimizations",
        reason="test",
        previous_state={"caching": "disabled"},
        new_state={"caching": "enabled"},
        success=True,
        db_path=db,
    )
    before_snap = {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "response_time": 200.0, "cpu_utilization": 80.0,
                   "memory_utilization": 70.0, "db_query_time": 40.0,
                   "traffic": 500.0, "active_users": 100, "system_load": 1.5}
    after_snap1 = {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "response_time": 80.0, "cpu_utilization": 30.0,
                   "memory_utilization": 35.0, "db_query_time": 20.0,
                   "traffic": 490.0, "active_users": 90, "system_load": 0.8}
    after_snap2 = {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "response_time": 999.0, "cpu_utilization": 90.0,
                   "memory_utilization": 80.0, "db_query_time": 90.0,
                   "traffic": 100.0, "active_users": 10, "system_load": 0.1}

    attach_before_metrics_to_latest_event(before_snap, db_path=db)
    r1 = attach_after_metrics_to_latest_event(after_snap1, db_path=db)
    r2 = attach_after_metrics_to_latest_event(after_snap2, db_path=db)  # should be no-op

    assert r1 is True
    assert r2 is False  # no matching event (after already set)

    events = get_recent_events(db_path=db)
    # after_metrics preserved from first write
    assert abs(events[0]["after_metrics"]["response_time"] - 80.0) < 0.01


def test_failed_optimization_no_after_metrics(db):
    """Failed optimization events should never have after_metrics populated."""
    record_event(
        action="apply_optimizations",
        reason="test failure",
        previous_state={"caching": "disabled"},
        new_state={"caching": "disabled"},
        success=False,
        error_message="Disk write failed",
        db_path=db,
    )
    # attach_after_metrics only targets events with before_metrics set; failed ones won't have it
    after_snap = {"response_time": 80.0, "cpu_utilization": 30.0,
                  "memory_utilization": 35.0, "db_query_time": 20.0,
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "traffic": 400.0, "active_users": 80, "system_load": 0.7}
    result = attach_after_metrics_to_latest_event(after_snap, db_path=db)
    assert result is False  # no event with before_metrics set


# ── Integration tests via API ─────────────────────────────────────────────────

def test_api_events_include_before_metrics_after_optimization(client):
    """Triggering optimization via /monitor → /events should return before_metrics."""
    severe_metric = {
        "metric": fresh_metric(cpu=95.0, response_time=300.0),
        "risk_signal": "normal"
    }
    response = client.post("/monitor", json=severe_metric)
    assert response.status_code == 200
    assert response.json()["optimization_applied"] is True

    events = client.get("/events?limit=5").json()
    apply_events = [e for e in events if e["action"] == "apply_optimizations"]
    assert len(apply_events) >= 1
    evt = apply_events[0]
    assert evt["before_metrics"] is not None
    assert abs(evt["before_metrics"]["cpu_utilization"] - 95.0) < 0.01
    assert abs(evt["before_metrics"]["response_time"] - 300.0) < 0.01


def test_api_events_include_after_metrics_on_second_optimized_metric(client):
    """After optimization, next /monitor in OPTIMIZED state sets after_metrics."""
    # Trigger optimization (hard threshold: cpu >= 90)
    r1 = client.post("/monitor", json={
        "metric": fresh_metric(cpu=95.0, response_time=300.0),
        "risk_signal": "normal"
    })
    assert r1.json()["optimization_applied"] is True

    # Send a second metric in OPTIMIZED state
    r2 = client.post("/monitor", json={
        "metric": fresh_metric(cpu=30.0, response_time=80.0),
        "risk_signal": "normal"
    })
    assert r2.status_code == 200
    assert r2.json()["controller_state"] in ("OPTIMIZED", "RECOVERY", "NORMAL")

    events = client.get("/events?limit=5").json()
    apply_events = [e for e in events if e["action"] == "apply_optimizations"]
    assert len(apply_events) >= 1
    evt = apply_events[0]
    assert evt["after_metrics"] is not None
    # Impact should now be computed
    assert evt["impact"] is not None
    assert evt["impact"]["cpu_utilization"] is not None


def test_api_events_backward_compat_no_snapshot_fields(client):
    """/events returns existing shape for events without snapshots (recovery events, etc.)."""
    # Trigger full cycle: opt → recovery
    client.post("/monitor", json={"metric": fresh_metric(cpu=95.0), "risk_signal": "normal"})
    events = client.get("/events?limit=20").json()
    for evt in events:
        # All events must have these keys (may be None)
        assert "before_metrics" in evt
        assert "after_metrics" in evt
        assert "impact" in evt
        assert "action" in evt
        assert "success" in evt


def test_api_events_recovery_event_has_no_impact(client):
    """restore_defaults events should not have before/after_metrics."""
    # Trigger optimization
    client.post("/monitor", json={"metric": fresh_metric(cpu=95.0), "risk_signal": "normal"})
    # Send 3 normal readings to complete recovery
    for _ in range(3):
        client.post("/monitor", json={"metric": fresh_metric(cpu=30.0), "risk_signal": "normal"})

    events = client.get("/events?limit=20").json()
    recovery_events = [e for e in events if e["action"] == "restore_defaults"]
    # If recovery completed, there should be a restore_defaults event
    if recovery_events:
        re = recovery_events[0]
        assert re["before_metrics"] is None
        assert re["after_metrics"] is None
        assert re["impact"] is None
