"""
Tests for monitoring freshness evaluation and hard safety thresholds.
"""

from datetime import datetime, timezone, timedelta
from backend.monitoring import assess_freshness, evaluate_hard_safety_thresholds
from backend.schemas import MetricSchema


def create_metric(timestamp: datetime, cpu: float = 40.0, memory: float = 50.0, response_time: float = 100.0, load: float = 1.0) -> MetricSchema:
    return MetricSchema(
        timestamp=timestamp,
        traffic=100.0,
        active_users=20,
        cpu_utilization=cpu,
        memory_utilization=memory,
        response_time=response_time,
        db_query_time=20.0,
        system_load=load
    )


def test_fresh_metric():
    now = datetime.now(timezone.utc)
    metric_time = now - timedelta(seconds=2)
    metric = create_metric(metric_time)

    is_stale, age = assess_freshness(metric, threshold_seconds=10.0, now=now)
    assert not is_stale
    assert 1.9 <= age <= 2.1


def test_stale_metric():
    now = datetime.now(timezone.utc)
    metric_time = now - timedelta(seconds=15)
    metric = create_metric(metric_time)

    is_stale, age = assess_freshness(metric, threshold_seconds=10.0, now=now)
    assert is_stale
    assert 14.9 <= age <= 15.1


def test_configurable_stale_threshold():
    now = datetime.now(timezone.utc)
    metric_time = now - timedelta(seconds=5)
    metric = create_metric(metric_time)

    # Stale when threshold is 3s
    is_stale, _ = assess_freshness(metric, threshold_seconds=3.0, now=now)
    assert is_stale

    # Fresh when threshold is 10s
    is_stale, _ = assess_freshness(metric, threshold_seconds=10.0, now=now)
    assert not is_stale


def test_far_future_timestamp_detected_as_stale():
    now = datetime.now(timezone.utc)
    metric_time = now + timedelta(seconds=30)
    metric = create_metric(metric_time)

    is_stale, _ = assess_freshness(metric, threshold_seconds=10.0, now=now)
    assert is_stale


def test_hard_safety_threshold_cpu():
    metric = create_metric(datetime.now(timezone.utc), cpu=92.0)
    is_breached, reasons = evaluate_hard_safety_thresholds(metric)
    assert is_breached
    assert any("CPU" in r for r in reasons)


def test_hard_safety_threshold_memory():
    metric = create_metric(datetime.now(timezone.utc), memory=95.0)
    is_breached, reasons = evaluate_hard_safety_thresholds(metric)
    assert is_breached
    assert any("Memory" in r for r in reasons)


def test_hard_safety_threshold_response_time():
    metric = create_metric(datetime.now(timezone.utc), response_time=1200.0)
    is_breached, reasons = evaluate_hard_safety_thresholds(metric)
    assert is_breached
    assert any("Response time" in r for r in reasons)


def test_hard_safety_threshold_system_load():
    metric = create_metric(datetime.now(timezone.utc), load=6.5)
    is_breached, reasons = evaluate_hard_safety_thresholds(metric)
    assert is_breached
    assert any("System load" in r for r in reasons)


def test_hard_safety_threshold_normal_conditions():
    metric = create_metric(datetime.now(timezone.utc), cpu=50.0, memory=50.0, response_time=200.0, load=1.5)
    is_breached, reasons = evaluate_hard_safety_thresholds(metric)
    assert not is_breached
    assert len(reasons) == 0
