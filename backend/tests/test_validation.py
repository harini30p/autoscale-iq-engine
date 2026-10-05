"""
Tests for metric schema validation in AutoScale IQ.
Verifies rejection of invalid inputs, negative numbers, bounds, NaN, Infinity, and malformed timestamps.
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from backend.schemas import MetricSchema


def valid_metric_payload():
    return {
        "timestamp": datetime.now(timezone.utc),
        "traffic": 250.0,
        "active_users": 50,
        "cpu_utilization": 45.5,
        "memory_utilization": 60.0,
        "response_time": 120.0,
        "db_query_time": 30.0,
        "system_load": 1.2,
    }


def test_valid_metric():
    data = valid_metric_payload()
    metric = MetricSchema(**data)
    assert metric.traffic == 250.0
    assert metric.active_users == 50
    assert metric.cpu_utilization == 45.5
    assert metric.timestamp.tzinfo is not None


def test_missing_required_fields():
    for required_field in [
        "timestamp", "traffic", "active_users", "cpu_utilization",
        "memory_utilization", "response_time", "db_query_time", "system_load"
    ]:
        data = valid_metric_payload()
        del data[required_field]
        with pytest.raises(ValidationError):
            MetricSchema(**data)


def test_negative_values():
    negative_test_cases = [
        ("traffic", -1.0),
        ("active_users", -5),
        ("cpu_utilization", -0.1),
        ("memory_utilization", -10.0),
        ("response_time", -50.0),
        ("db_query_time", -1.0),
        ("system_load", -0.5),
    ]
    for field, bad_val in negative_test_cases:
        data = valid_metric_payload()
        data[field] = bad_val
        with pytest.raises(ValidationError):
            MetricSchema(**data)


def test_cpu_utilization_upper_bound():
    data = valid_metric_payload()
    data["cpu_utilization"] = 100.1
    with pytest.raises(ValidationError):
        MetricSchema(**data)

    # 100.0 must be valid
    data["cpu_utilization"] = 100.0
    metric = MetricSchema(**data)
    assert metric.cpu_utilization == 100.0


def test_memory_utilization_upper_bound():
    data = valid_metric_payload()
    data["memory_utilization"] = 100.5
    with pytest.raises(ValidationError):
        MetricSchema(**data)

    # 100.0 must be valid
    data["memory_utilization"] = 100.0
    metric = MetricSchema(**data)
    assert metric.memory_utilization == 100.0


def test_nan_rejection():
    for field in ["traffic", "cpu_utilization", "memory_utilization", "response_time", "db_query_time", "system_load"]:
        data = valid_metric_payload()
        data[field] = float("nan")
        with pytest.raises(ValidationError):
            MetricSchema(**data)


def test_infinity_rejection():
    for field in ["traffic", "cpu_utilization", "memory_utilization", "response_time", "db_query_time", "system_load"]:
        data = valid_metric_payload()
        data[field] = float("inf")
        with pytest.raises(ValidationError):
            MetricSchema(**data)

        data[field] = float("-inf")
        with pytest.raises(ValidationError):
            MetricSchema(**data)


def test_invalid_timestamp_string():
    data = valid_metric_payload()
    data["timestamp"] = "not-a-valid-timestamp"
    with pytest.raises(ValidationError):
        MetricSchema(**data)


def test_naive_timestamp_converted_to_utc():
    naive_dt = datetime(2026, 10, 5, 14, 30, 0)
    data = valid_metric_payload()
    data["timestamp"] = naive_dt
    metric = MetricSchema(**data)
    assert metric.timestamp.tzinfo is not None
    assert metric.timestamp.tzinfo == timezone.utc
