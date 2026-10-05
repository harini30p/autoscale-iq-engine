"""
Monitoring and freshness validation logic for AutoScale IQ.
Handles stale metric detection and hard safety threshold evaluation.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple
from backend.config import (
    DEFAULT_STALE_THRESHOLD_SECONDS,
    CPU_CRITICAL_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    RESPONSE_TIME_CRITICAL_THRESHOLD,
    SYSTEM_LOAD_CRITICAL_THRESHOLD,
)
from backend.schemas import MetricSchema


def assess_freshness(
    metric: MetricSchema,
    threshold_seconds: float = DEFAULT_STALE_THRESHOLD_SECONDS,
    now: Optional[datetime] = None
) -> Tuple[bool, float]:
    """
    Evaluate whether an incoming metric is fresh or stale.
    Returns a tuple (is_stale: bool, age_seconds: float).
    
    A metric is considered stale if its timestamp is older than threshold_seconds
    relative to current UTC time.
    """
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    # Ensure metric timestamp is UTC aware
    metric_ts = metric.timestamp
    if metric_ts.tzinfo is None:
        metric_ts = metric_ts.replace(tzinfo=timezone.utc)

    age_seconds = (current_time - metric_ts).total_seconds()

    # If the metric is older than threshold_seconds or absurdly far in the future
    is_stale = age_seconds > threshold_seconds or age_seconds < -threshold_seconds
    return is_stale, max(0.0, age_seconds)


def evaluate_hard_safety_thresholds(metric: MetricSchema) -> Tuple[bool, List[str]]:
    """
    Check if any hard application safety thresholds are breached.
    Note: These are deterministic safety thresholds, not ML predictions.
    
    Returns (is_breached: bool, reasons: List[str]).
    """
    breaches = []

    if metric.cpu_utilization >= CPU_CRITICAL_THRESHOLD:
        breaches.append(
            f"CPU utilization ({metric.cpu_utilization:.1f}%) reached critical threshold ({CPU_CRITICAL_THRESHOLD:.1f}%)"
        )

    if metric.memory_utilization >= MEMORY_CRITICAL_THRESHOLD:
        breaches.append(
            f"Memory utilization ({metric.memory_utilization:.1f}%) reached critical threshold ({MEMORY_CRITICAL_THRESHOLD:.1f}%)"
        )

    if metric.response_time >= RESPONSE_TIME_CRITICAL_THRESHOLD:
        breaches.append(
            f"Response time ({metric.response_time:.1f}ms) exceeded critical threshold ({RESPONSE_TIME_CRITICAL_THRESHOLD:.1f}ms)"
        )

    if metric.system_load >= SYSTEM_LOAD_CRITICAL_THRESHOLD:
        breaches.append(
            f"System load ({metric.system_load:.2f}) exceeded critical threshold ({SYSTEM_LOAD_CRITICAL_THRESHOLD:.2f})"
        )

    return len(breaches) > 0, breaches
