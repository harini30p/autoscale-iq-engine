"""
Pydantic schemas and enums for AutoScale IQ.
Defines strict validation rules for system metrics, controller state, and API models.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class RiskSignal(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


class ControllerState(str, Enum):
    NORMAL = "NORMAL"
    WATCHING = "WATCHING"
    OPTIMIZED = "OPTIMIZED"
    RECOVERY = "RECOVERY"


def _check_finite_number(value: float, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{field_name} must be a finite number, got {value}")
    return value


class MetricSchema(BaseModel):
    timestamp: datetime = Field(..., description="UTC timestamp of the metric observation")
    traffic: float = Field(..., ge=0, description="Requests/invocations during interval (non-negative)")
    active_users: int = Field(..., ge=0, description="Active user count (non-negative integer)")
    cpu_utilization: float = Field(..., ge=0.0, le=100.0, description="CPU utilization percentage (0-100)")
    memory_utilization: float = Field(..., ge=0.0, le=100.0, description="Memory utilization percentage (0-100)")
    response_time: float = Field(..., ge=0.0, description="Response time in milliseconds (non-negative)")
    db_query_time: float = Field(..., ge=0.0, description="Database query time in milliseconds (non-negative)")
    system_load: float = Field(..., ge=0.0, description="System load indicator (non-negative)")

    @field_validator("timestamp")
    @classmethod
    def validate_utc_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            # Assume UTC if naive, attach timezone
            v = v.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC
            v = v.astimezone(timezone.utc)
        return v

    @field_validator("traffic", "cpu_utilization", "memory_utilization", "response_time", "db_query_time", "system_load")
    @classmethod
    def validate_finite_floats(cls, v: float, info) -> float:
        return _check_finite_number(v, info.field_name)

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2026-10-05T12:00:00Z",
                "traffic": 500.0,
                "active_users": 100,
                "cpu_utilization": 75.0,
                "memory_utilization": 70.0,
                "response_time": 250.0,
                "db_query_time": 80.0,
                "system_load": 1.5
            }
        }
    }


class SystemConfiguration(BaseModel):
    caching: str = Field(default="disabled", description="Caching status ('enabled' or 'disabled')")
    pagination_size: int = Field(default=50, description="Page size limit (e.g., 50 normal, 20 optimized)")
    heavy_components: str = Field(default="enabled", description="Heavy component status ('enabled' or 'disabled')")


class MetricRecordSchema(BaseModel):
    """Stored telemetry observation record returned from history endpoints."""
    id: Optional[int] = None
    timestamp: datetime
    traffic: float
    active_users: int
    cpu_utilization: float
    memory_utilization: float
    response_time: float
    db_query_time: float
    system_load: float


class MLPredictionRecordSchema(BaseModel):
    """Stored ML surge prediction record returned from history endpoints."""
    id: Optional[int] = None
    timestamp: datetime
    probability: float = Field(..., description="Calibrated surge probability")
    risk_signal: RiskSignal = Field(..., description="Classification from thresholds")
    watch_threshold: float
    critical_threshold: float
    feature_snapshot: Optional[List[float]] = None


class OptimizationEventSchema(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    action: str
    reason: str
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    # Optional metric snapshots — None for events created before this feature
    before_metrics: Optional["MetricSnapshotSchema"] = None
    after_metrics: Optional["MetricSnapshotSchema"] = None
    impact: Optional["ImpactMetrics"] = None


class MetricSnapshotSchema(BaseModel):
    """Lightweight metric snapshot captured at optimization trigger / first-post-optimization reading."""
    timestamp: Optional[datetime] = None
    traffic: Optional[float] = None
    response_time: Optional[float] = None
    db_query_time: Optional[float] = None
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    active_users: Optional[int] = None
    system_load: Optional[float] = None


class ImpactMetrics(BaseModel):
    """Percentage improvements (lower-is-better) between before and after metric snapshots.
    Positive value = improvement. Negative = degradation. None = not calculable.
    Formula: ((before - after) / before) * 100
    """
    response_time: Optional[float] = None
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    db_query_time: Optional[float] = None


# Rebuild OptimizationEventSchema forward refs
OptimizationEventSchema.model_rebuild()


class MonitorRequest(BaseModel):
    metric: MetricSchema
    risk_signal: RiskSignal = Field(default=RiskSignal.NORMAL, description="Generic risk signal from monitoring/ML layer")


class MLPredictionResponse(BaseModel):
    """
    Structured ML surge prediction result included in monitor responses.
    All probabilities are calibrated (isotonic regression).
    """
    surge_probability: float = Field(..., description="Calibrated surge probability [0, 1]")
    raw_probability: float = Field(..., description="Uncalibrated model output probability")
    risk_signal: RiskSignal = Field(..., description="Risk classification from ML thresholds")
    watch_threshold: float = Field(..., description="tau_watch threshold used")
    critical_threshold: float = Field(..., description="tau_crit threshold used")
    window_size: int = Field(..., description="Number of ticks in the rolling window")
    insufficient_data: bool = Field(..., description="True if window < 61 ticks")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")


class MonitorResponse(BaseModel):
    metric_valid: bool = True
    stale: bool
    metric_age_seconds: float
    risk_signal: RiskSignal
    controller_state: ControllerState
    optimization_applied: bool
    optimization_blocked_reason: Optional[str] = None
    current_configuration: SystemConfiguration
    manual_override: bool
    cooldown_active: bool
    consecutive_high_risk: int
    consecutive_normal: int
    ml_prediction: Optional[MLPredictionResponse] = Field(
        default=None,
        description="ML surge prediction result (None if ML unavailable)"
    )


class PredictRequest(BaseModel):
    """
    Direct surge prediction request (does not go through the safety controller).

    Only fields that are part of the locked 19-feature training schema are
    accepted. Resource/latency metrics (cpu, memory, response time, db query
    time) are NOT model inputs.
    """
    invocations: float = Field(..., ge=0, description="Current invocations/min")
    minute_of_day: int = Field(..., ge=0, le=1439, description="Minute of day (0–1439)")
    # Trigger one-hot — 7 categories matching the locked training schema
    trigger_timer: bool = Field(default=False, description="Timer trigger active")
    trigger_http: bool = Field(default=True, description="HTTP trigger active")
    trigger_queue: bool = Field(default=False, description="Queue trigger active")
    trigger_orchestration: bool = Field(default=False, description="Orchestration trigger active")
    trigger_event: bool = Field(default=False, description="Event trigger active")
    trigger_storage: bool = Field(default=False, description="Storage trigger active")
    trigger_others: bool = Field(default=False, description="Other trigger active")


class PredictResponse(BaseModel):
    """Response from the /predict endpoint."""
    ml_prediction: MLPredictionResponse
    model_name: str = "HistGradientBoostingClassifier 5M Config6"
    calibration: str = "Isotonic Regression"


class ValidateMetricResponse(BaseModel):
    valid: bool
    metric: MetricSchema
    message: str = "Metric is valid"


class StateResponse(BaseModel):
    controller_state: ControllerState
    current_configuration: SystemConfiguration
    manual_override: bool
    consecutive_high_risk: int
    consecutive_normal: int
    cooldown_active: bool
    cooldown_remaining_seconds: float
    last_optimization_timestamp: Optional[datetime] = None
    updated_at: datetime


class OverrideRequest(BaseModel):
    manual_override: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime
