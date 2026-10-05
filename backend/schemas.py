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


class OptimizationEventSchema(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    action: str
    reason: str
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class MonitorRequest(BaseModel):
    metric: MetricSchema
    risk_signal: RiskSignal = Field(default=RiskSignal.NORMAL, description="Generic risk signal from monitoring/ML layer")


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
