"""
Interactive Simulation Engine — tick generators for AutoScale IQ (Milestone 4).

This module produces named sequences of metric ticks compatible with MetricSchema.
It does NOT run ML inference, step the safety controller, or apply optimizations.
Playback posts each tick to the existing POST /monitor endpoint.

Timestamps are intentionally omitted. The playback layer must stamp each tick
with the current UTC time at send time so freshness checks succeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from backend.config import (
    CPU_CRITICAL_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
)


# Matches backend.ml_engine._MIN_WINDOW_SIZE (61 prior ticks + current = full window)
WARMUP_TICK_COUNT = 61

SCENARIO_BASELINE = "baseline"
SCENARIO_GRADUAL_SURGE = "gradual_surge"
SCENARIO_SUDDEN_SPIKE = "sudden_spike"
SCENARIO_HARD_SAFETY = "hard_safety"
SCENARIO_RECOVERY = "recovery"

PHASE_WARMUP = "warmup"
PHASE_BASELINE = "baseline"
PHASE_RAMP = "ramp"
PHASE_SURGE = "surge"
PHASE_SPIKE = "spike"
PHASE_HARD_SAFETY = "hard_safety"
PHASE_RECOVERY = "recovery"

# Steady low-load profile (aligned with the existing dashboard baseline sample)
_BASE = {
    "traffic": 65.0,
    "active_users": 15,
    "cpu_utilization": 28.0,
    "memory_utilization": 35.0,
    "response_time": 85.0,
    "db_query_time": 18.0,
    "system_load": 0.6,
}

# Elevated load below hard-safety thresholds — suitable for ML/controller demos
_SURGE = {
    "traffic": 600.0,
    "active_users": 120,
    "cpu_utilization": 72.0,
    "memory_utilization": 65.0,
    "response_time": 240.0,
    "db_query_time": 45.0,
    "system_load": 1.8,
}

# Deterministic hard-safety breach (CPU and memory at/above configured thresholds)
_HARD_SAFETY = {
    "traffic": 420.0,
    "active_users": 90,
    "cpu_utilization": max(92.0, CPU_CRITICAL_THRESHOLD),
    "memory_utilization": max(91.0, MEMORY_CRITICAL_THRESHOLD),
    "response_time": 380.0,
    "db_query_time": 70.0,
    "system_load": 2.4,
}

# Post-surge cooling profile (does not itself restore levers — controller does)
_RECOVERY = {
    "traffic": 70.0,
    "active_users": 18,
    "cpu_utilization": 32.0,
    "memory_utilization": 38.0,
    "response_time": 95.0,
    "db_query_time": 20.0,
    "system_load": 0.7,
}

REQUIRED_METRIC_KEYS = (
    "traffic",
    "active_users",
    "cpu_utilization",
    "memory_utilization",
    "response_time",
    "db_query_time",
    "system_load",
)


@dataclass(frozen=True)
class SimulationTick:
    """One generated observation. `metric` has no timestamp."""

    index: int
    phase: str
    metric: Mapping[str, Any]

    def as_metric_dict(self, timestamp: datetime) -> Dict[str, Any]:
        """Build a MetricSchema-compatible dict with a playback-assigned timestamp."""
        payload = dict(self.metric)
        payload["timestamp"] = timestamp.isoformat()
        return payload


@dataclass(frozen=True)
class ScenarioDefinition:
    """Named, deterministic sequence of simulation ticks."""

    name: str
    description: str
    ticks: Tuple[SimulationTick, ...]

    @property
    def tick_count(self) -> int:
        return len(self.ticks)

    def get_tick(self, index: int) -> SimulationTick:
        if index < 0 or index >= len(self.ticks):
            raise IndexError(
                f"Tick index {index} out of range for scenario '{self.name}' "
                f"(0..{len(self.ticks) - 1})"
            )
        return self.ticks[index]

    def phases(self) -> List[str]:
        """Phase names in first-seen order."""
        seen: List[str] = []
        for tick in self.ticks:
            if tick.phase not in seen:
                seen.append(tick.phase)
        return seen

    def phase_ranges(self) -> List[Dict[str, Any]]:
        """Inclusive index ranges for each contiguous phase block."""
        if not self.ticks:
            return []
        ranges: List[Dict[str, Any]] = []
        start = 0
        current = self.ticks[0].phase
        for i, tick in enumerate(self.ticks[1:], start=1):
            if tick.phase != current:
                ranges.append(
                    {"phase": current, "start_index": start, "end_index": i - 1}
                )
                start = i
                current = tick.phase
        ranges.append(
            {"phase": current, "start_index": start, "end_index": len(self.ticks) - 1}
        )
        return ranges

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tick_count": self.tick_count,
            "phases": self.phases(),
            "phase_ranges": self.phase_ranges(),
            "warmup_tick_count": WARMUP_TICK_COUNT,
        }


def _metric(**overrides: Any) -> Dict[str, Any]:
    values = dict(_BASE)
    values.update(overrides)
    return {
        "traffic": float(values["traffic"]),
        "active_users": int(values["active_users"]),
        "cpu_utilization": float(values["cpu_utilization"]),
        "memory_utilization": float(values["memory_utilization"]),
        "response_time": float(values["response_time"]),
        "db_query_time": float(values["db_query_time"]),
        "system_load": float(values["system_load"]),
    }


def _lerp(start: Mapping[str, Any], end: Mapping[str, Any], t: float) -> Dict[str, Any]:
    t = min(1.0, max(0.0, t))
    return _metric(
        traffic=float(start["traffic"]) + (float(end["traffic"]) - float(start["traffic"])) * t,
        active_users=int(round(
            float(start["active_users"]) + (float(end["active_users"]) - float(start["active_users"])) * t
        )),
        cpu_utilization=float(start["cpu_utilization"]) + (
            float(end["cpu_utilization"]) - float(start["cpu_utilization"])
        ) * t,
        memory_utilization=float(start["memory_utilization"]) + (
            float(end["memory_utilization"]) - float(start["memory_utilization"])
        ) * t,
        response_time=float(start["response_time"]) + (
            float(end["response_time"]) - float(start["response_time"])
        ) * t,
        db_query_time=float(start["db_query_time"]) + (
            float(end["db_query_time"]) - float(start["db_query_time"])
        ) * t,
        system_load=float(start["system_load"]) + (
            float(end["system_load"]) - float(start["system_load"])
        ) * t,
    )


def _repeat_phase(phase: str, profile: Mapping[str, Any], count: int) -> List[Tuple[str, Dict[str, Any]]]:
    metric = _metric(**profile)
    return [(phase, metric) for _ in range(count)]


def _ramp_phase(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    count: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (PHASE_RAMP, _lerp(start, end, (i + 1) / count))
        for i in range(count)
    ]


def _assemble(name: str, description: str, sequenced: Sequence[Tuple[str, Dict[str, Any]]]) -> ScenarioDefinition:
    ticks = tuple(
        SimulationTick(index=i, phase=phase, metric=metric)
        for i, (phase, metric) in enumerate(sequenced)
    )
    return ScenarioDefinition(name=name, description=description, ticks=ticks)


def _build_baseline() -> ScenarioDefinition:
    sequenced = (
        _repeat_phase(PHASE_WARMUP, _BASE, WARMUP_TICK_COUNT)
        + _repeat_phase(PHASE_BASELINE, _BASE, 20)
    )
    return _assemble(
        SCENARIO_BASELINE,
        "Steady low-load traffic after a full ML window warmup.",
        sequenced,
    )


def _build_gradual_surge() -> ScenarioDefinition:
    ramp = _ramp_phase(_BASE, _SURGE, 12)
    sequenced = (
        _repeat_phase(PHASE_WARMUP, _BASE, WARMUP_TICK_COUNT)
        + ramp
        + _repeat_phase(PHASE_SURGE, _SURGE, 10)
        + _repeat_phase(PHASE_RECOVERY, _RECOVERY, 15)
    )
    return _assemble(
        SCENARIO_GRADUAL_SURGE,
        "Flow 1: warmup and rising traffic drive ML risk through controller confirmations into optimization, followed by cooling ticks.",
        sequenced,
    )


def _build_sudden_spike() -> ScenarioDefinition:
    sequenced = (
        _repeat_phase(PHASE_WARMUP, _BASE, WARMUP_TICK_COUNT)
        + _repeat_phase(PHASE_SPIKE, _SURGE, 8)
        + _repeat_phase(PHASE_RECOVERY, _RECOVERY, 15)
    )
    return _assemble(
        SCENARIO_SUDDEN_SPIKE,
        "Warmup followed by an abrupt surge spike, then cooling recovery ticks.",
        sequenced,
    )


def _build_hard_safety() -> ScenarioDefinition:
    sequenced = (
        _repeat_phase(PHASE_WARMUP, _BASE, WARMUP_TICK_COUNT)
        + _repeat_phase(PHASE_HARD_SAFETY, _HARD_SAFETY, 8)
        + _repeat_phase(PHASE_RECOVERY, _RECOVERY, 12)
    )
    return _assemble(
        SCENARIO_HARD_SAFETY,
        "Flow 3: warmup then CPU/memory hard-threshold breaches; enable manual override in the panel to demonstrate blocked automation.",
        sequenced,
    )


def _build_recovery() -> ScenarioDefinition:
    ramp = _ramp_phase(_BASE, _SURGE, 12)
    sequenced = (
        _repeat_phase(PHASE_WARMUP, _BASE, WARMUP_TICK_COUNT)
        + ramp
        + _repeat_phase(PHASE_SURGE, _SURGE, 10)
        + _repeat_phase(PHASE_RECOVERY, _RECOVERY, 20)
    )
    return _assemble(
        SCENARIO_RECOVERY,
        "Flow 2: rising traffic establishes optimized state, then normal recovery ticks pass through the controller until defaults are restored.",
        sequenced,
    )


_SCENARIOS: Dict[str, ScenarioDefinition] = {
    SCENARIO_BASELINE: _build_baseline(),
    SCENARIO_GRADUAL_SURGE: _build_gradual_surge(),
    SCENARIO_SUDDEN_SPIKE: _build_sudden_spike(),
    SCENARIO_HARD_SAFETY: _build_hard_safety(),
    SCENARIO_RECOVERY: _build_recovery(),
}


def list_scenario_names() -> Tuple[str, ...]:
    return (
        SCENARIO_BASELINE,
        SCENARIO_GRADUAL_SURGE,
        SCENARIO_SUDDEN_SPIKE,
        SCENARIO_HARD_SAFETY,
        SCENARIO_RECOVERY,
    )


def list_scenarios() -> List[Dict[str, Any]]:
    """Metadata for every named scenario (no tick payloads)."""
    return [_SCENARIOS[name].metadata() for name in list_scenario_names()]


def get_scenario(name: str) -> ScenarioDefinition:
    if name not in _SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{name}'. Valid names: {', '.join(list_scenario_names())}"
        )
    return _SCENARIOS[name]


def get_scenario_metadata(name: str) -> Dict[str, Any]:
    return get_scenario(name).metadata()


def get_tick(name: str, index: int) -> SimulationTick:
    return get_scenario(name).get_tick(index)
