"""
Tests for SafetyController state machine, confirmations, safety thresholds,
cooldown, recovery hysteresis, and manual override.
"""

from datetime import datetime, timezone, timedelta
from backend.database import init_db, get_system_state
from backend.controller import SafetyController
from backend.schemas import ControllerState, MetricSchema, RiskSignal
from backend.config import (
    CACHE_NORMAL,
    CACHE_OPTIMIZED,
    PAGINATION_NORMAL,
    PAGINATION_OPTIMIZED,
    HEAVY_COMPONENTS_NORMAL,
    HEAVY_COMPONENTS_OPTIMIZED,
)


def create_metric(
    timestamp: datetime = None,
    cpu: float = 40.0,
    memory: float = 40.0,
    response_time: float = 100.0,
    load: float = 1.0
) -> MetricSchema:
    return MetricSchema(
        timestamp=timestamp or datetime.now(timezone.utc),
        traffic=100.0,
        active_users=20,
        cpu_utilization=cpu,
        memory_utilization=memory,
        response_time=response_time,
        db_query_time=20.0,
        system_load=load
    )


def test_initial_state(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file)

    state = controller.get_state()
    assert state.controller_state == ControllerState.NORMAL
    assert state.current_configuration.caching == CACHE_NORMAL
    assert state.manual_override is False
    assert state.consecutive_high_risk == 0
    assert state.consecutive_normal == 0


def test_elevated_risk_transitions_to_watching(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file)

    metric = create_metric()
    res = controller.process_metric(metric=metric, risk_signal=RiskSignal.ELEVATED, is_stale=False)

    assert res.controller_state == ControllerState.WATCHING
    assert res.optimization_applied is False
    assert res.current_configuration.caching == CACHE_NORMAL


def test_fewer_than_3_high_risk_does_not_optimize(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, high_risk_confirmations=3)

    metric = create_metric()

    # Reading 1
    res1 = controller.process_metric(metric=metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)
    assert res1.controller_state == ControllerState.WATCHING
    assert res1.consecutive_high_risk == 1
    assert res1.optimization_applied is False

    # Reading 2
    res2 = controller.process_metric(metric=metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)
    assert res2.controller_state == ControllerState.WATCHING
    assert res2.consecutive_high_risk == 2
    assert res2.optimization_applied is False
    assert res2.current_configuration.caching == CACHE_NORMAL


def test_3_consecutive_high_risk_triggers_optimization(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, high_risk_confirmations=3)

    metric = create_metric()

    # Reading 1 & 2
    controller.process_metric(metric=metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)
    controller.process_metric(metric=metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)

    # Reading 3
    res3 = controller.process_metric(metric=metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)
    assert res3.controller_state == ControllerState.OPTIMIZED
    assert res3.optimization_applied is True
    assert res3.current_configuration.caching == CACHE_OPTIMIZED
    assert res3.current_configuration.pagination_size == PAGINATION_OPTIMIZED
    assert res3.current_configuration.heavy_components == HEAVY_COMPONENTS_OPTIMIZED


def test_critical_safety_threshold_triggers_immediate_optimization(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file)

    # Metric breaches CPU critical threshold (95% >= 90%)
    severe_metric = create_metric(cpu=95.0)

    # Single reading with generic NORMAL signal should still trigger immediate optimization due to safety breach
    res = controller.process_metric(metric=severe_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    assert res.controller_state == ControllerState.OPTIMIZED
    assert res.optimization_applied is True
    assert res.current_configuration.caching == CACHE_OPTIMIZED


def test_cooldown_blocks_repeated_optimization(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, cooldown_seconds=60.0)

    t0 = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)
    severe_metric = create_metric(timestamp=t0, cpu=95.0)

    # Apply optimization at t0
    res1 = controller.process_metric(metric=severe_metric, risk_signal=RiskSignal.CRITICAL, is_stale=False, now=t0)
    assert res1.optimization_applied is True
    assert res1.controller_state == ControllerState.OPTIMIZED

    # Attempt repeated optimization 10 seconds later (within 60s cooldown)
    t1 = t0 + timedelta(seconds=10)
    res2 = controller.process_metric(metric=severe_metric, risk_signal=RiskSignal.CRITICAL, is_stale=False, now=t1)
    assert res2.optimization_applied is False
    assert res2.cooldown_active is True
    assert "cooldown" in (res2.optimization_blocked_reason or "").lower()


def test_recovery_requires_3_consecutive_normal_readings(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, high_risk_confirmations=3, recovery_confirmations=3)

    # Drive system into OPTIMIZED
    severe_metric = create_metric(cpu=95.0)
    controller.process_metric(metric=severe_metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)
    assert controller.get_state().controller_state == ControllerState.OPTIMIZED

    normal_metric = create_metric(cpu=30.0, memory=30.0, response_time=80.0, load=0.8)

    # Reading 1: transitions to RECOVERY (count 1)
    res1 = controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    assert res1.controller_state == ControllerState.RECOVERY
    assert res1.consecutive_normal == 1
    assert res1.current_configuration.caching == CACHE_OPTIMIZED

    # Reading 2: remains in RECOVERY (count 2)
    res2 = controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    assert res2.controller_state == ControllerState.RECOVERY
    assert res2.consecutive_normal == 2
    assert res2.current_configuration.caching == CACHE_OPTIMIZED

    # Reading 3: full recovery confirmed -> restores defaults and transitions to NORMAL
    res3 = controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    assert res3.controller_state == ControllerState.NORMAL
    assert res3.consecutive_normal == 0
    assert res3.current_configuration.caching == CACHE_NORMAL
    assert res3.current_configuration.pagination_size == PAGINATION_NORMAL
    assert res3.current_configuration.heavy_components == HEAVY_COMPONENTS_NORMAL


def test_interrupted_recovery_resets_count(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, recovery_confirmations=3)

    # Enter OPTIMIZED
    controller.process_metric(metric=create_metric(cpu=95.0), risk_signal=RiskSignal.CRITICAL, is_stale=False)

    normal_metric = create_metric()
    # 2 normal readings
    controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    assert controller.get_state().consecutive_normal == 2

    # Elevated reading interrupts recovery
    res = controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.ELEVATED, is_stale=False)
    assert res.controller_state == ControllerState.OPTIMIZED
    assert res.consecutive_normal == 0
    assert res.current_configuration.caching == CACHE_OPTIMIZED


def test_manual_override_blocks_automatic_optimization(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, high_risk_confirmations=1)

    # Enable manual override
    controller.set_manual_override(True)
    assert controller.get_state().manual_override is True

    # Severe metric that would otherwise trigger optimization immediately
    severe_metric = create_metric(cpu=99.0)
    res = controller.process_metric(metric=severe_metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)

    assert res.optimization_applied is False
    assert "Manual override" in (res.optimization_blocked_reason or "")
    assert res.current_configuration.caching == CACHE_NORMAL


def test_stale_metric_blocks_optimization_and_counter_increment(tmp_path):
    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, high_risk_confirmations=3)

    metric = create_metric()
    res = controller.process_metric(metric=metric, risk_signal=RiskSignal.CRITICAL, is_stale=True)

    assert res.optimization_applied is False
    assert res.consecutive_high_risk == 0
    assert "Stale metric" in (res.optimization_blocked_reason or "")


def test_recovery_failure_handling(tmp_path):
    """
    Regression test: When optimizer.restore_defaults() fails during sustained recovery:
    1. Controller must NOT transition to NORMAL.
    2. Controller must remain in a safe state (RECOVERY).
    3. Actual configuration must remain optimized.
    4. Failure reason must be returned in optimization_blocked_reason.
    5. Controller must not falsely claim recovery success.
    """
    from unittest.mock import patch

    db_file = str(tmp_path / "test_ctrl.db")
    init_db(db_file)
    controller = SafetyController(db_path=db_file, recovery_confirmations=3)

    # 1. Drive system into OPTIMIZED
    severe_metric = create_metric(cpu=95.0)
    res_opt = controller.process_metric(metric=severe_metric, risk_signal=RiskSignal.CRITICAL, is_stale=False)
    assert res_opt.controller_state == ControllerState.OPTIMIZED
    assert res_opt.current_configuration.caching == CACHE_OPTIMIZED

    normal_metric = create_metric(cpu=30.0, memory=30.0, response_time=80.0, load=0.8)

    # 2. Reading 1 & 2 of normal conditions
    controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)
    assert controller.get_state().consecutive_normal == 2

    # 3. 3rd normal reading: mock optimizer.restore_defaults to return failure
    opt_config = controller.optimizer.get_configuration()
    with patch.object(
        controller.optimizer,
        "restore_defaults",
        return_value=(False, opt_config, "Disk I/O error during restoration")
    ):
        res3 = controller.process_metric(metric=normal_metric, risk_signal=RiskSignal.NORMAL, is_stale=False)

        # Assertions
        assert res3.controller_state != ControllerState.NORMAL
        assert res3.controller_state == ControllerState.RECOVERY
        assert controller.get_state().controller_state == ControllerState.RECOVERY
        assert res3.optimization_applied is False
        assert res3.current_configuration.caching == CACHE_OPTIMIZED
        assert res3.current_configuration.pagination_size == PAGINATION_OPTIMIZED
        assert res3.current_configuration.heavy_components == HEAVY_COMPONENTS_OPTIMIZED
        assert "recovery failed" in (res3.optimization_blocked_reason or "").lower()
        assert "Disk I/O error" in (res3.optimization_blocked_reason or "")

