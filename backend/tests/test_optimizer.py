"""
Tests for ApplicationOptimizer.
Verifies configuration state changes (caching, pagination, heavy components),
idempotency, restore defaults, and failure handling.
"""

from unittest.mock import patch
from backend.database import init_db, get_recent_events, get_system_state
from backend.optimizer import ApplicationOptimizer
from backend.config import (
    CACHE_NORMAL,
    CACHE_OPTIMIZED,
    PAGINATION_NORMAL,
    PAGINATION_OPTIMIZED,
    HEAVY_COMPONENTS_NORMAL,
    HEAVY_COMPONENTS_OPTIMIZED,
)


def test_default_configuration(tmp_path):
    db_file = str(tmp_path / "test_opt.db")
    init_db(db_file)
    optimizer = ApplicationOptimizer(db_path=db_file)

    config = optimizer.get_configuration()
    assert config.caching == CACHE_NORMAL
    assert config.pagination_size == PAGINATION_NORMAL
    assert config.heavy_components == HEAVY_COMPONENTS_NORMAL
    assert optimizer.is_default_state()
    assert not optimizer.is_already_optimized()


def test_apply_optimizations(tmp_path):
    db_file = str(tmp_path / "test_opt.db")
    init_db(db_file)
    optimizer = ApplicationOptimizer(db_path=db_file)

    success, new_cfg, err = optimizer.apply_optimizations(reason="High traffic detected")
    assert success is True
    assert err is None
    assert new_cfg.caching == CACHE_OPTIMIZED
    assert new_cfg.pagination_size == PAGINATION_OPTIMIZED
    assert new_cfg.heavy_components == HEAVY_COMPONENTS_OPTIMIZED
    assert optimizer.is_already_optimized()

    # Verify event logged
    events = get_recent_events(db_path=db_file)
    assert len(events) == 1
    assert events[0]["action"] == "apply_optimizations"
    assert events[0]["success"] is True
    assert events[0]["previous_state"]["caching"] == CACHE_NORMAL
    assert events[0]["new_state"]["caching"] == CACHE_OPTIMIZED


def test_optimizer_idempotency(tmp_path):
    db_file = str(tmp_path / "test_opt.db")
    init_db(db_file)
    optimizer = ApplicationOptimizer(db_path=db_file)

    # First application
    optimizer.apply_optimizations(reason="First call")
    events_after_first = len(get_recent_events(db_path=db_file))

    # Second application (already optimized)
    success, cfg, err = optimizer.apply_optimizations(reason="Second call")
    assert success is True
    assert cfg.caching == CACHE_OPTIMIZED
    # No redundant duplicate event inserted
    events_after_second = len(get_recent_events(db_path=db_file))
    assert events_after_second == events_after_first


def test_restore_defaults(tmp_path):
    db_file = str(tmp_path / "test_opt.db")
    init_db(db_file)
    optimizer = ApplicationOptimizer(db_path=db_file)

    optimizer.apply_optimizations(reason="Setup")
    assert optimizer.is_already_optimized()

    success, restored_cfg, err = optimizer.restore_defaults(reason="Normal traffic resumed")
    assert success is True
    assert err is None
    assert restored_cfg.caching == CACHE_NORMAL
    assert restored_cfg.pagination_size == PAGINATION_NORMAL
    assert restored_cfg.heavy_components == HEAVY_COMPONENTS_NORMAL
    assert optimizer.is_default_state()

    events = get_recent_events(db_path=db_file)
    assert events[0]["action"] == "restore_defaults"
    assert events[0]["success"] is True


def test_restore_defaults_idempotency(tmp_path):
    db_file = str(tmp_path / "test_opt.db")
    init_db(db_file)
    optimizer = ApplicationOptimizer(db_path=db_file)

    # System is already in default state
    success, cfg, err = optimizer.restore_defaults(reason="Already normal")
    assert success is True
    assert optimizer.is_default_state()
    # No event logged for no-op
    events = get_recent_events(db_path=db_file)
    assert len(events) == 0


def test_optimizer_failure_handling(tmp_path):
    db_file = str(tmp_path / "test_opt.db")
    init_db(db_file)
    optimizer = ApplicationOptimizer(db_path=db_file)

    # Simulate database error during update_system_state
    with patch("backend.optimizer.update_system_state", side_effect=RuntimeError("Disk write failed")):
        success, cfg, err = optimizer.apply_optimizations(reason="Simulate failure")
        assert success is False
        assert "Disk write failed" in (err or "")
        # Configuration should remain in default state
        assert cfg.caching == CACHE_NORMAL

    # Verify failure was logged in events table
    events = get_recent_events(db_path=db_file)
    assert len(events) == 1
    assert events[0]["action"] == "apply_optimizations"
    assert events[0]["success"] is False
    assert "Disk write failed" in events[0]["error_message"]
