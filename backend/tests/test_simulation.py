"""
Tests for Milestone 4 simulation tick generators and simulation reset.

Does not change or re-test POST /monitor behavior except to confirm the
reset endpoint is additive and restores defaults.
"""

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import (
    CACHE_NORMAL,
    CPU_CRITICAL_THRESHOLD,
    HEAVY_COMPONENTS_NORMAL,
    MEMORY_CRITICAL_THRESHOLD,
    PAGINATION_NORMAL,
)
from backend.database import (
    init_db,
    reset_system_state_to_defaults,
    update_system_state,
)
from backend.main import create_app
from backend.ml_engine import ObservationTick, SurgePredictor
from backend.optimizer import ApplicationOptimizer
from backend.schemas import ControllerState, MetricSchema
from backend.simulation import (
    PHASE_HARD_SAFETY,
    PHASE_RAMP,
    PHASE_SPIKE,
    PHASE_SURGE,
    PHASE_WARMUP,
    REQUIRED_METRIC_KEYS,
    WARMUP_TICK_COUNT,
    get_scenario,
    get_scenario_metadata,
    get_tick,
    list_scenario_names,
    list_scenarios,
)


EXPECTED_SCENARIOS = (
    "baseline",
    "gradual_surge",
    "sudden_spike",
    "hard_safety",
    "recovery",
)


def _mock_predictor() -> SurgePredictor:
    model = MagicMock()
    calibrator = MagicMock()
    model.predict_proba = MagicMock(return_value=np.array([[0.98, 0.02]]))
    calibrator.predict_proba = MagicMock(side_effect=lambda x: np.array([x[0]]))

    predictor = object.__new__(SurgePredictor)
    predictor._model = model
    predictor._calibrator = calibrator
    predictor._tau_watch = 0.04
    predictor._tau_crit = 0.075
    predictor._max_window = 61
    predictor._window = deque(maxlen=61)
    predictor._inference_lock = threading.Lock()
    return predictor


# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------


def test_all_five_scenarios_exist():
    names = list_scenario_names()
    assert names == EXPECTED_SCENARIOS
    listed = {item["name"] for item in list_scenarios()}
    assert listed == set(EXPECTED_SCENARIOS)
    for name in EXPECTED_SCENARIOS:
        scenario = get_scenario(name)
        assert scenario.name == name
        assert scenario.tick_count > 0
        assert get_scenario_metadata(name)["name"] == name


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        get_scenario("not_a_real_scenario")


def test_generated_ticks_have_required_metric_structure():
    now = datetime.now(timezone.utc)
    for name in EXPECTED_SCENARIOS:
        scenario = get_scenario(name)
        for tick in scenario.ticks:
            assert set(tick.metric.keys()) == set(REQUIRED_METRIC_KEYS)
            assert "timestamp" not in tick.metric
            payload = tick.as_metric_dict(now)
            parsed = MetricSchema.model_validate(payload)
            assert parsed.traffic >= 0
            assert 0.0 <= parsed.cpu_utilization <= 100.0
            assert 0.0 <= parsed.memory_utilization <= 100.0


def test_get_tick_matches_sequence_and_bounds():
    scenario = get_scenario("gradual_surge")
    first = get_tick("gradual_surge", 0)
    last = get_tick("gradual_surge", scenario.tick_count - 1)
    assert first.index == 0
    assert last.index == scenario.tick_count - 1
    assert first == scenario.get_tick(0)
    with pytest.raises(IndexError):
        get_tick("gradual_surge", scenario.tick_count)


def test_scenario_phases_are_deterministic_and_ordered():
    expected_phases = {
        "baseline": [PHASE_WARMUP, "baseline"],
        "gradual_surge": [PHASE_WARMUP, PHASE_RAMP, PHASE_SURGE, "recovery"],
        "sudden_spike": [PHASE_WARMUP, PHASE_SPIKE, "recovery"],
        "hard_safety": [PHASE_WARMUP, PHASE_HARD_SAFETY, "recovery"],
        "recovery": [PHASE_WARMUP, PHASE_SURGE, "recovery"],
    }
    for name, phases in expected_phases.items():
        scenario = get_scenario(name)
        assert scenario.phases() == phases
        # Same object contents on every call
        again = get_scenario(name)
        assert scenario.ticks == again.ticks
        assert [t.phase for t in scenario.ticks] == [t.phase for t in again.ticks]
        # Phases do not go backwards
        order = {phase: i for i, phase in enumerate(phases)}
        last_rank = -1
        for tick in scenario.ticks:
            rank = order[tick.phase]
            assert rank >= last_rank
            last_rank = rank


def test_warmup_is_long_enough_for_61_tick_window():
    assert WARMUP_TICK_COUNT >= 61
    for name in EXPECTED_SCENARIOS:
        scenario = get_scenario(name)
        warmup = [t for t in scenario.ticks if t.phase == PHASE_WARMUP]
        assert len(warmup) >= 61
        assert all(t.phase == PHASE_WARMUP for t in scenario.ticks[:WARMUP_TICK_COUNT])
        # Main action happens after warmup
        if name != "baseline":
            assert any(t.phase != PHASE_WARMUP for t in scenario.ticks[WARMUP_TICK_COUNT:])


def test_gradual_surge_ramp_increases_traffic():
    scenario = get_scenario("gradual_surge")
    ramp = [t for t in scenario.ticks if t.phase == PHASE_RAMP]
    assert len(ramp) >= 2
    traffic = [t.metric["traffic"] for t in ramp]
    assert traffic == sorted(traffic)
    assert traffic[-1] > traffic[0]


def test_hard_safety_scenario_breaches_cpu_and_memory_thresholds():
    scenario = get_scenario("hard_safety")
    hard_ticks = [t for t in scenario.ticks if t.phase == PHASE_HARD_SAFETY]
    assert len(hard_ticks) > 0
    for tick in hard_ticks:
        assert tick.metric["cpu_utilization"] >= CPU_CRITICAL_THRESHOLD
        assert tick.metric["memory_utilization"] >= MEMORY_CRITICAL_THRESHOLD


def test_hard_safety_warmup_is_below_thresholds():
    scenario = get_scenario("hard_safety")
    warmup = [t for t in scenario.ticks if t.phase == PHASE_WARMUP]
    assert warmup
    for tick in warmup:
        assert tick.metric["cpu_utilization"] < CPU_CRITICAL_THRESHOLD
        assert tick.metric["memory_utilization"] < MEMORY_CRITICAL_THRESHOLD


# ---------------------------------------------------------------------------
# Database + ML window reset
# ---------------------------------------------------------------------------


def test_reset_system_state_restores_defaults(tmp_path):
    db_path = str(tmp_path / "sim_reset.db")
    init_db(db_path)
    optimizer = ApplicationOptimizer(db_path=db_path)
    ok, _, err = optimizer.apply_optimizations(reason="test setup")
    assert ok is True
    assert err is None
    update_system_state(
        controller_state=ControllerState.OPTIMIZED,
        consecutive_high_risk=3,
        consecutive_normal=1,
        manual_override=True,
        last_optimization_timestamp=datetime.now(timezone.utc),
        db_path=db_path,
    )

    state = reset_system_state_to_defaults(db_path=db_path)
    assert state["controller_state"] == ControllerState.NORMAL
    assert state["caching"] == CACHE_NORMAL
    assert state["pagination_size"] == PAGINATION_NORMAL
    assert state["heavy_components"] == HEAVY_COMPONENTS_NORMAL
    assert state["manual_override"] is False
    assert state["consecutive_high_risk"] == 0
    assert state["consecutive_normal"] == 0
    assert state["last_optimization_timestamp"] is None


def test_clear_window_preserves_loaded_model():
    predictor = _mock_predictor()
    model_ref = predictor._model
    calibrator_ref = predictor._calibrator
    tau = predictor.thresholds
    for i in range(20):
        predictor.push_tick(ObservationTick(invocations=float(i)))
    assert predictor.window_length == 20

    predictor.clear_window()

    assert predictor.window_length == 0
    assert predictor._model is model_ref
    assert predictor._calibrator is calibrator_ref
    assert predictor.thresholds == tau
    # Singleton unload must still be a separate test-only reset()
    SurgePredictor._instance = predictor
    try:
        predictor.clear_window()
        assert SurgePredictor.get_instance() is predictor
    finally:
        SurgePredictor.reset()


def test_simulation_reset_endpoint_restores_state(tmp_path):
    db_file = str(tmp_path / "sim_api.db")
    init_db(db_file)
    app = create_app(db_path=db_file, load_ml=False)
    with TestClient(app) as client:
        optimizer = ApplicationOptimizer(db_path=db_file)
        optimizer.apply_optimizations(reason="api reset setup")
        update_system_state(
            controller_state=ControllerState.OPTIMIZED,
            consecutive_high_risk=2,
            db_path=db_file,
        )

        response = client.post("/simulation/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["controller_state"] == "NORMAL"
        assert data["current_configuration"]["caching"] == CACHE_NORMAL
        assert data["current_configuration"]["pagination_size"] == PAGINATION_NORMAL
        assert data["current_configuration"]["heavy_components"] == HEAVY_COMPONENTS_NORMAL
        assert data["manual_override"] is False
        assert data["consecutive_high_risk"] == 0
        assert data["consecutive_normal"] == 0
        assert data["last_optimization_timestamp"] is None
        assert data["ml_window_cleared"] is False
        assert data["ml_window_length"] == 0

        state = client.get("/state").json()
        assert state["controller_state"] == "NORMAL"


def test_simulation_reset_clears_ml_window_without_unloading(tmp_path):
    db_file = str(tmp_path / "sim_ml_reset.db")
    init_db(db_file)
    SurgePredictor.reset()

    thresholds_file = tmp_path / "thresholds.json"
    thresholds_file.write_text(
        '{"watch_threshold": 0.04, "critical_threshold": 0.075}'
    )

    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.98, 0.02]]))
    mock_calibrator = MagicMock()
    mock_calibrator.predict_proba = MagicMock(side_effect=lambda x: np.array([x[0]]))

    def _fake_joblib_load(path):
        path_str = str(path)
        if "cal" in path_str:
            return mock_calibrator
        return mock_model

    with patch("backend.ml_engine.joblib.load", side_effect=_fake_joblib_load):
        app = create_app(
            db_path=db_file,
            ml_model_path=tmp_path / "model.joblib",
            ml_calibrator_path=tmp_path / "cal.joblib",
            ml_thresholds_path=thresholds_file,
            load_ml=True,
        )
        with TestClient(app) as client:
            predictor = SurgePredictor.get_instance()
            model_ref = predictor._model
            calibrator_ref = predictor._calibrator
            for i in range(15):
                predictor.push_tick(ObservationTick(invocations=float(i)))
            assert predictor.window_length == 15

            response = client.post("/simulation/reset")
            assert response.status_code == 200
            data = response.json()
            assert data["ml_window_cleared"] is True
            assert data["ml_window_length"] == 0
            assert predictor.window_length == 0
            assert SurgePredictor.get_instance() is predictor
            assert predictor._model is model_ref
            assert predictor._calibrator is calibrator_ref

    SurgePredictor.reset()


# ---------------------------------------------------------------------------
# Scenario HTTP API
# ---------------------------------------------------------------------------


def test_get_simulation_scenarios_list(tmp_path):
    db_file = str(tmp_path / "sim_list.db")
    init_db(db_file)
    app = create_app(db_path=db_file, load_ml=False)
    with TestClient(app) as client:
        response = client.get("/simulation/scenarios")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        names = [item["name"] for item in data]
        assert names == list(EXPECTED_SCENARIOS)
        for item in data:
            assert "ticks" not in item
            assert item["tick_count"] >= 61
            assert "description" in item
            assert "phases" in item
            assert "phase_ranges" in item
            assert item["warmup_tick_count"] >= 61


def test_get_simulation_scenario_detail_has_ticks_without_timestamps(tmp_path):
    db_file = str(tmp_path / "sim_detail.db")
    init_db(db_file)
    app = create_app(db_path=db_file, load_ml=False)
    with TestClient(app) as client:
        response = client.get("/simulation/scenarios/gradual_surge")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "gradual_surge"
        assert data["tick_count"] == len(data["ticks"])
        assert data["tick_count"] > 0
        first = data["ticks"][0]
        assert first["index"] == 0
        assert "phase" in first
        assert "metric" in first
        assert "timestamp" not in first
        assert "timestamp" not in first["metric"]
        for key in (
            "traffic",
            "active_users",
            "cpu_utilization",
            "memory_utilization",
            "response_time",
            "db_query_time",
            "system_load",
        ):
            assert key in first["metric"]


def test_get_unknown_simulation_scenario_returns_404(tmp_path):
    db_file = str(tmp_path / "sim_404.db")
    init_db(db_file)
    app = create_app(db_path=db_file, load_ml=False)
    with TestClient(app) as client:
        response = client.get("/simulation/scenarios/not_a_real_scenario")
        assert response.status_code == 404

