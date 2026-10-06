"""
FastAPI Application for AutoScale IQ — ML-Integrated Backend.
Provides REST API endpoints for metric monitoring, validation, ML surge prediction,
state inspection, event history, and safety override controls.

ML Pipeline (per /monitor invocation):
    Metric → ObservationTick → SurgePredictor → calibrated probability
    → RiskSignal → SafetyController → optimization decision
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging

from backend.config import (
    DEFAULT_DB_PATH,
    ML_MODEL_PATH,
    ML_CALIBRATOR_PATH,
    ML_THRESHOLDS_PATH,
)
from backend.database import (
    init_db,
    insert_metric,
    insert_ml_prediction,
    get_recent_events,
    get_recent_metrics,
    get_recent_predictions,
)
from backend.monitoring import assess_freshness
from backend.controller import SafetyController
from backend.ml_engine import ObservationTick, SurgePredictor
from backend.schemas import (
    HealthResponse,
    MetricRecordSchema,
    MetricSchema,
    MLPredictionRecordSchema,
    MLPredictionResponse,
    MonitorRequest,
    MonitorResponse,
    OptimizationEventSchema,
    OverrideRequest,
    PredictRequest,
    PredictResponse,
    RiskSignal,
    StateResponse,
    ValidateMetricResponse,
)

logger = logging.getLogger("autoscale_iq.main")


def _build_observation_tick(metric: MetricSchema) -> ObservationTick:
    """
    Convert a MetricSchema into an ObservationTick for the ML feature engine.
    Uses traffic as invocations; defaults to HTTP trigger type.

    Note: Resource/latency fields (cpu, memory, response time, db query time)
    are NOT part of the locked 19-feature ML schema and are not passed here.
    """
    return ObservationTick(
        invocations=metric.traffic,
        trigger_http=True,
        trigger_timer=False,
        trigger_queue=False,
        trigger_orchestration=False,
        trigger_event=False,
        trigger_storage=False,
        trigger_others=False,
    )


def _ml_result_to_schema(result) -> MLPredictionResponse:
    """Convert MLPredictionResult dataclass → Pydantic MLPredictionResponse."""
    return MLPredictionResponse(
        surge_probability=result.surge_probability,
        raw_probability=result.raw_probability,
        risk_signal=result.risk_signal,
        watch_threshold=result.watch_threshold,
        critical_threshold=result.critical_threshold,
        window_size=result.window_size,
        insufficient_data=result.insufficient_data,
        inference_time_ms=result.inference_time_ms,
    )


def create_app(
    db_path: str = DEFAULT_DB_PATH,
    ml_model_path=ML_MODEL_PATH,
    ml_calibrator_path=ML_CALIBRATOR_PATH,
    ml_thresholds_path=ML_THRESHOLDS_PATH,
    load_ml: bool = True,
) -> FastAPI:
    """
    Factory function to create the FastAPI app.

    Parameters
    ----------
    db_path:
        SQLite database path.
    ml_model_path / ml_calibrator_path / ml_thresholds_path:
        Paths to the locked ML artifacts.
    load_ml:
        Set False to skip ML loading (useful for lightweight unit tests).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize database
        init_db(app.state.db_path)

        # Initialize ML engine singleton
        if app.state.load_ml:
            try:
                SurgePredictor.initialize(
                    model_path=app.state.ml_model_path,
                    calibrator_path=app.state.ml_calibrator_path,
                    thresholds_path=app.state.ml_thresholds_path,
                )
                logger.info("SurgePredictor initialized successfully")
            except Exception as exc:
                logger.error("Failed to initialize SurgePredictor: %s", exc)
                # Do not crash — ML will be gracefully disabled
        yield

    app = FastAPI(
        title="AutoScale IQ Engine API",
        description=(
            "ML-Powered Self-Optimizing Web Application — "
            "Safety & Monitoring with Surge Prediction"
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    app.state.db_path = db_path
    app.state.ml_model_path = ml_model_path
    app.state.ml_calibrator_path = ml_calibrator_path
    app.state.ml_thresholds_path = ml_thresholds_path
    app.state.load_ml = load_ml

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_controller() -> SafetyController:
        return SafetyController(db_path=app.state.db_path)

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health_check():
        """Health check endpoint confirming service status."""
        return HealthResponse(
            status="ok",
            timestamp=datetime.now(timezone.utc)
        )

    # -------------------------------------------------------------------------
    # Metrics validation
    # -------------------------------------------------------------------------

    @app.post("/metrics/validate", response_model=ValidateMetricResponse, tags=["Metrics"])
    def validate_metric(metric: MetricSchema):
        """
        Validate incoming metric schema.
        Returns validation details or rejects malformed payloads with 422.
        """
        return ValidateMetricResponse(
            valid=True,
            metric=metric,
            message="Metric payload is valid"
        )

    # -------------------------------------------------------------------------
    # Monitor (main ingestion endpoint)
    # -------------------------------------------------------------------------

    @app.post("/monitor", response_model=MonitorResponse, tags=["Monitoring"])
    def monitor_metric(payload: MonitorRequest):
        """
        Ingest a metric observation, run ML surge prediction, classify risk,
        apply the safety state machine, and optionally trigger optimizations.

        ML Pipeline:
            MetricSchema → ObservationTick → SurgePredictor
            → calibrated probability → RiskSignal
            → SafetyController.process_metric()

        The risk_signal field in the request payload is OVERRIDDEN by the ML
        engine output when ML is available. The payload risk_signal acts as a
        fallback when the ML engine is unavailable.
        """
        metric = payload.metric
        fallback_risk_signal = payload.risk_signal

        # 1. Freshness check
        is_stale, age_seconds = assess_freshness(metric)

        # 2. Persist metric into SQLite
        try:
            insert_metric(metric, db_path=app.state.db_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist metric: {str(e)}"
            )

        # 3. ML inference (only when metric is fresh)
        ml_result = None
        effective_risk_signal: RiskSignal = fallback_risk_signal

        if app.state.load_ml and not is_stale:
            try:
                predictor = SurgePredictor.get_instance()
                tick = _build_observation_tick(metric)
                minute_of_day = metric.timestamp.hour * 60 + metric.timestamp.minute
                ml_result = predictor.predict(tick, minute_of_day)
                # Push tick into window AFTER prediction
                predictor.push_tick(tick)
                # Use ML risk signal as the effective signal
                effective_risk_signal = ml_result.risk_signal

                # Persist ML prediction (graceful degradation on persistence failure)
                try:
                    insert_ml_prediction(
                        probability=ml_result.surge_probability,
                        risk_signal=ml_result.risk_signal.value,
                        watch_threshold=ml_result.watch_threshold,
                        critical_threshold=ml_result.critical_threshold,
                        feature_snapshot=ml_result.feature_vector,
                        timestamp=metric.timestamp,
                        db_path=app.state.db_path,
                    )
                except Exception as db_exc:
                    logger.warning("Failed to persist ML prediction: %s", db_exc)
            except RuntimeError:
                # SurgePredictor not initialized — use fallback
                logger.warning("SurgePredictor unavailable; using fallback risk signal")
            except Exception as exc:
                logger.error("ML inference error: %s", exc, exc_info=True)

        # 4. Process through safety state machine
        controller = get_controller()
        result = controller.process_metric(
            metric=metric,
            risk_signal=effective_risk_signal,
            is_stale=is_stale,
        )

        return MonitorResponse(
            metric_valid=True,
            stale=is_stale,
            metric_age_seconds=round(age_seconds, 3),
            risk_signal=effective_risk_signal,
            controller_state=result.controller_state,
            optimization_applied=result.optimization_applied,
            optimization_blocked_reason=result.optimization_blocked_reason,
            current_configuration=result.current_configuration,
            manual_override=result.manual_override,
            cooldown_active=result.cooldown_active,
            consecutive_high_risk=result.consecutive_high_risk,
            consecutive_normal=result.consecutive_normal,
            ml_prediction=_ml_result_to_schema(ml_result) if ml_result else None,
        )

    # -------------------------------------------------------------------------
    # Direct ML prediction endpoint (no controller, no persistence)
    # -------------------------------------------------------------------------

    @app.post("/predict", response_model=PredictResponse, tags=["ML Prediction"])
    def predict_surge(payload: PredictRequest):
        """
        Run a direct ML surge prediction without going through the safety controller.
        Does NOT persist data or affect system state — purely read-only inference.
        """
        if not app.state.load_ml:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML engine not loaded in this configuration"
            )
        try:
            predictor = SurgePredictor.get_instance()
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML engine not initialized"
            )

        tick = ObservationTick(
            invocations=payload.invocations,
            trigger_timer=payload.trigger_timer,
            trigger_http=payload.trigger_http,
            trigger_queue=payload.trigger_queue,
            trigger_orchestration=payload.trigger_orchestration,
            trigger_event=payload.trigger_event,
            trigger_storage=payload.trigger_storage,
            trigger_others=payload.trigger_others,
        )
        result = predictor.predict(tick, payload.minute_of_day)

        return PredictResponse(
            ml_prediction=_ml_result_to_schema(result),
        )

    # -------------------------------------------------------------------------
    # State, events, history, override
    # -------------------------------------------------------------------------

    @app.get("/metrics/history", response_model=List[MetricRecordSchema], tags=["Metrics"])
    def get_metrics_history(
        limit: int = Query(default=60, ge=1, le=500, description="Max telemetry records to return")
    ):
        """
        Get recent telemetry metric observations in chronological order.
        Read-only; does not run ML inference or step the controller.
        """
        return get_recent_metrics(limit=limit, db_path=app.state.db_path)

    @app.get("/predictions/history", response_model=List[MLPredictionRecordSchema], tags=["ML Prediction"])
    def get_predictions_history(
        limit: int = Query(default=60, ge=1, le=500, description="Max prediction records to return")
    ):
        """
        Get recent ML surge prediction records in chronological order.
        Read-only; does not perform new inference or modify state.
        """
        return get_recent_predictions(limit=limit, db_path=app.state.db_path)

    @app.get("/state", response_model=StateResponse, tags=["State"])
    def get_system_state_endpoint():
        """Get current controller state, configuration levers, and cooldown info."""
        controller = get_controller()
        return controller.get_state()

    @app.get("/events", response_model=List[OptimizationEventSchema], tags=["Events"])
    def get_events(
        limit: int = Query(default=50, ge=1, le=500, description="Max events to return")
    ):
        """Get recent optimization and recovery events."""
        events = get_recent_events(limit=limit, db_path=app.state.db_path)
        return events

    @app.post("/override", response_model=StateResponse, tags=["Override"])
    def set_manual_override_endpoint(payload: OverrideRequest):
        """Enable or disable manual safety override."""
        controller = get_controller()
        return controller.set_manual_override(payload.manual_override)

    return app


# Default application instance for Uvicorn
app = create_app()
