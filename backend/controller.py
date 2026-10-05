"""
Safety Controller for AutoScale IQ.
Implements the core state machine (NORMAL -> WATCHING -> OPTIMIZED -> RECOVERY),
safety thresholds, confirmation counters, cooldown enforcement, and manual overrides.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Optional, Tuple

from backend.config import (
    DEFAULT_DB_PATH,
    DEFAULT_HIGH_RISK_CONFIRMATIONS,
    DEFAULT_RECOVERY_CONFIRMATIONS,
    DEFAULT_COOLDOWN_SECONDS,
)
from backend.database import get_system_state, update_system_state
from backend.monitoring import evaluate_hard_safety_thresholds
from backend.optimizer import ApplicationOptimizer
from backend.schemas import (
    ControllerState,
    MetricSchema,
    RiskSignal,
    StateResponse,
    SystemConfiguration,
)

logger = logging.getLogger("autoscale_iq.controller")


@dataclass
class ControllerStepResult:
    controller_state: ControllerState
    optimization_applied: bool
    optimization_blocked_reason: Optional[str]
    current_configuration: SystemConfiguration
    manual_override: bool
    cooldown_active: bool
    consecutive_high_risk: int
    consecutive_normal: int


class SafetyController:
    """
    Safety Controller managing system state transitions and optimization decisions.
    Enforces high-risk confirmation count, hard safety thresholds, cooldown,
    recovery hysteresis, and manual overrides.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        high_risk_confirmations: int = DEFAULT_HIGH_RISK_CONFIRMATIONS,
        recovery_confirmations: int = DEFAULT_RECOVERY_CONFIRMATIONS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ):
        self.db_path = db_path
        self.high_risk_confirmations = high_risk_confirmations
        self.recovery_confirmations = recovery_confirmations
        self.cooldown_seconds = cooldown_seconds
        self.optimizer = ApplicationOptimizer(db_path=self.db_path)

    def is_in_cooldown(
        self,
        last_opt_time: Optional[datetime],
        now: Optional[datetime] = None
    ) -> Tuple[bool, float]:
        """
        Check if the system is currently within the post-optimization cooldown window.
        Returns (is_active: bool, remaining_seconds: float).
        """
        if not last_opt_time:
            return False, 0.0

        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        if last_opt_time.tzinfo is None:
            last_opt_time = last_opt_time.replace(tzinfo=timezone.utc)

        elapsed = (current_time - last_opt_time).total_seconds()
        remaining = self.cooldown_seconds - elapsed
        if remaining > 0:
            return True, remaining
        return False, 0.0

    def get_state(self, now: Optional[datetime] = None) -> StateResponse:
        """Retrieve full structured state response including cooldown."""
        state = get_system_state(self.db_path)
        cooldown_active, cooldown_remaining = self.is_in_cooldown(
            state["last_optimization_timestamp"], now=now
        )
        return StateResponse(
            controller_state=state["controller_state"],
            current_configuration=SystemConfiguration(
                caching=state["caching"],
                pagination_size=state["pagination_size"],
                heavy_components=state["heavy_components"],
            ),
            manual_override=state["manual_override"],
            consecutive_high_risk=state["consecutive_high_risk"],
            consecutive_normal=state["consecutive_normal"],
            cooldown_active=cooldown_active,
            cooldown_remaining_seconds=max(0.0, cooldown_remaining),
            last_optimization_timestamp=state["last_optimization_timestamp"],
            updated_at=state["updated_at"],
        )

    def set_manual_override(self, manual_override: bool) -> StateResponse:
        """Enable or disable manual override and return updated state."""
        update_system_state(manual_override=manual_override, db_path=self.db_path)
        return self.get_state()

    def process_metric(
        self,
        metric: MetricSchema,
        risk_signal: RiskSignal,
        is_stale: bool,
        now: Optional[datetime] = None
    ) -> ControllerStepResult:
        """
        Process a metric reading through the safety state machine.
        Decides whether to watch, optimize, recover, or remain unchanged.
        """
        current_time = now or datetime.now(timezone.utc)
        state_dict = get_system_state(self.db_path)
        current_state: ControllerState = state_dict["controller_state"]
        manual_override: bool = state_dict["manual_override"]
        high_risk_count: int = state_dict["consecutive_high_risk"]
        normal_count: int = state_dict["consecutive_normal"]
        last_opt_time: Optional[datetime] = state_dict["last_optimization_timestamp"]

        cooldown_active, _ = self.is_in_cooldown(last_opt_time, now=current_time)
        current_config = self.optimizer.get_configuration()

        # 1. STALE METRIC CHECK
        if is_stale:
            # Stale metrics MUST NOT trigger automatic optimization or advance confirmation counters
            return ControllerStepResult(
                controller_state=current_state,
                optimization_applied=False,
                optimization_blocked_reason="Stale metric detected; automation paused for safety",
                current_configuration=current_config,
                manual_override=manual_override,
                cooldown_active=cooldown_active,
                consecutive_high_risk=high_risk_count,
                consecutive_normal=normal_count,
            )

        # 2. MANUAL OVERRIDE CHECK
        if manual_override:
            # Automation is completely disabled by user
            return ControllerStepResult(
                controller_state=current_state,
                optimization_applied=False,
                optimization_blocked_reason="Manual override active; automatic optimization blocked",
                current_configuration=current_config,
                manual_override=True,
                cooldown_active=cooldown_active,
                consecutive_high_risk=high_risk_count,
                consecutive_normal=normal_count,
            )

        # 3. EVALUATE HARD SAFETY THRESHOLDS (Deterministic safety bounds)
        is_severe, severe_reasons = evaluate_hard_safety_thresholds(metric)

        # 4. RECOVERY LOGIC (When already in OPTIMIZED or RECOVERY state)
        if current_state in (ControllerState.OPTIMIZED, ControllerState.RECOVERY):
            if is_severe or risk_signal == RiskSignal.CRITICAL:
                # System is experiencing high stress again, reset recovery counter
                normal_count = 0
                new_state = ControllerState.OPTIMIZED
                update_system_state(
                    controller_state=new_state,
                    consecutive_normal=0,
                    db_path=self.db_path
                )
                blocked_reason = (
                    "System in cooldown; repeated optimization blocked"
                    if cooldown_active
                    else "Already optimized; severe condition or critical risk active"
                )
                return ControllerStepResult(
                    controller_state=new_state,
                    optimization_applied=False,
                    optimization_blocked_reason=blocked_reason,
                    current_configuration=current_config,
                    manual_override=False,
                    cooldown_active=cooldown_active,
                    consecutive_high_risk=high_risk_count,
                    consecutive_normal=0,
                )

            if risk_signal == RiskSignal.NORMAL and not is_severe:
                normal_count += 1
                if normal_count >= self.recovery_confirmations:
                    # Sustained recovery confirmed -> attempt to restore defaults
                    success, restored_config, err = self.optimizer.restore_defaults(
                        reason=f"Sustained normal conditions confirmed ({normal_count}/{self.recovery_confirmations} consecutive readings)",
                        timestamp=current_time
                    )
                    if success:
                        new_state = ControllerState.NORMAL
                        update_system_state(
                            controller_state=new_state,
                            consecutive_normal=0,
                            consecutive_high_risk=0,
                            db_path=self.db_path
                        )
                        return ControllerStepResult(
                            controller_state=new_state,
                            optimization_applied=False,
                            optimization_blocked_reason=None,
                            current_configuration=restored_config,
                            manual_override=False,
                            cooldown_active=cooldown_active,
                            consecutive_high_risk=0,
                            consecutive_normal=0,
                        )
                    else:
                        # Recovery failed: do NOT transition to NORMAL and do NOT claim success.
                        # Preserve the actual optimized configuration and report failure reason.
                        new_state = ControllerState.RECOVERY
                        update_system_state(
                            controller_state=new_state,
                            consecutive_normal=normal_count,
                            db_path=self.db_path
                        )
                        blocked_msg = f"Recovery failed: {err}" if err else "Recovery failed to restore default configuration"
                        return ControllerStepResult(
                            controller_state=new_state,
                            optimization_applied=False,
                            optimization_blocked_reason=blocked_msg,
                            current_configuration=restored_config,
                            manual_override=False,
                            cooldown_active=cooldown_active,
                            consecutive_high_risk=0,
                            consecutive_normal=normal_count,
                        )
                else:
                    # Incrementing recovery confirmation
                    new_state = ControllerState.RECOVERY
                    update_system_state(
                        controller_state=new_state,
                        consecutive_normal=normal_count,
                        db_path=self.db_path
                    )
                    return ControllerStepResult(
                        controller_state=new_state,
                        optimization_applied=False,
                        optimization_blocked_reason=f"Recovery in progress ({normal_count}/{self.recovery_confirmations} normal readings)",
                        current_configuration=current_config,
                        manual_override=False,
                        cooldown_active=cooldown_active,
                        consecutive_high_risk=0,
                        consecutive_normal=normal_count,
                    )

            if risk_signal == RiskSignal.ELEVATED:
                # Elevated during recovery halts recovery progression
                normal_count = 0
                new_state = ControllerState.OPTIMIZED
                update_system_state(
                    controller_state=new_state,
                    consecutive_normal=0,
                    db_path=self.db_path
                )
                return ControllerStepResult(
                    controller_state=new_state,
                    optimization_applied=False,
                    optimization_blocked_reason="Recovery halted; risk signal elevated",
                    current_configuration=current_config,
                    manual_override=False,
                    cooldown_active=cooldown_active,
                    consecutive_high_risk=0,
                    consecutive_normal=0,
                )

        # 5. NORMAL / WATCHING STATE LOGIC (System not currently optimized)

        # Scenario A: Hard Safety Threshold Breached (Immediate safety trigger)
        if is_severe:
            reason_str = f"Hard safety threshold breached: {'; '.join(severe_reasons)}"
            if cooldown_active:
                return ControllerStepResult(
                    controller_state=current_state,
                    optimization_applied=False,
                    optimization_blocked_reason="System in cooldown; repeated optimization blocked",
                    current_configuration=current_config,
                    manual_override=False,
                    cooldown_active=True,
                    consecutive_high_risk=high_risk_count,
                    consecutive_normal=0,
                )

            # Apply immediate optimization for safety
            success, new_config, err = self.optimizer.apply_optimizations(
                reason=reason_str, timestamp=current_time
            )
            new_state = ControllerState.OPTIMIZED if success else current_state
            update_system_state(
                controller_state=new_state,
                consecutive_high_risk=0,
                consecutive_normal=0,
                db_path=self.db_path
            )
            return ControllerStepResult(
                controller_state=new_state,
                optimization_applied=success,
                optimization_blocked_reason=err if not success else None,
                current_configuration=new_config,
                manual_override=False,
                cooldown_active=False,
                consecutive_high_risk=0,
                consecutive_normal=0,
            )

        # Scenario B: Critical Risk Signal (Count toward High-Risk Confirmations)
        if risk_signal == RiskSignal.CRITICAL:
            high_risk_count += 1
            if high_risk_count >= self.high_risk_confirmations:
                if cooldown_active:
                    update_system_state(
                        controller_state=ControllerState.WATCHING,
                        consecutive_high_risk=high_risk_count,
                        db_path=self.db_path
                    )
                    return ControllerStepResult(
                        controller_state=ControllerState.WATCHING,
                        optimization_applied=False,
                        optimization_blocked_reason="System in cooldown; repeated optimization blocked",
                        current_configuration=current_config,
                        manual_override=False,
                        cooldown_active=True,
                        consecutive_high_risk=high_risk_count,
                        consecutive_normal=0,
                    )

                # Confirmed high risk -> apply optimization
                reason_str = f"High-risk confirmed ({high_risk_count}/{self.high_risk_confirmations} consecutive readings)"
                success, new_config, err = self.optimizer.apply_optimizations(
                    reason=reason_str, timestamp=current_time
                )
                new_state = ControllerState.OPTIMIZED if success else current_state
                update_system_state(
                    controller_state=new_state,
                    consecutive_high_risk=0,
                    consecutive_normal=0,
                    db_path=self.db_path
                )
                return ControllerStepResult(
                    controller_state=new_state,
                    optimization_applied=success,
                    optimization_blocked_reason=err if not success else None,
                    current_configuration=new_config,
                    manual_override=False,
                    cooldown_active=False,
                    consecutive_high_risk=0,
                    consecutive_normal=0,
                )
            else:
                # Need more confirmations -> enter/remain in WATCHING
                new_state = ControllerState.WATCHING
                update_system_state(
                    controller_state=new_state,
                    consecutive_high_risk=high_risk_count,
                    db_path=self.db_path
                )
                return ControllerStepResult(
                    controller_state=new_state,
                    optimization_applied=False,
                    optimization_blocked_reason=f"High risk detected ({high_risk_count}/{self.high_risk_confirmations} confirmations); awaiting sustained signal",
                    current_configuration=current_config,
                    manual_override=False,
                    cooldown_active=cooldown_active,
                    consecutive_high_risk=high_risk_count,
                    consecutive_normal=0,
                )

        # Scenario C: Elevated Risk Signal (Move to WATCHING, do not optimize yet)
        if risk_signal == RiskSignal.ELEVATED:
            new_state = ControllerState.WATCHING
            update_system_state(
                controller_state=new_state,
                consecutive_normal=0,
                db_path=self.db_path
            )
            return ControllerStepResult(
                controller_state=new_state,
                optimization_applied=False,
                optimization_blocked_reason="Elevated risk detected; watching system health",
                current_configuration=current_config,
                manual_override=False,
                cooldown_active=cooldown_active,
                consecutive_high_risk=high_risk_count,
                consecutive_normal=0,
            )

        # Scenario D: Normal Signal (Reset high-risk count, move to NORMAL if in WATCHING)
        high_risk_count = 0
        new_state = ControllerState.NORMAL
        update_system_state(
            controller_state=new_state,
            consecutive_high_risk=0,
            consecutive_normal=0,
            db_path=self.db_path
        )
        return ControllerStepResult(
            controller_state=new_state,
            optimization_applied=False,
            optimization_blocked_reason=None,
            current_configuration=current_config,
            manual_override=False,
            cooldown_active=cooldown_active,
            consecutive_high_risk=0,
            consecutive_normal=0,
        )
