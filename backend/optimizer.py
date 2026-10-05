"""
Application Optimizer for AutoScale IQ.
Manages application-level optimizations (caching, pagination, heavy components)
and persists configuration changes and event logs safely.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional, Tuple

from backend.config import (
    DEFAULT_DB_PATH,
    CACHE_NORMAL,
    CACHE_OPTIMIZED,
    PAGINATION_NORMAL,
    PAGINATION_OPTIMIZED,
    HEAVY_COMPONENTS_NORMAL,
    HEAVY_COMPONENTS_OPTIMIZED,
)
from backend.database import get_system_state, update_system_state, record_event
from backend.schemas import SystemConfiguration

logger = logging.getLogger("autoscale_iq.optimizer")


class ApplicationOptimizer:
    """
    Manages application-level optimization levers:
    - Caching: disabled -> enabled
    - Pagination: 50 -> 20 items/page
    - Heavy/Nonessential components: enabled -> disabled
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    def get_configuration(self) -> SystemConfiguration:
        """Retrieve current active system configuration."""
        state = get_system_state(self.db_path)
        return SystemConfiguration(
            caching=state["caching"],
            pagination_size=state["pagination_size"],
            heavy_components=state["heavy_components"]
        )

    def is_already_optimized(self, config: Optional[SystemConfiguration] = None) -> bool:
        """Check if all optimization levers are already in optimized state."""
        cfg = config or self.get_configuration()
        return (
            cfg.caching == CACHE_OPTIMIZED and
            cfg.pagination_size == PAGINATION_OPTIMIZED and
            cfg.heavy_components == HEAVY_COMPONENTS_OPTIMIZED
        )

    def is_default_state(self, config: Optional[SystemConfiguration] = None) -> bool:
        """Check if all configuration levers are in normal default state."""
        cfg = config or self.get_configuration()
        return (
            cfg.caching == CACHE_NORMAL and
            cfg.pagination_size == PAGINATION_NORMAL and
            cfg.heavy_components == HEAVY_COMPONENTS_NORMAL
        )

    def apply_optimizations(
        self,
        reason: str,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, SystemConfiguration, Optional[str]]:
        """
        Apply application-level optimizations (idempotent, safe).
        Returns (success: bool, configuration: SystemConfiguration, error_message: Optional[str]).
        """
        now = timestamp or datetime.now(timezone.utc)
        current_cfg = self.get_configuration()

        if self.is_already_optimized(current_cfg):
            # Already optimized, idempotent no-op
            return True, current_cfg, None

        previous_state = current_cfg.model_dump()
        target_state = {
            "caching": CACHE_OPTIMIZED,
            "pagination_size": PAGINATION_OPTIMIZED,
            "heavy_components": HEAVY_COMPONENTS_OPTIMIZED
        }

        try:
            # Perform configuration update in DB
            update_system_state(
                caching=CACHE_OPTIMIZED,
                pagination_size=PAGINATION_OPTIMIZED,
                heavy_components=HEAVY_COMPONENTS_OPTIMIZED,
                last_optimization_timestamp=now,
                db_path=self.db_path
            )

            # Record successful event log
            record_event(
                action="apply_optimizations",
                reason=reason,
                previous_state=previous_state,
                new_state=target_state,
                success=True,
                error_message=None,
                timestamp=now,
                db_path=self.db_path
            )

            new_cfg = self.get_configuration()
            return True, new_cfg, None

        except Exception as ex:
            error_msg = f"Failed to apply optimizations: {str(ex)}"
            logger.error(error_msg, exc_info=True)
            # Record failure in event log
            try:
                record_event(
                    action="apply_optimizations",
                    reason=reason,
                    previous_state=previous_state,
                    new_state=previous_state,  # state remains unchanged
                    success=False,
                    error_message=error_msg,
                    timestamp=now,
                    db_path=self.db_path
                )
            except Exception as log_ex:
                logger.error(f"Failed to record failure event: {log_ex}")

            return False, current_cfg, error_msg

    def restore_defaults(
        self,
        reason: str,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, SystemConfiguration, Optional[str]]:
        """
        Restore default application configuration (idempotent, safe).
        Returns (success: bool, configuration: SystemConfiguration, error_message: Optional[str]).
        """
        now = timestamp or datetime.now(timezone.utc)
        current_cfg = self.get_configuration()

        if self.is_default_state(current_cfg):
            # Already in default configuration, idempotent no-op
            return True, current_cfg, None

        previous_state = current_cfg.model_dump()
        target_state = {
            "caching": CACHE_NORMAL,
            "pagination_size": PAGINATION_NORMAL,
            "heavy_components": HEAVY_COMPONENTS_NORMAL
        }

        try:
            # Update configuration in DB
            update_system_state(
                caching=CACHE_NORMAL,
                pagination_size=PAGINATION_NORMAL,
                heavy_components=HEAVY_COMPONENTS_NORMAL,
                db_path=self.db_path
            )

            # Record successful recovery event log
            record_event(
                action="restore_defaults",
                reason=reason,
                previous_state=previous_state,
                new_state=target_state,
                success=True,
                error_message=None,
                timestamp=now,
                db_path=self.db_path
            )

            new_cfg = self.get_configuration()
            return True, new_cfg, None

        except Exception as ex:
            error_msg = f"Failed to restore default configuration: {str(ex)}"
            logger.error(error_msg, exc_info=True)
            try:
                record_event(
                    action="restore_defaults",
                    reason=reason,
                    previous_state=previous_state,
                    new_state=previous_state,
                    success=False,
                    error_message=error_msg,
                    timestamp=now,
                    db_path=self.db_path
                )
            except Exception as log_ex:
                logger.error(f"Failed to record failure event: {log_ex}")

            return False, current_cfg, error_msg
