"""
Configuration settings for AutoScale IQ Backend Safety Foundation.
Centralizes safety thresholds, confirmation counts, and database parameters.
"""

from pathlib import Path
import os

# Base directory for the backend
BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DEFAULT_DB_PATH = os.getenv("AUTOSCALE_DB_PATH", str(BASE_DIR / "autoscale.db"))

# Monitoring freshness settings
DEFAULT_STALE_THRESHOLD_SECONDS: float = 10.0

# Safety controller settings
DEFAULT_HIGH_RISK_CONFIRMATIONS: int = 3
DEFAULT_RECOVERY_CONFIRMATIONS: int = 3
DEFAULT_COOLDOWN_SECONDS: float = 30.0

# Hard safety thresholds (Safety rules, not ML predictions)
# Severe conditions breaching these thresholds trigger immediate optimization or escalation
CPU_CRITICAL_THRESHOLD: float = 90.0          # CPU % >= 90 is critical
MEMORY_CRITICAL_THRESHOLD: float = 90.0       # Memory % >= 90 is critical
RESPONSE_TIME_CRITICAL_THRESHOLD: float = 1000.0  # Response time >= 1000ms is critical
SYSTEM_LOAD_CRITICAL_THRESHOLD: float = 5.0   # System load indicator >= 5.0 is critical

# Application optimization configuration states
CACHE_NORMAL = "disabled"
CACHE_OPTIMIZED = "enabled"

PAGINATION_NORMAL = 50
PAGINATION_OPTIMIZED = 20

HEAVY_COMPONENTS_NORMAL = "enabled"
HEAVY_COMPONENTS_OPTIMIZED = "disabled"
