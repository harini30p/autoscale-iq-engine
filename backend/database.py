"""
SQLite database persistence layer for AutoScale IQ.
Manages metrics, future predictions placeholder, optimization events, and system state.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import (
    DEFAULT_DB_PATH,
    CACHE_NORMAL,
    PAGINATION_NORMAL,
    HEAVY_COMPONENTS_NORMAL,
)
from backend.schemas import MetricSchema, ControllerState, SystemConfiguration


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a connection to SQLite database with Row row_factory."""
    # Ensure directory exists if path contains directories
    db_file = Path(db_path)
    if db_file.parent and not db_file.parent.exists() and str(db_file.parent) != "":
        db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize database tables and default system state."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                traffic REAL NOT NULL,
                active_users INTEGER NOT NULL,
                cpu_utilization REAL NOT NULL,
                memory_utilization REAL NOT NULL,
                response_time REAL NOT NULL,
                db_query_time REAL NOT NULL,
                system_load REAL NOT NULL
            )
        """)

        # ML predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                probability REAL NOT NULL,
                risk_signal TEXT NOT NULL,
                watch_threshold REAL NOT NULL,
                critical_threshold REAL NOT NULL,
                feature_snapshot TEXT NOT NULL
            )
        """)

        # Optimization events log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                previous_state TEXT NOT NULL,
                new_state TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT
            )
        """)

        # Backward-compatible migration: add metric snapshot columns if missing.
        # These columns hold JSON and are NULL for events created before this migration.
        for col_def in [
            "before_metrics_json TEXT",
            "after_metrics_json TEXT",
        ]:
            col_name = col_def.split()[0]
            try:
                cursor.execute(
                    f"ALTER TABLE optimization_events ADD COLUMN {col_def}"
                )
            except Exception:
                # Column already exists — safe to ignore
                pass

        # System state single-row table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                controller_state TEXT NOT NULL DEFAULT 'NORMAL',
                caching TEXT NOT NULL DEFAULT 'disabled',
                pagination_size INTEGER NOT NULL DEFAULT 50,
                heavy_components TEXT NOT NULL DEFAULT 'enabled',
                manual_override INTEGER NOT NULL DEFAULT 0,
                consecutive_high_risk INTEGER NOT NULL DEFAULT 0,
                consecutive_normal INTEGER NOT NULL DEFAULT 0,
                last_optimization_timestamp TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        # Insert default system state if not present
        cursor.execute("SELECT COUNT(*) as cnt FROM system_state WHERE id = 1")
        row = cursor.fetchone()
        if row and row["cnt"] == 0:
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                INSERT INTO system_state (
                    id, controller_state, caching, pagination_size, heavy_components,
                    manual_override, consecutive_high_risk, consecutive_normal,
                    last_optimization_timestamp, updated_at
                ) VALUES (1, ?, ?, ?, ?, 0, 0, 0, NULL, ?)
            """, (
                ControllerState.NORMAL.value,
                CACHE_NORMAL,
                PAGINATION_NORMAL,
                HEAVY_COMPONENTS_NORMAL,
                now_iso
            ))
        conn.commit()


def insert_metric(metric: MetricSchema, db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert a validated metric record into SQLite."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO metrics (
                timestamp, traffic, active_users, cpu_utilization,
                memory_utilization, response_time, db_query_time, system_load
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.timestamp.isoformat(),
            metric.traffic,
            metric.active_users,
            metric.cpu_utilization,
            metric.memory_utilization,
            metric.response_time,
            metric.db_query_time,
            metric.system_load
        ))
        conn.commit()
        return cursor.lastrowid or 0


def record_event(
    action: str,
    reason: str,
    previous_state: Dict[str, Any],
    new_state: Dict[str, Any],
    success: bool,
    error_message: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Record an optimization or recovery event."""
    init_db(db_path)
    ts = timestamp or datetime.now(timezone.utc)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO optimization_events (
                timestamp, action, reason, previous_state, new_state, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ts.isoformat(),
            action,
            reason,
            json.dumps(previous_state),
            json.dumps(new_state),
            1 if success else 0,
            error_message
        ))
        conn.commit()
        return cursor.lastrowid or 0


def get_system_state(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieve the current persistent system state."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM system_state WHERE id = 1")
        row = cursor.fetchone()
        if not row:
            raise RuntimeError("System state table is not initialized")
        
        last_opt_ts = None
        if row["last_optimization_timestamp"]:
            try:
                last_opt_ts = datetime.fromisoformat(row["last_optimization_timestamp"])
            except ValueError:
                last_opt_ts = None

        return {
            "controller_state": ControllerState(row["controller_state"]),
            "caching": row["caching"],
            "pagination_size": row["pagination_size"],
            "heavy_components": row["heavy_components"],
            "manual_override": bool(row["manual_override"]),
            "consecutive_high_risk": row["consecutive_high_risk"],
            "consecutive_normal": row["consecutive_normal"],
            "last_optimization_timestamp": last_opt_ts,
            "updated_at": datetime.fromisoformat(row["updated_at"])
        }


def update_system_state(
    controller_state: Optional[ControllerState] = None,
    caching: Optional[str] = None,
    pagination_size: Optional[int] = None,
    heavy_components: Optional[str] = None,
    manual_override: Optional[bool] = None,
    consecutive_high_risk: Optional[int] = None,
    consecutive_normal: Optional[int] = None,
    last_optimization_timestamp: Optional[datetime] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Update system state fields atomically in the database."""
    current = get_system_state(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    new_controller_state = controller_state.value if controller_state else current["controller_state"].value
    new_caching = caching if caching is not None else current["caching"]
    new_pagination_size = pagination_size if pagination_size is not None else current["pagination_size"]
    new_heavy_components = heavy_components if heavy_components is not None else current["heavy_components"]
    new_manual_override = 1 if (manual_override if manual_override is not None else current["manual_override"]) else 0
    new_consecutive_high_risk = consecutive_high_risk if consecutive_high_risk is not None else current["consecutive_high_risk"]
    new_consecutive_normal = consecutive_normal if consecutive_normal is not None else current["consecutive_normal"]
    
    if last_optimization_timestamp is not None:
        new_last_opt = last_optimization_timestamp.isoformat()
    elif current["last_optimization_timestamp"]:
        new_last_opt = current["last_optimization_timestamp"].isoformat()
    else:
        new_last_opt = None

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE system_state
            SET controller_state = ?,
                caching = ?,
                pagination_size = ?,
                heavy_components = ?,
                manual_override = ?,
                consecutive_high_risk = ?,
                consecutive_normal = ?,
                last_optimization_timestamp = ?,
                updated_at = ?
            WHERE id = 1
        """, (
            new_controller_state,
            new_caching,
            new_pagination_size,
            new_heavy_components,
            new_manual_override,
            new_consecutive_high_risk,
            new_consecutive_normal,
            new_last_opt,
            now_iso
        ))
        conn.commit()

    return get_system_state(db_path)


def _compute_impact(before: Optional[Dict], after: Optional[Dict]) -> Optional[Dict]:
    """
    Compute improvement percentages between before and after metric snapshots.
    For response_time, cpu_utilization, memory_utilization, db_query_time,
    lower-is-better: improvement = ((before - after) / before) * 100.
    Returns None if before or after is absent.
    """
    if not before or not after:
        return None
    impact: Dict[str, Any] = {}
    for field in ("response_time", "cpu_utilization", "memory_utilization", "db_query_time"):
        b = before.get(field)
        a = after.get(field)
        if b is None or a is None:
            impact[field] = None
            continue
        try:
            b_f = float(b)
            a_f = float(a)
            if b_f == 0.0:
                impact[field] = None  # Avoid division-by-zero
            else:
                impact[field] = round(((b_f - a_f) / b_f) * 100, 2)
        except (TypeError, ValueError):
            impact[field] = None
    return impact


def get_recent_events(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve recent optimization/recovery events ordered by latest first.
    Includes optional before_metrics, after_metrics, and impact fields.
    Old records without snapshots return None for these fields.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # SELECT * to include the optional snapshot columns added by migration
        cursor.execute("""
            SELECT id, timestamp, action, reason, previous_state, new_state,
                   success, error_message, before_metrics_json, after_metrics_json
            FROM optimization_events
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        events = []
        for r in rows:
            before_m = None
            after_m = None
            try:
                if r["before_metrics_json"]:
                    before_m = json.loads(r["before_metrics_json"])
            except Exception:
                pass
            try:
                if r["after_metrics_json"]:
                    after_m = json.loads(r["after_metrics_json"])
            except Exception:
                pass
            events.append({
                "id": r["id"],
                "timestamp": datetime.fromisoformat(r["timestamp"]),
                "action": r["action"],
                "reason": r["reason"],
                "previous_state": json.loads(r["previous_state"]),
                "new_state": json.loads(r["new_state"]),
                "success": bool(r["success"]),
                "error_message": r["error_message"],
                "before_metrics": before_m,
                "after_metrics": after_m,
                "impact": _compute_impact(before_m, after_m),
            })
        return events


def attach_before_metrics_to_latest_event(
    before_metrics: Dict[str, Any],
    action: str = "apply_optimizations",
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Attach a before-metrics snapshot to the most recent matching event.
    Only updates an event that does not already have before_metrics set.
    Returns True if an update was made.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM optimization_events
            WHERE action = ? AND before_metrics_json IS NULL
            ORDER BY id DESC LIMIT 1
        """, (action,))
        row = cursor.fetchone()
        if not row:
            return False
        cursor.execute(
            "UPDATE optimization_events SET before_metrics_json = ? WHERE id = ?",
            (json.dumps(before_metrics), row["id"])
        )
        conn.commit()
        return cursor.rowcount > 0


def attach_after_metrics_to_latest_event(
    after_metrics: Dict[str, Any],
    action: str = "apply_optimizations",
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Attach an after-metrics snapshot to the most recent apply_optimizations event
    that already has before_metrics but does NOT yet have after_metrics.
    Returns True if an update was made (i.e., this is the first after reading).
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM optimization_events
            WHERE action = ? AND before_metrics_json IS NOT NULL AND after_metrics_json IS NULL
            ORDER BY id DESC LIMIT 1
        """, (action,))
        row = cursor.fetchone()
        if not row:
            return False
        cursor.execute(
            "UPDATE optimization_events SET after_metrics_json = ? WHERE id = ?",
            (json.dumps(after_metrics), row["id"])
        )
        conn.commit()
        return cursor.rowcount > 0


def get_recent_metrics(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve recent recorded metrics ordered chronologically (oldest first)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, traffic, active_users, cpu_utilization,
                   memory_utilization, response_time, db_query_time, system_load
            FROM metrics
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        metrics = []
        for r in reversed(rows):
            ts = r["timestamp"]
            try:
                ts_dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                ts_dt = ts
            metrics.append({
                "id": r["id"],
                "timestamp": ts_dt,
                "traffic": r["traffic"],
                "active_users": r["active_users"],
                "cpu_utilization": r["cpu_utilization"],
                "memory_utilization": r["memory_utilization"],
                "response_time": r["response_time"],
                "db_query_time": r["db_query_time"],
                "system_load": r["system_load"],
            })
        return metrics


def insert_ml_prediction(
    probability: float,
    risk_signal: str,
    watch_threshold: float,
    critical_threshold: float,
    feature_snapshot: Any,
    timestamp: Optional[datetime] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Insert an ML surge prediction record into SQLite."""
    init_db(db_path)
    ts = timestamp or datetime.now(timezone.utc)
    ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
    snapshot_str = json.dumps(feature_snapshot) if not isinstance(feature_snapshot, str) else feature_snapshot

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ml_predictions (
                timestamp, probability, risk_signal,
                watch_threshold, critical_threshold, feature_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            ts_str,
            float(probability),
            str(risk_signal),
            float(watch_threshold),
            float(critical_threshold),
            snapshot_str,
        ))
        conn.commit()
        return cursor.lastrowid or 0


def get_recent_predictions(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve recent ML predictions ordered chronologically (oldest first)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, probability, risk_signal,
                   watch_threshold, critical_threshold, feature_snapshot
            FROM ml_predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        predictions = []
        for r in reversed(rows):
            snapshot = r["feature_snapshot"]
            try:
                snapshot = json.loads(snapshot)
            except Exception:
                pass
            ts = r["timestamp"]
            try:
                ts_dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                ts_dt = ts
            predictions.append({
                "id": r["id"],
                "timestamp": ts_dt,
                "probability": r["probability"],
                "risk_signal": r["risk_signal"],
                "watch_threshold": r["watch_threshold"],
                "critical_threshold": r["critical_threshold"],
                "feature_snapshot": snapshot,
            })
        return predictions
