"""
Verification script for Milestone 1: Backend Safety Foundation.
Performs end-to-end verification of:
1. /health
2. /docs
3. Valid /monitor request
4. Stale metric behavior
5. 3x high-risk confirmation behavior
6. 3x recovery hysteresis behavior
7. Manual override behavior
8. SQLite database persistence
"""

from datetime import datetime, timezone, timedelta
import tempfile
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.database import get_recent_events, get_recent_metrics, get_system_state
from backend.schemas import ControllerState, RiskSignal


def run_verification(tmp_path_str: str):
    db_file = f"{tmp_path_str}/verification_autoscale.db"
    print(f"[*] Initializing verification app with DB: {db_file}")
    app = create_app(db_path=db_file)
    with TestClient(app) as client:
        # 1. Health check
        print("\n--- 1. Testing GET /health ---")
        res = client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        print(f"[+] Health Response: {res.json()}")

        # 2. OpenAPI /docs
        print("\n--- 2. Testing GET /docs and /openapi.json ---")
        res_docs = client.get("/docs")
        assert res_docs.status_code == 200, f"Expected 200, got {res_docs.status_code}"
        res_openapi = client.get("/openapi.json")
        assert res_openapi.status_code == 200, f"Expected 200, got {res_openapi.status_code}"
        print(f"[+] Docs verified. OpenAPI endpoints count: {len(res_openapi.json().get('paths', {}))}")

        # 3. Valid /monitor request
        print("\n--- 3. Testing Valid POST /monitor ---")
        now = datetime.now(timezone.utc)
        fresh_payload = {
            "metric": {
                "timestamp": now.isoformat(),
                "traffic": 300.0,
                "active_users": 60,
                "cpu_utilization": 55.0,
                "memory_utilization": 50.0,
                "response_time": 150.0,
                "db_query_time": 35.0,
                "system_load": 1.5
            },
            "risk_signal": "normal"
        }
        res_mon = client.post("/monitor", json=fresh_payload)
        assert res_mon.status_code == 200, f"Expected 200, got {res_mon.status_code}: {res_mon.text}"
        mon_data = res_mon.json()
        print(f"[+] Valid Monitor Response: controller_state={mon_data['controller_state']}, stale={mon_data['stale']}, config={mon_data['current_configuration']}")
        assert mon_data["metric_valid"] is True
        assert mon_data["stale"] is False
        assert mon_data["controller_state"] == "NORMAL"

        # 4. Stale metric request
        print("\n--- 4. Testing Stale POST /monitor ---")
        stale_time = now - timedelta(seconds=25)
        stale_payload = {
            "metric": {
                "timestamp": stale_time.isoformat(),
                "traffic": 300.0,
                "active_users": 60,
                "cpu_utilization": 95.0,
                "memory_utilization": 50.0,
                "response_time": 150.0,
                "db_query_time": 35.0,
                "system_load": 1.5
            },
            "risk_signal": "critical"
        }
        res_stale = client.post("/monitor", json=stale_payload)
        assert res_stale.status_code == 200
        stale_data = res_stale.json()
        print(f"[+] Stale Monitor Response: stale={stale_data['stale']}, optimization_applied={stale_data['optimization_applied']}, reason={stale_data['optimization_blocked_reason']}")
        assert stale_data["stale"] is True
        assert stale_data["optimization_applied"] is False

        # 5. 3-reading High Risk Confirmation
        print("\n--- 5. Testing 3x High-Risk Confirmation ---")
        for step in range(1, 4):
            step_now = datetime.now(timezone.utc)
            payload = {
                "metric": {
                    "timestamp": step_now.isoformat(),
                    "traffic": 400.0,
                    "active_users": 80,
                    "cpu_utilization": 70.0,
                    "memory_utilization": 70.0,
                    "response_time": 200.0,
                    "db_query_time": 40.0,
                    "system_load": 2.0
                },
                "risk_signal": "critical"
            }
            res_step = client.post("/monitor", json=payload)
            data_step = res_step.json()
            print(f"    Reading {step}: controller_state={data_step['controller_state']}, opt_applied={data_step['optimization_applied']}, consecutive_high_risk={data_step['consecutive_high_risk']}")
            if step < 3:
                assert data_step["controller_state"] == "WATCHING"
                assert data_step["optimization_applied"] is False
            else:
                assert data_step["controller_state"] == "OPTIMIZED"
                assert data_step["optimization_applied"] is True
                assert data_step["current_configuration"]["caching"] == "enabled"
                assert data_step["current_configuration"]["pagination_size"] == 20
                assert data_step["current_configuration"]["heavy_components"] == "disabled"

        # 6. 3-reading Recovery Hysteresis
        print("\n--- 6. Testing 3x Recovery Hysteresis ---")
        for step in range(1, 4):
            step_now = datetime.now(timezone.utc)
            payload = {
                "metric": {
                    "timestamp": step_now.isoformat(),
                    "traffic": 100.0,
                    "active_users": 20,
                    "cpu_utilization": 30.0,
                    "memory_utilization": 30.0,
                    "response_time": 80.0,
                    "db_query_time": 15.0,
                    "system_load": 0.8
                },
                "risk_signal": "normal"
            }
            res_rec = client.post("/monitor", json=payload)
            data_rec = res_rec.json()
            print(f"    Recovery reading {step}: controller_state={data_rec['controller_state']}, consecutive_normal={data_rec['consecutive_normal']}, caching={data_rec['current_configuration']['caching']}")
            if step < 3:
                assert data_rec["controller_state"] == "RECOVERY"
                assert data_rec["current_configuration"]["caching"] == "enabled"
            else:
                assert data_rec["controller_state"] == "NORMAL"
                assert data_rec["current_configuration"]["caching"] == "disabled"
                assert data_rec["current_configuration"]["pagination_size"] == 50
                assert data_rec["current_configuration"]["heavy_components"] == "enabled"

        # 7. Manual Override
        print("\n--- 7. Testing Manual Override ---")
        res_ovr = client.post("/override", json={"manual_override": True})
        assert res_ovr.status_code == 200
        assert res_ovr.json()["manual_override"] is True
        print(f"[+] Manual override enabled: {res_ovr.json()['manual_override']}")

        # Critical request under override
        res_ovr_mon = client.post("/monitor", json={
            "metric": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "traffic": 500.0,
                "active_users": 100,
                "cpu_utilization": 99.0,
                "memory_utilization": 99.0,
                "response_time": 2000.0,
                "db_query_time": 500.0,
                "system_load": 10.0
            },
            "risk_signal": "critical"
        })
        data_ovr_mon = res_ovr_mon.json()
        print(f"[+] Under override: opt_applied={data_ovr_mon['optimization_applied']}, reason={data_ovr_mon['optimization_blocked_reason']}")
        assert data_ovr_mon["optimization_applied"] is False
        assert "override" in data_ovr_mon["optimization_blocked_reason"].lower()

        # Disable override
        res_ovr_off = client.post("/override", json={"manual_override": False})
        assert res_ovr_off.json()["manual_override"] is False

        # 8. Database Persistence Check
        print("\n--- 8. Verifying SQLite State and Event Records ---")
        db_state = get_system_state(db_file)
        db_events = get_recent_events(10, db_file)
        db_metrics = get_recent_metrics(10, db_file)

        print(f"[+] Stored system_state: {db_state}")
        print(f"[+] Total stored events in DB: {len(db_events)}")
        for ev in db_events:
            print(f"    Event: action={ev['action']}, success={ev['success']}, reason={ev['reason']}")
        print(f"[+] Total stored metrics in DB: {len(db_metrics)}")

        assert len(db_events) >= 2, "Expected at least 1 optimization event and 1 recovery event"
        assert len(db_metrics) >= 8, "Expected all monitor requests to be persisted in metrics table"
        print("\n[SUCCESS] ALL VERIFICATION STEPS PASSED PERFECTLY!")


if __name__ == "__main__":
    import os
    import shutil
    td = tempfile.mkdtemp()
    try:
        run_verification(td)
    finally:
        try:
            shutil.rmtree(td, ignore_errors=True)
        except Exception:
            pass
