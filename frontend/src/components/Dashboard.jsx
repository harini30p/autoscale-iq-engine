import React, { useState, useEffect, useCallback } from 'react';
import Header from './Header';
import SystemStatusBanner from './SystemStatusBanner';
import MetricCardsGrid from './MetricCardsGrid';
import MLSurgePredictorPanel from './MLSurgePredictorPanel';
import EventsTable from './EventsTable';
import {
  checkHealth,
  getSystemState,
  getRecentEvents,
  setManualOverride,
  sendMonitorMetric,
} from '../services/api';

export default function Dashboard() {
  const [backendConnected, setBackendConnected] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [lastHealthCheck, setLastHealthCheck] = useState(null);
  const [systemState, setSystemState] = useState(null);
  const [events, setEvents] = useState([]);
  const [latestMetric, setLatestMetric] = useState(null);
  const [mlPrediction, setMlPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [overrideUpdating, setOverrideUpdating] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [simulating, setSimulating] = useState(false);

  // Fetch live system state, health, and events
  const fetchData = useCallback(async () => {
    setIsChecking(true);
    try {
      // 1. Check health
      const healthRes = await checkHealth();
      setBackendConnected(healthRes?.status === 'ok');
      setLastHealthCheck(healthRes?.timestamp || new Date().toISOString());

      // 2. Fetch system state
      const stateRes = await getSystemState();
      setSystemState(stateRes);

      // 3. Fetch recent events
      const eventsRes = await getRecentEvents(20);
      setEvents(Array.isArray(eventsRes) ? eventsRes : []);

      setError(null);
    } catch (err) {
      setBackendConnected(false);
      setError(err.message || 'Failed to connect to AutoScale IQ backend');
    } finally {
      setIsChecking(false);
    }
  }, []);

  // Initial load and periodic polling
  useEffect(() => {
    fetchData();

    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchData();
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchData, autoRefresh]);

  // Toggle manual override
  const handleToggleOverride = async (enableOverride) => {
    setOverrideUpdating(true);
    try {
      const updatedState = await setManualOverride(enableOverride);
      setSystemState(updatedState);
      setError(null);
    } catch (err) {
      setError(`Failed to update manual override: ${err.message}`);
    } finally {
      setOverrideUpdating(false);
    }
  };

  // Optional sample observation simulation to test live end-to-end integration
  const handleSimulateSampleObservation = async (spike = false) => {
    setSimulating(true);
    try {
      const samplePayload = {
        metric: {
          timestamp: new Date().toISOString(),
          traffic: spike ? 550.0 : 65.0,
          active_users: spike ? 120 : 15,
          cpu_utilization: spike ? 72.0 : 28.0,
          memory_utilization: spike ? 65.0 : 35.0,
          response_time: spike ? 240.0 : 85.0,
          db_query_time: spike ? 45.0 : 18.0,
          system_load: spike ? 1.8 : 0.6,
        },
        risk_signal: 'normal',
      };

      const monitorRes = await sendMonitorMetric(samplePayload);
      setLatestMetric(samplePayload.metric);
      if (monitorRes?.ml_prediction) {
        setMlPrediction(monitorRes.ml_prediction);
      }
      // Refresh system state and events
      await fetchData();
    } catch (err) {
      setError(`Simulation failed: ${err.message}`);
    } finally {
      setSimulating(false);
    }
  };

  return (
    <div className="dashboard-container">
      <Header
        backendConnected={backendConnected}
        isChecking={isChecking}
        lastHealthCheck={lastHealthCheck}
        onRefresh={fetchData}
      />

      <main className="dashboard-main">
        {error && (
          <div className="alert-banner alert-danger">
            <span className="alert-icon">⚠️</span>
            <div className="alert-content">
              <strong>Backend Connection Issue:</strong> {error}
            </div>
            <button
              type="button"
              className="btn btn-sm btn-outline"
              onClick={fetchData}
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* System State Banner */}
        <SystemStatusBanner
          systemState={systemState}
          loading={isChecking}
          error={error}
          onToggleOverride={handleToggleOverride}
          overrideUpdating={overrideUpdating}
        />

        {/* Main Observability Grid: Telemetry & ML Prediction */}
        <div className="observability-row">
          <MetricCardsGrid latestMetric={latestMetric} />
        </div>

        <div className="observability-row">
          <MLSurgePredictorPanel
            mlPrediction={mlPrediction}
            controllerState={systemState?.controller_state}
          />
        </div>

        {/* Interactive Telemetry Test Bar */}
        <div className="card test-bar-card">
          <div className="test-bar-left">
            <span className="test-bar-title">🧪 End-to-End Observation Simulator</span>
            <span className="test-bar-desc">
              Send a test metric observation through <code>/monitor</code> to observe live feature generation, ML calibration, and controller evaluations.
            </span>
          </div>
          <div className="test-bar-actions">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => handleSimulateSampleObservation(false)}
              disabled={simulating || !backendConnected}
            >
              {simulating ? 'Sending...' : 'Send Baseline Metric (65 req/min)'}
            </button>
            <button
              type="button"
              className="btn btn-accent btn-sm"
              onClick={() => handleSimulateSampleObservation(true)}
              disabled={simulating || !backendConnected}
            >
              {simulating ? 'Sending...' : '⚡ Send Surge Spike (550 req/min)'}
            </button>
            <label className="auto-refresh-toggle">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-poll (5s)
            </label>
          </div>
        </div>

        {/* Events Audit Table */}
        <EventsTable
          events={events}
          loading={isChecking}
          error={error}
        />
      </main>

      <footer className="app-footer">
        <span>AutoScale IQ Engine • ML-Integrated Safety Controller v0.2.0</span>
        <span>HistGradientBoostingClassifier 5M • Isotonic Calibration • Watch: 0.040 | Crit: 0.075</span>
      </footer>
    </div>
  );
}
