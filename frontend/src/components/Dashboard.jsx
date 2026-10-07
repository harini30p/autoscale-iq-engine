import React, { useState, useEffect, useCallback } from 'react';
import Header from './Header';
import SystemStatusBanner from './SystemStatusBanner';
import MetricCardsGrid from './MetricCardsGrid';
import MLSurgePredictorPanel from './MLSurgePredictorPanel';
import TelemetryChartsPanel from './TelemetryChartsPanel';
import OptimizationImpactPanel from './OptimizationImpactPanel';
import EventsTable from './EventsTable';
import SimulationPanel from './SimulationPanel';
import {
  checkHealth,
  getSystemState,
  getRecentEvents,
  getMetricHistory,
  getPredictionHistory,
  setManualOverride,
} from '../services/api';

export default function Dashboard() {
  const [backendConnected, setBackendConnected] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [lastHealthCheck, setLastHealthCheck] = useState(null);
  const [systemState, setSystemState] = useState(null);
  const [events, setEvents] = useState([]);
  const [metricHistory, setMetricHistory] = useState([]);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [latestMetric, setLatestMetric] = useState(null);
  const [mlPrediction, setMlPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [overrideUpdating, setOverrideUpdating] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fetch live system state, health, events, and telemetry/prediction history
  const fetchData = useCallback(async (options = {}) => {
    const quiet = options.quiet === true;
    if (!quiet) setIsChecking(true);
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

      // 4. Fetch telemetry history
      const metricsRes = await getMetricHistory(60);
      const metricsList = Array.isArray(metricsRes) ? metricsRes : [];
      setMetricHistory(metricsList);

      // 5. Fetch prediction history
      const predsRes = await getPredictionHistory(60);
      const predsList = Array.isArray(predsRes) ? predsRes : [];
      setPredictionHistory(predsList);

      // Populate latest metric from history if available
      if (metricsList.length > 0) {
        setLatestMetric((prev) => prev || metricsList[metricsList.length - 1]);
      }

      // Populate latest ML prediction from history if available
      if (predsList.length > 0) {
        const latestPred = predsList[predsList.length - 1];
        setMlPrediction((prev) => prev || {
          surge_probability: latestPred.probability,
          raw_probability: latestPred.probability,
          risk_signal: latestPred.risk_signal,
          watch_threshold: latestPred.watch_threshold,
          critical_threshold: latestPred.critical_threshold,
          feature_vector: latestPred.feature_snapshot || [],
          window_size: 61,
          insufficient_data: false,
          inference_time_ms: 0.5,
        });
      }

      setError(null);
    } catch (err) {
      setBackendConnected(false);
      setError(err.message || 'Failed to connect to AutoScale IQ backend');
    } finally {
      if (!quiet) setIsChecking(false);
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

  const handleSimulationTick = async (monitorRes, metric) => {
    if (metric) {
      setLatestMetric(metric);
    }
    if (monitorRes?.ml_prediction) {
      setMlPrediction(monitorRes.ml_prediction);
    }
    await fetchData({ quiet: true });
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

        {/* Optimization Impact — Before/After Measurement */}
        <OptimizationImpactPanel
          events={events}
          controllerState={systemState?.controller_state}
        />

        {/* Main Observability Grid: Telemetry & ML Prediction */}
        <div className="observability-row">
          <MetricCardsGrid
            latestMetric={latestMetric}
            historyCount={metricHistory.length}
          />
        </div>

        <div className="observability-row">
          <MLSurgePredictorPanel
            mlPrediction={mlPrediction}
            controllerState={systemState?.controller_state}
            predictionHistoryCount={predictionHistory.length}
          />
        </div>

        {/* Real-time telemetry & ML probability charts */}
        <TelemetryChartsPanel
          metricHistory={metricHistory}
          predictionHistory={predictionHistory}
        />

        <SimulationPanel
          backendConnected={backendConnected}
          autoRefresh={autoRefresh}
          onAutoRefreshChange={setAutoRefresh}
          onTickResult={handleSimulationTick}
        />

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
