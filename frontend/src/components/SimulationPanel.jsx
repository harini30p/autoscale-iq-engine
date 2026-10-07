/**
 * SimulationPanel — plays backend scenarios through POST /monitor.
 * Tick metrics come from GET /simulation/scenarios/:name (no timestamps).
 * Playback stamps current UTC immediately before each monitor request.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  getSimulationScenario,
  getSimulationScenarios,
  resetSimulation,
  sendMonitorMetric,
} from '../services/api';

export const PLAYBACK_SPEEDS = {
  slow: 2000,
  normal: 1000,
  fast: 500,
};

const STATUS = {
  READY: 'ready',
  RUNNING: 'running',
  PAUSED: 'paused',
  STOPPED: 'stopped',
  COMPLETED: 'completed',
};

const STATUS_LABELS = {
  ready: 'Ready',
  running: 'Running',
  paused: 'Paused',
  stopped: 'Stopped',
  completed: 'Completed',
};

const DEFAULT_SCENARIO = 'gradual_surge';

function formatScenarioName(name) {
  if (!name) return '';
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function delay(ms, timeoutRef, waitResolveRef) {
  return new Promise((resolve) => {
    waitResolveRef.current = resolve;
    timeoutRef.current = setTimeout(() => {
      timeoutRef.current = null;
      waitResolveRef.current = null;
      resolve();
    }, ms);
  });
}

export default function SimulationPanel({
  backendConnected = false,
  autoRefresh,
  onAutoRefreshChange,
  onTickResult,
}) {
  const [scenarios, setScenarios] = useState([]);
  const [selectedName, setSelectedName] = useState('');
  const [scenarioDetail, setScenarioDetail] = useState(null);
  const [status, setStatus] = useState(STATUS.READY);
  const [tickIndex, setTickIndex] = useState(0);
  const [speed, setSpeed] = useState('normal');
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [playBusy, setPlayBusy] = useState(false);
  const [error, setError] = useState(null);

  const sessionRef = useRef(0);
  const statusRef = useRef(STATUS.READY);
  const tickIndexRef = useRef(0);
  const scenarioDetailRef = useRef(null);
  const speedRef = useRef('normal');
  const timeoutRef = useRef(null);
  const waitResolveRef = useRef(null);
  const playLockRef = useRef(false);

  const setPlaybackStatus = (next) => {
    statusRef.current = next;
    setStatus(next);
  };

  const bumpSession = useCallback(() => {
    sessionRef.current += 1;
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (waitResolveRef.current) {
      const resolve = waitResolveRef.current;
      waitResolveRef.current = null;
      resolve();
    }
  }, []);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    scenarioDetailRef.current = scenarioDetail;
  }, [scenarioDetail]);

  useEffect(() => {
    return () => {
      bumpSession();
    };
  }, [bumpSession]);

  const loadScenarioList = useCallback(async () => {
    setLoadingList(true);
    setError(null);
    try {
      const list = await getSimulationScenarios();
      const items = Array.isArray(list) ? list : [];
      setScenarios(items);
      const names = items.map((s) => s.name);
      const preferred = names.includes(DEFAULT_SCENARIO)
        ? DEFAULT_SCENARIO
        : names[0] || '';
      setSelectedName((prev) => (prev && names.includes(prev) ? prev : preferred));
    } catch (err) {
      setError(`Failed to load simulation scenarios: ${err.message}`);
      setScenarios([]);
    } finally {
      setLoadingList(false);
    }
  }, []);

  const loadScenarioDetail = useCallback(async (name) => {
    if (!name) {
      setScenarioDetail(null);
      return;
    }
    setLoadingDetail(true);
    setError(null);
    try {
      const detail = await getSimulationScenario(name);
      setScenarioDetail(detail);
      scenarioDetailRef.current = detail;
    } catch (err) {
      setScenarioDetail(null);
      scenarioDetailRef.current = null;
      setError(`Failed to load scenario "${name}": ${err.message}`);
    } finally {
      setLoadingDetail(false);
    }
  }, []);

  useEffect(() => {
    if (!backendConnected) return;
    loadScenarioList();
  }, [backendConnected, loadScenarioList]);

  useEffect(() => {
    if (!selectedName || !backendConnected) return;
    loadScenarioDetail(selectedName);
  }, [selectedName, backendConnected, loadScenarioDetail]);

  const tickCount = scenarioDetail?.tick_count
    || scenarioDetail?.ticks?.length
    || 0;
  const ticks = scenarioDetail?.ticks || [];
  const currentTick = ticks[Math.min(tickIndex, Math.max(tickCount - 1, 0))];
  const currentPhase = tickCount === 0
    ? '—'
    : (status === STATUS.COMPLETED
      ? (ticks[tickCount - 1]?.phase || '—')
      : (currentTick?.phase || ticks[0]?.phase || '—'));
  const progressPct = tickCount === 0 ? 0 : Math.min(100, (tickIndex / tickCount) * 100);
  const isRunning = status === STATUS.RUNNING;
  const controlsLocked = isRunning || playBusy || loadingDetail || loadingList;

  const runPlayback = useCallback(async (session) => {
    while (session === sessionRef.current && statusRef.current === STATUS.RUNNING) {
      const detail = scenarioDetailRef.current;
      const loadedTicks = detail?.ticks || [];
      const idx = tickIndexRef.current;

      if (idx >= loadedTicks.length) {
        if (session === sessionRef.current) {
          setPlaybackStatus(STATUS.COMPLETED);
        }
        return;
      }

      const tick = loadedTicks[idx];
      const timestamp = new Date().toISOString();
      const metric = {
        ...tick.metric,
        timestamp,
      };

      try {
        const monitorRes = await sendMonitorMetric({
          metric,
          risk_signal: 'normal',
        });

        if (session !== sessionRef.current) return;

        tickIndexRef.current = idx + 1;
        setTickIndex(idx + 1);

        if (typeof onTickResult === 'function') {
          await onTickResult(monitorRes, metric);
        }

        if (session !== sessionRef.current) return;

        if (idx + 1 >= loadedTicks.length) {
          setPlaybackStatus(STATUS.COMPLETED);
          return;
        }

        if (statusRef.current !== STATUS.RUNNING) return;

        await delay(
          PLAYBACK_SPEEDS[speedRef.current] || PLAYBACK_SPEEDS.normal,
          timeoutRef,
          waitResolveRef,
        );

        if (session !== sessionRef.current) return;
        if (statusRef.current !== STATUS.RUNNING) return;
      } catch (err) {
        if (session !== sessionRef.current) return;
        setError(`Playback stopped: failed to send tick ${idx} via /monitor (${err.message})`);
        setPlaybackStatus(STATUS.PAUSED);
        return;
      }
    }
  }, [onTickResult]);

  const handlePlay = async () => {
    if (playLockRef.current) return;
    if (!backendConnected) {
      setError('Backend is unavailable. Cannot start simulation.');
      return;
    }
    if (!scenarioDetailRef.current?.ticks?.length) {
      setError('No scenario ticks loaded. Select a scenario and try again.');
      return;
    }

    playLockRef.current = true;
    setPlayBusy(true);
    setError(null);
    const needsFreshStart = statusRef.current !== STATUS.PAUSED;

    if (needsFreshStart) {
      try {
        await resetSimulation();
      } catch (err) {
        setError(`Simulation reset failed: ${err.message}`);
        setPlaybackStatus(STATUS.STOPPED);
        playLockRef.current = false;
        setPlayBusy(false);
        return;
      }
      tickIndexRef.current = 0;
      setTickIndex(0);
    }

    bumpSession();
    const session = sessionRef.current;
    setPlaybackStatus(STATUS.RUNNING);
    try {
      await runPlayback(session);
    } finally {
      playLockRef.current = false;
      setPlayBusy(false);
    }
  };

  const handlePause = () => {
    if (statusRef.current !== STATUS.RUNNING) return;
    setPlaybackStatus(STATUS.PAUSED);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (waitResolveRef.current) {
      const resolve = waitResolveRef.current;
      waitResolveRef.current = null;
      resolve();
    }
  };

  const handleStop = () => {
    bumpSession();
    tickIndexRef.current = 0;
    setTickIndex(0);
    setPlaybackStatus(STATUS.STOPPED);
  };

  const handleScenarioChange = (event) => {
    const next = event.target.value;
    bumpSession();
    tickIndexRef.current = 0;
    setTickIndex(0);
    setPlaybackStatus(STATUS.READY);
    setSelectedName(next);
  };

  return (
    <section className={`card sim-panel ${isRunning ? 'sim-panel-running' : ''}`}>
      <div className="sim-panel-header">
        <div>
          <h3 className="sim-panel-title">Interactive Simulation Engine</h3>
          <p className="sim-panel-subtitle">
            Plays backend scenarios through <code>/monitor</code> — ML, controller, and optimizer stay in the live path.
          </p>
        </div>
        <span className={`sim-status-pill sim-status-${status}`}>
          {STATUS_LABELS[status]}
        </span>
      </div>

      <div className="sim-panel-grid">
        <label className="sim-field">
          <span className="sim-label">Scenario</span>
          <select
            className="sim-select"
            value={selectedName}
            onChange={handleScenarioChange}
            disabled={controlsLocked || !backendConnected || scenarios.length === 0}
          >
            {scenarios.length === 0 && (
              <option value="">
                {loadingList ? 'Loading scenarios…' : 'No scenarios available'}
              </option>
            )}
            {scenarios.map((item) => (
              <option key={item.name} value={item.name}>
                {formatScenarioName(item.name)} ({item.tick_count} ticks)
              </option>
            ))}
          </select>
        </label>

        <label className="sim-field">
          <span className="sim-label">Speed</span>
          <select
            className="sim-select"
            value={speed}
            onChange={(e) => setSpeed(e.target.value)}
            disabled={!backendConnected}
          >
            <option value="slow">Slow — 2s / tick</option>
            <option value="normal">Normal — 1s / tick</option>
            <option value="fast">Fast — 500ms / tick</option>
          </select>
        </label>

        <div className="sim-field">
          <span className="sim-label">Phase</span>
          <span className="sim-phase-value">{currentPhase}</span>
        </div>
      </div>

      <p className="sim-description">
        {loadingDetail
          ? 'Loading scenario ticks…'
          : (scenarioDetail?.description || 'Select a scenario to load tick metadata from the backend.')}
      </p>

      <div className="sim-progress-row">
        <div className="sim-progress-meta">
          <span>
            Tick {Math.min(tickIndex, tickCount)} / {tickCount}
          </span>
          <span>{Math.round(progressPct)}%</span>
        </div>
        <div
          className="sim-progress-track"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={tickCount || 0}
          aria-valuenow={Math.min(tickIndex, tickCount)}
        >
          <div className="sim-progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
      </div>

      <div className="sim-controls">
        <button
          type="button"
          className="btn btn-accent btn-sm"
          onClick={handlePlay}
          disabled={isRunning || playBusy || !backendConnected || loadingDetail || tickCount === 0}
        >
          {status === STATUS.PAUSED ? 'Resume' : 'Play'}
        </button>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={handlePause}
          disabled={!isRunning}
        >
          Pause
        </button>
        <button
          type="button"
          className="btn btn-danger btn-sm"
          onClick={handleStop}
          disabled={status === STATUS.READY && tickIndex === 0}
        >
          Stop
        </button>
        {typeof autoRefresh === 'boolean' && onAutoRefreshChange && (
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => onAutoRefreshChange(e.target.checked)}
            />
            Auto-poll (5s)
          </label>
        )}
      </div>

      {error && (
        <div className="sim-error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}
