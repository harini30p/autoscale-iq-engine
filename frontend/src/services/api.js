/**
 * AutoScale IQ — Frontend API Client Layer
 * Handles communication with the FastAPI backend.
 * Base URL is configurable via VITE_API_BASE_URL.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Generic fetch wrapper with timeout and error handling.
 */
async function fetchJson(endpoint, options = {}) {
  const url = `${API_BASE_URL.replace(/\/$/, '')}/${endpoint.replace(/^\//, '')}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);

  try {
    const requestHeaders = {
      Accept: 'application/json',
      ...(options.headers || {}),
    };

    if (options.body !== undefined && !('Content-Type' in requestHeaders)) {
      requestHeaders['Content-Type'] = 'application/json';
    }

    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: requestHeaders,
    });

    clearTimeout(timeoutId);

    let payload = null;
    if (response.status !== 204 && response.headers.get('content-length') !== '0') {
      const rawText = await response.text();
      if (rawText) {
        try {
          payload = JSON.parse(rawText);
        } catch {
          payload = rawText;
        }
      }
    }

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status} ${response.statusText}`;
      if (typeof payload === 'string' && payload) {
        errorDetail = payload;
      } else if (payload?.detail) {
        errorDetail = typeof payload.detail === 'string'
          ? payload.detail
          : JSON.stringify(payload.detail);
      }
      throw new Error(errorDetail);
    }

    return payload;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out (backend unreachable)');
    }
    throw err;
  }
}

/**
 * Check backend health status (GET /health).
 * @returns {Promise<{status: string, timestamp: string}>}
 */
export async function checkHealth() {
  return fetchJson('/health');
}

/**
 * Get current safety controller state & active levers (GET /state).
 * @returns {Promise<Object>}
 */
export async function getSystemState() {
  return fetchJson('/state');
}

/**
 * Retrieve recent optimization and recovery events (GET /events).
 * @param {number} limit
 * @returns {Promise<Array<Object>>}
 */
export async function getRecentEvents(limit = 20) {
  return fetchJson(`/events?limit=${limit}`);
}

/**
 * Ingest a metric reading and run safety evaluation (POST /monitor).
 * @param {Object} payload - { metric: MetricSchema, risk_signal?: string }
 * @returns {Promise<Object>}
 */
export async function sendMonitorMetric(payload) {
  return fetchJson('/monitor', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

/**
 * Direct ML surge prediction inference (POST /predict).
 * @param {Object} payload - { invocations: number, minute_of_day: number, ...trigger_flags }
 * @returns {Promise<Object>}
 */
export async function predictSurge(payload) {
  return fetchJson('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

/**
 * Retrieve recent metric telemetry history (GET /metrics/history).
 * @param {number} limit
 * @returns {Promise<Array<Object>>}
 */
export async function getMetricHistory(limit = 60) {
  return fetchJson(`/metrics/history?limit=${limit}`);
}

/**
 * Retrieve recent ML prediction history (GET /predictions/history).
 * @param {number} limit
 * @returns {Promise<Array<Object>>}
 */
export async function getPredictionHistory(limit = 60) {
  return fetchJson(`/predictions/history?limit=${limit}`);
}

/**
 * Enable or disable manual safety override (POST /override).
 * @param {boolean} manualOverride
 * @returns {Promise<Object>}
 */
export async function setManualOverride(manualOverride) {
  return fetchJson('/override', {
    method: 'POST',
    body: JSON.stringify({ manual_override: manualOverride }),
  });
}

/**
 * List simulation scenario metadata (GET /simulation/scenarios).
 * Metadata only — no metric payloads.
 * @returns {Promise<Array<Object>>}
 */
export async function getSimulationScenarios() {
  return fetchJson('/simulation/scenarios');
}

/**
 * Load one scenario including tick payloads (GET /simulation/scenarios/:name).
 * Ticks have no timestamps; playback stamps UTC before POST /monitor.
 * @param {string} name
 * @returns {Promise<Object>}
 */
export async function getSimulationScenario(name) {
  return fetchJson(`/simulation/scenarios/${encodeURIComponent(name)}`);
}

/**
 * Restore controller defaults and clear the ML rolling window (POST /simulation/reset).
 * Does not delete historical metrics, predictions, or events.
 * @returns {Promise<Object>}
 */
export async function resetSimulation() {
  return fetchJson('/simulation/reset', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export { API_BASE_URL };

