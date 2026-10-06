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
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...(options.headers || {}),
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData?.detail) {
          errorDetail = typeof errorData.detail === 'string'
            ? errorData.detail
            : JSON.stringify(errorData.detail);
        }
      } catch {
        // Response body was not JSON
      }
      throw new Error(errorDetail);
    }

    return await response.json();
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

export { API_BASE_URL };

