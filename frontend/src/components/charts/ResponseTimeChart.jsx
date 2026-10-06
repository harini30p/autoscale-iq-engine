/**
 * ResponseTimeChart — Response time (ms) over time (from /metrics/history).
 * Uses real persisted metric history; no fake or random data.
 */
import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

function fmtTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '';
  }
}

function safeNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const rt = payload.find((p) => p.dataKey === 'response_time');
  const db = payload.find((p) => p.dataKey === 'db_query_time');
  return (
    <div className="chart-tooltip">
      <div className="ct-time">{label}</div>
      {rt && (
        <div className="ct-row">
          <span className="ct-dot" style={{ background: '#06b6d4' }} />
          <span className="ct-label">Response</span>
          <span className="ct-value">{rt.value != null ? `${rt.value.toFixed(1)} ms` : '—'}</span>
        </div>
      )}
      {db && (
        <div className="ct-row">
          <span className="ct-dot" style={{ background: '#8b5cf6' }} />
          <span className="ct-label">DB Query</span>
          <span className="ct-value">{db.value != null ? `${db.value.toFixed(1)} ms` : '—'}</span>
        </div>
      )}
    </div>
  );
}

export default function ResponseTimeChart({ metricHistory }) {
  const data = useMemo(() => {
    if (!Array.isArray(metricHistory) || metricHistory.length === 0) return [];
    return metricHistory.map((r) => ({
      time: fmtTime(r.timestamp),
      response_time: safeNum(r.response_time),
      db_query_time: safeNum(r.db_query_time),
    }));
  }, [metricHistory]);

  const hasData = data.length > 0;

  return (
    <div className="chart-card card">
      <div className="chart-header">
        <span className="chart-title">⏱ Response Time</span>
        <span className="chart-subtitle">Latency ms • live history</span>
        {hasData && <span className="chart-count-badge">{data.length} pts</span>}
      </div>

      {!hasData ? (
        <div className="chart-empty-state">
          <span className="chart-empty-icon">📭</span>
          <span className="chart-empty-text">Waiting for metric data…</span>
          <span className="chart-empty-sub">Send an observation via the simulator below</span>
        </div>
      ) : (
        <div className="chart-body">
          <div className="chart-legend">
            <span className="legend-item">
              <span className="legend-dot" style={{ background: '#06b6d4' }} />
              Response Time
            </span>
            <span className="legend-item">
              <span className="legend-dot" style={{ background: '#8b5cf6' }} />
              DB Query
            </span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fill: '#64748b', fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                width={48}
                tickFormatter={(v) => `${v}ms`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="response_time"
                stroke="#06b6d4"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#06b6d4' }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="db_query_time"
                stroke="#8b5cf6"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                dot={false}
                activeDot={{ r: 4, fill: '#8b5cf6' }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
