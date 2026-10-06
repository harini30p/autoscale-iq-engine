/**
 * TrafficChart — Requests/min over time (from /metrics/history).
 * Uses real persisted metric history; no fake or random data.
 */
import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';

// ── Helpers ──────────────────────────────────────────────────────────────────

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

// ── Custom Tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value;
  return (
    <div className="chart-tooltip">
      <div className="ct-time">{label}</div>
      <div className="ct-row">
        <span className="ct-dot" style={{ background: '#3b82f6' }} />
        <span className="ct-label">Traffic</span>
        <span className="ct-value">{val != null ? `${val.toFixed(1)} req/min` : '—'}</span>
      </div>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function TrafficChart({ metricHistory }) {
  const data = useMemo(() => {
    if (!Array.isArray(metricHistory) || metricHistory.length === 0) return [];
    return metricHistory
      .map((r) => ({
        time: fmtTime(r.timestamp),
        traffic: safeNum(r.traffic),
      }))
      .filter((d) => d.traffic !== null);
  }, [metricHistory]);

  const hasData = data.length > 0;

  return (
    <div className="chart-card card">
      <div className="chart-header">
        <span className="chart-title">📶 Traffic</span>
        <span className="chart-subtitle">Requests / min • live history</span>
        {hasData && (
          <span className="chart-count-badge">{data.length} pts</span>
        )}
      </div>

      {!hasData ? (
        <div className="chart-empty-state">
          <span className="chart-empty-icon">📭</span>
          <span className="chart-empty-text">Waiting for metric data…</span>
          <span className="chart-empty-sub">Send an observation via the simulator below</span>
        </div>
      ) : (
        <div className="chart-body">
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="trafficGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
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
                tickFormatter={(v) => `${v}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="traffic"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#trafficGrad)"
                dot={false}
                activeDot={{ r: 4, fill: '#3b82f6', stroke: '#1e3a8a', strokeWidth: 2 }}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
