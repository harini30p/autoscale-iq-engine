/**
 * ResourceUtilizationChart — CPU and Memory % over time (from /metrics/history).
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
  const cpu = payload.find((p) => p.dataKey === 'cpu_utilization');
  const mem = payload.find((p) => p.dataKey === 'memory_utilization');
  return (
    <div className="chart-tooltip">
      <div className="ct-time">{label}</div>
      {cpu && (
        <div className="ct-row">
          <span className="ct-dot" style={{ background: '#f59e0b' }} />
          <span className="ct-label">CPU</span>
          <span className="ct-value">{cpu.value != null ? `${cpu.value.toFixed(1)}%` : '—'}</span>
        </div>
      )}
      {mem && (
        <div className="ct-row">
          <span className="ct-dot" style={{ background: '#10b981' }} />
          <span className="ct-label">Memory</span>
          <span className="ct-value">{mem.value != null ? `${mem.value.toFixed(1)}%` : '—'}</span>
        </div>
      )}
    </div>
  );
}

export default function ResourceUtilizationChart({ metricHistory }) {
  const data = useMemo(() => {
    if (!Array.isArray(metricHistory) || metricHistory.length === 0) return [];
    return metricHistory.map((r) => ({
      time: fmtTime(r.timestamp),
      cpu_utilization: safeNum(r.cpu_utilization),
      memory_utilization: safeNum(r.memory_utilization),
    }));
  }, [metricHistory]);

  const hasData = data.length > 0;

  return (
    <div className="chart-card card">
      <div className="chart-header">
        <span className="chart-title">🖥 CPU &amp; Memory</span>
        <span className="chart-subtitle">Utilization % • live history</span>
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
              <span className="legend-dot" style={{ background: '#f59e0b' }} />
              CPU
            </span>
            <span className="legend-item">
              <span className="legend-dot" style={{ background: '#10b981' }} />
              Memory
            </span>
            <span className="legend-item legend-threshold">
              <span className="legend-line" style={{ borderColor: '#ef4444' }} />
              90% hard limit
            </span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
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
                width={38}
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              {/* Hard safety threshold at 90% CPU */}
              <ReferenceLine
                y={90}
                stroke="#ef4444"
                strokeDasharray="4 3"
                strokeOpacity={0.6}
                label={{ value: '90%', position: 'insideTopRight', fill: '#ef4444', fontSize: 10 }}
              />
              <Area
                type="monotone"
                dataKey="cpu_utilization"
                stroke="#f59e0b"
                strokeWidth={2}
                fill="url(#cpuGrad)"
                dot={false}
                activeDot={{ r: 4, fill: '#f59e0b' }}
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="memory_utilization"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#memGrad)"
                dot={false}
                activeDot={{ r: 4, fill: '#10b981' }}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
