/**
 * MLSurgeProbabilityChart — Calibrated surge probability over time
 * from /predictions/history. Shows WATCH (0.040) and CRITICAL (0.075)
 * threshold lines. No fake or random data.
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

// ── Constants — locked thresholds ─────────────────────────────────────────────
const WATCH_THRESHOLD = 0.040;
const CRITICAL_THRESHOLD = 0.075;

// ── Helpers ───────────────────────────────────────────────────────────────────

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

function riskColor(prob) {
  if (prob == null) return '#64748b';
  if (prob >= CRITICAL_THRESHOLD) return '#ef4444';
  if (prob >= WATCH_THRESHOLD) return '#f59e0b';
  return '#10b981';
}

function riskLabel(prob) {
  if (prob == null) return 'UNKNOWN';
  if (prob >= CRITICAL_THRESHOLD) return 'CRITICAL';
  if (prob >= WATCH_THRESHOLD) return 'ELEVATED';
  return 'NORMAL';
}

// ── Custom Tooltip ─────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const prob = payload[0]?.value;
  const color = riskColor(prob);
  return (
    <div className="chart-tooltip">
      <div className="ct-time">{label}</div>
      <div className="ct-row">
        <span className="ct-dot" style={{ background: color }} />
        <span className="ct-label">Surge Prob</span>
        <span className="ct-value" style={{ color }}>
          {prob != null ? `${(prob * 100).toFixed(2)}%` : '—'}
        </span>
      </div>
      <div className="ct-risk" style={{ color }}>
        {riskLabel(prob)}
      </div>
    </div>
  );
}

// ── Gradient fill that transitions with risk ──────────────────────────────────

function GradientDef() {
  return (
    <defs>
      <linearGradient id="surgeGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
        <stop offset="60%" stopColor="#f59e0b" stopOpacity={0.15} />
        <stop offset="95%" stopColor="#10b981" stopOpacity={0.05} />
      </linearGradient>
    </defs>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function MLSurgeProbabilityChart({ predictionHistory }) {
  const data = useMemo(() => {
    if (!Array.isArray(predictionHistory) || predictionHistory.length === 0) return [];
    return predictionHistory
      .map((r) => ({
        time: fmtTime(r.timestamp),
        probability: safeNum(r.probability),
        risk_signal: r.risk_signal,
      }))
      .filter((d) => d.probability !== null);
  }, [predictionHistory]);

  const hasData = data.length > 0;

  // Compute latest risk for the chart stroke color
  const latestProb = hasData ? data[data.length - 1]?.probability : null;
  const strokeColor = riskColor(latestProb);

  return (
    <div className="chart-card chart-card--wide card">
      <div className="chart-header">
        <span className="chart-title">🧠 ML Surge Probability</span>
        <span className="chart-subtitle">Calibrated • Isotonic Regression • live history</span>
        {hasData && <span className="chart-count-badge">{data.length} pts</span>}
      </div>

      {!hasData ? (
        <div className="chart-empty-state">
          <span className="chart-empty-icon">🔮</span>
          <span className="chart-empty-text">No ML predictions yet…</span>
          <span className="chart-empty-sub">Predictions are generated each time a fresh metric is ingested via /monitor</span>
        </div>
      ) : (
        <div className="chart-body">
          {/* Threshold legend */}
          <div className="chart-legend">
            <span className="legend-item">
              <span className="legend-line" style={{ borderColor: '#f59e0b' }} />
              Watch ≥ {(WATCH_THRESHOLD * 100).toFixed(1)}%
            </span>
            <span className="legend-item">
              <span className="legend-line" style={{ borderColor: '#ef4444' }} />
              Critical ≥ {(CRITICAL_THRESHOLD * 100).toFixed(1)}%
            </span>
            <span className="legend-item">
              <span className="legend-dot" style={{ background: strokeColor }} />
              <span style={{ color: strokeColor, fontWeight: 600 }}>{riskLabel(latestProb)}</span>
            </span>
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <GradientDef />
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
                width={52}
                domain={[0, Math.max(0.12, (latestProb ?? 0) * 1.3)]}
                tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
              />
              <Tooltip content={<CustomTooltip />} />

              {/* WATCH threshold — amber */}
              <ReferenceLine
                y={WATCH_THRESHOLD}
                stroke="#f59e0b"
                strokeDasharray="5 3"
                strokeOpacity={0.8}
                label={{
                  value: `WATCH ${(WATCH_THRESHOLD * 100).toFixed(1)}%`,
                  position: 'insideTopRight',
                  fill: '#f59e0b',
                  fontSize: 10,
                  fontWeight: 600,
                }}
              />

              {/* CRITICAL threshold — red */}
              <ReferenceLine
                y={CRITICAL_THRESHOLD}
                stroke="#ef4444"
                strokeDasharray="5 3"
                strokeOpacity={0.8}
                label={{
                  value: `CRIT ${(CRITICAL_THRESHOLD * 100).toFixed(1)}%`,
                  position: 'insideTopRight',
                  fill: '#ef4444',
                  fontSize: 10,
                  fontWeight: 600,
                }}
              />

              <Area
                type="monotone"
                dataKey="probability"
                stroke={strokeColor}
                strokeWidth={2}
                fill="url(#surgeGrad)"
                dot={false}
                activeDot={{ r: 5, fill: strokeColor, stroke: '#0f172a', strokeWidth: 2 }}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
