/**
 * OptimizationImpactPanel — Shows the most recent optimization cycle:
 *   Before / After metric comparison, lever changes, and impact percentages.
 *
 * Data flows entirely from the centralized Dashboard 5-second polling.
 * No synthetic or hardcoded values are used.
 */
import React, { useMemo } from 'react';

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtTs(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString([], {
      month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch {
    return String(iso);
  }
}

function pct(val) {
  if (val == null) return null;
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

function safeNum(v, decimals = 1) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(decimals) : '—';
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ImpactChip({ value }) {
  const n = pct(value);
  if (n === null) return <span className="impact-chip impact-na">—</span>;
  const isGood = n > 0;
  const isBad = n < 0;
  const cls = isGood ? 'impact-chip impact-good' : isBad ? 'impact-chip impact-bad' : 'impact-chip impact-neutral';
  const sign = isGood ? '▼' : isBad ? '▲' : '';
  const label = `${sign} ${Math.abs(n).toFixed(1)}%`;
  return <span className={cls} title={isGood ? 'Improvement' : isBad ? 'Degradation' : 'No change'}>{label}</span>;
}

function MetricRow({ label, unit, before, after, impact }) {
  const b = before != null ? `${safeNum(before, 1)} ${unit}` : '—';
  const a = after != null ? `${safeNum(after, 1)} ${unit}` : null;
  return (
    <tr className="impact-table-row">
      <td className="impact-col-label">{label}</td>
      <td className="impact-col-val">{b}</td>
      <td className="impact-col-val">
        {a != null
          ? <span className="after-val">{a}</span>
          : <span className="after-waiting">waiting…</span>
        }
      </td>
      <td className="impact-col-change"><ImpactChip value={impact} /></td>
    </tr>
  );
}

function LeverRow({ label, before, after }) {
  const changed = before !== undefined && after !== undefined && before !== after;
  return (
    <div className="lever-change-row">
      <span className="lever-change-label">{label}</span>
      <span className="lever-change-vals">
        <span className="lever-before">{String(before)}</span>
        <span className="lever-arrow">→</span>
        <span className={`lever-after ${changed ? 'lever-after-active' : ''}`}>{String(after)}</span>
      </span>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function OptimizationImpactPanel({ events, controllerState }) {
  // Find the most recent apply_optimizations event
  const latestOpt = useMemo(() => {
    if (!Array.isArray(events) || events.length === 0) return null;
    return events.find((e) => e.action === 'apply_optimizations') || null;
  }, [events]);

  // ── Empty state: no optimization ever recorded ──
  if (!latestOpt) {
    return (
      <div className="card oip-card oip-empty">
        <div className="oip-header">
          <span className="oip-title">⚡ Optimization Impact</span>
          <span className="oip-subtitle">Before / After Measurement</span>
        </div>
        <div className="oip-empty-body">
          <span className="oip-empty-icon">🔎</span>
          <span className="oip-empty-text">No optimization cycle recorded yet</span>
          <span className="oip-empty-sub">
            When the Safety Controller triggers an optimization (3× CRITICAL signal or hard threshold breach),
            the before and after metrics will appear here.
          </span>
        </div>
      </div>
    );
  }

  const { timestamp, reason, success, error_message,
          before_metrics: bm, after_metrics: am, impact,
          previous_state, new_state } = latestOpt;

  // ── Failure state ──
  if (!success) {
    return (
      <div className="card oip-card oip-failed">
        <div className="oip-header">
          <span className="oip-title">⚡ Optimization Impact</span>
          <span className="oip-subtitle">Before / After Measurement</span>
          <span className="oip-ts">{fmtTs(timestamp)}</span>
          <span className="oip-failure-badge">⛔ Optimization Failed</span>
        </div>
        <div className="oip-failure-body">
          <p className="oip-failure-reason">{reason}</p>
          {error_message && (
            <p className="oip-failure-error">{error_message}</p>
          )}
          <p className="oip-failure-note">No before/after measurements are shown for failed optimizations.</p>
        </div>
      </div>
    );
  }

  const hasAfter = am != null;

  // Determine status badge based on controller state and after measurement
  let badgeText = 'Optimization Recorded';
  let badgeCls = 'oip-success-badge oip-badge-recorded';
  let badgeIcon = '✓';

  if (!hasAfter) {
    badgeText = 'Awaiting Measurement';
    badgeCls = 'oip-success-badge oip-badge-awaiting';
    badgeIcon = '⏳';
  } else if (controllerState === 'OPTIMIZED') {
    badgeText = 'Optimization Active';
    badgeCls = 'oip-success-badge oip-badge-active';
    badgeIcon = '⚡';
  } else {
    badgeText = 'Optimization Recorded';
    badgeCls = 'oip-success-badge oip-badge-recorded';
    badgeIcon = '✓';
  }

  return (
    <div className="card oip-card">
      {/* ── Header ── */}
      <div className="oip-header">
        <span className="oip-title">⚡ Optimization Impact</span>
        <span className="oip-subtitle">Before / After Measurement</span>
        <span className="oip-ts">{fmtTs(timestamp)}</span>
        <span className={badgeCls}>{badgeIcon} {badgeText}</span>
      </div>

      {/* ── Reason ── */}
      <div className="oip-reason-bar">
        <span className="oip-reason-label">Trigger:</span>
        <span className="oip-reason-text">{reason}</span>
      </div>

      <div className="oip-body-grid">
        {/* ── Metric comparison table ── */}
        <div className="oip-metrics-section">
          <span className="oip-section-label">PERFORMANCE IMPACT</span>
          {!hasAfter && (
            <div className="oip-awaiting-bar">
              ⏳ Waiting for post-optimization measurement (arrives on next telemetry reading)
            </div>
          )}
          <div className="oip-table-wrapper">
            <table className="impact-table">
              <thead>
                <tr>
                  <th className="impact-th">Metric</th>
                  <th className="impact-th">Before</th>
                  <th className="impact-th">After</th>
                  <th className="impact-th">Change</th>
                </tr>
              </thead>
              <tbody>
                <MetricRow
                  label="Response Time"
                  unit="ms"
                  before={bm?.response_time}
                  after={am?.response_time}
                  impact={impact?.response_time}
                />
                <MetricRow
                  label="CPU Utilization"
                  unit="%"
                  before={bm?.cpu_utilization}
                  after={am?.cpu_utilization}
                  impact={impact?.cpu_utilization}
                />
                <MetricRow
                  label="Memory"
                  unit="%"
                  before={bm?.memory_utilization}
                  after={am?.memory_utilization}
                  impact={impact?.memory_utilization}
                />
                <MetricRow
                  label="DB Query Time"
                  unit="ms"
                  before={bm?.db_query_time}
                  after={am?.db_query_time}
                  impact={impact?.db_query_time}
                />
              </tbody>
            </table>
          </div>

          {/* Traffic context row — not labeled as improvement */}
          {bm && (
            <div className="oip-context-row">
              <span className="oip-context-label">Traffic context:</span>
              <span className="oip-context-val">
                Before: <strong>{safeNum(bm.traffic, 0)} req/min</strong>
                {am && <> → After: <strong>{safeNum(am.traffic, 0)} req/min</strong></>}
              </span>
            </div>
          )}
        </div>

        {/* ── Lever changes ── */}
        <div className="oip-levers-section">
          <span className="oip-section-label">APPLIED LEVERS</span>
          <div className="oip-levers-list">
            <LeverRow
              label="Caching"
              before={previous_state?.caching}
              after={new_state?.caching}
            />
            <LeverRow
              label="Pagination"
              before={previous_state?.pagination_size != null ? `${previous_state.pagination_size} items/pg` : undefined}
              after={new_state?.pagination_size != null ? `${new_state.pagination_size} items/pg` : undefined}
            />
            <LeverRow
              label="Heavy Components"
              before={previous_state?.heavy_components}
              after={new_state?.heavy_components}
            />
          </div>

          {/* Timestamps */}
          {bm?.timestamp && (
            <div className="oip-ts-detail">
              <span className="oip-ts-row">
                <span className="oip-ts-label">Before captured:</span>
                <span className="oip-ts-val">{fmtTs(bm.timestamp)}</span>
              </span>
              {am?.timestamp && (
                <span className="oip-ts-row">
                  <span className="oip-ts-label">After captured:</span>
                  <span className="oip-ts-val">{fmtTs(am.timestamp)}</span>
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
