/**
 * TelemetryChartsPanel — Aggregates all four real-data observability charts.
 * Receives metricHistory and predictionHistory directly from Dashboard.jsx
 * (the single source of truth for 5-second polling). No separate polling here.
 */
import React from 'react';
import TrafficChart from './charts/TrafficChart';
import ResponseTimeChart from './charts/ResponseTimeChart';
import ResourceUtilizationChart from './charts/ResourceUtilizationChart';
import MLSurgeProbabilityChart from './charts/MLSurgeProbabilityChart';

export default function TelemetryChartsPanel({ metricHistory, predictionHistory }) {
  return (
    <section className="telemetry-charts-section" aria-label="Telemetry Charts">
      {/* Section heading */}
      <div className="charts-section-header">
        <span className="charts-section-title">📈 Telemetry &amp; ML History</span>
        <span className="charts-section-sub">
          Live data from <code>/metrics/history</code> &amp; <code>/predictions/history</code> • polled every 5s
        </span>
      </div>

      {/* 2-column grid for telemetry charts */}
      <div className="charts-grid">
        <TrafficChart metricHistory={metricHistory} />
        <ResponseTimeChart metricHistory={metricHistory} />
        <ResourceUtilizationChart metricHistory={metricHistory} />
        {/* ML probability chart spans full width */}
        <MLSurgeProbabilityChart predictionHistory={predictionHistory} />
      </div>
    </section>
  );
}
