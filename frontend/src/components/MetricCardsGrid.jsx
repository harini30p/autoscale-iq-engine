import React from 'react';

export default function MetricCardsGrid({ latestMetric }) {
  const hasData = Boolean(latestMetric);

  const metricsConfig = [
    {
      id: 'traffic',
      name: 'Traffic Volume',
      unit: 'req/min',
      value: hasData ? latestMetric.traffic : null,
      description: 'Incoming invocation workload',
      thresholdInfo: 'ML Input Feature',
    },
    {
      id: 'response_time',
      name: 'Response Time',
      unit: 'ms',
      value: hasData ? latestMetric.response_time : null,
      description: 'End-to-end request latency',
      thresholdInfo: 'Hard limit: 1000ms',
    },
    {
      id: 'cpu',
      name: 'CPU Utilization',
      unit: '%',
      value: hasData ? latestMetric.cpu_utilization : null,
      description: 'Application server load',
      thresholdInfo: 'Hard limit: 90%',
    },
    {
      id: 'memory',
      name: 'Memory Utilization',
      unit: '%',
      value: hasData ? latestMetric.memory_utilization : null,
      description: 'Server RAM occupancy',
      thresholdInfo: 'Hard limit: 90%',
    },
    {
      id: 'active_users',
      name: 'Active Users',
      unit: 'users',
      value: hasData ? latestMetric.active_users : null,
      description: 'Concurrent user sessions',
      thresholdInfo: 'Telemetry Metric',
    },
    {
      id: 'db_query_time',
      name: 'DB Query Time',
      unit: 'ms',
      value: hasData ? latestMetric.db_query_time : null,
      description: 'Database operation latency',
      thresholdInfo: 'Database Metric',
    },
  ];

  return (
    <section className="metrics-section">
      <div className="section-header">
        <h2 className="section-title">Telemetry & System Workload</h2>
        <span className="section-subtitle">
          {hasData ? `Last updated: ${new Date(latestMetric.timestamp).toLocaleTimeString()}` : 'Live metric stream: Waiting for metric observations'}
        </span>
      </div>

      <div className="metric-cards-grid">
        {metricsConfig.map((item) => (
          <div key={item.id} className="card metric-card">
            <div className="metric-card-header">
              <span className="metric-name">{item.name}</span>
              <span className="metric-threshold-tag">{item.thresholdInfo}</span>
            </div>

            <div className="metric-value-container">
              {item.value !== null ? (
                <span className="metric-value">
                  {typeof item.value === 'number' ? item.value.toLocaleString() : item.value}
                  <span className="metric-unit"> {item.unit}</span>
                </span>
              ) : (
                <div className="metric-waiting">
                  <span className="waiting-pill">Waiting for data</span>
                  <span className="waiting-subtext">No active telemetry stream</span>
                </div>
              )}
            </div>

            <p className="metric-description">{item.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
