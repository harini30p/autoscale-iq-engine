import React from 'react';

export default function MLSurgePredictorPanel({
  mlPrediction,
  controllerState,
  predictionHistoryCount = 0,
}) {
  const hasPrediction = Boolean(mlPrediction);

  const risk = mlPrediction?.risk_signal || 'NORMAL';
  const prob = mlPrediction?.surge_probability ?? null;
  const rawProb = mlPrediction?.raw_probability ?? null;
  const watchThresh = mlPrediction?.watch_threshold ?? 0.040;
  const critThresh = mlPrediction?.critical_threshold ?? 0.075;
  const latency = mlPrediction?.inference_time_ms ?? null;
  const windowSize = mlPrediction?.window_size ?? null;

  const riskBadgeClass = {
    normal: 'risk-normal',
    elevated: 'risk-elevated',
    critical: 'risk-critical',
  }[risk.toLowerCase()] || 'risk-normal';

  return (
    <section className="ml-panel-section">
      <div className="section-header">
        <div className="title-with-badge">
          <h2 className="section-title">ML Surge Prediction Engine</h2>
          <span className="badge-model">HistGradientBoostingClassifier 5M</span>
          <span className="badge-calibrator">Isotonic Calibrated</span>
        </div>
        <span className="section-subtitle">
          Continuous 5-minute lookahead surge probability estimation
          {predictionHistoryCount > 0
            ? ` (${predictionHistoryCount} prediction${predictionHistoryCount === 1 ? '' : 's'} in history)`
            : ''}
        </span>
      </div>

      <div className="ml-panel-grid">
        {/* ML Risk Signal Card */}
        <div className="card ml-card">
          <div className="card-top-label">ML Risk Signal</div>
          <div className="ml-risk-body">
            {hasPrediction ? (
              <>
                <span className={`risk-signal-pill ${riskBadgeClass}`}>
                  {risk.toUpperCase()}
                </span>
                <span className="risk-explainer">
                  {risk.toLowerCase() === 'critical'
                    ? 'Surge probability >= 0.075 (Advancing confirmation counter)'
                    : risk.toLowerCase() === 'elevated'
                    ? 'Surge probability in [0.040, 0.075) (Watching system health)'
                    : 'Surge probability < 0.040 (Normal baseline operations)'}
                </span>
              </>
            ) : (
              <div className="metric-waiting">
                <span className="waiting-pill">Waiting for ML observation</span>
                <span className="waiting-subtext">Predictions trigger during metric ingestion</span>
              </div>
            )}
          </div>
          <div className="card-footer-info">
            <span>Thresholds: Watch = {watchThresh.toFixed(3)} | Crit = {critThresh.toFixed(3)}</span>
          </div>
        </div>

        {/* Calibrated Surge Probability Card */}
        <div className="card ml-card">
          <div className="card-top-label">Calibrated Surge Probability</div>
          <div className="ml-prob-body">
            {prob !== null ? (
              <>
                <div className="prob-number-row">
                  <span className="prob-big-val">{(prob * 100).toFixed(2)}%</span>
                  {rawProb !== null && (
                    <span className="raw-prob-label">(Raw: {(rawProb * 100).toFixed(2)}%)</span>
                  )}
                </div>

                <div className="prob-bar-track">
                  <div
                    className={`prob-bar-fill ${riskBadgeClass}`}
                    style={{ width: `${Math.min(100, Math.max(0, prob * 100 * 5))}%` }}
                  />
                  {/* Markers for thresholds */}
                  <div className="thresh-marker watch-marker" style={{ left: `${watchThresh * 100 * 5}%` }} title="tau_watch = 0.040" />
                  <div className="thresh-marker crit-marker" style={{ left: `${critThresh * 100 * 5}%` }} title="tau_crit = 0.075" />
                </div>
              </>
            ) : (
              <div className="metric-waiting">
                <span className="waiting-pill">Waiting for data</span>
                <span className="waiting-subtext">Probability output pending</span>
              </div>
            )}
          </div>
          <div className="card-footer-info">
            <span>{latency !== null ? `Inference Latency: ${latency.toFixed(2)}ms` : 'Target: 5-minute surge'}</span>
            <span>{windowSize !== null ? `Window: ${windowSize}/61 ticks` : ''}</span>
          </div>
        </div>

        {/* Advisory Safety Flow Card */}
        <div className="card ml-card">
          <div className="card-top-label">Safety Decision Flow</div>
          <div className="flow-body">
            <div className="flow-step">
              <span className="flow-num">1</span>
              <span className="flow-text">19-Feature Vector Generation</span>
            </div>
            <div className="flow-step">
              <span className="flow-num">2</span>
              <span className="flow-text">Isotonic Calibrated Inference</span>
            </div>
            <div className="flow-step">
              <span className="flow-num">3</span>
              <span className="flow-text">SafetyController (3x Confirmations)</span>
            </div>
            <div className="flow-step current">
              <span className="flow-num">4</span>
              <span className="flow-text">
                Current Controller: <strong>{controllerState || 'NORMAL'}</strong>
              </span>
            </div>
          </div>
          <div className="card-footer-info">
            <span>ML is advisory only; safety controller remains authoritative</span>
          </div>
        </div>
      </div>
    </section>
  );
}
