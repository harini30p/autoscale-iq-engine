import React from 'react';

const RECOVERY_CONFIRMATIONS = 3;
const HIGH_RISK_CONFIRMATIONS = 3;

export default function SystemStatusBanner({
  systemState,
  loading,
  error,
  onToggleOverride,
  overrideUpdating,
}) {
  if (loading && !systemState) {
    return (
      <div className="card status-banner skeleton-pulse">
        <div className="status-banner-header">
          <span className="banner-title">Loading System State...</span>
        </div>
      </div>
    );
  }

  if (error && !systemState) {
    return (
      <div className="card status-banner banner-error">
        <div className="status-banner-header">
          <span className="banner-title">⚠️ Unable to load system state</span>
        </div>
        <p className="banner-error-desc">{error}</p>
      </div>
    );
  }

  const state = systemState?.controller_state || 'UNKNOWN';
  const config = systemState?.current_configuration || {
    caching: 'disabled',
    pagination_size: 50,
    heavy_components: 'enabled',
  };
  const isOverride = Boolean(systemState?.manual_override);
  const isCooldown = Boolean(systemState?.cooldown_active);
  const cooldownSecs = Math.round(systemState?.cooldown_remaining_seconds || 0);
  const consecutiveNormal = systemState?.consecutive_normal ?? 0;
  const consecutiveHighRisk = systemState?.consecutive_high_risk ?? 0;

  const stateThemeMap = {
    NORMAL: 'badge-state-normal',
    WATCHING: 'badge-state-watching',
    OPTIMIZED: 'badge-state-optimized',
    RECOVERY: 'badge-state-recovery',
    UNKNOWN: 'badge-state-unknown',
  };

  const isRecovery = state === 'RECOVERY';
  const isOptimized = state === 'OPTIMIZED';
  const isWatching = state === 'WATCHING';

  return (
    <div className="card status-banner">
      <div className="status-banner-top">
        <div className="status-state-group">
          <span className="label-sm">Safety Controller State</span>
          <div className="state-display">
            <span className={`state-badge ${stateThemeMap[state] || 'badge-state-unknown'}`}>
              {state}
            </span>
            {isCooldown && (
              <span className="cooldown-badge" title="Cooldown prevents repeated oscillations">
                ⏳ Cooldown Active ({cooldownSecs}s)
              </span>
            )}
            {isOverride && (
              <span className="override-badge" title="Automated optimizations are paused">
                🔒 Manual Override Active
              </span>
            )}
          </div>
        </div>

        <div className="status-controls">
          <button
            type="button"
            className={`btn btn-sm ${isOverride ? 'btn-danger' : 'btn-outline'}`}
            onClick={() => onToggleOverride(!isOverride)}
            disabled={overrideUpdating}
            title="Toggle safety manual override"
          >
            {overrideUpdating
              ? 'Updating...'
              : isOverride
              ? 'Disable Manual Override'
              : 'Enable Manual Override'}
          </button>
        </div>
      </div>

      {/* Recovery / Watching progress indicator */}
      {(isRecovery || isWatching) && (
        <div className="recovery-progress-section">
          {isRecovery && (
            <>
              <div className="recovery-progress-header">
                <span className="recovery-progress-label">
                  🔄 Recovery in progress
                </span>
                <span className="recovery-progress-count">
                  {consecutiveNormal} / {RECOVERY_CONFIRMATIONS} normal confirmations
                </span>
              </div>
              <div className="recovery-progress-track">
                <div
                  className="recovery-progress-fill"
                  style={{
                    width: `${Math.min(100, (consecutiveNormal / RECOVERY_CONFIRMATIONS) * 100)}%`,
                  }}
                />
              </div>
              <span className="recovery-progress-sub">
                Defaults restored when {RECOVERY_CONFIRMATIONS} consecutive normal readings are confirmed
              </span>
            </>
          )}
          {isWatching && (
            <>
              <div className="recovery-progress-header">
                <span className="recovery-progress-label watching-label">
                  👁 Watching — elevated risk detected
                </span>
                <span className="recovery-progress-count watching-count">
                  {consecutiveHighRisk} / {HIGH_RISK_CONFIRMATIONS} critical confirmations
                </span>
              </div>
              <div className="recovery-progress-track watching-track">
                <div
                  className="recovery-progress-fill watching-fill"
                  style={{
                    width: `${Math.min(100, (consecutiveHighRisk / HIGH_RISK_CONFIRMATIONS) * 100)}%`,
                  }}
                />
              </div>
              <span className="recovery-progress-sub">
                Optimization triggers after {HIGH_RISK_CONFIRMATIONS} consecutive critical readings
              </span>
            </>
          )}
        </div>
      )}

      {isOptimized && (
        <div className="optimized-notice">
          <span>⚡ Optimizations active — monitoring for recovery conditions</span>
          <span className="recovery-progress-sub">
            {RECOVERY_CONFIRMATIONS} consecutive normal readings required to restore defaults
          </span>
        </div>
      )}

      <div className="status-divider" />

      <div className="status-levers-grid">
        <div className="lever-item">
          <span className="lever-label">Caching Tier</span>
          <span className={`lever-value ${config.caching === 'enabled' ? 'val-active' : 'val-default'}`}>
            {config.caching}
          </span>
        </div>

        <div className="lever-item">
          <span className="lever-label">Pagination Limit</span>
          <span className={`lever-value ${config.pagination_size === 20 ? 'val-active' : 'val-default'}`}>
            {config.pagination_size} items / page
          </span>
        </div>

        <div className="lever-item">
          <span className="lever-label">Heavy Components</span>
          <span className={`lever-value ${config.heavy_components === 'disabled' ? 'val-active' : 'val-default'}`}>
            {config.heavy_components}
          </span>
        </div>

        <div className="lever-item">
          <span className="lever-label">High-Risk Confirmations</span>
          <span className="lever-value">
            {consecutiveHighRisk} / {HIGH_RISK_CONFIRMATIONS}
          </span>
        </div>

        <div className="lever-item">
          <span className="lever-label">Recovery Confirmations</span>
          <span className="lever-value">
            {consecutiveNormal} / {RECOVERY_CONFIRMATIONS}
          </span>
        </div>
      </div>
    </div>
  );
}
