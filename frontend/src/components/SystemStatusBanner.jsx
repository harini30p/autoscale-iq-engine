import React from 'react';

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

  const stateThemeMap = {
    NORMAL: 'badge-state-normal',
    WATCHING: 'badge-state-watching',
    OPTIMIZED: 'badge-state-optimized',
    RECOVERY: 'badge-state-recovery',
    UNKNOWN: 'badge-state-unknown',
  };

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
            {systemState?.consecutive_high_risk ?? 0} / 3
          </span>
        </div>

        <div className="lever-item">
          <span className="lever-label">Recovery Confirmations</span>
          <span className="lever-value">
            {systemState?.consecutive_normal ?? 0} / 3
          </span>
        </div>
      </div>
    </div>
  );
}
