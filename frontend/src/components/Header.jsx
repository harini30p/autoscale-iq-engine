import React from 'react';
import { API_BASE_URL } from '../services/api';

export default function Header({ backendConnected, isChecking, lastHealthCheck, onRefresh }) {
  let statusClass = 'status-checking';
  let statusText = 'Checking connection...';

  if (!isChecking) {
    if (backendConnected) {
      statusClass = 'status-connected';
      statusText = 'Backend Connected';
    } else {
      statusClass = 'status-disconnected';
      statusText = 'Backend Offline';
    }
  }

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="logo-icon">⚡</div>
        <div>
          <h1 className="brand-title">AutoScale IQ</h1>
          <p className="brand-subtitle">ML-Powered Self-Optimizing Engine & Safety Controller</p>
        </div>
      </div>

      <div className="header-controls">
        <div className="connection-badge" title={`Connected to ${API_BASE_URL}`}>
          <span className={`status-dot ${statusClass}`} />
          <span className="status-label">{statusText}</span>
          {lastHealthCheck && (
            <span className="status-time">
              {new Date(lastHealthCheck).toLocaleTimeString()}
            </span>
          )}
        </div>

        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onRefresh}
          disabled={isChecking}
          title="Refresh system state and events"
        >
          {isChecking ? '↻ Refreshing...' : '↻ Refresh'}
        </button>
      </div>
    </header>
  );
}
