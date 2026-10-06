import React from 'react';

export default function EventsTable({ events, loading, error }) {
  const hasEvents = Boolean(events && events.length > 0);

  return (
    <section className="events-section">
      <div className="section-header">
        <h2 className="section-title">Optimization & Recovery History</h2>
        <span className="section-subtitle">
          Audit trail of automated controller actions and state transitions
        </span>
      </div>

      <div className="card events-card">
        {loading && !hasEvents ? (
          <div className="table-status-message">Loading event history...</div>
        ) : error && !hasEvents ? (
          <div className="table-status-message error">⚠️ Unable to load events: {error}</div>
        ) : !hasEvents ? (
          <div className="table-empty-state">
            <span className="empty-icon">📋</span>
            <span className="empty-title">No optimization events recorded yet</span>
            <span className="empty-desc">
              When the safety controller triggers an optimization or completes recovery, events will appear here in real time.
            </span>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="events-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Trigger Reason</th>
                  <th>State Transition</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {events.map((evt) => {
                  const isApply = evt.action === 'apply_optimizations';
                  const isSuccess = Boolean(evt.success);

                  return (
                    <tr key={evt.id || evt.timestamp}>
                      <td className="cell-time">
                        {new Date(evt.timestamp).toLocaleString()}
                      </td>
                      <td>
                        <span className={`action-pill ${isApply ? 'action-apply' : 'action-restore'}`}>
                          {isApply ? '⚡ Optimize' : '🔄 Restore Defaults'}
                        </span>
                      </td>
                      <td className="cell-reason" title={evt.reason}>
                        {evt.reason}
                      </td>
                      <td className="cell-state-transition">
                        <span className="state-from">
                          {evt.previous_state?.caching ? `caching:${evt.previous_state.caching}` : 'default'}
                        </span>
                        <span className="state-arrow">→</span>
                        <span className="state-to">
                          {evt.new_state?.caching ? `caching:${evt.new_state.caching}` : 'optimized'}
                        </span>
                      </td>
                      <td>
                        <span className={`status-pill ${isSuccess ? 'pill-success' : 'pill-failure'}`}>
                          {isSuccess ? 'Success' : 'Failed'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
