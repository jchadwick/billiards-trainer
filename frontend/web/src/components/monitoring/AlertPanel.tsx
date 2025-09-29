/**
 * Alert panel component for displaying system alerts and notifications
 * Supports different alert levels and dismissible notifications
 */

import React, { useState } from 'react';
import type { Alert } from '../../types/monitoring';

export interface AlertPanelProps {
  alerts: Alert[];
  onDismiss?: (alertId: string) => void;
  maxAlerts?: number;
  showTimestamp?: boolean;
  className?: string;
}

const alertConfig = {
  info: {
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    borderColor: 'border-blue-200 dark:border-blue-800',
    textColor: 'text-blue-800 dark:text-blue-200',
    icon: 'ℹ',
    iconColor: 'text-blue-400',
  },
  warning: {
    bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
    borderColor: 'border-yellow-200 dark:border-yellow-800',
    textColor: 'text-yellow-800 dark:text-yellow-200',
    icon: '⚠',
    iconColor: 'text-yellow-400',
  },
  error: {
    bgColor: 'bg-red-50 dark:bg-red-900/20',
    borderColor: 'border-red-200 dark:border-red-800',
    textColor: 'text-red-800 dark:text-red-200',
    icon: '✗',
    iconColor: 'text-red-400',
  },
  critical: {
    bgColor: 'bg-red-100 dark:bg-red-900/40',
    borderColor: 'border-red-300 dark:border-red-700',
    textColor: 'text-red-900 dark:text-red-100',
    icon: '🚨',
    iconColor: 'text-red-500',
  },
};

export const AlertPanel: React.FC<AlertPanelProps> = ({
  alerts,
  onDismiss,
  maxAlerts = 10,
  showTimestamp = true,
  className = '',
}) => {
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());

  const visibleAlerts = alerts
    .filter(alert => !alert.dismissed && !dismissedAlerts.has(alert.id))
    .slice(0, maxAlerts);

  const handleDismiss = (alertId: string) => {
    setDismissedAlerts(prev => new Set([...prev, alertId]));
    onDismiss?.(alertId);
  };

  const formatTimestamp = (timestamp: Date): string => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) {
      return `${days}d ago`;
    } else if (hours > 0) {
      return `${hours}h ago`;
    } else if (minutes > 0) {
      return `${minutes}m ago`;
    } else {
      return 'Just now';
    }
  };

  if (visibleAlerts.length === 0) {
    return null;
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {visibleAlerts.map((alert) => {
        const config = alertConfig[alert.level];

        return (
          <div
            key={alert.id}
            className={`
              ${config.bgColor}
              ${config.borderColor}
              border
              rounded-lg
              p-4
              ${alert.level === 'critical' ? 'animate-pulse' : ''}
            `}
          >
            <div className="flex items-start">
              <div className={`flex-shrink-0 ${config.iconColor} text-lg mr-3`}>
                {config.icon}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4 className={`text-sm font-medium ${config.textColor}`}>
                    {alert.title}
                  </h4>

                  <div className="flex items-center space-x-2">
                    {showTimestamp && (
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {formatTimestamp(alert.timestamp)}
                      </span>
                    )}

                    {!alert.persistent && (
                      <button
                        onClick={() => handleDismiss(alert.id)}
                        className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                        title="Dismiss"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </div>

                <p className={`mt-1 text-sm ${config.textColor}`}>
                  {alert.message}
                </p>

                {alert.action && (
                  <div className="mt-3">
                    <button
                      onClick={alert.action.onClick}
                      className={`
                        text-sm
                        font-medium
                        ${alert.level === 'info' ? 'text-blue-600 hover:text-blue-500' : ''}
                        ${alert.level === 'warning' ? 'text-yellow-600 hover:text-yellow-500' : ''}
                        ${alert.level === 'error' || alert.level === 'critical' ? 'text-red-600 hover:text-red-500' : ''}
                        transition-colors
                      `}
                    >
                      {alert.action.label} →
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}

      {alerts.length > maxAlerts && (
        <div className="text-center py-2">
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Showing {maxAlerts} of {alerts.length} alerts
          </span>
        </div>
      )}
    </div>
  );
};

export default AlertPanel;
