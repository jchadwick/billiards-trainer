/**
 * Status indicator component for showing health status
 * Supports different status types with appropriate colors and animations
 */

import React from 'react';
import type { HealthStatus } from '../../types/api';

export interface StatusIndicatorProps {
  status: HealthStatus | 'unknown' | 'loading';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  animated?: boolean;
  description?: string;
  className?: string;
}

const statusConfig = {
  healthy: {
    color: 'bg-green-500',
    textColor: 'text-green-700',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    icon: '✓',
    label: 'Healthy',
  },
  degraded: {
    color: 'bg-yellow-500',
    textColor: 'text-yellow-700',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
    icon: '⚠',
    label: 'Degraded',
  },
  unhealthy: {
    color: 'bg-red-500',
    textColor: 'text-red-700',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    icon: '✗',
    label: 'Unhealthy',
  },
  unknown: {
    color: 'bg-gray-400',
    textColor: 'text-gray-700',
    bgColor: 'bg-gray-50',
    borderColor: 'border-gray-200',
    icon: '?',
    label: 'Unknown',
  },
  loading: {
    color: 'bg-blue-500',
    textColor: 'text-blue-700',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    icon: '⟳',
    label: 'Loading',
  },
};

const sizeConfig = {
  sm: {
    dot: 'h-2 w-2',
    icon: 'text-xs',
    text: 'text-xs',
    badge: 'px-2 py-1',
  },
  md: {
    dot: 'h-3 w-3',
    icon: 'text-sm',
    text: 'text-sm',
    badge: 'px-3 py-1',
  },
  lg: {
    dot: 'h-4 w-4',
    icon: 'text-base',
    text: 'text-base',
    badge: 'px-4 py-2',
  },
};

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  size = 'md',
  showLabel = true,
  animated = true,
  description,
  className = '',
}) => {
  const config = statusConfig[status];
  const sizeClasses = sizeConfig[size];

  if (!showLabel) {
    // Simple dot indicator
    return (
      <div
        className={`inline-flex items-center ${className}`}
        title={label || config.label}
      >
        <div
          className={`
            ${sizeClasses.dot}
            ${config.color}
            rounded-full
            ${animated && status === 'loading' ? 'animate-spin' : ''}
            ${animated && status === 'healthy' ? 'animate-pulse' : ''}
          `}
        />
        {description && (
          <span className={`ml-2 ${sizeClasses.text} text-gray-600 dark:text-gray-400`}>
            {description}
          </span>
        )}
      </div>
    );
  }

  // Badge-style indicator
  return (
    <div className={`inline-flex items-center ${className}`}>
      <span
        className={`
          inline-flex items-center
          ${sizeClasses.badge}
          ${sizeClasses.text}
          font-medium
          rounded-full
          ${config.bgColor}
          ${config.textColor}
          ${config.borderColor}
          border
          dark:bg-gray-800
          dark:text-gray-300
          dark:border-gray-600
        `}
      >
        <span
          className={`
            ${sizeClasses.icon}
            mr-1
            ${animated && status === 'loading' ? 'animate-spin' : ''}
          `}
        >
          {config.icon}
        </span>
        {label || config.label}
      </span>
      {description && (
        <span className={`ml-2 ${sizeClasses.text} text-gray-600 dark:text-gray-400`}>
          {description}
        </span>
      )}
    </div>
  );
};

export default StatusIndicator;
