/**
 * Statistic card component for displaying key metrics
 * Used throughout the dashboard for showing important numbers
 */

import React from 'react';

export interface StatCardProps {
  title: string;
  value: string | number;
  unit?: string;
  change?: number; // Percentage change
  changeLabel?: string;
  icon?: string;
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'gray';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  className?: string;
}

const colorConfig = {
  blue: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    text: 'text-blue-600 dark:text-blue-400',
    icon: 'text-blue-500',
  },
  green: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    text: 'text-green-600 dark:text-green-400',
    icon: 'text-green-500',
  },
  yellow: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    text: 'text-yellow-600 dark:text-yellow-400',
    icon: 'text-yellow-500',
  },
  red: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    text: 'text-red-600 dark:text-red-400',
    icon: 'text-red-500',
  },
  purple: {
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    text: 'text-purple-600 dark:text-purple-400',
    icon: 'text-purple-500',
  },
  gray: {
    bg: 'bg-gray-50 dark:bg-gray-800',
    text: 'text-gray-600 dark:text-gray-400',
    icon: 'text-gray-500',
  },
};

const sizeConfig = {
  sm: {
    padding: 'p-4',
    title: 'text-sm',
    value: 'text-lg',
    icon: 'text-lg',
  },
  md: {
    padding: 'p-6',
    title: 'text-sm',
    value: 'text-2xl',
    icon: 'text-xl',
  },
  lg: {
    padding: 'p-8',
    title: 'text-base',
    value: 'text-3xl',
    icon: 'text-2xl',
  },
};

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  unit = '',
  change,
  changeLabel,
  icon,
  color = 'gray',
  size = 'md',
  loading = false,
  className = '',
}) => {
  const colorClasses = colorConfig[color];
  const sizeClasses = sizeConfig[size];

  const formatValue = (val: string | number): string => {
    if (typeof val === 'number') {
      if (val >= 1000000) {
        return `${(val / 1000000).toFixed(1)}M`;
      } else if (val >= 1000) {
        return `${(val / 1000).toFixed(1)}K`;
      }
      return val.toFixed(0);
    }
    return String(val);
  };

  const getChangeColor = (change: number): string => {
    if (change > 0) return 'text-green-600 dark:text-green-400';
    if (change < 0) return 'text-red-600 dark:text-red-400';
    return 'text-gray-600 dark:text-gray-400';
  };

  const getChangeIcon = (change: number): string => {
    if (change > 0) return '↗';
    if (change < 0) return '↘';
    return '→';
  };

  return (
    <div
      className={`
        ${colorClasses.bg}
        ${sizeClasses.padding}
        rounded-lg
        border
        border-gray-200
        dark:border-gray-700
        ${className}
      `}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className={`${sizeClasses.title} font-medium text-gray-500 dark:text-gray-400`}>
            {title}
          </p>

          {loading ? (
            <div className="mt-2">
              <div className="animate-pulse bg-gray-300 dark:bg-gray-600 h-8 w-16 rounded" />
            </div>
          ) : (
            <p className={`${sizeClasses.value} font-bold ${colorClasses.text} mt-2`}>
              {formatValue(value)}
              {unit && <span className="text-sm font-normal ml-1">{unit}</span>}
            </p>
          )}

          {change !== undefined && !loading && (
            <div className={`flex items-center mt-1 ${sizeClasses.title}`}>
              <span className={getChangeColor(change)}>
                {getChangeIcon(change)} {Math.abs(change).toFixed(1)}%
              </span>
              {changeLabel && (
                <span className="ml-1 text-gray-500 dark:text-gray-400">
                  {changeLabel}
                </span>
              )}
            </div>
          )}
        </div>

        {icon && (
          <div className={`${sizeClasses.icon} ${colorClasses.icon} ml-4`}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};

export default StatCard;
