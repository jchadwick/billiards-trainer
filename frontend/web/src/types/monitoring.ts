/**
 * Monitoring and metrics types
 */

import type { AlertLevel } from './api';

export interface Alert {
  id: string;
  level: AlertLevel;
  title: string;
  message: string;
  timestamp: Date;
  dismissed?: boolean;
  persistent?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface MetricPoint {
  timestamp: Date;
  value: number;
  label?: string;
}

export interface ChartMetrics {
  data: MetricPoint[];
  label: string;
  unit?: string;
  color?: string;
}
