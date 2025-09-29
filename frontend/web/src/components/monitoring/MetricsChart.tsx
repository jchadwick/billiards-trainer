/**
 * Real-time metrics chart component using Chart.js
 * Supports line charts, bar charts, and real-time data updates
 */

import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import type { ChartOptions, ChartData } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import type { MetricPoint } from '../../types/monitoring';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export interface MetricsChartProps {
  title: string;
  data: MetricPoint[];
  type?: 'line' | 'bar';
  maxDataPoints?: number;
  color?: string;
  unit?: string;
  height?: number;
  realTime?: boolean;
  showGrid?: boolean;
  animated?: boolean;
  yAxisMax?: number;
  yAxisMin?: number;
}

export const MetricsChart: React.FC<MetricsChartProps> = ({
  title,
  data,
  type = 'line',
  maxDataPoints = 50,
  color = 'rgb(59, 130, 246)', // blue-500
  unit = '',
  height = 300,
  realTime = true,
  showGrid = true,
  animated = true,
  yAxisMax,
  yAxisMin,
}) => {
  const chartRef = useRef<ChartJS<'line' | 'bar'>>(null);

  // Limit data points for performance
  const limitedData = data.slice(-maxDataPoints);

  // Prepare chart data
  const chartData: ChartData<'line' | 'bar'> = {
    labels: limitedData.map(point =>
      point.timestamp.toLocaleTimeString('en-US', {
        hour12: false,
        minute: '2-digit',
        second: '2-digit'
      })
    ),
    datasets: [
      {
        label: title,
        data: limitedData.map(point => point.value),
        borderColor: color,
        backgroundColor: type === 'bar' ? color : `${color}20`,
        borderWidth: 2,
        fill: type === 'line',
        tension: 0.4,
        pointRadius: type === 'line' ? 0 : 4,
        pointHoverRadius: 6,
      },
    ],
  };

  // Chart options
  const options: ChartOptions<'line' | 'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: animated ? {
      duration: realTime ? 750 : 1000,
      easing: 'easeInOutQuart',
    } : false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: title,
        font: {
          size: 14,
          weight: 'bold',
        },
        color: '#374151', // gray-700
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const value = context.parsed.y;
            return `${title}: ${value.toFixed(2)}${unit}`;
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: showGrid,
          color: '#f3f4f6', // gray-100
        },
        ticks: {
          maxTicksLimit: 8,
          color: '#6b7280', // gray-500
        },
      },
      y: {
        display: true,
        grid: {
          display: showGrid,
          color: '#f3f4f6', // gray-100
        },
        ticks: {
          color: '#6b7280', // gray-500
          callback: (value) => `${value}${unit}`,
        },
        min: yAxisMin,
        max: yAxisMax,
      },
    },
    elements: {
      point: {
        hoverBorderWidth: 3,
      },
    },
  };

  // Update chart when data changes (for real-time updates)
  useEffect(() => {
    if (realTime && chartRef.current) {
      const chart = chartRef.current;
      chart.data = chartData;
      chart.update('none'); // Update without animation for real-time
    }
  }, [data, realTime]);

  const ChartComponent = type === 'line' ? Line : Bar;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
      <div style={{ height: `${height}px` }}>
        <ChartComponent
          ref={chartRef}
          data={chartData}
          options={options}
        />
      </div>

      {/* Chart statistics */}
      {data.length > 0 && (
        <div className="mt-3 flex justify-between items-center text-sm text-gray-600 dark:text-gray-400">
          <div className="flex space-x-4">
            <span>
              Current: <span className="font-medium">{data[data.length - 1]?.value.toFixed(2)}{unit}</span>
            </span>
            <span>
              Avg: <span className="font-medium">
                {(data.reduce((sum, point) => sum + point.value, 0) / data.length).toFixed(2)}{unit}
              </span>
            </span>
            <span>
              Max: <span className="font-medium">{Math.max(...data.map(p => p.value)).toFixed(2)}{unit}</span>
            </span>
          </div>
          <span className="text-xs">
            {data.length} points
          </span>
        </div>
      )}
    </div>
  );
};

export default MetricsChart;
