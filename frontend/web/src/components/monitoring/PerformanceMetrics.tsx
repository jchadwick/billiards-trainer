/**
 * Performance metrics component showing detailed system performance data
 * Displays CPU, memory, disk, network, and processing speed metrics with charts
 */

import React, { useEffect, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { MetricsChart } from './MetricsChart';
import type { MetricPoint } from '../../types/monitoring';
import { StatCard } from './StatCard';
import { ProgressBar } from './ProgressBar';
import type { SystemMetrics } from '../../types/api';

interface PerformanceData {
  cpu: MetricPoint[];
  memory: MetricPoint[];
  disk: MetricPoint[];
  networkIn: MetricPoint[];
  networkOut: MetricPoint[];
  apiLatency: MetricPoint[];
  frameRate: MetricPoint[];
  processingTime: MetricPoint[];
}

export const PerformanceMetrics: React.FC = observer(() => {
  const { connectionStore } = useStores();
  const [performanceData, setPerformanceData] = useState<PerformanceData>({
    cpu: [],
    memory: [],
    disk: [],
    networkIn: [],
    networkOut: [],
    apiLatency: [],
    frameRate: [],
    processingTime: [],
  });
  const [currentMetrics, setCurrentMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPerformanceData();
    const interval = setInterval(loadPerformanceData, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const loadPerformanceData = async () => {
    try {
      const now = new Date();

      // Simulate realistic performance metrics
      const metrics: SystemMetrics = {
        cpu_usage: 35 + Math.sin(Date.now() / 10000) * 20 + Math.random() * 10,
        memory_usage: 65 + Math.cos(Date.now() / 15000) * 15 + Math.random() * 5,
        disk_usage: 45 + Math.random() * 2, // Disk usage changes slowly
        network_io: {
          bytes_sent: 50000 + Math.random() * 100000,
          bytes_received: 150000 + Math.random() * 200000,
        },
        api_requests_per_second: 8 + Math.random() * 12,
        websocket_connections: connectionStore.state.isConnected ? 2 + Math.floor(Math.random() * 3) : 0,
        average_response_time: 45 + Math.random() * 80,
      };

      setCurrentMetrics(metrics);

      // Additional performance metrics specific to billiards system
      const frameRate = 25 + Math.random() * 5; // 25-30 FPS
      const processingTime = 15 + Math.random() * 10; // 15-25ms processing time

      setPerformanceData(prev => ({
        cpu: [...prev.cpu.slice(-49), { timestamp: now, value: metrics.cpu_usage }],
        memory: [...prev.memory.slice(-49), { timestamp: now, value: metrics.memory_usage }],
        disk: [...prev.disk.slice(-49), { timestamp: now, value: metrics.disk_usage }],
        networkIn: [...prev.networkIn.slice(-49), { timestamp: now, value: metrics.network_io.bytes_received / 1024 }], // KB
        networkOut: [...prev.networkOut.slice(-49), { timestamp: now, value: metrics.network_io.bytes_sent / 1024 }], // KB
        apiLatency: [...prev.apiLatency.slice(-49), { timestamp: now, value: metrics.average_response_time }],
        frameRate: [...prev.frameRate.slice(-49), { timestamp: now, value: frameRate }],
        processingTime: [...prev.processingTime.slice(-49), { timestamp: now, value: processingTime }],
      }));

      setLoading(false);
    } catch (error) {
      console.error('Failed to load performance data:', error);
      setLoading(false);
    }
  };

  const getLastValue = (data: MetricPoint[]): number => {
    return data.length > 0 ? data[data.length - 1].value : 0;
  };

  const getAverage = (data: MetricPoint[]): number => {
    if (data.length === 0) return 0;
    return data.reduce((sum, point) => sum + point.value, 0) / data.length;
  };

  const getPerformanceColor = (value: number, thresholds: { good: number; warning: number }): 'green' | 'yellow' | 'red' => {
    if (value <= thresholds.good) return 'green';
    if (value <= thresholds.warning) return 'yellow';
    return 'red';
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <StatCard key={i} title="Loading..." value="" loading={true} />
          ))}
        </div>
      </div>
    );
  }

  const currentCpu = getLastValue(performanceData.cpu);
  const currentMemory = getLastValue(performanceData.memory);
  const currentFrameRate = getLastValue(performanceData.frameRate);
  const currentProcessingTime = getLastValue(performanceData.processingTime);

  return (
    <div className="space-y-6">
      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="CPU Usage"
          value={currentCpu.toFixed(1)}
          unit="%"
          icon="💻"
          color={getPerformanceColor(currentCpu, { good: 60, warning: 80 })}
          change={currentCpu - getAverage(performanceData.cpu.slice(-10))}
          changeLabel="vs 10-point avg"
        />

        <StatCard
          title="Memory Usage"
          value={currentMemory.toFixed(1)}
          unit="%"
          icon="🧠"
          color={getPerformanceColor(currentMemory, { good: 70, warning: 85 })}
          change={currentMemory - getAverage(performanceData.memory.slice(-10))}
          changeLabel="vs 10-point avg"
        />

        <StatCard
          title="Frame Rate"
          value={currentFrameRate.toFixed(1)}
          unit="FPS"
          icon="🎥"
          color={currentFrameRate >= 25 ? 'green' : currentFrameRate >= 20 ? 'yellow' : 'red'}
          change={((currentFrameRate - 30) / 30) * 100}
          changeLabel="vs target 30 FPS"
        />

        <StatCard
          title="Processing Time"
          value={currentProcessingTime.toFixed(1)}
          unit="ms"
          icon="⚡"
          color={currentProcessingTime <= 20 ? 'green' : currentProcessingTime <= 35 ? 'yellow' : 'red'}
          change={((30 - currentProcessingTime) / 30) * 100}
          changeLabel="vs target 30ms"
        />
      </div>

      {/* System Resource Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsChart
          title="CPU Usage Over Time"
          data={performanceData.cpu}
          unit="%"
          color="rgb(239, 68, 68)" // red-500
          yAxisMax={100}
          yAxisMin={0}
          height={250}
        />

        <MetricsChart
          title="Memory Usage Over Time"
          data={performanceData.memory}
          unit="%"
          color="rgb(34, 197, 94)" // green-500
          yAxisMax={100}
          yAxisMin={0}
          height={250}
        />
      </div>

      {/* Network and API Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Network Traffic
          </h3>
          <div className="space-y-4">
            <MetricsChart
              title="Network In (KB/s)"
              data={performanceData.networkIn}
              unit=" KB/s"
              color="rgb(59, 130, 246)" // blue-500
              height={200}
              showGrid={false}
            />
            <MetricsChart
              title="Network Out (KB/s)"
              data={performanceData.networkOut}
              unit=" KB/s"
              color="rgb(168, 85, 247)" // purple-500
              height={200}
              showGrid={false}
            />
          </div>
        </div>

        <MetricsChart
          title="API Response Time"
          data={performanceData.apiLatency}
          unit="ms"
          color="rgb(245, 158, 11)" // amber-500
          yAxisMin={0}
          height={250}
        />
      </div>

      {/* Vision System Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsChart
          title="Video Frame Rate"
          data={performanceData.frameRate}
          unit=" FPS"
          color="rgb(20, 184, 166)" // teal-500
          yAxisMin={15}
          yAxisMax={35}
          height={250}
        />

        <MetricsChart
          title="Frame Processing Time"
          data={performanceData.processingTime}
          unit="ms"
          color="rgb(217, 70, 239)" // fuchsia-500
          yAxisMin={0}
          yAxisMax={50}
          height={250}
        />
      </div>

      {/* Resource Usage Bars */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Current Resource Usage
        </h3>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">CPU</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">{currentCpu.toFixed(1)}%</span>
            </div>
            <ProgressBar
              value={currentCpu}
              color={getPerformanceColor(currentCpu, { good: 60, warning: 80 })}
              animated={true}
            />
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Memory</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">{currentMemory.toFixed(1)}%</span>
            </div>
            <ProgressBar
              value={currentMemory}
              color={getPerformanceColor(currentMemory, { good: 70, warning: 85 })}
              animated={true}
            />
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Disk</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">{getLastValue(performanceData.disk).toFixed(1)}%</span>
            </div>
            <ProgressBar
              value={getLastValue(performanceData.disk)}
              color={getPerformanceColor(getLastValue(performanceData.disk), { good: 70, warning: 85 })}
              animated={false}
            />
          </div>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="API Requests/sec"
          value={currentMetrics?.api_requests_per_second.toFixed(1) || '0'}
          icon="📊"
          color="blue"
        />

        <StatCard
          title="WebSocket Connections"
          value={currentMetrics?.websocket_connections || 0}
          icon="🔗"
          color="purple"
        />

        <StatCard
          title="System Load"
          value={((currentCpu + currentMemory) / 2).toFixed(1)}
          unit="%"
          icon="⚖️"
          color={getPerformanceColor((currentCpu + currentMemory) / 2, { good: 60, warning: 80 })}
        />
      </div>
    </div>
  );
});

export default PerformanceMetrics;
