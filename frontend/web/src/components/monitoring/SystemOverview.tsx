/**
 * System overview component showing high-level system status and key metrics
 * Provides a dashboard summary with critical system information
 */

import React, { useEffect, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { StatCard } from './StatCard';
import { StatusIndicator } from './StatusIndicator';
import { AlertPanel } from './AlertPanel';
import { MetricsChart } from './MetricsChart';
import { apiClient } from '../../api/client';
import type { Alert, MetricPoint } from '../../types/monitoring';
import type { HealthResponse, SystemMetrics } from '../../types/api';

export const SystemOverview: React.FC = observer(() => {
  const { connectionStore, systemStore } = useStores();
  const [systemHealth, setSystemHealth] = useState<HealthResponse | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [cpuHistory, setCpuHistory] = useState<MetricPoint[]>([]);
  const [memoryHistory, setMemoryHistory] = useState<MetricPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSystemData();
    const interval = setInterval(loadSystemData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadSystemData = async () => {
    try {
      // Fetch real health and metrics data from backend
      const [healthResponse, metricsResponse] = await Promise.all([
        apiClient.get('/health?include_details=true&include_metrics=true'),
        apiClient.get('/health/metrics')
      ]);

      setSystemHealth(healthResponse);
      setSystemMetrics(metricsResponse);

      // Update CPU and memory history
      const now = new Date();
      setCpuHistory(prev => [
        ...prev.slice(-49), // Keep last 49 points
        { timestamp: now, value: metricsResponse.cpu_usage }
      ]);
      setMemoryHistory(prev => [
        ...prev.slice(-49),
        { timestamp: now, value: metricsResponse.memory_usage }
      ]);

      // Generate alerts based on system status
      updateAlerts(healthResponse, metricsResponse);

      setLoading(false);
    } catch (error) {
      console.error('Failed to load system data:', error);

      // Fallback to minimal data based on connection state
      const fallbackHealth: HealthResponse = {
        status: connectionStore.state.isConnected ? 'healthy' : 'unhealthy',
        timestamp: new Date().toISOString(),
        uptime: 0,
        version: '1.0.0',
        components: {
          api: {
            name: 'API Server',
            status: connectionStore.state.isConnected ? 'healthy' : 'unhealthy',
            message: connectionStore.state.isConnected ? 'Connected' : 'Disconnected',
            last_check: new Date().toISOString(),
            uptime: 0,
            errors: connectionStore.state.error ? [connectionStore.state.error] : [],
          },
        },
      };

      const fallbackMetrics: SystemMetrics = {
        cpu_usage: 0,
        memory_usage: 0,
        disk_usage: 0,
        network_io: {},
        api_requests_per_second: 0,
        websocket_connections: connectionStore.state.isConnected ? 1 : 0,
        average_response_time: 0,
      };

      setSystemHealth(fallbackHealth);
      setSystemMetrics(fallbackMetrics);
      setLoading(false);
    }
  };

  const updateAlerts = (health: HealthResponse, metrics: SystemMetrics) => {
    const newAlerts: Alert[] = [];

    // High CPU usage alert
    if (metrics.cpu_usage > 80) {
      newAlerts.push({
        id: 'high-cpu',
        level: 'warning',
        title: 'High CPU Usage',
        message: `CPU usage is ${metrics.cpu_usage.toFixed(1)}%`,
        timestamp: new Date(),
      });
    }

    // High memory usage alert
    if (metrics.memory_usage > 85) {
      newAlerts.push({
        id: 'high-memory',
        level: 'warning',
        title: 'High Memory Usage',
        message: `Memory usage is ${metrics.memory_usage.toFixed(1)}%`,
        timestamp: new Date(),
      });
    }

    // Component health alerts
    Object.entries(health.components).forEach(([key, component]) => {
      if (component.status === 'unhealthy') {
        newAlerts.push({
          id: `component-${key}`,
          level: 'error',
          title: `${component.name} Unhealthy`,
          message: component.message || 'Component is not responding',
          timestamp: new Date(),
        });
      } else if (component.status === 'degraded') {
        newAlerts.push({
          id: `component-${key}`,
          level: 'warning',
          title: `${component.name} Degraded`,
          message: component.message || 'Component performance is degraded',
          timestamp: new Date(),
        });
      }
    });

    setAlerts(newAlerts);
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <StatCard
              key={i}
              title="Loading..."
              value=""
              loading={true}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="System Status"
          value={systemHealth?.status || 'unknown'}
          icon="🖥️"
          color={systemHealth?.status === 'healthy' ? 'green' :
                 systemHealth?.status === 'degraded' ? 'yellow' : 'red'}
        />

        <StatCard
          title="Uptime"
          value={systemHealth ? formatUptime(systemHealth.uptime) : '0h 0m'}
          icon="⏱️"
          color="blue"
        />

        <StatCard
          title="Active Connections"
          value={systemMetrics?.websocket_connections || 0}
          icon="🔗"
          color="purple"
        />

        <StatCard
          title="API Response Time"
          value={systemMetrics?.average_response_time.toFixed(0) || '0'}
          unit="ms"
          icon="⚡"
          color={systemMetrics && systemMetrics.average_response_time < 100 ? 'green' :
                 systemMetrics && systemMetrics.average_response_time < 200 ? 'yellow' : 'red'}
        />
      </div>

      {/* Resource Usage Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsChart
          title="CPU Usage"
          data={cpuHistory}
          unit="%"
          color="rgb(59, 130, 246)" // blue
          yAxisMax={100}
          yAxisMin={0}
        />

        <MetricsChart
          title="Memory Usage"
          data={memoryHistory}
          unit="%"
          color="rgb(16, 185, 129)" // green
          yAxisMax={100}
          yAxisMin={0}
        />
      </div>

      {/* Component Status */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Component Status
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {systemHealth && Object.entries(systemHealth.components).map(([key, component]) => (
            <div key={key} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center">
                <StatusIndicator
                  status={component.status}
                  size="sm"
                  showLabel={false}
                />
                <span className="ml-2 text-sm font-medium text-gray-900 dark:text-white">
                  {component.name}
                </span>
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {component.uptime ? formatUptime(component.uptime) : '—'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="CPU Usage"
          value={systemMetrics?.cpu_usage.toFixed(1) || '0'}
          unit="%"
          icon="💻"
          color={systemMetrics && systemMetrics.cpu_usage < 70 ? 'green' :
                 systemMetrics && systemMetrics.cpu_usage < 85 ? 'yellow' : 'red'}
        />

        <StatCard
          title="Memory Usage"
          value={systemMetrics?.memory_usage.toFixed(1) || '0'}
          unit="%"
          icon="🧠"
          color={systemMetrics && systemMetrics.memory_usage < 75 ? 'green' :
                 systemMetrics && systemMetrics.memory_usage < 90 ? 'yellow' : 'red'}
        />

        <StatCard
          title="Disk Usage"
          value={systemMetrics?.disk_usage.toFixed(1) || '0'}
          unit="%"
          icon="💾"
          color={systemMetrics && systemMetrics.disk_usage < 80 ? 'green' :
                 systemMetrics && systemMetrics.disk_usage < 90 ? 'yellow' : 'red'}
        />
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            System Alerts
          </h3>
          <AlertPanel alerts={alerts} />
        </div>
      )}
    </div>
  );
});

export default SystemOverview;
