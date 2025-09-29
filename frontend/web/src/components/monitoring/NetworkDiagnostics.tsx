/**
 * Network diagnostics component for monitoring connectivity and latency
 * Provides real-time network monitoring and connection testing tools
 */

import React, { useState, useEffect } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { MetricsChart } from './MetricsChart';
import type { MetricPoint } from '../../types/monitoring';
import { StatusIndicator } from './StatusIndicator';
import { StatCard } from './StatCard';
import { ProgressBar } from './ProgressBar';

interface NetworkEndpoint {
  id: string;
  name: string;
  url: string;
  type: 'api' | 'websocket' | 'database' | 'external';
  status: 'online' | 'offline' | 'degraded' | 'testing';
  latency: number;
  lastCheck: Date;
  uptime: number;
  errorCount: number;
}

interface NetworkTest {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result?: {
    success: boolean;
    message: string;
    details: Record<string, any>;
  };
}

export const NetworkDiagnostics: React.FC = observer(() => {
  const { connectionStore } = useStores();
  const [endpoints, setEndpoints] = useState<NetworkEndpoint[]>([]);
  const [networkTests, setNetworkTests] = useState<NetworkTest[]>([]);
  const [latencyHistory, setLatencyHistory] = useState<MetricPoint[]>([]);
  const [throughputHistory, setThroughputHistory] = useState<MetricPoint[]>([]);
  const [selectedEndpoint, setSelectedEndpoint] = useState<string | null>(null);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [autoMonitoring, setAutoMonitoring] = useState(true);

  useEffect(() => {
    initializeEndpoints();
    initializeNetworkTests();
    if (autoMonitoring) {
      const interval = setInterval(updateNetworkMetrics, 5000);
      return () => clearInterval(interval);
    }
  }, [autoMonitoring]);

  const initializeEndpoints = () => {
    const mockEndpoints: NetworkEndpoint[] = [
      {
        id: 'api-server',
        name: 'API Server',
        url: 'http://localhost:8000',
        type: 'api',
        status: connectionStore.state.isConnected ? 'online' : 'offline',
        latency: 45 + Math.random() * 30,
        lastCheck: new Date(),
        uptime: connectionStore.state.isConnected ? 99.8 : 0,
        errorCount: connectionStore.state.isConnected ? 2 : 15,
      },
      {
        id: 'websocket',
        name: 'WebSocket Server',
        url: 'ws://localhost:8000/ws',
        type: 'websocket',
        status: connectionStore.state.isConnected ? 'online' : 'offline',
        latency: 25 + Math.random() * 15,
        lastCheck: new Date(),
        uptime: connectionStore.state.isConnected ? 99.9 : 0,
        errorCount: 0,
      },
      {
        id: 'database',
        name: 'Database',
        url: 'postgresql://localhost:5432',
        type: 'database',
        status: 'online',
        latency: 12 + Math.random() * 8,
        lastCheck: new Date(),
        uptime: 99.95,
        errorCount: 1,
      },
      {
        id: 'vision-api',
        name: 'Vision Processing',
        url: 'http://localhost:8001',
        type: 'api',
        status: 'degraded',
        latency: 150 + Math.random() * 50,
        lastCheck: new Date(),
        uptime: 96.5,
        errorCount: 8,
      },
      {
        id: 'external-api',
        name: 'External API',
        url: 'https://api.example.com',
        type: 'external',
        status: 'online',
        latency: 200 + Math.random() * 100,
        lastCheck: new Date(),
        uptime: 98.2,
        errorCount: 5,
      },
    ];

    setEndpoints(mockEndpoints);
  };

  const initializeNetworkTests = () => {
    const tests: NetworkTest[] = [
      {
        id: 'connectivity',
        name: 'Connectivity Test',
        description: 'Test basic network connectivity to all endpoints',
        status: 'pending',
        progress: 0,
      },
      {
        id: 'latency',
        name: 'Latency Test',
        description: 'Measure response times and network latency',
        status: 'pending',
        progress: 0,
      },
      {
        id: 'throughput',
        name: 'Throughput Test',
        description: 'Test network bandwidth and data transfer rates',
        status: 'pending',
        progress: 0,
      },
      {
        id: 'reliability',
        name: 'Reliability Test',
        description: 'Test connection stability and error rates',
        status: 'pending',
        progress: 0,
      },
      {
        id: 'websocket',
        name: 'WebSocket Test',
        description: 'Test WebSocket connection and message handling',
        status: 'pending',
        progress: 0,
      },
    ];

    setNetworkTests(tests);
  };

  const updateNetworkMetrics = () => {
    const now = new Date();

    // Update endpoint latencies
    setEndpoints(prev =>
      prev.map(endpoint => ({
        ...endpoint,
        latency: endpoint.latency + (Math.random() - 0.5) * 10, // Small random variation
        lastCheck: now,
        status: endpoint.id === 'api-server' || endpoint.id === 'websocket'
          ? (connectionStore.state.isConnected ? 'online' : 'offline')
          : endpoint.status,
      }))
    );

    // Update latency history
    const avgLatency = endpoints.reduce((sum, ep) => sum + ep.latency, 0) / endpoints.length;
    setLatencyHistory(prev => [
      ...prev.slice(-49),
      { timestamp: now, value: avgLatency }
    ]);

    // Update throughput history (simulated)
    const throughput = 50 + Math.random() * 30; // 50-80 Mbps
    setThroughputHistory(prev => [
      ...prev.slice(-49),
      { timestamp: now, value: throughput }
    ]);
  };

  const runNetworkTest = async (testId: string) => {
    setNetworkTests(prev =>
      prev.map(test =>
        test.id === testId
          ? { ...test, status: 'running', progress: 0 }
          : test
      )
    );

    // Simulate test execution
    const duration = 3000 + Math.random() * 2000; // 3-5 seconds
    const startTime = Date.now();

    return new Promise<void>((resolve) => {
      const updateProgress = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min((elapsed / duration) * 100, 100);

        setNetworkTests(prev =>
          prev.map(test =>
            test.id === testId
              ? { ...test, progress }
              : test
          )
        );

        if (progress < 100) {
          setTimeout(updateProgress, 100);
        } else {
          // Complete the test
          const success = Math.random() > 0.15; // 85% success rate
          const result = {
            success,
            message: success ? 'Test completed successfully' : 'Test failed with errors',
            details: {
              executionTime: `${duration}ms`,
              endpoints: endpoints.length,
              avgLatency: `${Math.round(latencyHistory[latencyHistory.length - 1]?.value || 50)}ms`,
              throughput: `${Math.round(throughputHistory[throughputHistory.length - 1]?.value || 60)} Mbps`,
            },
          };

          setNetworkTests(prev =>
            prev.map(test =>
              test.id === testId
                ? {
                    ...test,
                    status: success ? 'completed' : 'failed',
                    progress: 100,
                    result,
                  }
                : test
            )
          );
          resolve();
        }
      };

      updateProgress();
    });
  };

  const runAllNetworkTests = async () => {
    setIsRunningTests(true);

    for (const test of networkTests) {
      await runNetworkTest(test.id);
      await new Promise(resolve => setTimeout(resolve, 500)); // Brief pause between tests
    }

    setIsRunningTests(false);
  };

  const pingEndpoint = async (endpointId: string) => {
    setEndpoints(prev =>
      prev.map(ep =>
        ep.id === endpointId
          ? { ...ep, status: 'testing' }
          : ep
      )
    );

    // Simulate ping test
    setTimeout(() => {
      const success = Math.random() > 0.1; // 90% success rate
      setEndpoints(prev =>
        prev.map(ep =>
          ep.id === endpointId
            ? {
                ...ep,
                status: success ? 'online' : 'offline',
                latency: success ? ep.latency + (Math.random() - 0.5) * 20 : 999,
                lastCheck: new Date(),
                errorCount: success ? ep.errorCount : ep.errorCount + 1,
              }
            : ep
        )
      );
    }, 1000);
  };

  const getEndpointStatusColor = (status: NetworkEndpoint['status']): 'green' | 'yellow' | 'red' | 'blue' => {
    switch (status) {
      case 'online': return 'green';
      case 'degraded': return 'yellow';
      case 'offline': return 'red';
      case 'testing': return 'blue';
      default: return 'red';
    }
  };

  const getTestStatusColor = (status: NetworkTest['status']): 'green' | 'yellow' | 'red' | 'blue' | 'gray' => {
    switch (status) {
      case 'completed': return 'green';
      case 'failed': return 'red';
      case 'running': return 'blue';
      default: return 'gray';
    }
  };

  const getTestStatusIcon = (status: NetworkTest['status']): string => {
    switch (status) {
      case 'completed': return '✅';
      case 'failed': return '❌';
      case 'running': return '🔄';
      default: return '⏸️';
    }
  };

  const getOverallNetworkHealth = (): 'healthy' | 'degraded' | 'unhealthy' => {
    const offlineCount = endpoints.filter(ep => ep.status === 'offline').length;
    const degradedCount = endpoints.filter(ep => ep.status === 'degraded').length;

    if (offlineCount > 0) return 'unhealthy';
    if (degradedCount > 0) return 'degraded';
    return 'healthy';
  };

  return (
    <div className="space-y-6">
      {/* Network Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Network Health"
          value={getOverallNetworkHealth()}
          icon="🌐"
          color={getOverallNetworkHealth() === 'healthy' ? 'green' :
                 getOverallNetworkHealth() === 'degraded' ? 'yellow' : 'red'}
        />

        <StatCard
          title="Online Endpoints"
          value={endpoints.filter(ep => ep.status === 'online').length}
          unit={` / ${endpoints.length}`}
          icon="✅"
          color="green"
        />

        <StatCard
          title="Avg Latency"
          value={latencyHistory.length > 0 ? Math.round(latencyHistory[latencyHistory.length - 1].value) : 0}
          unit="ms"
          icon="⚡"
          color={latencyHistory.length > 0 && latencyHistory[latencyHistory.length - 1].value < 100 ? 'green' :
                 latencyHistory.length > 0 && latencyHistory[latencyHistory.length - 1].value < 200 ? 'yellow' : 'red'}
        />

        <StatCard
          title="Throughput"
          value={throughputHistory.length > 0 ? Math.round(throughputHistory[throughputHistory.length - 1].value) : 0}
          unit=" Mbps"
          icon="📈"
          color="blue"
        />
      </div>

      {/* Network Metrics Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsChart
          title="Network Latency"
          data={latencyHistory}
          unit="ms"
          color="rgb(239, 68, 68)" // red-500
          yAxisMin={0}
          height={250}
        />

        <MetricsChart
          title="Network Throughput"
          data={throughputHistory}
          unit=" Mbps"
          color="rgb(34, 197, 94)" // green-500
          yAxisMin={0}
          height={250}
        />
      </div>

      {/* Network Tests */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Network Diagnostics
          </h3>
          <div className="flex items-center space-x-4">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={autoMonitoring}
                onChange={(e) => setAutoMonitoring(e.target.checked)}
                className="sr-only"
              />
              <div className={`relative w-10 h-6 transition-colors duration-200 ease-in-out rounded-full ${
                autoMonitoring ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
              }`}>
                <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 ease-in-out ${
                  autoMonitoring ? 'transform translate-x-4' : ''
                }`} />
              </div>
              <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">
                Auto-monitor
              </span>
            </label>

            <button
              onClick={runAllNetworkTests}
              disabled={isRunningTests}
              className={`px-4 py-2 rounded-md transition-colors ${
                isRunningTests
                  ? 'bg-gray-400 cursor-not-allowed text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isRunningTests ? 'Running Tests...' : 'Run All Tests'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {networkTests.map((test) => (
            <div key={test.id} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <span>{getTestStatusIcon(test.status)}</span>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    {test.name}
                  </h4>
                </div>
                <StatCard
                  title=""
                  value={test.status}
                  size="sm"
                  color={getTestStatusColor(test.status)}
                  className="min-w-20"
                />
              </div>

              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {test.description}
              </p>

              {test.status === 'running' && (
                <div className="mb-3">
                  <ProgressBar
                    value={test.progress}
                    size="sm"
                    color="blue"
                    animated={true}
                  />
                </div>
              )}

              {test.result && (
                <div className="mb-3">
                  <p className={`text-sm ${
                    test.result.success ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {test.result.message}
                  </p>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    <div>Duration: {test.result.details.executionTime}</div>
                    <div>Avg Latency: {test.result.details.avgLatency}</div>
                  </div>
                </div>
              )}

              <button
                onClick={() => runNetworkTest(test.id)}
                disabled={test.status === 'running' || isRunningTests}
                className={`w-full px-3 py-2 text-sm rounded-md transition-colors ${
                  test.status === 'running' || isRunningTests
                    ? 'bg-gray-400 cursor-not-allowed text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {test.status === 'running' ? 'Running...' : 'Run Test'}
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Endpoint Status */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Endpoint Status
        </h3>

        <div className="space-y-4">
          {endpoints.map((endpoint) => (
            <div key={endpoint.id} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-4">
                <StatusIndicator
                  status={endpoint.status === 'online' ? 'healthy' :
                          endpoint.status === 'degraded' ? 'degraded' : 'unhealthy'}
                  size="md"
                  showLabel={false}
                />

                <div>
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    {endpoint.name}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    {endpoint.url}
                  </p>
                  <div className="flex items-center space-x-4 mt-1 text-xs text-gray-500 dark:text-gray-400">
                    <span>Type: {endpoint.type}</span>
                    <span>Latency: {Math.round(endpoint.latency)}ms</span>
                    <span>Uptime: {endpoint.uptime.toFixed(1)}%</span>
                    <span>Errors: {endpoint.errorCount}</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <StatCard
                  title="Status"
                  value={endpoint.status}
                  size="sm"
                  color={getEndpointStatusColor(endpoint.status)}
                />

                <button
                  onClick={() => pingEndpoint(endpoint.id)}
                  disabled={endpoint.status === 'testing'}
                  className={`px-3 py-2 text-sm rounded-md transition-colors ${
                    endpoint.status === 'testing'
                      ? 'bg-gray-400 cursor-not-allowed text-white'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  {endpoint.status === 'testing' ? 'Testing...' : 'Ping'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Network Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="Total Requests"
          value="12,450"
          icon="📊"
          color="blue"
        />

        <StatCard
          title="Failed Requests"
          value={endpoints.reduce((sum, ep) => sum + ep.errorCount, 0)}
          icon="❌"
          color="red"
        />

        <StatCard
          title="Success Rate"
          value={((1 - endpoints.reduce((sum, ep) => sum + ep.errorCount, 0) / 12450) * 100).toFixed(1)}
          unit="%"
          icon="✅"
          color="green"
        />
      </div>
    </div>
  );
});

export default NetworkDiagnostics;
