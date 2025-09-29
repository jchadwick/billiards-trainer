/**
 * Health check component providing system health diagnostic tools
 * Includes automated health tests and manual diagnostic controls
 */

import React, { useState, useEffect } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { StatusIndicator } from './StatusIndicator';
import { ProgressBar } from './ProgressBar';
import { StatCard } from './StatCard';
import type { ComponentHealth, HealthStatus } from '../../types/api';

interface HealthTest {
  id: string;
  name: string;
  description: string;
  category: 'connectivity' | 'performance' | 'integrity' | 'security';
  status: 'pending' | 'running' | 'passed' | 'failed' | 'warning';
  progress: number;
  duration?: number;
  lastRun?: Date;
  result?: {
    message: string;
    details?: Record<string, any>;
    recommendations?: string[];
  };
}

interface DiagnosticSuite {
  id: string;
  name: string;
  description: string;
  tests: HealthTest[];
  isRunning: boolean;
  progress: number;
}

export const HealthCheck: React.FC = observer(() => {
  const { connectionStore } = useStores();
  const [diagnosticSuites, setDiagnosticSuites] = useState<DiagnosticSuite[]>([]);
  const [systemHealth, setSystemHealth] = useState<ComponentHealth[]>([]);
  const [selectedSuite, setSelectedSuite] = useState<string | null>(null);
  const [isRunningFullDiagnostic, setIsRunningFullDiagnostic] = useState(false);
  const [lastHealthCheck, setLastHealthCheck] = useState<Date | null>(null);

  useEffect(() => {
    initializeDiagnostics();
    loadSystemHealth();
  }, []);

  const initializeDiagnostics = () => {
    const suites: DiagnosticSuite[] = [
      {
        id: 'connectivity',
        name: 'Connectivity Tests',
        description: 'Test network connectivity and service availability',
        isRunning: false,
        progress: 0,
        tests: [
          {
            id: 'api-health',
            name: 'API Health Check',
            description: 'Verify API server is responding correctly',
            category: 'connectivity',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'websocket-connection',
            name: 'WebSocket Connection',
            description: 'Test WebSocket connectivity and message handling',
            category: 'connectivity',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'database-connection',
            name: 'Database Connection',
            description: 'Verify database connectivity and query performance',
            category: 'connectivity',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'external-services',
            name: 'External Services',
            description: 'Check connectivity to external APIs and services',
            category: 'connectivity',
            status: 'pending',
            progress: 0,
          },
        ],
      },
      {
        id: 'performance',
        name: 'Performance Tests',
        description: 'Evaluate system performance and resource utilization',
        isRunning: false,
        progress: 0,
        tests: [
          {
            id: 'cpu-stress',
            name: 'CPU Performance',
            description: 'Test CPU performance under load',
            category: 'performance',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'memory-usage',
            name: 'Memory Usage',
            description: 'Check memory allocation and garbage collection',
            category: 'performance',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'disk-io',
            name: 'Disk I/O',
            description: 'Test disk read/write performance',
            category: 'performance',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'frame-processing',
            name: 'Frame Processing',
            description: 'Benchmark video frame processing speed',
            category: 'performance',
            status: 'pending',
            progress: 0,
          },
        ],
      },
      {
        id: 'integrity',
        name: 'System Integrity',
        description: 'Verify system components and data integrity',
        isRunning: false,
        progress: 0,
        tests: [
          {
            id: 'config-validation',
            name: 'Configuration Validation',
            description: 'Validate system configuration settings',
            category: 'integrity',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'calibration-accuracy',
            name: 'Calibration Accuracy',
            description: 'Verify camera-projector calibration accuracy',
            category: 'integrity',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'data-consistency',
            name: 'Data Consistency',
            description: 'Check database data consistency and integrity',
            category: 'integrity',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'file-integrity',
            name: 'File Integrity',
            description: 'Verify system files and assets integrity',
            category: 'integrity',
            status: 'pending',
            progress: 0,
          },
        ],
      },
      {
        id: 'security',
        name: 'Security Tests',
        description: 'Perform security checks and vulnerability assessment',
        isRunning: false,
        progress: 0,
        tests: [
          {
            id: 'authentication',
            name: 'Authentication System',
            description: 'Test user authentication and authorization',
            category: 'security',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'ssl-certificates',
            name: 'SSL Certificates',
            description: 'Verify SSL certificate validity and security',
            category: 'security',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'access-controls',
            name: 'Access Controls',
            description: 'Test role-based access control implementation',
            category: 'security',
            status: 'pending',
            progress: 0,
          },
          {
            id: 'data-encryption',
            name: 'Data Encryption',
            description: 'Verify data encryption at rest and in transit',
            category: 'security',
            status: 'pending',
            progress: 0,
          },
        ],
      },
    ];

    setDiagnosticSuites(suites);
  };

  const loadSystemHealth = async () => {
    // Simulate loading system health from API
    const mockHealth: ComponentHealth[] = [
      {
        name: 'API Server',
        status: connectionStore.state.isConnected ? 'healthy' : 'unhealthy',
        message: connectionStore.state.isConnected ? 'All endpoints responding' : 'Server unreachable',
        last_check: new Date().toISOString(),
        uptime: connectionStore.state.isConnected ? 3600 : 0,
        errors: connectionStore.state.error ? [connectionStore.state.error] : [],
      },
      {
        name: 'Database',
        status: 'healthy',
        message: 'Connection pool healthy',
        last_check: new Date().toISOString(),
        uptime: 3600,
        errors: [],
      },
      {
        name: 'Vision System',
        status: 'healthy',
        message: 'Camera operational, tracking active',
        last_check: new Date().toISOString(),
        uptime: 3580,
        errors: [],
      },
      {
        name: 'Projector',
        status: 'degraded',
        message: 'Minor calibration drift detected',
        last_check: new Date().toISOString(),
        uptime: 3400,
        errors: ['Calibration accuracy: 95%'],
      },
    ];

    setSystemHealth(mockHealth);
    setLastHealthCheck(new Date());
  };

  const runDiagnosticSuite = async (suiteId: string) => {
    setDiagnosticSuites(prev =>
      prev.map(suite =>
        suite.id === suiteId
          ? { ...suite, isRunning: true, progress: 0 }
          : suite
      )
    );

    const suite = diagnosticSuites.find(s => s.id === suiteId);
    if (!suite) return;

    for (let i = 0; i < suite.tests.length; i++) {
      const test = suite.tests[i];

      // Update test status to running
      setDiagnosticSuites(prev =>
        prev.map(s =>
          s.id === suiteId
            ? {
                ...s,
                progress: (i / s.tests.length) * 100,
                tests: s.tests.map(t =>
                  t.id === test.id
                    ? { ...t, status: 'running', progress: 0 }
                    : t
                )
              }
            : s
        )
      );

      // Simulate test execution with progress
      await runSingleTest(suiteId, test.id);
    }

    // Mark suite as complete
    setDiagnosticSuites(prev =>
      prev.map(suite =>
        suite.id === suiteId
          ? { ...suite, isRunning: false, progress: 100 }
          : suite
      )
    );
  };

  const runSingleTest = async (suiteId: string, testId: string) => {
    const duration = 2000 + Math.random() * 3000; // 2-5 seconds
    const startTime = Date.now();

    return new Promise<void>((resolve) => {
      const updateProgress = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min((elapsed / duration) * 100, 100);

        setDiagnosticSuites(prev =>
          prev.map(s =>
            s.id === suiteId
              ? {
                  ...s,
                  tests: s.tests.map(t =>
                    t.id === testId
                      ? { ...t, progress }
                      : t
                  )
                }
              : s
          )
        );

        if (progress < 100) {
          setTimeout(updateProgress, 100);
        } else {
          // Complete the test with random result
          const success = Math.random() > 0.2; // 80% success rate
          const status = success ? 'passed' : Math.random() > 0.5 ? 'failed' : 'warning';

          setDiagnosticSuites(prev =>
            prev.map(s =>
              s.id === suiteId
                ? {
                    ...s,
                    tests: s.tests.map(t =>
                      t.id === testId
                        ? {
                            ...t,
                            status,
                            progress: 100,
                            duration: duration,
                            lastRun: new Date(),
                            result: {
                              message: status === 'passed' ? 'Test completed successfully' :
                                      status === 'warning' ? 'Test completed with warnings' :
                                      'Test failed - issues detected',
                              details: {
                                executionTime: `${duration}ms`,
                                result: status === 'passed' ? 'OK' : 'FAILED',
                              },
                              recommendations: status !== 'passed' ? [
                                'Review system logs for detailed error information',
                                'Consider restarting the affected service',
                                'Contact support if issues persist',
                              ] : undefined,
                            }
                          }
                        : t
                    )
                  }
                : s
            )
          );
          resolve();
        }
      };

      updateProgress();
    });
  };

  const runFullDiagnostic = async () => {
    setIsRunningFullDiagnostic(true);

    for (const suite of diagnosticSuites) {
      await runDiagnosticSuite(suite.id);
    }

    setIsRunningFullDiagnostic(false);
    await loadSystemHealth();
  };

  const getTestStatusColor = (status: HealthTest['status']): 'green' | 'yellow' | 'red' | 'blue' | 'gray' => {
    switch (status) {
      case 'passed': return 'green';
      case 'warning': return 'yellow';
      case 'failed': return 'red';
      case 'running': return 'blue';
      default: return 'gray';
    }
  };

  const getTestStatusIcon = (status: HealthTest['status']): string => {
    switch (status) {
      case 'passed': return '✅';
      case 'warning': return '⚠️';
      case 'failed': return '❌';
      case 'running': return '🔄';
      default: return '⏸️';
    }
  };

  const getOverallHealth = (): HealthStatus => {
    const unhealthyCount = systemHealth.filter(h => h.status === 'unhealthy').length;
    const degradedCount = systemHealth.filter(h => h.status === 'degraded').length;

    if (unhealthyCount > 0) return 'unhealthy';
    if (degradedCount > 0) return 'degraded';
    return 'healthy';
  };

  return (
    <div className="space-y-6">
      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Overall Health"
          value={getOverallHealth()}
          icon="🏥"
          color={getOverallHealth() === 'healthy' ? 'green' :
                 getOverallHealth() === 'degraded' ? 'yellow' : 'red'}
        />

        <StatCard
          title="Components"
          value={systemHealth.filter(h => h.status === 'healthy').length}
          unit={` / ${systemHealth.length} healthy`}
          icon="⚙️"
          color="blue"
        />

        <StatCard
          title="Last Check"
          value={lastHealthCheck ? lastHealthCheck.toLocaleTimeString() : 'Never'}
          icon="🕐"
          color="purple"
        />

        <StatCard
          title="Uptime"
          value="3h 45m"
          icon="⏱️"
          color="green"
        />
      </div>

      {/* Quick Actions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Health Diagnostics
          </h3>
          <div className="flex space-x-3">
            <button
              onClick={loadSystemHealth}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
            >
              Quick Health Check
            </button>
            <button
              onClick={runFullDiagnostic}
              disabled={isRunningFullDiagnostic}
              className={`px-4 py-2 rounded-md transition-colors ${
                isRunningFullDiagnostic
                  ? 'bg-gray-400 cursor-not-allowed text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isRunningFullDiagnostic ? 'Running...' : 'Full Diagnostic'}
            </button>
          </div>
        </div>

        {/* Component Health Status */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {systemHealth.map((component) => (
            <div key={component.name} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                  {component.name}
                </h4>
                <StatusIndicator status={component.status} size="sm" showLabel={false} />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {component.message}
              </p>
              {component.errors.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-red-600 dark:text-red-400">
                    {component.errors[0]}
                  </p>
                </div>
              )}
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Last check: {new Date(component.last_check).toLocaleTimeString()}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Diagnostic Suites */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {diagnosticSuites.map((suite) => (
          <div key={suite.id} className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  {suite.name}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  {suite.description}
                </p>
              </div>
              <button
                onClick={() => runDiagnosticSuite(suite.id)}
                disabled={suite.isRunning}
                className={`px-3 py-2 text-sm rounded-md transition-colors ${
                  suite.isRunning
                    ? 'bg-gray-400 cursor-not-allowed text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {suite.isRunning ? 'Running...' : 'Run Tests'}
              </button>
            </div>

            {suite.isRunning && (
              <div className="mb-4">
                <ProgressBar
                  value={suite.progress}
                  label="Suite Progress"
                  color="blue"
                  animated={true}
                />
              </div>
            )}

            <div className="space-y-3">
              {suite.tests.map((test) => (
                <div key={test.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span>{getTestStatusIcon(test.status)}</span>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {test.name}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">
                      {test.description}
                    </p>
                    {test.status === 'running' && (
                      <div className="mt-2">
                        <ProgressBar
                          value={test.progress}
                          size="sm"
                          color="blue"
                          showPercentage={false}
                          animated={true}
                        />
                      </div>
                    )}
                    {test.result && (
                      <div className="mt-2">
                        <p className={`text-xs ${
                          test.status === 'passed' ? 'text-green-600' :
                          test.status === 'warning' ? 'text-yellow-600' :
                          'text-red-600'
                        }`}>
                          {test.result.message}
                        </p>
                        {test.lastRun && (
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            Completed: {test.lastRun.toLocaleTimeString()}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="ml-4">
                    <StatCard
                      title=""
                      value={test.status}
                      size="sm"
                      color={getTestStatusColor(test.status)}
                      className="min-w-20"
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Suite Summary */}
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
              <div className="grid grid-cols-4 gap-2 text-center">
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">Passed</p>
                  <p className="text-sm font-medium text-green-600">
                    {suite.tests.filter(t => t.status === 'passed').length}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">Warning</p>
                  <p className="text-sm font-medium text-yellow-600">
                    {suite.tests.filter(t => t.status === 'warning').length}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">Failed</p>
                  <p className="text-sm font-medium text-red-600">
                    {suite.tests.filter(t => t.status === 'failed').length}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">Pending</p>
                  <p className="text-sm font-medium text-gray-600">
                    {suite.tests.filter(t => t.status === 'pending').length}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});

export default HealthCheck;
