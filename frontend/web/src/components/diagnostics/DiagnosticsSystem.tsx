/**
 * Comprehensive automated diagnostics system component
 * Provides comprehensive system validation, hardware detection, network testing,
 * performance analysis, interactive troubleshooting, and end-to-end validation
 *
 * Implements specifications FR-UI-086 to FR-UI-105
 */

import React, { useState, useEffect, useCallback } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { HardwareDiagnostics } from './HardwareDiagnostics';
import { NetworkDiagnostics } from './NetworkDiagnostics';
import { PerformanceAnalysis } from './PerformanceAnalysis';
import { TroubleshootingWizard } from './TroubleshootingWizard';
import { SystemValidation } from './SystemValidation';
import { DiagnosticReport } from './DiagnosticReport';
import { StatCard } from '../monitoring/StatCard';
import { ProgressBar } from '../monitoring/ProgressBar';
import { AlertPanel } from '../monitoring/AlertPanel';
import { StatusIndicator } from '../monitoring/StatusIndicator';

export type DiagnosticCategory =
  | 'hardware'
  | 'network'
  | 'performance'
  | 'troubleshooting'
  | 'system-validation'
  | 'overview';

export interface DiagnosticTest {
  id: string;
  name: string;
  description: string;
  category: DiagnosticCategory;
  priority: 'critical' | 'high' | 'medium' | 'low';
  status: 'pending' | 'running' | 'passed' | 'failed' | 'warning' | 'skipped';
  progress: number;
  duration?: number;
  startTime?: Date;
  endTime?: Date;
  result?: {
    success: boolean;
    message: string;
    details: Record<string, any>;
    recommendations?: string[];
    errorCode?: string;
  };
  dependencies?: string[];
  automated: boolean;
}

export interface DiagnosticSuite {
  id: string;
  name: string;
  description: string;
  category: DiagnosticCategory;
  tests: DiagnosticTest[];
  isRunning: boolean;
  progress: number;
  estimatedDuration: number;
  lastRun?: Date;
}

export interface SystemIssue {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: DiagnosticCategory;
  title: string;
  description: string;
  impact: string;
  recommendations: string[];
  autoFixAvailable: boolean;
  detectedAt: Date;
  relatedTests: string[];
}

export const DiagnosticsSystem: React.FC = observer(() => {
  const { systemStore, connectionStore } = useStores();
  const [activeCategory, setActiveCategory] = useState<DiagnosticCategory>('overview');
  const [diagnosticSuites, setDiagnosticSuites] = useState<DiagnosticSuite[]>([]);
  const [isRunningFullDiagnostic, setIsRunningFullDiagnostic] = useState(false);
  const [overallProgress, setOverallProgress] = useState(0);
  const [detectedIssues, setDetectedIssues] = useState<SystemIssue[]>([]);
  const [autoSchedulingEnabled, setAutoSchedulingEnabled] = useState(false);
  const [lastFullDiagnostic, setLastFullDiagnostic] = useState<Date | null>(null);
  const [diagnosticReport, setDiagnosticReport] = useState<any>(null);
  const [showReport, setShowReport] = useState(false);

  // Initialize diagnostic suites
  useEffect(() => {
    initializeDiagnosticSuites();
    loadPreviousResults();
  }, []);

  const initializeDiagnosticSuites = useCallback(() => {
    const suites: DiagnosticSuite[] = [
      {
        id: 'hardware',
        name: 'Hardware Diagnostics',
        description: 'Camera, projector, system resources, and audio validation',
        category: 'hardware',
        isRunning: false,
        progress: 0,
        estimatedDuration: 120, // 2 minutes
        tests: [
          {
            id: 'camera-detection',
            name: 'Camera Detection',
            description: 'Detect and validate camera hardware availability',
            category: 'hardware',
            priority: 'critical',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'camera-functionality',
            name: 'Camera Functionality Test',
            description: 'Test camera capture, resolution, and frame rate',
            category: 'hardware',
            priority: 'critical',
            status: 'pending',
            progress: 0,
            dependencies: ['camera-detection'],
            automated: true,
          },
          {
            id: 'projector-detection',
            name: 'Projector Detection',
            description: 'Detect and validate projector connectivity',
            category: 'hardware',
            priority: 'high',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'projector-calibration',
            name: 'Projector Calibration Validation',
            description: 'Verify projector calibration accuracy',
            category: 'hardware',
            priority: 'high',
            status: 'pending',
            progress: 0,
            dependencies: ['projector-detection'],
            automated: true,
          },
          {
            id: 'system-resources',
            name: 'System Resources Check',
            description: 'CPU, memory, disk space, and temperature monitoring',
            category: 'hardware',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'audio-system',
            name: 'Audio System Validation',
            description: 'Test audio input/output devices and functionality',
            category: 'hardware',
            priority: 'low',
            status: 'pending',
            progress: 0,
            automated: true,
          },
        ],
      },
      {
        id: 'network',
        name: 'Network Diagnostics',
        description: 'Connectivity, bandwidth, latency, and API endpoint testing',
        category: 'network',
        isRunning: false,
        progress: 0,
        estimatedDuration: 90, // 1.5 minutes
        tests: [
          {
            id: 'backend-connectivity',
            name: 'Backend Connectivity',
            description: 'Test API server connectivity and response times',
            category: 'network',
            priority: 'critical',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'websocket-connection',
            name: 'WebSocket Connection',
            description: 'Test real-time WebSocket connectivity and messaging',
            category: 'network',
            priority: 'critical',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'bandwidth-test',
            name: 'Bandwidth Measurement',
            description: 'Measure network bandwidth and data transfer rates',
            category: 'network',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'latency-test',
            name: 'Network Latency Test',
            description: 'Measure network latency and packet loss',
            category: 'network',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'api-endpoints',
            name: 'API Endpoints Validation',
            description: 'Test all critical API endpoints availability',
            category: 'network',
            priority: 'high',
            status: 'pending',
            progress: 0,
            dependencies: ['backend-connectivity'],
            automated: true,
          },
        ],
      },
      {
        id: 'performance',
        name: 'Performance Analysis',
        description: 'Frame rate, detection accuracy, memory usage, and bottleneck identification',
        category: 'performance',
        isRunning: false,
        progress: 0,
        estimatedDuration: 180, // 3 minutes
        tests: [
          {
            id: 'frame-rate-analysis',
            name: 'Frame Rate Analysis',
            description: 'Monitor and analyze video processing frame rates',
            category: 'performance',
            priority: 'high',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'detection-accuracy',
            name: 'Detection Accuracy Metrics',
            description: 'Test ball and cue detection accuracy',
            category: 'performance',
            priority: 'high',
            status: 'pending',
            progress: 0,
            dependencies: ['camera-functionality'],
            automated: true,
          },
          {
            id: 'memory-profiling',
            name: 'Memory Usage Profiling',
            description: 'Analyze memory allocation patterns and leaks',
            category: 'performance',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'bottleneck-identification',
            name: 'Performance Bottleneck Analysis',
            description: 'Identify system performance bottlenecks',
            category: 'performance',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'gpu-performance',
            name: 'GPU Performance Test',
            description: 'Test GPU acceleration and performance',
            category: 'performance',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
        ],
      },
      {
        id: 'system-validation',
        name: 'System Validation',
        description: 'End-to-end workflow testing and data integrity verification',
        category: 'system-validation',
        isRunning: false,
        progress: 0,
        estimatedDuration: 240, // 4 minutes
        tests: [
          {
            id: 'end-to-end-workflow',
            name: 'End-to-End Workflow Test',
            description: 'Test complete game session workflow',
            category: 'system-validation',
            priority: 'critical',
            status: 'pending',
            progress: 0,
            dependencies: ['camera-functionality', 'backend-connectivity'],
            automated: true,
          },
          {
            id: 'module-communication',
            name: 'Module Communication Validation',
            description: 'Test inter-module communication and data flow',
            category: 'system-validation',
            priority: 'high',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'data-integrity',
            name: 'Data Integrity Verification',
            description: 'Verify data consistency and integrity across modules',
            category: 'system-validation',
            priority: 'high',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'configuration-validation',
            name: 'Configuration Validation',
            description: 'Validate system configuration and settings',
            category: 'system-validation',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
          {
            id: 'security-validation',
            name: 'Security Validation',
            description: 'Test authentication and authorization systems',
            category: 'system-validation',
            priority: 'medium',
            status: 'pending',
            progress: 0,
            automated: true,
          },
        ],
      },
    ];

    setDiagnosticSuites(suites);
  }, []);

  const loadPreviousResults = useCallback(async () => {
    try {
      // Load previous diagnostic results from localStorage or API
      const saved = localStorage.getItem('billiards-diagnostics-results');
      if (saved) {
        const { lastRun, results } = JSON.parse(saved);
        setLastFullDiagnostic(new Date(lastRun));
        // Process saved results...
      }
    } catch (error) {
      console.error('Failed to load previous diagnostic results:', error);
    }
  }, []);

  const runFullDiagnostic = useCallback(async () => {
    setIsRunningFullDiagnostic(true);
    setOverallProgress(0);
    setDetectedIssues([]);

    try {
      const startTime = Date.now();
      const allTests = diagnosticSuites.flatMap(suite => suite.tests);
      let completedTests = 0;

      // Run each suite in sequence
      for (const suite of diagnosticSuites) {
        await runDiagnosticSuite(suite.id);
        completedTests += suite.tests.length;
        setOverallProgress((completedTests / allTests.length) * 100);
      }

      // Generate comprehensive report
      const report = generateDiagnosticReport();
      setDiagnosticReport(report);
      setLastFullDiagnostic(new Date());

      // Save results
      localStorage.setItem('billiards-diagnostics-results', JSON.stringify({
        lastRun: new Date().toISOString(),
        results: report,
      }));

    } catch (error) {
      console.error('Full diagnostic failed:', error);
    } finally {
      setIsRunningFullDiagnostic(false);
      setOverallProgress(100);
    }
  }, [diagnosticSuites]);

  const runDiagnosticSuite = useCallback(async (suiteId: string) => {
    // Implementation will be detailed in individual diagnostic components
    // This is a placeholder that delegates to category-specific components
    return Promise.resolve();
  }, []);

  const generateDiagnosticReport = useCallback(() => {
    const allTests = diagnosticSuites.flatMap(suite => suite.tests);
    const passed = allTests.filter(test => test.status === 'passed').length;
    const failed = allTests.filter(test => test.status === 'failed').length;
    const warnings = allTests.filter(test => test.status === 'warning').length;

    return {
      timestamp: new Date(),
      summary: {
        total: allTests.length,
        passed,
        failed,
        warnings,
        successRate: (passed / allTests.length) * 100,
      },
      issues: detectedIssues,
      suites: diagnosticSuites,
      systemInfo: {
        version: systemStore.version,
        uptime: systemStore.uptimeFormatted,
        health: systemStore.overallHealth,
      },
    };
  }, [diagnosticSuites, detectedIssues, systemStore]);

  const getOverallHealthStatus = () => {
    const criticalIssues = detectedIssues.filter(issue => issue.severity === 'critical');
    const highIssues = detectedIssues.filter(issue => issue.severity === 'high');

    if (criticalIssues.length > 0) return 'critical';
    if (highIssues.length > 0) return 'warning';
    return 'healthy';
  };

  const categories = [
    { id: 'overview', label: 'Overview', icon: '📊', description: 'System status overview' },
    { id: 'hardware', label: 'Hardware', icon: '🔧', description: 'Camera, projector, system resources' },
    { id: 'network', label: 'Network', icon: '🌐', description: 'Connectivity and bandwidth testing' },
    { id: 'performance', label: 'Performance', icon: '⚡', description: 'Frame rate and bottleneck analysis' },
    { id: 'troubleshooting', label: 'Troubleshooting', icon: '🔍', description: 'Interactive problem resolution' },
    { id: 'system-validation', label: 'Validation', icon: '✅', description: 'End-to-end system testing' },
  ] as const;

  const renderCategoryContent = () => {
    switch (activeCategory) {
      case 'overview':
        return renderOverview();
      case 'hardware':
        return <HardwareDiagnostics suites={diagnosticSuites.filter(s => s.category === 'hardware')} />;
      case 'network':
        return <NetworkDiagnostics suites={diagnosticSuites.filter(s => s.category === 'network')} />;
      case 'performance':
        return <PerformanceAnalysis suites={diagnosticSuites.filter(s => s.category === 'performance')} />;
      case 'troubleshooting':
        return <TroubleshootingWizard issues={detectedIssues} />;
      case 'system-validation':
        return <SystemValidation suites={diagnosticSuites.filter(s => s.category === 'system-validation')} />;
      default:
        return renderOverview();
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="System Health"
          value={getOverallHealthStatus()}
          icon="🏥"
          color={getOverallHealthStatus() === 'healthy' ? 'green' :
                 getOverallHealthStatus() === 'warning' ? 'yellow' : 'red'}
        />
        <StatCard
          title="Detected Issues"
          value={detectedIssues.length}
          icon="⚠️"
          color={detectedIssues.length === 0 ? 'green' : 'red'}
        />
        <StatCard
          title="Last Full Diagnostic"
          value={lastFullDiagnostic ? lastFullDiagnostic.toLocaleDateString() : 'Never'}
          icon="🕐"
          color="blue"
        />
        <StatCard
          title="Auto Scheduling"
          value={autoSchedulingEnabled ? 'Enabled' : 'Disabled'}
          icon="⏰"
          color={autoSchedulingEnabled ? 'green' : 'gray'}
        />
      </div>

      {/* Quick Actions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Automated Diagnostics
          </h3>
          <div className="flex space-x-3">
            <button
              onClick={runFullDiagnostic}
              disabled={isRunningFullDiagnostic}
              className={`px-4 py-2 rounded-md transition-colors ${
                isRunningFullDiagnostic
                  ? 'bg-gray-400 cursor-not-allowed text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isRunningFullDiagnostic ? 'Running Diagnostics...' : 'Run Full Diagnostic'}
            </button>
            {diagnosticReport && (
              <button
                onClick={() => setShowReport(true)}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors"
              >
                View Report
              </button>
            )}
          </div>
        </div>

        {isRunningFullDiagnostic && (
          <div className="mb-4">
            <ProgressBar
              value={overallProgress}
              label="Overall Progress"
              color="blue"
              animated={true}
            />
          </div>
        )}

        {/* Auto Scheduling Toggle */}
        <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div>
            <h4 className="text-sm font-medium text-gray-900 dark:text-white">
              Automated Diagnostic Scheduling
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Run diagnostics automatically on a scheduled basis
            </p>
          </div>
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={autoSchedulingEnabled}
              onChange={(e) => setAutoSchedulingEnabled(e.target.checked)}
              className="sr-only"
            />
            <div className={`relative w-10 h-6 transition-colors duration-200 ease-in-out rounded-full ${
              autoSchedulingEnabled ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
            }`}>
              <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 ease-in-out ${
                autoSchedulingEnabled ? 'transform translate-x-4' : ''
              }`} />
            </div>
          </label>
        </div>
      </div>

      {/* Detected Issues */}
      {detectedIssues.length > 0 && (
        <AlertPanel
          alerts={detectedIssues.map(issue => ({
            id: issue.id,
            type: issue.severity === 'critical' ? 'error' :
                  issue.severity === 'high' ? 'warning' : 'info',
            message: issue.title,
            details: issue.description,
          }))}
          title="Detected Issues"
          onDismiss={(alertId) => {
            setDetectedIssues(prev => prev.filter(issue => issue.id !== alertId));
          }}
        />
      )}

      {/* Diagnostic Suites Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
              <StatusIndicator
                status={suite.isRunning ? 'degraded' : 'healthy'}
                size="md"
                showLabel={false}
              />
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

            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Tests</p>
                <p className="text-lg font-medium text-gray-900 dark:text-white">
                  {suite.tests.length}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Passed</p>
                <p className="text-lg font-medium text-green-600">
                  {suite.tests.filter(t => t.status === 'passed').length}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Failed</p>
                <p className="text-lg font-medium text-red-600">
                  {suite.tests.filter(t => t.status === 'failed').length}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Automated Diagnostics
              </h1>
              <div className="ml-4 flex items-center">
                <div className={`h-3 w-3 rounded-full ${
                  connectionStore.state.isConnected ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">
                  {connectionStore.state.isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto">
            {categories.map((category) => (
              <button
                key={category.id}
                onClick={() => setActiveCategory(category.id)}
                className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  activeCategory === category.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
                title={category.description}
              >
                <span className="mr-2">{category.icon}</span>
                <span>{category.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderCategoryContent()}
      </main>

      {/* Diagnostic Report Modal */}
      {showReport && diagnosticReport && (
        <DiagnosticReport
          report={diagnosticReport}
          onClose={() => setShowReport(false)}
        />
      )}
    </div>
  );
});

export default DiagnosticsSystem;
