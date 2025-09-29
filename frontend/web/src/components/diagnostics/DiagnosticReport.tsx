/**
 * Diagnostic Report Component
 * Provides comprehensive diagnostic report generation, viewing, and export functionality
 *
 * Features:
 * - Comprehensive report generation
 * - Multiple export formats (JSON, CSV, PDF)
 * - Detailed test results and metrics
 * - Issue summaries and recommendations
 * - Historical data comparison
 */

import React, { useState, useCallback } from 'react';
import { observer } from 'mobx-react-lite';
import { StatCard } from '../monitoring/StatCard';
import { ProgressBar } from '../monitoring/ProgressBar';
import { StatusIndicator } from '../monitoring/StatusIndicator';

interface DiagnosticReportData {
  timestamp: Date;
  summary: {
    total: number;
    passed: number;
    failed: number;
    warnings: number;
    successRate: number;
  };
  issues: Array<{
    id: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    category: string;
    title: string;
    description: string;
    impact: string;
    recommendations: string[];
    autoFixAvailable: boolean;
    detectedAt: Date;
  }>;
  suites: Array<{
    id: string;
    name: string;
    category: string;
    tests: Array<{
      id: string;
      name: string;
      status: string;
      duration?: number;
      result?: any;
    }>;
  }>;
  systemInfo: {
    version: string;
    uptime: string;
    health: string;
  };
  performance?: {
    averageFrameRate: number;
    averageLatency: number;
    memoryUsage: number;
    cpuUsage: number;
  };
  network?: {
    connectivity: string;
    averageLatency: number;
    throughput: number;
    endpointsOnline: number;
    totalEndpoints: number;
  };
  recommendations: string[];
}

interface DiagnosticReportProps {
  report: DiagnosticReportData;
  onClose: () => void;
}

export const DiagnosticReport: React.FC<DiagnosticReportProps> = observer(({ report, onClose }) => {
  const [selectedTab, setSelectedTab] = useState<'overview' | 'details' | 'issues' | 'recommendations'>('overview');
  const [isExporting, setIsExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<'json' | 'csv' | 'pdf'>('json');

  const exportReport = useCallback(async (format: 'json' | 'csv' | 'pdf') => {
    setIsExporting(true);

    try {
      switch (format) {
        case 'json':
          await exportAsJSON();
          break;
        case 'csv':
          await exportAsCSV();
          break;
        case 'pdf':
          await exportAsPDF();
          break;
      }
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  }, [report]);

  const exportAsJSON = async () => {
    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `diagnostic-report-${report.timestamp.toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportAsCSV = async () => {
    const csvData = generateCSVData();
    const dataBlob = new Blob([csvData], { type: 'text/csv' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `diagnostic-report-${report.timestamp.toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportAsPDF = async () => {
    // For PDF export, we would typically use a library like jsPDF or html2pdf
    // For this implementation, we'll simulate the process
    const pdfContent = generatePDFContent();
    console.log('PDF export would generate:', pdfContent);

    // Simulate PDF generation delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    alert('PDF export functionality would be implemented with a PDF library like jsPDF');
  };

  const generateCSVData = () => {
    const headers = ['Test Name', 'Category', 'Status', 'Duration (ms)', 'Score', 'Issues'];
    const rows = [];

    // Add header
    rows.push(headers.join(','));

    // Add test data
    report.suites.forEach(suite => {
      suite.tests.forEach(test => {
        const row = [
          `"${test.name}"`,
          `"${suite.category}"`,
          test.status,
          test.duration || 0,
          test.result?.score || 0,
          `"${test.result?.issues?.join('; ') || ''}"`,
        ];
        rows.push(row.join(','));
      });
    });

    return rows.join('\n');
  };

  const generatePDFContent = () => {
    return {
      title: 'Billiards Trainer - Diagnostic Report',
      timestamp: report.timestamp.toLocaleString(),
      summary: report.summary,
      sections: [
        {
          title: 'Executive Summary',
          content: `System diagnostic completed with ${report.summary.successRate.toFixed(1)}% success rate. ${report.summary.failed} tests failed out of ${report.summary.total} total tests.`,
        },
        {
          title: 'Critical Issues',
          content: report.issues.filter(issue => issue.severity === 'critical').map(issue => issue.title).join(', ') || 'None',
        },
        {
          title: 'Recommendations',
          content: report.recommendations.join('\n• '),
        },
      ],
    };
  };

  const getSeverityColor = (severity: string): 'green' | 'yellow' | 'red' | 'purple' => {
    switch (severity) {
      case 'critical': return 'purple';
      case 'high': return 'red';
      case 'medium': return 'yellow';
      case 'low': return 'green';
      default: return 'yellow';
    }
  };

  const getCategoryIcon = (category: string): string => {
    switch (category.toLowerCase()) {
      case 'hardware': return '🔧';
      case 'network': return '🌐';
      case 'performance': return '⚡';
      case 'troubleshooting': return '🔍';
      case 'system-validation': return '✅';
      default: return '📊';
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Success Rate"
          value={report.summary.successRate.toFixed(1)}
          unit="%"
          icon="🎯"
          color={report.summary.successRate >= 90 ? 'green' :
                 report.summary.successRate >= 70 ? 'yellow' : 'red'}
        />
        <StatCard
          title="Tests Passed"
          value={report.summary.passed}
          unit={` / ${report.summary.total}`}
          icon="✅"
          color="green"
        />
        <StatCard
          title="Tests Failed"
          value={report.summary.failed}
          icon="❌"
          color="red"
        />
        <StatCard
          title="Warnings"
          value={report.summary.warnings}
          icon="⚠️"
          color="yellow"
        />
      </div>

      {/* System Information */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          System Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <span className="text-sm text-gray-500 dark:text-gray-400">Version:</span>
            <span className="ml-2 text-gray-900 dark:text-white">{report.systemInfo.version}</span>
          </div>
          <div>
            <span className="text-sm text-gray-500 dark:text-gray-400">Uptime:</span>
            <span className="ml-2 text-gray-900 dark:text-white">{report.systemInfo.uptime}</span>
          </div>
          <div>
            <span className="text-sm text-gray-500 dark:text-gray-400">Health:</span>
            <span className="ml-2">
              <StatusIndicator
                status={report.systemInfo.health === 'healthy' ? 'healthy' : 'unhealthy'}
                size="sm"
                showLabel={true}
              />
            </span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      {report.performance && (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Performance Metrics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Frame Rate:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.performance.averageFrameRate.toFixed(1)} fps</span>
            </div>
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Latency:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.performance.averageLatency.toFixed(0)} ms</span>
            </div>
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Memory:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.performance.memoryUsage.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">CPU:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.performance.cpuUsage.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Network Metrics */}
      {report.network && (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Network Metrics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Connectivity:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.network.connectivity}</span>
            </div>
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Latency:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.network.averageLatency.toFixed(0)} ms</span>
            </div>
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Throughput:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.network.throughput.toFixed(1)} Mbps</span>
            </div>
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Endpoints Online:</span>
              <span className="ml-2 text-gray-900 dark:text-white">{report.network.endpointsOnline}/{report.network.totalEndpoints}</span>
            </div>
          </div>
        </div>
      )}

      {/* Test Suite Summary */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Test Suite Summary
        </h3>
        <div className="space-y-4">
          {report.suites.map((suite) => {
            const passedTests = suite.tests.filter(t => t.status === 'passed').length;
            const totalTests = suite.tests.length;
            const successRate = totalTests > 0 ? (passedTests / totalTests) * 100 : 0;

            return (
              <div key={suite.id} className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 rounded-lg">
                <div className="flex items-center space-x-4">
                  <span className="text-2xl">{getCategoryIcon(suite.category)}</span>
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                      {suite.name}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      {passedTests}/{totalTests} tests passed
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="w-32">
                    <ProgressBar
                      value={successRate}
                      size="sm"
                      color={successRate >= 90 ? 'green' : successRate >= 70 ? 'yellow' : 'red'}
                      showPercentage={false}
                    />
                  </div>
                  <span className={`text-sm font-medium ${
                    successRate >= 90 ? 'text-green-600' :
                    successRate >= 70 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {successRate.toFixed(0)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderDetails = () => (
    <div className="space-y-6">
      {report.suites.map((suite) => (
        <div key={suite.id} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
            <span>{getCategoryIcon(suite.category)}</span>
            <span>{suite.name}</span>
          </h3>

          <div className="space-y-3">
            {suite.tests.map((test) => (
              <div key={test.id} className="p-4 bg-white dark:bg-gray-800 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                    {test.name}
                  </h4>
                  <div className="flex items-center space-x-2">
                    {test.duration && (
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {test.duration}ms
                      </span>
                    )}
                    <StatCard
                      title=""
                      value={test.status}
                      size="sm"
                      color={test.status === 'passed' ? 'green' :
                             test.status === 'warning' ? 'yellow' : 'red'}
                      className="min-w-16"
                    />
                  </div>
                </div>

                {test.result && (
                  <div className="text-sm space-y-1">
                    {test.result.score !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-gray-500 dark:text-gray-400">Score:</span>
                        <span className={`font-medium ${
                          test.result.score >= 80 ? 'text-green-600' :
                          test.result.score >= 60 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {test.result.score}/100
                        </span>
                      </div>
                    )}

                    {test.result.message && (
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Result:</span>
                        <span className="ml-2 text-gray-900 dark:text-white">{test.result.message}</span>
                      </div>
                    )}

                    {test.result.details && Object.keys(test.result.details).length > 0 && (
                      <div className="mt-2">
                        <span className="text-gray-500 dark:text-gray-400">Details:</span>
                        <div className="ml-2 grid grid-cols-2 gap-2 text-xs">
                          {Object.entries(test.result.details).map(([key, value]) => (
                            <div key={key}>
                              <span className="text-gray-500 dark:text-gray-400">{key}:</span>
                              <span className="ml-1 text-gray-900 dark:text-white">
                                {typeof value === 'number' ? value.toFixed(1) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {test.result.issues && test.result.issues.length > 0 && (
                      <div className="mt-2">
                        <span className="text-red-600 dark:text-red-400 text-xs">
                          Issues: {test.result.issues.join(', ')}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );

  const renderIssues = () => (
    <div className="space-y-6">
      {report.issues.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🎉</div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No Issues Detected
          </h3>
          <p className="text-gray-600 dark:text-gray-300">
            All diagnostic tests completed successfully with no issues found.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {report.issues.map((issue) => (
            <div key={issue.id} className={`p-4 rounded-lg border-l-4 ${
              issue.severity === 'critical' ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20' :
              issue.severity === 'high' ? 'border-red-500 bg-red-50 dark:bg-red-900/20' :
              issue.severity === 'medium' ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' :
              'border-green-500 bg-green-50 dark:bg-green-900/20'
            }`}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <StatCard
                      title=""
                      value={issue.severity}
                      size="sm"
                      color={getSeverityColor(issue.severity)}
                      className="min-w-16"
                    />
                    <span className="text-sm text-gray-500 dark:text-gray-400">{issue.category}</span>
                    {issue.autoFixAvailable && (
                      <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded">
                        Auto-fix available
                      </span>
                    )}
                  </div>

                  <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    {issue.title}
                  </h4>

                  <p className="text-gray-600 dark:text-gray-300 mb-2">
                    {issue.description}
                  </p>

                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
                    <strong>Impact:</strong> {issue.impact}
                  </p>

                  {issue.recommendations.length > 0 && (
                    <div>
                      <h5 className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                        Recommendations:
                      </h5>
                      <ul className="text-sm text-gray-600 dark:text-gray-300 list-disc list-inside">
                        {issue.recommendations.map((rec, index) => (
                          <li key={index}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <div className="text-xs text-gray-500 dark:text-gray-400 ml-4">
                  Detected: {issue.detectedAt.toLocaleString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderRecommendations = () => (
    <div className="space-y-6">
      {report.recommendations.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">✨</div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No Recommendations
          </h3>
          <p className="text-gray-600 dark:text-gray-300">
            Your system is running optimally with no specific recommendations at this time.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
            <h3 className="text-lg font-medium text-blue-900 dark:text-blue-100 mb-4">
              System Optimization Recommendations
            </h3>
            <div className="space-y-3">
              {report.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">
                    {index + 1}
                  </div>
                  <p className="text-blue-800 dark:text-blue-200">
                    {recommendation}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Priority Actions */}
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
            <h3 className="text-lg font-medium text-yellow-900 dark:text-yellow-100 mb-4">
              Priority Actions
            </h3>
            <div className="space-y-2">
              {report.issues.filter(issue => issue.severity === 'critical' || issue.severity === 'high').map((issue) => (
                <div key={issue.id} className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    issue.severity === 'critical' ? 'bg-purple-500' : 'bg-red-500'
                  }`} />
                  <span className="text-yellow-800 dark:text-yellow-200">
                    Resolve {issue.title.toLowerCase()}
                  </span>
                </div>
              ))}
              {report.issues.filter(issue => issue.severity === 'critical' || issue.severity === 'high').length === 0 && (
                <p className="text-yellow-800 dark:text-yellow-200">
                  No critical or high priority issues detected.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'details', label: 'Test Details', icon: '🔍' },
    { id: 'issues', label: 'Issues', icon: '⚠️', count: report.issues.length },
    { id: 'recommendations', label: 'Recommendations', icon: '💡', count: report.recommendations.length },
  ] as const;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              Diagnostic Report
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Generated on {report.timestamp.toLocaleString()}
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value as any)}
              className="text-sm border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="json">JSON</option>
              <option value="csv">CSV</option>
              <option value="pdf">PDF</option>
            </select>
            <button
              onClick={() => exportReport(exportFormat)}
              disabled={isExporting}
              className={`px-4 py-2 text-sm rounded-md transition-colors ${
                isExporting
                  ? 'bg-gray-400 cursor-not-allowed text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isExporting ? 'Exporting...' : 'Export'}
            </button>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedTab(tab.id)}
                className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  selectedTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                <span>{tab.label}</span>
                {tab.count !== undefined && tab.count > 0 && (
                  <span className="ml-2 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 text-xs px-2 py-1 rounded-full">
                    {tab.count}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {selectedTab === 'overview' && renderOverview()}
          {selectedTab === 'details' && renderDetails()}
          {selectedTab === 'issues' && renderIssues()}
          {selectedTab === 'recommendations' && renderRecommendations()}
        </div>
      </div>
    </div>
  );
});

export default DiagnosticReport;