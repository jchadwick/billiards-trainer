/**
 * Debug panel to test frontend-backend integration
 */

import React, { useState, useEffect } from 'react';
import { observer } from 'mobx-react-lite';
import { integrationTester, IntegrationTestResult } from '../../utils/integration-test';
import { useRootStore } from '../../stores/context';

interface IntegrationTestPanelProps {
  className?: string;
}

export const IntegrationTestPanel: React.FC<IntegrationTestPanelProps> = observer(({ className = '' }) => {
  const rootStore = useRootStore();
  const [testResults, setTestResults] = useState<IntegrationTestResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const runTests = async () => {
    setIsRunning(true);
    try {
      const results = await integrationTester.runAllTests();
      setTestResults(results);
      integrationTester.printResults();
    } catch (error) {
      console.error('Failed to run integration tests:', error);
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(runTests, 10000); // Run every 10 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusIcon = (success: boolean) => success ? '✅' : '❌';
  const getStatusColor = (success: boolean) => success ? 'text-green-600' : 'text-red-600';

  const summary = integrationTester.getTestSummary();

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">🧪 Integration Tests</h2>
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="mr-2"
            />
            Auto-refresh
          </label>
          <button
            onClick={runTests}
            disabled={isRunning}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isRunning
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isRunning ? 'Running...' : 'Run Tests'}
          </button>
        </div>
      </div>

      {testResults.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center space-x-4 mb-4">
            <div className="text-lg font-medium">
              📊 Results: {summary.passed}/{summary.total} passed
            </div>
            <div className={`text-lg font-medium ${summary.successRate === 100 ? 'text-green-600' : 'text-orange-600'}`}>
              ({summary.successRate.toFixed(1)}%)
            </div>
          </div>

          <div className="space-y-3">
            {testResults.map((result, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg border-l-4 ${
                  result.success ? 'border-green-400 bg-green-50' : 'border-red-400 bg-red-50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-lg">{getStatusIcon(result.success)}</span>
                    <span className="font-medium text-gray-900">{result.test}</span>
                  </div>
                  <span className={`text-sm font-medium ${getStatusColor(result.success)}`}>
                    {result.success ? 'PASS' : 'FAIL'}
                  </span>
                </div>
                <div className="mt-2 text-sm text-gray-700">{result.message}</div>
                {result.details && (
                  <details className="mt-2">
                    <summary className="text-xs text-gray-500 cursor-pointer">Show details</summary>
                    <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                      {JSON.stringify(result.details, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="border-t pt-6">
        <h3 className="text-lg font-medium mb-4">🔄 Store Status</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">System Store</div>
            <div className={`text-sm ${rootStore.system.isHealthy ? 'text-green-600' : 'text-red-600'}`}>
              {rootStore.system.status.isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>

          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">Config Store</div>
            <div className={`text-sm ${rootStore.config.isValid ? 'text-green-600' : 'text-red-600'}`}>
              {rootStore.config.hasUnsavedChanges ? 'Dirty' : 'Clean'}
            </div>
          </div>

          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">Connection Store</div>
            <div className={`text-sm ${rootStore.connection.state.isConnected ? 'text-green-600' : 'text-red-600'}`}>
              {rootStore.connection.connectionStatus}
            </div>
          </div>

          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">Auth Store</div>
            <div className={`text-sm ${rootStore.auth.isAuthenticated ? 'text-green-600' : 'text-orange-600'}`}>
              {rootStore.auth.isAuthenticated ? 'Authenticated' : 'Not authenticated'}
            </div>
          </div>

          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">Vision Store</div>
            <div className={`text-sm ${rootStore.vision.isConnected ? 'text-green-600' : 'text-gray-600'}`}>
              {rootStore.vision.isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>

          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">UI Store</div>
            <div className={`text-sm ${rootStore.ui.isAnyLoading ? 'text-orange-600' : 'text-green-600'}`}>
              {rootStore.ui.isAnyLoading ? 'Loading' : 'Ready'}
            </div>
          </div>
        </div>
      </div>

      <div className="border-t pt-6 mt-6">
        <h3 className="text-lg font-medium mb-4">🛠️ Quick Actions</h3>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => rootStore.system.refreshMetrics()}
            className="px-3 py-2 bg-blue-100 text-blue-800 rounded-lg text-sm hover:bg-blue-200 transition-colors"
          >
            Refresh Metrics
          </button>
          <button
            onClick={() => rootStore.config.loadFromBackend()}
            className="px-3 py-2 bg-green-100 text-green-800 rounded-lg text-sm hover:bg-green-200 transition-colors"
          >
            Reload Config
          </button>
          <button
            onClick={() => rootStore.connection.connect()}
            className="px-3 py-2 bg-purple-100 text-purple-800 rounded-lg text-sm hover:bg-purple-200 transition-colors"
          >
            Test Connection
          </button>
          <button
            onClick={() => rootStore.system.clearErrors()}
            className="px-3 py-2 bg-orange-100 text-orange-800 rounded-lg text-sm hover:bg-orange-200 transition-colors"
          >
            Clear Errors
          </button>
        </div>
      </div>
    </div>
  );
});

export default IntegrationTestPanel;
