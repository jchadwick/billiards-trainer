/**
 * Main monitoring dashboard layout component
 * Provides the overall structure and navigation for system monitoring
 */

import React, { useState, useEffect } from 'react';
import { observer } from 'mobx-react-lite';
import { useStores } from '../../stores/context';
import { SystemOverview } from './SystemOverview';
import { PerformanceMetrics } from './PerformanceMetrics';
import { ModuleStatus } from './ModuleStatus';
import { ErrorLog } from './ErrorLog';
import { EventTimeline } from './EventTimeline';
import { HealthCheck } from './HealthCheck';
import { NetworkDiagnostics } from './NetworkDiagnostics';
import { ErrorBoundary } from './ErrorBoundary';

export type DashboardTab =
  | 'overview'
  | 'performance'
  | 'modules'
  | 'errors'
  | 'events'
  | 'diagnostics'
  | 'network';

interface DashboardLayoutProps {
  initialTab?: DashboardTab;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = observer(({
  initialTab = 'overview'
}) => {
  const { connectionStore, systemStore } = useStores();
  const [activeTab, setActiveTab] = useState<DashboardTab>(initialTab);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds

  // Auto-refresh effect
  useEffect(() => {
    if (!isAutoRefresh) return;

    const interval = setInterval(() => {
      // Refresh data based on current tab
      refreshCurrentTabData();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [isAutoRefresh, refreshInterval, activeTab]);

  const refreshCurrentTabData = async () => {
    try {
      switch (activeTab) {
        case 'overview':
        case 'performance':
          // Refresh system metrics
          await systemStore?.refreshMetrics?.();
          break;
        case 'modules':
          // Refresh module status
          await systemStore?.refreshModuleStatus?.();
          break;
        case 'network':
          // Refresh connection status
          await connectionStore?.refreshStatus?.();
          break;
        // Add other refresh logic as needed
      }
    } catch (error) {
      console.error('Failed to refresh data:', error);
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'performance', label: 'Performance', icon: '⚡' },
    { id: 'modules', label: 'Modules', icon: '🔧' },
    { id: 'errors', label: 'Errors', icon: '⚠️' },
    { id: 'events', label: 'Events', icon: '📅' },
    { id: 'diagnostics', label: 'Diagnostics', icon: '🔍' },
    { id: 'network', label: 'Network', icon: '🌐' },
  ] as const;

  const renderTabContent = () => {
    const content = (() => {
      switch (activeTab) {
        case 'overview':
          return <SystemOverview />;
        case 'performance':
          return <PerformanceMetrics />;
        case 'modules':
          return <ModuleStatus />;
        case 'errors':
          return <ErrorLog />;
        case 'events':
          return <EventTimeline />;
        case 'diagnostics':
          return <HealthCheck />;
        case 'network':
          return <NetworkDiagnostics />;
        default:
          return <SystemOverview />;
      }
    })();

    return (
      <ErrorBoundary
        onError={(error, errorInfo) => {
          console.error(`Error in ${activeTab} tab:`, error, errorInfo);
          // Could send to error tracking service here
        }}
        fallback={
          <div className="flex items-center justify-center min-h-96">
            <div className="text-center">
              <div className="text-6xl mb-4">⚠️</div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Failed to load {activeTab} dashboard
              </h3>
              <p className="text-gray-500 dark:text-gray-400 mb-4">
                There was an error loading this section of the monitoring dashboard.
              </p>
              <button
                onClick={() => window.location.reload()}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors"
              >
                Reload Dashboard
              </button>
            </div>
          </div>
        }
      >
        {content}
      </ErrorBoundary>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                System Monitoring
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

            {/* Controls */}
            <div className="flex items-center space-x-4">
              {/* Auto-refresh toggle */}
              <div className="flex items-center">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={isAutoRefresh}
                    onChange={(e) => setIsAutoRefresh(e.target.checked)}
                    className="sr-only"
                  />
                  <div className={`relative w-10 h-6 transition-colors duration-200 ease-in-out rounded-full ${
                    isAutoRefresh ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
                  }`}>
                    <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 ease-in-out ${
                      isAutoRefresh ? 'transform translate-x-4' : ''
                    }`} />
                  </div>
                  <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">
                    Auto-refresh
                  </span>
                </label>
              </div>

              {/* Refresh interval selector */}
              {isAutoRefresh && (
                <select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                  className="text-sm border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value={1000}>1s</option>
                  <option value={2000}>2s</option>
                  <option value={5000}>5s</option>
                  <option value={10000}>10s</option>
                  <option value={30000}>30s</option>
                </select>
              )}

              {/* Manual refresh button */}
              <button
                onClick={refreshCurrentTabData}
                className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
                title="Manual refresh"
              >
                🔄
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Mobile tab selector */}
          <div className="block sm:hidden">
            <select
              value={activeTab}
              onChange={(e) => setActiveTab(e.target.value as DashboardTab)}
              className="block w-full py-2 px-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              {tabs.map((tab) => (
                <option key={tab.id} value={tab.id}>
                  {tab.icon} {tab.label}
                </option>
              ))}
            </select>
          </div>

          {/* Desktop tabs */}
          <div className="hidden sm:flex space-x-8 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as DashboardTab)}
                className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                <span className="hidden md:inline">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderTabContent()}
      </main>
    </div>
  );
});

export default DashboardLayout;
