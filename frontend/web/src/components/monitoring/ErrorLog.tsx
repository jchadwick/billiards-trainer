/**
 * Error log component for displaying system errors and warnings with filtering
 * Provides searchable, filterable log viewer with export capabilities
 */

import React, { useState, useEffect, useMemo } from 'react';
import { observer } from 'mobx-react-lite';
import type { LogLevel } from '../../types/api';

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: LogLevel;
  component: string;
  message: string;
  details?: Record<string, any>;
  stackTrace?: string;
  userId?: string;
  sessionId?: string;
}

export const ErrorLog: React.FC = observer(() => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<LogLevel | 'ALL'>('ALL');
  const [filterComponent, setFilterComponent] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);

  // Mock log data generation
  useEffect(() => {
    generateMockLogs();
    if (autoRefresh) {
      const interval = setInterval(generateMockLogs, 10000); // Add new logs every 10 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const generateMockLogs = () => {
    const components = ['Vision', 'API', 'Core', 'Projector', 'Database', 'WebSocket'];
    const levels: LogLevel[] = ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'CRITICAL'];

    const sampleMessages = {
      ERROR: [
        'Failed to connect to camera device',
        'Database connection timeout',
        'Projector calibration failed',
        'WebSocket connection lost',
        'Invalid configuration parameter',
      ],
      WARNING: [
        'High memory usage detected',
        'Frame processing time exceeded threshold',
        'Calibration accuracy below optimal',
        'Disk space running low',
        'Network latency increased',
      ],
      INFO: [
        'System startup completed',
        'User logged in successfully',
        'Configuration reloaded',
        'Calibration process started',
        'Game session ended',
      ],
      DEBUG: [
        'Processing frame batch',
        'WebSocket message received',
        'Cache invalidated',
        'Background task completed',
        'Performance metrics updated',
      ],
      CRITICAL: [
        'System shutdown initiated due to critical error',
        'Memory corruption detected',
        'Hardware failure in vision system',
        'Security breach attempt blocked',
      ],
    };

    const generateRandomLog = (): LogEntry => {
      const level = levels[Math.floor(Math.random() * levels.length)];
      const component = components[Math.floor(Math.random() * components.length)];
      const messages = sampleMessages[level];
      const message = messages[Math.floor(Math.random() * messages.length)];

      return {
        id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date(Date.now() - Math.random() * 86400000), // Random time in last 24 hours
        level,
        component,
        message,
        details: level === 'ERROR' || level === 'CRITICAL' ? {
          errorCode: `E${Math.floor(Math.random() * 9999).toString().padStart(4, '0')}`,
          affectedSystems: [component],
          retryCount: Math.floor(Math.random() * 5),
        } : undefined,
        stackTrace: level === 'ERROR' || level === 'CRITICAL' ?
          `Error at line ${Math.floor(Math.random() * 100)}\n  at ${component}Module.process()\n  at EventLoop.tick()` :
          undefined,
        sessionId: `session-${Math.random().toString(36).substr(2, 8)}`,
      };
    };

    setLogs(prevLogs => {
      // Add 1-3 new logs
      const newLogs = Array.from({ length: Math.floor(Math.random() * 3) + 1 }, generateRandomLog);
      const allLogs = [...prevLogs, ...newLogs]
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        .slice(0, 1000); // Keep only last 1000 logs

      return allLogs;
    });

    setLoading(false);
  };

  // Apply filters
  useEffect(() => {
    let filtered = logs;

    // Filter by level
    if (filterLevel !== 'ALL') {
      filtered = filtered.filter(log => log.level === filterLevel);
    }

    // Filter by component
    if (filterComponent !== 'ALL') {
      filtered = filtered.filter(log => log.component === filterComponent);
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(log =>
        log.message.toLowerCase().includes(term) ||
        log.component.toLowerCase().includes(term) ||
        log.id.toLowerCase().includes(term)
      );
    }

    setFilteredLogs(filtered);
  }, [logs, filterLevel, filterComponent, searchTerm]);

  const components = useMemo(() => {
    const componentSet = new Set(logs.map(log => log.component));
    return Array.from(componentSet).sort();
  }, [logs]);

  const levelCounts = useMemo(() => {
    const counts: Record<LogLevel, number> = {
      CRITICAL: 0,
      ERROR: 0,
      WARNING: 0,
      INFO: 0,
      DEBUG: 0,
    };

    logs.forEach(log => {
      counts[log.level]++;
    });

    return counts;
  }, [logs]);

  const getLevelColor = (level: LogLevel): string => {
    switch (level) {
      case 'CRITICAL': return 'text-red-900 bg-red-100 border-red-200';
      case 'ERROR': return 'text-red-700 bg-red-50 border-red-200';
      case 'WARNING': return 'text-yellow-700 bg-yellow-50 border-yellow-200';
      case 'INFO': return 'text-blue-700 bg-blue-50 border-blue-200';
      case 'DEBUG': return 'text-gray-700 bg-gray-50 border-gray-200';
    }
  };

  const getLevelIcon = (level: LogLevel): string => {
    switch (level) {
      case 'CRITICAL': return '🚨';
      case 'ERROR': return '❌';
      case 'WARNING': return '⚠️';
      case 'INFO': return 'ℹ️';
      case 'DEBUG': return '🔍';
    }
  };

  const exportLogs = () => {
    const csvContent = filteredLogs.map(log =>
      `"${log.timestamp.toISOString()}","${log.level}","${log.component}","${log.message.replace(/"/g, '""')}"`
    ).join('\n');

    const csvHeader = 'Timestamp,Level,Component,Message\n';
    const csvData = csvHeader + csvContent;

    const blob = new Blob([csvData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `system-logs-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="flex flex-wrap items-center space-x-4">
            {/* Search */}
            <div className="flex-1 min-w-64">
              <input
                type="text"
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </div>

            {/* Level filter */}
            <select
              value={filterLevel}
              onChange={(e) => setFilterLevel(e.target.value as LogLevel | 'ALL')}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="ALL">All Levels</option>
              <option value="CRITICAL">Critical</option>
              <option value="ERROR">Error</option>
              <option value="WARNING">Warning</option>
              <option value="INFO">Info</option>
              <option value="DEBUG">Debug</option>
            </select>

            {/* Component filter */}
            <select
              value={filterComponent}
              onChange={(e) => setFilterComponent(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="ALL">All Components</option>
              {components.map(component => (
                <option key={component} value={component}>{component}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-4">
            {/* Auto-refresh toggle */}
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="sr-only"
              />
              <div className={`relative w-10 h-6 transition-colors duration-200 ease-in-out rounded-full ${
                autoRefresh ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
              }`}>
                <div className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 ease-in-out ${
                  autoRefresh ? 'transform translate-x-4' : ''
                }`} />
              </div>
              <span className="ml-2 text-sm text-gray-600 dark:text-gray-300">
                Auto-refresh
              </span>
            </label>

            {/* Export button */}
            <button
              onClick={exportLogs}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
            >
              Export CSV
            </button>
          </div>
        </div>
      </div>

      {/* Level Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {Object.entries(levelCounts).map(([level, count]) => (
          <div
            key={level}
            className={`p-4 rounded-lg border ${getLevelColor(level as LogLevel)}`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{level}</p>
                <p className="text-2xl font-bold">{count}</p>
              </div>
              <span className="text-xl">{getLevelIcon(level as LogLevel)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Log List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            System Logs ({filteredLogs.length})
          </h3>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              <p className="text-gray-500 dark:text-gray-400">Loading logs...</p>
            </div>
          ) : filteredLogs.length === 0 ? (
            <div className="p-8 text-center">
              <p className="text-gray-500 dark:text-gray-400">No logs found matching your criteria.</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {filteredLogs.map((log) => (
                <div
                  key={log.id}
                  className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                  onClick={() => setSelectedLog(log)}
                >
                  <div className="flex items-start space-x-3">
                    <span className="text-lg">{getLevelIcon(log.level)}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getLevelColor(log.level)}`}>
                            {log.level}
                          </span>
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            {log.component}
                          </span>
                        </div>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {log.timestamp.toLocaleString()}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                        {log.message}
                      </p>
                      {log.details && (
                        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                          Session: {log.sessionId} • Error Code: {log.details.errorCode}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Log Detail Modal */}
      {selectedLog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  Log Details
                </h3>
                <button
                  onClick={() => setSelectedLog(null)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Level</label>
                    <div className={`mt-1 px-2 py-1 text-sm font-medium rounded border ${getLevelColor(selectedLog.level)} inline-block`}>
                      {selectedLog.level}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Component</label>
                    <p className="mt-1 text-sm text-gray-900 dark:text-white">{selectedLog.component}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Timestamp</label>
                    <p className="mt-1 text-sm text-gray-900 dark:text-white">{selectedLog.timestamp.toLocaleString()}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Session ID</label>
                    <p className="mt-1 text-sm text-gray-900 dark:text-white font-mono">{selectedLog.sessionId}</p>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Message</label>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white">{selectedLog.message}</p>
                </div>

                {selectedLog.details && (
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Details</label>
                    <pre className="mt-1 text-sm text-gray-900 dark:text-white bg-gray-100 dark:bg-gray-700 p-3 rounded overflow-x-auto">
                      {JSON.stringify(selectedLog.details, null, 2)}
                    </pre>
                  </div>
                )}

                {selectedLog.stackTrace && (
                  <div>
                    <label className="text-sm font-medium text-gray-500 dark:text-gray-400">Stack Trace</label>
                    <pre className="mt-1 text-sm text-gray-900 dark:text-white bg-gray-100 dark:bg-gray-700 p-3 rounded overflow-x-auto">
                      {selectedLog.stackTrace}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default ErrorLog;
