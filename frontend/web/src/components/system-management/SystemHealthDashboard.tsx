import React from 'react'
import { observer } from 'mobx-react-lite'
import { useSystemStore } from '../../hooks/useStores'
import { Card, CardContent, CardHeader, CardTitle, Button } from '../ui'
import { StatusIndicator } from '../monitoring/StatusIndicator'

export const SystemHealthDashboard = observer(() => {
  const systemStore = useSystemStore()

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success'
      case 'degraded':
        return 'warning'
      case 'unhealthy':
        return 'error'
      default:
        return 'secondary'
    }
  }

  const systemHealth = systemStore.status.isConnected ? 'healthy' : 'unhealthy'
  const componentStatuses = [
    { name: 'API', status: systemStore.status.isConnected ? 'healthy' : 'unhealthy' },
    { name: 'WebSocket', status: systemStore.status.websocketStatus === 'connected' ? 'healthy' : 'unhealthy' },
    { name: 'Core', status: 'healthy' }, // Will be connected to real data
    { name: 'Vision', status: 'healthy' },
    { name: 'Config', status: 'healthy' },
    { name: 'Projector', status: 'degraded' }, // Example status
  ]

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-current" style={{
            color: systemHealth === 'healthy' ? '#10b981' : systemHealth === 'degraded' ? '#f59e0b' : '#ef4444'
          }} />
          <span>System Health Dashboard</span>
        </CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={() => systemStore.refreshHealth()}
        >
          <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Overall System Status */}
          <div className="flex items-center justify-center p-6 bg-secondary-50 dark:bg-secondary-800 rounded-lg">
            <div className="text-center">
              <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
                systemHealth === 'healthy'
                  ? 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
                  : systemHealth === 'degraded'
                  ? 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
                  : 'bg-error-100 text-error-800 dark:bg-error-900 dark:text-error-200'
              }`}>
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  systemHealth === 'healthy' ? 'bg-success-500' :
                  systemHealth === 'degraded' ? 'bg-warning-500' : 'bg-error-500'
                }`} />
                System {systemHealth === 'healthy' ? 'Healthy' : systemHealth === 'degraded' ? 'Degraded' : 'Unhealthy'}
              </div>
              <div className="text-sm text-secondary-600 dark:text-secondary-400 mt-2">
                {systemHealth === 'healthy'
                  ? 'All systems operating normally'
                  : systemHealth === 'degraded'
                  ? 'Some services experiencing issues'
                  : 'Critical issues detected'
                }
              </div>
            </div>
          </div>

          {/* Component Status Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {componentStatuses.map((component) => (
              <div key={component.name} className="text-center p-3 bg-white dark:bg-secondary-900 rounded-lg border border-secondary-200 dark:border-secondary-700">
                <StatusIndicator
                  status={component.status as any}
                  size="lg"
                  className="mx-auto mb-2"
                />
                <div className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                  {component.name}
                </div>
                <div className={`text-xs mt-1 capitalize ${
                  component.status === 'healthy' ? 'text-success-600' :
                  component.status === 'degraded' ? 'text-warning-600' : 'text-error-600'
                }`}>
                  {component.status}
                </div>
              </div>
            ))}
          </div>

          {/* System Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-secondary-900 p-4 rounded-lg border border-secondary-200 dark:border-secondary-700">
              <div className="text-sm font-medium text-secondary-600 dark:text-secondary-400">
                CPU Usage
              </div>
              <div className="text-2xl font-bold text-secondary-900 dark:text-secondary-100 mt-1">
                {systemStore.lastMetrics?.cpu_usage?.toFixed(1) || '0.0'}%
              </div>
              <div className="w-full bg-secondary-200 dark:bg-secondary-700 rounded-full h-2 mt-2">
                <div
                  className="bg-primary-600 h-2 rounded-full"
                  style={{ width: `${systemStore.lastMetrics?.cpu_usage || 0}%` }}
                />
              </div>
            </div>

            <div className="bg-white dark:bg-secondary-900 p-4 rounded-lg border border-secondary-200 dark:border-secondary-700">
              <div className="text-sm font-medium text-secondary-600 dark:text-secondary-400">
                Memory Usage
              </div>
              <div className="text-2xl font-bold text-secondary-900 dark:text-secondary-100 mt-1">
                {systemStore.lastMetrics?.memory_usage?.toFixed(1) || '0.0'}%
              </div>
              <div className="w-full bg-secondary-200 dark:bg-secondary-700 rounded-full h-2 mt-2">
                <div
                  className="bg-success-600 h-2 rounded-full"
                  style={{ width: `${systemStore.lastMetrics?.memory_usage || 0}%` }}
                />
              </div>
            </div>

            <div className="bg-white dark:bg-secondary-900 p-4 rounded-lg border border-secondary-200 dark:border-secondary-700">
              <div className="text-sm font-medium text-secondary-600 dark:text-secondary-400">
                Disk Usage
              </div>
              <div className="text-2xl font-bold text-secondary-900 dark:text-secondary-100 mt-1">
                {systemStore.lastMetrics?.disk_usage?.toFixed(1) || '0.0'}%
              </div>
              <div className="w-full bg-secondary-200 dark:bg-secondary-700 rounded-full h-2 mt-2">
                <div
                  className="bg-warning-600 h-2 rounded-full"
                  style={{ width: `${systemStore.lastMetrics?.disk_usage || 0}%` }}
                />
              </div>
            </div>

            <div className="bg-white dark:bg-secondary-900 p-4 rounded-lg border border-secondary-200 dark:border-secondary-700">
              <div className="text-sm font-medium text-secondary-600 dark:text-secondary-400">
                Active Connections
              </div>
              <div className="text-2xl font-bold text-secondary-900 dark:text-secondary-100 mt-1">
                {systemStore.lastMetrics?.websocket_connections || 0}
              </div>
              <div className="text-xs text-secondary-500 mt-1">
                WebSocket clients
              </div>
            </div>
          </div>

          {/* Recent Alerts */}
          {systemStore.recentErrors.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-3">
                Recent Alerts
              </h4>
              <div className="space-y-2">
                {systemStore.recentErrors.slice(0, 3).map((error, index) => (
                  <div key={index} className={`p-3 rounded-lg border-l-4 ${
                    error.level === 'critical'
                      ? 'bg-error-50 border-error-500 dark:bg-error-900/20'
                      : error.level === 'warning'
                      ? 'bg-warning-50 border-warning-500 dark:bg-warning-900/20'
                      : 'bg-info-50 border-info-500 dark:bg-info-900/20'
                  }`}>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                          {error.source}: {error.message}
                        </div>
                        <div className="text-xs text-secondary-500 mt-1">
                          {error.timestamp.toLocaleString()}
                        </div>
                      </div>
                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                        error.level === 'critical'
                          ? 'bg-error-100 text-error-800 dark:bg-error-900 dark:text-error-200'
                          : error.level === 'warning'
                          ? 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
                          : 'bg-info-100 text-info-800 dark:bg-info-900 dark:text-info-200'
                      }`}>
                        {error.level}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
})
