import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { useSystemStore, useUIStore } from '../../hooks/useStores'
import { Card, CardContent, CardHeader, CardTitle, Button, Modal } from '../ui'
import { StatusIndicator } from '../monitoring/StatusIndicator'

interface Service {
  name: string
  displayName: string
  description: string
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping'
  port?: number
  pid?: number
  uptime?: number
  memoryUsage?: number
  cpuUsage?: number
  autoRestart: boolean
  critical: boolean
}

const services: Service[] = [
  {
    name: 'api_server',
    displayName: 'API Server',
    description: 'FastAPI server handling REST endpoints',
    status: 'running',
    port: 8000,
    pid: 12345,
    uptime: 3600, // seconds
    memoryUsage: 128, // MB
    cpuUsage: 5.2, // %
    autoRestart: true,
    critical: true,
  },
  {
    name: 'websocket_server',
    displayName: 'WebSocket Server',
    description: 'Real-time communication server',
    status: 'running',
    port: 8001,
    pid: 12346,
    uptime: 3580,
    memoryUsage: 64,
    cpuUsage: 2.1,
    autoRestart: true,
    critical: true,
  },
  {
    name: 'health_monitor',
    displayName: 'Health Monitor',
    description: 'System health and performance monitoring',
    status: 'running',
    pid: 12347,
    uptime: 3590,
    memoryUsage: 32,
    cpuUsage: 1.5,
    autoRestart: true,
    critical: false,
  },
  {
    name: 'config_watcher',
    displayName: 'Configuration Watcher',
    description: 'Hot reload configuration changes',
    status: 'running',
    pid: 12348,
    uptime: 3600,
    memoryUsage: 16,
    cpuUsage: 0.3,
    autoRestart: true,
    critical: false,
  },
  {
    name: 'database_connection',
    displayName: 'Database Pool',
    description: 'PostgreSQL connection pool manager',
    status: 'running',
    pid: 12349,
    uptime: 3610,
    memoryUsage: 48,
    cpuUsage: 0.8,
    autoRestart: true,
    critical: true,
  },
  {
    name: 'background_tasks',
    displayName: 'Background Tasks',
    description: 'Async background job processor',
    status: 'error',
    pid: null,
    uptime: 0,
    memoryUsage: 0,
    cpuUsage: 0,
    autoRestart: false,
    critical: false,
  },
]

export const ServiceManagement = observer(() => {
  const systemStore = useSystemStore()
  const uiStore = useUIStore()
  const [actioningService, setActioningService] = useState<string | null>(null)
  const [confirmAction, setConfirmAction] = useState<{
    service: string
    action: 'start' | 'stop' | 'restart'
  } | null>(null)

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success'
      case 'stopped':
        return 'secondary'
      case 'error':
        return 'error'
      case 'starting':
      case 'stopping':
        return 'warning'
      default:
        return 'secondary'
    }
  }

  const formatUptime = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
    return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`
  }

  const handleServiceAction = async (serviceName: string, action: 'start' | 'stop' | 'restart') => {
    setActioningService(serviceName)

    try {
      // This will need to be implemented with real API calls
      let result = { success: false, error: 'API not implemented yet' }

      switch (action) {
        case 'start':
          // result = await systemStore.startService(serviceName)
          break
        case 'stop':
          // result = await systemStore.stopService(serviceName)
          break
        case 'restart':
          // result = await systemStore.restartService(serviceName)
          break
      }

      if (result.success) {
        uiStore.showSuccess(
          'Service Action Successful',
          `Service ${serviceName} ${action} completed successfully.`
        )
      } else {
        uiStore.showError(
          'Service Action Failed',
          result.error || `Failed to ${action} service ${serviceName}.`
        )
      }
    } catch (error) {
      uiStore.showError(
        'Service Action Error',
        error instanceof Error ? error.message : `Unexpected error during ${action} of ${serviceName}.`
      )
    } finally {
      setActioningService(null)
      setConfirmAction(null)
    }
  }

  const toggleAutoRestart = async (serviceName: string) => {
    try {
      // This will need to be implemented with real API calls
      // await systemStore.toggleServiceAutoRestart(serviceName)
      uiStore.showInfo(
        'Auto-restart Updated',
        `Auto-restart setting for ${serviceName} has been updated.`
      )
    } catch (error) {
      uiStore.showError(
        'Auto-restart Update Failed',
        error instanceof Error ? error.message : 'Failed to update auto-restart setting.'
      )
    }
  }

  const canPerformAction = (service: Service, action: 'start' | 'stop' | 'restart') => {
    if (actioningService && actioningService !== service.name) return false

    switch (action) {
      case 'start':
        return service.status === 'stopped' || service.status === 'error'
      case 'stop':
        return service.status === 'running'
      case 'restart':
        return service.status === 'running' || service.status === 'error'
      default:
        return false
    }
  }

  const runningServices = services.filter(s => s.status === 'running')
  const errorServices = services.filter(s => s.status === 'error')
  const totalMemory = runningServices.reduce((sum, s) => sum + (s.memoryUsage || 0), 0)
  const avgCpu = runningServices.length > 0
    ? runningServices.reduce((sum, s) => sum + (s.cpuUsage || 0), 0) / runningServices.length
    : 0

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Service Management</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Service Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-success-50 dark:bg-success-900/20 rounded-lg border border-success-200 dark:border-success-800">
                <div className="text-lg font-bold text-success-700 dark:text-success-300">
                  {runningServices.length}
                </div>
                <div className="text-xs text-success-600 dark:text-success-400">
                  Running
                </div>
              </div>
              <div className="text-center p-3 bg-error-50 dark:bg-error-900/20 rounded-lg border border-error-200 dark:border-error-800">
                <div className="text-lg font-bold text-error-700 dark:text-error-300">
                  {errorServices.length}
                </div>
                <div className="text-xs text-error-600 dark:text-error-400">
                  Errors
                </div>
              </div>
              <div className="text-center p-3 bg-info-50 dark:bg-info-900/20 rounded-lg border border-info-200 dark:border-info-800">
                <div className="text-lg font-bold text-info-700 dark:text-info-300">
                  {totalMemory}MB
                </div>
                <div className="text-xs text-info-600 dark:text-info-400">
                  Memory
                </div>
              </div>
              <div className="text-center p-3 bg-warning-50 dark:bg-warning-900/20 rounded-lg border border-warning-200 dark:border-warning-800">
                <div className="text-lg font-bold text-warning-700 dark:text-warning-300">
                  {avgCpu.toFixed(1)}%
                </div>
                <div className="text-xs text-warning-600 dark:text-warning-400">
                  Avg CPU
                </div>
              </div>
            </div>

            {/* Service List */}
            <div className="space-y-3">
              {services.map((service) => (
                <div
                  key={service.name}
                  className={`p-4 border rounded-lg bg-white dark:bg-secondary-900 ${
                    service.critical
                      ? 'border-warning-200 dark:border-warning-800'
                      : 'border-secondary-200 dark:border-secondary-700'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <StatusIndicator
                          status={getStatusColor(service.status) as any}
                          size="sm"
                        />
                        <div>
                          <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 flex items-center space-x-2">
                            <span>{service.displayName}</span>
                            {service.critical && (
                              <span className="px-2 py-0.5 text-xs font-medium bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200 rounded">
                                Critical
                              </span>
                            )}
                          </h4>
                          <p className="text-xs text-secondary-600 dark:text-secondary-400">
                            {service.description}
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-xs text-secondary-500">
                        <div>
                          <span className={`px-2 py-1 rounded-full font-medium capitalize ${
                            service.status === 'running'
                              ? 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
                              : service.status === 'error'
                              ? 'bg-error-100 text-error-800 dark:bg-error-900 dark:text-error-200'
                              : service.status === 'starting' || service.status === 'stopping'
                              ? 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
                              : 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900 dark:text-secondary-200'
                          }`}>
                            {actioningService === service.name ? 'Processing...' : service.status}
                          </span>
                        </div>
                        {service.port && (
                          <div>Port: {service.port}</div>
                        )}
                        {service.pid && (
                          <div>PID: {service.pid}</div>
                        )}
                        {service.uptime && service.uptime > 0 && (
                          <div>Uptime: {formatUptime(service.uptime)}</div>
                        )}
                        {service.memoryUsage && service.memoryUsage > 0 && (
                          <div>Memory: {service.memoryUsage}MB</div>
                        )}
                        {service.cpuUsage && service.cpuUsage > 0 && (
                          <div>CPU: {service.cpuUsage.toFixed(1)}%</div>
                        )}
                      </div>

                      <div className="flex items-center space-x-4 mt-2">
                        <label className="flex items-center space-x-2 text-xs">
                          <input
                            type="checkbox"
                            checked={service.autoRestart}
                            onChange={() => toggleAutoRestart(service.name)}
                            className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                          />
                          <span className="text-secondary-600 dark:text-secondary-400">
                            Auto-restart
                          </span>
                        </label>
                      </div>
                    </div>

                    <div className="flex space-x-2 ml-4">
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={!canPerformAction(service, 'start') || actioningService === service.name}
                        onClick={() => setConfirmAction({ service: service.name, action: 'start' })}
                      >
                        {actioningService === service.name ? (
                          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                            <path fill="currentColor" className="opacity-75" d="M4 12a8 8 0 018-8v8z" />
                          </svg>
                        ) : (
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h8m-9-4h10a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2v-8a2 2 0 012-2z" />
                          </svg>
                        )}
                        Start
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        disabled={!canPerformAction(service, 'stop') || actioningService === service.name}
                        onClick={() => setConfirmAction({ service: service.name, action: 'stop' })}
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                        </svg>
                        Stop
                      </Button>

                      <Button
                        variant="primary"
                        size="sm"
                        disabled={!canPerformAction(service, 'restart') || actioningService === service.name}
                        onClick={() => setConfirmAction({ service: service.name, action: 'restart' })}
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        Restart
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Confirmation Modal */}
      {confirmAction && (
        <Modal
          isOpen={true}
          onClose={() => setConfirmAction(null)}
          title={`Confirm Service ${confirmAction.action.charAt(0).toUpperCase() + confirmAction.action.slice(1)}`}
        >
          <div className="space-y-4">
            <p className="text-secondary-700 dark:text-secondary-300">
              Are you sure you want to {confirmAction.action} the{' '}
              <span className="font-medium">
                {services.find(s => s.name === confirmAction.service)?.displayName}
              </span>{' '}
              service?
            </p>

            {confirmAction.action === 'stop' && services.find(s => s.name === confirmAction.service)?.critical && (
              <div className="p-3 bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-lg">
                <p className="text-sm text-error-700 dark:text-error-300">
                  This is a critical service. Stopping it may significantly impact system functionality.
                </p>
              </div>
            )}

            <div className="flex justify-end space-x-3 pt-4">
              <Button variant="outline" onClick={() => setConfirmAction(null)}>
                Cancel
              </Button>
              <Button
                variant={confirmAction.action === 'stop' ? 'danger' : 'primary'}
                onClick={() =>
                  handleServiceAction(confirmAction.service, confirmAction.action)
                }
              >
                {confirmAction.action.charAt(0).toUpperCase() + confirmAction.action.slice(1)} Service
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </>
  )
})