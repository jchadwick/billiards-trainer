import React from 'react'
import { observer } from 'mobx-react-lite'
import { useSystemStore } from '../../hooks/useStores'
import { Card, CardContent, CardHeader, CardTitle } from '../ui'
import { SystemHealthDashboard } from './SystemHealthDashboard'
import { ModuleControlInterface } from './ModuleControlInterface'
import { InterModuleCommunication } from './InterModuleCommunication'
import { ServiceManagement } from './ServiceManagement'
import { ModuleDependencyGraph } from './ModuleDependencyGraph'
import { AutoRestartOptions } from './AutoRestartOptions'

export const SystemManagement = observer(() => {
  const systemStore = useSystemStore()

  return (
    <div className="space-y-6">
      {/* System Health Dashboard */}
      <SystemHealthDashboard />

      {/* Module Control and Service Management Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModuleControlInterface />
        <ServiceManagement />
      </div>

      {/* Inter-module Communication and Dependencies */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <InterModuleCommunication />
        <ModuleDependencyGraph />
      </div>

      {/* Auto-restart Options */}
      <AutoRestartOptions />

      {/* System Information */}
      <Card>
        <CardHeader>
          <CardTitle>System Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="text-sm font-medium text-secondary-600 dark:text-secondary-400">
                System Status
              </div>
              <div className={`text-lg font-semibold ${
                systemStore.isConnected
                  ? 'text-success-600'
                  : 'text-error-600'
              }`}>
                {systemStore.isConnected ? 'Connected' : 'Disconnected'}
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium text-secondary-600 dark:text-secondary-400">
                Connection Uptime
              </div>
              <div className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                {systemStore.connectionUptime > 0
                  ? `${Math.floor(systemStore.connectionUptime / 1000)}s`
                  : 'N/A'
                }
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium text-secondary-600 dark:text-secondary-400">
                Active Errors
              </div>
              <div className={`text-lg font-semibold ${
                systemStore.criticalErrors.length > 0
                  ? 'text-error-600'
                  : 'text-success-600'
              }`}>
                {systemStore.criticalErrors.length} Critical
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
})