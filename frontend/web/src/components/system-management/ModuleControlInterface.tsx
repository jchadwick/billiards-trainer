import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { useSystemStore, useUIStore } from '../../hooks/useStores'
import { Card, CardContent, CardHeader, CardTitle, Button, Modal } from '../ui'
import { StatusIndicator } from '../monitoring/StatusIndicator'

interface ModuleInfo {
  name: string
  displayName: string
  description: string
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping'
  dependencies: string[]
  canRestart: boolean
}

const modules: ModuleInfo[] = [
  {
    name: 'config',
    displayName: 'Configuration',
    description: 'Configuration management and hot reloading',
    status: 'running',
    dependencies: [],
    canRestart: true,
  },
  {
    name: 'core',
    displayName: 'Core Engine',
    description: 'Game logic and physics calculations',
    status: 'running',
    dependencies: ['config'],
    canRestart: true,
  },
  {
    name: 'vision',
    displayName: 'Vision Processing',
    description: 'Camera capture and computer vision',
    status: 'running',
    dependencies: ['config', 'core'],
    canRestart: true,
  },
  {
    name: 'api',
    displayName: 'API Server',
    description: 'REST API and WebSocket connections',
    status: 'running',
    dependencies: ['config', 'core'],
    canRestart: false, // API server restart requires full system restart
  },
  {
    name: 'projector',
    displayName: 'Projector',
    description: 'Visual overlay projection system',
    status: 'error',
    dependencies: ['config', 'core', 'vision'],
    canRestart: true,
  },
]

export const ModuleControlInterface = observer(() => {
  const systemStore = useSystemStore()
  const uiStore = useUIStore()
  const [actioningModule, setActioningModule] = useState<string | null>(null)
  const [confirmAction, setConfirmAction] = useState<{
    module: string
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

  const handleModuleAction = async (moduleName: string, action: 'start' | 'stop' | 'restart') => {
    setActioningModule(moduleName)

    try {
      let result;

      switch (action) {
        case 'start':
          result = await systemStore.startModule(moduleName)
          break
        case 'stop':
          result = await systemStore.stopModule(moduleName)
          break
        case 'restart':
          result = await systemStore.restartModule(moduleName)
          break
      }

      if (result.success) {
        uiStore.showSuccess(
          'Module Action Successful',
          `Module ${moduleName} ${action} completed successfully.`
        )
      } else {
        uiStore.showError(
          'Module Action Failed',
          result.error || `Failed to ${action} module ${moduleName}.`
        )
      }
    } catch (error) {
      uiStore.showError(
        'Module Action Error',
        error instanceof Error ? error.message : `Unexpected error during ${action} of ${moduleName}.`
      )
    } finally {
      setActioningModule(null)
      setConfirmAction(null)
    }
  }

  const canPerformAction = (module: ModuleInfo, action: 'start' | 'stop' | 'restart') => {
    if (actioningModule && actioningModule !== module.name) return false

    switch (action) {
      case 'start':
        return module.status === 'stopped' || module.status === 'error'
      case 'stop':
        return module.status === 'running' && module.canRestart
      case 'restart':
        return module.canRestart && (module.status === 'running' || module.status === 'error')
      default:
        return false
    }
  }

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Module Control Interface</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {modules.map((module) => (
              <div
                key={module.name}
                className="p-4 border border-secondary-200 dark:border-secondary-700 rounded-lg bg-white dark:bg-secondary-900"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <StatusIndicator
                        status={getStatusColor(module.status) as any}
                        size="sm"
                      />
                      <div>
                        <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                          {module.displayName}
                        </h4>
                        <p className="text-xs text-secondary-600 dark:text-secondary-400">
                          {module.description}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4 text-xs text-secondary-500">
                      <span className={`px-2 py-1 rounded-full font-medium capitalize ${
                        module.status === 'running'
                          ? 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
                          : module.status === 'error'
                          ? 'bg-error-100 text-error-800 dark:bg-error-900 dark:text-error-200'
                          : module.status === 'starting' || module.status === 'stopping'
                          ? 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
                          : 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900 dark:text-secondary-200'
                      }`}>
                        {actioningModule === module.name ? 'Processing...' : module.status}
                      </span>

                      {module.dependencies.length > 0 && (
                        <span>
                          Dependencies: {module.dependencies.join(', ')}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex space-x-2 ml-4">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!canPerformAction(module, 'start') || actioningModule === module.name}
                      onClick={() => setConfirmAction({ module: module.name, action: 'start' })}
                    >
                      {actioningModule === module.name ? (
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
                      disabled={!canPerformAction(module, 'stop') || actioningModule === module.name}
                      onClick={() => setConfirmAction({ module: module.name, action: 'stop' })}
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
                      disabled={!canPerformAction(module, 'restart') || actioningModule === module.name}
                      onClick={() => setConfirmAction({ module: module.name, action: 'restart' })}
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

          <div className="mt-6 p-4 bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg">
            <div className="flex items-start space-x-3">
              <svg className="w-5 h-5 text-warning-600 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 17.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <div>
                <h4 className="text-sm font-medium text-warning-800 dark:text-warning-200">
                  Module Control Notice
                </h4>
                <p className="text-sm text-warning-700 dark:text-warning-300 mt-1">
                  Stopping or restarting modules may temporarily affect system functionality. Dependencies will be automatically handled.
                  The API server requires a full system restart and cannot be individually controlled.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Confirmation Modal */}
      {confirmAction && (
        <Modal
          isOpen={true}
          onClose={() => setConfirmAction(null)}
          title={`Confirm Module ${confirmAction.action.charAt(0).toUpperCase() + confirmAction.action.slice(1)}`}
        >
          <div className="space-y-4">
            <p className="text-secondary-700 dark:text-secondary-300">
              Are you sure you want to {confirmAction.action} the{' '}
              <span className="font-medium">
                {modules.find(m => m.name === confirmAction.module)?.displayName}
              </span>{' '}
              module?
            </p>

            {confirmAction.action === 'stop' && (
              <div className="p-3 bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg">
                <p className="text-sm text-warning-700 dark:text-warning-300">
                  This action may temporarily disrupt dependent modules and system functionality.
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
                  handleModuleAction(confirmAction.module, confirmAction.action)
                }
              >
                {confirmAction.action.charAt(0).toUpperCase() + confirmAction.action.slice(1)} Module
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </>
  )
})
