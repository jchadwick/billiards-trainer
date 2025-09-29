import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { useSystemStore, useUIStore } from '../../hooks/useStores'
import { Card, CardContent, CardHeader, CardTitle, Button, Checkbox, Select, Input } from '../ui'

interface AutoRestartPolicy {
  enabled: boolean
  maxAttempts: number
  retryDelay: number // seconds
  backoffMultiplier: number
  maxDelay: number // seconds
  conditions: {
    onCrash: boolean
    onHealthCheckFailure: boolean
    onDependencyFailure: boolean
    onResourceExhaustion: boolean
  }
  excludeModules: string[]
}

interface RecoveryAction {
  id: string
  module: string
  action: 'restart' | 'recreate' | 'rollback'
  timestamp: Date
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  attempts: number
  nextAttempt?: Date
  error?: string
}

export const AutoRestartOptions = observer(() => {
  const systemStore = useSystemStore()
  const uiStore = useUIStore()

  const [autoRestartPolicy, setAutoRestartPolicy] = useState<AutoRestartPolicy>({
    enabled: true,
    maxAttempts: 3,
    retryDelay: 5,
    backoffMultiplier: 2,
    maxDelay: 60,
    conditions: {
      onCrash: true,
      onHealthCheckFailure: true,
      onDependencyFailure: false,
      onResourceExhaustion: false,
    },
    excludeModules: [],
  })

  const [recoveryActions] = useState<RecoveryAction[]>([
    {
      id: 'recovery_1',
      module: 'projector',
      action: 'restart',
      timestamp: new Date(Date.now() - 120000), // 2 minutes ago
      status: 'completed',
      attempts: 2,
    },
    {
      id: 'recovery_2',
      module: 'vision',
      action: 'restart',
      timestamp: new Date(Date.now() - 60000), // 1 minute ago
      status: 'failed',
      attempts: 3,
      error: 'Camera device not found',
    },
    {
      id: 'recovery_3',
      module: 'core',
      action: 'restart',
      timestamp: new Date(Date.now() - 10000), // 10 seconds ago
      status: 'in_progress',
      attempts: 1,
      nextAttempt: new Date(Date.now() + 15000), // 15 seconds from now
    },
  ])

  const availableModules = ['config', 'core', 'vision', 'api', 'projector']

  const handlePolicyUpdate = async (updates: Partial<AutoRestartPolicy>) => {
    try {
      const newPolicy = { ...autoRestartPolicy, ...updates }
      setAutoRestartPolicy(newPolicy)

      // This would call the backend API to update the policy
      // await systemStore.updateAutoRestartPolicy(newPolicy)

      uiStore.showSuccess(
        'Auto-restart Policy Updated',
        'The auto-restart policy has been updated successfully.'
      )
    } catch (error) {
      uiStore.showError(
        'Policy Update Failed',
        error instanceof Error ? error.message : 'Failed to update auto-restart policy.'
      )
    }
  }

  const handleManualRecovery = async (module: string, action: 'restart' | 'recreate' | 'rollback') => {
    try {
      // This would call the backend API for manual recovery
      // await systemStore.triggerManualRecovery(module, action)

      uiStore.showInfo(
        'Recovery Initiated',
        `Manual ${action} initiated for ${module} module.`
      )
    } catch (error) {
      uiStore.showError(
        'Recovery Failed',
        error instanceof Error ? error.message : `Failed to initiate ${action} for ${module}.`
      )
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
      case 'in_progress':
        return 'bg-info-100 text-info-800 dark:bg-info-900 dark:text-info-200'
      case 'failed':
        return 'bg-error-100 text-error-800 dark:bg-error-900 dark:text-error-200'
      case 'pending':
        return 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
      default:
        return 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900 dark:text-secondary-200'
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Automatic Restart & Recovery Options</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Auto-restart Policy Configuration */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                Auto-restart Policy
              </h4>
              <Checkbox
                checked={autoRestartPolicy.enabled}
                onChange={(checked) => handlePolicyUpdate({ enabled: checked })}
                label="Enable automatic restart"
              />
            </div>

            {autoRestartPolicy.enabled && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 bg-secondary-50 dark:bg-secondary-800 rounded-lg">
                <div>
                  <label className="block text-xs font-medium text-secondary-700 dark:text-secondary-300 mb-1">
                    Max Restart Attempts
                  </label>
                  <Input
                    type="number"
                    value={autoRestartPolicy.maxAttempts}
                    onChange={(e) => handlePolicyUpdate({ maxAttempts: parseInt(e.target.value) || 3 })}
                    min={1}
                    max={10}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-secondary-700 dark:text-secondary-300 mb-1">
                    Initial Delay (seconds)
                  </label>
                  <Input
                    type="number"
                    value={autoRestartPolicy.retryDelay}
                    onChange={(e) => handlePolicyUpdate({ retryDelay: parseInt(e.target.value) || 5 })}
                    min={1}
                    max={300}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-secondary-700 dark:text-secondary-300 mb-1">
                    Backoff Multiplier
                  </label>
                  <Input
                    type="number"
                    step="0.1"
                    value={autoRestartPolicy.backoffMultiplier}
                    onChange={(e) => handlePolicyUpdate({ backoffMultiplier: parseFloat(e.target.value) || 2 })}
                    min={1}
                    max={5}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-secondary-700 dark:text-secondary-300 mb-1">
                    Max Delay (seconds)
                  </label>
                  <Input
                    type="number"
                    value={autoRestartPolicy.maxDelay}
                    onChange={(e) => handlePolicyUpdate({ maxDelay: parseInt(e.target.value) || 60 })}
                    min={10}
                    max={3600}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Restart Conditions */}
          {autoRestartPolicy.enabled && (
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                Restart Conditions
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <Checkbox
                  checked={autoRestartPolicy.conditions.onCrash}
                  onChange={(checked) => handlePolicyUpdate({
                    conditions: { ...autoRestartPolicy.conditions, onCrash: checked }
                  })}
                  label="Module crash or unexpected exit"
                />
                <Checkbox
                  checked={autoRestartPolicy.conditions.onHealthCheckFailure}
                  onChange={(checked) => handlePolicyUpdate({
                    conditions: { ...autoRestartPolicy.conditions, onHealthCheckFailure: checked }
                  })}
                  label="Health check failure"
                />
                <Checkbox
                  checked={autoRestartPolicy.conditions.onDependencyFailure}
                  onChange={(checked) => handlePolicyUpdate({
                    conditions: { ...autoRestartPolicy.conditions, onDependencyFailure: checked }
                  })}
                  label="Dependency module failure"
                />
                <Checkbox
                  checked={autoRestartPolicy.conditions.onResourceExhaustion}
                  onChange={(checked) => handlePolicyUpdate({
                    conditions: { ...autoRestartPolicy.conditions, onResourceExhaustion: checked }
                  })}
                  label="Resource exhaustion (memory/CPU)"
                />
              </div>
            </div>
          )}

          {/* Module Exclusions */}
          {autoRestartPolicy.enabled && (
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                Exclude Modules from Auto-restart
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {availableModules.map((module) => (
                  <Checkbox
                    key={module}
                    checked={autoRestartPolicy.excludeModules.includes(module)}
                    onChange={(checked) => {
                      const excludeModules = checked
                        ? [...autoRestartPolicy.excludeModules, module]
                        : autoRestartPolicy.excludeModules.filter(m => m !== module)
                      handlePolicyUpdate({ excludeModules })
                    }}
                    label={module.charAt(0).toUpperCase() + module.slice(1)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Manual Recovery Actions */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                Manual Recovery
              </h4>
              <div className="flex space-x-2">
                <Select
                  value=""
                  onChange={(e) => {
                    const [module, action] = e.target.value.split(':')
                    if (module && action) {
                      handleManualRecovery(module, action as any)
                    }
                  }}
                  className="text-sm"
                >
                  <option value="">Trigger Recovery...</option>
                  {availableModules.map((module) => (
                    <React.Fragment key={module}>
                      <option value={`${module}:restart`}>Restart {module}</option>
                      <option value={`${module}:recreate`}>Recreate {module}</option>
                      <option value={`${module}:rollback`}>Rollback {module}</option>
                    </React.Fragment>
                  ))}
                </Select>
              </div>
            </div>

            <div className="bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg p-3">
              <div className="flex items-start space-x-2">
                <svg className="w-4 h-4 text-warning-600 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 17.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <div>
                  <p className="text-sm text-warning-700 dark:text-warning-300">
                    <span className="font-medium">Manual recovery actions:</span>
                  </p>
                  <ul className="text-xs text-warning-600 dark:text-warning-400 mt-1 space-y-1">
                    <li>• <strong>Restart:</strong> Stop and start the module</li>
                    <li>• <strong>Recreate:</strong> Completely reinitialize the module</li>
                    <li>• <strong>Rollback:</strong> Revert to last known good state</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Recovery Actions */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
              Recent Recovery Actions
            </h4>
            <div className="space-y-2">
              {recoveryActions.length === 0 ? (
                <div className="text-center py-4 text-secondary-500">
                  <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm">No recent recovery actions</p>
                </div>
              ) : (
                recoveryActions.map((action) => (
                  <div key={action.id} className="flex items-center justify-between p-3 bg-white dark:bg-secondary-900 rounded border border-secondary-200 dark:border-secondary-700">
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {action.status === 'in_progress' ? (
                          <svg className="w-4 h-4 animate-spin text-info-600" fill="none" viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                            <path fill="currentColor" className="opacity-75" d="M4 12a8 8 0 018-8v8z" />
                          </svg>
                        ) : action.status === 'completed' ? (
                          <svg className="w-4 h-4 text-success-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        ) : action.status === 'failed' ? (
                          <svg className="w-4 h-4 text-error-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        ) : (
                          <svg className="w-4 h-4 text-warning-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                            {action.action.charAt(0).toUpperCase() + action.action.slice(1)} {action.module}
                          </span>
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusBadge(action.status)}`}>
                            {action.status.replace('_', ' ')}
                          </span>
                        </div>
                        <div className="text-xs text-secondary-500 space-x-4">
                          <span>{action.timestamp.toLocaleString()}</span>
                          <span>Attempt {action.attempts}</span>
                          {action.nextAttempt && (
                            <span>Next: {action.nextAttempt.toLocaleTimeString()}</span>
                          )}
                        </div>
                        {action.error && (
                          <div className="text-xs text-error-600 dark:text-error-400 mt-1">
                            Error: {action.error}
                          </div>
                        )}
                      </div>
                    </div>
                    {action.status === 'failed' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleManualRecovery(action.module, action.action)}
                      >
                        Retry
                      </Button>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
})