import React, { useState, useEffect } from 'react'
import { observer } from 'mobx-react-lite'
import { useSystemStore } from '../../hooks/useStores'
import { Card, CardContent, CardHeader, CardTitle, Button } from '../ui'

interface CommunicationFlow {
  id: string
  source: string
  target: string
  messageType: string
  payload: any
  timestamp: Date
  status: 'sent' | 'received' | 'failed' | 'pending'
  latency?: number
}

interface ModuleConnectionStatus {
  module: string
  connections: {
    target: string
    status: 'connected' | 'disconnected' | 'error'
    latency: number
    messageCount: number
  }[]
}

export const InterModuleCommunication = observer(() => {
  const systemStore = useSystemStore()
  const [communicationFlows, setCommunicationFlows] = useState<CommunicationFlow[]>([])
  const [moduleConnections, setModuleConnections] = useState<ModuleConnectionStatus[]>([
    {
      module: 'api',
      connections: [
        { target: 'core', status: 'connected', latency: 2.3, messageCount: 156 },
        { target: 'vision', status: 'connected', latency: 4.1, messageCount: 89 },
        { target: 'config', status: 'connected', latency: 1.2, messageCount: 23 },
        { target: 'projector', status: 'error', latency: 0, messageCount: 0 },
      ]
    },
    {
      module: 'core',
      connections: [
        { target: 'vision', status: 'connected', latency: 3.2, messageCount: 78 },
        { target: 'config', status: 'connected', latency: 0.8, messageCount: 45 },
        { target: 'projector', status: 'disconnected', latency: 0, messageCount: 12 },
      ]
    },
    {
      module: 'vision',
      connections: [
        { target: 'core', status: 'connected', latency: 2.8, messageCount: 134 },
        { target: 'projector', status: 'error', latency: 0, messageCount: 8 },
      ]
    }
  ])

  // Simulate real-time communication flow updates
  useEffect(() => {
    const generateMockFlow = (): CommunicationFlow => {
      const modules = ['api', 'core', 'vision', 'config', 'projector']
      const messageTypes = ['config_update', 'ball_detection', 'trajectory_data', 'calibration_point', 'health_check']
      const statuses: CommunicationFlow['status'][] = ['sent', 'received', 'failed', 'pending']

      const source = modules[Math.floor(Math.random() * modules.length)]
      let target = modules[Math.floor(Math.random() * modules.length)]
      while (target === source) {
        target = modules[Math.floor(Math.random() * modules.length)]
      }

      return {
        id: `flow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        source,
        target,
        messageType: messageTypes[Math.floor(Math.random() * messageTypes.length)],
        payload: { data: 'sample_data', size: Math.floor(Math.random() * 1000) },
        timestamp: new Date(),
        status: statuses[Math.floor(Math.random() * statuses.length)],
        latency: Math.random() * 10
      }
    }

    const interval = setInterval(() => {
      setCommunicationFlows(prev => {
        const newFlow = generateMockFlow()
        const updated = [newFlow, ...prev.slice(0, 19)] // Keep last 20 flows
        return updated
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
      case 'sent':
      case 'received':
        return 'text-success-600'
      case 'disconnected':
      case 'pending':
        return 'text-warning-600'
      case 'error':
      case 'failed':
        return 'text-error-600'
      default:
        return 'text-secondary-600'
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'connected':
      case 'sent':
      case 'received':
        return 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
      case 'disconnected':
      case 'pending':
        return 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
      case 'error':
      case 'failed':
        return 'bg-error-100 text-error-800 dark:bg-error-900 dark:text-error-200'
      default:
        return 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900 dark:text-secondary-200'
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Inter-module Communication</CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            // Refresh communication data
            console.log('Refreshing communication data...')
          }}
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
          {/* Module Connection Status */}
          <div>
            <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-3">
              Module Connections
            </h4>
            <div className="space-y-3">
              {moduleConnections.map((moduleStatus) => (
                <div key={moduleStatus.module} className="p-3 bg-secondary-50 dark:bg-secondary-800 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100 capitalize">
                      {moduleStatus.module}
                    </span>
                    <span className="text-xs text-secondary-500">
                      {moduleStatus.connections.length} connections
                    </span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {moduleStatus.connections.map((connection, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-white dark:bg-secondary-900 rounded border border-secondary-200 dark:border-secondary-700">
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${
                            connection.status === 'connected' ? 'bg-success-500' :
                            connection.status === 'disconnected' ? 'bg-warning-500' : 'bg-error-500'
                          }`} />
                          <span className="text-xs font-medium text-secondary-900 dark:text-secondary-100">
                            → {connection.target}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2 text-xs text-secondary-500">
                          {connection.status === 'connected' && (
                            <>
                              <span>{connection.latency.toFixed(1)}ms</span>
                              <span>({connection.messageCount})</span>
                            </>
                          )}
                          <span className={`px-1.5 py-0.5 rounded-full text-xs font-medium ${getStatusBadge(connection.status)}`}>
                            {connection.status}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Real-time Message Flow */}
          <div>
            <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-3">
              Real-time Message Flow
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {communicationFlows.length === 0 ? (
                <div className="text-center py-4 text-secondary-500">
                  <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <p className="text-sm">Waiting for communication data...</p>
                </div>
              ) : (
                communicationFlows.map((flow) => (
                  <div key={flow.id} className="flex items-center justify-between p-2 bg-white dark:bg-secondary-900 rounded border border-secondary-200 dark:border-secondary-700">
                    <div className="flex items-center space-x-3 flex-1">
                      <div className="flex items-center space-x-2 text-xs">
                        <span className="font-medium text-secondary-900 dark:text-secondary-100">
                          {flow.source}
                        </span>
                        <svg className="w-3 h-3 text-secondary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <span className="font-medium text-secondary-900 dark:text-secondary-100">
                          {flow.target}
                        </span>
                      </div>
                      <span className="text-xs text-secondary-600 dark:text-secondary-400">
                        {flow.messageType.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      {flow.latency && (
                        <span className="text-xs text-secondary-500">
                          {flow.latency.toFixed(1)}ms
                        </span>
                      )}
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(flow.status)}`}>
                        {flow.status}
                      </span>
                      <span className="text-xs text-secondary-500">
                        {flow.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Communication Statistics */}
          <div>
            <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 mb-3">
              Communication Statistics
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-white dark:bg-secondary-900 rounded border border-secondary-200 dark:border-secondary-700">
                <div className="text-lg font-bold text-success-600">
                  {communicationFlows.filter(f => f.status === 'sent' || f.status === 'received').length}
                </div>
                <div className="text-xs text-secondary-600 dark:text-secondary-400">
                  Successful
                </div>
              </div>
              <div className="text-center p-3 bg-white dark:bg-secondary-900 rounded border border-secondary-200 dark:border-secondary-700">
                <div className="text-lg font-bold text-error-600">
                  {communicationFlows.filter(f => f.status === 'failed').length}
                </div>
                <div className="text-xs text-secondary-600 dark:text-secondary-400">
                  Failed
                </div>
              </div>
              <div className="text-center p-3 bg-white dark:bg-secondary-900 rounded border border-secondary-200 dark:border-secondary-700">
                <div className="text-lg font-bold text-warning-600">
                  {communicationFlows.filter(f => f.status === 'pending').length}
                </div>
                <div className="text-xs text-secondary-600 dark:text-secondary-400">
                  Pending
                </div>
              </div>
              <div className="text-center p-3 bg-white dark:bg-secondary-900 rounded border border-secondary-200 dark:border-secondary-700">
                <div className="text-lg font-bold text-secondary-900 dark:text-secondary-100">
                  {communicationFlows.length > 0
                    ? (communicationFlows.reduce((sum, f) => sum + (f.latency || 0), 0) / communicationFlows.filter(f => f.latency).length).toFixed(1)
                    : '0.0'
                  }ms
                </div>
                <div className="text-xs text-secondary-600 dark:text-secondary-400">
                  Avg Latency
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
})
